"""Train a small MNIST network, quantize it to INT8, and export everything the
RTL needs: weight / parameter / layer-table memory images, test images, and
bit-exact golden outputs from an integer-only reference model.

    python model/train.py [--arch cnn|mlp|stress] [--n-test N]

Architectures
    mlp:  784 -> fc 64 (ReLU) -> fc 10
    cnn:  28x28x1 -> conv4x4/2 x8 (ReLU) -> conv3x3/2 x16 (ReLU) -> fc 10
          The first layer is stored and computed as a 2x2 convolution over a
          space-to-depth input (14x14x4), which is the same arithmetic.
    stress: hardware corner cases (padding, pooling, 1x1 conv, odd fc width),
          not meant for accuracy

Layer kinds
    ("conv", k, stride, out_ch[, pad])   zero padding, ReLU unless last
    ("pool", k, stride)                  max pooling, no parameters
    ("fc", out)                          ReLU unless last

Hardware arithmetic (must match rtl/nn_accel.sv exactly):
    acc  = sum(w_int8 * x_int8) + bias_int32
    hidden: y = clamp((acc * M + 2^(S-1)) >> S, 0, 127), M and S per channel
    last  : logits = acc (raw int32), class = argmax
    pool  : y = max over the window, per channel

Memory layout: activations are stored HWC with the channel count padded to a
multiple of P=8, so one pixel is a whole number of 64-bit words and every
kernel row of a convolution window is a contiguous run of words.
"""
import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
P = 8  # MAC lanes == bytes per memory word

ARCHS = {
    "mlp": {"input": (1, 1, 784), "layers": [("fc", 64), ("fc", 10)],
            "epochs": 10, "lr": 0.05},
    # "s2d": 2 stores the image space-to-depth: each 2x2 block of pixels
    # becomes one 4-channel pixel, so "input" is 14x14x4. A 2x2 stride-1
    # convolution there equals a 4x4 stride-2 convolution on the image, but
    # its window is 4 memory words instead of 16.
    "cnn": {"input": (14, 14, 4), "s2d": 2,
            "layers": [("conv", 2, 1, 8), ("conv", 3, 2, 16), ("fc", 10)],
            "epochs": 12, "lr": 0.05},
    # Not a useful network: a hardware stress test. It has padded
    # convolutions at stride 1 and 2, max pooling over one-word and two-word
    # pixels (the second with an odd input size), a 1x1 convolution whose
    # windows are a single word, a single-channel input padded to 8, and a
    # fully-connected width that is not a multiple of 8.
    "stress": {"input": (28, 28, 1),
               "layers": [("conv", 3, 1, 8, 1), ("pool", 2, 2), ("conv", 1, 1, 8),
                          ("conv", 3, 2, 16, 1), ("pool", 2, 2), ("fc", 20), ("fc", 10)],
               "epochs": 2, "lr": 0.05},
}


def load_mnist():
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, name in FILES.items():
        path = DATA / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(MIRROR + name, path)
        raw = gzip.open(path, "rb").read()
        if key.endswith("_x"):
            _, n, r, c = struct.unpack(">IIII", raw[:16])
            out[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(n, r * c)
        else:
            out[key] = np.frombuffer(raw, np.uint8, offset=8)
    return out


# ------------------------------------------------------------------ geometry
def prep(arch, x):
    """Flat 784-pixel images -> the (N, H, W, C) tensor the first layer reads.
    Space-to-depth is a pure rearrangement: channel order is (dy, dx)."""
    b = arch.get("s2d")
    if not b:
        return x.reshape(-1, *arch["input"])
    n, side = len(x), 28 // b
    x = x.reshape(n, side, b, side, b).transpose(0, 1, 3, 2, 4)
    return x.reshape(n, side, side, b * b)


def geometry(arch):
    """Resolve every layer to kernel, stride, padding and in/out shapes."""
    h, w, c = arch["input"]
    out = []
    for spec in arch["layers"]:
        kind = spec[0]
        if kind == "fc":
            kh, kw, s, p, oc = h, w, 1, 0, spec[1]
        elif kind == "pool":
            kh = kw = spec[1]
            s, p, oc = spec[2], 0, c
        else:
            kh = kw = spec[1]
            s, oc, p = spec[2], spec[3], (spec[4] if len(spec) > 4 else 0)
        oh, ow = (h + 2 * p - kh) // s + 1, (w + 2 * p - kw) // s + 1
        out.append(dict(kind=kind, kh=kh, kw=kw, s=s, p=p, oc=oc,
                        h=h, w=w, c=c, oh=oh, ow=ow))
        h, w, c = oh, ow, oc
    assert out[-1]["kind"] != "pool", "the last layer must produce logits"
    return out


def im2col(x, g):
    """(N,H,W,C) -> (N,OH,OW,kh*kw*C), patch order (ky, kx, c), zero padded."""
    if g["p"]:
        x = np.pad(x, ((0, 0), (g["p"], g["p"]), (g["p"], g["p"]), (0, 0)))
    n, _, _, c = x.shape
    cols = np.empty((n, g["oh"], g["ow"], g["kh"], g["kw"], c), x.dtype)
    for ky in range(g["kh"]):
        for kx in range(g["kw"]):
            cols[:, :, :, ky, kx, :] = x[:, ky:ky + g["s"] * g["oh"]:g["s"],
                                         kx:kx + g["s"] * g["ow"]:g["s"], :]
    return cols.reshape(n, g["oh"], g["ow"], -1)


def col2im(dcols, g):
    n, p, c = dcols.shape[0], g["p"], g["c"]
    d = dcols.reshape(n, g["oh"], g["ow"], g["kh"], g["kw"], c)
    dx = np.zeros((n, g["h"] + 2 * p, g["w"] + 2 * p, c), dcols.dtype)
    for ky in range(g["kh"]):
        for kx in range(g["kw"]):
            dx[:, ky:ky + g["s"] * g["oh"]:g["s"],
               kx:kx + g["s"] * g["ow"]:g["s"], :] += d[:, :, :, ky, kx, :]
    return dx[:, p:p + g["h"], p:p + g["w"], :]


def pool_windows(a, g):
    """(N,H,W,C) -> (N,OH,OW,kh*kw,C): the values each pooled output sees."""
    n = len(a)
    return im2col(a, g).reshape(n, g["oh"], g["ow"], g["kh"] * g["kw"], g["c"])


# ------------------------------------------------------------------ training
def train(arch, geo, xq, y, mom=0.9, bs=64, seed=0):
    """Float training on inputs scaled to [0, 1]. Returns one entry per layer:
    (W, b) with W shaped (out_ch, kh*kw*c), or None for a pooling layer."""
    rng = np.random.default_rng(seed)
    x = prep(arch, xq.astype(np.float32) / 127.0)
    last = len(geo) - 1
    params = []
    for g in geo:
        if g["kind"] == "pool":
            params.append(None)
            continue
        fan_in = g["kh"] * g["kw"] * g["c"]
        params.append([rng.normal(0, np.sqrt(2 / fan_in), (g["oc"], fan_in)).astype(np.float32),
                       np.zeros(g["oc"], np.float32)])
    vel = [None if p is None else [np.zeros_like(p[0]), np.zeros_like(p[1])] for p in params]
    n, lr = len(x), arch["lr"]
    for ep in range(arch["epochs"]):
        perm = rng.permutation(n)
        loss_sum = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            a, yb = x[idx], y[idx]
            cache = []
            for li, g in enumerate(geo):
                if g["kind"] == "pool":
                    win = pool_windows(a, g)
                    arg = win.argmax(3)
                    cache.append(arg)
                    a = np.take_along_axis(win, arg[:, :, :, None, :], 3)[:, :, :, 0, :]
                else:
                    cols = im2col(a, g)
                    z = cols @ params[li][0].T + params[li][1]
                    cache.append((cols, z))
                    a = z if li == last else np.maximum(z, 0)
            z = a.reshape(len(idx), -1)
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            loss_sum += -np.log(p[np.arange(len(idx)), yb] + 1e-9).sum()
            p[np.arange(len(idx)), yb] -= 1
            dz = (p / len(idx)).reshape(cache[-1][1].shape)
            for li in reversed(range(len(geo))):
                g = geo[li]
                if g["kind"] == "pool":
                    arg = cache[li]
                    d = np.zeros(arg.shape[:3] + (g["kh"] * g["kw"], g["c"]), np.float32)
                    np.put_along_axis(d, arg[:, :, :, None, :], dz[:, :, :, None, :], 3)
                    dz = col2im(d.reshape(arg.shape[:3] + (-1,)), g)
                    continue
                cols, z = cache[li]
                if li != last:
                    dz = dz * (z > 0)
                flat = dz.reshape(-1, g["oc"])
                grads = [flat.T @ cols.reshape(-1, cols.shape[-1]), flat.sum(0)]
                if li:
                    dz = col2im(dz @ params[li][0], g)
                for prm, v, gr in zip(params[li], vel[li], grads):
                    v *= mom
                    v -= lr * gr
                    prm += v
        print(f"epoch {ep + 1:2d}  loss {loss_sum / n:.4f}")
        lr *= 0.8
    return [None if p is None else (p[0], p[1]) for p in params]


def float_predict(arch, geo, fparams, xq):
    a = prep(arch, xq.astype(np.float32) / 127.0)
    for li, (g, fp) in enumerate(zip(geo, fparams)):
        if g["kind"] == "pool":
            a = pool_windows(a, g).max(3)
        else:
            z = im2col(a, g) @ fp[0].T + fp[1]
            a = z if li == len(geo) - 1 else np.maximum(z, 0)
    return a.reshape(len(a), -1).argmax(1)


# -------------------------------------------------------------- quantization
def int_layer(a, g, q, last):
    """One layer of the integer reference model."""
    if g["kind"] == "pool":
        return pool_windows(a, g).max(3)
    acc = im2col(a, g) @ q["w"].T + q["b"]
    assert np.abs(acc).max() < 2**31
    if last:
        return acc
    return np.clip((acc * q["M"] + np.left_shift(1, q["S"] - 1)) >> q["S"], 0, 127)


def requant_factor(f):
    """Scalar f ~= M / 2^S with M in [2^14, 2^15] and 1 <= S <= 40."""
    s = 1
    while f * (1 << s) < (1 << 14) and s < 40:
        s += 1
    m = int(round(f * (1 << s)))
    assert 0 <= m < (1 << 16)
    return m, s


def quantize(arch, geo, fparams, calib_xq):
    """Post-training quantization, calibrated layer by layer on integer
    activations so the scales match what the hardware will actually see.
    Hidden layers get one weight scale per output channel. The last layer
    keeps a single scale, because its raw sums are compared with each other."""
    a = prep(arch, calib_xq).astype(np.int64)
    s_in = 1 / 127.0
    qparams = []
    for li, (g, fp) in enumerate(zip(geo, fparams)):
        last = li == len(geo) - 1
        if g["kind"] == "pool":
            qparams.append(None)
            a = int_layer(a, g, None, False)
            continue
        w, b = fp
        oc = g["oc"]
        if last:
            s_w = np.full(oc, np.abs(w).max() / 127)
        else:
            s_w = np.maximum(np.abs(w).max(1), 1e-12) / 127
        q = dict(w=np.round(w / s_w[:, None]).astype(np.int64),
                 b=np.round(b / (s_w * s_in)).astype(np.int64),
                 M=np.zeros(oc, np.int64), S=np.ones(oc, np.int64))
        if not last:
            acc = im2col(a, g) @ q["w"].T + q["b"]
            s_out = np.percentile(np.maximum(acc, 0) * (s_w * s_in), 99.9) / 127
            for c in range(oc):
                q["M"][c], q["S"][c] = requant_factor(s_w[c] * s_in / s_out)
            a = int_layer(a, g, q, False)
            s_in = s_out
        qparams.append(q)
    return qparams


def golden(arch, geo, qparams, xq):
    """Integer-only reference; mirrors the RTL datapath bit for bit."""
    a = prep(arch, xq).astype(np.int64)
    for li, (g, q) in enumerate(zip(geo, qparams)):
        a = int_layer(a, g, q, li == len(geo) - 1)
    logits = a.reshape(len(a), -1)
    return logits, logits.argmax(1)


# -------------------------------------------------------------------- export
def pad_c(a):
    """Zero-pad the last (channel) axis to a multiple of P."""
    pad = (-a.shape[-1]) % P
    return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, pad)])


def pack_words(a):
    """int array (rows, n) with n % P == 0 -> flat list of 64-bit words."""
    a = (a.astype(np.int64) & 0xFF).reshape(-1, P).astype(np.uint64)
    shifts = np.arange(P, dtype=np.uint64) * np.uint64(8)
    return (a << shifts).sum(-1, dtype=np.uint64)


def write_hex(path, values, digits):
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v) & ((1 << (4 * digits)) - 1):0{digits}x}\n")


# Layer descriptor, LSB first. Keep in sync with rtl/nn_accel.sv.
CFG_FIELDS = [("kw", 4), ("kh", 4), ("rd_words", 12), ("pix_step", 12),
              ("row_step", 12), ("step_x", 12), ("step_y", 16), ("org", 16),
              ("in_w", 8), ("in_h", 8), ("pad", 4), ("stride", 4),
              ("out_w", 8), ("out_h", 8), ("out_ch", 12), ("w_base", 20),
              ("b_base", 12), ("pool", 1), ("last", 1)]
CFG_BITS = 176


def cfg_word(**fields):
    word, pos = 0, 0
    for name, width in CFG_FIELDS:
        value = fields.pop(name)
        assert 0 <= value < (1 << width), (name, value, width)
        word |= value << pos
        pos += width
    assert not fields and pos <= CFG_BITS
    return word


def compile_program(geo, qparams):
    """Turn the network into the accelerator's memory images.

    The read pointer visits `rd_words` words of each window pixel, then moves
    `pix_step` to the next pixel, or `row_step` to the next kernel row. A
    convolution reads every word of a pixel; pooling reads one word per pixel
    and is repeated for each word of the output pixel. `org` is the address of
    the first window's first word, which is negative (two's complement) when
    the layer is padded; out-of-image words are zeroed by the hardware."""
    weights, params, cfg = [], [], []
    n_w = n_p = 0
    for li, (g, q) in enumerate(zip(geo, qparams)):
        last = li == len(geo) - 1
        cw = -(-g["c"] // P)                   # words per input pixel
        common = dict(step_x=g["s"] * cw, step_y=g["s"] * g["w"] * cw,
                      out_w=g["ow"], out_h=g["oh"], w_base=n_w, b_base=n_p,
                      last=int(last))
        if g["kind"] == "pool":
            cfg.append(cfg_word(
                kw=g["kw"], kh=g["kh"], rd_words=1, pix_step=cw,
                row_step=(g["w"] - g["kw"] + 1) * cw, org=0,
                in_w=g["w"], in_h=g["h"], pad=0, stride=g["s"],
                out_ch=cw, pool=1, **common))
            continue
        if g["kind"] == "fc":
            # one "pixel" that is the whole input
            window = dict(kw=1, kh=1, rd_words=g["h"] * g["w"] * cw, pix_step=1,
                          row_step=1, org=0, in_w=1, in_h=1, pad=0, stride=1)
        else:
            assert g["oc"] % P == 0, "conv layers need out_ch % 8 == 0"
            window = dict(kw=g["kw"], kh=g["kh"], rd_words=cw, pix_step=1,
                          row_step=(g["w"] - g["kw"]) * cw + 1,
                          org=(-(g["p"] * g["w"] + g["p"]) * cw) & 0xFFFF,
                          in_w=g["w"], in_h=g["h"], pad=g["p"], stride=g["s"])
        cfg.append(cfg_word(out_ch=g["oc"], pool=0, **window, **common))
        w = q["w"].reshape(g["oc"], g["kh"], g["kw"], g["c"])
        w_words = pack_words(pad_c(w).reshape(g["oc"], -1))
        weights.append(w_words)
        # per-output parameter word: bias, multiplier M, shift S
        params.append((q["b"] & 0xFFFFFFFF) | (q["M"] << 32) | (q["S"] << 48))
        n_w += len(w_words)
        n_p += g["oc"]
    return np.concatenate(weights), np.concatenate(params), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS, default="cnn")
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()
    arch, n_test = ARCHS[args.arch], args.n_test
    geo = geometry(arch)
    out = ROOT / "build" / args.arch
    out.mkdir(parents=True, exist_ok=True)

    d = load_mnist()
    xtr, xte = (d["train_x"] >> 1).astype(np.int64), (d["test_x"] >> 1).astype(np.int64)
    ytr, yte = d["train_y"].astype(np.int64), d["test_y"].astype(np.int64)

    fparams = train(arch, geo, xtr, ytr)
    fl = float_predict(arch, geo, fparams, xte)
    qparams = quantize(arch, geo, fparams, xtr[:10000])
    logits, pred = golden(arch, geo, qparams, xte)
    print(f"arch           : {args.arch}")
    print(f"float accuracy : {(fl == yte).mean() * 100:.2f}%")
    print(f"INT8  accuracy : {(pred == yte).mean() * 100:.2f}%")

    weights, params, cfg = compile_program(geo, qparams)
    images = pad_c(prep(arch, xte[:n_test])).reshape(n_test, -1)
    in_words = images.shape[1] // P
    write_hex(out / "weights.hex", weights, 16)
    write_hex(out / "params.hex", params, 14)
    write_hex(out / "cfg.hex", cfg, CFG_BITS // 4)
    write_hex(out / "meta.hex", [in_words, len(cfg), logits.shape[1]], 8)
    write_hex(out / "images.hex", pack_words(images), 16)
    write_hex(out / "labels.hex", yte[:n_test], 2)
    write_hex(out / "golden_class.hex", pred[:n_test], 2)
    write_hex(out / "golden_logits.hex", logits[:n_test].reshape(-1), 8)
    # test image 0 as input words, for the FPGA demo ROM
    write_hex(out / "demo_image.hex", pack_words(images[:1]), 16)
    print(f"demo image     : test image 0, label {yte[0]}, predicted {pred[0]}")
    macs = sum(g["oh"] * g["ow"] * g["oc"] * g["kh"] * g["kw"] * g["c"]
               for g in geo if g["kind"] != "pool")
    print(f"weight words   : {len(weights)}  ({len(weights) * 8} bytes)")
    print(f"parameter words: {len(params)}")
    print(f"input words    : {in_words}")
    print(f"multiply-adds  : {macs} per inference")
    print(f"wrote {n_test} test vectors -> {out}")


if __name__ == "__main__":
    main()

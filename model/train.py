"""Train a small MNIST network, quantize it to INT8, and export everything the
RTL needs: weight/bias/layer-table memory images, test images, and bit-exact
golden outputs from an integer-only reference model.

    python model/train.py [--arch cnn|mlp|stress] [--n-test N]

Architectures
    mlp:  784 -> fc 64 (ReLU) -> fc 10
    cnn:  28x28x1 -> conv3x3/2 x8 (ReLU) -> conv3x3/2 x16 (ReLU) -> fc 10
    stress: hardware corner cases (1x1 conv, odd fc width), not for accuracy

Hardware arithmetic (must match rtl/nn_accel.sv exactly):
    acc  = sum(w_int8 * x_int8) + bias_int32
    hidden: y = clamp((acc * M + 2^(S-1)) >> S, 0, 127)
    last  : logits = acc (raw int32), class = argmax

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

# Each layer: kind, kernel size, stride, output channels. An fc layer is a
# convolution whose kernel covers its whole input.
ARCHS = {
    "mlp": {"input": (1, 1, 784), "layers": [("fc", 64), ("fc", 10)],
            "epochs": 10, "lr": 0.05},
    "cnn": {"input": (28, 28, 1),
            "layers": [("conv", 3, 2, 8), ("conv", 3, 2, 16), ("fc", 10)],
            "epochs": 12, "lr": 0.05},
    # Not a useful network: a hardware stress test. The 1x1 convolution is a
    # hidden layer with one-word windows, so the front end has to wait for
    # the post unit; the 20-wide fc output is not a multiple of 8.
    "stress": {"input": (28, 28, 1),
               "layers": [("conv", 3, 2, 8), ("conv", 1, 1, 8), ("fc", 20), ("fc", 10)],
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
def geometry(arch):
    """Resolve every layer to (k_h, k_w, stride, out_ch) plus in/out shapes."""
    h, w, c = arch["input"]
    out = []
    for spec in arch["layers"]:
        if spec[0] == "fc":
            kh, kw, s, oc = h, w, 1, spec[1]
        else:
            _, k, s, oc = spec
            kh = kw = k
        oh, ow = (h - kh) // s + 1, (w - kw) // s + 1
        out.append(dict(kh=kh, kw=kw, s=s, oc=oc, h=h, w=w, c=c, oh=oh, ow=ow))
        h, w, c = oh, ow, oc
    return out


def im2col(x, g):
    """(N,H,W,C) -> (N,OH,OW,kh*kw*C), patch order (ky, kx, c)."""
    n, _, _, c = x.shape
    cols = np.empty((n, g["oh"], g["ow"], g["kh"], g["kw"], c), x.dtype)
    for ky in range(g["kh"]):
        for kx in range(g["kw"]):
            cols[:, :, :, ky, kx, :] = x[:, ky:ky + g["s"] * g["oh"]:g["s"],
                                         kx:kx + g["s"] * g["ow"]:g["s"], :]
    return cols.reshape(n, g["oh"], g["ow"], -1)


def col2im(dcols, g, c):
    n = dcols.shape[0]
    d = dcols.reshape(n, g["oh"], g["ow"], g["kh"], g["kw"], c)
    dx = np.zeros((n, g["h"], g["w"], c), dcols.dtype)
    for ky in range(g["kh"]):
        for kx in range(g["kw"]):
            dx[:, ky:ky + g["s"] * g["oh"]:g["s"],
               kx:kx + g["s"] * g["ow"]:g["s"], :] += d[:, :, :, ky, kx, :]
    return dx


# ------------------------------------------------------------------ training
def train(arch, geo, xq, y, mom=0.9, bs=64, seed=0):
    """Float training on inputs scaled to [0, 1]. Returns [(W, b)] per layer,
    W shaped (out_ch, kh*kw*c)."""
    rng = np.random.default_rng(seed)
    x = (xq.astype(np.float32) / 127.0).reshape(-1, *arch["input"])
    params = []
    for g in geo:
        fan_in = g["kh"] * g["kw"] * g["c"]
        params.append(rng.normal(0, np.sqrt(2 / fan_in), (g["oc"], fan_in)).astype(np.float32))
        params.append(np.zeros(g["oc"], np.float32))
    vel = [np.zeros_like(p) for p in params]
    n, lr = len(x), arch["lr"]
    for ep in range(arch["epochs"]):
        perm = rng.permutation(n)
        loss_sum = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            a, yb = x[idx], y[idx]
            cache = []
            for li, g in enumerate(geo):
                cols = im2col(a, g)
                z = cols @ params[2 * li].T + params[2 * li + 1]
                cache.append((cols, z))
                a = z if li == len(geo) - 1 else np.maximum(z, 0)
            z = a.reshape(len(idx), -1)
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            loss_sum += -np.log(p[np.arange(len(idx)), yb] + 1e-9).sum()
            p[np.arange(len(idx)), yb] -= 1
            dz = (p / len(idx)).reshape(cache[-1][1].shape)
            grads = [None] * len(params)
            for li in reversed(range(len(geo))):
                g = geo[li]
                cols, z = cache[li]
                if li != len(geo) - 1:
                    dz = dz * (z > 0)
                flat = dz.reshape(-1, g["oc"])
                grads[2 * li] = flat.T @ cols.reshape(-1, cols.shape[-1])
                grads[2 * li + 1] = flat.sum(0)
                if li:
                    dz = col2im(dz @ params[2 * li], g, g["c"])
            for prm, v, gr in zip(params, vel, grads):
                v *= mom
                v -= lr * gr
                prm += v
        print(f"epoch {ep + 1:2d}  loss {loss_sum / n:.4f}")
        lr *= 0.8
    return [(params[2 * i], params[2 * i + 1]) for i in range(len(geo))]


def float_predict(arch, geo, fparams, xq):
    a = (xq.astype(np.float32) / 127.0).reshape(-1, *arch["input"])
    for li, (g, (w, b)) in enumerate(zip(geo, fparams)):
        z = im2col(a, g) @ w.T + b
        a = z if li == len(geo) - 1 else np.maximum(z, 0)
    return a.reshape(len(a), -1).argmax(1)


# -------------------------------------------------------------- quantization
def int_layer(a, g, q, last):
    acc = im2col(a, g) @ q["w"].T + q["b"]
    assert np.abs(acc).max() < 2**31
    if last:
        return acc
    return np.clip((acc * q["M"] + (1 << (q["S"] - 1))) >> q["S"], 0, 127)


def quantize(arch, geo, fparams, calib_xq):
    """Post-training quantization, calibrated layer by layer on integer
    activations so the scales match what the hardware will actually see."""
    a = calib_xq.reshape(-1, *arch["input"]).astype(np.int64)
    s_in = 1 / 127.0
    qparams = []
    for li, (g, (w, b)) in enumerate(zip(geo, fparams)):
        last = li == len(geo) - 1
        s_w = np.abs(w).max() / 127
        q = dict(w=np.round(w / s_w).astype(np.int64),
                 b=np.round(b / (s_w * s_in)).astype(np.int64), M=0, S=0)
        if not last:
            acc = im2col(a, g) @ q["w"].T + q["b"]
            s_out = np.percentile(np.maximum(acc, 0) * (s_w * s_in), 99.9) / 127
            f = s_w * s_in / s_out      # ~= M / 2^S with M in [2^14, 2^15)
            while f * (1 << q["S"]) < (1 << 14):
                q["S"] += 1
            q["M"] = int(round(f * (1 << q["S"])))
            assert 0 < q["M"] < (1 << 16) and 1 <= q["S"] < 48
            a = int_layer(a, g, q, False)
            s_in = s_out
        qparams.append(q)
    return qparams


def golden(arch, geo, qparams, xq):
    """Integer-only reference; mirrors the RTL datapath bit for bit."""
    a = xq.reshape(-1, *arch["input"]).astype(np.int64)
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


def cfg_word(row_len, n_rows, row_skip, step_x, step_y, out_w, out_h, out_ch,
             w_base, b_base, mult, shift, last):
    fields = [(row_len, 12), (n_rows, 4), (row_skip, 12), (step_x, 12),
              (step_y, 16), (out_w, 8), (out_h, 8), (out_ch, 12),
              (w_base, 20), (b_base, 12), (mult, 16), (shift, 6), (last, 1)]
    word, pos = 0, 0
    for value, width in fields:
        assert 0 <= value < (1 << width), (value, width)
        word |= value << pos
        pos += width
    return word


def compile_program(geo, qparams):
    """Turn the network into the accelerator's memory images."""
    weights, biases, cfg = [], [], []
    for li, (g, q) in enumerate(zip(geo, qparams)):
        last = li == len(geo) - 1
        cw = -(-g["c"] // P)                   # words per input pixel
        assert last or g["oh"] * g["ow"] == 1 or g["oc"] % P == 0, \
            "conv layers need out_ch % 8 == 0"
        w = q["w"].reshape(g["oc"], g["kh"], g["kw"], g["c"])
        w_words = pack_words(pad_c(w).reshape(g["oc"], -1))
        cfg.append(cfg_word(
            row_len=g["kw"] * cw, n_rows=g["kh"], row_skip=(g["w"] - g["kw"]) * cw,
            step_x=g["s"] * cw, step_y=g["s"] * g["w"] * cw,
            out_w=g["ow"], out_h=g["oh"], out_ch=g["oc"],
            w_base=sum(len(x) for x in weights), b_base=sum(len(x) for x in biases),
            mult=q["M"], shift=q["S"], last=int(last)))
        weights.append(w_words)
        biases.append(q["b"])
    return np.concatenate(weights), np.concatenate(biases), cfg


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

    weights, biases, cfg = compile_program(geo, qparams)
    images = pad_c(xte[:n_test].reshape(n_test, *arch["input"])).reshape(n_test, -1)
    in_words = images.shape[1] // P
    write_hex(out / "weights.hex", weights, 16)
    write_hex(out / "bias.hex", biases, 8)
    write_hex(out / "cfg.hex", cfg, 36)
    write_hex(out / "meta.hex", [in_words, len(cfg), logits.shape[1]], 8)
    write_hex(out / "images.hex", pack_words(images), 16)
    write_hex(out / "labels.hex", yte[:n_test], 2)
    write_hex(out / "golden_class.hex", pred[:n_test], 2)
    write_hex(out / "golden_logits.hex", logits[:n_test].reshape(-1), 8)
    macs = sum(g["oh"] * g["ow"] * g["oc"] * g["kh"] * g["kw"] * g["c"] for g in geo)
    print(f"weight words   : {len(weights)}  ({len(weights) * 8} bytes)")
    print(f"biases         : {len(biases)}")
    print(f"input words    : {in_words}")
    print(f"multiply-adds  : {macs} per inference")
    print(f"wrote {n_test} test vectors -> {out}")


if __name__ == "__main__":
    main()

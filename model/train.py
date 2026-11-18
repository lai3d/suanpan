"""Train a small MNIST network with numpy and compile it for the accelerator.

    python model/train.py [--arch cnn|mlp|stress] [--n-test N] [--onnx FILE]

Architectures
    mlp:  784 -> fc 64 (ReLU) -> fc 10
    cnn:  28x28x1 -> conv4x4/2 x8 (ReLU) -> conv3x3/2 x16 (ReLU) -> fc 10
          Trained directly in its space-to-depth form: a 2x2 convolution over
          a 14x14x4 input, which is the same arithmetic.
    stress: hardware corner cases (padding, pooling, 1x1 conv, odd fc width),
          not meant for accuracy

Quantization, the integer golden model and the memory-image export live in
nnc.py and are shared with the ONNX path (tools/compile.py). --onnx also
writes the trained float model as ONNX, in its plain image-space form.
"""
import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

import nnc

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}

# "input" is the tensor the first layer reads. See nnc.py for the layer specs.
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
    """MNIST as flat uint8 images and labels; downloaded on first use."""
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


def prep(arch, x):
    """Flat 784-pixel images -> the (N, H, W, C) tensor the first layer reads."""
    if arch.get("s2d"):
        return nnc.s2d_input(x.reshape(len(x), 28, 28, 1), arch["s2d"])
    return x.reshape(-1, *arch["input"])


def col2im(dcols, g):
    n, p, c = dcols.shape[0], g["p"], g["c"]
    d = dcols.reshape(n, g["oh"], g["ow"], g["kh"], g["kw"], c)
    dx = np.zeros((n, g["h"] + 2 * p, g["w"] + 2 * p, c), dcols.dtype)
    for ky in range(g["kh"]):
        for kx in range(g["kw"]):
            dx[:, ky:ky + g["s"] * g["oh"]:g["s"],
               kx:kx + g["s"] * g["ow"]:g["s"], :] += d[:, :, :, ky, kx, :]
    return dx[:, p:p + g["h"], p:p + g["w"], :]


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
                    win = nnc.pool_windows(a, g)
                    arg = win.argmax(3)
                    cache.append(arg)
                    a = np.take_along_axis(win, arg[:, :, :, None, :], 3)[:, :, :, 0, :]
                else:
                    cols = nnc.im2col(a, g)
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


def plain_form(arch, fparams):
    """The network as a framework would describe it: 28x28x1 image input and
    no space-to-depth. Same function, different bookkeeping."""
    specs, fparams = list(arch["layers"]), list(fparams)
    if arch.get("s2d"):
        _, k, _, oc = specs[0]
        specs[0] = ("conv", 2 * k, 2, oc)
        fparams[0] = (nnc.s2d_weights_inv(fparams[0][0], 2 * k, 2 * k, 1), fparams[0][1])
    return (28, 28, 1), specs, fparams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS, default="cnn")
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--onnx", type=Path, help="also write the float model as ONNX")
    args = ap.parse_args()
    arch, n_test = ARCHS[args.arch], args.n_test
    geo = nnc.geometry(arch["input"], arch["layers"])
    out = ROOT / "build" / args.arch

    d = load_mnist()
    xtr, xte = (d["train_x"] >> 1).astype(np.int64), (d["test_x"] >> 1).astype(np.int64)
    ytr, yte = d["train_y"].astype(np.int64), d["test_y"].astype(np.int64)

    fparams = train(arch, geo, xtr, ytr)
    fl = nnc.float_forward(geo, fparams, prep(arch, xte.astype(np.float32) / 127.0)).argmax(1)
    qparams = nnc.quantize(geo, fparams, prep(arch, xtr[:10000]))
    info = nnc.export(out, geo, qparams, prep(arch, xte[:n_test]), yte[:n_test])
    logits, pred = nnc.golden(geo, qparams, prep(arch, xte))
    print(f"arch           : {args.arch}")
    print(f"float accuracy : {(fl == yte).mean() * 100:.2f}%")
    print(f"INT8  accuracy : {(pred == yte).mean() * 100:.2f}%")
    print(f"demo image     : test image 0, label {yte[0]}, predicted {pred[0]}")
    print(f"weight words   : {info['weight_words']}  ({info['weight_words'] * 8} bytes)")
    print(f"parameter words: {info['param_words']}")
    print(f"input words    : {info['in_words']}")
    print(f"MAC cycles     : {info['mac_cycles']} per inference")
    print(f"wrote {n_test} test vectors -> {out}")
    if args.onnx:
        import onnx_io
        onnx_io.to_onnx(args.onnx, *plain_form(arch, fparams))
        print(f"wrote float model -> {args.onnx}")


if __name__ == "__main__":
    main()

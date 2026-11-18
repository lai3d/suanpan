"""Train a small MNIST MLP, quantize it to INT8, and export everything the
RTL needs: weight/bias/config memory images, test images, and bit-exact
golden outputs from an integer-only reference model.

Network:  784 -> 64 (ReLU) -> 10
Hardware arithmetic (must match rtl/nn_accel.sv exactly):
    acc  = sum(w_int8 * x_int8) + bias_int32
    hidden: y = clamp((acc * M + 2^(S-1)) >> S, 0, 127)
    last  : logits = acc (raw int32), class = argmax
"""
import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "build" / "mem"
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
P = 8          # MAC lanes == bytes per memory word
HIDDEN = 64


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


def to_q(x_u8):
    """uint8 pixel -> non-negative int8 activation (0..127)."""
    return (x_u8 >> 1).astype(np.int64)


def train(xq, y, epochs=10, lr=0.05, mom=0.9, bs=64, seed=0):
    rng = np.random.default_rng(seed)
    x = xq.astype(np.float32) / 127.0
    w1 = rng.normal(0, np.sqrt(2 / 784), (HIDDEN, 784)).astype(np.float32)
    b1 = np.zeros(HIDDEN, np.float32)
    w2 = rng.normal(0, np.sqrt(2 / HIDDEN), (10, HIDDEN)).astype(np.float32)
    b2 = np.zeros(10, np.float32)
    params = [w1, b1, w2, b2]
    vel = [np.zeros_like(p) for p in params]
    n = len(x)
    for ep in range(epochs):
        perm = rng.permutation(n)
        loss_sum = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = x[idx], y[idx]
            h = np.maximum(xb @ w1.T + b1, 0)
            z = h @ w2.T + b2
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            loss_sum += -np.log(p[np.arange(len(idx)), yb] + 1e-9).sum()
            dz = p
            dz[np.arange(len(idx)), yb] -= 1
            dz /= len(idx)
            dh = (dz @ w2) * (h > 0)
            grads = [dh.T @ xb, dh.sum(0), dz.T @ h, dz.sum(0)]
            for prm, v, g in zip(params, vel, grads):
                v *= mom
                v -= lr * g
                prm += v
        print(f"epoch {ep + 1:2d}  loss {loss_sum / n:.4f}")
        lr *= 0.8
    return w1, b1, w2, b2


def quantize(w1, b1, w2, b2, calib_xq):
    sx = 1 / 127.0
    s1 = np.abs(w1).max() / 127
    w1q = np.round(w1 / s1).astype(np.int64)
    b1q = np.round(b1 / (s1 * sx)).astype(np.int64)
    # calibrate hidden activation range
    h = np.maximum(calib_xq @ w1q.T + b1q, 0) * (s1 * sx)
    sh = np.percentile(h, 99.9) / 127
    # requant factor s1*sx/sh  ~=  M / 2^S  with M in [2^14, 2^15)
    f = s1 * sx / sh
    S = 0
    while f * (1 << S) < (1 << 14):
        S += 1
    M = int(round(f * (1 << S)))
    assert 0 < M < (1 << 16) and 1 <= S < 48
    s2 = np.abs(w2).max() / 127
    w2q = np.round(w2 / s2).astype(np.int64)
    b2q = np.round(b2 / (s2 * sh)).astype(np.int64)
    return dict(w1=w1q, b1=b1q, w2=w2q, b2=b2q, M=M, S=S)


def golden(q, xq):
    """Integer-only reference; mirrors the RTL datapath bit for bit."""
    acc1 = xq @ q["w1"].T + q["b1"]
    h = np.clip((acc1 * q["M"] + (1 << (q["S"] - 1))) >> q["S"], 0, 127)
    logits = h @ q["w2"].T + q["b2"]
    assert np.abs(acc1).max() < 2**31 and np.abs(logits).max() < 2**31
    return logits, logits.argmax(1)


def pack_words(a):
    """int8 matrix (rows, n) -> list of 64-bit words, each row zero-padded to P."""
    rows, n = a.shape
    pad = (-n) % P
    a = np.pad(a, ((0, 0), (0, pad))).astype(np.int64) & 0xFF
    a = a.reshape(rows, -1, P)
    shifts = np.arange(P, dtype=np.uint64) * np.uint64(8)
    return (a.astype(np.uint64) << shifts).sum(-1, dtype=np.uint64).reshape(-1)


def write_hex(path, values, digits):
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v) & ((1 << (4 * digits)) - 1):0{digits}x}\n")


def cfg_word(in_words, out_size, w_base, b_base, mult=0, shift=0, last=0):
    return (in_words | out_size << 12 | w_base << 24 | b_base << 44
            | mult << 56 | shift << 72 | last << 78)


def main():
    n_test = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    d = load_mnist()
    xtr, xte = to_q(d["train_x"]), to_q(d["test_x"])
    ytr, yte = d["train_y"].astype(np.int64), d["test_y"].astype(np.int64)

    w1, b1, w2, b2 = train(xtr, ytr)
    fl = (np.maximum(xte / 127.0 @ w1.T + b1, 0) @ w2.T + b2).argmax(1)
    q = quantize(w1, b1, w2, b2, xtr[:10000])
    logits, pred = golden(q, xte)
    print(f"float accuracy : {(fl == yte).mean() * 100:.2f}%")
    print(f"INT8  accuracy : {(pred == yte).mean() * 100:.2f}%   (M={q['M']}, S={q['S']})")

    OUT.mkdir(parents=True, exist_ok=True)
    w1w, w2w = pack_words(q["w1"]), pack_words(q["w2"])
    write_hex(OUT / "weights.hex", np.concatenate([w1w, w2w]), 16)
    write_hex(OUT / "bias.hex", np.concatenate([q["b1"], q["b2"]]), 8)
    write_hex(OUT / "cfg.hex", [
        cfg_word(784 // P, HIDDEN, 0, 0, q["M"], q["S"]),
        cfg_word(HIDDEN // P, 10, len(w1w), HIDDEN, last=1),
    ], 20)
    write_hex(OUT / "images.hex", pack_words(xte[:n_test]), 16)
    write_hex(OUT / "labels.hex", yte[:n_test], 2)
    write_hex(OUT / "golden_class.hex", pred[:n_test], 2)
    write_hex(OUT / "golden_logits.hex", logits[:n_test].reshape(-1), 8)
    print(f"wrote {n_test} test vectors, {len(w1w) + len(w2w)} weight words -> {OUT}")


if __name__ == "__main__":
    main()

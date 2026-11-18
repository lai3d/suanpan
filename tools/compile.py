"""Compile an ONNX model for the suanpan accelerator.

    python tools/compile.py model.onnx --out build/mynet --data mnist
    python tools/compile.py model.onnx --out build/mynet --calib calib.npz --test test.npz

The model must take images scaled to [0, 1]. Calibration data sets the
quantization scales; test data becomes the vectors the RTL testbench checks.
Both are .npz files with `x` (N, C, H, W) or (N, H, W), either float in
[0, 1] or uint8, and optionally labels `y`. `--data mnist` uses the MNIST
training set for calibration and its test set for the vectors.

Output: the memory images and golden vectors in --out, ready for
    make sim MEM=<out>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

import nnc          # noqa: E402
import onnx_io      # noqa: E402


def to_int_nhwc(x):
    """Images as the hardware sees them: (N, H, W, C) integers 0..127."""
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[:, None]
    if x.ndim != 4:
        raise nnc.CompileError(f"data has shape {x.shape}; expected (N, C, H, W)")
    if x.dtype == np.uint8:
        x = x >> 1
    else:
        if x.min() < -1e-6 or x.max() > 1 + 1e-6:
            raise nnc.CompileError("float data must be scaled to [0, 1]; "
                                   f"found [{x.min():.3g}, {x.max():.3g}]")
        x = np.round(x * 127)
    return np.ascontiguousarray(x.transpose(0, 2, 3, 1)).astype(np.int64)


def load_npz(path):
    d = np.load(path)
    return to_int_nhwc(d["x"]), (d["y"].astype(np.int64) if "y" in d else None)


def load_data(args):
    if args.data == "mnist":
        import train
        d = train.load_mnist()
        calib = to_int_nhwc(d["train_x"][:args.n_calib].reshape(-1, 1, 28, 28))
        test = to_int_nhwc(d["test_x"][:args.n_test].reshape(-1, 1, 28, 28))
        return calib, test, d["test_y"][:args.n_test].astype(np.int64)
    if not (args.calib and args.test):
        raise nnc.CompileError("give --data mnist, or both --calib and --test")
    calib, _ = load_npz(args.calib)
    test, labels = load_npz(args.test)
    return calib[:args.n_calib], test[:args.n_test], None if labels is None else labels[:args.n_test]


def compile_onnx(args):
    input_shape, specs, fparams = onnx_io.from_onnx(args.model)
    raw_shape = input_shape
    calib, test, labels = load_data(args)
    if calib.shape[1:] != raw_shape or test.shape[1:] != raw_shape:
        raise nnc.CompileError(f"model input is {raw_shape} (H, W, C) but the data is "
                               f"{calib.shape[1:]} / {test.shape[1:]}")
    notes = []

    # float reference on the model exactly as imported
    geo0 = nnc.geometry(input_shape, specs)
    float_pred = nnc.float_forward(geo0, fparams, test / 127.0).argmax(1)

    # rewrites: same function, cheaper layout
    n_ch = [s[3] for s in specs if s[0] == "conv"]
    specs, fparams = nnc.pad_conv_channels(input_shape, specs, fparams)
    if [s[3] for s in specs if s[0] == "conv"] != n_ch:
        notes.append("convolution channels rounded up to a multiple of 8 with zero filters")
    input_shape, flatten = nnc.rewrite_fc_input(input_shape, specs)
    if flatten:
        calib, test = (a.reshape(len(a), 1, 1, -1) for a in (calib, test))
        notes.append("input flattened for the leading fully-connected layer")
    input_shape, specs, fparams, s2d = nnc.rewrite_s2d(input_shape, specs, fparams)
    if s2d:
        calib, test = nnc.s2d_input(calib, s2d), nnc.s2d_input(test, s2d)
        notes.append("first convolution rewritten onto a space-to-depth input")

    geo = nnc.geometry(input_shape, specs)
    qparams = nnc.quantize(geo, fparams, calib)
    info = nnc.export(args.out, geo, qparams, test, labels)

    print(f"model          : {args.model}")
    print(f"input          : {raw_shape[0]}x{raw_shape[1]}x{raw_shape[2]} (H x W x C)")
    for g in geo:
        shape = f"{g['oh']}x{g['ow']}x{g['oc']}"
        if g["kind"] == "pool":
            print(f"  pool {g['kh']}x{g['kw']}/{g['s']:<14} -> {shape}")
        elif g["kind"] == "fc":
            print(f"  fc {g['oc']:<19} -> {shape}")
        else:
            print(f"  conv {g['kh']}x{g['kw']}/{g['s']} pad {g['p']} x{g['oc']:<5} -> {shape}")
    for note in notes:
        print(f"rewrite        : {note}")
    agree = (float_pred == info["pred"]).mean() * 100
    if labels is not None:
        print(f"float accuracy : {(float_pred == labels).mean() * 100:.2f}%")
        print(f"INT8  accuracy : {(info['pred'] == labels).mean() * 100:.2f}%")
    print(f"INT8 vs float  : same class on {agree:.2f}% of {len(test)} test images")
    print(f"weight words   : {info['weight_words']}  ({info['weight_words'] * 8} bytes)")
    print(f"parameter words: {info['param_words']}")
    print(f"largest map    : {info['act_words']} words")
    print(f"MAC cycles     : {info['mac_cycles']} per inference")
    print(f"wrote {len(test)} test vectors -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path, help="ONNX model")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--data", choices=["mnist"], help="built-in dataset")
    ap.add_argument("--calib", type=Path, help=".npz with calibration images")
    ap.add_argument("--test", type=Path, help=".npz with test images")
    ap.add_argument("--n-calib", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=10000)
    args = ap.parse_args()
    try:
        compile_onnx(args)
    except nnc.CompileError as e:
        sys.exit(f"compile error: {e}")


if __name__ == "__main__":
    main()

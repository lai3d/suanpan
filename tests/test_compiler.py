"""Tests for the compiler that need no RTL simulation.

    python tests/test_compiler.py

Run `make model ARCH=<arch>` for cnn, mlp and stress first: the round-trip
test compares against their exported files.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

import nnc          # noqa: E402
import onnx_io      # noqa: E402

rng = np.random.default_rng(1)
TMP = Path(tempfile.mkdtemp())


def rand(*shape):
    return rng.normal(0, 0.5, shape).astype(np.float32)


def save(nodes, inits, in_shape, name="m.onnx"):
    graph = helper.make_graph(
        nodes, "t",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", *in_shape])],
        [helper.make_tensor_value_info(nodes[-1].output[0], TensorProto.FLOAT, None)],
        [numpy_helper.from_array(a, n) for n, a in inits.items()])
    path = TMP / name
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))
    return path


def conv_ref(x, w, b, stride=1, pad=0):
    """Direct NCHW convolution, written independently of nnc.im2col."""
    n, _, h, wd = x.shape
    oc, _, kh, kw = w.shape
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh, ow = (h + 2 * pad - kh) // stride + 1, (wd + 2 * pad - kw) // stride + 1
    y = np.zeros((n, oc, oh, ow), np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = x[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            y[:, :, i, j] = np.tensordot(patch, w, ([1, 2, 3], [1, 2, 3])) + b
    return y


def run_imported(path, x_nchw):
    shape, specs, fparams = onnx_io.from_onnx(path)
    geo = nnc.geometry(shape, specs)
    return nnc.float_forward(geo, fparams, x_nchw.transpose(0, 2, 3, 1))


def expect_error(fragment, fn):
    try:
        fn()
    except nnc.CompileError as e:
        assert fragment in str(e), f"wrong message: {e}"
        return
    raise AssertionError(f"expected a CompileError mentioning {fragment!r}")


# --------------------------------------------------------------------- tests
def test_import_matches_reference():
    """Conv (padded, strided) + BatchNorm + ReLU + MaxPool + Flatten + Gemm,
    checked against an independent numpy implementation."""
    w1, b1 = rand(8, 3, 3, 3), rand(8)
    gamma, beta, mean = rand(8) + 1.5, rand(8), rand(8)
    var = np.abs(rand(8)) + 0.5
    w2, b2 = rand(5, 8 * 3 * 3), rand(5)
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c"], kernel_shape=[3, 3],
                         strides=[2, 2], pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization", ["c", "g", "be", "mu", "va"], ["bn"], epsilon=1e-3),
        helper.make_node("Relu", ["bn"], ["r"]),
        helper.make_node("MaxPool", ["r"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Flatten", ["p"], ["f"], axis=1),
        helper.make_node("Gemm", ["f", "w2", "b2"], ["y"], transB=1),
    ]
    path = save(nodes, dict(w1=w1, b1=b1, g=gamma, be=beta, mu=mean, va=var, w2=w2, b2=b2),
                (3, 12, 12))
    x = rng.random((4, 3, 12, 12)).astype(np.float32)
    y = conv_ref(x, w1, b1, 2, 1)
    y = (y - mean[None, :, None, None]) / np.sqrt(var + 1e-3)[None, :, None, None]
    y = np.maximum(y * gamma[None, :, None, None] + beta[None, :, None, None], 0)
    y = y.reshape(4, 8, 3, 2, 3, 2).max((3, 5))
    ref = y.reshape(4, -1) @ w2.T + b2
    assert np.allclose(run_imported(path, x), ref, atol=1e-4)


def test_gemm_orientations():
    """transB=1 (PyTorch), transB=0, and MatMul + Add give the same layer."""
    w, b = rand(4, 6), rand(4)
    x = rng.random((3, 6, 1, 1)).astype(np.float32)
    ref = x.reshape(3, 6) @ w.T + b
    flat = helper.make_node("Flatten", ["input"], ["f"], axis=1)
    variants = [
        ([flat, helper.make_node("Gemm", ["f", "w", "b"], ["y"], transB=1)], dict(w=w, b=b)),
        ([flat, helper.make_node("Gemm", ["f", "w", "b"], ["y"])], dict(w=w.T.copy(), b=b)),
        ([flat, helper.make_node("MatMul", ["f", "w"], ["m"]),
          helper.make_node("Add", ["m", "b"], ["y"])], dict(w=w.T.copy(), b=b)),
    ]
    for nodes, inits in variants:
        assert np.allclose(run_imported(save(nodes, inits, (6, 1, 1)), x), ref, atol=1e-5)


def test_rewrites_preserve_function():
    """Channel padding and the space-to-depth rewrite must not change outputs."""
    shape = (12, 12, 1)
    specs = [("conv", 4, 2, 6), ("pool", 2, 1), ("conv", 3, 1, 8, 1), ("fc", 7), ("fc", 3)]
    geo = nnc.geometry(shape, specs)
    fparams = [None if g["kind"] == "pool" else (rand(g["oc"], g["kh"] * g["kw"] * g["c"]), rand(g["oc"]))
               for g in geo]
    x = rng.random((5, 12, 12, 1)).astype(np.float32)
    ref = nnc.float_forward(geo, fparams, x)

    specs2, fparams2 = nnc.pad_conv_channels(shape, specs, fparams)
    assert specs2[0][3] == 8 and specs2[2][3] == 8
    assert np.allclose(nnc.float_forward(nnc.geometry(shape, specs2), fparams2, x), ref, atol=1e-4)

    shape3, specs3, fparams3, s2d = nnc.rewrite_s2d(shape, specs2, fparams2)
    assert s2d == 2 and shape3 == (6, 6, 4) and specs3[0][1] == (2, 2)
    out = nnc.float_forward(nnc.geometry(shape3, specs3), fparams3, nnc.s2d_input(x))
    assert np.allclose(out, ref, atol=1e-4)

    # and the inverse weight mapping really is the inverse
    w = rand(8, 16)
    assert np.array_equal(nnc.s2d_weights(nnc.s2d_weights_inv(w, 4, 4, 1), 4, 4, 1), w)


def hardware_walk(geo, weights, cfg, stored):
    """Layer 0 as the accelerator executes it, from the exported images only:
    walk each window with the descriptor's pointer steps and sum word-by-word
    dot products. Independent of nnc.im2col and of the RTL."""
    f, pos = {}, 0
    for name, width in nnc.CFG_FIELDS:
        f[name] = (cfg[0] >> pos) & ((1 << width) - 1)
        pos += width
    def lanes(word):                      # 8 signed bytes of a 64-bit word
        b = np.array([(int(word) >> (8 * k)) & 0xFF for k in range(8)], np.int64)
        return np.where(b > 127, b - 256, b)
    wl = np.array([lanes(w) for w in weights])
    mem = np.array([lanes(w) for w in stored])
    out = np.zeros((f["out_h"], f["out_w"], f["out_ch"]), np.int64)
    row_base = f["org"]
    for oy in range(f["out_h"]):
        pix_base = row_base
        for ox in range(f["out_w"]):
            w_ptr = f["w_base"]
            for oc in range(f["out_ch"]):
                a_ptr, acc = pix_base, 0
                for kr in range(f["kh"]):
                    for kx in range(f["kw"]):
                        for kcw in range(f["rd_words"]):
                            acc += int(wl[w_ptr] @ mem[a_ptr])
                            w_ptr += 1
                            last_w = kcw == f["rd_words"] - 1
                            if last_w and kx == f["kw"] - 1:
                                a_ptr += f["row_step"]
                            elif last_w:
                                a_ptr += f["pix_step"]
                            else:
                                a_ptr += 1
                out[oy, ox, oc] = acc
            pix_base += f["step_x"]
        row_base += f["step_y"]
    return out


def test_overlap_packing_is_exact():
    """Overlap packing changes only where bytes live. Executing the exported
    layer table the way the hardware does must give the reference sums."""
    for shape, spec in [((9, 9, 3), ("conv", 3, 1, 8, 1)),     # CIFAR-like, 2 pixels / word
                        ((10, 10, 1), ("conv", 5, 1, 8, 2)),    # LeNet-like, 8 pixels / word
                        ((9, 9, 1), ("conv", 3, 2, 8, 1)),      # stride 2 with padding
                        ((8, 8, 4), ("conv", 2, 1, 8)),         # after space-to-depth
                        ((8, 8, 2), ("conv", 5, 1, 8, 2))]:     # 4 pixels / word, 2 words / row
        specs = [spec, ("fc", 3)]
        geo = nnc.geometry(shape, specs)
        assert nnc.apply_overlap(geo), (shape, spec)
        g = geo[0]
        fan = g["kh"] * g["kw"] * g["c"]
        fparams = [(rand(8, fan), rand(8)), (rand(3, g["oh"] * g["ow"] * 8), rand(3))]
        a = rng.integers(0, 128, (40, *shape))
        q = nnc.quantize(geo, fparams, a)
        weights, _, cfg = nnc.compile_program(geo, q)
        stored = nnc.pack_words(nnc.pad_c(nnc.overlap_pack(a[:1], g)).reshape(1, -1))
        ref = nnc.im2col(a[:1].astype(np.int64), g)[0] @ q[0]["w"].T        # no bias
        got = hardware_walk(geo, weights, cfg, stored)
        assert np.array_equal(got, ref), (shape, spec)
        # and it really is shorter than one word per pixel
        f_kw = cfg[0] & 0xF
        assert f_kw < g["kw"]

    # not applied when it cannot help: 8 channels fill a word, 1x1 has no row
    for shape, spec in [((8, 8, 8), ("conv", 3, 1, 8)), ((8, 8, 1), ("conv", 1, 1, 8))]:
        assert not nnc.apply_overlap(nnc.geometry(shape, [spec, ("fc", 3)]))


def test_s2d_not_applied_when_unsafe():
    for specs in ([("conv", 3, 2, 8), ("fc", 3)],          # odd kernel
                  [("conv", 4, 1, 8), ("fc", 3)],          # stride 1
                  [("conv", 4, 2, 8, 1), ("fc", 3)]):      # padded
        fp = [(rand(8, specs[0][1] ** 2), rand(8)), None]
        assert nnc.rewrite_s2d((12, 12, 1), specs, fp)[3] is None
    fp = [(rand(8, 16 * 4), rand(8)), None]                # 4 channels -> 16 > 8
    assert nnc.rewrite_s2d((12, 12, 4), [("conv", 4, 2, 8), ("fc", 3)], fp)[3] is None


def test_quantized_tracks_float():
    """The integer model must pick the float model's class wherever the float
    model is not nearly tied. (A random network on noise has many near-ties,
    so overall agreement is not a meaningful threshold.)"""
    shape, specs = (8, 8, 1), [("conv", 3, 1, 8, 1), ("pool", 2, 2), ("fc", 4)]
    geo = nnc.geometry(shape, specs)
    fparams = [(rand(8, 9), rand(8)), None, (rand(4, 4 * 4 * 8), rand(4))]
    a = rng.integers(0, 128, (400, 8, 8, 1))
    q = nnc.quantize(geo, fparams, a)
    _, pred = nnc.golden(geo, q, a)
    logits = nnc.float_forward(geo, fparams, a / 127.0)
    top = np.sort(logits, 1)
    clear = (top[:, -1] - top[:, -2]) > 0.25 * logits.std()
    agree = (pred == logits.argmax(1))
    assert clear.mean() > 0.5, clear.mean()
    assert agree[clear].mean() > 0.99, agree[clear].mean()
    assert agree.mean() > 0.9, agree.mean()


def test_errors_are_explained():
    w, b = rand(8, 1, 3, 3), rand(8)
    fw, fb = rand(3, 8 * 6 * 6), rand(3)
    conv = helper.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[3, 3])
    relu = helper.make_node("Relu", ["c"], ["r"])
    flat = lambda src: helper.make_node("Flatten", [src], ["f"], axis=1)
    gemm = helper.make_node("Gemm", ["f", "fw", "fb"], ["y"], transB=1)
    inits = dict(w=w, b=b, fw=fw, fb=fb)

    avg = helper.make_node("AveragePool", ["r"], ["p"], kernel_shape=[2, 2])
    expect_error("unsupported operator AveragePool",
                 lambda: onnx_io.from_onnx(save([conv, relu, avg, flat("p"), gemm], inits, (1, 8, 8))))
    expect_error("not followed by ReLU",
                 lambda: onnx_io.from_onnx(save([conv, flat("c"), gemm], inits, (1, 8, 8))))
    dil = helper.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[3, 3], dilations=[2, 2])
    expect_error("dilation", lambda: onnx_io.from_onnx(save([dil, relu, flat("r"), gemm], inits, (1, 8, 8))))
    lop = helper.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 0, 1, 0])
    expect_error("padding must be the same",
                 lambda: onnx_io.from_onnx(save([lop, relu, flat("r"), gemm], inits, (1, 8, 8))))

    # too big for the hardware: the message must name what does not fit
    big = nnc.geometry((64, 64, 8), [("conv", 3, 1, 8, 1), ("fc", 10)])
    req = dict(layers=2, weight_words=10, param_words=18, act_words=64 * 64, in_words=4096, n_out=10)
    expect_error("A_AW=12", lambda: nnc.check_limits(req))
    assert big[0]["oh"] == 64
    expect_error("9 layers", lambda: nnc.check_limits(dict(req, act_words=10, layers=9)))


def test_roundtrip_is_byte_identical():
    """train.py's float model -> ONNX -> compile.py must reproduce train.py's
    own export exactly, for every architecture."""
    files = ["weights", "params", "cfg", "meta", "images", "labels",
             "golden_class", "golden_logits", "demo_image"]
    for arch in ("cnn", "mlp", "stress"):
        src = ROOT / "build" / arch
        assert (src / "model.onnx").exists(), f"run `make model ARCH={arch}` first"
        out = TMP / arch
        # stress is exported without overlap packing (see ARCHS in train.py)
        extra = ["--no-overlap"] if arch == "stress" else []
        subprocess.run([sys.executable, str(ROOT / "tools" / "compile.py"), str(src / "model.onnx"),
                        "--out", str(out), "--data", "mnist", *extra], check=True, capture_output=True)
        for f in files:
            assert (out / f"{f}.hex").read_bytes() == (src / f"{f}.hex").read_bytes(), \
                f"{arch}/{f}.hex differs after the ONNX round trip"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"{len(tests)} tests passed")

"""ONNX front end and back end for the suanpan compiler.

from_onnx reads a model exported by PyTorch, TensorFlow, etc. into the
compiler's own description (see nnc.py). to_onnx writes one back out, which
is how the compiler is tested against itself.

Supported graph: a chain, optionally with residual connections, of
    Conv            2-D, group 1, no dilation, same padding on every side
    BatchNormalization   directly after a Conv; folded into it
    Relu            after every hidden Conv / Gemm (before or after a MaxPool)
    MaxPool         no padding, floor rounding
    Add             a residual connection: Conv -> [BatchNormalization] -> Add
                    -> Relu, where the other input of Add is an earlier tensor
                    that has already been through ReLU (an identity shortcut)
    Flatten / Reshape    before the first Gemm
    Gemm, or MatMul followed by Add
    Dropout, Identity, and a final Softmax / LogSoftmax are ignored
The model must take images scaled to [0, 1] with no other normalisation, and
its last layer must produce the logits.
"""
import numpy as np

from nnc import CompileError, geometry


def to_onnx(path, input_shape, specs, fparams):
    """Write the float network as ONNX. Input is NCHW, one node per layer."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    geo = geometry(input_shape, specs)
    h, w, c = input_shape
    nodes, inits = [], []
    cur, flat = "input", False

    def const(name, array):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(array, np.float32), name))
        return name

    for li, (g, fp) in enumerate(zip(geo, fparams)):
        last = li == len(geo) - 1
        out = f"t{li}"
        if g["kind"] == "pool":
            nodes.append(helper.make_node("MaxPool", [cur], [out],
                                          kernel_shape=[g["kh"], g["kw"]],
                                          strides=[g["s"], g["s"]]))
            cur = out
            continue
        if g["kind"] == "conv":
            wt = fp[0].reshape(g["oc"], g["kh"], g["kw"], g["c"]).transpose(0, 3, 1, 2)
            nodes.append(helper.make_node(
                "Conv", [cur, const(f"w{li}", wt), const(f"b{li}", fp[1])], [out],
                kernel_shape=[g["kh"], g["kw"]], strides=[g["s"], g["s"]],
                pads=[g["p"]] * 4))
        else:
            if not flat:
                nodes.append(helper.make_node("Flatten", [cur], [f"f{li}"], axis=1))
                cur, flat = f"f{li}", True
            # ONNX flattens (c, h, w); the compiler's order is (h, w, c)
            wt = fp[0].reshape(g["oc"], g["h"], g["w"], g["c"]).transpose(0, 3, 1, 2)
            nodes.append(helper.make_node(
                "Gemm", [cur, const(f"w{li}", wt.reshape(g["oc"], -1)),
                         const(f"b{li}", fp[1])], [out], transB=1))
        cur = out
        if not last:
            nodes.append(helper.make_node("Relu", [cur], [f"r{li}"]))
            cur = f"r{li}"

    n_out = geo[-1]["oh"] * geo[-1]["ow"] * geo[-1]["oc"]
    graph = helper.make_graph(
        nodes, "suanpan",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", c, h, w])],
        [helper.make_tensor_value_info(cur, TensorProto.FLOAT, ["N", n_out])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def from_onnx(path):
    """Read an ONNX model.
    Returns (input_shape (H, W, C), specs, fparams, links); see nnc.py."""
    import onnx
    from onnx import helper, numpy_helper

    graph = onnx.load(str(path)).graph
    consts = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            consts[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)

    inputs = [i for i in graph.input if i.name not in consts]
    if len(inputs) != 1:
        raise CompileError(f"expected one model input, found {len(inputs)}")
    dims = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim][1:]
    if len(dims) == 3:
        c, h, w = dims
    elif len(dims) == 1:
        c, h, w = dims[0], 1, 1
    else:
        raise CompileError(f"unsupported input shape {dims}; expected (N, C, H, W)")
    if min(c, h, w) < 1:
        raise CompileError("the input shape must be fixed apart from the batch size")
    input_shape = (h, w, c)

    # Every ONNX value maps to a compiler tensor (tid: 0 is the input, i+1 the
    # output of layer i) plus what is still owed before it is that tensor:
    #   pending  layer whose ReLU has not been seen yet (None: none owed)
    #   raw      set while the value is the bare output of conv / fc layer
    #            `raw`, which is when BatchNormalization or Add can be folded in
    vals = {inputs[0].name: dict(tid=0, pending=None, raw=None, shape=(c, h, w),
                                 flat=(h, w) == (1, 1))}
    specs, fparams, relu, links = [], [], [], []
    ended = False                         # a final Softmax has been seen

    def need(cond, node, msg):
        if not cond:
            raise CompileError(f"{node.op_type} '{node.name or node.output[0]}': {msg}")

    def weight(node, i):
        need(len(node.input) > i and node.input[i] in consts, node,
             "weights must be constants stored in the model")
        return np.asarray(consts[node.input[i]], np.float32)

    def value(node, i=0):
        need(node.input[i] in vals, node, f"input '{node.input[i]}' is not produced "
             "by a supported operator")
        return vals[node.input[i]]

    def activated(node, v):
        need(v["pending"] is None, node,
             f"its input comes from layer {v['pending']} without a ReLU; the hardware "
             "applies ReLU after every hidden layer")

    def layer(spec, fp, v, shape, weighted):
        specs.append(spec)
        fparams.append(fp)
        relu.append(False)
        links.append([v["tid"], None])
        li = len(specs) - 1
        return dict(tid=li + 1, shape=shape, flat=shape[1:] == (1, 1) and weighted and spec[0] == "fc",
                    pending=li if weighted else v["pending"], raw=li if weighted else None)

    for node in graph.node:
        op = node.op_type
        at = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if op == "Constant":
            continue
        need(not ended, node, "nothing may follow the final Softmax")
        out = node.output[0]
        if op in ("Dropout", "Identity"):
            vals[out] = value(node)
        elif op == "Conv":
            v = value(node)
            activated(node, v)
            c, h, w = v["shape"]
            need(not v["flat"] or (h, w) == (1, 1), node,
                 "convolution after the tensor was flattened")
            wt = weight(node, 1)
            need(wt.ndim == 4, node, "only 2-D convolution is supported")
            oc, wc, kh, kw = wt.shape
            need(at.get("group", 1) == 1 and wc == c, node,
                 "grouped / depthwise convolution is not supported")
            need(all(d == 1 for d in at.get("dilations", [1, 1])), node,
                 "dilation is not supported")
            need(at.get("auto_pad", b"NOTSET") in (b"NOTSET", "NOTSET", b"VALID", "VALID"),
                 node, "auto_pad must be NOTSET or VALID; export explicit pads")
            st = at.get("strides", [1, 1])
            pads = at.get("pads", [0, 0, 0, 0])
            need(st[0] == st[1], node, "stride must be the same in both directions")
            need(len(set(pads)) == 1, node, "padding must be the same on all four sides")
            b = weight(node, 2) if len(node.input) > 2 else np.zeros(oc, np.float32)
            shape = (oc, (h + 2 * pads[0] - kh) // st[0] + 1, (w + 2 * pads[0] - kw) // st[0] + 1)
            vals[out] = layer(("conv", (kh, kw), st[0], oc, pads[0]),
                              (wt.transpose(0, 2, 3, 1).reshape(oc, -1), b), v, shape, True)
        elif op == "BatchNormalization":
            v = value(node)
            li = v["raw"]
            need(li is not None and specs[li][0] == "conv" and links[li][1] is None, node,
                 "must directly follow a convolution")
            gamma, beta, mean, var = (weight(node, i) for i in range(1, 5))
            k = gamma / np.sqrt(var + at.get("epsilon", 1e-5))
            wt, b = fparams[li]
            fparams[li] = (wt * k[:, None], (b - mean) * k + beta)
            vals[out] = v
        elif op == "Relu":
            v = value(node)
            if v["pending"] is not None:
                relu[v["pending"]] = True
            vals[out] = dict(v, pending=None, raw=None)      # ReLU of a ReLU is itself
        elif op == "MaxPool":
            v = value(node)
            c, h, w = v["shape"]
            need(not v["flat"], node, "pooling after the tensor was flattened")
            ks = at["kernel_shape"]
            st = at.get("strides", [1, 1])
            need(len(ks) == 2 and st[0] == st[1], node, "only square-stride 2-D pooling")
            need(not any(at.get("pads", [0])), node, "padded pooling is not supported")
            need(not at.get("ceil_mode", 0), node, "ceil_mode is not supported")
            need(all(d == 1 for d in at.get("dilations", [1, 1])), node,
                 "dilation is not supported")
            # max pooling commutes with ReLU, so an owed ReLU passes through
            shape = (c, (h - ks[0]) // st[0] + 1, (w - ks[1]) // st[0] + 1)
            vals[out] = layer(("pool", (ks[0], ks[1]), st[0]), None, v, shape, False)
        elif op in ("Flatten", "Reshape"):
            need(op != "Flatten" or at.get("axis", 1) == 1, node, "axis must be 1")
            vals[out] = dict(value(node), flat=True)
        elif op in ("Gemm", "MatMul"):
            v = value(node)
            activated(node, v)
            c, h, w = v["shape"]
            need(v["flat"], node, "the tensor must be flattened first")
            wt = weight(node, 1)
            need(wt.ndim == 2, node, "weight must be a matrix")
            if op == "MatMul":
                wt = wt.T
                b = np.zeros(len(wt), np.float32)
            else:
                need(at.get("alpha", 1.0) == 1.0 and at.get("beta", 1.0) == 1.0
                     and not at.get("transA", 0), node, "alpha, beta and transA must be default")
                if at.get("transB", 0) == 0:
                    wt = wt.T                 # Gemm computes x @ B: B is (in, out)
                b = weight(node, 2) if len(node.input) > 2 else np.zeros(len(wt), np.float32)
            oc = len(wt)
            need(wt.shape[1] == c * h * w, node,
                 f"expects {wt.shape[1]} inputs but the tensor has {c * h * w}")
            # ONNX flattens (c, h, w); the compiler's order is (h, w, c)
            wt = wt.reshape(oc, c, h, w).transpose(0, 2, 3, 1).reshape(oc, -1)
            vals[out] = layer(("fc", oc), (wt, b), v, (oc, 1, 1), True)
        elif op == "Add":
            const = [i for i in (0, 1) if node.input[i] in consts]
            if const:                     # the bias of a MatMul
                v = value(node, 1 - const[0])
                li = v["raw"]
                need(li is not None and specs[li][0] == "fc" and links[li][1] is None, node,
                     "adding a constant is only supported as the bias of a MatMul")
                b = weight(node, const[0])
                fparams[li] = (fparams[li][0], fparams[li][1] + b.reshape(-1))
                vals[out] = v
                continue
            a, b = value(node, 0), value(node, 1)
            # one side is the bare output of a conv / fc layer and absorbs the
            # add; the other is the shortcut and must already be a stored tensor
            if a["raw"] is None or (b["raw"] is not None and a["pending"] is None):
                a, b = b, a
            li = a["raw"]
            need(li is not None and links[li][1] is None, node,
                 "one input must come straight from a convolution or fully-connected "
                 "layer (after its BatchNormalization, before its ReLU)")
            need(b["pending"] is None, node,
                 "the shortcut must be an activated tensor (an identity shortcut); a "
                 "projection shortcut without ReLU is not supported")
            need(a["shape"] == b["shape"], node,
                 f"shapes differ: {a['shape']} and {b['shape']}")
            links[li][1] = b["tid"]
            vals[out] = dict(a, raw=None)             # still owes its ReLU
        elif op in ("Softmax", "LogSoftmax"):
            vals[out] = value(node)                   # argmax is unchanged
            ended = True
        else:
            raise CompileError(
                f"unsupported operator {op} ('{node.name or node.output[0]}'); supported: "
                "Conv, BatchNormalization, Relu, MaxPool, Add, Flatten, Gemm")

    weighted = [i for i, sp in enumerate(specs) if sp[0] != "pool"]
    if not weighted:
        raise CompileError("the model has no convolution or fully-connected layer")
    for i in weighted[:-1]:
        if not relu[i]:
            raise CompileError(f"layer {i} ({specs[i][0]}) is not followed by ReLU; "
                               "the hardware applies ReLU after every hidden layer")
    if relu[weighted[-1]]:
        raise CompileError("the last layer must output raw logits, without ReLU")
    final = vals.get(graph.output[0].name)
    if weighted[-1] != len(specs) - 1 or final is None or final["tid"] != len(specs):
        raise CompileError("the model must end with a convolution or fully-connected layer")
    return input_shape, specs, fparams, [tuple(l) for l in links]

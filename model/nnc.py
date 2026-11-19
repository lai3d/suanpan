"""suanpan network compiler: turns a float network into the accelerator's
memory images.

    float layers -> rewrites -> INT8 quantization -> layer table + memories
                                      |
                                      +-> integer golden model (bit-exact
                                          reference for the RTL)

A network is described by
    input shape   (H, W, C)
    specs         one tuple per layer:
                    ("conv", k, stride, out_ch[, pad])   k is int or (kh, kw)
                    ("pool", k, stride)                  max pooling
                    ("fc", out)
    fparams       per layer (W, b) or None for pooling, with W shaped
                  (out_ch, kh*kw*C) in (ky, kx, c) order
    links         optional, per layer (src, add). Tensor 0 is the network
                  input and tensor i+1 is the output of layer i. `src` is the
                  tensor the layer reads (default: the previous one), `add` a
                  tensor of the output's shape that is added before the ReLU
                  (a residual shortcut), or None.

Hidden conv / fc layers are followed by ReLU; the last layer is not.

Hardware arithmetic (must match rtl/nn_accel.sv exactly):
    acc  = sum(w_int8 * x_int8) + bias_int32
    hidden: y = clamp((acc * M + 2^(S-1)) >> S, 0, 127), M and S per channel
    hidden with a shortcut x (int8 from another layer):
            y = clamp((acc * M + ((x * K) << (S - 14)) + 2^(S-1)) >> S, 0, 127)
    last  : logits = acc (raw int32), class = argmax
    pool  : y = max over the window, per channel

Memory layout: activations are stored HWC with the channel count padded to a
multiple of P=8, so one pixel is a whole number of 64-bit words.
"""
from pathlib import Path

import numpy as np

P = 8  # MAC lanes == bytes per memory word

# Capacity of the hardware as configured by default (see rtl/nn_accel.sv).
LIMITS = dict(W_AW=14, A_AW=11, B_AW=9, MAX_LAYERS=16, MAX_OUT=16)
N_BUF = 3          # activation buffers
SKIP_SHIFT = 14    # the shortcut multiplier K is a fraction of 2^14


class CompileError(Exception):
    """The network cannot be mapped onto the accelerator."""


# ------------------------------------------------------------------ geometry
def geometry(input_shape, specs, links=None):
    """Resolve every layer to kernel, stride, padding and in/out shapes, and
    to the tensors it reads (see the module docstring for `links`)."""
    shapes = [tuple(input_shape)]
    out = []
    for li, spec in enumerate(specs):
        src, add = links[li] if links else (li, None)
        if not 0 <= src <= li:
            raise CompileError(f"layer {li}: reads tensor {src}, which does not exist yet")
        h, w, c = shapes[src]
        kind = spec[0]
        if kind == "fc":
            kh, kw, s, p, oc = h, w, 1, 0, spec[1]
        else:
            k = spec[1]
            kh, kw = k if isinstance(k, tuple) else (k, k)
            if kind == "pool":
                s, p, oc = spec[2], 0, c
            else:
                s, oc, p = spec[2], spec[3], (spec[4] if len(spec) > 4 else 0)
        oh, ow = (h + 2 * p - kh) // s + 1, (w + 2 * p - kw) // s + 1
        if oh < 1 or ow < 1:
            raise CompileError(f"layer {li}: {kh}x{kw} window does not fit "
                               f"a {h}x{w} input")
        if add is not None:
            if kind == "pool":
                raise CompileError(f"layer {li}: a shortcut cannot be added to a pooling layer")
            if not 0 <= add <= li or shapes[add] != (oh, ow, oc):
                raise CompileError(f"layer {li}: shortcut tensor {add} does not have the "
                                   f"output's shape {oh}x{ow}x{oc}")
        out.append(dict(kind=kind, kh=kh, kw=kw, s=s, p=p, oc=oc,
                        h=h, w=w, c=c, oh=oh, ow=ow, src=src, add=add))
        shapes.append((oh, ow, oc))
    if out[-1]["kind"] == "pool":
        raise CompileError("the last layer must produce logits, not a pooled map")
    if out[-1]["add"] is not None:
        raise CompileError("the last layer cannot have a shortcut")
    return out


def input_is_private(geo):
    """True if only layer 0 reads the network input, so its storage layout
    can be changed freely."""
    return geo[0]["src"] == 0 and all(g["src"] != 0 and g["add"] != 0 for g in geo[1:]) \
        and geo[0]["add"] is None


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


def pool_windows(a, g):
    """(N,H,W,C) -> (N,OH,OW,kh*kw,C): the values each pooled output sees."""
    n = len(a)
    return im2col(a, g).reshape(n, g["oh"], g["ow"], g["kh"] * g["kw"], g["c"])


def float_forward(geo, fparams, a):
    """Float reference: (N,H,W,C) inputs scaled to [0, 1] -> logits."""
    t = [a]
    for li, (g, fp) in enumerate(zip(geo, fparams)):
        x = t[g["src"]]
        if g["kind"] == "pool":
            t.append(pool_windows(x, g).max(3))
            continue
        z = im2col(x, g) @ fp[0].T + fp[1]
        if g["add"] is not None:
            z = z + t[g["add"]]
        t.append(z if li == len(geo) - 1 else np.maximum(z, 0))
    return t[-1].reshape(len(a), -1)


# ------------------------------------------------------------------ rewrites
def s2d_input(x, b=2):
    """Space-to-depth: (N,H,W,C) -> (N,H/b,W/b,b*b*C), channel (dy*b+dx)*C+c.
    A pure rearrangement of the input."""
    n, h, w, c = x.shape
    x = x.reshape(n, h // b, b, w // b, b, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(n, h // b, w // b, b * b * c)


def s2d_weights(w, kh, kw, c, b=2):
    """Kernel (oc, kh*kw*c) on the image -> (oc, kh/b*kw/b*b*b*c) on its
    space-to-depth form. Same numbers, reordered."""
    oc = len(w)
    w = w.reshape(oc, kh // b, b, kw // b, b, c).transpose(0, 1, 3, 2, 4, 5)
    return w.reshape(oc, -1)


def s2d_weights_inv(w, kh, kw, c, b=2):
    """Inverse of s2d_weights; kh, kw, c describe the kernel on the image."""
    oc = len(w)
    w = w.reshape(oc, kh // b, kw // b, b, b, c).transpose(0, 1, 3, 2, 4, 5)
    return w.reshape(oc, -1)


def rewrite_s2d(input_shape, specs, fparams):
    """If the first layer is an even-sized stride-2 convolution on a narrow
    input, store the input space-to-depth and halve the kernel. The arithmetic
    is identical, but a kxk window becomes (k/2)x(k/2) memory words instead of
    kxk. Returns (input_shape, specs, fparams, s2d) with s2d = 2 or None."""
    h, w, c = input_shape
    spec = specs[0]
    if spec[0] != "conv":
        return input_shape, specs, fparams, None
    k = spec[1]
    kh, kw = k if isinstance(k, tuple) else (k, k)
    pad = spec[4] if len(spec) > 4 else 0
    if not (spec[2] == 2 and pad == 0 and kh % 2 == 0 and kw % 2 == 0
            and h % 2 == 0 and w % 2 == 0 and 4 * c <= P):
        return input_shape, specs, fparams, None
    new_spec = ("conv", (kh // 2, kw // 2), 1, spec[3])
    new_w = s2d_weights(fparams[0][0], kh, kw, c)
    return ((h // 2, w // 2, 4 * c), [new_spec] + list(specs[1:]),
            [(new_w, fparams[0][1])] + list(fparams[1:]), 2)


def rewrite_fc_input(input_shape, specs):
    """A network that starts with a fully-connected layer reads its input as
    one long pixel, so it is packed 8 values to a word instead of one."""
    if specs[0][0] != "fc":
        return input_shape, False
    h, w, c = input_shape
    return (1, 1, h * w * c), (h, w) != (1, 1)


def apply_overlap(geo):
    """Overlap packing for a first convolution with few input channels.

    Normally a pixel is one memory word however few channels it has, so a
    3-channel image wastes 5 of the 8 lanes. Here every input word holds its
    own pixel *and the next n-1 pixels of the row* (n = 8 // channels), so a
    kernel row of kw pixels is read as ceil(kw / n) words, n pixels apart,
    instead of kw words. The input is stored with its zero padding included,
    which removes the need to mask partially valid words.

    Only the memory layout changes: the same products are summed, so results
    are bit-identical. Marks geo[0] and returns True if it applies."""
    g = geo[0]
    n = P // g["c"]
    if g["kind"] != "conv" or n < 2 or -(-g["kw"] // n) >= g["kw"]:
        return False
    if not input_is_private(geo):
        return False
    if g["w"] + 2 * g["p"] > 255 or g["h"] + 2 * g["p"] > 255:
        return False
    g["ovl"] = n
    return True


def overlap_pack(a, g):
    """(N,H,W,C) -> (N,H',W',n*C): word x of a row holds pixels x .. x+n-1 of
    the zero-padded image (zeros past the right edge)."""
    n, p = g["ovl"], g["p"]
    a = np.pad(a, ((0, 0), (p, p), (p, p + n - 1), (0, 0)))
    w = a.shape[2] - (n - 1)
    return np.concatenate([a[:, :, j:j + w, :] for j in range(n)], 3)


def pad_conv_channels(input_shape, specs, fparams, links=None):
    """The hardware writes convolution outputs a whole word at a time, so
    their channel count is rounded up to a multiple of 8 with all-zero
    filters. The extra channels are always 0 and every consumer's weights for
    them are 0, so results do not change."""
    specs, fparams = list(specs), list(fparams)
    chans, extra = [input_shape[2]], [0]     # per tensor: channels, zero channels added
    for li, spec in enumerate(specs):
        src = links[li][0] if links else li
        c, ex = chans[src], extra[src]
        if spec[0] == "pool":
            chans.append(c)
            extra.append(ex)
            continue
        w, b = fparams[li]
        if ex:                     # widen this layer's input to match
            oc = len(w)
            w = w.reshape(oc, -1, c - ex)
            w = np.concatenate([w, np.zeros((oc, w.shape[1], ex), w.dtype)], 2)
            w = w.reshape(oc, -1)
        oc = spec[1] if spec[0] == "fc" else spec[3]
        last = li == len(specs) - 1
        pad = 0
        if spec[0] == "conv" and not last and oc % P:
            pad = (-oc) % P
            w = np.concatenate([w, np.zeros((pad, w.shape[1]), w.dtype)])
            b = np.concatenate([b, np.zeros(pad, b.dtype)])
            spec = spec[:3] + (oc + pad,) + spec[4:]
        specs[li], fparams[li] = spec, (w, b)
        chans.append(oc + pad)
        extra.append(pad)
    return specs, fparams


# -------------------------------------------------------------- quantization
def int_layer(a, g, q, last, skip=None):
    """One layer of the integer reference model. `skip` is the shortcut
    tensor (int, 0..127) for a layer that has one."""
    if g["kind"] == "pool":
        return pool_windows(a, g).max(3)
    acc = im2col(a, g) @ q["w"].T + q["b"]
    if np.abs(acc).max() >= 2**31:
        raise CompileError("accumulator overflows 32 bits")
    if last:
        return acc
    v = acc * q["M"]
    if skip is not None:
        v = v + np.left_shift(skip * q["K"], q["S"] - SKIP_SHIFT)
    return np.clip((v + np.left_shift(1, q["S"] - 1)) >> q["S"], 0, 127)


def requant_factor(f, s_min=1):
    """Scalar f ~= M / 2^S with M in [2^14, 2^15] where possible, M < 2^16,
    and s_min <= S <= 40."""
    s = s_min
    while f * (1 << s) < (1 << 14) and s < 40:
        s += 1
    m = int(round(f * (1 << s)))
    if not 0 <= m < (1 << 16):
        raise CompileError(f"requantization factor {f:.3g} is too large for the hardware")
    return m, s


def quantize(geo, fparams, calib_a):
    """Post-training quantization, calibrated layer by layer on integer
    activations so the scales match what the hardware will actually see.
    `calib_a` is (N,H,W,C) int, 0..127. Hidden layers get one weight scale per
    output channel. The last layer keeps a single scale, because its raw sums
    are compared with each other. A layer with a shortcut also gets K, which
    brings the shortcut tensor to the layer's output scale."""
    t = [calib_a.astype(np.int64)]       # integer tensors
    scale = [1 / 127.0]                  # real value of one unit, per tensor
    qparams = []
    for li, (g, fp) in enumerate(zip(geo, fparams)):
        last = li == len(geo) - 1
        x, s_in = t[g["src"]], scale[g["src"]]
        if g["kind"] == "pool":
            qparams.append(None)
            t.append(int_layer(x, g, None, False))
            scale.append(s_in)
            continue
        w, b = fp
        oc = g["oc"]
        if last:
            s_w = np.full(oc, np.abs(w).max() / 127)
        else:
            s_w = np.maximum(np.abs(w).max(1), 1e-12) / 127
        q = dict(w=np.round(w / s_w[:, None]).astype(np.int64),
                 b=np.round(b / (s_w * s_in)).astype(np.int64),
                 M=np.zeros(oc, np.int64), S=np.ones(oc, np.int64), K=0)
        if last:
            qparams.append(q)
            break
        skip = None if g["add"] is None else t[g["add"]]
        real = (im2col(x, g) @ q["w"].T + q["b"]) * (s_w * s_in)
        if skip is not None:
            real = real + skip * scale[g["add"]]
        s_out = np.percentile(np.maximum(real, 0), 99.9) / 127
        if s_out <= 0:
            raise CompileError(f"layer {li}: every activation is zero on the "
                               "calibration data")
        for c in range(oc):
            q["M"][c], q["S"][c] = requant_factor(s_w[c] * s_in / s_out,
                                                  SKIP_SHIFT if skip is not None else 1)
        if skip is not None:
            q["K"] = int(round(scale[g["add"]] / s_out * (1 << SKIP_SHIFT)))
            if not 0 < q["K"] < (1 << 16):
                raise CompileError(f"layer {li}: shortcut scale ratio is out of range")
        t.append(int_layer(x, g, q, False, skip))
        scale.append(s_out)
        qparams.append(q)
    return qparams


def golden(geo, qparams, a):
    """Integer-only reference; mirrors the RTL datapath bit for bit."""
    t = [a.astype(np.int64)]
    for li, (g, q) in enumerate(zip(geo, qparams)):
        skip = None if g["add"] is None else t[g["add"]]
        t.append(int_layer(t[g["src"]], g, q, li == len(geo) - 1, skip))
    logits = t[-1].reshape(len(a), -1)
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
              ("b_base", 12), ("pool", 1), ("last", 1),
              ("src_buf", 2), ("dst_buf", 2), ("skip_buf", 2), ("skip_en", 1),
              ("skip_mult", 16)]
CFG_BITS = 208


def cfg_word(layer, **fields):
    word, pos = 0, 0
    for name, width in CFG_FIELDS:
        value = fields.pop(name)
        if not 0 <= value < (1 << width):
            raise CompileError(f"layer {layer}: {name} = {value} does not fit "
                               f"in {width} bits")
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
    buf = allocate_buffers(geo)
    for li, (g, q) in enumerate(zip(geo, qparams)):
        last = li == len(geo) - 1
        cw = -(-g["c"] // P)                   # words per input pixel
        has_skip = g["add"] is not None
        common = dict(step_x=g["s"] * cw, step_y=g["s"] * g["w"] * cw,
                      out_w=g["ow"], out_h=g["oh"], w_base=n_w, b_base=n_p,
                      last=int(last), src_buf=buf[g["src"]],
                      dst_buf=0 if last else buf[li + 1],
                      skip_buf=buf[g["add"]] if has_skip else 0,
                      skip_en=int(has_skip), skip_mult=q["K"] if has_skip else 0)
        if g["kind"] == "pool":
            cfg.append(cfg_word(
                li, kw=g["kw"], kh=g["kh"], rd_words=1, pix_step=cw,
                row_step=(g["w"] - g["kw"] + 1) * cw, org=0,
                in_w=g["w"], in_h=g["h"], pad=0, stride=g["s"],
                out_ch=cw, pool=1, **common))
            continue
        w = q["w"].reshape(g["oc"], g["kh"], g["kw"], g["c"])
        if g.get("ovl"):
            # overlap-packed input: one word per position, n pixels in it;
            # a kernel row is kwp words read n positions apart
            n = g["ovl"]
            wp, hp = g["w"] + 2 * g["p"], g["h"] + 2 * g["p"]
            kwp = -(-g["kw"] // n)
            window = dict(kw=kwp, kh=g["kh"], rd_words=1, pix_step=n,
                          row_step=wp - (kwp - 1) * n, org=0,
                          in_w=wp, in_h=hp, pad=0, stride=g["s"])
            common.update(step_x=g["s"], step_y=g["s"] * wp)
            w = np.pad(w, ((0, 0), (0, 0), (0, kwp * n - g["kw"]), (0, 0)))
            w = w.reshape(g["oc"], g["kh"], kwp, n * g["c"])
        elif g["kind"] == "fc":
            # one "pixel" that is the whole input
            window = dict(kw=1, kh=1, rd_words=g["h"] * g["w"] * cw, pix_step=1,
                          row_step=1, org=0, in_w=1, in_h=1, pad=0, stride=1)
        else:
            if g["oc"] % P and not last:
                raise CompileError(f"layer {li}: convolution has {g['oc']} output "
                                   "channels; the hardware needs a multiple of 8 "
                                   "(pad_conv_channels does this)")
            window = dict(kw=g["kw"], kh=g["kh"], rd_words=cw, pix_step=1,
                          row_step=(g["w"] - g["kw"]) * cw + 1,
                          org=(-(g["p"] * g["w"] + g["p"]) * cw) & 0xFFFF,
                          in_w=g["w"], in_h=g["h"], pad=g["p"], stride=g["s"])
        cfg.append(cfg_word(li, out_ch=g["oc"], pool=0, **window, **common))
        w_words = pack_words(pad_c(w).reshape(g["oc"], -1))
        weights.append(w_words)
        # per-output parameter word: bias, multiplier M, shift S
        params.append((q["b"] & 0xFFFFFFFF) | (q["M"] << 32) | (q["S"] << 48))
        n_w += len(w_words)
        n_p += g["oc"]
    return np.concatenate(weights), np.concatenate(params), cfg


def tensor_words(geo):
    """Memory words of every tensor: the input, then each layer's output."""
    g0 = next(g for g in geo if g["src"] == 0)
    words = [g0["h"] * g0["w"] * -(-g0["c"] // P)]
    if geo[0].get("ovl"):      # stored with its padding, one word per position
        words[0] = (geo[0]["h"] + 2 * geo[0]["p"]) * (geo[0]["w"] + 2 * geo[0]["p"])
    return words + [g["oh"] * g["ow"] * -(-g["oc"] // P) for g in geo]


def allocate_buffers(geo):
    """Assign every stored tensor to one of the N_BUF activation buffers.
    A buffer can be reused once nothing later reads the tensor in it. The
    input is in buffer 0; a plain chain alternates between buffers 0 and 1.
    Returns the buffer of each tensor (None for the logits)."""
    n = len(geo)
    last_use = [0] * (n + 1)
    for li, g in enumerate(geo):
        last_use[g["src"]] = li
        if g["add"] is not None:
            last_use[g["add"]] = li
    buf = [0] + [None] * n
    holder = {0: 0}                           # buffer -> tensor it holds
    for li in range(n - 1):                   # the last layer stores nothing
        busy = {b for b, t in holder.items() if last_use[t] >= li}
        free = [b for b in range(N_BUF) if b not in busy]
        if not free:
            raise CompileError(f"layer {li}: more than {N_BUF} feature maps must be kept "
                               "at once; the accelerator has that many buffers")
        buf[li + 1] = free[0]
        holder[free[0]] = li + 1
    return buf


def requirements(geo, weights, params):
    """What the network needs from the hardware."""
    words = tensor_words(geo)
    buf = allocate_buffers(geo)
    sizes = [1] * N_BUF
    for t, b in enumerate(buf):
        if b is not None:
            sizes[b] = max(sizes[b], words[t])
    return dict(layers=len(geo), weight_words=len(weights), param_words=len(params),
                act_words=max(sizes), bufa_words=sizes[0], bufb_words=sizes[1],
                bufc_words=sizes[2], in_words=words[0],
                n_out=geo[-1]["oh"] * geo[-1]["ow"] * geo[-1]["oc"])


def check_limits(req, limits=LIMITS):
    """Raise CompileError, naming every limit the network exceeds."""
    problems = []
    if req["layers"] > limits["MAX_LAYERS"]:
        problems.append(f"{req['layers']} layers, the layer table holds {limits['MAX_LAYERS']}")
    for what, key, aw in [("weight words", "weight_words", "W_AW"),
                          ("activation words in one feature map", "act_words", "A_AW"),
                          ("parameter words", "param_words", "B_AW")]:
        if req[key] > (1 << limits[aw]):
            need = int(np.ceil(np.log2(req[key])))
            problems.append(f"{req[key]} {what}, {aw}={limits[aw]} holds "
                            f"{1 << limits[aw]} (needs {aw}={need})")
    if req["n_out"] > limits["MAX_OUT"]:
        problems.append(f"{req['n_out']} outputs, the testbench handles {limits['MAX_OUT']}")
    if problems:
        raise CompileError("network does not fit the accelerator as configured:\n  "
                           + "\n  ".join(problems))


def write_net_params(out, req):
    """Verilog header describing the network, for a design built around it
    (fpga/tangnano20k/top.sv). Paths are relative to the repo root when the
    output directory is inside it."""
    out = Path(out)
    try:
        rel = out.resolve().relative_to(Path(__file__).resolve().parent.parent)
    except ValueError:
        rel = out.resolve()

    def aw(n):
        return max(1, int(np.ceil(np.log2(n))))
    lines = [
        "// Generated by the suanpan compiler. Do not edit.",
        f"localparam int NET_LAYERS   = {req['layers']};",
        f"localparam int NET_IN_WORDS = {req['in_words']};",
        f"localparam int NET_N_OUT    = {req['n_out']};",
        f"localparam int NET_W_WORDS  = {req['weight_words']};",
        f"localparam int NET_P_WORDS  = {req['param_words']};",
        f"localparam int NET_A_WORDS  = {req['bufa_words']};",
        f"localparam int NET_B_WORDS  = {req['bufb_words']};",
        f"localparam int NET_C_WORDS  = {req['bufc_words']};",
        f"localparam int NET_W_AW     = {aw(req['weight_words'])};",
        f"localparam int NET_P_AW     = {aw(req['param_words'])};",
        f"localparam int NET_A_AW     = {aw(req['act_words'])};",
        f'localparam     NET_DIR      = "{rel.as_posix()}";',
    ] + [f'localparam     NET_{n.upper()}_FILE = "{(rel / (n + ".hex")).as_posix()}";'
         for n in ("weights", "params", "cfg", "demo_image")]
    (out / "net_params.vh").write_text("\n".join(lines) + "\n")


def cycles(geo):
    """Clock cycles of multiply-accumulate / pooling per inference: one per
    memory word read."""
    total = 0
    for g in geo:
        cw = -(-g["c"] // P)
        outs = g["oh"] * g["ow"]
        if g["kind"] == "pool":
            total += outs * cw * g["kh"] * g["kw"]
        elif g.get("ovl"):
            total += outs * g["oc"] * g["kh"] * -(-g["kw"] // g["ovl"])
        else:
            total += outs * g["oc"] * g["kh"] * g["kw"] * cw
    return total


def export(out, geo, qparams, test_a, labels=None, limits=LIMITS):
    """Write every file the RTL testbench and the FPGA build read.
    `test_a` is (N,H,W,C) int 0..127, already in the network's input layout
    (overlap packing, if geo[0] is marked for it, is applied here)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    logits, pred = golden(geo, qparams, test_a)
    weights, params, cfg = compile_program(geo, qparams)
    req = requirements(geo, weights, params)
    check_limits(req, limits)
    n = len(test_a)
    stored = overlap_pack(test_a, geo[0]) if geo[0].get("ovl") else test_a
    images = pad_c(stored).reshape(n, -1)
    assert images.shape[1] // P == req["in_words"]
    write_hex(out / "weights.hex", weights, 16)
    write_hex(out / "params.hex", params, 14)
    write_hex(out / "cfg.hex", cfg, CFG_BITS // 4)
    write_hex(out / "meta.hex", [req["in_words"], len(cfg), logits.shape[1]], 8)
    write_hex(out / "images.hex", pack_words(images), 16)
    write_hex(out / "labels.hex", labels if labels is not None else pred, 2)
    write_hex(out / "golden_class.hex", pred, 2)
    write_hex(out / "golden_logits.hex", logits.reshape(-1), 8)
    # test image 0 as input words, for the FPGA demo ROM
    write_hex(out / "demo_image.hex", pack_words(images[:1]), 16)
    write_net_params(out, req)
    return dict(req, logits=logits, pred=pred, mac_cycles=cycles(geo))

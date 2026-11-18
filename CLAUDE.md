# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

suanpan is an INT8 neural-network inference accelerator written in SystemVerilog, developed in three stages (see `docs/three-stages.md`). `README.md` and `docs/three-stages.md` each have a Chinese translation next to them (`*.zh-CN.md`); whenever you change one, change the other in the same commit, including every number. Stage 1 is done: RTL verified in simulation against a bit-exact integer model, plus pre-layout synthesis estimates. Stage 2 is in progress: a Tang Nano 20K demo builds to a bitstream and passes gate-level simulation, but has not been run on a board. There is no tapeout flow.

## Commands

Tools: `iverilog`, `verilator`, `yosys`, and Python with `numpy` (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`). The Makefile calls `.venv/bin/python`; override with `PY=python`.

```bash
make model      # train + quantize + export memory images to build/<arch>/
make sim        # 100 images under Icarus Verilog (N=<count> to change)
make sim-full   # all 10,000 test images under Verilator
make wave       # one image, dumps build/tb.vcd
make lint       # verilator --lint-only -Wall on the RTL
make synth      # Yosys + ABC onto SKY130, PERIOD=<ps>

make fpga-sim      # Tang Nano 20K wrapper in RTL simulation
make fpga          # synth_gowin + nextpnr-himbaechel + gowin_pack -> build/fpga/top.fs
make fpga-gatesim  # simulate the placed-and-routed netlist
```

The `fpga` targets need the OSS CAD Suite on `PATH` (Homebrew has no `nextpnr-himbaechel`). Vendor tools are not an option on macOS.

- `ARCH=cnn` (default), `ARCH=mlp` or `ARCH=stress` selects the network for `model`, `sim`, `sim-full` and `wave`. Each has its own `build/<arch>/` directory.
- The quickest check after an RTL change is `make sim N=5`. Before finishing, run `make lint` and `make sim-full` for `ARCH=cnn`, `ARCH=mlp` and `ARCH=stress`; CI runs all three. The MLP exercises the fully-connected path as a degenerate convolution. `stress` is not a useful network: it exists to hit hardware corner cases (a hidden layer with one-word windows, so outputs complete on consecutive cycles; a single-channel input padded to 8; and a layer width that is not a multiple of 8). If you add a corner case to the RTL, extend `stress` to reach it.
- `sim` and `sim-full` exit non-zero unless the testbench prints `RESULT: PASS`.
- Verilator's generated makefile cannot build in a path containing spaces. If the checkout lives under one, pass `VOBJ=<dir without spaces>` to `make sim-full`.
- `make model` downloads MNIST into `data/` on first run. `build/`, `data/` and `.venv/` are not committed.

## The one invariant

`model/train.py` contains an integer-only reference (`int_layer` / `golden`) that must match the RTL datapath **bit for bit**. The testbench compares every logit, not just the predicted class. Any change to arithmetic in `rtl/nn_accel.sv` (rounding, saturation, widths, operand order) needs the same change in `int_layer`, and vice versa. The shared definition is:

```
acc = sum(w_int8 * x_int8) + bias_int32
hidden layer: y = clamp((acc * M + 2^(S-1)) >> S, 0, 127)
last layer:   logits = raw acc, class = argmax
```

## Architecture

The hardware is network-agnostic. A network is three memory images produced by `train.py` (weights, biases, layer table) plus a `meta.hex` the testbench reads; the RTL has no knowledge of MNIST or of a particular topology.

**One datapath for both layer types.** There is no convolution engine. Activations are stored HWC with the channel count zero-padded to a multiple of `P=8`, so one pixel is a whole number of 64-bit words and each kernel row of a window is a contiguous run of words. An output value is a dot product read through a pointer that walks `row_len` words, skips `row_skip` words to the next image row, and repeats `n_rows` times. The window origin moves by `step_x` / `step_y` words, which is how stride is expressed. A fully-connected layer is the special case of one output pixel whose single kernel row spans the whole input.

**The layer table is the contract between Python and RTL.** Each layer is one 144-bit descriptor. Its bit layout is defined twice and must stay in sync: `cfg_word()` in `model/train.py` and the header comment plus the `S_LOAD` state in `rtl/nn_accel.sv`. `compile_program()` in `train.py` is where a layer's geometry is turned into those fields and where weights are packed in the order the hardware reads them (per output channel: `ky`, `kx`, padded channel).

**Control flow in `nn_accel.sv`.** Only address generation is a state machine; everything after it is a feed-forward pipeline. The front end (`state`) issues one memory word per cycle in `S_MAC` and, on the last word of an output, moves the channel / pixel / row counters on so that the next output's first word follows in the very next cycle. Words flow through four stages (`v0`..`v3`: registered operands, registered products, adder tree, accumulate) carrying tags: `f` first word of an output, `l` last word, `e` last word of the layer. The accumulator starts from the bias on the first word. When a last word reaches stage 3 (`handoff`) the sum enters the post pipeline (`pv[0..6]`): four 9x17 multiplies, two combine stages, round, shift, clamp, write. Output bytes are assembled into a full word before being written. Two activation buffers ping-pong, selected by `layer[0]`.

Things that follow from this design:

- An output costs exactly as many cycles as its window has words, so windows can be one word long and several outputs can be in flight at once. Per-output state must travel as tags or be counted on the post side (`o_cnt`); a single "current output" register is not enough.
- `b_q` holds an output's bias for only one cycle when windows are one word long, which is why it is captured at stage 0 and carried along (`bias_r0..2`).
- The front end drains everything in flight (`S_DRAIN`) before changing layer, because the next layer reads the buffer the current one is still writing and the post pipeline uses the per-layer constants.

**Input layout is part of the model.** The default CNN stores its input space-to-depth (`"s2d": 2` in `ARCHS`, see `prep()` in `train.py`): 2x2 pixel blocks become 4-channel pixels. It is a pure rearrangement done by the exporter; the hardware does not know about it. Anything that feeds images to the accelerator (the testbench's `images.hex`, the FPGA demo ROM) must use the same layout.

**Memories are black boxes in synthesis.** `rtl/sram_sp.sv` is a behavioral synchronous-read memory standing in for SRAM macros. `syn/synth.sh` reads it with `-lib`, so reported area and timing cover logic only. The testbench loads weight and bias images by hierarchical reference (`dut.u_wmem.mem`).

## Things that are easy to get wrong

- **Yosys adder mapping.** Yosys/ABC defaults are area-oriented and turn adders into ripple-carry chains, which hides the real critical path and makes timing jump between revisions for no RTL reason. `syn/synth.sh` deliberately maps to Kogge-Stone, passes `-noabc` to `synth`, and uses `syn/abc.script`; keep all three when changing the flow.
- **Reading ABC's timing.** ABC's area recovery relaxes every path up to the slowest one, so the path `stime -p` prints is not necessarily the cause. When timing regresses, compare against `main` under the same flow before touching RTL.
- **Shared adders are explicit on purpose.** `acc_y` (accumulate or bias) and `add_y` (Horner step or rounding) are each written as one adder with muxed operands. Writing them as separate `+` expressions in different FSM states makes Yosys merge them into a slower structure.
- **Icarus is the stricter front end.** It rejects enum assignment through a ternary (`state <= c ? A : B`) and warns on variable part-selects inside `always_comb`; use `if/else` and per-lane loops as the existing code does. Code must build under both Icarus and Verilator.
- **Layer constraints.** Valid convolutions only (no padding), no pooling, convolution output channels must be a multiple of 8, and one requantization scale per layer. `compile_program()` asserts the channel rule.
- **Buffer sizes are parameters.** `A_AW`, `W_AW`, `B_AW` in `nn_accel.sv` bound the largest feature map, weight count and bias count. The testbench has its own `A_AW` and `MAX_*` limits that must cover the network being run.

## FPGA flow

- **A failing gate-level simulation is not automatically a design bug.** Yosys' Gowin simulation library declares the block RAM (`SPX9`) and the multipliers as black boxes, so their outputs are X. `fpga/gowin_sim.v` holds behavioural models written from Yosys' own mapping files (`brams_map.v`, `dsp_map.v`), and `fpga/gatesim.sh` swaps them in. If synthesis starts using a cell that has no model there, the symptom is X on the outputs with a correct cycle count.
- **Partially initialized ROMs look uninitialized in a dump.** Yosys prints `INIT` MSB-first, so a ROM whose upper addresses are unused starts with a long run of `x`. Count defined bits before concluding a `$readmemh` did not load.
- **Block RAMs are limited to 18-bit data** in `fpga/tangnano20k/build.sh`, by running the memory-mapping step of `synth_gowin` by hand with a copy of Yosys' `brams.txt` that omits width 36. Any 36-bit `SPX9` makes nextpnr report hold violations on its `ADB*` pins, for every seed. Small, wide memories are the ones Yosys maps that way, so shrinking a buffer can make the violations appear.
- **Board facts.** Pins are from Sipeed's example repo (clock 4, LEDs 15-20 active low, button 88). The button's polarity is unverified, which is why the wrapper triggers on any debounced change.

## Reporting numbers

Synthesis results are pre-layout estimates: typical corner, no wire load, no SRAM timing, no power. When updating figures in `README.md`, keep that caveat and rerun `make synth` rather than reusing old values; small differences between runs or Yosys versions are mapper noise, not design changes.

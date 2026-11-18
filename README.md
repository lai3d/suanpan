# suanpan (算盘)

An AI chip designed from scratch, one stage at a time. The name is the Chinese
abacus: like its beads, the core of the chip multiplies and accumulates.

**Stage 1 (current): RTL + simulation.** A from-scratch hardware accelerator for fully-connected INT8 inference,
verified bit-exactly in simulation against an integer golden model on MNIST.

## Status

| Check | Result |
|---|---|
| Model | 784 → 64 (ReLU) → 10, INT8 weights, post-training quantized |
| Accuracy (float / INT8) | 97.52% / 97.48% on the 10,000-image MNIST test set |
| RTL vs golden model | 100,000 logits + 10,000 classes, 0 mismatches (Verilator) |
| Latency | 6,929 clock cycles per inference |

No synthesis has been run yet, so clock frequency, area and power are unknown.

## Architecture

```
            ┌──────────┐   64-bit word = 8 × int8
 host ────► │ buffer A │◄──────────────┐
            └────┬─────┘               │ requantized outputs
                 │ ping-pong      ┌────┴─────┐
            ┌────┴─────┐          │ ×M, >>S  │
            │ buffer B │          │ ReLU/clamp│
            └────┬─────┘          └────▲─────┘
                 ▼                     │
 weights ─► [ 8× int8 multiplier + adder tree ] ─► 32-bit accumulator ─► logits / argmax
 bias ──────────────────────────────────────────────────┘
```

- `rtl/dot_unit.sv` — 8-lane signed 8×8 dot product (the compute core).
- `rtl/nn_accel.sv` — controller FSM, MAC pipeline, requantization, argmax.
  The network is a table of layer descriptors in config memory, so a different
  MLP only needs new memory images, not new RTL.
- `rtl/sram_sp.sv` — synchronous-read memory (SRAM macro / block RAM).
- `model/train.py` — numpy training, quantization, memory-image export, and the
  bit-exact integer reference model.
- `tb/tb_nn_accel.sv` — self-checking testbench.

## Run

Requires `iverilog`, `verilator` and Python with `numpy` (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`).

```bash
make model      # downloads MNIST, trains, writes build/mem/*.hex
make sim        # 100 images, Icarus Verilog
make sim-full   # all 10,000 images, Verilator
make wave       # one image, dumps build/tb.vcd
make lint
```

## License

Apache-2.0. See [LICENSE](LICENSE).

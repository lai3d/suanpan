# Designing your own chip: three stages

[中文版](three-stages.zh-CN.md)

"Designing a chip yourself" can mean three quite different things. They differ
by orders of magnitude in cost and in what you hold at the end. This project
takes them one at a time and only moves on when the previous stage has
numbers to justify it.

| Stage | What you end up with | Cost | Status in this repo |
|---|---|---|---|
| 1. RTL + simulation | A design in Verilog that runs inference in a simulator | Free, open tools | Done |
| 2. FPGA | The same design running on real hardware | A development board | Bitstream builds; not yet run on a board |
| 3. ASIC tapeout | An actual chip | From thousands of dollars upward | Not started |

## Stage 1: RTL and simulation

You describe the hardware in a hardware description language and run it in a
simulator. Nothing physical exists, but the design is real: every register,
adder and memory is specified, and a synthesis tool can turn it into gates.

What this stage proves:

- **The design is correct.** Here that means bit-exact: all 10,000 MNIST test
  images produce the same logits as an integer reference model.
- **Roughly how big and fast it would be.** Synthesis onto an open cell
  library (SKY130) gives a gate count and a longest-path estimate.

What it cannot tell you: real wire delays, clock distribution, memory timing,
power. Those need a physical implementation.

Tools: Icarus Verilog and Verilator to simulate, Yosys to synthesize. All open
source. See the [README](../README.md) for what this stage produced.

## Stage 2: FPGA

An FPGA is a chip whose logic can be reprogrammed. Loading the design onto one
makes it run on real hardware, at real speed, with real memories and real
clocks, without manufacturing anything.

What this stage adds:

- **Evidence beyond simulation.** Timing is checked by a place-and-route tool
  against the actual device, and the result can be observed with a logic
  analyser or simply on LEDs.
- **A measured clock rate and resource usage** instead of estimates.
- **A way to connect to the outside world**: a host, a camera, a sensor.

What it costs: a development board. For this design the binding constraint is
block RAM, not logic. The first target is the Sipeed Tang Nano 20K, because it
is inexpensive, has enough block RAM, carries its own USB programmer, and is
supported by an open-source toolchain that runs on macOS and Linux.

Status: the demo in [`fpga/tangnano20k`](../fpga/tangnano20k) synthesizes,
places, routes and packs into a bitstream with the open flow, meeting timing
with a wide margin. A gate-level simulation of the routed netlist gives the
same answer as the RTL. It has **not** been run on a physical board yet.

An FPGA is not a small ASIC. It is slower and far less power-efficient than
the same design as a custom chip, and FPGA-specific blocks (block RAM, DSP
multipliers) do not carry over. What carries over is the RTL and the
confidence that it works.

## Stage 3: ASIC tapeout

Here the design is manufactured. Synthesis is followed by physical design:
floorplanning, placement, clock-tree synthesis, routing, and sign-off checks
for timing, design rules and layout-versus-schematic. The output is a layout
file sent to a foundry.

Two very different versions of this exist:

- **Mature process, shared wafer.** Open process design kits such as SkyWater
  130 nm, together with open flows such as OpenLane, let an individual submit
  a small design on a multi-project shuttle. The cost is in the range of
  thousands to low tens of thousands of dollars and the turnaround is months.
- **Advanced process.** At 7 nm and below, mask sets, licensed tools and IP,
  and the engineering team put the cost in the tens of millions of dollars.
  This is company-scale work.

Cost figures here are rough orders of magnitude, not quotes; check current
shuttle pricing before planning around them.

Stage 3 is also where the parts this project has postponed become unavoidable:
real SRAM macros instead of behavioural memories, a pad ring, a clock and
reset strategy, power analysis, and a test plan for the fabricated part.

## What a useful AI chip needs beyond the datapath

The multiply-accumulate core is the easy part. Three other things decide
whether a design is useful:

- **Memory.** Both the ASIC estimate and the FPGA build show the same thing:
  memory dominates area and is the scarce resource. Bandwidth to the weights,
  not multiplier count, usually limits throughput.
- **A programming model.** This design is driven by a table of layer
  descriptors written by the host, so a new network needs new memory images
  rather than new RTL. That table is the seed of an instruction set.
- **A software stack.** Something has to turn a trained model into those
  memory images. Here it is one Python script; for a real product it is a
  compiler, and it is routinely the larger half of the work.

## A realistic view

A small custom design will not beat a general-purpose GPU at being general.
It can win decisively in a narrow case: one class of model, a fixed
quantization format, a tight power budget. The sensible path is the one this
project follows: pick a workload, prove the design in simulation, prove it
again on an FPGA, and only then decide whether the numbers justify silicon.

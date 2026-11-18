PY   ?= .venv/bin/python
RTL  := rtl/sram_sp.sv rtl/dot_unit.sv rtl/nn_accel.sv
TB   := tb/tb_nn_accel.sv
N    ?= 100
# Network to build and simulate: cnn (default) or mlp.
ARCH ?= cnn
MEM  := build/$(ARCH)
# Verilator's generated makefile cannot build in a path containing spaces;
# override VOBJ if this checkout lives under one.
VOBJ ?= build/obj

.PHONY: model sim sim-full wave lint synth fpga fpga-sim fpga-gatesim fpga-load fpga-flash clean

# Train + quantize the network, export memory images and golden vectors.
model $(MEM)/weights.hex:
	$(PY) model/train.py --arch $(ARCH)

# Quick check with Icarus Verilog (N images, default 100).
sim: $(MEM)/weights.hex
	iverilog -g2012 -o build/tb.vvp $(RTL) $(TB)
	vvp -n build/tb.vvp +MEM=$(MEM) +N=$(N) | tee build/sim.log
	@grep -q "RESULT: PASS" build/sim.log

# Full 10,000-image test set with Verilator (compiled, much faster).
sim-full: $(MEM)/weights.hex
	verilator --binary --timing -Wno-fatal -O3 --Mdir $(VOBJ) \
	    --top-module tb_nn_accel -o tb $(RTL) $(TB) > build/verilator.log 2>&1 \
	    || (cat build/verilator.log; exit 1)
	$(VOBJ)/tb +MEM=$(MEM) +N=10000 | tee build/sim-full.log
	@grep -q "RESULT: PASS" build/sim-full.log

# Dump a waveform for one image -> build/tb.vcd
wave: $(MEM)/weights.hex
	iverilog -g2012 -o build/tb.vvp $(RTL) $(TB)
	vvp -n build/tb.vvp +MEM=$(MEM) +N=1 +VCD

lint:
	verilator --lint-only -Wall -Wno-UNUSEDSIGNAL --top-module nn_accel $(RTL)

# Logic synthesis onto SKY130 (needs yosys); PERIOD is the target in ps.
PERIOD ?= 10000
synth:
	syn/synth.sh $(PERIOD)

# --- Tang Nano 20K demo (needs the OSS CAD Suite on PATH) -----------------
FPGA_TOP := fpga/uart.sv fpga/tangnano20k/top.sv

# RTL simulation of the board wrapper: the LEDs must show the golden class.
fpga-sim: build/cnn/weights.hex
	iverilog -g2012 -o build/tb_fpga.vvp $(RTL) $(FPGA_TOP) tb/tb_fpga_top.sv
	vvp -n build/tb_fpga.vvp | tee build/fpga-sim.log
	@grep -q "RESULT: PASS" build/fpga-sim.log

# Synthesis, place and route, bitstream -> build/fpga/top.fs
fpga: build/cnn/weights.hex
	fpga/tangnano20k/build.sh

# Simulate the exact netlist that was placed and routed.
fpga-gatesim: fpga
	fpga/gatesim.sh build/fpga/top_synth.v tb/tb_fpga_top.sv tb_fpga_top

# Load the bitstream into the FPGA's SRAM (lost at power-off) ...
fpga-load:
	openFPGALoader -b tangnano20k build/fpga/top.fs

# ... or write it to the board's flash so it survives a power cycle.
fpga-flash:
	openFPGALoader -b tangnano20k -f build/fpga/top.fs

clean:
	rm -rf build

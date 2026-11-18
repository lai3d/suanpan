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

.PHONY: model sim sim-full wave lint synth clean

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

clean:
	rm -rf build

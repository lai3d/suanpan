PY   ?= .venv/bin/python
RTL  := rtl/sram_sp.sv rtl/dot_unit.sv rtl/nn_accel.sv
TB   := tb/tb_nn_accel.sv
N    ?= 100
# Verilator's generated makefile cannot build in a path containing spaces;
# override VOBJ if this checkout lives under one.
VOBJ ?= build/obj

.PHONY: model sim sim-full wave lint clean

# Train + quantize the network, export memory images and golden vectors.
model build/mem/weights.hex:
	$(PY) model/train.py

# Quick check with Icarus Verilog (N images, default 100).
sim: build/mem/weights.hex
	iverilog -g2012 -o build/tb.vvp $(RTL) $(TB)
	vvp -n build/tb.vvp +N=$(N)

# Full 10,000-image test set with Verilator (compiled, much faster).
sim-full: build/mem/weights.hex
	verilator --binary --timing -Wno-fatal -O3 --Mdir $(VOBJ) \
	    --top-module tb_nn_accel -o tb $(RTL) $(TB) > build/verilator.log 2>&1 \
	    || (cat build/verilator.log; exit 1)
	$(VOBJ)/tb +N=10000

# Dump a waveform for one image -> build/tb.vcd
wave: build/mem/weights.hex
	iverilog -g2012 -o build/tb.vvp $(RTL) $(TB)
	vvp -n build/tb.vvp +N=1 +VCD

lint:
	verilator --lint-only -Wall -Wno-UNUSEDSIGNAL --top-module nn_accel $(RTL)

clean:
	rm -rf build

#!/bin/sh
# Gate-level simulation of a netlist produced by synth_gowin.
#   usage: [NET=build/cnn] fpga/gatesim.sh <netlist.v> <testbench.sv> <top module> [plusargs...]
# Yosys' Gowin simulation library declares the block RAM as a black box; this
# swaps in the behavioural models from fpga/gowin_sim.v.
set -e
NETLIST=$1; TB=$2; TOP=$3; shift 3
OUT=${OUT:-build/fpga}
mkdir -p $OUT
LIB=$(yosys-config --datdir)/gowin/cells_sim.v
awk '/^\(\* blackbox \*\)$/ {held=$0; next}
     /^module SPX9 / {skip=1; held=""}
     skip {if ($0 ~ /^endmodule/) skip=0; next}
     {if (held != "") {print held; held=""} print}' $LIB > $OUT/cells_sim.v
iverilog -g2012 -DGATE -I${NET:-build/cnn} -s $TOP -o $OUT/gatesim.vvp $OUT/cells_sim.v fpga/gowin_sim.v $NETLIST $TB
vvp -n $OUT/gatesim.vvp "$@" | tee $OUT/gatesim.log
grep -q "RESULT: PASS" $OUT/gatesim.log

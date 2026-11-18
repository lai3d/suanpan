#!/bin/sh
# Logic synthesis of nn_accel onto the open SKY130 high-density cell library.
#   usage: syn/synth.sh [target_period_ps]     (run from the repo root)
#
# Adders are mapped to Kogge-Stone prefix adders and ABC runs a delay-oriented
# script; the Yosys defaults produce ripple-carry chains that hide the real
# critical path. `-noabc` matters too: synth's built-in generic ABC pass is
# area-oriented and, depending on the surrounding logic, collapses the prefix
# adders back into carry chains, which makes results jump between revisions.
#
# Memories (sram_sp) stay black boxes: on a real chip they are SRAM macros,
# not standard cells, and are accounted for separately.
set -e
PERIOD=${1:-10000}
RTL=${RTL:-rtl}
LIB=build/pdk/sky130_fd_sc_hd__tt_025C_1v80.lib
LIB_URL=https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
OUT=build/syn
mkdir -p build/pdk $OUT
[ -f $LIB ] || curl -sSfL -o $LIB $LIB_URL

yosys -q -l $OUT/yosys_$PERIOD.log -p "
    read_verilog -sv -lib $RTL/sram_sp.sv
    read_verilog -sv $RTL/dot_unit.sv $RTL/nn_accel.sv
    hierarchy -check -top nn_accel
    synth -top nn_accel -flatten -run :fine
    techmap -map +/techmap.v -map +/choices/kogge-stone.v
    synth -top nn_accel -noabc -run fine:
    dfflibmap -liberty $LIB
    abc -liberty $LIB -constr syn/abc.constr -script $PWD/syn/abc.script -D $PERIOD -dont_use sky130_fd_sc_hd__lpflow*
    opt_clean -purge
    tee -o $OUT/stat_$PERIOD.txt stat -liberty $LIB
    write_verilog -noattr $OUT/nn_accel_$PERIOD.v
"
echo "target period : $PERIOD ps"
grep -E "ABC: WireLoad" $OUT/yosys_$PERIOD.log | tail -1 | sed 's/^ABC: *//'
grep -E "Chip area|sequential" $OUT/stat_$PERIOD.txt

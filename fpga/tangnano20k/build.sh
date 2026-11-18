#!/bin/sh
# Build the Tang Nano 20K demo with the open-source Gowin flow.
#   usage: [BAUD=115200] [OUT=build/fpga] fpga/tangnano20k/build.sh
#          (run from the repo root)
# Needs yosys, nextpnr-himbaechel and gowin_pack on PATH (OSS CAD Suite),
# and the CNN memory images from `make model`.
set -e
OUT=${OUT:-build/fpga}
BAUD=${BAUD:-115200}
CLKS_PER_BIT=$((27000000 / BAUD))
DEVICE=GW2AR-LV18QN88C8/I7
FAMILY=GW2A-18C
mkdir -p $OUT

# Block RAMs are limited to 18-bit data. Yosys would otherwise use the 36-bit
# single-port mode for small wide memories, and nextpnr reports hold violations
# on the extra address pins (ADB*) of every such block, for every placement.
# The stock memory library is copied with width 36 removed, and the memory
# mapping step of synth_gowin is run by hand with that copy.
sed 's/widths 1 2 4 9 18 36 per_port;/widths 1 2 4 9 18 per_port;/' \
    "$(yosys-config --datdir)/gowin/brams.txt" > $OUT/brams_no36.txt

yosys -q -l $OUT/synth.log -p "
    read_verilog -sv rtl/sram_sp.sv rtl/dot_unit.sv rtl/nn_accel.sv fpga/uart.sv fpga/tangnano20k/top.sv
    chparam -set CLKS_PER_BIT $CLKS_PER_BIT top
    synth_gowin -top top -run :map_ram
    memory_libmap -lib +/gowin/lutrams.txt -lib $OUT/brams_no36.txt
    techmap -map +/gowin/lutrams_map.v -map +/gowin/brams_map.v
    synth_gowin -top top -run map_ffram: -json $OUT/top.json
    write_verilog -noattr $OUT/top_synth.v
"

nextpnr-himbaechel --json $OUT/top.json --write $OUT/top_pnr.json \
    --device $DEVICE --vopt family=$FAMILY \
    --vopt cst=fpga/tangnano20k/tangnano20k.cst --freq 27 > $OUT/pnr.log 2>&1 \
    || { grep -E "^ERROR" $OUT/pnr.log | sort | uniq -c; exit 1; }

gowin_pack -d $FAMILY -o $OUT/top.fs $OUT/top_pnr.json 2>/dev/null

grep -E "Max frequency" $OUT/pnr.log | tail -1 | sed 's/^Info: //'
grep -E "(LUT4|DFF|BSRAM|MULT9X9|MULT18X18|RAM16SDP4|ALU):" $OUT/pnr.log | sed 's/^Info: *//'
echo "UART: $BAUD baud ($CLKS_PER_BIT clocks per bit)"
echo "bitstream: $OUT/top.fs"

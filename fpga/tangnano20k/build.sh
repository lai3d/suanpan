#!/bin/sh
# Build the Tang Nano 20K demo with the open-source Gowin flow.
#   usage: fpga/tangnano20k/build.sh        (run from the repo root)
# Needs yosys, nextpnr-himbaechel and gowin_pack on PATH (OSS CAD Suite),
# and the CNN memory images from `make model`.
set -e
OUT=build/fpga
DEVICE=GW2AR-LV18QN88C8/I7
FAMILY=GW2A-18C
mkdir -p $OUT

# The bias table is forced into logic. As a block RAM it becomes the design's
# only 36-bit-wide SPX9, and nextpnr then reports hold violations on its
# address pins for every placement. It is 34 words, so logic costs nothing.
yosys -q -l $OUT/synth.log -p "
    read_verilog -sv rtl/sram_sp.sv rtl/dot_unit.sv rtl/nn_accel.sv fpga/tangnano20k/top.sv
    hierarchy -top top
    flatten
    setattr -set ram_style \"logic\" m:u_accel.u_bmem.mem
    synth_gowin -top top -json $OUT/top.json
    write_verilog -noattr $OUT/top_synth.v
"

nextpnr-himbaechel --json $OUT/top.json --write $OUT/top_pnr.json \
    --device $DEVICE --vopt family=$FAMILY \
    --vopt cst=fpga/tangnano20k/tangnano20k.cst --freq 27 > $OUT/pnr.log 2>&1 \
    || { grep -E "^ERROR" $OUT/pnr.log | sort | uniq -c; exit 1; }

gowin_pack -d $FAMILY -o $OUT/top.fs $OUT/top_pnr.json 2>/dev/null

grep -E "Max frequency" $OUT/pnr.log | tail -1 | sed 's/^Info: //'
grep -E "(LUT4|DFF|BSRAM|MULT9X9|MULT18X18|RAM16SDP4|ALU):" $OUT/pnr.log | sed 's/^Info: *//'
echo "bitstream: $OUT/top.fs"

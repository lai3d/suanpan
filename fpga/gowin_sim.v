// Behavioural simulation models for the Gowin GW2A hard blocks this design
// uses, for gate-level simulation only. Yosys ships these cells as black
// boxes, so without this file every block RAM output is X.
//
// These models follow the conventions of Yosys' own Gowin mapping files
// (dsp_map.v, brams_map.v). They show that the synthesized netlist is
// equivalent to the RTL under those conventions; they are not vendor models
// and do not prove what the silicon does.
//
// Multipliers: only the unregistered configuration is modelled (AREG = BREG =
// OUT_REG = PIPE_REG = 0). ASIGN / BSIGN select signed operands, ASEL / BSEL
// select the shift-chain inputs SIA / SIB.
module MULT9X9(A, SIA, B, SIB, ASIGN, BSIGN, ASEL, BSEL, CE, CLK, RESET, DOUT, SOA, SOB);
    input  [8:0] A, SIA, B, SIB;
    input  ASIGN, BSIGN, ASEL, BSEL, CE, CLK, RESET;
    output [17:0] DOUT;
    output [8:0] SOA, SOB;
    parameter AREG = 1'b0;
    parameter BREG = 1'b0;
    parameter OUT_REG = 1'b0;
    parameter PIPE_REG = 1'b0;
    parameter ASIGN_REG = 1'b0;
    parameter BSIGN_REG = 1'b0;
    parameter SOA_REG = 1'b0;
    parameter MULT_RESET_MODE = "SYNC";
    wire [8:0] a = ASEL ? SIA : A;
    wire [8:0] b = BSEL ? SIB : B;
    wire signed [17:0] ax = {{9{ASIGN & a[8]}}, a};
    wire signed [17:0] bx = {{9{BSIGN & b[8]}}, b};
    assign DOUT = ax * bx;
    assign SOA = a;
    assign SOB = b;
endmodule

module MULT18X18(A, SIA, B, SIB, ASIGN, BSIGN, ASEL, BSEL, CE, CLK, RESET, DOUT, SOA, SOB);
    input  [17:0] A, SIA, B, SIB;
    input  ASIGN, BSIGN, ASEL, BSEL, CE, CLK, RESET;
    output [35:0] DOUT;
    output [17:0] SOA, SOB;
    parameter AREG = 1'b0;
    parameter BREG = 1'b0;
    parameter OUT_REG = 1'b0;
    parameter PIPE_REG = 1'b0;
    parameter ASIGN_REG = 1'b0;
    parameter BSIGN_REG = 1'b0;
    parameter SOA_REG = 1'b0;
    parameter MULT_RESET_MODE = "SYNC";
    wire [17:0] a = ASEL ? SIA : A;
    wire [17:0] b = BSEL ? SIB : B;
    wire signed [35:0] ax = {{18{ASIGN & a[17]}}, a};
    wire signed [35:0] bx = {{18{BSIGN & b[17]}}, b};
    assign DOUT = ax * bx;
    assign SOA = a;
    assign SOB = b;
endmodule

// Single-port block RAM, 9/18/36-bit data. The 18,432 bits are addressed by
// AD[13:3] (9-bit words), AD[13:4] (18-bit) or AD[13:5] (36-bit); for the two
// wider modes the low address bits are per-byte write enables.
//   WRITE_MODE 0: output unchanged on write, 1: new data, 2: old data
// Only READ_MODE = 0 (no output pipeline register) is modelled.
module SPX9 (DO, DI, BLKSEL, AD, WRE, CLK, CE, OCE, RESET);
    parameter READ_MODE = 1'b0;
    parameter WRITE_MODE = 2'b00;
    parameter BIT_WIDTH = 36;
    parameter BLK_SEL = 3'b000;
    parameter RESET_MODE = "SYNC";
    parameter INIT_RAM_00 = 288'h0;
    parameter INIT_RAM_01 = 288'h0;
    parameter INIT_RAM_02 = 288'h0;
    parameter INIT_RAM_03 = 288'h0;
    parameter INIT_RAM_04 = 288'h0;
    parameter INIT_RAM_05 = 288'h0;
    parameter INIT_RAM_06 = 288'h0;
    parameter INIT_RAM_07 = 288'h0;
    parameter INIT_RAM_08 = 288'h0;
    parameter INIT_RAM_09 = 288'h0;
    parameter INIT_RAM_0A = 288'h0;
    parameter INIT_RAM_0B = 288'h0;
    parameter INIT_RAM_0C = 288'h0;
    parameter INIT_RAM_0D = 288'h0;
    parameter INIT_RAM_0E = 288'h0;
    parameter INIT_RAM_0F = 288'h0;
    parameter INIT_RAM_10 = 288'h0;
    parameter INIT_RAM_11 = 288'h0;
    parameter INIT_RAM_12 = 288'h0;
    parameter INIT_RAM_13 = 288'h0;
    parameter INIT_RAM_14 = 288'h0;
    parameter INIT_RAM_15 = 288'h0;
    parameter INIT_RAM_16 = 288'h0;
    parameter INIT_RAM_17 = 288'h0;
    parameter INIT_RAM_18 = 288'h0;
    parameter INIT_RAM_19 = 288'h0;
    parameter INIT_RAM_1A = 288'h0;
    parameter INIT_RAM_1B = 288'h0;
    parameter INIT_RAM_1C = 288'h0;
    parameter INIT_RAM_1D = 288'h0;
    parameter INIT_RAM_1E = 288'h0;
    parameter INIT_RAM_1F = 288'h0;
    parameter INIT_RAM_20 = 288'h0;
    parameter INIT_RAM_21 = 288'h0;
    parameter INIT_RAM_22 = 288'h0;
    parameter INIT_RAM_23 = 288'h0;
    parameter INIT_RAM_24 = 288'h0;
    parameter INIT_RAM_25 = 288'h0;
    parameter INIT_RAM_26 = 288'h0;
    parameter INIT_RAM_27 = 288'h0;
    parameter INIT_RAM_28 = 288'h0;
    parameter INIT_RAM_29 = 288'h0;
    parameter INIT_RAM_2A = 288'h0;
    parameter INIT_RAM_2B = 288'h0;
    parameter INIT_RAM_2C = 288'h0;
    parameter INIT_RAM_2D = 288'h0;
    parameter INIT_RAM_2E = 288'h0;
    parameter INIT_RAM_2F = 288'h0;
    parameter INIT_RAM_30 = 288'h0;
    parameter INIT_RAM_31 = 288'h0;
    parameter INIT_RAM_32 = 288'h0;
    parameter INIT_RAM_33 = 288'h0;
    parameter INIT_RAM_34 = 288'h0;
    parameter INIT_RAM_35 = 288'h0;
    parameter INIT_RAM_36 = 288'h0;
    parameter INIT_RAM_37 = 288'h0;
    parameter INIT_RAM_38 = 288'h0;
    parameter INIT_RAM_39 = 288'h0;
    parameter INIT_RAM_3A = 288'h0;
    parameter INIT_RAM_3B = 288'h0;
    parameter INIT_RAM_3C = 288'h0;
    parameter INIT_RAM_3D = 288'h0;
    parameter INIT_RAM_3E = 288'h0;
    parameter INIT_RAM_3F = 288'h0;
    output reg [35:0] DO;
    input  [35:0] DI;
    input  [2:0]  BLKSEL;
    input  [13:0] AD;
    input  WRE, CLK, CE, OCE, RESET;

    reg [18431:0] mem;
    integer k;
    initial begin
        DO = 36'd0;
        mem[0*288 +: 288] = INIT_RAM_00;
        mem[1*288 +: 288] = INIT_RAM_01;
        mem[2*288 +: 288] = INIT_RAM_02;
        mem[3*288 +: 288] = INIT_RAM_03;
        mem[4*288 +: 288] = INIT_RAM_04;
        mem[5*288 +: 288] = INIT_RAM_05;
        mem[6*288 +: 288] = INIT_RAM_06;
        mem[7*288 +: 288] = INIT_RAM_07;
        mem[8*288 +: 288] = INIT_RAM_08;
        mem[9*288 +: 288] = INIT_RAM_09;
        mem[10*288 +: 288] = INIT_RAM_0A;
        mem[11*288 +: 288] = INIT_RAM_0B;
        mem[12*288 +: 288] = INIT_RAM_0C;
        mem[13*288 +: 288] = INIT_RAM_0D;
        mem[14*288 +: 288] = INIT_RAM_0E;
        mem[15*288 +: 288] = INIT_RAM_0F;
        mem[16*288 +: 288] = INIT_RAM_10;
        mem[17*288 +: 288] = INIT_RAM_11;
        mem[18*288 +: 288] = INIT_RAM_12;
        mem[19*288 +: 288] = INIT_RAM_13;
        mem[20*288 +: 288] = INIT_RAM_14;
        mem[21*288 +: 288] = INIT_RAM_15;
        mem[22*288 +: 288] = INIT_RAM_16;
        mem[23*288 +: 288] = INIT_RAM_17;
        mem[24*288 +: 288] = INIT_RAM_18;
        mem[25*288 +: 288] = INIT_RAM_19;
        mem[26*288 +: 288] = INIT_RAM_1A;
        mem[27*288 +: 288] = INIT_RAM_1B;
        mem[28*288 +: 288] = INIT_RAM_1C;
        mem[29*288 +: 288] = INIT_RAM_1D;
        mem[30*288 +: 288] = INIT_RAM_1E;
        mem[31*288 +: 288] = INIT_RAM_1F;
        mem[32*288 +: 288] = INIT_RAM_20;
        mem[33*288 +: 288] = INIT_RAM_21;
        mem[34*288 +: 288] = INIT_RAM_22;
        mem[35*288 +: 288] = INIT_RAM_23;
        mem[36*288 +: 288] = INIT_RAM_24;
        mem[37*288 +: 288] = INIT_RAM_25;
        mem[38*288 +: 288] = INIT_RAM_26;
        mem[39*288 +: 288] = INIT_RAM_27;
        mem[40*288 +: 288] = INIT_RAM_28;
        mem[41*288 +: 288] = INIT_RAM_29;
        mem[42*288 +: 288] = INIT_RAM_2A;
        mem[43*288 +: 288] = INIT_RAM_2B;
        mem[44*288 +: 288] = INIT_RAM_2C;
        mem[45*288 +: 288] = INIT_RAM_2D;
        mem[46*288 +: 288] = INIT_RAM_2E;
        mem[47*288 +: 288] = INIT_RAM_2F;
        mem[48*288 +: 288] = INIT_RAM_30;
        mem[49*288 +: 288] = INIT_RAM_31;
        mem[50*288 +: 288] = INIT_RAM_32;
        mem[51*288 +: 288] = INIT_RAM_33;
        mem[52*288 +: 288] = INIT_RAM_34;
        mem[53*288 +: 288] = INIT_RAM_35;
        mem[54*288 +: 288] = INIT_RAM_36;
        mem[55*288 +: 288] = INIT_RAM_37;
        mem[56*288 +: 288] = INIT_RAM_38;
        mem[57*288 +: 288] = INIT_RAM_39;
        mem[58*288 +: 288] = INIT_RAM_3A;
        mem[59*288 +: 288] = INIT_RAM_3B;
        mem[60*288 +: 288] = INIT_RAM_3C;
        mem[61*288 +: 288] = INIT_RAM_3D;
        mem[62*288 +: 288] = INIT_RAM_3E;
        mem[63*288 +: 288] = INIT_RAM_3F;
        // undefined initial bits are zero in the bitstream
        for (k = 0; k < 18432; k = k + 1) if (mem[k] === 1'bx) mem[k] = 1'b0;
        if (READ_MODE != 0) $display("SPX9: READ_MODE=1 is not modelled");
    end

    localparam NBYTES = BIT_WIDTH / 9;
    wire [10:0] word = (BIT_WIDTH == 9) ? AD[13:3] : (BIT_WIDTH == 18) ? {1'b0, AD[13:4]} : {2'b00, AD[13:5]};
    wire [14:0] base = word * BIT_WIDTH;
    wire [3:0]  be   = (BIT_WIDTH == 9) ? 4'b0001 : (BIT_WIDTH == 18) ? {2'b00, AD[1:0]} : AD[3:0];

    reg [35:0] old, nxt;
    integer b;
    always @(posedge CLK) begin
        if (RESET && RESET_MODE == "SYNC") begin
            DO <= 36'd0;
        end else if (CE && BLKSEL == BLK_SEL) begin
            old = 36'd0;
            for (b = 0; b < BIT_WIDTH; b = b + 1) old[b] = mem[base + b];
            nxt = old;
            if (WRE) begin
                for (b = 0; b < NBYTES; b = b + 1)
                    if (be[b]) nxt[b*9 +: 9] = DI[b*9 +: 9];
                for (b = 0; b < BIT_WIDTH; b = b + 1) mem[base + b] = nxt[b];
                if (WRITE_MODE == 1)      DO <= nxt;
                else if (WRITE_MODE == 2) DO <= old;
            end else begin
                DO <= old;
            end
        end
    end
    always @(posedge RESET) if (RESET_MODE == "ASYNC") DO <= 36'd0;
endmodule

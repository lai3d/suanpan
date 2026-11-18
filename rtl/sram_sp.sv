// Single-port synchronous-read memory. Maps to SRAM macros on ASIC and to
// block RAM on FPGA. Optional $readmemh preload for ROM-style use.
// DEPTH may be less than 2^AW; addresses at or above it must not be written
// and read as don't-care.
module sram_sp #(
    parameter int DW    = 64,
    parameter int AW    = 8,
    parameter int DEPTH = 1 << AW,
    parameter     FILE  = ""
) (
    input  logic          clk,
    input  logic          we,
    input  logic [AW-1:0] addr,
    input  logic [DW-1:0] wdata,
    output logic [DW-1:0] rdata
);
    logic [DW-1:0] mem [0:DEPTH-1];

    initial begin
        if (FILE != "") $readmemh(FILE, mem);
    end

    always_ff @(posedge clk) begin
        if (we) mem[addr] <= wdata;
        rdata <= mem[addr];
    end
endmodule

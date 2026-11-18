// Single-port synchronous-read memory. Maps to SRAM macros on ASIC and to
// block RAM on FPGA. Optional $readmemh preload for ROM-style use.
module sram_sp #(
    parameter int DW   = 64,
    parameter int AW   = 8,
    parameter     FILE = ""
) (
    input  logic          clk,
    input  logic          we,
    input  logic [AW-1:0] addr,
    input  logic [DW-1:0] wdata,
    output logic [DW-1:0] rdata
);
    logic [DW-1:0] mem [0:(1<<AW)-1];

    initial begin
        if (FILE != "") $readmemh(FILE, mem);
    end

    always_ff @(posedge clk) begin
        if (we) mem[addr] <= wdata;
        rdata <= mem[addr];
    end
endmodule

// P-lane signed 8x8 dot product: sum_k  w[k] * x[k]
// This is the compute core of the chip. Two pipeline stages:
//   stage 1: P multipliers, products registered (when en)
//   stage 2: balanced adder tree over the registered products (combinational;
//            the caller registers `sum`)
// P must be a power of two.
module dot_unit #(
    parameter int P = 8
) (
    input  logic                         clk,
    input  logic                         en,
    input  logic        [P*8-1:0]        w,   // P signed int8 weights
    input  logic        [P*8-1:0]        x,   // P signed int8 activations
    output logic signed [15+$clog2(P):0] sum
);
    localparam int W = 16 + $clog2(P);

    logic [P*16-1:0] prod;
    always_ff @(posedge clk) begin
        if (en)
            for (int k = 0; k < P; k++)
                prod[k*16 +: 16] <= $signed(w[k*8 +: 8]) * $signed(x[k*8 +: 8]);
    end

    // Binary tree stored heap-style: node i has children 2i+1 and 2i+2,
    // the P leaves are nodes P-1 .. 2P-2, the root is node 0.
    logic [(2*P-1)*W-1:0] t;
    always_comb begin
        t = '0;
        for (int k = 0; k < P; k++)
            t[(P-1+k)*W +: W] = W'($signed(prod[k*16 +: 16]));
        for (int i = P - 2; i >= 0; i--)
            t[i*W +: W] = t[(2*i+1)*W +: W] + t[(2*i+2)*W +: W];
        sum = $signed(t[0 +: W]);
    end
endmodule

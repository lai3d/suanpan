// P-lane signed 8x8 dot product: sum_k  w[k] * x[k]
// This is the compute core of the chip: P multipliers feeding an adder tree.
module dot_unit #(
    parameter int P = 8
) (
    input  logic        [P*8-1:0] w,   // P signed int8 weights
    input  logic        [P*8-1:0] x,   // P signed int8 activations
    output logic signed [15+$clog2(P):0] sum
);
    always_comb begin
        sum = '0;
        for (int k = 0; k < P; k++)
            sum = sum + $signed(w[k*8 +: 8]) * $signed(x[k*8 +: 8]);
    end
endmodule

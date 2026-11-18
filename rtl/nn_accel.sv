// nn_accel: a minimal INT8 inference accelerator for convolutional and
// fully-connected layers.
//
// Datapath (per output value):
//     acc  = sum(w_int8 * x_int8)          P products per cycle
//     acc += bias_int32
//     hidden layer: y = clamp((acc * M + 2^(S-1)) >>> S, 0, 127)  -> next buffer
//     last layer  : raw acc streamed out as a logit, argmax tracked
//
// Timing: the MAC loop is a 3-stage pipeline (multiply, adder tree,
// accumulate). Requantisation runs once per output, so it is spread over
// several cycles with one small 9x17 multiplier (acc is consumed a byte at a
// time, MSB first) instead of a single wide multiply-shift-clamp path.
//
// Layers. Activations are stored HWC with the channel count padded to a
// multiple of P, so a pixel is a whole number of P-byte words and each kernel
// row of a convolution window is a contiguous run of words. One output value
// is the dot product of the weights with n_rows runs of row_len words; after
// each run the read pointer skips row_skip words to reach the next image row.
// The window slides by step_x words per output column and step_y words per
// output row. A fully-connected layer is the special case of one output pixel
// whose single "kernel row" covers the whole input.
//
// The network is a small table of layer descriptors (the "program"), held in
// registers that the host writes through the cfg_* port while the core is
// idle. Each 144-bit entry is:
//     [11:0]    row_len    words per kernel row
//     [15:12]   n_rows     kernel rows
//     [27:16]   row_skip   words skipped after each kernel row
//     [39:28]   step_x     window step per output column, in words
//     [55:40]   step_y     window step per output row, in words
//     [63:56]   out_w      output columns
//     [71:64]   out_h      output rows
//     [83:72]   out_ch     outputs per output pixel
//     [103:84]  w_base     first weight word of the layer
//     [115:104] b_base     first bias of the layer
//     [131:116] mult       requantisation multiplier M
//     [137:132] shift      requantisation shift S (>= 1)
//     [138]     last       final layer: emit logits instead of requantising
//
// Weights for one output pixel are out_ch consecutive kernels, and the same
// kernels are reused for every pixel. Activations ping-pong between two
// buffers: layer 0 reads A / writes B, layer 1 reads B / writes A, and so on.
// The host loads the input into A.
module nn_accel #(
    parameter int P          = 8,    // MAC lanes (bytes per memory word)
    parameter int W_AW       = 13,   // weight memory address bits (words)
    parameter int A_AW       = 10,   // activation buffer address bits (words)
    parameter int B_AW       = 7,    // bias memory address bits
    parameter int MAX_LAYERS = 4
) (
    input  logic                clk,
    input  logic                rst_n,

    // host: program the layer table (only while idle)
    input  logic                cfg_we,
    input  logic [$clog2(MAX_LAYERS)-1:0] cfg_addr,
    input  logic [143:0]        cfg_data,

    // host: load input activations into buffer A (only while idle)
    input  logic                in_we,
    input  logic [A_AW-1:0]     in_addr,
    input  logic [P*8-1:0]      in_data,

    // control
    input  logic                start,
    input  logic [2:0]          num_layers,
    output logic                busy,
    output logic                done,       // 1-cycle pulse
    output logic [7:0]          class_idx,  // argmax of the final layer

    // final-layer logits, streamed as they are produced
    output logic                logit_valid,
    output logic [7:0]          logit_idx,
    output logic signed [31:0]  logit_data
);
    localparam int LP = $clog2(P);

    typedef enum logic [3:0] {
        S_IDLE, S_LOAD, S_LAYER, S_NEURON, S_MAC, S_DRAIN, S_BIAS,
        S_MUL, S_MULF, S_RADD, S_SHIFT, S_CLAMP, S_OUT, S_NEXT, S_DONE
    } state_t;
    state_t state;

    // ---------------------------------------------------------------- config
    logic [143:0] cfg_mem [0:MAX_LAYERS-1];
    always_ff @(posedge clk) if (cfg_we) cfg_mem[cfg_addr] <= cfg_data;

    logic [2:0]   layer;
    logic [143:0] cfg;
    assign cfg = cfg_mem[layer[$clog2(MAX_LAYERS)-1:0]];

    // fields of the current layer, latched in S_LOAD  (*_last = count - 1)
    logic [11:0] c_row_last, c_row_skip, c_step_x, c_ch_last, c_b_base;
    logic [3:0]  c_nrow_last;
    logic [15:0] c_step_y;
    logic [7:0]  c_ow_last, c_oh_last;
    logic [19:0] c_w_base;
    logic [15:0] c_mult;
    logic [5:0]  c_shift;
    logic        c_last;
    logic signed [47:0] c_round;   // 2^(shift-1), decoded once per layer

    // -------------------------------------------------------------- counters
    logic [15:0]     oidx;       // output index within the layer (all pixels)
    logic [11:0]     och;        // output channel within the pixel
    logic [7:0]      px, py;     // output pixel
    logic [11:0]     kc;         // word within the kernel row
    logic [3:0]      kr;         // kernel row
    logic [A_AW-1:0] a_ptr;      // activation read pointer
    logic [A_AW-1:0] pix_base;   // window origin for the current output pixel
    logic [A_AW-1:0] row_base;   // window origin for the current output row
    logic [W_AW-1:0] w_ptr;      // running weight address
    logic [B_AW-1:0] b_ptr;      // running bias address
    logic            src_sel;    // 0: read A / write B,  1: read B / write A
    assign src_sel = layer[0];

    logic row_end, ch_end, col_end, layer_end;
    assign row_end   = (kc == c_row_last);
    assign ch_end    = (och == c_ch_last);
    assign col_end   = (px == c_ow_last);
    assign layer_end = ch_end && col_end && (py == c_oh_last);

    // -------------------------------------------------------------- memories
    logic [P*8-1:0] w_q, a_q, bufa_q, bufb_q;
    logic [31:0]    b_q;
    logic           dst_we;
    logic [A_AW-1:0] dst_addr;
    logic [P*8-1:0] dst_data;

    sram_sp #(.DW(P*8), .AW(W_AW)) u_wmem (
        .clk, .we(1'b0), .addr(w_ptr), .wdata('0), .rdata(w_q));

    sram_sp #(.DW(32), .AW(B_AW)) u_bmem (
        .clk, .we(1'b0), .addr(b_ptr), .wdata('0), .rdata(b_q));

    logic            bufa_we, bufb_we;
    logic [A_AW-1:0] bufa_addr, bufb_addr;
    assign bufa_we   = in_we | (dst_we & src_sel);
    assign bufb_we   = dst_we & ~src_sel;
    assign bufa_addr = in_we ? in_addr : (dst_we & src_sel) ? dst_addr : a_ptr;
    assign bufb_addr = bufb_we ? dst_addr : a_ptr;

    sram_sp #(.DW(P*8), .AW(A_AW)) u_bufa (
        .clk, .we(bufa_we), .addr(bufa_addr),
        .wdata(in_we ? in_data : dst_data), .rdata(bufa_q));
    sram_sp #(.DW(P*8), .AW(A_AW)) u_bufb (
        .clk, .we(bufb_we), .addr(bufb_addr), .wdata(dst_data), .rdata(bufb_q));

    assign a_q = src_sel ? bufb_q : bufa_q;

    // ------------------------------------------------------------- MAC pipe
    //   cycle n   : address issued (state == S_MAC)
    //   cycle n+1 : memory data valid, P products registered     (v1)
    //   cycle n+2 : adder tree over the products, registered      (v2)
    //   cycle n+3 : partial sum added into the accumulator        (v3)
    logic signed [15+LP:0] dot, dot_r;
    logic                  v1, v2, v3;
    logic signed [31:0]    acc;

    // one shared 32-bit adder: MAC accumulation or bias add
    logic signed [31:0] acc_y;
    assign acc_y = acc + ((state == S_BIAS) ? $signed(b_q) : 32'(dot_r));

    dot_unit #(.P(P)) u_dot (.clk, .en(v1), .w(w_q), .x(a_q), .sum(dot));

    // ------------------------------------------------------------- requant
    // prod = acc * M by Horner's rule over the four bytes of acc, MSB first:
    //   prod = ((b3*M * 256 + b2*M) * 256 + b1*M) * 256 + b0*M
    // acc doubles as the shift register that feeds the bytes out.
    logic signed [8:0]  chunk;   // top byte of acc; signed only for b3
    logic signed [25:0] pp;      // chunk * M
    logic signed [47:0] prod;
    logic [1:0]         mcnt;
    logic [7:0]         q;
    assign chunk = {(mcnt == 2'd0) & acc[31], acc[31:24]};

    // one shared 48-bit adder: Horner step (prod*256 + pp) or rounding add
    logic signed [47:0] add_a, add_b, add_y;
    assign add_a = (state == S_RADD) ? prod    : (prod <<< 8);
    assign add_b = (state == S_RADD) ? c_round : 48'(pp);
    assign add_y = add_a + add_b;

    // output word assembly: collect P bytes, then one full-word write
    logic [P*8-1:0] out_word, out_word_n;
    logic           word_full;
    always_comb begin
        for (int k = 0; k < P; k++)
            out_word_n[k*8 +: 8] = (oidx[LP-1:0] == LP'(k)) ? q : out_word[k*8 +: 8];
    end
    assign word_full = (oidx[LP-1:0] == LP'(P-1)) || layer_end;
    assign dst_we    = (state == S_OUT) && !c_last && word_full;
    assign dst_addr  = oidx[LP +: A_AW];
    assign dst_data  = out_word_n;

    logic signed [31:0] best;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            layer       <= '0;
            oidx        <= '0;
            och         <= '0;
            px          <= '0;
            py          <= '0;
            kc          <= '0;
            kr          <= '0;
            a_ptr       <= '0;
            pix_base    <= '0;
            row_base    <= '0;
            w_ptr       <= '0;
            b_ptr       <= '0;
            c_row_last  <= '0;
            c_nrow_last <= '0;
            c_row_skip  <= '0;
            c_step_x    <= '0;
            c_step_y    <= '0;
            c_ow_last   <= '0;
            c_oh_last   <= '0;
            c_ch_last   <= '0;
            c_w_base    <= '0;
            c_b_base    <= '0;
            c_mult      <= '0;
            c_shift     <= '0;
            c_last      <= 1'b0;
            c_round     <= '0;
            v1          <= 1'b0;
            v2          <= 1'b0;
            v3          <= 1'b0;
            dot_r       <= '0;
            acc         <= '0;
            pp          <= '0;
            prod        <= '0;
            mcnt        <= '0;
            q           <= '0;
            out_word    <= '0;
            best        <= '0;
            busy        <= 1'b0;
            done        <= 1'b0;
            class_idx   <= '0;
            logit_valid <= 1'b0;
            logit_idx   <= '0;
            logit_data  <= '0;
        end else begin
            done        <= 1'b0;
            logit_valid <= 1'b0;

            v1 <= (state == S_MAC);
            v2 <= v1;
            v3 <= v2;
            if (v2) dot_r <= dot;
            if (v3) acc   <= acc_y;

            case (state)
                S_IDLE: if (start) begin
                    layer <= '0;
                    busy  <= 1'b1;
                    state <= S_LOAD;
                end

                S_LOAD: begin
                    c_row_last  <= cfg[11:0]  - 12'd1;
                    c_nrow_last <= cfg[15:12] - 4'd1;
                    c_row_skip  <= cfg[27:16];
                    c_step_x    <= cfg[39:28];
                    c_step_y    <= cfg[55:40];
                    c_ow_last   <= cfg[63:56] - 8'd1;
                    c_oh_last   <= cfg[71:64] - 8'd1;
                    c_ch_last   <= cfg[83:72] - 12'd1;
                    c_w_base    <= cfg[103:84];
                    c_b_base    <= cfg[115:104];
                    c_mult      <= cfg[131:116];
                    c_shift     <= cfg[137:132];
                    c_last      <= cfg[138];
                    state       <= S_LAYER;
                end

                S_LAYER: begin
                    w_ptr    <= c_w_base[W_AW-1:0];
                    b_ptr    <= c_b_base[B_AW-1:0];
                    oidx     <= '0;
                    och      <= '0;
                    px       <= '0;
                    py       <= '0;
                    pix_base <= '0;
                    row_base <= '0;
                    out_word <= '0;
                    c_round  <= 48'sd1 <<< (c_shift - 6'd1);
                    state    <= S_NEURON;
                end

                S_NEURON: begin
                    acc   <= '0;
                    a_ptr <= pix_base;
                    kc    <= '0;
                    kr    <= '0;
                    state <= S_MAC;
                end

                // Walk the window: row_len words, skip to the next image row,
                // repeat for n_rows rows. Weights are read strictly in order.
                S_MAC: begin
                    w_ptr <= w_ptr + 1'b1;
                    if (row_end) begin
                        kc    <= '0;
                        kr    <= kr + 4'd1;
                        a_ptr <= a_ptr + c_row_skip[A_AW-1:0] + 1'b1;
                        if (kr == c_nrow_last) state <= S_DRAIN;
                    end else begin
                        kc    <= kc + 12'd1;
                        a_ptr <= a_ptr + 1'b1;
                    end
                end

                S_DRAIN: if (!v1 && !v2 && !v3) state <= S_BIAS;

                S_BIAS: begin
                    acc  <= acc_y;
                    mcnt <= '0;
                    if (c_last) state <= S_OUT;
                    else        state <= S_MUL;
                end

                S_MUL: begin
                    pp   <= chunk * $signed({1'b0, c_mult});
                    acc  <= {acc[23:0], 8'h00};
                    prod <= (mcnt == 2'd0) ? '0 : add_y;
                    mcnt <= mcnt + 2'd1;
                    if (mcnt == 2'd3) state <= S_MULF;
                end

                S_MULF: begin
                    prod  <= add_y;
                    state <= S_RADD;
                end

                S_RADD: begin
                    prod  <= add_y;
                    state <= S_SHIFT;
                end

                S_SHIFT: begin
                    prod  <= prod >>> c_shift;
                    state <= S_CLAMP;
                end

                S_CLAMP: begin
                    // negative -> 0; any bit above bit 6 set -> saturate at 127
                    q     <= prod[47] ? 8'd0 : (|prod[46:7]) ? 8'd127 : prod[7:0];
                    state <= S_OUT;
                end

                S_OUT: begin
                    if (c_last) begin
                        logit_valid <= 1'b1;
                        logit_idx   <= oidx[7:0];
                        logit_data  <= acc;
                        if (oidx == '0 || acc > best) begin
                            best      <= acc;
                            class_idx <= oidx[7:0];
                        end
                    end else begin
                        out_word <= word_full ? '0 : out_word_n;
                    end

                    oidx <= oidx + 16'd1;
                    if (!ch_end) begin
                        // next output channel of the same pixel
                        och   <= och + 12'd1;
                        b_ptr <= b_ptr + 1'b1;
                    end else begin
                        // next output pixel: same kernels, shifted window
                        och   <= '0;
                        w_ptr <= c_w_base[W_AW-1:0];
                        b_ptr <= c_b_base[B_AW-1:0];
                        if (!col_end) begin
                            px       <= px + 8'd1;
                            pix_base <= pix_base + c_step_x[A_AW-1:0];
                        end else begin
                            px       <= '0;
                            py       <= py + 8'd1;
                            pix_base <= row_base + c_step_y[A_AW-1:0];
                            row_base <= row_base + c_step_y[A_AW-1:0];
                        end
                    end
                    if (layer_end) state <= S_NEXT;
                    else           state <= S_NEURON;
                end

                S_NEXT: begin
                    if (c_last || layer == num_layers - 3'd1) begin
                        state <= S_DONE;
                    end else begin
                        layer <= layer + 3'd1;
                        state <= S_LOAD;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

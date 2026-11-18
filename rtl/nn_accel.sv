// nn_accel: a minimal INT8 inference accelerator for convolutional and
// fully-connected layers.
//
// Datapath (per output value):
//     acc  = sum(w_int8 * x_int8)          P products per cycle
//     acc += bias_int32
//     hidden layer: y = clamp((acc * M + 2^(S-1)) >>> S, 0, 127)  -> next buffer
//     last layer  : raw acc streamed out as a logit, argmax tracked
//
// Timing: everything after address generation is a feed-forward pipeline.
// The MAC loop has three stages (multiply, adder tree, accumulate). When the
// last word of an output reaches the accumulator its sum drops into the post
// pipeline: four 9x17 multiplies (one per byte of acc, times M), recombine in
// two steps, round, shift, clamp, write back.
//
// Throughput: because the post pipeline accepts a new sum every cycle, the
// front end never waits. It issues the windows of consecutive outputs back to
// back, so an output costs exactly as many cycles as its window has words.
// The bias is the initial value of the accumulator and costs nothing.
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
    parameter int MAX_LAYERS = 4,
    // optional $readmemh images for the weight and bias memories (ROM use)
    parameter     W_FILE     = "",
    parameter     B_FILE     = ""
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

    // front end: walks layers, output pixels and kernel windows
    typedef enum logic [2:0] {
        S_IDLE, S_LOAD, S_LAYER, S_MAC, S_DRAIN, S_NEXT, S_DONE
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

    sram_sp #(.DW(P*8), .AW(W_AW), .FILE(W_FILE)) u_wmem (
        .clk, .we(1'b0), .addr(w_ptr), .wdata('0), .rdata(w_q));

    sram_sp #(.DW(32), .AW(B_AW), .FILE(B_FILE)) u_bmem (
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
    //   cycle n+1 : memory data valid, registered                 (v0)
    //   cycle n+2 : P products registered                         (v1)
    //   cycle n+3 : adder tree over the products, registered      (v2)
    //   cycle n+4 : partial sum added into the accumulator        (v3)
    // Stage 0 keeps the memory read (and the buffer select mux) out of the
    // multiplier path.
    // Each word carries three tags down the pipe: f = first word of an
    // output, l = last word of an output, e = last word of the layer.
    logic signed [15+LP:0] dot, dot_r;
    logic                  v0, v1, v2, v3;
    logic                  f0, f1, f2, f3;
    logic                  l0, l1, l2, l3;
    logic                  e0, e1, e2, e3;
    logic [P*8-1:0]        w_r, a_r;           // stage-0 operands
    logic signed [31:0]    acc;
    logic signed [31:0]    bias_r0, bias_r1, bias_r2;   // bias, aligned with f0..f2

    logic mac_first, mac_last;
    assign mac_first = (kc == '0) && (kr == '0);
    assign mac_last  = row_end && (kr == c_nrow_last);

    // window origin of the next output, so its first word can be issued in
    // the cycle right after this output's last word
    logic [A_AW-1:0] pix_next;
    always_comb begin
        if (!ch_end)       pix_next = pix_base;
        else if (!col_end) pix_next = pix_base + c_step_x[A_AW-1:0];
        else               pix_next = row_base + c_step_y[A_AW-1:0];
    end

    // one 32-bit adder: the first word of an output starts from the bias
    logic signed [31:0] acc_y;
    assign acc_y = (f3 ? bias_r2 : acc) + 32'(dot_r);

    // the sum of an output is complete when its last word reaches stage 3
    logic handoff;
    assign handoff = v3 && l3;

    dot_unit #(.P(P)) u_dot (.clk, .en(v1), .w(w_r), .x(a_r), .sum(dot));

    // -------------------------------------------------------- post pipeline
    //   p0 : finished sum                       (also emits last-layer logits)
    //   p1 : b3*M, b2*M, b1*M, b0*M   where acc = {b3, b2, b1, b0}, b3 signed
    //   p2 : hi = b3*M << 8 + b2*M,   lo = b1*M << 8 + b0*M
    //   p3 : acc * M = (hi << 16) + lo
    //   p4 : + 2^(S-1)
    //   p5 : >>> S
    //   p6 : clamp to 0..127, then assembled into a word and written
    localparam int PN = 7;                   // post pipeline stages
    logic [PN-1:0]      pv;                  // stage valid
    logic [PN-1:0]      pe;                  // end-of-layer tag
    logic signed [31:0] p0_acc;
    logic signed [25:0] p1_pp [0:3];         // byte * M
    logic signed [34:0] p2_hi, p2_lo;
    logic signed [47:0] p3_prod, p4_rnd, p5_shf;
    logic [7:0]         q;
    logic [15:0]        o_cnt;               // outputs written in this layer

    // output word assembly: collect P bytes, then one full-word write
    logic [P*8-1:0] out_word, out_word_n;
    logic           word_full;
    always_comb begin
        for (int k = 0; k < P; k++)
            out_word_n[k*8 +: 8] = (o_cnt[LP-1:0] == LP'(k)) ? q : out_word[k*8 +: 8];
    end
    assign word_full = (o_cnt[LP-1:0] == LP'(P-1)) || pe[PN-1];
    assign dst_we    = pv[PN-1] && word_full;
    assign dst_addr  = o_cnt[LP +: A_AW];
    assign dst_data  = out_word_n;

    logic signed [31:0] best;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            layer       <= '0;
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
            v0          <= 1'b0;
            f0          <= 1'b0;
            l0          <= 1'b0;
            e0          <= 1'b0;
            w_r         <= '0;
            a_r         <= '0;
            bias_r0     <= '0;
            v1          <= 1'b0;
            v2          <= 1'b0;
            v3          <= 1'b0;
            f1          <= 1'b0;
            f2          <= 1'b0;
            f3          <= 1'b0;
            l1          <= 1'b0;
            l2          <= 1'b0;
            l3          <= 1'b0;
            e1          <= 1'b0;
            e2          <= 1'b0;
            e3          <= 1'b0;
            dot_r       <= '0;
            acc         <= '0;
            bias_r1     <= '0;
            bias_r2     <= '0;
            pv          <= '0;
            pe          <= '0;
            p0_acc      <= '0;
            for (int i = 0; i < 4; i++) p1_pp[i] <= '0;
            p2_hi       <= '0;
            p2_lo       <= '0;
            p3_prod     <= '0;
            p4_rnd      <= '0;
            p5_shf      <= '0;
            q           <= '0;
            o_cnt       <= '0;
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

            // ------------------------------------------------ MAC pipeline
            v0 <= (state == S_MAC);
            f0 <= (state == S_MAC) && mac_first;
            l0 <= (state == S_MAC) && mac_last;
            e0 <= (state == S_MAC) && mac_last && layer_end;
            v1 <= v0;  f1 <= f0;  l1 <= l0;  e1 <= e0;
            v2 <= v1;  f2 <= f1;  l2 <= l1;  e2 <= e1;
            v3 <= v2;  f3 <= f2;  l3 <= l2;  e3 <= e2;
            if (v0) begin
                w_r <= w_q;
                a_r <= a_q;
            end
            // b_q holds this output's bias only in the cycle its first word
            // is at stage 0 (windows can be a single word long)
            if (v0 && f0) bias_r0 <= $signed(b_q);
            if (v1 && f1) bias_r1 <= bias_r0;
            if (v2 && f2) bias_r2 <= bias_r1;
            if (v2) dot_r <= dot;
            if (v3) acc   <= acc_y;

            // ----------------------------------------------- post pipeline
            pv <= {pv[PN-2:1], pv[0] & ~c_last, handoff};
            pe <= {pe[PN-2:0], e3};
            if (handoff) p0_acc <= acc_y;
            if (pv[0]) begin
                p1_pp[3] <= $signed({p0_acc[31], p0_acc[31:24]}) * $signed({1'b0, c_mult});
                p1_pp[2] <= $signed({1'b0, p0_acc[23:16]}) * $signed({1'b0, c_mult});
                p1_pp[1] <= $signed({1'b0, p0_acc[15:8]})  * $signed({1'b0, c_mult});
                p1_pp[0] <= $signed({1'b0, p0_acc[7:0]})   * $signed({1'b0, c_mult});
            end
            if (pv[1]) begin
                p2_hi <= (35'(p1_pp[3]) <<< 8) + 35'(p1_pp[2]);
                p2_lo <= (35'(p1_pp[1]) <<< 8) + 35'(p1_pp[0]);
            end
            if (pv[2]) p3_prod <= (48'(p2_hi) <<< 16) + 48'(p2_lo);
            if (pv[3]) p4_rnd  <= p3_prod + c_round;
            if (pv[4]) p5_shf  <= p4_rnd >>> c_shift;
            // negative -> 0; any bit above bit 6 set -> saturate at 127
            if (pv[5]) q <= p5_shf[47] ? 8'd0 : (|p5_shf[46:7]) ? 8'd127 : p5_shf[7:0];
            if (pv[PN-1]) begin
                out_word <= word_full ? '0 : out_word_n;
                o_cnt    <= o_cnt + 16'd1;
            end

            // last layer: the sum itself is the result
            if (pv[0] && c_last) begin
                logit_valid <= 1'b1;
                logit_idx   <= o_cnt[7:0];
                logit_data  <= p0_acc;
                o_cnt       <= o_cnt + 16'd1;
                if (o_cnt == '0 || p0_acc > best) begin
                    best      <= p0_acc;
                    class_idx <= o_cnt[7:0];
                end
            end

            // --------------------------------------------------- front end
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
                    och      <= '0;
                    px       <= '0;
                    py       <= '0;
                    kc       <= '0;
                    kr       <= '0;
                    a_ptr    <= '0;
                    pix_base <= '0;
                    row_base <= '0;
                    o_cnt    <= '0;
                    out_word <= '0;
                    c_round  <= 48'sd1 <<< (c_shift - 6'd1);
                    state    <= S_MAC;
                end

                // Walk the window: row_len words, skip to the next image row,
                // repeat for n_rows rows. Weights are read strictly in order.
                // On the last word, every counter moves on to the next output
                // so that its first word follows in the very next cycle.
                S_MAC: begin
                    w_ptr <= w_ptr + 1'b1;
                    if (row_end) begin
                        kc    <= '0;
                        kr    <= kr + 4'd1;
                        a_ptr <= a_ptr + c_row_skip[A_AW-1:0] + 1'b1;
                    end else begin
                        kc    <= kc + 12'd1;
                        a_ptr <= a_ptr + 1'b1;
                    end

                    if (mac_last) begin
                        kc       <= '0;
                        kr       <= '0;
                        a_ptr    <= pix_next;
                        pix_base <= pix_next;
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
                                px <= px + 8'd1;
                            end else begin
                                px       <= '0;
                                py       <= py + 8'd1;
                                row_base <= pix_next;
                            end
                        end
                        if (layer_end) state <= S_DRAIN;
                    end
                end

                // end of layer: the next layer reads the buffer this one is
                // still writing, so let everything in flight finish first
                S_DRAIN: if (!v0 && !v1 && !v2 && !v3 && pv == '0) state <= S_NEXT;

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

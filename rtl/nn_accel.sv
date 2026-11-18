// nn_accel: a minimal INT8 inference accelerator for convolutional, pooling
// and fully-connected layers.
//
// Datapath (per output value):
//     acc  = sum(w_int8 * x_int8) + bias    P products per cycle
//     hidden layer: y = clamp((acc * M + 2^(S-1)) >>> S, 0, 127)  -> next buffer
//     last layer  : raw acc streamed out as a logit, argmax tracked
//     pooling     : y = lane-wise maximum over the window, written as a word
// bias, M and S come from one parameter word per output channel.
//
// Timing: everything after address generation is a feed-forward pipeline.
// The MAC loop has four stages (register operands, multiply, adder tree,
// accumulate). When the last word of an output reaches the accumulator its
// sum drops into the post pipeline: four 9x17 multiplies (one per byte of
// acc, times M), recombine in two steps, round, shift, clamp, write back.
//
// Throughput: because the post pipeline accepts a new sum every cycle, the
// front end never waits. It issues the windows of consecutive outputs back to
// back, so an output costs exactly as many cycles as its window has words.
// The bias is the initial value of the accumulator and costs nothing.
//
// Layers. Activations are stored HWC with the channel count padded to a
// multiple of P, so a pixel is a whole number of P-byte words. A window is
// kh rows of kw pixels. The read pointer visits rd_words words of each pixel,
// then moves pix_step to the next pixel or row_step to the next kernel row;
// the window origin moves by step_x / step_y words per output column / row.
//   convolution      reads every word of a pixel (rd_words = words per pixel)
//   fully connected  is one output pixel whose single "pixel" is the whole input
//   pooling          reads one word per pixel; each output is one word of the
//                    output pixel, so the window is walked once per word
// Padding is not stored. The window origin starts outside the image (org is
// negative, two's complement) and every word whose pixel lies outside
// in_w x in_h is replaced by zero as it is read.
//
// The network is a small table of layer descriptors (the "program") in a
// memory the host writes through the cfg_* port while the core is idle.
// Each 176-bit entry is, LSB first (keep in sync with model/train.py):
//     [3:0]     kw         kernel width in pixels
//     [7:4]     kh         kernel rows
//     [19:8]    rd_words   words read per window pixel
//     [31:20]   pix_step   pointer step to the next pixel of a kernel row
//     [43:32]   row_step   pointer step to the next kernel row
//     [55:44]   step_x     window step per output column, in words
//     [71:56]   step_y     window step per output row, in words
//     [87:72]   org        address of the first window's first word
//     [95:88]   in_w       input width in pixels
//     [103:96]  in_h       input height in pixels
//     [107:104] pad        padding in pixels
//     [111:108] stride     stride in pixels
//     [119:112] out_w      output columns
//     [127:120] out_h      output rows
//     [139:128] out_ch     outputs per output pixel
//     [159:140] w_base     first weight word of the layer
//     [171:160] b_base     first parameter word of the layer
//     [172]     pool       max pooling instead of multiply-accumulate
//     [173]     last       final layer: emit logits instead of requantising
//
// Parameter word, one per output channel: [31:0] bias, [47:32] M, [53:48] S.
//
// Weights for one output pixel are out_ch consecutive kernels, and the same
// kernels are reused for every pixel. Activations ping-pong between two
// buffers: layer 0 reads A / writes B, layer 1 reads B / writes A, and so on.
// The host loads the input into A.
module nn_accel #(
    parameter int P          = 8,    // MAC lanes (bytes per memory word)
    parameter int W_AW       = 13,   // weight memory address bits (words)
    parameter int A_AW       = 11,   // activation buffer address bits (words)
    parameter int B_AW       = 8,    // parameter memory address bits
    parameter int MAX_LAYERS = 8,
    // optional $readmemh images for the weight and parameter memories (ROM use)
    parameter     W_FILE     = "",
    parameter     B_FILE     = ""
) (
    input  logic                clk,
    input  logic                rst_n,

    // host: program the layer table (only while idle)
    input  logic                cfg_we,
    input  logic [$clog2(MAX_LAYERS)-1:0] cfg_addr,
    input  logic [175:0]        cfg_data,

    // host: load input activations into buffer A (only while idle)
    input  logic                in_we,
    input  logic [A_AW-1:0]     in_addr,
    input  logic [P*8-1:0]      in_data,

    // control
    input  logic                start,
    input  logic [3:0]          num_layers,
    output logic                busy,
    output logic                done,       // 1-cycle pulse
    output logic [7:0]          class_idx,  // argmax of the final layer

    // final-layer logits, streamed as they are produced
    output logic                logit_valid,
    output logic [7:0]          logit_idx,
    output logic signed [31:0]  logit_data
);
    localparam int LP = $clog2(P);
    localparam int LW = $clog2(MAX_LAYERS);

    // front end: walks layers, output pixels and kernel windows
    typedef enum logic [2:0] {
        S_IDLE, S_FETCH, S_LOAD, S_LAYER, S_MAC, S_DRAIN, S_NEXT, S_DONE
    } state_t;
    state_t state;

    // ---------------------------------------------------------------- config
    logic [3:0]   layer;
    logic [175:0] cfg;

    sram_sp #(.DW(176), .AW(LW)) u_cfg (
        .clk, .we(cfg_we), .addr(cfg_we ? cfg_addr : layer[LW-1:0]),
        .wdata(cfg_data), .rdata(cfg));

    // fields of the current layer, latched in S_LOAD  (*_last = count - 1)
    logic [3:0]  c_kw_last, c_kh_last, c_pad, c_stride;
    logic [11:0] c_rd_last, c_pix_step, c_row_step, c_step_x, c_ch_last, c_b_base;
    logic [15:0] c_step_y, c_org;
    logic [7:0]  c_in_w, c_in_h, c_ow_last, c_oh_last;
    logic [19:0] c_w_base;
    logic        c_pool, c_last;

    // -------------------------------------------------------------- counters
    logic [11:0]     och;        // output channel (or output word, if pooling)
    logic [7:0]      px, py;     // output pixel
    logic [11:0]     kcw;        // word within the window pixel
    logic [3:0]      kx, kr;     // pixel within the kernel row, kernel row
    logic signed [8:0] ix0, iy0;        // image coordinates of the window origin
    logic signed [8:0] ix_cur, iy_cur;  // image coordinates of the current word
    logic [A_AW-1:0] a_ptr;      // activation read pointer
    logic [A_AW-1:0] pix_base;   // window origin for the current output pixel
    logic [A_AW-1:0] row_base;   // window origin for the current output row
    logic [W_AW-1:0] w_ptr;      // running weight address
    logic [B_AW-1:0] b_ptr;      // running parameter address
    logic            src_sel;    // 0: read A / write B,  1: read B / write A
    assign src_sel = layer[0];

    logic pix_end, row_end, mac_first, mac_last, ch_end, col_end, layer_end, oob;
    assign pix_end   = (kcw == c_rd_last);
    assign row_end   = pix_end && (kx == c_kw_last);
    assign mac_last  = row_end && (kr == c_kh_last);
    assign mac_first = (kcw == '0) && (kx == '0) && (kr == '0);
    assign ch_end    = (och == c_ch_last);
    assign col_end   = (px == c_ow_last);
    assign layer_end = ch_end && col_end && (py == c_oh_last);
    // the current word belongs to a pixel outside the image: read it as zero
    assign oob = iy_cur[8] || (iy_cur[7:0] >= c_in_h) ||
                 ix_cur[8] || (ix_cur[7:0] >= c_in_w);

    // Where the next output starts, so that its first word can be issued in
    // the cycle right after this output's last word.
    logic [A_AW-1:0]   pix_next, a_start;
    logic [11:0]       och_next;
    logic signed [8:0] ix0_next, iy0_next, neg_pad;
    assign neg_pad  = -$signed({5'd0, c_pad});
    assign och_next = ch_end ? 12'd0 : och + 12'd1;
    always_comb begin
        pix_next = pix_base;
        ix0_next = ix0;
        iy0_next = iy0;
        if (ch_end) begin
            if (!col_end) begin
                pix_next = pix_base + c_step_x[A_AW-1:0];
                ix0_next = ix0 + $signed({5'd0, c_stride});
            end else begin
                pix_next = row_base + c_step_y[A_AW-1:0];
                ix0_next = neg_pad;
                iy0_next = iy0 + $signed({5'd0, c_stride});
            end
        end
    end
    // pooling reads word `och` of every window pixel
    assign a_start = pix_next + (c_pool ? och_next[A_AW-1:0] : '0);

    // -------------------------------------------------------------- memories
    logic [P*8-1:0] w_q, a_q, bufa_q, bufb_q;
    logic [53:0]    b_q;         // {S[5:0], M[15:0], bias[31:0]}
    logic           dst_we;
    logic [A_AW-1:0] dst_addr;
    logic [P*8-1:0] dst_data;

    sram_sp #(.DW(P*8), .AW(W_AW), .FILE(W_FILE)) u_wmem (
        .clk, .we(1'b0), .addr(w_ptr), .wdata('0), .rdata(w_q));

    sram_sp #(.DW(54), .AW(B_AW), .FILE(B_FILE)) u_bmem (
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
    // multiplier path, and is where out-of-image words become zero.
    // Each word carries tags down the pipe: f = first word of an output,
    // l = last word of an output, e = last word of the layer, z = zero it.
    logic signed [15+LP:0] dot, dot_r;
    logic                  v0, v1, v2, v3;
    logic                  f0, f1, f2, f3;
    logic                  l0, l1, l2, l3;
    logic                  e0, e1, e2, e3;
    logic                  z0;
    logic [P*8-1:0]        w_r, a_r;           // stage-0 operands
    logic signed [31:0]    acc;
    logic [53:0]           par_r0, par_r1, par_r2;   // parameters, aligned with f0..f2

    // one 32-bit adder: the first word of an output starts from the bias
    logic signed [31:0] acc_y;
    assign acc_y = (f3 ? $signed(par_r2[31:0]) : acc) + 32'(dot_r);

    // the sum of an output is complete when its last word reaches stage 3
    logic handoff;
    assign handoff = v3 && l3 && !c_pool;

    dot_unit #(.P(P)) u_dot (.clk, .en(v1), .w(w_r), .x(a_r), .sum(dot));

    // ------------------------------------------------------------- pooling
    // Lane-wise running maximum of the stage-0 words of one window.
    // Activations are 0..127, so an unsigned compare is enough.
    logic [P*8-1:0] pool_acc, pool_y, pw_data;
    logic           pw_v;
    always_comb begin
        for (int k = 0; k < P; k++)
            pool_y[k*8 +: 8] = (f1 || a_r[k*8 +: 8] > pool_acc[k*8 +: 8])
                               ? a_r[k*8 +: 8] : pool_acc[k*8 +: 8];
    end

    // -------------------------------------------------------- post pipeline
    //   p0 : finished sum                       (also emits last-layer logits)
    //   p1 : b3*M, b2*M, b1*M, b0*M   where acc = {b3, b2, b1, b0}, b3 signed
    //   p2 : hi = b3*M << 8 + b2*M,   lo = b1*M << 8 + b0*M
    //   p3 : acc * M = (hi << 16) + lo
    //   p4 : + 2^(S-1)
    //   p5 : >>> S
    //   p6 : clamp to 0..127, then assembled into a word and written
    // M and S belong to the output, so they travel with it.
    localparam int PN = 7;                   // post pipeline stages
    logic [PN-1:0]      pv;                  // stage valid
    logic [PN-1:0]      pe;                  // end-of-layer tag
    logic signed [31:0] p0_acc;
    logic [15:0]        p0_m;
    logic [5:0]         p0_s, p1_s, p2_s, p3_s, p4_s;
    logic signed [25:0] p1_pp [0:3];         // byte * M
    logic signed [34:0] p2_hi, p2_lo;
    logic signed [47:0] p3_prod, p3_rc, p4_rnd, p5_shf;
    logic [7:0]         q;
    logic [15:0]        o_cnt;               // outputs (bytes, or words if pooling) written

    // output word assembly: collect P bytes, then one full-word write
    logic [P*8-1:0] out_word, out_word_n;
    logic           word_full;
    always_comb begin
        for (int k = 0; k < P; k++)
            out_word_n[k*8 +: 8] = (o_cnt[LP-1:0] == LP'(k)) ? q : out_word[k*8 +: 8];
    end
    assign word_full = (o_cnt[LP-1:0] == LP'(P-1)) || pe[PN-1];
    assign dst_we    = c_pool ? pw_v : (pv[PN-1] && word_full);
    assign dst_addr  = c_pool ? o_cnt[A_AW-1:0] : o_cnt[LP +: A_AW];
    assign dst_data  = c_pool ? pw_data : out_word_n;

    logic signed [31:0] best;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            layer       <= '0;
            och         <= '0;
            px          <= '0;
            py          <= '0;
            kcw         <= '0;
            kx          <= '0;
            kr          <= '0;
            ix0         <= '0;
            iy0         <= '0;
            ix_cur      <= '0;
            iy_cur      <= '0;
            a_ptr       <= '0;
            pix_base    <= '0;
            row_base    <= '0;
            w_ptr       <= '0;
            b_ptr       <= '0;
            c_kw_last   <= '0;
            c_kh_last   <= '0;
            c_pad       <= '0;
            c_stride    <= '0;
            c_rd_last   <= '0;
            c_pix_step  <= '0;
            c_row_step  <= '0;
            c_step_x    <= '0;
            c_step_y    <= '0;
            c_org       <= '0;
            c_in_w      <= '0;
            c_in_h      <= '0;
            c_ow_last   <= '0;
            c_oh_last   <= '0;
            c_ch_last   <= '0;
            c_w_base    <= '0;
            c_b_base    <= '0;
            c_pool      <= 1'b0;
            c_last      <= 1'b0;
            v0          <= 1'b0;
            v1          <= 1'b0;
            v2          <= 1'b0;
            v3          <= 1'b0;
            f0          <= 1'b0;
            f1          <= 1'b0;
            f2          <= 1'b0;
            f3          <= 1'b0;
            l0          <= 1'b0;
            l1          <= 1'b0;
            l2          <= 1'b0;
            l3          <= 1'b0;
            e0          <= 1'b0;
            e1          <= 1'b0;
            e2          <= 1'b0;
            e3          <= 1'b0;
            z0          <= 1'b0;
            w_r         <= '0;
            a_r         <= '0;
            par_r0      <= '0;
            par_r1      <= '0;
            par_r2      <= '0;
            dot_r       <= '0;
            acc         <= '0;
            pool_acc    <= '0;
            pw_data     <= '0;
            pw_v        <= 1'b0;
            pv          <= '0;
            pe          <= '0;
            p0_acc      <= '0;
            p0_m        <= '0;
            p0_s        <= '0;
            p1_s        <= '0;
            p2_s        <= '0;
            p3_s        <= '0;
            p4_s        <= '0;
            for (int i = 0; i < 4; i++) p1_pp[i] <= '0;
            p2_hi       <= '0;
            p2_lo       <= '0;
            p3_prod     <= '0;
            p3_rc       <= '0;
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
            z0 <= oob;
            v1 <= v0;  f1 <= f0;  l1 <= l0;  e1 <= e0;
            v2 <= v1;  f2 <= f1;  l2 <= l1;  e2 <= e1;
            v3 <= v2;  f3 <= f2;  l3 <= l2;  e3 <= e2;
            if (v0) begin
                w_r <= w_q;
                a_r <= z0 ? '0 : a_q;
            end
            // b_q holds this output's parameters only in the cycle its first
            // word is at stage 0 (windows can be a single word long)
            if (v0 && f0) par_r0 <= b_q;
            if (v1 && f1) par_r1 <= par_r0;
            if (v2 && f2) par_r2 <= par_r1;
            if (v2) dot_r <= dot;
            if (v3) acc   <= acc_y;

            // ----------------------------------------------------- pooling
            pw_v <= v1 && l1 && c_pool;
            if (v1 && c_pool) begin
                pool_acc <= pool_y;
                if (l1) pw_data <= pool_y;
            end
            if (pw_v) o_cnt <= o_cnt + 16'd1;

            // ----------------------------------------------- post pipeline
            pv <= {pv[PN-2:1], pv[0] & ~c_last, handoff};
            pe <= {pe[PN-2:0], e3};
            if (handoff) begin
                p0_acc <= acc_y;
                p0_m   <= par_r2[47:32];
                p0_s   <= par_r2[53:48];
            end
            if (pv[0]) begin
                p1_pp[3] <= $signed({p0_acc[31], p0_acc[31:24]}) * $signed({1'b0, p0_m});
                p1_pp[2] <= $signed({1'b0, p0_acc[23:16]}) * $signed({1'b0, p0_m});
                p1_pp[1] <= $signed({1'b0, p0_acc[15:8]})  * $signed({1'b0, p0_m});
                p1_pp[0] <= $signed({1'b0, p0_acc[7:0]})   * $signed({1'b0, p0_m});
                p1_s     <= p0_s;
            end
            if (pv[1]) begin
                p2_hi <= (35'(p1_pp[3]) <<< 8) + 35'(p1_pp[2]);
                p2_lo <= (35'(p1_pp[1]) <<< 8) + 35'(p1_pp[0]);
                p2_s  <= p1_s;
            end
            if (pv[2]) begin
                p3_prod <= (48'(p2_hi) <<< 16) + 48'(p2_lo);
                p3_rc   <= 48'sd1 <<< (p2_s - 6'd1);
                p3_s    <= p2_s;
            end
            if (pv[3]) begin
                p4_rnd <= p3_prod + p3_rc;
                p4_s   <= p3_s;
            end
            if (pv[4]) p5_shf <= p4_rnd >>> p4_s;
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
                    state <= S_FETCH;
                end

                // the layer table is a synchronous memory: one cycle to read
                S_FETCH: state <= S_LOAD;

                S_LOAD: begin
                    c_kw_last  <= cfg[3:0] - 4'd1;
                    c_kh_last  <= cfg[7:4] - 4'd1;
                    c_rd_last  <= cfg[19:8] - 12'd1;
                    c_pix_step <= cfg[31:20];
                    c_row_step <= cfg[43:32];
                    c_step_x   <= cfg[55:44];
                    c_step_y   <= cfg[71:56];
                    c_org      <= cfg[87:72];
                    c_in_w     <= cfg[95:88];
                    c_in_h     <= cfg[103:96];
                    c_pad      <= cfg[107:104];
                    c_stride   <= cfg[111:108];
                    c_ow_last  <= cfg[119:112] - 8'd1;
                    c_oh_last  <= cfg[127:120] - 8'd1;
                    c_ch_last  <= cfg[139:128] - 12'd1;
                    c_w_base   <= cfg[159:140];
                    c_b_base   <= cfg[171:160];
                    c_pool     <= cfg[172];
                    c_last     <= cfg[173];
                    state      <= S_LAYER;
                end

                S_LAYER: begin
                    w_ptr    <= c_w_base[W_AW-1:0];
                    b_ptr    <= c_b_base[B_AW-1:0];
                    och      <= '0;
                    px       <= '0;
                    py       <= '0;
                    kcw      <= '0;
                    kx       <= '0;
                    kr       <= '0;
                    ix0      <= neg_pad;
                    iy0      <= neg_pad;
                    ix_cur   <= neg_pad;
                    iy_cur   <= neg_pad;
                    a_ptr    <= c_org[A_AW-1:0];
                    pix_base <= c_org[A_AW-1:0];
                    row_base <= c_org[A_AW-1:0];
                    o_cnt    <= '0;
                    out_word <= '0;
                    state    <= S_MAC;
                end

                // Walk the window one word per cycle. Weights are read
                // strictly in order. On the last word, every counter moves on
                // to the next output so that its first word follows in the
                // very next cycle.
                S_MAC: begin
                    w_ptr <= w_ptr + 1'b1;
                    if (mac_last) begin
                        kcw      <= '0;
                        kx       <= '0;
                        kr       <= '0;
                        a_ptr    <= a_start;
                        pix_base <= pix_next;
                        ix0      <= ix0_next;
                        iy0      <= iy0_next;
                        ix_cur   <= ix0_next;
                        iy_cur   <= iy0_next;
                        och      <= och_next;
                        if (!ch_end) begin
                            // next output of the same pixel
                            b_ptr <= b_ptr + 1'b1;
                        end else begin
                            // next output pixel: same kernels, shifted window
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
                    end else if (row_end) begin
                        kcw    <= '0;
                        kx     <= '0;
                        kr     <= kr + 4'd1;
                        a_ptr  <= a_ptr + c_row_step[A_AW-1:0];
                        ix_cur <= ix0;
                        iy_cur <= iy_cur + 9'sd1;
                    end else if (pix_end) begin
                        kcw    <= '0;
                        kx     <= kx + 4'd1;
                        a_ptr  <= a_ptr + c_pix_step[A_AW-1:0];
                        ix_cur <= ix_cur + 9'sd1;
                    end else begin
                        kcw   <= kcw + 12'd1;
                        a_ptr <= a_ptr + 1'b1;
                    end
                end

                // end of layer: the next layer reads the buffer this one is
                // still writing, so let everything in flight finish first
                S_DRAIN: if (!v0 && !v1 && !v2 && !v3 && !pw_v && pv == '0) state <= S_NEXT;

                S_NEXT: begin
                    if (c_last || layer == num_layers - 4'd1) begin
                        state <= S_DONE;
                    end else begin
                        layer <= layer + 4'd1;
                        state <= S_FETCH;
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

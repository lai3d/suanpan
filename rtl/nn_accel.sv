// nn_accel: a minimal INT8 fully-connected inference accelerator.
//
// Datapath (per output neuron):
//     acc  = sum(w_int8 * x_int8)          P products per cycle
//     acc += bias_int32
//     hidden layer: y = clamp((acc * M + 2^(S-1)) >>> S, 0, 127)  -> next buffer
//     last layer  : raw acc streamed out as a logit, argmax tracked
//
// The network is described by a small layer table ("program") in cfg memory;
// each 80-bit entry is:
//     [11:0]  in_words   input length in P-byte words
//     [23:12] out_size   number of output neurons
//     [43:24] w_base     first weight word of the layer
//     [55:44] b_base     first bias of the layer
//     [71:56] mult       requantisation multiplier M
//     [77:72] shift      requantisation shift S (>= 1)
//     [78]    last       final layer: emit logits instead of requantising
//
// Activations ping-pong between two buffers: layer 0 reads A / writes B,
// layer 1 reads B / writes A, and so on. The host loads the input into A.
module nn_accel #(
    parameter int P          = 8,    // MAC lanes (bytes per memory word)
    parameter int W_AW       = 13,   // weight memory address bits (words)
    parameter int A_AW       = 7,    // activation buffer address bits (words)
    parameter int B_AW       = 7,    // bias memory address bits
    parameter int MAX_LAYERS = 4,
    parameter     W_FILE     = "",
    parameter     B_FILE     = "",
    parameter     CFG_FILE   = ""
) (
    input  logic                clk,
    input  logic                rst_n,

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
        S_IDLE, S_LAYER, S_NEURON, S_MAC, S_DRAIN,
        S_BIAS, S_MUL, S_RND, S_OUT, S_NEXT, S_DONE
    } state_t;
    state_t state;

    // ---------------------------------------------------------------- config
    logic [79:0] cfg_mem [0:MAX_LAYERS-1];
    initial if (CFG_FILE != "") $readmemh(CFG_FILE, cfg_mem);

    logic [2:0]  layer;
    logic [79:0] cfg;
    assign cfg = cfg_mem[layer[$clog2(MAX_LAYERS)-1:0]];

    logic [11:0] c_in_words, c_out_size, c_b_base;
    logic [19:0] c_w_base;
    logic [15:0] c_mult;
    logic [5:0]  c_shift;
    logic        c_last;
    assign c_in_words = cfg[11:0];
    assign c_out_size = cfg[23:12];
    assign c_w_base   = cfg[43:24];
    assign c_b_base   = cfg[55:44];
    assign c_mult     = cfg[71:56];
    assign c_shift    = cfg[77:72];
    assign c_last     = cfg[78];

    // -------------------------------------------------------------- counters
    logic [11:0]     neuron;   // output neuron index within the layer
    logic [11:0]     widx;     // input word index within the neuron
    logic [W_AW-1:0] w_ptr;    // running weight address
    logic            src_sel;  // 0: read A / write B,  1: read B / write A
    assign src_sel = layer[0];

    // -------------------------------------------------------------- memories
    logic [P*8-1:0] w_q, a_q, bufa_q, bufb_q;
    logic [31:0]    b_q;
    logic           dst_we;
    logic [A_AW-1:0] dst_addr;
    logic [P*8-1:0] dst_data;

    sram_sp #(.DW(P*8), .AW(W_AW), .FILE(W_FILE)) u_wmem (
        .clk, .we(1'b0), .addr(w_ptr), .wdata('0), .rdata(w_q));

    sram_sp #(.DW(32), .AW(B_AW), .FILE(B_FILE)) u_bmem (
        .clk, .we(1'b0), .addr(B_AW'(c_b_base + neuron)), .wdata('0), .rdata(b_q));

    logic            bufa_we, bufb_we;
    logic [A_AW-1:0] bufa_addr, bufb_addr;
    assign bufa_we   = in_we | (dst_we & src_sel);
    assign bufb_we   = dst_we & ~src_sel;
    assign bufa_addr = in_we ? in_addr : (dst_we & src_sel) ? dst_addr : widx[A_AW-1:0];
    assign bufb_addr = bufb_we ? dst_addr : widx[A_AW-1:0];

    sram_sp #(.DW(P*8), .AW(A_AW)) u_bufa (
        .clk, .we(bufa_we), .addr(bufa_addr),
        .wdata(in_we ? in_data : dst_data), .rdata(bufa_q));
    sram_sp #(.DW(P*8), .AW(A_AW)) u_bufb (
        .clk, .we(bufb_we), .addr(bufb_addr), .wdata(dst_data), .rdata(bufb_q));

    assign a_q = src_sel ? bufb_q : bufa_q;

    // ------------------------------------------------------------- MAC pipe
    //   cycle n   : address issued (state == S_MAC)
    //   cycle n+1 : memory data valid, P products summed        (v1)
    //   cycle n+2 : partial sum added into the accumulator       (v2)
    logic signed [15+LP:0] dot, dot_r;
    logic                  v1, v2;
    logic signed [31:0]    acc;

    dot_unit #(.P(P)) u_dot (.w(w_q), .x(a_q), .sum(dot));

    // ------------------------------------------------------------- requant
    logic signed [47:0] prod, rnd;
    logic [7:0]         q;
    assign rnd = (prod + (48'sd1 <<< (c_shift - 6'd1))) >>> c_shift;

    // output word assembly: collect P bytes, then one full-word write
    logic [P*8-1:0] out_word, out_word_n;
    logic           word_full;
    always_comb begin
        for (int k = 0; k < P; k++)
            out_word_n[k*8 +: 8] = (neuron[LP-1:0] == LP'(k)) ? q : out_word[k*8 +: 8];
    end
    assign word_full = (neuron[LP-1:0] == LP'(P-1)) || (neuron == c_out_size - 12'd1);
    assign dst_we    = (state == S_OUT) && !c_last && word_full;
    assign dst_addr  = neuron[LP +: A_AW];
    assign dst_data  = out_word_n;

    logic signed [31:0] best;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            layer       <= '0;
            neuron      <= '0;
            widx        <= '0;
            w_ptr       <= '0;
            v1          <= 1'b0;
            v2          <= 1'b0;
            dot_r       <= '0;
            acc         <= '0;
            prod        <= '0;
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
            if (v1) dot_r <= dot;
            if (v2) acc   <= acc + 32'(dot_r);

            case (state)
                S_IDLE: if (start) begin
                    layer <= '0;
                    busy  <= 1'b1;
                    state <= S_LAYER;
                end

                S_LAYER: begin
                    w_ptr    <= c_w_base[W_AW-1:0];
                    neuron   <= '0;
                    out_word <= '0;
                    state    <= S_NEURON;
                end

                S_NEURON: begin
                    acc   <= '0;
                    widx  <= '0;
                    state <= S_MAC;
                end

                S_MAC: begin
                    w_ptr <= w_ptr + 1'b1;
                    widx  <= widx + 12'd1;
                    if (widx == c_in_words - 12'd1) state <= S_DRAIN;
                end

                S_DRAIN: if (!v1 && !v2) state <= S_BIAS;

                S_BIAS: begin
                    acc   <= acc + $signed(b_q);
                    if (c_last) state <= S_OUT;
                    else        state <= S_MUL;
                end

                S_MUL: begin
                    prod  <= acc * $signed({1'b0, c_mult});
                    state <= S_RND;
                end

                S_RND: begin
                    q     <= (rnd < 0) ? 8'd0 : (rnd > 127) ? 8'd127 : rnd[7:0];
                    state <= S_OUT;
                end

                S_OUT: begin
                    if (c_last) begin
                        logit_valid <= 1'b1;
                        logit_idx   <= neuron[7:0];
                        logit_data  <= acc;
                        if (neuron == '0 || acc > best) begin
                            best      <= acc;
                            class_idx <= neuron[7:0];
                        end
                    end else begin
                        out_word <= word_full ? '0 : out_word_n;
                    end
                    if (neuron == c_out_size - 12'd1) begin
                        state <= S_NEXT;
                    end else begin
                        neuron <= neuron + 12'd1;
                        state  <= S_NEURON;
                    end
                end

                S_NEXT: begin
                    if (c_last || layer == num_layers - 3'd1) begin
                        state <= S_DONE;
                    end else begin
                        layer <= layer + 3'd1;
                        state <= S_LAYER;
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

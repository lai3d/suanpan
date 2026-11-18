// Tang Nano 20K demo: an image classifier with LED output and a UART host
// interface. The network is whichever one the compiler's net_params.vh
// describes; the memories are sized to it.
//
// After power-on (or any press/release of button S1) the wrapper
//   1. writes the layer table into the accelerator,
//   2. copies the demo image from ROM into activation buffer A. The ROM
//      already holds it as input words in the network's own layout,
//   3. starts inference and waits for `done`.
//
// LEDs are active low:
//   led[3:0]  predicted class, binary
//   led[4]    lit while running
//   led[5]    lit when a result is being shown
//
// UART (8N1, through the board's USB bridge). Commands are accepted only
// while idle, one at a time:
//   'P'                         -> 'K'                       (ping)
//   'I' + NET_IN_WORDS*8 bytes  -> class (1 byte) + NET_N_OUT logits
//         input words, each 8 bytes little-endian;   logits are int32,
//         i.e. the image in buffer A's layout        little-endian
// If an image stalls for RX_TIMEOUT cycles the command is dropped, so a lost
// byte cannot leave the receiver out of step for good.
//
// Build with the network's output directory on the include path, e.g.
// -Ibuild/cnn; the memory images named in net_params.vh are read at synthesis
// time, relative to the repo root.
module top #(
    parameter int POR_BITS  = 16,       // power-on reset length (2^POR_BITS cycles)
    parameter int CLKS_PER_BIT = 234,   // 27 MHz / 115200 baud
    parameter int RX_TIMEOUT   = 3375000  // 125 ms at 27 MHz
) (
    input  logic       clk,           // 27 MHz
    input  logic       btn,           // S1
    input  logic       uart_rx,
    output logic       uart_tx,
    output logic [5:0] led
);
    `include "net_params.vh"
    localparam int N_LAYERS = NET_LAYERS;
    localparam int IN_WORDS = NET_IN_WORDS;
    localparam int N_OUT    = NET_N_OUT;
    localparam int A_AW     = NET_A_AW;

    // ------------------------------------------------- reset and re-run
    logic [POR_BITS:0] por = '0;
    logic              rst_n;
    assign rst_n = por[POR_BITS];
    always_ff @(posedge clk) if (!rst_n) por <= por + 1'b1;

    // Any change of the (debounced) button re-runs the demo, so the wrapper
    // does not depend on the button's polarity.
    logic [2:0]  btn_sync;
    logic [18:0] btn_cnt;
    logic        btn_state, rerun;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            btn_sync  <= '0;
            btn_cnt   <= '0;
            btn_state <= 1'b0;
            rerun     <= 1'b0;
        end else begin
            btn_sync <= {btn_sync[1:0], btn};
            rerun    <= 1'b0;
            if (btn_sync[2] == btn_state) begin
                btn_cnt <= '0;
            end else begin
                btn_cnt <= btn_cnt + 1'b1;
                if (&btn_cnt) begin          // stable for ~19 ms at 27 MHz
                    btn_state <= btn_sync[2];
                    rerun     <= 1'b1;
                end
            end
        end
    end

    // ------------------------------------------------------------ UART
    logic [7:0] rx_data, tx_data;
    logic       rx_valid, tx_valid, tx_ready;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk, .rst_n, .rx(uart_rx), .data(rx_data), .valid(rx_valid));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk, .rst_n, .data(tx_data), .valid(tx_valid), .ready(tx_ready), .tx(uart_tx));

    // ------------------------------------------------------------ ROMs
    logic [175:0] cfg_rom [0:7];
    logic [63:0]  img_rom [0:IN_WORDS-1];
    initial begin
        $readmemh(NET_CFG_FILE, cfg_rom);
        $readmemh(NET_DEMO_IMAGE_FILE, img_rom);
    end

    // ----------------------------------------------------- accelerator
    logic         cfg_we, in_we, start, busy, done, logit_valid;
    logic [2:0]   cfg_addr;
    logic [175:0] cfg_data;
    logic [A_AW-1:0] in_addr;
    logic [63:0]  in_data;
    logic [7:0]   class_idx, logit_idx;
    logic signed [31:0] logit_data;

    nn_accel #(
        .W_AW(NET_W_AW), .A_AW(A_AW), .B_AW(NET_P_AW),
        .W_DEPTH(NET_W_WORDS), .BUFA_DEPTH(NET_A_WORDS), .BUFB_DEPTH(NET_B_WORDS),
        .B_DEPTH(NET_P_WORDS),
        .W_FILE(NET_WEIGHTS_FILE),
        .B_FILE(NET_PARAMS_FILE)
    ) u_accel (
        .clk, .rst_n, .cfg_we, .cfg_addr, .cfg_data, .in_we, .in_addr, .in_data,
        .start, .num_layers(4'(N_LAYERS)), .busy, .done, .class_idx,
        .logit_valid, .logit_idx, .logit_data
    );

    // logits of the latest run, for the UART reply
    logic [31:0] logit_mem [0:15];
    always_ff @(posedge clk) if (logit_valid) logit_mem[logit_idx[3:0]] <= logit_data;

    // ------------------------------------------------------- sequencer
    typedef enum logic [2:0] {
        T_CFG, T_IMG, T_START, T_WAIT, T_REPLY, T_IDLE, T_RX, T_PONG
    } tstate_t;
    tstate_t     tstate;
    logic [11:0] cnt;          // layer index, then input word index
    logic [63:0] pixel;        // synchronous ROM read
    logic        pixel_v;
    logic [A_AW-1:0] pixel_addr;
    logic [3:0]  result;
    logic        have_result;
    logic        from_uart;    // the run in progress was requested over UART

    // UART image reception: bytes are shifted into a word, LSB first
    logic [63:0]     wsh;
    logic [2:0]      bcnt;
    logic            uw_we;
    logic [A_AW-1:0] uw_addr;
    logic [63:0]     uw_data;
    logic [$clog2(RX_TIMEOUT+1)-1:0] tcnt;

    // UART reply: byte 0 is the class, then N_OUT little-endian int32
    localparam int REPLY_LEN = 1 + 4 * N_OUT;
    logic [6:0]  rcnt;
    logic [6:0]  ridx;
    logic [31:0] rword;
    logic [7:0]  rbyte;
    assign ridx  = rcnt - 7'd1;
    assign rword = logit_mem[ridx[5:2]];
    always_comb begin
        case (ridx[1:0])
            2'd0:    rbyte = rword[7:0];
            2'd1:    rbyte = rword[15:8];
            2'd2:    rbyte = rword[23:16];
            default: rbyte = rword[31:24];
        endcase
        if (rcnt == '0) rbyte = {4'd0, result};
    end

    always_ff @(posedge clk) pixel <= img_rom[cnt[$clog2(IN_WORDS)-1:0]];

    assign cfg_data = cfg_rom[cnt[2:0]];
    assign cfg_addr = cnt[2:0];
    assign cfg_we   = (tstate == T_CFG);
    assign in_we    = pixel_v | uw_we;
    assign in_addr  = uw_we ? uw_addr : pixel_addr;
    assign in_data  = uw_we ? uw_data : pixel;
    assign start    = (tstate == T_START);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tstate      <= T_CFG;
            cnt         <= '0;
            pixel_v     <= 1'b0;
            pixel_addr  <= '0;
            result      <= '0;
            have_result <= 1'b0;
            from_uart   <= 1'b0;
            wsh         <= '0;
            bcnt        <= '0;
            uw_we       <= 1'b0;
            uw_addr     <= '0;
            uw_data     <= '0;
            tcnt        <= '0;
            rcnt        <= '0;
            tx_valid    <= 1'b0;
            tx_data     <= '0;
        end else begin
            pixel_v    <= (tstate == T_IMG);
            pixel_addr <= cnt[A_AW-1:0];
            uw_we      <= 1'b0;
            tx_valid   <= 1'b0;

            case (tstate)
                T_CFG: begin
                    if (cnt == 12'(N_LAYERS - 1)) begin
                        cnt    <= '0;
                        tstate <= T_IMG;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                T_IMG: begin
                    if (cnt == 12'(IN_WORDS - 1)) begin
                        cnt    <= '0;
                        tstate <= T_START;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                // the last input word (from ROM or UART) is written during
                // T_START, before the accelerator reads it
                T_START: tstate <= T_WAIT;

                T_WAIT: if (done) begin
                    result      <= class_idx[3:0];
                    have_result <= 1'b1;
                    rcnt        <= '0;
                    if (from_uart) tstate <= T_REPLY;
                    else           tstate <= T_IDLE;
                end

                T_REPLY: if (tx_ready && !tx_valid) begin
                    tx_data  <= rbyte;
                    tx_valid <= 1'b1;
                    rcnt     <= rcnt + 7'd1;
                    if (rcnt == 7'(REPLY_LEN - 1)) tstate <= T_IDLE;
                end

                T_IDLE: begin
                    if (rerun) begin
                        have_result <= 1'b0;
                        from_uart   <= 1'b0;
                        cnt         <= '0;
                        tstate      <= T_CFG;
                    end else if (rx_valid && rx_data == "P") begin
                        tstate <= T_PONG;
                    end else if (rx_valid && rx_data == "I") begin
                        have_result <= 1'b0;
                        from_uart   <= 1'b1;
                        cnt         <= '0;
                        bcnt        <= '0;
                        tcnt        <= '0;
                        tstate      <= T_RX;
                    end
                end

                T_RX: begin
                    tcnt <= tcnt + 1'b1;
                    if (rx_valid) begin
                        tcnt <= '0;
                        wsh  <= {rx_data, wsh[63:8]};
                        bcnt <= bcnt + 3'd1;
                        if (bcnt == 3'd7) begin
                            uw_we   <= 1'b1;
                            uw_addr <= cnt[A_AW-1:0];
                            uw_data <= {rx_data, wsh[63:8]};
                            cnt     <= cnt + 1'b1;
                            if (cnt == 12'(IN_WORDS - 1)) tstate <= T_START;
                        end
                    end else if (tcnt == ($bits(tcnt))'(RX_TIMEOUT)) begin
                        tstate <= T_IDLE;                // stalled: drop the command
                    end
                end

                T_PONG: if (tx_ready && !tx_valid) begin
                    tx_data  <= "K";
                    tx_valid <= 1'b1;
                    tstate   <= T_IDLE;
                end

                default: tstate <= T_CFG;
            endcase
        end
    end

    assign led = ~{have_result, busy, result};
endmodule

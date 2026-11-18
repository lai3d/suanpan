// Tang Nano 20K demo: classify one MNIST digit held in ROM and show the
// result on the LEDs.
//
// After power-on (or any press/release of button S1) the wrapper
//   1. writes the layer table into the accelerator,
//   2. copies the 28x28 demo image from ROM into activation buffer A
//      (one pixel per word: channel 0 = pixel, channels 1..7 = 0),
//   3. starts inference and waits for `done`.
//
// LEDs are active low:
//   led[3:0]  predicted class, binary
//   led[4]    lit while running
//   led[5]    lit when a result is being shown
//
// The network is the default CNN; memory images come from `make model`.
module top #(
    // Memory images, read at synthesis time (paths relative to the repo root).
    parameter     W_FILE    = "build/cnn/weights.hex",
    parameter     B_FILE    = "build/cnn/bias.hex",
    parameter     CFG_FILE  = "build/cnn/cfg.hex",
    parameter     IMG_FILE  = "build/cnn/demo_image.hex",
    parameter int N_LAYERS  = 3,
    parameter int IN_WORDS  = 784,
    parameter int POR_BITS  = 16      // power-on reset length (2^POR_BITS cycles)
) (
    input  logic       clk,           // 27 MHz
    input  logic       btn,           // S1
    output logic [5:0] led
);
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

    // ------------------------------------------------------------ ROMs
    logic [143:0] cfg_rom [0:3];
    logic [7:0]   img_rom [0:1023];
    initial begin
        $readmemh(CFG_FILE, cfg_rom);
        $readmemh(IMG_FILE, img_rom);
    end

    // ----------------------------------------------------- accelerator
    logic         cfg_we, in_we, start, busy, done;
    logic [1:0]   cfg_addr;
    logic [143:0] cfg_data;
    logic [9:0]   in_addr;
    logic [63:0]  in_data;
    logic [7:0]   class_idx;

    /* verilator lint_off PINCONNECTEMPTY */
    nn_accel #(
        .W_AW(10), .A_AW(10),
        .W_FILE(W_FILE),
        .B_FILE(B_FILE)
    ) u_accel (
        .clk, .rst_n, .cfg_we, .cfg_addr, .cfg_data, .in_we, .in_addr, .in_data,
        .start, .num_layers(3'(N_LAYERS)), .busy, .done, .class_idx,
        .logit_valid(), .logit_idx(), .logit_data()
    );
    /* verilator lint_on PINCONNECTEMPTY */

    // ------------------------------------------------------- sequencer
    typedef enum logic [2:0] {T_CFG, T_IMG, T_START, T_WAIT, T_SHOW} tstate_t;
    tstate_t     tstate;
    logic [10:0] cnt;          // layer index, then pixel index
    logic [7:0]  pixel;        // synchronous ROM read
    logic        pixel_v;
    logic [9:0]  pixel_addr;
    logic [3:0]  result;
    logic        have_result;

    always_ff @(posedge clk) pixel <= img_rom[cnt[9:0]];

    assign cfg_data = cfg_rom[cnt[1:0]];
    assign cfg_addr = cnt[1:0];
    assign cfg_we   = (tstate == T_CFG);
    assign in_we    = pixel_v;
    assign in_addr  = pixel_addr;
    assign in_data  = {56'd0, pixel};
    assign start    = (tstate == T_START);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tstate      <= T_CFG;
            cnt         <= '0;
            pixel_v     <= 1'b0;
            pixel_addr  <= '0;
            result      <= '0;
            have_result <= 1'b0;
        end else begin
            pixel_v    <= (tstate == T_IMG);
            pixel_addr <= cnt[9:0];

            case (tstate)
                T_CFG: begin
                    if (cnt == 11'(N_LAYERS - 1)) begin
                        cnt    <= '0;
                        tstate <= T_IMG;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                T_IMG: begin
                    if (cnt == 11'(IN_WORDS - 1)) begin
                        cnt    <= '0;
                        tstate <= T_START;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                // pixel_v trails T_IMG by one cycle, so the last pixel is
                // written during T_START, before the accelerator reads it
                T_START: tstate <= T_WAIT;

                T_WAIT: if (done) begin
                    result      <= class_idx[3:0];
                    have_result <= 1'b1;
                    tstate      <= T_SHOW;
                end

                T_SHOW: if (rerun) begin
                    have_result <= 1'b0;
                    cnt         <= '0;
                    tstate      <= T_CFG;
                end

                default: tstate <= T_CFG;
            endcase
        end
    end

    assign led = ~{have_result, busy, result};
endmodule

// Self-checking testbench: runs N MNIST test images through the accelerator
// and compares every logit and the predicted class against the bit-exact
// integer golden model exported by model/train.py.
//
//   +N=<count>   number of images to run (default 100)
//   +VCD         dump waves to build/tb.vcd
`timescale 1ns/1ps
module tb_nn_accel;
    localparam int P        = 8;
    localparam int IN_WORDS = 98;      // 784 / 8
    localparam int N_OUT    = 10;
    localparam int MAX_N    = 10000;

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;              // 100 MHz

    logic               in_we = 0;
    logic [6:0]         in_addr = 0;
    logic [63:0]        in_data = 0;
    logic               start = 0;
    logic               cfg_we = 0;
    logic [1:0]         cfg_addr = 0;
    logic [79:0]        cfg_data = 0;
    logic [79:0]        cfg_rom [0:1];
    logic               busy, done, logit_valid;
    logic [7:0]         class_idx, logit_idx;
    logic signed [31:0] logit_data;

    nn_accel #(
        .P(P),
        .W_FILE("build/mem/weights.hex"),
        .B_FILE("build/mem/bias.hex")
    ) dut (
        .clk, .rst_n, .cfg_we, .cfg_addr, .cfg_data, .in_we, .in_addr, .in_data,
        .start, .num_layers(3'd2), .busy, .done, .class_idx,
        .logit_valid, .logit_idx, .logit_data
    );

    logic [63:0] images   [0:MAX_N*IN_WORDS-1];
    logic [7:0]  labels   [0:MAX_N-1];
    logic [7:0]  g_class  [0:MAX_N-1];
    logic [31:0] g_logits [0:MAX_N*N_OUT-1];

    int n, img, errors, correct, logits_seen;
    longint cycles, busy_cycles;

    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (busy) busy_cycles <= busy_cycles + 1;
        if (logit_valid) begin
            logits_seen <= logits_seen + 1;
            if (logit_data !== $signed(g_logits[img*N_OUT + logit_idx])) begin
                errors <= errors + 1;
                if (errors < 10)
                    $display("MISMATCH img %0d logit %0d: rtl %0d  golden %0d", img,
                             logit_idx, logit_data, $signed(g_logits[img*N_OUT + logit_idx]));
            end
        end
    end

    initial begin
        if (!$value$plusargs("N=%d", n)) n = 100;
        if ($test$plusargs("VCD")) begin
            $dumpfile("build/tb.vcd");
            $dumpvars(0, tb_nn_accel);
        end
        $readmemh("build/mem/cfg.hex", cfg_rom);
        $readmemh("build/mem/images.hex", images);
        $readmemh("build/mem/labels.hex", labels);
        $readmemh("build/mem/golden_class.hex", g_class);
        $readmemh("build/mem/golden_logits.hex", g_logits);
        errors = 0; correct = 0; logits_seen = 0; cycles = 0; busy_cycles = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        @(posedge clk);

        // program the layer table
        for (int l = 0; l < 2; l++) begin
            cfg_we   <= 1;
            cfg_addr <= l[1:0];
            cfg_data <= cfg_rom[l];
            @(posedge clk);
        end
        cfg_we <= 0;
        @(posedge clk);

        for (img = 0; img < n; img++) begin
            for (int i = 0; i < IN_WORDS; i++) begin
                in_we   <= 1;
                in_addr <= i[6:0];
                in_data <= images[img*IN_WORDS + i];
                @(posedge clk);
            end
            in_we <= 0;
            start <= 1;
            @(posedge clk);
            start <= 0;
            @(posedge clk);
            while (!done) @(posedge clk);

            if (class_idx !== g_class[img]) begin
                errors++;
                $display("MISMATCH img %0d class: rtl %0d  golden %0d",
                         img, class_idx, g_class[img]);
            end
            if (class_idx === labels[img]) correct++;
            @(posedge clk);
        end

        $display("--------------------------------------------------");
        $display("images run        : %0d", n);
        $display("logits checked    : %0d (expected %0d)", logits_seen, n * N_OUT);
        $display("mismatches        : %0d", errors);
        $display("accuracy vs label : %0d / %0d", correct, n);
        $display("cycles / inference: %0d (compute only)", busy_cycles / n);
        if (errors == 0 && logits_seen == n * N_OUT) $display("RESULT: PASS");
        else                                         $display("RESULT: FAIL");
        $finish;
    end

    initial begin
        #2_000_000_000;   // generous watchdog: 2 s of simulated time
        $display("RESULT: FAIL (timeout)");
        $finish;
    end
endmodule

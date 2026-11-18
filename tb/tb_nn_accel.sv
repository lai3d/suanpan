// Self-checking testbench: runs N MNIST test images through the accelerator
// and compares every logit and the predicted class against the bit-exact
// integer golden model exported by model/train.py. The network (layer table,
// weights, input size) comes entirely from the memory images, so the same
// testbench runs any architecture the exporter produces.
//
//   +MEM=<dir>   directory with the exported images (default build/cnn)
//   +N=<count>   number of images to run (default 100)
//   +VCD         dump waves to build/tb.vcd
`timescale 1ns/1ps
module tb_nn_accel;
    localparam int P          = 8;
    localparam int A_AW       = 10;
    localparam int MAX_LAYERS = 8;
    localparam int MAX_N      = 10000;
    localparam int MAX_IN     = 1 << A_AW;   // input words per image
    localparam int MAX_OUT    = 16;

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;              // 100 MHz

    logic               cfg_we = 0;
    logic [2:0]         cfg_addr = 0;
    logic [175:0]       cfg_data = 0;
    logic               in_we = 0;
    logic [A_AW-1:0]    in_addr = 0;
    logic [63:0]        in_data = 0;
    logic               start = 0;
    logic [3:0]         num_layers = 0;
    logic               busy, done, logit_valid;
    logic [7:0]         class_idx, logit_idx;
    logic signed [31:0] logit_data;

    // GATE: a synthesized netlist has no parameters and no named memories;
    // its weight and bias ROMs are baked in at synthesis time.
`ifdef GATE
    nn_accel dut (
`else
    nn_accel #(.P(P), .A_AW(A_AW), .MAX_LAYERS(MAX_LAYERS)) dut (
`endif
        .clk, .rst_n, .cfg_we, .cfg_addr, .cfg_data, .in_we, .in_addr, .in_data,
        .start, .num_layers, .busy, .done, .class_idx,
        .logit_valid, .logit_idx, .logit_data
    );

    logic [175:0] cfg_rom  [0:MAX_LAYERS-1];
    logic [31:0]  meta     [0:2];            // in_words, n_layers, n_out
    logic [63:0]  images   [0:MAX_N*MAX_IN-1];
    logic [7:0]   labels   [0:MAX_N-1];
    logic [7:0]   g_class  [0:MAX_N-1];
    logic [31:0]  g_logits [0:MAX_N*MAX_OUT-1];

    string mem;
    int n, img, in_words, n_layers, n_out, errors, correct, logits_seen;
    longint busy_cycles;

    always @(posedge clk) begin
        if (busy) busy_cycles <= busy_cycles + 1;
        if (logit_valid) begin
            logits_seen <= logits_seen + 1;
            if (logit_data !== $signed(g_logits[img*n_out + int'(logit_idx)])) begin
                errors <= errors + 1;
                if (errors < 10)
                    $display("MISMATCH img %0d logit %0d: rtl %0d  golden %0d", img, logit_idx,
                             logit_data, $signed(g_logits[img*n_out + int'(logit_idx)]));
            end
        end
    end

    initial begin
        if (!$value$plusargs("MEM=%s", mem)) mem = "build/cnn";
        if (!$value$plusargs("N=%d", n)) n = 100;
        if ($test$plusargs("VCD")) begin
            $dumpfile("build/tb.vcd");
            $dumpvars(0, tb_nn_accel);
        end
        // weight and parameter memories are ROM images on the chip
`ifndef GATE
        $readmemh({mem, "/weights.hex"}, dut.u_wmem.mem);
        $readmemh({mem, "/params.hex"}, dut.u_bmem.mem);
`endif
        $readmemh({mem, "/cfg.hex"}, cfg_rom);
        $readmemh({mem, "/meta.hex"}, meta);
        $readmemh({mem, "/images.hex"}, images);
        $readmemh({mem, "/labels.hex"}, labels);
        $readmemh({mem, "/golden_class.hex"}, g_class);
        $readmemh({mem, "/golden_logits.hex"}, g_logits);
        in_words = int'(meta[0]);
        n_layers = int'(meta[1]);
        n_out    = int'(meta[2]);
        errors = 0; correct = 0; logits_seen = 0; busy_cycles = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        @(posedge clk);

        // program the layer table
        num_layers <= n_layers[3:0];
        for (int l = 0; l < n_layers; l++) begin
            cfg_we   <= 1;
            cfg_addr <= l[2:0];
            cfg_data <= cfg_rom[l];
            @(posedge clk);
        end
        cfg_we <= 0;
        @(posedge clk);

        for (img = 0; img < n; img++) begin
            for (int i = 0; i < in_words; i++) begin
                in_we   <= 1;
                in_addr <= i[A_AW-1:0];
                in_data <= images[img*in_words + i];
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
        $display("network           : %0s (%0d layers)", mem, n_layers);
        $display("images run        : %0d", n);
        $display("logits checked    : %0d (expected %0d)", logits_seen, n * n_out);
        $display("mismatches        : %0d", errors);
        $display("accuracy vs label : %0d / %0d", correct, n);
        $display("cycles / inference: %0d (compute only)", busy_cycles / longint'(n));
        if (errors == 0 && logits_seen == n * n_out) $display("RESULT: PASS");
        else                                         $display("RESULT: FAIL");
        $finish;
    end
endmodule

// Testbench for the Tang Nano 20K wrapper. It plays the host: it watches the
// LEDs and talks to the UART.
//
//   power-on        LEDs show the golden class of the demo image
//   ping            'P' -> 'K'
//   images over UART (N of them): class and every logit match the golden model
//   stalled image   a truncated command is dropped and the link recovers
//   button          re-runs the demo image
//
// Compile with the network's directory on the include path (-Ibuild/cnn):
// net_params.vh says how big the images are and where the vectors live.
//
//   +N=<count>   images to send over UART (default 3; 0 in GATE mode)
//   +CPB=<n>     clocks per UART bit; must match the design under test
//
// GATE: the synthesized netlist has no parameters, so it runs with the real
// power-on delay, baud rate and timeout. The stalled-image test is skipped
// there because the real timeout is millions of cycles.
`timescale 1ns/1ps
module tb_fpga_top;
    `include "net_params.vh"
    localparam int IN_WORDS = NET_IN_WORDS;
    localparam int N_OUT    = NET_N_OUT;
    localparam int MAX_N    = 16;

    logic clk = 0, btn = 0, uart_rx = 1;
    logic uart_tx;
    logic [5:0] led;
    always #18.5 clk = ~clk;   // 27 MHz

`ifdef GATE
    localparam int CPB_DEFAULT = 234;
    top dut (.clk(clk), .btn(btn), .uart_rx(uart_rx), .uart_tx(uart_tx), .led(led));
`else
    localparam int CPB_DEFAULT = 8;
    localparam int TIMEOUT     = 4000;
    top #(.POR_BITS(4), .CLKS_PER_BIT(CPB_DEFAULT), .RX_TIMEOUT(TIMEOUT)) dut (
        .clk, .btn, .uart_rx, .uart_tx, .led);
`endif

    logic [63:0] images   [0:MAX_N*IN_WORDS-1];
    logic [7:0]  g_class  [0:MAX_N-1];
    logic [31:0] g_logits [0:MAX_N*N_OUT-1];
    int cpb, n, errors = 0;

    // ------------------------------------------------------------ host UART
    task automatic send_byte(input logic [7:0] b);
        uart_rx = 0;                      repeat (cpb) @(posedge clk);
        for (int i = 0; i < 8; i++) begin
            uart_rx = b[i];               repeat (cpb) @(posedge clk);
        end
        uart_rx = 1;                      repeat (cpb) @(posedge clk);
    endtask

    // waits up to `limit` clocks for a start bit; ok = 0 on timeout
    task automatic recv_byte(output logic [7:0] b, output logic ok, input int limit);
        int guard = 0;
        ok = 0; b = 8'hxx;
        while (uart_tx !== 1'b0 && guard < limit) begin @(posedge clk); guard++; end
        if (uart_tx === 1'b0) begin
            repeat (cpb + cpb / 2) @(posedge clk);        // middle of bit 0
            for (int i = 0; i < 8; i++) begin
                b[i] = uart_tx;           repeat (cpb) @(posedge clk);
            end
            ok = (uart_tx === 1'b1);                      // stop bit
        end
    endtask

    task automatic ping(input string what);
        logic [7:0] b; logic ok;
        fork
            send_byte("P");
            recv_byte(b, ok, 40 * cpb + 2000);
        join
        if (!ok || b !== "K") begin
            $display("FAIL (%0s): ping answered %0h ok=%0d", what, b, ok); errors++;
        end else $display("%0s: ping ok", what);
    endtask

    task automatic classify(input int idx);
        logic [7:0] b, cls; logic ok;
        logic [31:0] lg;
        int bad = 0;
        send_byte("I");
        for (int w = 0; w < IN_WORDS; w++)
            for (int k = 0; k < 8; k++) send_byte(images[idx*IN_WORDS + w][k*8 +: 8]);
        recv_byte(cls, ok, 2000000);
        if (!ok) begin $display("FAIL: image %0d: no reply", idx); errors++; return; end
        for (int o = 0; o < N_OUT; o++) begin
            for (int k = 0; k < 4; k++) begin
                recv_byte(b, ok, 40 * cpb + 2000);
                if (!ok) bad++;
                lg[k*8 +: 8] = b;
            end
            if (lg !== g_logits[idx*N_OUT + o]) bad++;
        end
        if (cls !== g_class[idx] || bad != 0 || ~led[3:0] !== g_class[idx][3:0]) begin
            $display("FAIL: image %0d: class %0d (golden %0d), %0d bad logit bytes/values, LEDs %0d",
                     idx, cls, g_class[idx], bad, ~led[3:0]);
            errors++;
        end else $display("image %0d over UART: class %0d, %0d logits match", idx, cls, N_OUT);
    endtask

    task automatic wait_result(input string what);
        int guard = 0;
        while (led[5] !== 1'b0 && guard < 2000000) begin @(posedge clk); guard++; end
        if (led[5] !== 1'b0) begin
            $display("FAIL (%0s): no result after %0d cycles", what, guard);
            errors++;
        end else if (~led[3:0] !== g_class[0][3:0]) begin
            $display("FAIL (%0s): LEDs show %0d, golden %0d", what, ~led[3:0], g_class[0]);
            errors++;
        end else begin
            $display("%0s: LEDs show %0d after %0d cycles (golden %0d)",
                     what, ~led[3:0], guard, g_class[0]);
        end
    endtask

    initial begin
        if (!$value$plusargs("CPB=%d", cpb)) cpb = CPB_DEFAULT;
`ifdef GATE
        if (!$value$plusargs("N=%d", n)) n = 0;
`else
        if (!$value$plusargs("N=%d", n)) n = 3;
`endif
        $readmemh({NET_DIR, "/images.hex"}, images);
        $readmemh({NET_DIR, "/golden_class.hex"}, g_class);
        $readmemh({NET_DIR, "/golden_logits.hex"}, g_logits);

        wait_result("power-on");
        ping("after power-on");

        // images 1..n: image 0 is the one already in ROM
        for (int i = 1; i <= n; i++) classify(i);

`ifndef GATE
        // a command that stalls must be dropped, and the link must recover
        send_byte("I");
        repeat (10) send_byte(8'h5a);
        repeat (TIMEOUT + 100) @(posedge clk);
        ping("after stalled image");
        if (n > 0) classify(1);
`endif

        // button: long enough to pass the debouncer, expect the ROM image again
        btn = 1;
        while (led[5] === 1'b0) @(posedge clk);
        wait_result("after button press");

        if (errors == 0) $display("RESULT: PASS");
        else             $display("RESULT: FAIL");
        $finish;
    end
endmodule

// Testbench for the Tang Nano 20K demo wrapper: after power-on the LEDs must
// show the golden class of the demo image, and a button press must re-run it.
`timescale 1ns/1ps
module tb_fpga_top;
    logic clk = 0, btn = 0;
    logic [5:0] led;
    always #18.5 clk = ~clk;   // 27 MHz

    // GATE: the synthesized netlist has no parameters (and a full-length
    // power-on reset), so it is instantiated as-is.
`ifdef GATE
    top dut (.clk(clk), .btn(btn), .led(led));
`else
    top #(.POR_BITS(4)) dut (.clk, .btn, .led);
`endif

    logic [7:0] g_class [0:0];
    int errors = 0;

    task automatic wait_result(input string what);
        int guard = 0;
        while (led[5] !== 1'b0 && guard < 200000) begin @(posedge clk); guard++; end
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
        $readmemh("build/cnn/golden_class.hex", g_class);
        wait_result("power-on");

        // press the button long enough to pass the debouncer, expect a re-run
        btn = 1;
        while (led[5] === 1'b0) @(posedge clk);   // result cleared
        wait_result("after button press");

        if (errors == 0) $display("RESULT: PASS");
        else             $display("RESULT: FAIL");
        $finish;
    end

    initial begin
        #2_000_000_000;
        $display("RESULT: FAIL (timeout)");
        $finish;
    end
endmodule

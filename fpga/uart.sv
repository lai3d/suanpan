// Minimal 8N1 UART, one start bit, one stop bit, LSB first.
// CLKS_PER_BIT = clock frequency / baud rate.

module uart_rx #(
    parameter int CLKS_PER_BIT = 234
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       rx,
    output logic [7:0] data,
    output logic       valid      // 1-cycle pulse, data is stable with it
);
    localparam int CW = $clog2(CLKS_PER_BIT + 1);
    typedef enum logic [1:0] {R_IDLE, R_START, R_DATA, R_STOP} rstate_t;
    rstate_t       state;
    logic [2:0]    sync;          // metastability filter, idle high
    logic [CW-1:0] cnt;
    logic [2:0]    nbit;
    logic [7:0]    sh;
    logic          rxs;
    assign rxs = sync[2];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= R_IDLE;
            sync  <= 3'b111;
            cnt   <= '0;
            nbit  <= '0;
            sh    <= '0;
            data  <= '0;
            valid <= 1'b0;
        end else begin
            sync  <= {sync[1:0], rx};
            valid <= 1'b0;
            case (state)
                R_IDLE: if (!rxs) begin
                    cnt   <= CW'(CLKS_PER_BIT / 2);      // to the middle of the start bit
                    state <= R_START;
                end
                R_START: if (cnt == '0) begin
                    cnt   <= CW'(CLKS_PER_BIT - 1);
                    nbit  <= '0;
                    if (!rxs) state <= R_DATA;           // still low: a real start bit
                    else      state <= R_IDLE;
                end else cnt <= cnt - 1'b1;
                R_DATA: if (cnt == '0) begin
                    cnt  <= CW'(CLKS_PER_BIT - 1);
                    sh   <= {rxs, sh[7:1]};
                    nbit <= nbit + 3'd1;
                    if (nbit == 3'd7) state <= R_STOP;
                end else cnt <= cnt - 1'b1;
                R_STOP: if (cnt == '0) begin
                    valid <= rxs;                        // drop the byte on a framing error
                    data  <= sh;
                    state <= R_IDLE;
                end else cnt <= cnt - 1'b1;
            endcase
        end
    end
endmodule

module uart_tx #(
    parameter int CLKS_PER_BIT = 234
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic [7:0] data,
    input  logic       valid,     // accepted in a cycle where ready is high
    output logic       ready,
    output logic       tx
);
    localparam int CW = $clog2(CLKS_PER_BIT + 1);
    logic [CW-1:0] cnt;
    logic [3:0]    nbit;          // bits left, including start and stop
    logic [9:0]    sh;

    assign ready = (nbit == '0);
    assign tx    = ready ? 1'b1 : sh[0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt  <= '0;
            nbit <= '0;
            sh   <= '1;
        end else if (ready) begin
            if (valid) begin
                sh   <= {1'b1, data, 1'b0};              // stop, data, start
                nbit <= 4'd10;
                cnt  <= CW'(CLKS_PER_BIT - 1);
            end
        end else if (cnt == '0) begin
            sh   <= {1'b1, sh[9:1]};
            nbit <= nbit - 4'd1;
            cnt  <= CW'(CLKS_PER_BIT - 1);
        end else begin
            cnt <= cnt - 1'b1;
        end
    end
endmodule

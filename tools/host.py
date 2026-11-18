"""Host side of the Tang Nano 20K UART interface.

    python tools/host.py --port /dev/tty.usbserial-XXXX ping
    python tools/host.py --port /dev/tty.usbserial-XXXX run [--n 100]
    python tools/host.py frames [--n 3]         # no board: show what would be sent

`run` sends MNIST test images to the board and compares the returned class
and logits with the golden model exported by `make model`, the same check the
simulation testbench makes.

Protocol (see fpga/tangnano20k/top.sv):
    'P'                      -> 'K'
    'I' + input words (LE)   -> class (1 byte) + N_OUT int32 logits (LE)
"""
import argparse
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_OUT = 10


def read_hex(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(int(line, 16))
            if limit and len(out) >= limit:
                break
    return out


def load_vectors(mem, n):
    """Images as UART frames plus the golden class and logits for each."""
    in_words, _, n_out = read_hex(mem / "meta.hex")
    assert n_out == N_OUT
    words = read_hex(mem / "images.hex", n * in_words)
    classes = read_hex(mem / "golden_class.hex", n)
    logits = read_hex(mem / "golden_logits.hex", n * n_out)
    frames = [b"I" + b"".join(struct.pack("<Q", w) for w in words[i * in_words:(i + 1) * in_words])
              for i in range(n)]
    signed = [v - (1 << 32) if v >> 31 else v for v in logits]
    return frames, classes, [signed[i * n_out:(i + 1) * n_out] for i in range(n)]


def open_port(args):
    import serial                      # pyserial; only needed with a board
    return serial.Serial(args.port, args.baud, timeout=args.timeout)


def cmd_ping(args):
    with open_port(args) as port:
        port.reset_input_buffer()
        port.write(b"P")
        reply = port.read(1)
    print("ping:", "ok" if reply == b"K" else f"FAILED (got {reply!r})")
    return 0 if reply == b"K" else 1


def cmd_run(args):
    frames, classes, logits = load_vectors(args.mem, args.n)
    labels = read_hex(args.mem / "labels.hex", args.n)
    reply_len = 1 + 4 * N_OUT
    bad = correct = 0
    start = time.time()
    with open_port(args) as port:
        port.reset_input_buffer()
        for i, frame in enumerate(frames):
            port.write(frame)
            reply = port.read(reply_len)
            if len(reply) != reply_len:
                print(f"image {i}: timeout, got {len(reply)} of {reply_len} bytes")
                bad += 1
                time.sleep(0.3)        # longer than the board's receive timeout
                port.reset_input_buffer()
                continue
            got = list(struct.unpack(f"<{N_OUT}i", reply[1:]))
            if reply[0] != classes[i] or got != logits[i]:
                bad += 1
                if bad <= 10:
                    print(f"image {i}: MISMATCH class {reply[0]} (golden {classes[i]})")
            correct += reply[0] == labels[i]
    took = time.time() - start
    print(f"images sent       : {len(frames)}")
    print(f"mismatches        : {bad}")
    print(f"accuracy vs label : {correct} / {len(frames)}")
    print(f"time              : {took:.1f} s ({took / len(frames) * 1000:.1f} ms per image)")
    print("RESULT:", "PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


def cmd_frames(args):
    frames, classes, logits = load_vectors(args.mem, args.n)
    for i, frame in enumerate(frames):
        print(f"image {i}: {len(frame)} bytes, starts {frame[:9].hex()}, "
              f"expect class {classes[i]}, logit[0] {logits[i][0]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["ping", "run", "frames"])
    ap.add_argument("--port", help="serial device of the board's USB bridge")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=100, help="number of test images")
    ap.add_argument("--mem", type=Path, default=ROOT / "build" / "cnn")
    args = ap.parse_args()
    if args.command != "frames" and not args.port:
        ap.error("--port is required to talk to a board")
    return {"ping": cmd_ping, "run": cmd_run, "frames": cmd_frames}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

"""Build a small residual network with random weights and random inputs, as
an RTL regression for the shortcut path and the third activation buffer.

    python tests/make_residual_net.py          # writes build/restest/
    make sim MEM=build/restest N=200

It is not trained: the point is that the RTL matches the golden model on a
network where a tensor must be kept while two layers run and then added in.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import test_compiler as tc  # noqa: E402


def main():
    out = ROOT / "build" / "restest"
    out.mkdir(parents=True, exist_ok=True)
    nodes, weights = tc.residual_graph()
    shutil.copy(tc.save(nodes, weights, (8, 6, 6)), out / "model.onnx")
    rng = np.random.default_rng(7)
    np.savez(out / "calib.npz", x=rng.integers(0, 256, (300, 8, 6, 6), dtype=np.uint8))
    np.savez(out / "test.npz", x=rng.integers(0, 256, (200, 8, 6, 6), dtype=np.uint8))
    subprocess.run([sys.executable, str(ROOT / "tools" / "compile.py"), str(out / "model.onnx"),
                    "--out", str(out), "--calib", str(out / "calib.npz"),
                    "--test", str(out / "test.npz")], check=True)


if __name__ == "__main__":
    main()

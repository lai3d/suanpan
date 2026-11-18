"""Train LeNet-5 on MNIST with PyTorch and export it to ONNX.

This is an example of a model that was not built for the accelerator: plain
PyTorch layers, 6 and 16 convolution channels, padding and max pooling.

    uv run --python 3.13 --with torch --with onnx examples/train_lenet5.py
    python tools/compile.py examples/lenet5.onnx --out build/lenet5 --data mnist
    make sim-full MEM=build/lenet5

Inputs are scaled to [0, 1] as (pixel >> 1) / 127, which is exactly what the
accelerator's 7-bit input can represent.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from datasets import load_mnist  # noqa: E402


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    torch.manual_seed(0)
    d = load_mnist()
    def images(a):
        return torch.tensor((a >> 1).astype(np.float32) / 127.0)
    xtr, ytr = images(d["train_x"]), torch.tensor(d["train_y"])
    xte, yte = images(d["test_x"]), torch.tensor(d["test_y"])

    model = LeNet5()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for epoch in range(4):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            nn.functional.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xte).argmax(1) == yte).float().mean().item() * 100
        print(f"epoch {epoch + 1}  test accuracy {acc:.2f}%")

    out = ROOT / "examples" / "lenet5.onnx"
    torch.onnx.export(model, xte[:1], str(out), input_names=["input"],
                      output_names=["logits"], opset_version=13, dynamo=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Train a small CNN on CIFAR-10 with PyTorch and export it to ONNX.

    uv run --python 3.13 --with torch --with onnx examples/train_cifar10.py
    python tools/compile.py examples/cifar10.onnx --out build/cifar10 --data cifar10
    make sim-full MEM=build/cifar10 N_FULL=1000

The network is sized for the accelerator: a single chain, at most 8 layers,
ReLU after every hidden layer, max pooling, and feature maps of at most 2048
memory words. Batch normalisation is used for training and folded into the
convolutions by the compiler.

Inputs are scaled to [0, 1] as (pixel >> 1) / 127, which is exactly what the
accelerator's 7-bit input can represent. There is no mean / std normalisation.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from datasets import load_cifar10  # noqa: E402

EPOCHS = 30
BATCH = 128


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                    nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def augment(x):
    """Random crop (4 pixels of zero padding) and horizontal flip."""
    n = len(x)
    padded = nn.functional.pad(x, (4, 4, 4, 4))
    dx, dy = torch.randint(0, 9, (n,)), torch.randint(0, 9, (n,))
    idx = torch.arange(32)
    rows = (dy[:, None] + idx[None, :])[:, :, None].expand(n, 32, 32)
    cols = (dx[:, None] + idx[None, :])[:, None, :].expand(n, 32, 32)
    flip = torch.rand(n) < 0.5
    cols = torch.where(flip[:, None, None], cols.flip(2), cols)
    batch = torch.arange(n)[:, None, None].expand(n, 32, 32)
    return padded[batch, :, rows, cols].permute(0, 3, 1, 2)


def main():
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = load_cifar10()
    def images(a):
        return torch.tensor((a >> 1).astype(np.float32) / 127.0)
    xtr, ytr = images(d["train_x"]), torch.tensor(d["train_y"])
    xte, yte = images(d["test_x"]).to(dev), torch.tensor(d["test_y"]).to(dev)

    model = Net().to(dev)
    steps = EPOCHS * (len(xtr) // BATCH + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-3, total_steps=steps)
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = augment(xtr[idx]).to(dev), ytr[idx].to(dev)
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb, label_smoothing=0.05).backward()
            opt.step()
            sched.step()
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(xte).argmax(1) == yte).float().mean().item() * 100
            print(f"epoch {epoch + 1:2d}  test accuracy {acc:.2f}%  ({time.time() - start:.0f} s, {dev})",
                  flush=True)

    out = ROOT / "examples" / "cifar10.onnx"
    model = model.to("cpu").eval()
    torch.onnx.export(model, xte[:1].cpu(), str(out), input_names=["input"],
                      output_names=["logits"], opset_version=13, dynamo=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

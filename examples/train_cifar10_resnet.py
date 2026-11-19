"""Train a small residual network on CIFAR-10 with PyTorch and export it.

    uv run --python 3.13 --with torch --with onnx examples/train_cifar10_resnet.py
    python tools/compile.py examples/cifar10_resnet.onnx --out build/resnet --data cifar10
    make sim-full MEM=build/resnet N_FULL=200

Three residual blocks at 16, 32 and 64 channels, with a strided convolution
between them. The shortcuts are identity shortcuts: each block adds its own
(already activated) input to the output of its second convolution. Down-
sampling is done by the plain convolutions between blocks, because the
accelerator stores only non-negative activations and so cannot hold the
output of a projection shortcut, which has no ReLU.

Inputs are scaled to [0, 1] as (pixel >> 1) / 127. There is no mean / std
normalisation, and no global average pooling.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "examples"))
from datasets import load_cifar10  # noqa: E402
from train_cifar10 import augment  # noqa: E402

EPOCHS = 40
BATCH = 128


def conv_bn(cin, cout, stride=1):
    return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.BatchNorm2d(cout)]


class Block(nn.Module):
    """y = relu(x + bn(conv(relu(bn(conv(x))))))"""
    def __init__(self, c):
        super().__init__()
        self.a = nn.Sequential(*conv_bn(c, c), nn.ReLU())
        self.b = nn.Sequential(*conv_bn(c, c))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.b(self.a(x)) + x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *conv_bn(3, 16), nn.ReLU(), Block(16),
            *conv_bn(16, 32, 2), nn.ReLU(), Block(32),
            *conv_bn(32, 64, 2), nn.ReLU(), Block(64),
            nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


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
                acc = sum((model(xte[i:i + 1000]).argmax(1) == yte[i:i + 1000]).sum().item()
                          for i in range(0, len(xte), 1000)) / len(xte) * 100
            print(f"epoch {epoch + 1:2d}  test accuracy {acc:.2f}%  ({time.time() - start:.0f} s, {dev})",
                  flush=True)

    out = ROOT / "examples" / "cifar10_resnet.onnx"
    model = model.to("cpu").eval()
    torch.onnx.export(model, xte[:1].cpu(), str(out), input_names=["input"],
                      output_names=["logits"], opset_version=13, dynamo=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Datasets used by the examples. Each is downloaded on first use into data/.

Images are returned as uint8 in (N, C, H, W) order, labels as int64.
"""
import gzip
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"


def load_mnist():
    """Returns train_x (60000, 1, 28, 28), train_y, test_x, test_y."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, name in MNIST_FILES.items():
        path = DATA / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(MNIST_URL + name, path)
        raw = gzip.open(path, "rb").read()
        if key.endswith("_x"):
            _, n, r, c = struct.unpack(">IIII", raw[:16])
            out[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(n, 1, r, c)
        else:
            out[key] = np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    return out


def load_cifar10():
    """Returns train_x (50000, 3, 32, 32), train_y, test_x, test_y."""
    root = DATA / "cifar10"
    batches = root / "cifar-10-batches-bin"
    if not (batches / "test_batch.bin").exists():
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "cifar-10-binary.tar.gz"
        if not archive.exists():
            print("downloading cifar-10-binary.tar.gz (163 MB)")
            # download under another name so an interrupted transfer is never
            # mistaken for the complete archive
            part = archive.with_suffix(".part")
            urllib.request.urlretrieve(CIFAR10_URL, part)
            part.rename(archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")      # plain files only
        archive.unlink()

    def read(name):
        # each record: 1 label byte + 3072 pixel bytes (R, G, B planes)
        raw = np.frombuffer((batches / name).read_bytes(), np.uint8).reshape(-1, 3073)
        return raw[:, 1:].reshape(-1, 3, 32, 32), raw[:, 0].astype(np.int64)

    train = [read(f"data_batch_{i}.bin") for i in range(1, 6)]
    test_x, test_y = read("test_batch.bin")
    return dict(train_x=np.concatenate([t[0] for t in train]),
                train_y=np.concatenate([t[1] for t in train]),
                test_x=test_x, test_y=test_y)


LOADERS = {"mnist": load_mnist, "cifar10": load_cifar10}


def write_subset(name, out_dir, n_calib=128, n_test=32):
    """Save a few images as the .npz files tools/compile.py reads, so a
    network can be compiled and simulated without the full dataset (CI)."""
    d = LOADERS[name]()
    out_dir = Path(out_dir)
    np.savez_compressed(out_dir / f"{name}_calib.npz", x=d["train_x"][:n_calib])
    np.savez_compressed(out_dir / f"{name}_test.npz", x=d["test_x"][:n_test],
                        y=d["test_y"][:n_test])


if __name__ == "__main__":
    import sys
    write_subset(sys.argv[1], sys.argv[2])

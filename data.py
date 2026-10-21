"""CIFAR-10 / CIFAR-100 kept on the GPU, with augmentation done on the GPU.

The whole training set is stored once on the GPU as uint8 (about 150 MB for CIFAR-10).
This replaces the usual CPU DataLoader, which is the bottleneck on an A100 and would make
our timings depend on how busy the node's CPUs are. Every batch is cut from that tensor.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


class GPUData:
    """Images (N x 3 x 32 x 32, uint8) and labels (N) living on one device."""

    def __init__(self, x, y, name, device):
        self.x = x.to(device)
        self.y = y.to(device)
        self.mean = torch.tensor(MEAN[name], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD[name], device=device).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.y)

    def batch(self, idx, augment=False, gen=None):
        """Return normalised images and labels for the given indices."""
        x = self.x[idx].float().div_(255)
        if augment:
            x = random_crop_flip(x, gen)
        return (x - self.mean) / self.std, self.y[idx]


def random_crop_flip(x, gen):
    """Standard CIFAR augmentation on a whole batch at once:
    pad 4 black pixels on each side, take a random 32x32 crop, flip left-right with p = 0.5."""
    B, dev = x.size(0), x.device
    xp = F.pad(x, (4, 4, 4, 4))                                   # B x 3 x 40 x 40
    i = torch.randint(0, 9, (B,), generator=gen, device=dev)      # crop offsets 0..8
    j = torch.randint(0, 9, (B,), generator=gen, device=dev)
    r = torch.arange(32, device=dev)
    rows = (i[:, None] + r)[:, None, :, None]                     # B x 1 x 32 x 1
    cols = (j[:, None] + r)[:, None, None, :]                     # B x 1 x 1 x 32
    b = torch.arange(B, device=dev)[:, None, None, None]
    c = torch.arange(3, device=dev)[None, :, None, None]
    x = xp[b, c, rows, cols]                                      # B x 3 x 32 x 32
    flip = torch.rand(B, generator=gen, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def cifar_file(root, name):
    """Path of the file written by download_data.py, with a clear error if it is missing."""
    path = os.path.join(os.path.expanduser(root), f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run `python download_data.py {root}` first")
    return path


def load_cifar(root, name="cifar10", device="cuda"):
    """Read <root>/<name>.npz (made by download_data.py) and move it to `device`."""
    d = np.load(cifar_file(root, name))
    out = []
    for split in ("train", "test"):
        x = torch.from_numpy(d[f"x_{split}"]).permute(0, 3, 1, 2).contiguous()   # N x 3 x 32 x 32
        y = torch.from_numpy(d[f"y_{split}"]).long()
        out.append(GPUData(x, y, name, device))
    return out[0], out[1]                                                       # train, test


def fake_cifar(name="cifar10", device="cpu", n_train=2000, n_test=500, seed=0):
    """Random images with CIFAR's shape. Only for checking that the code runs."""
    g = torch.Generator().manual_seed(seed)
    K = NUM_CLASSES[name]
    make = lambda n: GPUData(torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g),
                             torch.randint(0, K, (n,), generator=g), name, device)
    return make(n_train), make(n_test)

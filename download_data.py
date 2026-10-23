"""Download CIFAR-10 and CIFAR-100 once and save each as one file. Run on the Prajna login node:

    python download_data.py              # -> ~/datadiet_data/cifar10.npz and cifar100.npz
    python download_data.py ./data       # another folder (e.g. on Kaggle or Colab)

Source: the Hugging Face copy of the official datasets (uoft-cs/cifar10, uoft-cs/cifar100), about 270 MB,
which downloads in seconds; the original server at cs.toronto.edu is often extremely slow.
Everyone in the team should use this script, so that image number i is the same picture for everybody
(score files store one score per image number).

Each .npz file holds: x_train (50000, 32, 32, 3) uint8, y_train (50000,), x_test (10000, 32, 32, 3),
y_test (10000,) and classes (the class names). GPU jobs read these files and never need the internet.
"""
import io, json, os, sys, urllib.request
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

URL = "https://huggingface.co/datasets/uoft-cs/{repo}/resolve/main/{folder}/{split}-00000-of-00001.parquet"
SOURCES = {"cifar10": dict(repo="cifar10", folder="plain_text", label="label"),
           "cifar100": dict(repo="cifar100", folder="cifar100", label="fine_label")}


def fetch(url, dest):
    """Download `url` to `dest`, showing progress; a finished file is not downloaded again."""
    if os.path.exists(dest):
        return dest
    with urllib.request.urlopen(url, timeout=60) as r, open(dest + ".part", "wb") as fh:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            print(f"\r  downloading {os.path.basename(dest)}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB",
                  end="", flush=True)
    print()
    os.replace(dest + ".part", dest)
    return dest


def read_split(path, label_col):
    """Decode the PNG pictures of one parquet file into a uint8 array, with labels and class names."""
    table = pq.read_table(path)
    x = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
                  for d in table.column("img").to_pylist()])
    y = np.asarray(table.column(label_col).to_pylist(), dtype=np.int64)
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"][label_col]["names"]
    return x, y, names


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datadiet_data")
    os.makedirs(root, exist_ok=True)
    for name, s in SOURCES.items():
        out = os.path.join(root, f"{name}.npz")
        if os.path.exists(out):
            print(f"{name:8s} already downloaded: {out}")
            continue
        parts = {}
        for split in ("train", "test"):
            url = URL.format(repo=s["repo"], folder=s["folder"], split=split)
            parts[split] = read_split(fetch(url, os.path.join(root, f"{name}_{split}.parquet")), s["label"])
        (x_train, y_train, classes), (x_test, y_test, _) = parts["train"], parts["test"]
        assert x_train.shape == (50000, 32, 32, 3) and x_test.shape == (10000, 32, 32, 3)
        assert len(classes) == y_train.max() + 1 and (np.bincount(y_train) == 50000 // len(classes)).all()
        np.savez(out + ".part.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                 classes=np.array(classes))
        os.replace(out + ".part.npz", out)
        for split in ("train", "test"):
            os.remove(os.path.join(root, f"{name}_{split}.parquet"))
        print(f"{name:8s} train {len(y_train)} / test {len(y_test)} images, {len(classes)} classes  ->  {out}")


if __name__ == "__main__":
    main()

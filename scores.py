"""Difficulty scores for every training image (guide Ch. 7). Larger score = harder.

  EL2N       = || softmax(f(x)) - onehot(y) ||_2 on the plain (un-augmented) image, model in eval
               mode, averaged over several independently trained scorers.
  forgetting = number of right -> wrong flips during training; never-learned images get
               (max count + 1), the hardest score of all (Toneva et al.).

One file per scorer run and snapshot epoch:  scores/<dataset>/<model>_seed<s>_e<epoch>.npz
holding `el2n`, `forgetting`, `forget_raw`, `learned`. The final snapshot also writes the
scorer's logits (`..._logits.npy`, float16) for the optional distillation experiment.

A "score set" is named <kind>_<model>_e<epoch>, e.g. el2n_resnet18_e20 or forget_tiny_e5,
and means: that score averaged over all scorer seeds.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from train import autocast


@torch.no_grad()
def el2n_and_logits(model, data, bs=1000):
    """EL2N score and raw logits for every image of `data`, in index order."""
    model.eval()                                   # batch norm uses its stored statistics
    device = data.y.device
    K = None
    el2n, logits = [], []
    for s in range(0, len(data), bs):
        x, y = data.batch(torch.arange(s, min(s + bs, len(data)), device=device))   # no augmentation
        with autocast(device):
            z = model(x.contiguous(memory_format=torch.channels_last))
        z = z.float()
        K = z.size(1)
        err = F.softmax(z, dim=1) - F.one_hot(y, K).float()      # the error vector p - y
        el2n.append(err.norm(dim=1).cpu())                        # its length
        logits.append(z.half().cpu())
    return torch.cat(el2n).numpy(), torch.cat(logits).numpy()


def forgetting_scores(forgets, learned):
    """Turn raw counts into scores: never-learned images get max + 1."""
    f = forgets.float().cpu().numpy()
    learned = learned.cpu().numpy()
    f[~learned] = (f.max() + 1) if len(f) else 0
    return f


def score_path(score_dir, model, seed, epoch):
    return os.path.join(score_dir, f"{model}_seed{seed}_e{epoch}.npz")


def logits_path(score_dir, model, seed, epoch):
    return os.path.join(score_dir, f"{model}_seed{seed}_e{epoch}_logits.npy")


def parse_score_set(name):
    """'el2n_resnet18_e20' -> ('el2n', 'resnet18', 20)"""
    kind, model, e = name.split("_")
    assert kind in ("el2n", "forget") and e.startswith("e"), f"bad score set name {name!r}"
    return kind, model, int(e[1:])


def load_score_set(score_dir, name, seeds):
    """Average the score over all scorer seeds. Raises a clear error if files are missing."""
    kind, model, epoch = parse_score_set(name)
    key = "el2n" if kind == "el2n" else "forgetting"
    paths = [score_path(score_dir, model, s, epoch) for s in seeds]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"score set {name} needs {missing}. Run the 'scores' experiment first.")
    return np.mean([np.load(p)[key] for p in paths], axis=0)


def load_teacher_logits(score_dir, model, epoch, seeds):
    """Average logits of the scorer runs: an ensemble teacher whose cost is already paid."""
    paths = [logits_path(score_dir, model, s, epoch) for s in seeds]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"teacher logits missing: {missing}. Run the 'scores' experiment first.")
    return np.mean([np.load(p).astype(np.float32) for p in paths], axis=0)

"""Check the code against the worked examples of the guide (and a few invariants).

    python tests/test_guide_examples.py

Runs on a CPU in a few seconds. Every check prints OK or stops with an AssertionError.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from data import GPUData, random_crop_flip
from prune import select
from scores import el2n_and_logits, forgetting_scores
from train import update_forgetting, kd_loss


def ok(msg):
    print(f"OK  {msg}")


# --- 7.2 EL2N: clear cat 0.013, ordinary cat 0.430, mislabelled 1.009 --------------------------
class FixedLogits(nn.Module):
    """Returns log(p) for image i, so softmax gives back exactly p."""
    def __init__(self, probs):
        super().__init__()
        self.logits = torch.log(torch.tensor(probs))
    def forward(self, x):
        return self.logits[x[:, 0, 0, 0].round().long()]   # image i carries its index in pixel 0


probs = [(0.989, 0.007, 0.004), (0.659, 0.242, 0.099), (0.659, 0.242, 0.099)]
labels = torch.tensor([0, 0, 1])                            # cat, cat, dog
x = torch.zeros(3, 3, 32, 32, dtype=torch.uint8)
x[:, 0, 0, 0] = torch.arange(3, dtype=torch.uint8)
data = GPUData(x, labels, "cifar10", "cpu")
data.mean.zero_(); data.std.fill_(1 / 255)                  # undo normalisation so pixel 0 == index
el2n, _ = el2n_and_logits(FixedLogits(probs), data)
assert np.allclose(el2n, [0.013, 0.430, 1.009], atol=1e-3), el2n
ok("EL2N worked example: " + ", ".join(f"{v:.4f}" for v in el2n) + "  (guide: 0.013, 0.430, 1.009)")

# --- 7.4 EL2N bound: true-class prob 0.9 -> EL2N in [0.1, 0.141] -------------------------------
p = torch.tensor([[0.9, 0.1, 0.0], [0.9, 0.05, 0.05]])
e = (p - F.one_hot(torch.tensor([0, 0]), 3)).norm(dim=1)
assert 0.1 - 1e-6 <= e.min() and e.max() <= 0.1415
ok("EL2N bound for p_y = 0.9: " + ", ".join(f"{v:.3f}" for v in e.tolist()) + " within [0.1, 0.141]")

# --- 7.6 forgetting: 0,1,1,0,1,1,1,0,0,1 -> 2 ; 1,0,1,0,1,0,1,1 -> 3 ; never learned -> max+1 ----
seqs = [[0, 1, 1, 0, 1, 1, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0, 1, 1, 1, 1], [0] * 10]
prev = torch.zeros(3, dtype=torch.bool); forgets = torch.zeros(3, dtype=torch.int32)
learned = torch.zeros(3, dtype=torch.bool)
for t in range(10):
    update_forgetting(prev, forgets, learned, torch.arange(3), torch.tensor([s[t] == 1 for s in seqs]))
assert forgets.tolist() == [2, 3, 0] and learned.tolist() == [True, True, False]
assert forgetting_scores(forgets, learned).tolist() == [2, 3, 4]
ok("forgetting counts [2, 3, 0]; never-learned image scored max+1 = 4")

# --- 6.2 strategies: A..J, f = 0.4 ---------------------------------------------------------------
names = np.array(list("ABCDEFGHIJ"))
s = np.array([0.02, 0.91, 0.10, 0.45, 0.05, 1.30, 0.33, 0.08, 0.60, 0.15])
hard, easy = set(names[select(10, 0.4, "hard", s)]), set(names[select(10, 0.4, "easy", s)])
off = set(names[select(10, 0.4, "hard", s, offset=0.1)])
assert hard == set("FBID") and easy == set("AEHC") and off == set("BIDG")
assert set(names[select(10, 0.2, "easy", s)]) == set("AE")
ok("pruning worked example: hard {F,B,I,D}, easy {A,E,H,C}, offset {B,I,D,G}, easy@0.2 {A,E}")

# --- 5.6 class-balanced pruning keeps f of every class ------------------------------------------
lab = np.repeat(np.arange(10), 100)
kept = select(1000, 0.3, "hard", np.random.default_rng(0).random(1000), labels=lab, class_balanced=True)
assert (np.bincount(lab[kept]) == 30).all()
ok("class-balanced: exactly 30 of 100 images kept in every class")

# --- random subsets: reproducible, right size, overlap ~ f (Prop. 9.2) ---------------------------
a, b = select(50000, 0.3, "random", seed=0), select(50000, 0.3, "random", seed=0)
c = select(50000, 0.3, "random", seed=1)
assert np.array_equal(a, b) and len(np.unique(a)) == 15000
ov = len(np.intersect1d(a, c)) / len(a)
assert abs(ov - 0.3) < 0.02
ok(f"random subsets reproducible; overlap of two random 30% sets = {ov:.3f} (expected 0.3)")

# --- 9.4 Spearman 0.8 and Kendall 0.6 on five images ---------------------------------------------
r, q = [1, 2, 3, 4, 5], [2, 1, 3, 5, 4]
rho, tau = stats.spearmanr(r, q).statistic, stats.kendalltau(r, q).statistic
assert np.isclose(rho, 0.8) and np.isclose(tau, 0.6)
ok(f"Spearman {rho:.1f}, Kendall {tau:.1f}")

# --- 12.3 / 12.4 confidence interval and Welch t-test --------------------------------------------
A, B = np.array([93.1, 93.5, 92.9]), np.array([92.4, 92.9, 92.6])
half = stats.t.ppf(0.975, 2) * A.std(ddof=1) / np.sqrt(3)
t, pv = stats.ttest_ind(A, B, equal_var=False)
assert np.isclose(half, 0.76, atol=0.01) and np.isclose(t, 2.33, atol=0.01) and 0.07 < pv < 0.09
ok(f"95% CI 93.17 +- {half:.2f}; Welch t = {t:.2f}, p = {pv:.3f} (not significant)")

# --- 10.3 break-even worked examples --------------------------------------------------------------
self_score = 3 * 1 * (10 + 1 / 3) / 60
tiny_score = 3 * 0.1 * (5 + 1 / 3) / 60
assert round(self_score, 3) == 0.517 and round(tiny_score, 3) == 0.027
ok(f"S/F self-scoring {self_score:.3f} (break-even f {1 - self_score:.3f}); tiny CNN {tiny_score:.3f}")

# --- 11.2 distillation loss: alpha = 0 is plain cross-entropy, equal logits give zero KL ---------
z, y = torch.randn(8, 10), torch.randint(0, 10, (8,))
assert torch.isclose(kd_loss(z, torch.randn(8, 10), y, 4.0, 0.0), F.cross_entropy(z, y))
assert torch.isclose(kd_loss(z, z, y, 4.0, 1.0), torch.tensor(0.0), atol=1e-6)
ok("distillation loss: alpha=0 equals cross-entropy; identical teacher gives KL = 0")

# --- augmentation: every output is a 32x32 crop of the padded image, possibly flipped ------------
img = torch.rand(64, 3, 32, 32)
out = random_crop_flip(img, torch.Generator().manual_seed(1))
pad = F.pad(img, (4, 4, 4, 4))
for i in range(64):
    assert any(torch.equal(out[i], w) or torch.equal(out[i], w.flip(2))
               for w in (pad[i, :, a:a + 32, b:b + 32] for a in range(9) for b in range(9)))
ok("augmentation: random crop (padding 4) + horizontal flip")
print("\nall checks passed")

"""CIFAR-100: accuracy vs total time (scoring + training) for every scorer and scoring epoch,
and the best choice at each time budget. Same idea as the CIFAR-10 slide "Counting the scoring cost".

All times are L40S measurements:
  F         = mean training time of the 3 full-data runs (baselines)
  training  = mean L40S training time of keep-hard runs at the same keep share (main + cheap + epochs
              runs that ran on an L40S; time depends only on how many pictures are kept)
  S         = scoring time of the 3 scorer seeds, two ways:
              measured   = time logged up to that epoch + EL2N pass (includes ~60 s GPU start-up per process)
              start-up removed = epochs x steady time per epoch + EL2N pass (steady per-epoch time taken
                                 from the epochs after the first)
Run from the project root:  python results/cifar100/c100_cost_budget.py
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/cifar100/"
OUT = R + "analysis/"
read = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + pat))], ignore_index=True)

base = read("train_baselines_shard*.csv")
prune = pd.concat([read(p) for p in ("train_main_shard*.csv", "train_cheap_shard*.csv", "train_epochs_shard*.csv")
                   if glob.glob(R + p)], ignore_index=True)
prune = prune[prune.method == "hard"]
score = read("scoring_shard*.csv")

F = base[base.method == "full"].train_seconds.mean()
acc_full = base[base.method == "full"].test_acc.mean() * 100

# training time per keep share, L40S only
l40 = prune[prune.hardware.str.contains("L40S")]
train_sec = l40.groupby("data_fraction").train_seconds.mean()

# scoring time per (model, epoch): measured and start-up removed
S_meas, S_clean = {}, {}
for (m, seed), g in score.groupby(["model", "seed"]):
    g = g.set_index("snapshot_epoch").sort_index()
    eps = list(g.index)
    steady = (g.train_seconds[eps[-1]] - g.train_seconds[eps[1]]) / (eps[-1] - eps[1])   # s per epoch
    for e in eps:
        S_meas[(m, e)] = S_meas.get((m, e), 0) + g.train_seconds[e] + g.el2n_seconds[e]
        el2n = g.el2n_seconds[eps[-1]] if e == eps[0] else g.el2n_seconds[e]               # first pass is also warm-up
        S_clean[(m, e)] = S_clean.get((m, e), 0) + e * steady + el2n

rows = []
for sset, g in prune.groupby("score_set"):
    _, m, e = sset.split("_"); e = int(e[1:])
    for f, gg in g.groupby("data_fraction"):
        rows.append(dict(scorer=m, epoch=e, f=f, acc=gg.test_acc.mean() * 100, sd=gg.test_acc.std() * 100,
                         n=len(gg), S_meas=S_meas[(m, e)] / F, S_clean=S_clean[(m, e)] / F, T_train=train_sec[f] / F))
P = pd.DataFrame(rows)
P["cost"] = (P.S_clean + P.T_train) * 100
P["cost_meas"] = (P.S_meas + P.T_train) * 100

def series(method):
    g = base[base.method == method].groupby("data_fraction" if method == "random" else "epochs")
    return pd.DataFrame(dict(acc=g.test_acc.mean() * 100, cost=g.train_seconds.mean() / F * 100)).sort_values("cost")
short, rand = series("short"), series("random")

# ---- figure ---------------------------------------------------------------------------------------------
BLUE, GREY, INK = "#2a78d6", "#7a8494", "#14213d"
SC = {("resnet18", 20): ("#eb6834", "D", "-"), ("resnet18", 5): ("#eb6834", "D", "--"), ("resnet18", 1): ("#eb6834", "D", ":"),
      ("tiny", 20): ("#1baf7a", "^", "-"), ("tiny", 5): ("#1baf7a", "^", "--"), ("tiny", 1): ("#1baf7a", "^", ":")}
plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=200)
fig.subplots_adjust(left=0.07, right=0.72, top=0.95, bottom=0.13)
ax.grid(color="#e3e7ee", lw=0.9); ax.set_axisbelow(True)
ax.plot(list(short.cost) + [100], list(short.acc) + [acc_full], color=BLUE, marker="o", ms=8, lw=2.2, label="Fewer epochs, all data")
ax.plot(list(rand.cost) + [100], list(rand.acc) + [acc_full], color=GREY, marker="s", ms=8, lw=2.2, ls="--", label="Random subset")
for (m, e), (c, mk, ls) in SC.items():
    g = P[(P.scorer == m) & (P.epoch == e)].sort_values("cost")
    if len(g):
        ax.plot(g.cost, g.acc, color=c, marker=mk, ms=7, lw=1.8, ls=ls,
                label=f"Keep-hard, {'ResNet-18' if m == 'resnet18' else 'tiny CNN'} e{e}")
ax.axvline(100, color=INK, lw=1.1, ls=":")
ax.set_xlabel("Total time, % of one full run (scoring + training)")
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(28, 79)
ax.text(101, 29.5, "one full run", fontsize=10, color=INK)
ax.legend(fontsize=10.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.savefig(OUT + "cost_scatter_all_scorers.png", facecolor="white")

# ---- tables ---------------------------------------------------------------------------------------------
P.sort_values(["scorer", "epoch", "f"], ascending=[True, True, False]).to_csv(OUT + "cost_all_scorers.csv", index=False)
best = []
for B in (90, 75, 50, 30):
    cand = P[P.cost <= B + 5]
    sh = short[short.cost <= B + 5]
    p = cand.loc[cand.acc.idxmax()] if len(cand) else None
    s = sh.loc[sh.acc.idxmax()] if len(sh) else None
    best.append(dict(budget=f"about {B}% of F",
                     pruning=(f"{'R18' if p.scorer == 'resnet18' else 'tiny'} e{p.epoch}, keep {p.f:.0%}: "
                              f"{p.acc:.2f}% at {p.cost:.0f}%") if p is not None else "none",
                     fewer_epochs=f"{s.name} epochs: {s.acc:.2f}% at {s.cost:.0f}%" if s is not None else "none"))
pd.DataFrame(best).to_csv(OUT + "best_per_budget.csv", index=False)
print(f"F = {F:.1f} s, full = {acc_full:.2f}%")
print(pd.DataFrame(best).to_string(index=False))
print(P.groupby(["scorer", "epoch"])[["S_meas", "S_clean"]].first().round(3))


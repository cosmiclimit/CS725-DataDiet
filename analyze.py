"""Turn all results into the tables and figures for the presentation and report.

  python analyze.py                       # CIFAR-10
  python analyze.py --dataset cifar100
  python analyze.py --images              # also the 20 hardest / 20 easiest images (needs the CIFAR files)
  python analyze.py --smoke               # analyse the fake-data smoke results

Everything goes to results/<dataset>/analysis/, with REPORT.md collecting all tables.
Sections whose inputs do not exist yet are skipped with a note, so this can run at any stage.

Tables                                   Figures
  runs_summary.csv    every setting        baselines.png      accuracy vs compute for the baselines
  cost_table.csv      guide 10.7           strategies_*.png   hard / easy / random vs % data kept
  scorer_costs.csv    S/F and break-even   scorers.png        keep-hard with every EL2N scorer
  scorer_table.csv    guide 9.5            score_epochs.png   accuracy vs scoring epoch
  agreement.csv       Spearman, Kendall,   cost_scatter.png   total cost vs accuracy (top-left = best)
                      kept-set overlap     hardest_easiest.png (with --images)
  significance.csv    Welch t-tests vs random
  class_balance.csv   images kept per class
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                              # no screen needed (works on the cluster)
import matplotlib.pyplot as plt
from scipy import stats
from data import cifar_file
from experiments import config, ref_set
from prune import select
from scores import load_score_set, parse_score_set, score_path

# categorical colours in a fixed order (validated palette), grey for the full-data reference
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY, INK, GRID = "#52514e", "#0b0b0b", "#e5e4e0"
NOTES = []


def note(msg):
    print("  note:", msg)
    NOTES.append(msg)


# ----------------------------------------------------------------------------- loading
def load_runs(res_dir):
    files = sorted(glob.glob(os.path.join(res_dir, "train_*.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("job_id", keep="last")
    df["score_set"] = df["score_set"].fillna("-")
    return df


def load_scoring(res_dir):
    files = sorted(glob.glob(os.path.join(res_dir, "scoring_shard*.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(["model", "seed", "snapshot_epoch"], keep="last")


def label(r):
    """Human-readable name of a setting."""
    if r["method"] in ("full", "short", "random"):
        base = {"full": "full data", "short": "short schedule", "random": "random"}[r["method"]]
    else:
        kind, model, e = parse_score_set(r["score_set"])
        base = f"{r['method']} ({'EL2N' if kind == 'el2n' else 'forgetting'}, {model}, e{e})"
    if r.get("balanced", 0):
        base += " class-balanced"
    if r.get("offset", 0):
        base += f" offset {r['offset']:g}"
    if r.get("kd", 0):
        base += " + distill"
    return base


def summarize(df):
    keys = ["model", "method", "score_set", "data_fraction", "epochs", "balanced", "offset", "kd"]
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        acc = g["test_acc"].to_numpy() * 100
        n = len(acc)
        sd = acc.std(ddof=1) if n > 1 else np.nan
        ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        row = dict(zip(keys, k))
        row.update(acc_mean=acc.mean(), acc_std=sd, ci95=ci, seeds=n,
                   train_min=g["train_seconds"].mean() / 60, hardware=g["hardware"].iloc[0])
        row["setting"] = label(row)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- helpers
def md_table(df, floatfmt=".2f"):
    """Markdown table without extra dependencies."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def full_reference(ax, s, x_text):
    """Grey dashed line and band for full-data accuracy (mean +- std)."""
    full = s[(s["method"] == "full")]
    if not len(full):
        return
    m, sd = full["acc_mean"].iloc[0], np.nan_to_num(full["acc_std"].iloc[0])
    ax.axhspan(m - sd, m + sd, color=GREY, alpha=0.12, lw=0)
    ax.axhline(m, color=GREY, lw=1.5, ls="--")
    ax.annotate(f"full data: {m:.2f}%", (x_text, m), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=GREY)


def line(ax, sub, x, color, lab, **kw):
    sub = sub.sort_values(x)
    ax.errorbar(sub[x], sub["acc_mean"], yerr=np.nan_to_num(sub["acc_std"]), color=color, lw=2,
                marker=kw.pop("marker", "o"), ms=5, capsize=3, label=lab, **kw)


def plain(s):
    """Settings without class balance / offset / distillation, ResNet-18 target."""
    return s[(s["balanced"] == 0) & (s["offset"] == 0) & (s["kd"] == 0) & (s["model"] == "resnet18")]


# ----------------------------------------------------------------------------- figures
def fig_baselines(s, c, out):
    p = plain(s)
    if not len(p[p["method"].isin(["random", "short"])]):
        return note("no random/short baseline runs yet: baselines.png skipped")
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    full_reference(ax, p, 30)
    full = p[p["method"] == "full"]
    if len(full):
        ax.errorbar([100], full["acc_mean"], yerr=np.nan_to_num(full["acc_std"]), color=GREY, marker="o", capsize=3)
    for method, color, lab in [("random", PALETTE[0], f"random pruning ({c['epochs']} ep)"),
                               ("short", PALETTE[1], f"all data, short schedule (f x {c['epochs']} ep)")]:
        sub = p[p["method"] == method].copy()
        if len(sub):
            sub["cost_pct"] = (sub["data_fraction"] * sub["epochs"] / c["epochs"] * 100).round(1)
            line(ax, sub, "cost_pct", color, lab)
    style(ax, "Baselines: accuracy vs training compute (mean ± std over seeds)",
          "training compute (% of one full-data run)", "test accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_strategies(s, score_set, out):
    p = plain(s)
    sub = p[(p["score_set"] == score_set) & p["method"].isin(["hard", "easy"])]
    if not len(sub):
        return note(f"no hard/easy runs with {score_set} yet: {os.path.basename(out)} skipped")
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    full_reference(ax, p, 30)
    p = p.assign(pct=p["data_fraction"] * 100)
    line(ax, p[p["method"] == "random"], "pct", PALETTE[0], "random")
    line(ax, p[(p["score_set"] == score_set) & (p["method"] == "hard")], "pct", PALETTE[2], "keep hard")
    line(ax, p[(p["score_set"] == score_set) & (p["method"] == "easy")], "pct", PALETTE[6], "keep easy")
    kind, model, e = parse_score_set(score_set)
    style(ax, f"Which images to keep? ({'EL2N' if kind == 'el2n' else 'forgetting'} from {model}, epoch {e})",
          "% of training data kept", "test accuracy (%)")
    ax.set_xticks([30, 50, 70, 90])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def el2n_sets(c):
    """Reference first, then the other EL2N score sets (fixed order = fixed colours)."""
    last = c["snapshots"][-1]
    order = [(m, e) for e in [last] + c["snapshots"][:-1][::-1] for m in ("resnet18", "tiny")]
    return [f"el2n_{m}_e{e}" for m, e in order]


def fig_scorers(s, c, out):
    p = plain(s)
    hard = p[(p["method"] == "hard") & p["score_set"].str.startswith("el2n")]
    if hard["score_set"].nunique() < 2:
        return note("fewer than two EL2N scorers with keep-hard runs: scorers.png skipped")
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    full_reference(ax, p, 30)
    p = p.assign(pct=p["data_fraction"] * 100)
    line(ax, p[p["method"] == "random"], "pct", GREY, "random", ls=":", marker="s")
    for i, ss in enumerate(el2n_sets(c)):
        sub = p[(p["method"] == "hard") & (p["score_set"] == ss)]
        if len(sub):
            _, m, e = parse_score_set(ss)
            line(ax, sub, "pct", PALETTE[(i + 2) % len(PALETTE)], f"keep hard, {m} e{e}")
    style(ax, "Keep-hard accuracy for each scorer and scoring epoch", "% of training data kept", "test accuracy (%)")
    ax.set_xticks([30, 50, 70, 90])
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_score_epochs(s, c, out):
    p = plain(s)
    hard = p[(p["method"] == "hard") & p["score_set"].str.startswith("el2n")]
    if not len(hard):
        return note("no keep-hard EL2N runs yet: score_epochs.png skipped")
    rows = []
    for _, r in hard.iterrows():
        _, m, e = parse_score_set(r["score_set"])
        rows.append(dict(r, scorer=m, score_epoch=e))
    h = pd.DataFrame(rows)
    fracs = [f for f in (0.5, 0.3) if f in set(h["data_fraction"])] or sorted(set(h["data_fraction"]))[:2]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    for fi, f in enumerate(fracs):
        rnd = p[(p["method"] == "random") & (p["data_fraction"] == f)]
        if len(rnd):
            ax.axhline(rnd["acc_mean"].iloc[0], color=GREY, lw=1, ls=[":", "--"][fi % 2])
            ax.annotate(f"random {f:.0%}", (c["snapshots"][0], rnd["acc_mean"].iloc[0]), xytext=(0, 3),
                        textcoords="offset points", fontsize=7, color=GREY)
        for mi, m in enumerate(("resnet18", "tiny")):
            sub = h[(h["scorer"] == m) & (h["data_fraction"] == f)]
            if len(sub):
                line(ax, sub, "score_epoch", PALETTE[2 + mi], f"{m} scorer, keep {f:.0%}",
                     ls=["-", "--"][fi % 2], marker=["o", "s"][fi % 2])
    ax.set_xscale("log")
    ax.set_xticks(c["snapshots"])
    ax.set_xticklabels([str(e) for e in c["snapshots"]])
    style(ax, "How early can we score? Keep-hard accuracy vs scoring epoch", "scoring epoch (log scale)",
          "test accuracy (%)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_cost_scatter(cost, c, out):
    """One point per method and keep fraction. Shows the baselines and keep-hard with each EL2N scorer
    (at most 8 colours); every other setting is in cost_table.csv."""
    if not len(cost):
        return
    colors = {"full data": GREY, "random": PALETTE[0], "short schedule": PALETTE[1]}
    for i, ss in enumerate(el2n_sets(c)):
        colors[f"hard:{ss}"] = PALETTE[(2 + i) % len(PALETTE)]
    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=200)
    for key, color in colors.items():
        sub = cost[cost["key"] == key]
        if not len(sub):
            continue
        ax.errorbar(sub["total_pct"], sub["acc_mean"], yerr=np.nan_to_num(sub["acc_std"]), fmt="o", ms=6,
                    color=color, capsize=2, label=sub["group"].iloc[0], markeredgecolor="white",
                    markeredgewidth=1)
    ax.axvline(100, color=GREY, lw=1, ls="--")
    ax.annotate("one full-data run", (100, ax.get_ylim()[1]), xytext=(4, -4), textcoords="offset points",
                fontsize=7, color=GREY, ha="left", va="top")
    style(ax, "Total cost (scoring + training) vs accuracy -- top-left is best",
          "total GPU time (% of one full-data run)", "test accuracy (%)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def scorer_costs(scoring, c, F_sec):
    """Cost S of every score set = sum over scorer runs of (training up to the snapshot + EL2N pass)."""
    if not len(scoring):
        return pd.DataFrame()
    rows = []
    for (m, e), g in scoring.groupby(["model", "snapshot_epoch"]):
        g = g[g["seed"].isin(c["scorer_seeds"])]
        for kind in ("el2n", "forget"):
            S = g["train_seconds"].sum() + (g["el2n_seconds"].sum() if kind == "el2n" else 0)
            rows.append(dict(score_set=f"{kind}_{m}_e{e}", scorer=m, epoch=e, runs=len(g), cost_sec=S,
                             scorer_test_acc=g["scorer_test_acc"].mean() * 100,
                             S_over_F=S / F_sec if F_sec else np.nan,
                             break_even_f=1 - S / F_sec if F_sec else np.nan))
    return pd.DataFrame(rows).sort_values(["scorer", "epoch", "score_set"])


def cost_table(s, sc, c, F_sec):
    """Guide 10.7: scoring + training time vs one full-data run, with accuracy."""
    p = s[s["model"] == "resnet18"]
    S = dict(zip(sc["score_set"], sc["cost_sec"])) if len(sc) else {}
    ref = ref_set(c)
    rows = []
    for _, r in p.iterrows():
        if r["method"] in ("hard", "easy"):
            if r["score_set"] not in S:
                continue
            scoring = S[r["score_set"]]
        else:
            scoring = 0.0
        if r["kd"] and r["method"] == "random":
            scoring += S.get(ref, np.nan)          # the teacher's training cost counts too
        variant = r["balanced"] or r["offset"] or r["kd"]
        if r["method"] == "full":
            group = key = "full data"
        elif r["method"] == "short":
            group = key = "short schedule"
        elif r["method"] == "random":
            group = "random" + (" + distill" if r["kd"] else "")
            key = group
        else:
            group = r["setting"]
            key = f"{r['method']}:{r['score_set']}" + (":variant" if variant else "")
        f = r["data_fraction"] if r["method"] != "short" else r["epochs"] / c["epochs"]
        total = scoring + r["train_min"] * 60
        rows.append(dict(group=group, key=key, f=f, scoring_min=scoring / 60, training_min=r["train_min"],
                         total_min=total / 60, total_pct=100 * total / F_sec if F_sec else np.nan,
                         acc_mean=r["acc_mean"], acc_std=r["acc_std"], seeds=r["seeds"]))
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    order = {"full data": 0, "short schedule": 1, "random": 2}
    df["o"] = df["group"].map(lambda g: order.get(g, 3))
    return df.sort_values(["o", "group", "f"], ascending=[True, True, False]).drop(columns="o")


def agreement(score_dir, c):
    """Compare every score set with the reference: Spearman, Kendall and kept-set overlap (guide 9.4)."""
    ref = ref_set(c)
    try:
        R = load_score_set(score_dir, ref, c["scorer_seeds"])
    except FileNotFoundError:
        note("reference score files missing: agreement.csv skipped")
        return pd.DataFrame()
    last = c["snapshots"][-1]
    others = [ss for ss in el2n_sets(c) if ss != ref] + [f"forget_resnet18_e{last}", f"forget_tiny_e{last}"]
    rows = []
    for ss in others:
        try:
            X = load_score_set(score_dir, ss, c["scorer_seeds"])
        except FileNotFoundError:
            continue
        row = dict(score_set=ss, spearman=stats.spearmanr(R, X).statistic,
                   kendall=stats.kendalltau(R, X).statistic)
        for f in c["fractions"]:
            a = select(len(R), f, "hard", R, seed=0)
            b = select(len(X), f, "hard", X, seed=0)
            row[f"overlap@{f:.0%}"] = len(np.intersect1d(a, b)) / len(a)
            row[f"random@{f:.0%}"] = f                           # expected overlap by chance
        rows.append(row)
    return pd.DataFrame(rows)


def scorer_table(s, sc, agr, c):
    """Guide 9.5: scorer, epochs, runs, cost, agreement with the reference, keep-hard accuracy."""
    if not len(sc):
        return pd.DataFrame()
    p = plain(s)
    rows = []
    for _, r in sc[sc["score_set"].str.startswith("el2n")].iterrows():
        row = dict(scorer=r["scorer"], epoch=r["epoch"], runs=r["runs"], cost_sec=r["cost_sec"],
                   scorer_acc=r["scorer_test_acc"])
        a = agr[agr["score_set"] == r["score_set"]] if len(agr) else pd.DataFrame()
        row["spearman_vs_ref"] = a["spearman"].iloc[0] if len(a) else (1.0 if r["score_set"] == ref_set(c) else np.nan)
        for f in (0.5, 0.3):
            h = p[(p["method"] == "hard") & (p["score_set"] == r["score_set"]) & (p["data_fraction"] == f)]
            row[f"hard_acc@{f:.0%}"] = h["acc_mean"].iloc[0] if len(h) else np.nan
        rows.append(row)
    for f in (0.5, 0.3):
        rnd = p[(p["method"] == "random") & (p["data_fraction"] == f)]
        if len(rnd):
            for row in rows:
                row[f"random_acc@{f:.0%}"] = rnd["acc_mean"].iloc[0]
    return pd.DataFrame(rows)


def significance(df):
    """Welch t-test of every pruned setting against random pruning at the same keep fraction."""
    rows = []
    p = df[df["model"] == "resnet18"]
    rnd = p[(p["method"] == "random") & (p["kd"] == 0)]
    for _, g in p[p["method"].isin(["hard", "easy"]) | (p["kd"] == 1)].groupby(
            ["method", "score_set", "data_fraction", "balanced", "offset", "kd"]):
        f = g["data_fraction"].iloc[0]
        base = rnd[rnd["data_fraction"] == f]["test_acc"].to_numpy() * 100
        x = g["test_acc"].to_numpy() * 100
        if len(x) < 2 or len(base) < 2:
            continue
        t, pval = stats.ttest_ind(x, base, equal_var=False)
        rows.append(dict(setting=label(g.iloc[0].to_dict()), f=f, mean=x.mean(), random_mean=base.mean(),
                         diff=x.mean() - base.mean(), t=t, p_value=pval,
                         significant_5pct="yes" if pval < 0.05 else "no"))
    return pd.DataFrame(rows)


def class_balance(score_dir, c, dataset):
    """How many images of each class survive keep-hard / keep-easy with the reference scores."""
    ref = ref_set(c)
    _, model, e = parse_score_set(ref)
    path = score_path(score_dir, model, c["scorer_seeds"][0], e)
    if not os.path.exists(path):
        return pd.DataFrame()
    labels = np.load(path)["labels"]
    R = load_score_set(score_dir, ref, c["scorer_seeds"])
    K = labels.max() + 1
    rows = []
    for f in c["fractions"]:
        for strat in ("hard", "easy"):
            counts = np.bincount(labels[select(len(R), f, strat, R, seed=0)], minlength=K)
            row = dict(strategy=strat, f=f"{f:.0%}", min_per_class=counts.min(), max_per_class=counts.max(),
                       expected=round(f * len(labels) / K))
            if K <= 10:
                row.update({f"class{k}": int(v) for k, v in enumerate(counts)})
            rows.append(row)
    return pd.DataFrame(rows)


def hardest_easiest(score_dir, c, dataset, data_root, out, k=20):
    try:
        d = np.load(cifar_file(data_root, dataset))
        images, labels, classes = d["x_train"], d["y_train"], d["classes"]
        R = load_score_set(score_dir, ref_set(c), c["scorer_seeds"])
    except Exception as e:                       # dataset or scores missing
        return note(f"hardest/easiest images skipped ({e})")
    order = np.argsort(-R)
    fig, axes = plt.subplots(4, 10, figsize=(12, 6), dpi=150)
    for row, (title, idx) in enumerate([("hardest", order[:k]), ("easiest", order[::-1][:k])]):
        for n, i in enumerate(idx):
            ax = axes[2 * row + n // 10, n % 10]
            ax.imshow(images[i])
            ax.set_title(f"{classes[labels[i]]}\n{R[i]:.2f}", fontsize=7)
            ax.axis("off")
        axes[2 * row, 0].text(-0.6, 0.5, title, transform=axes[2 * row, 0].transAxes, rotation=90,
                              va="center", ha="center", fontsize=11)
    fig.suptitle(f"{k} hardest (top) and {k} easiest (bottom) training images by {ref_set(c)} "
                 "(label, EL2N). Check the hardest ones for wrong labels.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--scores-dir", default="scores")
    ap.add_argument("--data-root", default=os.path.expanduser("~/datadiet_data"))
    ap.add_argument("--images", action="store_true", help="draw the hardest / easiest images (needs CIFAR files)")
    ap.add_argument("--smoke", action="store_true", help="analyse the smoke-test outputs")
    a = ap.parse_args()

    c = config(a.smoke)
    res_dir = os.path.join(a.results_dir, "smoke" if a.smoke else "", a.dataset)
    score_dir = os.path.join(a.scores_dir, "smoke" if a.smoke else "", a.dataset)
    out = os.path.join(res_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    P = lambda name: os.path.join(out, name)

    runs, scoring = load_runs(res_dir), load_scoring(res_dir)
    print(f"{len(runs)} training runs and {len(scoring)} scorer snapshots in {res_dir}")
    report = [f"# Results: {a.dataset.upper()}", ""]
    if len(runs):
        hw = ", ".join(sorted(runs["hardware"].astype(str).unique()))
        report += [f"{len(runs)} training runs, {len(scoring)} scorer snapshots. Hardware: {hw}. "
                   f"Target: ResNet-18, {c['epochs']} epochs (equal-epochs setting). "
                   "Accuracy in %, mean ± sample std over seeds; ci95 = half-width of the 95% t-interval.", ""]
        if runs["hardware"].nunique() > 1:
            note("results come from more than one GPU type -- timings are not comparable across them")

    s = summarize(runs) if len(runs) else pd.DataFrame()
    full = s[(s["method"] == "full") & (s["model"] == "resnet18") & (s["epochs"] == c["epochs"])] if len(s) else s
    F_sec = full["train_min"].iloc[0] * 60 if len(full) else None
    if len(s):
        cols = ["setting", "data_fraction", "epochs", "seeds", "acc_mean", "acc_std", "ci95", "train_min"]
        tab = s.sort_values(["method", "score_set", "data_fraction"])[cols]
        tab.to_csv(P("runs_summary.csv"), index=False)
        report += ["## All settings", "", md_table(tab), ""]
        fig_baselines(s, c, P("baselines.png"))
        for ss in sorted(set(s["score_set"]) - {"-"}):
            fig_strategies(s, ss, P(f"strategies_{ss}.png"))
        fig_scorers(s, c, P("scorers.png"))
        fig_score_epochs(s, c, P("score_epochs.png"))
    else:
        note("no training runs yet")
    if F_sec is None:
        note("no full-data run yet: costs cannot be expressed relative to a full run")

    sc = scorer_costs(scoring, c, F_sec)
    if len(sc):
        sc.to_csv(P("scorer_costs.csv"), index=False)
        report += ["## Scoring cost", "", "S = total GPU seconds of all scorer runs up to the snapshot (+ the EL2N "
                   "forward pass). Pruning with equal epochs saves compute only if f < 1 - S/F (break-even f).", "",
                   md_table(sc, ".3f"), ""]
    if len(s) and F_sec:
        cost = cost_table(s, sc, c, F_sec)
        if len(cost):
            cost.drop(columns="key").to_csv(P("cost_table.csv"), index=False)
            report += ["## Cost table (guide 10.7)", "", f"F = one full-data run = {F_sec / 60:.2f} min. "
                       "total_pct < 100 means cheaper than training on all data.", "",
                       md_table(cost.drop(columns="key")), ""]
            fig_cost_scatter(cost, c, P("cost_scatter.png"))

    agr = agreement(score_dir, c)
    if len(agr):
        agr.to_csv(P("agreement.csv"), index=False)
        report += ["## Scorer agreement with the reference (guide 9.4)", "",
                   f"Reference: {ref_set(c)}. Overlap = share of the reference's kept (hardest) images also kept "
                   "by the other scorer; compare with random@f, the overlap expected by chance.", "",
                   md_table(agr, ".3f"), ""]
    st = scorer_table(s, sc, agr, c) if len(sc) and len(s) else pd.DataFrame()
    if len(st):
        st.to_csv(P("scorer_table.csv"), index=False)
        report += ["## Scorer table (guide 9.5)", "", md_table(st), ""]
    if len(runs):
        sig = significance(runs)
        if len(sig):
            sig.to_csv(P("significance.csv"), index=False)
            report += ["## Is it better than random? (Welch t-test, guide 12.4)", "", md_table(sig, ".3f"), ""]
    cb = class_balance(score_dir, c, a.dataset)
    if len(cb):
        cb.to_csv(P("class_balance.csv"), index=False)
        report += ["## Images kept per class (reference scores, guide 5.6)", "", md_table(cb, ".0f"), ""]
    if a.images:
        hardest_easiest(score_dir, c, a.dataset, a.data_root, P("hardest_easiest.png"))

    figs = sorted(os.path.basename(f) for f in glob.glob(P("*.png")))
    report += ["## Figures", ""] + [f"![{f}]({f})" for f in figs] + [""]
    if NOTES:
        report += ["## Notes", ""] + [f"- {n}" for n in NOTES] + [""]
    with open(P("REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print(f"wrote {len(figs)} figures and REPORT.md to {out}")


if __name__ == "__main__":
    main()

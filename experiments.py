"""Every experiment of the project, written as a list of jobs.

A job is a dict. kind='train' trains one target model; kind='score' trains one scorer and saves
its scores at the snapshot epochs. run.py executes the jobs; analyze.py reads their results.

Settings that must be identical across methods (guide Ch. 12.5) live in `config()`.
"""

EXPERIMENTS = {
    "check":      "2-epoch timing runs of ResNet-18 and the tiny CNN on all data (first GPU test)",
    "baselines":  "full data (60 ep), random pruning, short schedule (f x 60 ep) -- every keep fraction",
    "scores":     "train tiny CNN and ResNet-18 scorers (3 seeds), save EL2N + forgetting at epochs 1/5/20",
    "main":       "keep hard / easy with the reference scores (EL2N, ResNet-18, epoch 20)",
    "cheap":      "keep hard / easy with the cheap scores (EL2N, tiny CNN, epoch 20)",
    "epochs":     "keep hard with early scores (EL2N at epochs 1 and 5, both scorers)",
    "forgetting": "keep hard / easy with forgetting scores (ResNet-18, epoch 20) -- stretch goal",
    "variants":   "keep hard, class-balanced and with a 5% offset (reference scores) -- optional",
    "distill":    "keep hard / random at 50% and 30% trained with distillation from the scorers -- bonus",
}
NEEDS_SCORES = {"main", "cheap", "epochs", "forgetting", "variants", "distill"}
MODEL_COST = {"resnet18": 1.0, "tiny": 0.2, "resnet50": 2.5}   # rough, only for splitting work


def config(smoke=False):
    if smoke:   # tiny version of everything, for checking the code on a laptop CPU
        return dict(fractions=[0.5], seeds=[0], scorer_seeds=[1000], epochs=2,
                    snapshots=[1, 2, 3], scorers=["tiny", "resnet18"], target="resnet18")
    return dict(
        fractions=[0.9, 0.7, 0.5, 0.3],       # keep fractions
        seeds=[0, 1, 2],                      # target seeds -- the same for every method
        scorer_seeds=[1000, 1001, 1002],      # scorers use other seeds than the targets
        epochs=60,                            # target training length (equal-epochs setting)
        snapshots=[1, 5, 20],                 # scoring epochs (of a 60-epoch cosine schedule)
        scorers=["tiny", "resnet18"],
        target="resnet18",
    )


def ref_set(c):
    return f"el2n_resnet18_e{c['snapshots'][-1]}"


def train_job(c, method, fraction, seed, score_set="-", epochs=None, model=None,
              balanced=False, offset=0.0, kd=False):
    epochs = epochs or c["epochs"]
    model = model or c["target"]
    data_fraction = 1.0 if method in ("full", "short") else fraction
    return dict(kind="train", model=model, method=method, score_set=score_set,
                data_fraction=data_fraction, epochs=epochs, seed=seed,
                balanced=balanced, offset=offset, kd=kd,
                cost=data_fraction * epochs / c["epochs"] * MODEL_COST[model])


def score_job(c, model, seed):
    snaps = c["snapshots"]
    return dict(kind="score", model=model, seed=seed, snapshots=snaps,
                train_epochs=max(snaps), schedule_epochs=c["epochs"],
                cost=max(snaps) / c["epochs"] * MODEL_COST[model])


def job_id(dataset, j):
    if j["kind"] == "score":
        return f"score|{dataset}|{j['model']}|{j['seed']}|{j['train_epochs']}of{j['schedule_epochs']}"
    return (f"train|{dataset}|{j['model']}|{j['method']}|{j['score_set']}|{float(j['data_fraction'])}|"
            f"{j['epochs']}|{j['seed']}|bal{int(j['balanced'])}|off{float(j['offset'])}|kd{int(j['kd'])}")


def build(name, c):
    F, S, E = c["fractions"], c["seeds"], c["epochs"]
    last = c["snapshots"][-1]
    REF, TINY = ref_set(c), f"el2n_tiny_e{last}"
    small = [f for f in F if f in (0.5, 0.3)] or F[-1:]       # fractions for the optional extras
    jobs = []
    if name == "check":
        jobs = [train_job(c, "full", 1.0, 0, epochs=2, model="resnet18"),
                train_job(c, "full", 1.0, 0, epochs=2, model="tiny")]
    elif name == "baselines":
        for s in S:
            jobs.append(train_job(c, "full", 1.0, s))
            for f in F:
                jobs.append(train_job(c, "random", f, s))
                jobs.append(train_job(c, "short", f, s, epochs=max(1, round(f * E))))
    elif name == "scores":
        jobs = [score_job(c, m, s) for m in c["scorers"] for s in c["scorer_seeds"]]
    elif name in ("main", "cheap", "forgetting"):
        sset = {"main": REF, "cheap": TINY, "forgetting": f"forget_resnet18_e{last}"}[name]
        jobs = [train_job(c, m, f, s, sset) for s in S for f in F for m in ("hard", "easy")]
    elif name == "epochs":
        sets = [f"el2n_{m}_e{e}" for m in c["scorers"] for e in c["snapshots"][:-1]]
        jobs = [train_job(c, "hard", f, s, ss) for s in S for f in F for ss in sets]
    elif name == "variants":
        jobs = ([train_job(c, "hard", f, s, REF, balanced=True) for s in S for f in small] +
                [train_job(c, "hard", f, s, REF, offset=0.05) for s in S for f in small])
    elif name == "distill":
        jobs = ([train_job(c, "hard", f, s, REF, kd=True) for s in S for f in small] +
                [train_job(c, "random", f, s, kd=True) for s in S for f in small])
    else:
        raise SystemExit(f"unknown experiment {name!r}; choose from {', '.join(EXPERIMENTS)}")
    return jobs

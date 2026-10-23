"""Run one experiment (see experiments.py), optionally split across several GPUs.

  python run.py --list                                  # what experiments exist
  python run.py baselines --dry-run                     # show the plan, train nothing
  python run.py baselines --smoke                       # tiny CPU check on fake data
  python run.py baselines --shard 0 --num-shards 4      # what each GPU runs inside a SLURM job
  python run.py main --dataset cifar100

Outputs
  results/<dataset>/train_<experiment>_shard<k>.csv    one row per trained target model
  results/<dataset>/scoring_shard<k>.csv               one row per scorer snapshot (with its cost)
  scores/<dataset>/<model>_seed<s>_e<epoch>.npz        the scores themselves

Every job has a unique id. Jobs whose id is already in a results file (or whose score files
exist) are skipped, so a killed job can simply be submitted again, and two experiments never
repeat the same run.
"""
import argparse, csv, glob, os, platform, socket, sys, time
import numpy as np
import torch
from data import load_cifar, fake_cifar, NUM_CLASSES
from experiments import EXPERIMENTS, NEEDS_SCORES, config, build, job_id
from models import make_model
from prune import select
from scores import (el2n_and_logits, forgetting_scores, score_path, logits_path,
                    load_score_set, load_teacher_logits, parse_score_set)
from train import train, evaluate, sync


def shard_of(jobs, num_shards):
    """Balance the work: give each job (most expensive first) to the least-loaded shard."""
    load, owner = [0.0] * num_shards, {}
    for i in sorted(range(len(jobs)), key=lambda i: -jobs[i]["cost"]):
        k = load.index(min(load))
        owner[i] = k
        load[k] += jobs[i]["cost"]
    return owner, load


def done_ids(res_dir, score_dir, dataset, jobs, c):
    done = set()
    for path in glob.glob(os.path.join(res_dir, "train_*.csv")):
        with open(path, newline="") as fh:
            done |= {row["job_id"] for row in csv.DictReader(fh)}
    for j in jobs:
        if j["kind"] == "score" and all(os.path.exists(score_path(score_dir, j["model"], j["seed"], e))
                                        for e in j["snapshots"]):
            done.add(job_id(dataset, j))
    return done


def log_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def describe(j):
    if j["kind"] == "score":
        return f"score  {j['model']:8s} seed={j['seed']} epochs={j['train_epochs']} snapshots={j['snapshots']}"
    extra = "".join([" balanced" if j["balanced"] else "", f" offset={j['offset']}" if j["offset"] else "",
                     " +distill" if j["kd"] else ""])
    ss = f" scores={j['score_set']}" if j["score_set"] != "-" else ""
    return (f"train  {j['model']:8s} {j['method']:6s} data={j['data_fraction']:.1f} epochs={j['epochs']:2d} "
            f"seed={j['seed']}{ss}{extra}")


def common_fields(env):
    return dict(hardware=env["hw"], host=env["host"], torch=torch.__version__,
                slurm_job=os.environ.get("SLURM_JOB_ID", ""), finished=time.strftime("%Y-%m-%d %H:%M:%S"))


def run_score_job(j, dataset, train_data, test_data, score_dir, res_dir, shard, env):
    """Train a scorer; at every snapshot epoch save EL2N + forgetting and log the cost so far."""
    K = NUM_CLASSES[dataset]
    os.makedirs(score_dir, exist_ok=True)
    device = train_data.y.device

    def on_epoch_end(ep, model, train_seconds, fstate):
        if ep not in j["snapshots"]:
            return
        sync(device)
        t0 = time.time()
        el2n, logits = el2n_and_logits(model, train_data)        # one forward pass = scoring cost
        sync(device)
        el2n_seconds = time.time() - t0
        acc = evaluate(model, test_data)                           # diagnostic only, not a cost
        forgets, learned = fstate
        np.savez_compressed(score_path(score_dir, j["model"], j["seed"], ep),
                            el2n=el2n.astype(np.float32), forgetting=forgetting_scores(forgets, learned),
                            forget_raw=forgets.cpu().numpy().astype(np.int16), learned=learned.cpu().numpy(),
                            labels=train_data.y.cpu().numpy().astype(np.int16))
        if ep == max(j["snapshots"]):
            np.save(logits_path(score_dir, j["model"], j["seed"], ep), logits)
        print(f"  snapshot e{ep}: scorer test acc {acc:.4f}, EL2N mean {el2n.mean():.3f}, "
              f"forgetting>0 for {(forgets > 0).float().mean().item():.1%}, "
              f"never learned {(~learned).float().mean().item():.1%}", flush=True)
        log_row(os.path.join(res_dir, f"scoring_shard{shard}.csv"), dict(
            job_id=job_id(dataset, j), dataset=dataset, model=j["model"], seed=j["seed"], snapshot_epoch=ep,
            schedule_epochs=j["schedule_epochs"], train_seconds=round(train_seconds, 2),
            el2n_seconds=round(el2n_seconds, 2), scorer_test_acc=round(acc, 4), **common_fields(env)))

    torch.manual_seed(j["seed"])
    train(make_model(j["model"], K), train_data, test_data, j["train_epochs"], seed=j["seed"],
          schedule_epochs=j["schedule_epochs"], track_forgetting=True, on_epoch_end=on_epoch_end,
          evaluate_at_end=False, log_every=1 if j["train_epochs"] <= 5 else 5)


def run_train_job(j, dataset, train_data, test_data, score_dir, res_path, c, env, experiment):
    K = NUM_CLASSES[dataset]
    N = len(train_data)
    labels = train_data.y.cpu().numpy()
    idx = None
    if j["method"] == "random":
        idx = select(N, j["data_fraction"], "random", seed=j["seed"])
    elif j["method"] in ("hard", "easy"):
        scores = load_score_set(score_dir, j["score_set"], c["scorer_seeds"])
        idx = select(N, j["data_fraction"], j["method"], scores, seed=j["seed"], labels=labels,
                     class_balanced=j["balanced"], offset=j["offset"])
    soft = None
    if j["kd"]:     # teacher = the reference scorers, averaged (their cost is counted in analysis)
        _, tmodel, tepoch = parse_score_set(f"el2n_resnet18_e{c['snapshots'][-1]}")
        soft = torch.as_tensor(load_teacher_logits(score_dir, tmodel, tepoch, c["scorer_seeds"]),
                               device=train_data.y.device)
    torch.manual_seed(j["seed"])                   # seeds the weight initialisation
    res = train(make_model(j["model"], K), train_data, test_data, j["epochs"], indices=idx,
                seed=j["seed"], soft_targets=soft, log_every=1 if j["epochs"] <= 5 else 10)
    print(f"  -> test acc {res['test_acc']:.4f}, train time {res['train_seconds'] / 60:.2f} min", flush=True)
    log_row(res_path, dict(
        job_id=job_id(dataset, j), experiment=experiment, dataset=dataset, model=j["model"],
        method=j["method"], score_set=j["score_set"], data_fraction=j["data_fraction"],
        balanced=int(j["balanced"]), offset=j["offset"], kd=int(j["kd"]), epochs=j["epochs"],
        seed=j["seed"], n_kept=N if idx is None else len(idx), test_acc=round(res["test_acc"], 4),
        train_seconds=round(res["train_seconds"], 2), sec_per_epoch=round(res["train_seconds"] / j["epochs"], 3),
        final_train_acc=round(res["train_acc"], 4), **common_fields(env)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", help="one of: " + ", ".join(EXPERIMENTS))
    ap.add_argument("--list", action="store_true", help="list the experiments and exit")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--data-root", default=os.path.expanduser("~/datadiet_data"))
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--scores-dir", default="scores")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seeds", nargs="+", type=int, help="override the target seeds (e.g. --seeds 0)")
    ap.add_argument("--fractions", nargs="+", type=float, help="override the keep fractions")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--smoke", action="store_true",
                    help="code check (GPU if available, else CPU): fake data, 2-epoch targets, scorers to epoch 3, "
                         "outputs in results/smoke and scores/smoke")
    a = ap.parse_args()

    if a.list or not a.experiment:
        for name, text in EXPERIMENTS.items():
            print(f"  {name:11s} {text}")
        return
    c = config(a.smoke)
    if a.seeds:
        c["seeds"] = a.seeds
    if a.fractions:
        c["fractions"] = a.fractions
    results_root = os.path.join(a.results_dir, "smoke") if a.smoke else a.results_dir
    scores_root = os.path.join(a.scores_dir, "smoke") if a.smoke else a.scores_dir
    if a.experiment == "check" and not a.smoke:
        results_root = os.path.join(a.results_dir, "check")        # timing checks are not results
    res_dir = os.path.join(results_root, a.dataset)
    score_dir = os.path.join(scores_root, a.dataset)

    jobs = build(a.experiment, c)
    owner, load = shard_of(jobs, a.num_shards)
    mine = [j for i, j in enumerate(jobs) if owner[i] == a.shard]
    done = done_ids(res_dir, score_dir, a.dataset, jobs, c)
    todo = [j for j in mine if job_id(a.dataset, j) not in done]
    print(f"{a.experiment} on {a.dataset}: {len(jobs)} jobs = {sum(load):.1f} full-run equivalents; "
          f"shard {a.shard}/{a.num_shards} has {len(mine)} jobs ({load[a.shard]:.1f}), {len(todo)} still to do")
    for j in mine:
        print(f"  [{'todo' if j in todo else 'done'}] {describe(j)}")
    if a.experiment in NEEDS_SCORES and todo:      # fail now, not after loading data
        needed = {j["score_set"] for j in todo if j["score_set"] != "-"}
        if any(j["kd"] for j in todo):
            needed.add(f"el2n_resnet18_e{c['snapshots'][-1]}")
        missing = []
        for ss in sorted(needed):
            try:
                load_score_set(score_dir, ss, c["scorer_seeds"])
            except FileNotFoundError:
                missing.append(ss)
        if missing:
            print(f"\n{'NOTE' if a.dry_run else 'ERROR'}: score files for {', '.join(missing)} are not in "
                  f"{score_dir}/. Run the 'scores' experiment (same --dataset) first.")
            if not a.dry_run:
                sys.exit(2)
    if a.dry_run or not todo:
        if not todo:
            print("nothing left to do for this shard")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # smoke tests use a GPU if there is one
    hw = torch.cuda.get_device_name(device) if device.type == "cuda" else (platform.processor() or "cpu")
    env = dict(hw=hw, host=socket.gethostname())
    print(f"host {env['host']} | {hw} | torch {torch.__version__} | slurm job {os.environ.get('SLURM_JOB_ID', '-')}",
          flush=True)
    if device.type == "cpu" and not a.smoke:
        print("WARNING: no GPU visible -- real runs on a CPU would take days.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True      # pick the fastest conv kernels for our shapes

    t0 = time.time()
    train_data, test_data = (fake_cifar(a.dataset, device, n_train=1000, n_test=200) if a.smoke
                             else load_cifar(a.data_root, a.dataset, device))
    print(f"data on {device}: {len(train_data)} train / {len(test_data)} test ({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, f"train_{a.experiment}_shard{a.shard}.csv")
    for n, j in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {describe(j)}", flush=True)
        if j["kind"] == "score":
            run_score_job(j, a.dataset, train_data, test_data, score_dir, res_dir, a.shard, env)
        else:
            run_train_job(j, a.dataset, train_data, test_data, score_dir, res_path, c, env, a.experiment)
    print("\nall jobs of this shard finished")


if __name__ == "__main__":
    main()

"""Estimate the GPU time of every experiment from the 2-epoch timing check.

  python estimate.py                 # after `python run.py check` (run by slurm/gpu_check.sbatch)
  python estimate.py --gpus 8        # wall-clock estimate for 8 GPUs per job

Uses the measured seconds per epoch of ResNet-18 and the tiny CNN on all 50,000 images.
"""
import argparse, glob, os
import pandas as pd
from experiments import EXPERIMENTS, config, build

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", default="results/check/cifar10")
ap.add_argument("--gpus", type=int, default=4)
a = ap.parse_args()

files = glob.glob(os.path.join(a.results_dir, "train_check_shard*.csv"))
if not files:
    raise SystemExit(f"no timing results in {a.results_dir}; run `python run.py check` on a GPU first")
df = pd.concat([pd.read_csv(f) for f in files])
spe = dict(zip(df["model"], df["sec_per_epoch"]))
c = config()
F = spe["resnet18"] * c["epochs"] / 60                        # minutes for one full-data run
ratio = spe.get("tiny", spe["resnet18"]) / spe["resnet18"]
print(f"{df['hardware'].iloc[0]}: ResNet-18 {spe['resnet18']:.2f} s/epoch -> one full run F = {F:.1f} min; "
      f"tiny CNN {spe.get('tiny', float('nan')):.2f} s/epoch (r = {ratio:.2f})")
print(f"\n{'experiment':11s} {'jobs':>5s} {'full runs':>10s} {'GPU-hours':>10s} {'wall time on ' + str(a.gpus) + ' GPUs':>20s}")
total = 0.0
for name in EXPERIMENTS:
    if name == "check":
        continue
    jobs = build(name, c)
    fr = 0.0
    for j in jobs:
        epochs = j["train_epochs"] if j["kind"] == "score" else j["epochs"] * j["data_fraction"]
        fr += epochs / c["epochs"] * (ratio if j["model"] == "tiny" else 1.0)
    hours = fr * F / 60
    total += hours
    print(f"{name:11s} {len(jobs):5d} {fr:10.1f} {hours:10.2f} {hours / a.gpus * 60:17.0f} min")
print(f"{'total C-10':11s} {'':5s} {'':10s} {total:10.2f} {total / a.gpus * 60:17.0f} min"
      "   (CIFAR-100 main settings: roughly another 40% of this)")

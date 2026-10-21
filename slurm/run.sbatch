#!/bin/bash
# Run one experiment on one DGX node, one process per GPU, each with its own share of the jobs.
#   sbatch slurm/run.sbatch baselines
#   sbatch slurm/run.sbatch main --dataset cifar100
#   sbatch --gres=gpu:8 --time=01:00:00 slurm/run.sbatch epochs      # more GPUs / other time limit
# Everything after the experiment name is passed on to run.py.
# If the job dies or hits its time limit, submit the same command again: finished jobs are skipped.
#SBATCH --job-name=datadiet
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
if [ $# -lt 1 ]; then echo "usage: sbatch slurm/run.sbatch <experiment> [run.py options]"; exit 1; fi
EXP=$1; shift
cd "$SLURM_SUBMIT_DIR"
source ~/venvs/datadiet/bin/activate
mkdir -p logs

# SLURM tells us which GPUs we got, e.g. CUDA_VISIBLE_DEVICES=2,3,5,6
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
N=${#GPUS[@]}
echo "== $EXP $* | node $(hostname) | $N GPUs: ${GPUS[*]} | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
# plan first: stops here with a clear error if, e.g., score files are missing
python run.py "$EXP" "$@" --dry-run --num-shards "$N" > "logs/${EXP}-${SLURM_JOB_ID}-plan.txt"
head -1 "logs/${EXP}-${SLURM_JOB_ID}-plan.txt"

# one process per GPU; each writes its own log file and its own CSV
PIDS=()
for k in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$k]} python run.py "$EXP" "$@" --shard "$k" --num-shards "$N" \
    > "logs/${EXP}-${SLURM_JOB_ID}-shard${k}.log" 2>&1 &
  PIDS+=($!)
done

# wait for each process and report any that crashed
FAILED=0
for k in "${!PIDS[@]}"; do
  if wait "${PIDS[$k]}"; then echo "shard $k finished"; else echo "shard $k FAILED -- see logs/${EXP}-${SLURM_JOB_ID}-shard${k}.log"; FAILED=1; fi
done
echo "== done $(date)"
exit $FAILED

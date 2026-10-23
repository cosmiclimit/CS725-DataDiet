#!/bin/bash
# Run SEVERAL experiments one after the other inside ONE job (one process per GPU for each experiment).
# Useful because Prajna limits how many jobs one user may have in the queue (5 on the l40 partition).
#   sbatch slurm/run_many.sbatch epochs forgetting variants distill
#   sbatch slurm/run_many.sbatch baselines scores main cheap --dataset cifar100
# Experiment names come first; everything from the first --option on is passed to run.py.
# If the job dies, submit the same command again: finished runs are skipped.
#SBATCH --job-name=many
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
EXPS=()
while [ $# -gt 0 ] && [[ "$1" != --* ]]; do EXPS+=("$1"); shift; done
if [ ${#EXPS[@]} -eq 0 ]; then echo "usage: sbatch slurm/run_many.sbatch <experiment> [<experiment> ...] [run.py options]"; exit 1; fi
cd "$SLURM_SUBMIT_DIR"
source ~/venvs/datadiet/bin/activate
mkdir -p logs

IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
N=${#GPUS[@]}
echo "== experiments: ${EXPS[*]} | options: $* | node $(hostname) | $N GPUs: ${GPUS[*]} | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

FAILED=0
for EXP in "${EXPS[@]}"; do
  echo "== start $EXP $(date)"
  if ! python run.py "$EXP" "$@" --dry-run --num-shards "$N" > "logs/${EXP}-${SLURM_JOB_ID}-plan.txt"; then
    echo "$EXP: plan failed -- see logs/${EXP}-${SLURM_JOB_ID}-plan.txt"; FAILED=1; continue
  fi
  head -1 "logs/${EXP}-${SLURM_JOB_ID}-plan.txt"
  PIDS=()
  for k in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$k]} python run.py "$EXP" "$@" --shard "$k" --num-shards "$N" \
      > "logs/${EXP}-${SLURM_JOB_ID}-shard${k}.log" 2>&1 &
    PIDS+=($!)
  done
  for k in "${!PIDS[@]}"; do
    if wait "${PIDS[$k]}"; then echo "$EXP shard $k finished"
    else echo "$EXP shard $k FAILED -- see logs/${EXP}-${SLURM_JOB_ID}-shard${k}.log"; FAILED=1; fi
  done
done
echo "== done $(date)"
exit $FAILED

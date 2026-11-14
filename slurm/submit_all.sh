#!/bin/bash
# Submit a whole dataset's experiments at once, in the right order.
#   bash slurm/submit_all.sh                         # CIFAR-10: all experiments
#   bash slurm/submit_all.sh cifar100                # CIFAR-100: the main settings
#   bash slurm/submit_all.sh cifar10 main cheap      # only the experiments you name
#
# 'baselines' and 'scores' start straight away; every experiment that needs scores waits for the
# 'scores' job (if it is submitted in the same call) and is cancelled automatically if scoring fails.
# Prajna lets 4 of your jobs run at the same time; the rest wait in the queue.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${1:-cifar10}
shift || true
if [ $# -gt 0 ]; then
  EXPS=("$@")
elif [ "$DATASET" = "cifar100" ]; then
  EXPS=(baselines scores main cheap)
else
  EXPS=(baselines scores main cheap epochs forgetting variants distill)
fi

# time limits for 4 A100s (generous; check estimate.py after the GPU check and lower them if you like)
declare -A TIME=([baselines]=02:00:00 [scores]=01:00:00 [main]=02:00:00 [cheap]=02:00:00
                 [epochs]=03:00:00 [forgetting]=02:00:00 [variants]=01:00:00 [distill]=01:00:00)
NEEDS_SCORES=" main cheap epochs forgetting variants distill "

SCORES_JOB=""
for EXP in "${EXPS[@]}"; do
  [ "$EXP" = "scores" ] || continue
  SCORES_JOB=$(sbatch --parsable --job-name="scores-${DATASET}" --time="${TIME[scores]}" \
               slurm/run.sbatch scores --dataset "$DATASET")
  echo "submitted scores      -> job $SCORES_JOB"
done

for EXP in "${EXPS[@]}"; do
  [ "$EXP" = "scores" ] && continue
  if [ -z "${TIME[$EXP]+x}" ]; then echo "unknown experiment '$EXP' -- skipped"; continue; fi
  DEP=()
  if [[ "$NEEDS_SCORES" == *" $EXP "* && -n "$SCORES_JOB" ]]; then
    DEP=(--dependency="afterok:${SCORES_JOB}" --kill-on-invalid-dep=yes)
  fi
  JOB=$(sbatch --parsable --job-name="${EXP}-${DATASET}" --time="${TIME[$EXP]}" "${DEP[@]}" \
        slurm/run.sbatch "$EXP" --dataset "$DATASET")
  printf "submitted %-11s -> job %s %s\n" "$EXP" "$JOB" "${DEP:+(waits for scores)}"
done
echo "watch them with:  squeue --me"

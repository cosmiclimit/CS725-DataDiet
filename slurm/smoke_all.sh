#!/bin/bash
# Run every experiment in smoke mode (fake data, tiny sizes) and the analysis: a 1-3 minute check that
# all code paths work. Uses the GPU if there is one (e.g. on Colab or Kaggle), otherwise the CPU.
#   bash slurm/smoke_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf results/smoke scores/smoke
for e in baselines scores main cheap epochs forgetting variants distill; do
  echo "==== $e"
  python run.py "$e" --smoke | grep -E "host |->|snapshot|finished|ERROR"
done
python analyze.py --smoke | tail -1
echo "SMOKE TEST PASSED"

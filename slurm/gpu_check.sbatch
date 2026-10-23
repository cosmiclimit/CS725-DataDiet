#!/bin/bash
# First GPU job (about 5 minutes): is the A100 visible to PyTorch, and how long does one epoch take?
# Submit from the repo folder:   sbatch slurm/gpu_check.sbatch
# Output: gpu_check-<jobid>.out in the repo folder.
#SBATCH --job-name=gpu_check
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/venvs/datadiet/bin/activate

echo "== node $(hostname), GPUs given to this job: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU visible:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

echo "== 2-epoch timing runs (ResNet-18 and tiny CNN, all 50,000 images; results/check is not a result)"
rm -rf results/check
python run.py check

echo "== estimated GPU time for every experiment"
python estimate.py

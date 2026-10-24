# CS725 DataDiet: data pruning with EL2N (mid-term)

**Question.** If we score every CIFAR-10 training image by difficulty and train a ResNet-18 on only a
fraction of them, how close do we get to full-data accuracy, and does it save compute once the cost of
scoring is counted?

This is the mid-term state of the project: **CIFAR-10 only, three experiments** (`baselines`, `scores`,
`main`). Slides: [slides/midterm_slides.pdf](slides/midterm_slides.pdf). All tables and figures:
[results/cifar10/analysis/REPORT.md](results/cifar10/analysis/REPORT.md).

## Method in short

- **Score:** EL2N (Paul et al., *Deep Learning on a Data Diet*, NeurIPS 2021) = ‖softmax(f(x)) − y‖₂, the
  distance between the model's output and the one-hot label. Taken from a ResNet-18 at epoch 20 and
  averaged over 3 scorer runs.
- **Easy** = low EL2N (the scorer is confident and right). **Hard** = high EL2N (the scorer is far from the
  right answer; the very hardest are often ambiguous or mislabelled).
- **Strategies:** keep hard, keep easy, or random, at 90 / 70 / 50 / 30 % of the 50,000 training images.
- **Baselines:** full data; random pruning; and the *short schedule* (all data, fewer epochs, same compute
  as pruning).
- **Training (every run):** ResNet-18, SGD + Nesterov, lr 0.1, batch 128, weight decay 5e-4, cosine
  schedule, crop + flip, mixed precision, 60 epochs on the kept subset (equal epochs), seeds 0 / 1 / 2.
  NVIDIA L40S on IIT Bombay's Prajna cluster.

## Experiments

| Name | What | Runs |
|---|---|---|
| `baselines` | full data (60 epochs), random pruning at 90/70/50/30 %, short schedule (54/42/30/18 epochs) | 27 |
| `scores` | 3 ResNet-18 scorer runs, EL2N snapshots at epochs 1, 5 and 20 | 3 scorer runs |
| `main` | keep hard / keep easy with ResNet-18 epoch-20 EL2N at 90/70/50/30 % | 24 |

## Mid-term results (test accuracy %, mean over 3 seeds)

Full data: **94.32 ± 0.38**, one full run F = 2.38 min.

| Kept | Random | Short schedule (same compute) | **Keep hard** | Keep easy | Hard vs random (Welch p) |
|---|---|---|---|---|---|
| 90 % | 93.84 | 94.22 | **94.16** | 92.38 | +0.33 (0.035) |
| 70 % | 92.94 | 93.99 | **94.09** | 89.50 | +1.15 (0.007) |
| 50 % | 90.97 | 92.92 | **92.63** | 85.53 | +1.65 (0.002) |
| 30 % | 86.42 | 91.29 | 80.15 | 78.45 | −6.27 (< 0.001) |

- Keep-hard beats random at 50–90 %; the hardest 70 % is within 0.23 points of full data.
- Keep-hard collapses at 30 %: the hardest images include many ambiguous or mislabelled ones, and the
  kept classes become very unbalanced (666 cars vs 2,790 cats, against 1,500 for a balanced subset).
- Keep-easy is worse than random everywhere.
- The short schedule beats random pruning at every compute budget.
- **Cost:** three ResNet-18 scorers cost S = 1.16 F, so keep-hard costs 145–200 % of one full run. Pruning
  with equal epochs saves compute only if f < 1 − S/F. This is why the next step is cheaper scorers.

## Reproduce

```bash
pip install -r requirements.txt
python tests/test_guide_examples.py          # code vs worked examples (CPU, seconds)
python download_data.py                      # CIFAR-10/100 to ~/datadiet_data
python run.py baselines --dry-run            # the plan of an experiment, nothing trained
sbatch slurm/gpu_check.sbatch                # Prajna: GPU check + timing; then `python estimate.py`
sbatch slurm/run_many.sbatch baselines scores main   # Prajna: the three mid-term experiments in one job
python analyze.py --images                   # tables, figures and REPORT.md in results/cifar10/analysis/
```

## Files

| File | Purpose |
|---|---|
| `download_data.py`, `data.py` | CIFAR download; GPU-resident data loading with crop + flip |
| `models.py`, `train.py` | ResNet-18 (CIFAR version) and tiny CNN; the training loop with timing |
| `scores.py` | EL2N and forgetting scores, score files in `scores/cifar10/` |
| `prune.py`, `experiments.py`, `run.py` | keep hard / easy / random; experiment definitions; the main program |
| `analyze.py`, `estimate.py` | tables, figures, significance tests; GPU-time estimate |
| `slurm/` | Prajna job scripts |
| `results/cifar10/` | raw results (`train_*_shard*.csv`, `scoring_shard*.csv`) and `analysis/` |

## After the mid-term

From the proposal: a cheap tiny-CNN scorer, early scores (epochs 1 and 5), the forgetting score,
class-balanced and offset keep-hard, distillation from the scorers, CIFAR-100 and the full cost table.
New ideas (Review 2): an ensemble of cheap scorers, middle-band / coverage selection (CCS, Moderate
coreset), training-dynamics scores (AUM), a scorer trained on a subset, dynamic pruning (InfoBatch), and
200-epoch training.

## Team

| Member | Role |
|---|---|
| Rahul Paul | training pipeline and baselines; ran the experiments on Prajna |
| Ayan | scoring (EL2N, score files, tests) |
| Banibrata Manna | pruning grid (strategies, experiments, hard / easy / random) |
| Tejas Itankar | analysis and presentation (tables, figures, significance tests, slides) |

# CS725 DataDiet: cheap scorers for data pruning

**Question.** If we score every CIFAR training image by difficulty and train a ResNet-18 on only a
fraction of them, how close do we get to full-data accuracy? Can a small, cheaply trained model do the
scoring, how early can it decide, and does pruning save compute once the cost of scoring is counted?

Static data pruning with EL2N (Paul et al. 2021) and forgetting scores (Toneva et al. 2019): keep hard /
easy / random at 90 / 70 / 50 / 30 % of CIFAR-10 and CIFAR-100, with a tiny-CNN or ResNet-18 scorer at
epochs 1, 5 and 20, and a cost table that counts scoring + training against full training and an
equal-cost short schedule.

- **Final slides:** [slides/DataDiet Final Presentation.pptx](slides/DataDiet%20Final%20Presentation.pptx)
- **Mid-term slides:** [slides/DataDiet Mid-Term Presentation.pptx](slides/DataDiet%20Mid-Term%20Presentation.pptx)
- **All tables and figures:** [CIFAR-10 report](results/cifar10/analysis/REPORT.md) ·
  [CIFAR-100 report](results/cifar100/analysis/REPORT.md)

## Method in short

- **Score:** EL2N (Paul et al., *Deep Learning on a Data Diet*, NeurIPS 2021) = ‖softmax(f(x)) − y‖₂, the
  distance between the model's output and the one-hot label, averaged over 3 scorer runs. Also the
  forgetting score (Toneva et al. 2019): how often an image goes from correct to wrong during training.
- **Scorers:** the reference ResNet-18, and a cheap tiny CNN. Snapshots at epochs 1, 5 and 20.
- **Easy** = low EL2N (the scorer is confident and right). **Hard** = high EL2N (the scorer is far from the
  right answer; the very hardest are often ambiguous or mislabelled).
- **Strategies:** keep hard, keep easy, or random, at 90 / 70 / 50 / 30 % of the training images.
- **Baselines:** full data; random pruning; and the *short schedule* (all data, fewer epochs, same compute
  as pruning).
- **Training (every run):** ResNet-18, SGD + Nesterov, lr 0.1, batch 128, weight decay 5e-4, cosine
  schedule, crop + flip, mixed precision, 60 epochs on the kept subset (equal epochs), seeds 0 / 1 / 2.
  NVIDIA L40S on IIT Bombay's Prajna cluster.

## Experiments

| Name | What | Runs |
|---|---|---|
| `baselines` | full data (60 epochs), random pruning at 90/70/50/30 %, short schedule (54/42/30/18 epochs) | 27 |
| `scores` | tiny CNN + ResNet-18 scorers × 3 seeds; EL2N + forgetting at epochs 1 / 5 / 20 | 6 scorer runs |
| `main` | keep hard / keep easy with ResNet-18 epoch-20 EL2N (reference scores) | 24 |
| `cheap` | keep hard / keep easy with tiny-CNN epoch-20 EL2N | 24 |
| `epochs` | keep hard with epoch-1 and epoch-5 scores (both scorers) | 48 |
| `forgetting` | keep hard / easy with ResNet-18 forgetting scores | 24 |
| `variants` | class-balanced and offset (skip the hardest 5 %) keep-hard | 12 |
| `distill` | pruned training with distillation from the scorers | 12 |

CIFAR-10 ran every experiment (171 training runs). CIFAR-100 ran `baselines`, `scores`, `main` and
`cheap` (75 training runs).

## Mid-term results (CIFAR-10, test accuracy %, mean over 3 seeds)

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
  with equal epochs saves compute only if f < 1 − S/F. This is why the next step was cheaper scorers.

## Final results: CIFAR-10

### Cheap and early scorers (keep hard, accuracy %; cost = scoring + training as % of one full run)

| Scorer | S / F | 90 % | 70 % | 50 % | 30 % |
|---|---|---|---|---|---|
| ResNet-18, epoch 20 (reference) | 1.16 | 94.16 (200 %) | 94.09 (181 %) | 92.63 (163 %) | 80.15 (145 %) |
| ResNet-18, epoch 5 | 0.50 | 94.43 (132 %) | 93.93 (113 %) | 92.29 (95 %) | 74.40 (78 %) |
| ResNet-18, epoch 1 | 0.40 | 93.94 (122 %) | 92.61 (102 %) | 87.23 (85 %) | 66.73 (67 %) |
| Tiny CNN, epoch 20 | 0.30 | 94.42 (111 %) | 93.84 (94 %) | 91.03 (75 %) | 74.70 (58 %) |
| Tiny CNN, epoch 5 | 0.11 | 94.39 (93 %) | 93.51 (73 %) | 91.17 (56 %) | 71.02 (39 %) |
| Tiny CNN, epoch 1 | 0.08 | 94.22 (89 %) | 92.78 (70 %) | 88.37 (53 %) | 70.52 (35 %) |
| *Random* | 0 | 93.84 (92 %) | 92.94 (76 %) | 90.97 (48 %) | 86.42 (30 %) |
| *Short schedule* | 0 | 94.22 (79 %) | 93.99 (73 %) | 92.92 (43 %) | 91.29 (26 %) |

- A tiny CNN cuts scoring cost by 4× (epoch 20) to 15× (epoch 1). Its epoch-20 ranking agrees well with
  the reference (Spearman 0.70; 76 % of the reference's hardest 50 % are kept, vs 50 % by chance).
- Cheap keep-hard is the first pruning that is actually cheaper than full data: tiny CNN epoch 5 at 90 %
  gives 94.39 % for 93 % of the compute, and beats random at 90 % (p = 0.03).
- Very early scores (epoch 1) are close to random at 70–90 % and much worse at 30–50 %.
- **The short schedule is still the safest way to spend a fixed compute budget:** for less compute it is
  within about 0.2 points of the best keep-hard run at 50–90 %, and far better at 30 %.

### Forgetting, variants and distillation (keep hard, accuracy %)

| Setting | 90 % | 70 % | 50 % | 30 % |
|---|---|---|---|---|
| EL2N (reference) | 94.16 | 94.09 | 92.63 | 80.15 |
| Forgetting (ResNet-18, epoch 20) | 94.16 | 93.88 | **93.11** | 81.81 ± 5.65 |
| Class-balanced EL2N | | | 92.37 | 80.98 |
| Offset EL2N (skip hardest 5 %) | | | 92.36 | 85.76 |
| EL2N + distillation | | | **92.92** | **90.44** |
| Random + distillation | | | 91.08 | 89.09 |

- Forgetting is the best score at 50 % (+2.13 over random, p = 0.001) but very unstable at 30 %.
- At 30 % the problem is mainly the noisy hardest images, not class balance: skipping the hardest 5 %
  helps (+5.6), balancing the classes barely does (+0.8).
- Distillation from the scorer rescues keep-hard at 30 % (80.15 → 90.44, +4.0 over random), but it
  needs the expensive ResNet-18 teacher, so the cost stays at ~150 % of a full run.

## Final results: CIFAR-100

Full data: **76.34 ± 0.16**, F = 2.17 min. Scoring is relatively more expensive here: S / F = 2.52 for the
ResNet-18 epoch-20 scorers and 0.77 for the tiny CNN.

| Kept | Random | Short schedule | Keep hard (ResNet-18) | Keep hard (tiny CNN) | Keep easy (ResNet-18) |
|---|---|---|---|---|---|
| 90 % | 75.17 | 75.94 | 75.81 | **76.34** | 75.13 |
| 70 % | 72.33 | 75.51 | 72.53 | 72.94 | 69.89 |
| 50 % | 68.26 | 74.14 | 60.57 | 62.03 | 64.80 |
| 30 % | 58.07 | 72.28 | 31.06 ± 4.95 | 37.77 | 54.16 |

- With 100 classes and only 500 images each, keep-hard breaks down much earlier: it is no better than
  random at 70–90 % and much worse at 30–50 % (some classes keep as few as 18 images at 30 %).
- The tiny CNN is as good as or better than the ResNet-18 scorer at every fraction, at 30 % of its cost.
- No pruning setting beats the short schedule; with scoring counted, every keep-hard run costs more than
  one full run (108–343 %).

## Conclusions

1. A cheap tiny-CNN scorer ranks images almost as well as a ResNet-18 at a fraction of the cost, and
   early snapshots (epoch 5) are nearly as useful as late ones.
2. Keep-hard helps on CIFAR-10 at moderate pruning (50–90 %), but fails at aggressive pruning and on the
   harder, smaller-per-class CIFAR-100.
3. Once scoring cost is counted honestly, training on all data for fewer epochs (the short schedule) is
   the strongest baseline at every compute budget we tried.

## Reproduce

```bash
pip install -r requirements.txt
python tests/test_guide_examples.py          # code vs worked examples (CPU, seconds)
python run.py --list                         # the experiments
python run.py baselines --dry-run            # the plan of an experiment, nothing trained
python run.py baselines --smoke              # tiny CPU run on fake data
bash slurm/smoke_all.sh                      # every experiment on fake data
python download_data.py                      # CIFAR-10/100 to ~/datadiet_data
sbatch slurm/gpu_check.sbatch                # Prajna: GPU check + timing; then `python estimate.py`
bash slurm/submit_all.sh                     # Prajna: every CIFAR-10 experiment, in order
bash slurm/submit_all.sh cifar100            # Prajna: the CIFAR-100 settings
python analyze.py --dataset cifar10 --images # tables, figures and REPORT.md in results/cifar10/analysis/
python analyze.py --dataset cifar100 --images
```

## Files

| File | Purpose |
|---|---|
| `download_data.py`, `data.py` | CIFAR download; GPU-resident data loading with crop + flip |
| `models.py`, `train.py` | ResNet-18 (CIFAR version) and tiny CNN; the training loop with timing |
| `scores.py` | EL2N and forgetting scores |
| `prune.py`, `experiments.py`, `run.py` | keep hard / easy / random; experiment definitions; the main program |
| `analyze.py`, `estimate.py` | tables, figures, significance tests; GPU-time estimate |
| `slurm/` | Prajna job scripts, `submit_all.sh` and `smoke_all.sh` |
| `scores/<dataset>/` | score files `<model>_seed<s>_e<epoch>.npz` (teacher logits are not committed) |
| `results/<dataset>/` | raw results (`train_*_shard*.csv`, `scoring_shard*.csv`) and `analysis/` |
| `slides/` | mid-term and final presentations |

## Team

| Member | Role |
|---|---|
| Rahul Paul | training pipeline and baselines; ran the experiments on Prajna |
| Ayan Barman | scoring (EL2N, forgetting, score files, tests) |
| Banibrata Manna | pruning grid (strategies, experiments, hard / easy / random, variants) |
| Tejas Itankar | analysis and presentation (tables, figures, significance tests, slides) |

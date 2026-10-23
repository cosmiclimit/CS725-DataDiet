"""Train one model and measure its test accuracy and training time.

Recipe (identical for every run in the project, guide Ch. 12.5): SGD with Nesterov momentum 0.9,
lr 0.1, batch 128, weight decay 5e-4, cosine learning-rate decay, random crop + flip,
mixed precision (float16) on the GPU.

Extras used by the scoring runs and the bonus experiment:
  schedule_epochs  - length of the cosine schedule. A scorer can follow a 60-epoch schedule but
                     stop after 20 epochs ("early in a normal training run", as in Paul et al.).
  track_forgetting - count forgetting events (right -> wrong) for every training image.
  on_epoch_end     - called after each epoch, e.g. to compute EL2N scores; its time is NOT
                     counted as training time (it is measured separately by the caller).
  soft_targets     - teacher logits for every training image: knowledge-distillation loss (11.1).
"""
import math
import time
import torch
import torch.nn.functional as F


def sync(device):
    """GPU work is asynchronous: wait for it to finish before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def evaluate(model, data, bs=1000):
    """Accuracy on the full `data` set (no augmentation)."""
    model.eval()
    device = data.y.device
    correct = 0
    for s in range(0, len(data), bs):
        x, y = data.batch(torch.arange(s, min(s + bs, len(data)), device=device))
        with autocast(device):
            out = model(x.contiguous(memory_format=torch.channels_last))
        correct += (out.argmax(1) == y).sum().item()
    return correct / len(data)


def update_forgetting(prev, forgets, learned, idx, hit):
    """One observation of the images `idx` (unique): a forgetting event is correct -> incorrect."""
    forgets[idx] += (prev[idx] & ~hit).int()
    prev[idx] = hit
    learned[idx] |= hit


def kd_loss(student_logits, teacher_logits, y, T, alpha):
    """(1 - alpha) * cross-entropy + alpha * T^2 * KL(teacher_T || student_T), guide eq. (11.1)."""
    ce = F.cross_entropy(student_logits, y)
    kl = F.kl_div(F.log_softmax(student_logits.float() / T, dim=1),
                  F.softmax(teacher_logits.float() / T, dim=1), reduction="batchmean")
    return (1 - alpha) * ce + alpha * T * T * kl


def train(model, train_data, test_data, epochs, indices=None, seed=0, lr=0.1, bs=128, wd=5e-4,
          schedule_epochs=None, track_forgetting=False, on_epoch_end=None,
          soft_targets=None, kd_T=4.0, kd_alpha=0.5, evaluate_at_end=True, log_every=10):
    """Train `model` for `epochs` on `train_data` (only the examples in `indices`, if given).

    Returns a dict: test_acc, train_seconds (training loop only, GPU-synchronised),
    epoch_seconds (cumulative training time after each epoch), final train loss / accuracy,
    and, if track_forgetting, the forgetting state (counts, learned) as GPU tensors."""
    device = train_data.y.device
    use_amp = device.type == "cuda"
    idx_all = (torch.arange(len(train_data), device=device) if indices is None
               else torch.as_tensor(indices, device=device, dtype=torch.long))
    n = len(idx_all)
    steps = math.ceil(n / bs)                      # batches per epoch (last one may be smaller)
    schedule_epochs = schedule_epochs or epochs

    model = model.to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=schedule_epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)                          # controls data order and augmentation

    N = len(train_data)
    if track_forgetting:
        prev = torch.zeros(N, dtype=torch.bool, device=device)      # correct the last time seen?
        forgets = torch.zeros(N, dtype=torch.int32, device=device)  # number of right -> wrong flips
        learned = torch.zeros(N, dtype=torch.bool, device=device)   # ever classified correctly?

    train_seconds, epoch_seconds = 0.0, []
    for ep in range(1, epochs + 1):
        sync(device)
        t0 = time.time()
        model.train()
        perm = idx_all[torch.randperm(n, generator=gen, device=device)]   # reshuffle each epoch
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for s in range(steps):
            b = perm[s * bs:(s + 1) * bs]
            x, y = train_data.batch(b, augment=True, gen=gen)
            with autocast(device):
                out = model(x.contiguous(memory_format=torch.channels_last))
                if soft_targets is None:
                    loss = F.cross_entropy(out, y)
                else:
                    loss = kd_loss(out, soft_targets[b], y, kd_T, kd_alpha)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            hit = out.argmax(1) == y
            loss_sum += loss.detach() * len(y)     # kept on the GPU: no sync inside the loop
            correct += hit.sum()
            if track_forgetting:                   # indices in a batch are unique
                update_forgetting(prev, forgets, learned, b, hit)
        sync(device)
        train_seconds += time.time() - t0
        epoch_seconds.append(train_seconds)
        if ep % log_every == 0 or ep == epochs:
            print(f"  epoch {ep:3d}/{epochs}  train loss {loss_sum.item() / n:.3f}  "
                  f"train acc {correct.item() / n:.3f}  {train_seconds:6.0f}s", flush=True)
        if on_epoch_end is not None:               # e.g. scoring; not counted as training time
            state = (forgets, learned) if track_forgetting else None
            on_epoch_end(ep, model, train_seconds, state)

    out = {"train_seconds": train_seconds, "epoch_seconds": epoch_seconds,
           "train_loss": loss_sum.item() / n, "train_acc": correct.item() / n}
    if evaluate_at_end:
        out["test_acc"] = evaluate(model, test_data)
    if track_forgetting:
        out["forgetting_state"] = (forgets, learned)
    return out

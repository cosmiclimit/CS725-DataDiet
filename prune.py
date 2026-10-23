"""Choose which training examples to keep (guide Ch. 5.6 and 6)."""
import numpy as np


def select(n, fraction, strategy, scores=None, seed=0, labels=None, class_balanced=False, offset=0.0):
    """Return the sorted indices of the examples to keep.

    n         : dataset size.        fraction : share to keep, in (0, 1].
    strategy  : 'random', 'hard' (highest scores) or 'easy' (lowest scores).
    class_balanced : rank within each class and keep `fraction` of every class (needs labels).
    offset    : keep-hard only -- first skip this share of the very hardest (likely mislabelled).

    The random subset depends only on (n, fraction, seed), so every experiment that asks for
    the same random subset gets exactly the same images. Ties in the scores (common for the
    forgetting score) are broken randomly with the same seed, not by image order."""
    rng = np.random.default_rng(seed)

    def pick(idx):
        k = int(round(fraction * len(idx)))
        if strategy == "random":
            return rng.choice(idx, k, replace=False)
        s = np.asarray(scores)[idx]
        order = idx[np.lexsort((rng.random(len(idx)), -s))]      # hardest first, random tie-break
        if strategy == "hard":
            skip = int(round(offset * len(idx)))
            return order[skip:skip + k]
        if strategy == "easy":
            return order[::-1][:k]
        raise ValueError(f"unknown strategy {strategy!r}")

    if class_balanced:
        labels = np.asarray(labels)
        kept = [pick(np.flatnonzero(labels == c)) for c in np.unique(labels)]
        return np.sort(np.concatenate(kept))
    return np.sort(pick(np.arange(n)))

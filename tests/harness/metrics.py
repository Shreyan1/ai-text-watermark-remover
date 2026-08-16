"""Detection metrics — pure Python, no numpy.

The two the paper reports: AUROC and TPR at a fixed FPR (they use FPR=1%).
"""

from __future__ import annotations


def auroc(positives: list[float], negatives: list[float]) -> float:
    """Area under ROC via the Mann-Whitney U statistic (rank-based). 1.0 = perfect
    separation, 0.5 = chance."""
    if not positives or not negatives:
        return float("nan")
    combined = [(s, 1) for s in positives] + [(s, 0) for s in negatives]
    combined.sort(key=lambda t: t[0])
    # Assign average ranks (handle ties).
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl == 1)
    n_pos, n_neg = len(positives), len(negatives)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def tpr_at_fpr(positives: list[float], negatives: list[float], fpr: float = 0.01) -> float:
    """True-positive rate when the threshold is set to allow at most `fpr` false
    positives on the negatives — the paper's headline TPR@FPR=1%."""
    if not positives or not negatives:
        return float("nan")
    neg_sorted = sorted(negatives, reverse=True)
    idx = max(0, min(len(neg_sorted) - 1, int(fpr * len(neg_sorted))))
    threshold = neg_sorted[idx]
    return sum(1 for s in positives if s > threshold) / len(positives)


def summary(positives: list[float], negatives: list[float]) -> dict:
    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "mean_pos": _mean(positives),
        "mean_neg": _mean(negatives),
        "auroc": auroc(positives, negatives),
        "tpr@fpr=1%": tpr_at_fpr(positives, negatives, 0.01),
        "n_pos": len(positives),
        "n_neg": len(negatives),
    }

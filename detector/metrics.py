import numpy as np


def compute_eer(labels, ai_scores):
    """Equal error rate and the P(AI) threshold where it occurs.

    labels: 0 = human, 1 = AI.  Predict AI when score >= threshold.
    """
    labels = np.asarray(labels)
    scores = np.asarray(ai_scores, dtype=np.float64)
    n_ai, n_human = (labels == 1).sum(), (labels == 0).sum()
    if n_ai == 0 or n_human == 0:
        return float("nan"), 0.5

    order = np.argsort(scores)
    s, l = scores[order], labels[order]
    # Threshold = s[i]: everything before i is called human.
    miss = np.concatenate([[0], np.cumsum(l == 1)]) / n_ai               # AI called human
    false_alarm = 1 - np.concatenate([[0], np.cumsum(l == 0)]) / n_human  # human called AI
    i = int(np.argmin(np.abs(miss - false_alarm)))
    threshold = s[i] if i < len(s) else s[-1] + 1e-6
    return float((miss[i] + false_alarm[i]) / 2), float(threshold)


def summarize(labels, ai_scores, threshold):
    labels = np.asarray(labels)
    pred = np.asarray(ai_scores) >= threshold
    eer, eer_thr = compute_eer(labels, ai_scores)
    human_acc = (~pred[labels == 0]).mean() if (labels == 0).any() else float("nan")
    ai_acc = pred[labels == 1].mean() if (labels == 1).any() else float("nan")
    return {
        "n": len(labels),
        "eer": eer,
        "eer_threshold": eer_thr,
        "accuracy": float((pred == labels).mean()),
        "human_recall": float(human_acc),
        "ai_recall": float(ai_acc),
    }

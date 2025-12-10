# src/metrics.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, roc_curve
)

def auc_with_flip(labels: np.ndarray, probs_pos: np.ndarray):
    """
    Computes ROC AUC in normal and flipped directions; returns the best.
    Returns:
      auc (float), use_flipped (bool), scores_for_auc (np.ndarray), fpr (np.ndarray), tpr (np.ndarray)
    """
    labels = np.asarray(labels)
    probs_pos = np.asarray(probs_pos)

    try:
        auc_as_is = roc_auc_score(labels, probs_pos)
        auc_flipped = roc_auc_score(labels, 1.0 - probs_pos)
        use_flipped = auc_flipped > auc_as_is
        scores_for_auc = (1.0 - probs_pos) if use_flipped else probs_pos
        auc = max(auc_as_is, auc_flipped)
        fpr, tpr, _ = roc_curve(labels, scores_for_auc)
    except ValueError:
        auc, use_flipped = float("nan"), False
        scores_for_auc = np.array([])
        fpr, tpr = np.array([]), np.array([])

    return auc, use_flipped, scores_for_auc, fpr, tpr


def tune_threshold(labels: np.ndarray, probs_pos: np.ndarray, steps: int = 101) -> Tuple[float, float, float, float]:
    """
    Sweeps thresholds in [0,1] to maximize F1. Returns (best_thr, precision, recall, f1).
    """
    labels = np.asarray(labels)
    probs_pos = np.asarray(probs_pos)
    if len(np.unique(labels)) < 2:
        return 0.5, float("nan"), float("nan"), float("nan")

    thresholds = np.linspace(0.0, 1.0, steps)
    best_idx, best_f1 = 0, -1.0
    for i, t in enumerate(thresholds):
        preds = (probs_pos >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_idx = f1, i

    best_thr = float(thresholds[best_idx])
    preds_thr = (probs_pos >= best_thr).astype(int)
    precision = precision_score(labels, preds_thr, zero_division=0)
    recall = recall_score(labels, preds_thr, zero_division=0)
    f1 = f1_score(labels, preds_thr, zero_division=0)
    return best_thr, precision, recall, f1

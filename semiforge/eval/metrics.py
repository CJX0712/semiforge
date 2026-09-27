"""Metrics + benchmark runner. Evaluation is transductive on the unlabeled set."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("E301 accuracy: shape mismatch")
    if y_true.size == 0:
        raise ValueError("E302 accuracy: empty arrays")
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        raise ValueError("E302 macro_f1: empty arrays")
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1s))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred)}

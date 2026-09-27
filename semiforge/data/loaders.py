"""Real-dataset loader (digits subsample) - ships with scikit-learn, offline OK."""

from __future__ import annotations

import numpy as np

from semiforge.core.types import SSLDataset

try:
    from sklearn.datasets import load_digits

    _SKLEARN = True
except ImportError:  # pragma: no cover - sklearn is a hard dep, guard anyway
    _SKLEARN = False


def _mask_labeled(y_true: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    y_labeled = np.full_like(y_true, -1)
    for c in np.unique(y_true):
        idx = np.flatnonzero(y_true == c)
        rng.shuffle(idx)
        k = max(1, round(frac * idx.size))
        y_labeled[idx[:k]] = c
    return y_labeled


def available_digits() -> bool:
    return _SKLEARN


def load_digits_ssl(n: int = 500, frac_labeled: float = 0.10, seed: int = 42) -> SSLDataset:
    if not _SKLEARN:
        raise RuntimeError("E106 scikit-learn not available for digits loader")
    X, y = load_digits(return_X_y=True)
    # stratified subsample
    rng = np.random.default_rng(seed)
    take: list[np.ndarray] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        take.append(idx[: max(1, n // 10)])
    idx = np.concatenate(take)
    rng.shuffle(idx)
    X, y = X[idx].astype(float), y[idx].astype(int)
    y_labeled = _mask_labeled(y, frac_labeled, rng)
    # 0-1 scaling per feature keeps kNN graphs meaningful
    X = X / 16.0
    return SSLDataset(X, y_labeled, y, name="digits", meta={"frac_labeled": frac_labeled})

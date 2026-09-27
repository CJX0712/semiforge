"""Synthetic semi-supervised datasets with controlled difficulty."""

from __future__ import annotations

import numpy as np

from semiforge.core.types import SSLDataset


def _mask_labeled(y_true: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly keep `frac` of each class labeled; guarantee >=1 per class."""
    y_labeled = np.full_like(y_true, -1)
    for c in np.unique(y_true):
        idx = np.flatnonzero(y_true == c)
        rng.shuffle(idx)
        k = max(1, round(frac * idx.size))
        y_labeled[idx[:k]] = c
    return y_labeled


def make_moons(
    n: int = 600, frac_labeled: float = 0.05, noise: float = 0.22, seed: int = 42
) -> SSLDataset:
    """Interleaving half-moons (classic SSL benchmark, Rastmoor & Brann 2005)."""
    rng = np.random.default_rng(seed)
    n_out = n // 2
    t = rng.uniform(0, np.pi, n_out)
    upper = np.stack([np.cos(t), np.sin(t)], axis=1)
    lower = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.vstack([upper, lower]) + rng.normal(0, noise, (2 * n_out, 2))
    y = np.concatenate([np.zeros(n_out, int), np.ones(n_out, int)])
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    y_labeled = _mask_labeled(y, frac_labeled, rng)
    return SSLDataset(
        X, y_labeled, y, name="moons", meta={"frac_labeled": frac_labeled, "noise": noise}
    )


def make_blobs(
    n: int = 600,
    frac_labeled: float = 0.05,
    class_sep: float = 1.6,
    n_classes: int = 3,
    seed: int = 42,
) -> SSLDataset:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, class_sep, (n_classes, 2))
    idx = rng.integers(0, n_classes, n)
    X = centers[idx] + rng.normal(0, 1.0, (n, 2))
    y = idx
    y_labeled = _mask_labeled(y, frac_labeled, rng)
    return SSLDataset(
        X,
        y_labeled,
        y,
        name="blobs",
        meta={"frac_labeled": frac_labeled, "class_sep": class_sep},
    )


def make_circle(
    n: int = 600, frac_labeled: float = 0.05, noise: float = 0.12, seed: int = 42
) -> SSLDataset:
    """Concentric ring vs ball - label propagation's showcase."""
    rng = np.random.default_rng(seed)
    n_out = n // 2
    ang = rng.uniform(0, 2 * np.pi, n_out)
    outer = np.stack([np.cos(ang), np.sin(ang)], axis=1) * 1.4
    inner = rng.normal(0, 0.45, (n - n_out, 2))
    X = np.vstack([outer, inner]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(n_out, int), np.ones(n - n_out, int)])
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    y_labeled = _mask_labeled(y, frac_labeled, rng)
    return SSLDataset(
        X, y_labeled, y, name="circle", meta={"frac_labeled": frac_labeled, "noise": noise}
    )


def make_spiral(
    n: int = 600, frac_labeled: float = 0.05, noise: float = 0.18, seed: int = 42
) -> SSLDataset:
    """Two interleaved spirals (harder, 2-arm)."""
    rng = np.random.default_rng(seed)
    n_arm = n // 2
    t = np.linspace(0.35, 3.0 * np.pi, n_arm)
    arm1 = np.stack([t * np.cos(t), t * np.sin(t)], axis=1) / (3.0 * np.pi)
    arm2 = -arm1
    X = np.vstack([arm1, arm2]) + rng.normal(0, noise, (2 * n_arm, 2))
    y = np.concatenate([np.zeros(n_arm, int), np.ones(n - n_arm, int)])
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    y_labeled = _mask_labeled(y, frac_labeled, rng)
    return SSLDataset(
        X, y_labeled, y, name="spiral", meta={"frac_labeled": frac_labeled, "noise": noise}
    )


def make_highdim(
    n: int = 600, d: int = 20, frac_labeled: float = 0.05, sep: float = 1.2, seed: int = 42
) -> SSLDataset:
    """High-dimensional Gaussian mixture - stresses feature-based methods."""
    rng = np.random.default_rng(seed)
    n_classes = 3
    centers = rng.normal(0, sep, (n_classes, d))
    idx = rng.integers(0, n_classes, n)
    X = centers[idx] + rng.normal(0, 1.0, (n, d))
    y = idx
    y_labeled = _mask_labeled(y, frac_labeled, rng)
    return SSLDataset(
        X, y_labeled, y, name="highdim", meta={"frac_labeled": frac_labeled, "d": d, "sep": sep}
    )


GENERATORS = {
    "moons": make_moons,
    "blobs": make_blobs,
    "circle": make_circle,
    "spiral": make_spiral,
    "highdim": make_highdim,
}

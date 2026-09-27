"""Pure-numpy label propagation (Zhu, Ghahramani & Lafferty 2002).

This is the offline fallback AND the graph prior used by the CBCP-Hybrid
flagship. Deterministic, no sklearn required.
"""

from __future__ import annotations

import numpy as np

from semiforge.core.errors import ModelError


def knn_affinity(X: np.ndarray, k: int = 10, sigma: float | None = None) -> np.ndarray:
    """Symmetric kNN affinity matrix with gaussian kernel weights."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    k = min(k, n - 1)
    sq = (X * X).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    W = np.zeros((n, n))
    # local scaling: sigma = median distance to k-th neighbor per row
    kth = np.sort(d2, axis=1)[:, k - 1]
    if sigma is None:
        sigma = np.sqrt(np.maximum(kth, 1e-12))
    for i in range(n):
        nb = np.argsort(d2[i])[:k]
        w = np.exp(-d2[i, nb] / (sigma[i] * sigma[nb] + 1e-12))
        W[i, nb] = w
    W = np.maximum(W, W.T)  # symmetrize
    np.fill_diagonal(W, 0.0)
    return W


def propagate(
    W: np.ndarray,
    y_labeled: np.ndarray,
    alpha: float = 0.75,
    n_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """Iterative clamped label propagation. Returns soft posteriors F (n, C).

    F <- alpha * S @ F + (1 - alpha) * Y0, with labeled rows clamped to Y0.
    S = D^{-1/2} W D^{-1/2}.
    """
    n = W.shape[0]
    classes = np.unique(y_labeled[y_labeled != -1])
    if classes.size == 0:
        raise ModelError("E202 propagation needs at least one labeled point")
    C = classes.size
    cls_index = {int(c): j for j, c in enumerate(classes)}
    Y0 = np.full((n, C), 1.0 / C)
    for i in range(n):
        c = int(y_labeled[i])
        if c != -1:
            row = np.zeros(C)
            row[cls_index[c]] = 1.0
            Y0[i] = row
    deg = W.sum(axis=1)
    dinv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    S = W * dinv_sqrt[:, None] * dinv_sqrt[None, :]
    F = Y0.copy()
    for _ in range(n_iter):
        F_new = alpha * (S @ F) + (1 - alpha) * Y0
        labeled_rows = y_labeled != -1
        F_new[labeled_rows] = Y0[labeled_rows]
        delta = np.abs(F_new - F).max()
        F = F_new
        if delta < tol:
            break
    # row-normalize unlabeled rows -> proper probability posteriors
    unl = y_labeled == -1
    s = F[unl].sum(axis=1, keepdims=True)
    F[unl] = F[unl] / np.maximum(s, 1e-12)
    return F


def graph_propagate(
    X: np.ndarray, y_labeled: np.ndarray, k: int = 10, alpha: float = 0.75, n_iter: int = 200
) -> np.ndarray:
    W = knn_affinity(X, k=k)
    return propagate(W, y_labeled, alpha=alpha, n_iter=n_iter)

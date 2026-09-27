"""Estimator factories with offline fallback.

Backend probing is done at call time (module-level function) so tests can
monkeypatch `estimators.available_sklearn` reliably.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    _SKLEARN = True
except ImportError:  # pragma: no cover
    _SKLEARN = False


def available_sklearn() -> bool:
    return _SKLEARN


class NumpyLogisticRegression:
    """Pure-numpy multinomial logistic regression (softmax + full-batch Adam).

    Deterministic, zero-dependency fallback. Good enough as a base learner
    and as the supervised-baseline backend when sklearn is unavailable.
    """

    def __init__(
        self, lr: float = 0.1, n_iter: int = 600, l2: float = 1e-3, seed: int = 0
    ) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.seed = seed
        self.classes_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.b_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NumpyLogisticRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        self.classes_ = np.unique(y)
        k = self.classes_.size
        Y = np.zeros((n, k))
        for j, c in enumerate(self.classes_):
            Y[y == c, j] = 1.0
        # standardize
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + 1e-12
        Xs = (X - self.mu_) / self.sd_
        W = np.zeros((d, k))
        b = np.zeros(k)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        mb = np.zeros_like(b)
        vb = np.zeros_like(b)
        for t in range(1, self.n_iter + 1):
            Z = Xs @ W + b
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            G = (P - Y) / n
            gW = Xs.T @ G + self.l2 * W
            gb = G.sum(axis=0)
            mW = 0.9 * mW + 0.1 * gW
            vW = 0.999 * vW + 0.001 * gW * gW
            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb * gb
            mhW = mW / (1 - 0.9**t)
            vhW = vW / (1 - 0.999**t)
            mhb = mb / (1 - 0.9**t)
            vhb = vb / (1 - 0.999**t)
            W -= self.lr * mhW / (np.sqrt(vhW) + 1e-8)
            b -= self.lr * mhb / (np.sqrt(vhb) + 1e-8)
        self.W_, self.b_ = W, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None:
            raise RuntimeError("E201 estimator not fitted")
        Xs = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        Z = Xs @ self.W_ + self.b_
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_rf(seed: int = 0, n_estimators: int = 120):
    """Random forest: sklearn backend, numpy LR fallback."""
    if available_sklearn():
        return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1)
    return NumpyLogisticRegression(seed=seed)


def make_lr(seed: int = 0):
    """Logistic regression: sklearn backend, numpy fallback.

    sklearn 1.9 note: no multi_class arg; lbfgs handles multinomial.
    """
    if available_sklearn():
        return LogisticRegression(solver="lbfgs", max_iter=2000, C=1.0)
    return NumpyLogisticRegression(seed=seed)

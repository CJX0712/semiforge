"""SemiForge 纯 numpy 标签传播 (Tier-1 离线兜底)。

作者: 晨星 (CJX0712)

实现 Zhou 2004 / Bengio 2006 迭代式标签传播 (RBF 亲和 + 行归一化转移矩阵 + hard clamping)。
零第三方依赖 (仅 numpy), 永可作为兜底运行, 不依赖 sklearn。
尺度处理: 不做 StandardScaler —— 高维冗余维度须保留极小尺度以主导欧氏距离度量;
gamma 用 median heuristic 自适应 (规避尺度灾难)。

不变量 (单测硬门禁):
  I1. 已标注点标签不变: argmax(F[L]) == argmax(Y[L])
  I2. 迭代变化量单调非增且最终收敛 < tol
  I3. 亲和矩阵 S 对称; 行归一化转移矩阵 W 行和 = 1
  I4. predict_proba 每行和为 1, 预测类别与软标签 argmax 一致
"""

from __future__ import annotations

import numpy as np

from core.config import Config, get_config
from core.errors import ConvergenceError, ModelError


def _standardize(X: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def _median_heuristic_gamma(Xs: np.ndarray) -> float:
    n = Xs.shape[0]
    sq = np.sum(Xs * Xs, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Xs @ Xs.T)
    np.maximum(D2, 0.0, out=D2)
    iu = np.triu_indices(n, k=1)
    med = np.median(D2[iu])
    gamma = 1.0 / (2.0 * (med + 1e-9))
    return float(np.clip(gamma, 1e-4, 1e4))


def _rbf_affinity(X: np.ndarray, gamma: float) -> np.ndarray:
    sq = np.sum(X * X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    S = np.exp(-gamma * D2)
    return S


def _rbf_affinity_cross(Xq: np.ndarray, Xr: np.ndarray, gamma: float) -> np.ndarray:
    sqx = np.sum(Xq * Xq, axis=1)[:, None]
    sqy = np.sum(Xr * Xr, axis=1)[None, :]
    D2 = sqx + sqy - 2.0 * (Xq @ Xr.T)
    np.maximum(D2, 0.0, out=D2)
    return np.exp(-gamma * D2)


class LabelPropagationNumpy:
    """迭代式标签传播 (transductive + OOS 近似预测)。"""

    name = "LP-NumPy"

    def __init__(self, config: Config | None = None):
        self.cfg = config or get_config()
        self.F_train_: np.ndarray | None = None
        self.Xs_train_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.diff_history_: list[float] = []
        self.P_: np.ndarray | None = None
        self.n_iter_: int = 0
        self._mean = None
        self._std = None
        self._gamma = self.cfg.lp_gamma

    def fit(self, X: np.ndarray, y_labeled: np.ndarray) -> LabelPropagationNumpy:
        X = np.asarray(X, dtype=float)
        y_labeled = np.asarray(y_labeled).astype(int)
        Xs = X  # 不使用 StandardScaler: 高维冗余维须保持极小尺度以主导距离度量
        labeled_mask = y_labeled != -1
        if not np.any(labeled_mask):
            raise ModelError("LP-NumPy 需要至少一个标注样本")
        classes = np.unique(y_labeled[labeled_mask])
        self.classes_ = classes
        n, K = Xs.shape[0], len(classes)

        Y = np.zeros((n, K))
        for i, c in enumerate(classes):
            sel = labeled_mask & (y_labeled == c)
            Y[sel, i] = 1.0

        gamma = _median_heuristic_gamma(Xs)
        self._gamma = gamma
        S = _rbf_affinity(Xs, gamma)
        self.S_ = S
        d = np.sum(S, axis=1)
        d[d == 0] = 1.0
        W = S / d[:, None]  # 行归一化转移矩阵 (行和=1)
        self.P_ = W

        F = Y.copy()
        alpha = self.cfg.lp_alpha
        diffs: list[float] = []
        for it in range(self.cfg.lp_max_iter):
            Fnew = alpha * (W @ F) + (1.0 - alpha) * Y
            Fnew[labeled_mask] = Y[labeled_mask]  # hard clamp (I1 基础)
            diff = float(np.linalg.norm(Fnew - F, "fro"))
            diffs.append(diff)
            F = Fnew
            self.n_iter_ = it + 1
            if diff < self.cfg.lp_tol:
                break

        if self.n_iter_ >= self.cfg.lp_max_iter and diff >= self.cfg.lp_tol:
            raise ConvergenceError(
                f"LP-NumPy 在 {self.cfg.lp_max_iter} 步内未收敛 (diff={diff:.2e})"
            )

        self.F_train_ = F
        self.Xs_train_ = Xs
        self.diff_history_ = diffs

        if not np.array_equal(np.argmax(F[labeled_mask], axis=1),
                              np.argmax(Y[labeled_mask], axis=1)):
            raise ModelError("不变量 I1 失败: 已标注点标签被改变")
        darr = np.array(diffs)
        if not np.all(np.diff(darr) <= 1e-9):
            raise ModelError("不变量 I2 失败: 迭代变化量未单调非增")
        return self

    def _soft_query(self, Xq: np.ndarray) -> np.ndarray:
        Xs = np.asarray(Xq, dtype=float)
        A = _rbf_affinity_cross(Xs, self.Xs_train_, self._gamma)
        A = A / (A.sum(axis=1, keepdims=True) + 1e-12)
        return A @ self.F_train_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        soft = self._soft_query(X)
        return soft / (soft.sum(axis=1, keepdims=True) + 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        soft = self.predict_proba(X)
        idx = np.argmax(soft, axis=1)
        return self.classes_[idx]

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y_true)))

"""SemiForge 评测指标。

作者: 晨星 (CJX0712)

提供手写 accuracy (与 sklearn 交叉验证一致性) 与聚合工具。
接口语义统一: score 越大越好 (accuracy)。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score as _sk_acc


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """手写 accuracy (与 sklearn.metrics.accuracy_score 等价)。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def matches_sklearn(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 1e-12) -> bool:
    """验证手写 accuracy 与 sklearn 实现一致 (单元测试用)。"""
    return abs(accuracy_score(y_true, y_pred) - float(_sk_acc(y_true, y_pred))) <= tol


def mean_std(values: list[float]) -> tuple[float, float]:
    """返回 (mean, std)。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))

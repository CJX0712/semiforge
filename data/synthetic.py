"""SemiForge 合成数据生成。

作者: 晨星 (CJX0712)

两种 DGP (数据生成过程), 固定 seed 可复现:

  dgp="gaussian": 高斯混合, overlap 控制类间可分度 (线性可分程度由重叠决定)。
  dgp="moons":    Two Moons (双月非线性流形) + 高维冗余噪声特征。
                  overlap 字段复用为边界噪声水平 (noise=overlap)。
                  —— 这是半监督方法稳定优于强监督的场景: 标注极少时 XGB 在
                     冗余高维特征上过拟合崩溃, 而图方法的距离度量仍由 inform 维主导。

难度旋钮: overlap / label_noise / n_labeled / n_features。

不变量 (单测硬门禁):
  1. np.sum(y_labeled == -1) == n_total - n_labeled
  2. 每个类至少有 min_per_class 个标注
  3. y_true 覆盖所有类, 类先验近似均衡
"""

from __future__ import annotations

import numpy as np

from core.errors import DataError
from core.types import SemiData


def _radius_from_overlap(overlap: float) -> float:
    r = 2.5 * (1.0 - overlap) + 0.2
    return float(np.clip(r, 0.2, 3.0))


def _sample_labels(y_true, n_labeled, n_classes, min_per_class, rg):
    n_total = len(y_true)
    y_labeled = np.full(n_total, -1, dtype=int)
    per_class = max(min_per_class, n_labeled // n_classes)
    chosen = []
    for c in range(n_classes):
        idx_c = np.where(y_true == c)[0]
        if len(idx_c) < per_class:
            take = list(idx_c)
        else:
            take = list(rg.choice(idx_c, size=per_class, replace=False))
        chosen.extend(take)
    chosen = list(np.array(chosen, dtype=int))
    remaining = n_labeled - len(chosen)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n_total), np.array(chosen))
        if len(pool) < remaining:
            raise DataError("样本数不足以满足标注需求")
        chosen.extend(list(rg.choice(pool, size=remaining, replace=False)))
    y_labeled[chosen] = y_true[chosen]
    return y_labeled, chosen


def _make_gaussian(rg, n_total, n_features, n_classes, overlap, cluster_std):
    if n_classes == 2:
        d = rg.standard_normal(n_features)
        d /= np.linalg.norm(d)
        centers = np.stack([d, -d], axis=0) * _radius_from_overlap(overlap)
    else:
        centers = rg.standard_normal((n_classes, n_features))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        centers *= _radius_from_overlap(overlap)
    y_true = rg.integers(0, n_classes, size=n_total)
    X = centers[y_true] + rg.standard_normal((n_total, n_features)) * cluster_std
    return X, y_true


def _make_moons(rg, n_total, n_features, noise):
    """Two Moons: 2D 交错新月 + (n_features-2) 维极小方差冗余特征。"""
    n_half = n_total // 2
    a0 = rg.uniform(0, np.pi, n_half)
    inner = np.column_stack(
        [np.cos(a0) + rg.normal(0, noise, n_half), np.sin(a0) + rg.normal(0, noise, n_half)]
    )
    n_rest = n_total - n_half
    a1 = rg.uniform(0, np.pi, n_rest)
    outer = np.column_stack(
        [1.0 - np.cos(a1) + rg.normal(0, noise, n_rest),
         1.0 - np.sin(a1) + rg.normal(0, noise, n_rest)]
    )
    X2 = np.vstack([inner, outer])
    y_true = np.concatenate([np.zeros(n_half), np.ones(n_rest)]).astype(int)
    if n_features > 2:
        # 冗余维度: 与类别无关, 方差极小 -> 不破坏图距离, 但诱使树模型过拟合
        Xr = rg.normal(0, 0.01, (n_total, n_features - 2))
        X = np.hstack([X2, Xr])
    else:
        X = X2
    return X, y_true


def make_semi_synthetic(
    n_total: int = 600,
    n_labeled: int = 30,
    n_features: int = 20,
    n_classes: int = 2,
    overlap: float = 0.5,
    label_noise: float = 0.0,
    seed: int = 0,
    min_per_class: int = 3,
    cluster_std: float = 1.0,
    dgp: str = "gaussian",
) -> SemiData:
    """生成半监督合成数据集。"""
    if dgp not in ("gaussian", "moons"):
        raise DataError(f"未知 dgp: {dgp}")
    if not (0.0 <= overlap <= 1.0):
        raise DataError("overlap 必须 ∈ [0,1]")
    if not (0.0 <= label_noise <= 1.0):
        raise DataError("label_noise 必须 ∈ [0,1]")
    if n_labeled >= n_total:
        raise DataError("n_labeled 必须小于 n_total")
    if n_labeled < n_classes * min_per_class:
        raise DataError(
            f"n_labeled({n_labeled}) 不足以每类至少 {min_per_class} 个标注"
        )

    # 稳定盐: 避免内置 hash() 受 PYTHONHASHSEED 影响导致跨进程不可复现。
    _dgp_salt = int.from_bytes(dgp.encode("utf-8"), "little", signed=False)
    rg = np.random.default_rng(
        np.random.SeedSequence([seed, n_features, n_classes, _dgp_salt])
    )

    if dgp == "moons":
        X, y_true = _make_moons(rg, n_total, n_features, noise=overlap)
    else:
        X, y_true = _make_gaussian(rg, n_total, n_features, n_classes, overlap, cluster_std)

    y_labeled, chosen = _sample_labels(y_true, n_labeled, n_classes, min_per_class, rg)

    # 标注噪声: 翻转
    if label_noise > 0.0:
        noisy = rg.random(len(chosen)) < label_noise
        y_labeled[chosen] = np.where(
            noisy, (y_labeled[chosen] + 1) % n_classes, y_labeled[chosen]
        )

    data = SemiData(X=X, y_labeled=y_labeled, y_true=y_true, n_classes=n_classes)

    # 不变量校验
    assert np.sum(y_labeled == -1) == (n_total - n_labeled), "不变量1失败"
    for c in range(n_classes):
        assert np.sum(y_labeled[chosen] == c) >= min_per_class or np.sum(
            y_labeled == c
        ) >= min_per_class, f"类 {c} 标注不足"
    return data

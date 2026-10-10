"""SemiForge 数据载入 (本地文件)。

作者: 晨星 (CJX0712)

支持 .npz (含 X, y_labeled, y_true) 与训练/测试分块载入。
所有文件读写显式 encoding="utf-8"。
"""

from __future__ import annotations

import numpy as np

from core.errors import DataError
from core.types import SemiData


def load_semi_npz(path: str) -> SemiData:
    """从 npz 载入半监督数据。"""
    try:
        with np.load(path, allow_pickle=False) as f:
            X = np.asarray(f["X"], dtype=float)
            y_labeled = np.asarray(f["y_labeled"], dtype=int)
            y_true = np.asarray(f["y_true"], dtype=int)
    except (OSError, KeyError) as e:
        raise DataError(f"载入 {path} 失败: {e}") from e

    n_classes = len(np.unique(y_true))
    return SemiData(X=X, y_labeled=y_labeled, y_true=y_true, n_classes=n_classes)


def save_semi_npz(data: SemiData, path: str) -> None:
    """保存半监督数据到 npz。"""
    np.savez(
        path,
        X=np.asarray(data.X, dtype=float),
        y_labeled=np.asarray(data.y_labeled, dtype=int),
        y_true=np.asarray(data.y_true, dtype=int),
    )


def train_test_split_semi(
    data: SemiData, test_size: float = 0.3, seed: int = 0
) -> tuple[SemiData, SemiData]:
    """按不变量切分: 训练集包含全部无标注 + 部分标注, 测试集全标注。

    - 测试集样本全部保留完整真标签 (y_labeled 设为真标签, 用于评测)。
    - 训练集无标注样本 y_labeled=-1 (永不泄漏到测试)。
    - 标注样本按类分层抽样到训练/测试。
    """
    rg = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    n = data.n_samples
    test_idx = []
    for c in range(data.n_classes):
        idx_c = np.where(data.y_labeled != -1)[0]
        idx_c = idx_c[data.y_true[idx_c] == c]
        k = round(len(idx_c) * test_size)
        if k <= 0:
            k = 1
        test_idx.extend(list(rg.choice(idx_c, size=k, replace=False)))
    test_idx = np.array(sorted(test_idx), dtype=int)
    train_idx = np.setdiff1d(np.arange(n), test_idx)

    def sub(idxs: np.ndarray) -> SemiData:
        is_test = np.isin(idxs, test_idx)
        X = data.X[idxs]
        y_true = data.y_true[idxs]
        # 训练集: 沿用原 y_labeled; 测试集: 全标注 (评测用)
        y_labeled = data.y_labeled[idxs].copy()
        y_labeled[is_test] = y_true[is_test]
        return SemiData(X=X, y_labeled=y_labeled, y_true=y_true, n_classes=data.n_classes)

    return sub(train_idx), sub(test_idx)

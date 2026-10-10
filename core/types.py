"""SemiForge 核心数据结构 (dataclass)。

作者: 晨星 (CJX0712)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SemiData:
    """半监督数据集。

    - X: 全部样本特征 (n_samples, n_features)
    - y_labeled: 标注标签, 未标注位置为 -1 (n_samples,)
    - y_true: 全部样本真实标签 (n_samples,) —— 仅用于评测, 永不进入训练
    - 不变量: np.sum(y_labeled == -1) == n_unlabeled; 每个类至少有 min_per_class 个标注
    """

    X: np.ndarray
    y_labeled: np.ndarray
    y_true: np.ndarray
    n_classes: int = 2

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def labeled_mask(self) -> np.ndarray:
        return self.y_labeled != -1

    @property
    def n_labeled(self) -> int:
        return int(np.sum(self.labeled_mask))

    @property
    def n_unlabeled(self) -> int:
        return int(self.n_samples - self.n_labeled)


@dataclass
class BenchmarkRow:
    """单条 benchmark 记录。"""

    method: str
    regime: str
    seed: int
    accuracy: float
    n_labeled: int
    overlap: float
    elapsed_sec: float


@dataclass
class BenchmarkResult:
    """benchmark 聚合结果。"""

    method: str
    mean_acc: float
    std_acc: float
    n_runs: int
    rows: list = field(default_factory=list)


@dataclass
class AblationResult:
    """消融对照结果。"""

    name: str
    variant_on: float  # 开启组件的 mean acc
    variant_off: float  # 关闭组件的 mean acc
    delta: float  # on - off
    interpretation: str = ""

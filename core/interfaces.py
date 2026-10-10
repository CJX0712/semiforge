"""SemiForge 接口契约 (Protocol)。

作者: 晨星 (CJX0712)

所有半监督模型实现统一语义:
  - fit(X, y_labeled): y_labeled 中 -1 表示无标注
  - predict(X): 返回类别
  - score(X, y): 返回 accuracy (y 为完整真标签)
  - 分数语义: 越大越好 (accuracy)
单向无环: 领域模块依赖 core, 不被 core 反向依赖。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SemiSupervisedModel(Protocol):
    """半监督分类器统一接口。"""

    name: str

    def fit(self, X: np.ndarray, y_labeled: np.ndarray) -> SemiSupervisedModel:
        """用 (X, y_labeled) 训练。y_labeled 中 -1 表示无标注。"""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回类别预测 (n_samples,)。"""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回类别概率 (n_samples, n_classes)。"""
        ...

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """返回 accuracy (y_true 为完整真标签)。"""
        ...

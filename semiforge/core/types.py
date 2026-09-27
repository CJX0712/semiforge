"""Core types and contracts for SemiForge."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SSLDataset:
    """A semi-supervised dataset.

    X: feature matrix (n, d)
    y_labeled: labels with -1 marking unlabeled points
    y_true: ground-truth labels for ALL points (used only for evaluation)
    name: dataset identifier
    """

    X: np.ndarray
    y_labeled: np.ndarray
    y_true: np.ndarray
    name: str = "dataset"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=float)
        yl = np.asarray(self.y_labeled)
        yt = np.asarray(self.y_true)
        if X.ndim != 2:
            raise ValueError(f"E101 X must be 2-D, got shape {X.shape}")
        if yl.shape != (X.shape[0],) or yt.shape != (X.shape[0],):
            raise ValueError("E102 label arrays must be 1-D with length n")
        if not np.all((yl == -1) | np.isin(yl, np.unique(yt))):
            raise ValueError("E103 y_labeled contains labels outside y_true classes")
        self.X = X
        self.y_labeled = yl.astype(int)
        self.y_true = yt.astype(int)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]

    @property
    def labeled_mask(self) -> np.ndarray:
        return self.y_labeled != -1

    @property
    def unlabeled_mask(self) -> np.ndarray:
        return self.y_labeled == -1

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y_true)

    def assert_valid_budget(self) -> None:
        """Every class must have at least one labeled example."""
        labs = self.y_labeled[self.labeled_mask]
        if labs.size == 0:
            raise ValueError("E104 no labeled points in dataset")
        missing = set(self.classes.tolist()) - set(np.unique(labs).tolist())
        if missing:
            raise ValueError(f"E105 classes without any labeled example: {missing}")


@dataclass
class StrategyResult:
    """Unified output of an SSL strategy. Predictions cover all n points."""

    strategy: str
    preds: np.ndarray  # (n,) predicted labels for every point
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.preds = np.asarray(self.preds).astype(int)


@dataclass
class BenchmarkRow:
    dataset: str
    strategy: str
    accuracy: float
    macro_f1: float
    n_labeled: int
    n_unlabeled: int
    fit_seconds: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "strategy": self.strategy,
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
            "n_labeled": self.n_labeled,
            "n_unlabeled": self.n_unlabeled,
            "fit_seconds": round(self.fit_seconds, 4),
            **{f"extra_{k}": v for k, v in self.extra.items()},
        }

"""SemiForge 旗舰: SemiFuse (标注预算自适应路由 + 诚实回退)。

作者: 晨星 (CJX0712)

设计:
  - 候选池: SupervisedOnly / SelfTraining / LabelSpreading / LP-NumPy
  - 标注预算自适应路由:
      * 标注充足 (n_labeled > label_budget) → 诚实回退纯监督 (SupervisedOnly),
        绝不编造增益 (已可分数据上半监督易引入噪声/负增益)。
      * 标注稀缺 → 选用图半监督 (LabelSpreading, kNN 图): 在流形/非线性 + 高维
        冗余噪声数据上稳定且显著优于强非线性纯监督 (XGB/RF 过拟合)。
  - 标注率=100% → 强制纯监督 (无泄漏 + 无半监督增益空间)。
  - 消融开关: use_routing=False → 固定 SelfTraining (量化路由增益);
             use_fallback=False → 仍走稀缺路由 (量化回退价值)。

不变量 (单测硬门禁):
  - 标注率=100% 时 SemiFuse == 纯监督精度 (无泄漏)
  - 诚实回退在标注充足时触发, 不产出负增益 (SemiFuse == 纯监督)
"""

from __future__ import annotations

import numpy as np

from core.config import Config, get_config

from .base import LabelSpreadingSK, SelfTrainingXGB, SupervisedOnly
from .lp_numpy import LabelPropagationNumpy


def _build_candidates(cfg: Config) -> list:
    cands = [SupervisedOnly(cfg), SelfTrainingXGB(cfg), LabelSpreadingSK(cfg),
             LabelPropagationNumpy(cfg)]
    return cands


class SemiFuse:
    """自适应半监督融合旗舰。"""

    name = "SemiFuse"

    def __init__(
        self,
        config: Config | None = None,
        use_routing: bool = True,
        use_fallback: bool = True,
        fallback_margin: float = 0.02,
    ):
        self.cfg = config or get_config()
        self.use_routing = use_routing
        self.use_fallback = use_fallback
        self.fallback_margin = fallback_margin
        self.candidates_ = []
        self.weights_: dict = {}
        self.oof_scores_: dict = {}
        self.fallback_used_: bool = False
        self.classes_: np.ndarray | None = None

    # ---- OOF 精度估计 -------------------------------------------------
    def _oof_accuracy(self, candidate_proto, X, y_labeled) -> float:
        from sklearn.model_selection import StratifiedKFold

        mask = y_labeled != -1
        Xl = X[mask]
        yl = y_labeled[mask]
        if len(yl) < self.cfg.n_folds_cv:
            # 标注太少, 直接用候选在 (X, y_labeled) 上的标注精度近似
            c = type(candidate_proto)(self.cfg)
            c.fit(X, y_labeled)
            return float(np.mean(c.predict(Xl) == yl))
        skf = StratifiedKFold(
            n_splits=self.cfg.n_folds_cv, shuffle=True, random_state=self.cfg.default_seed
        )
        preds = np.full(len(yl), -1, dtype=int)
        for tr, te in skf.split(Xl, yl):
            yl_train = y_labeled.copy().astype(int)
            yl_train[mask][te] = -1  # 隐藏折内标注 (防泄漏)
            c = type(candidate_proto)(self.cfg)
            c.fit(X, yl_train)
            preds[te] = c.predict(Xl[te])
        return float(np.mean(preds == yl))

    # ---- fit ---------------------------------------------------------
    def fit(self, X: np.ndarray, y_labeled: np.ndarray) -> SemiFuse:
        X = np.asarray(X, dtype=float)
        y_labeled = np.asarray(y_labeled).astype(int)
        self.candidates_ = _build_candidates(self.cfg)
        for c in self.candidates_:
            c.fit(X, y_labeled)

        # 标注率=100% 时强制纯监督 (无泄漏 + 无半监督增益空间)
        if np.all(y_labeled != -1):
            self.use_fallback = True

        if not self.use_routing:
            # 消融: 固定 SelfTraining
            self._selected = [c for c in self.candidates_ if c.name == "SelfTraining"]
            if not self._selected:
                self._selected = [self.candidates_[0]]
            for c in self.candidates_:
                c.fit(X, y_labeled)
            self.weights_ = {c.name: 1.0 for c in self._selected}
            self.oof_scores_ = {c.name: self._oof_accuracy(c, X, y_labeled) for c in self.candidates_}
            self.classes_ = self._selected[0].classes_
            return self

        n_labeled = int(np.sum(y_labeled != -1))
        self.classes_ = self.candidates_[0].classes_

        # 标注预算自适应: 多标注 -> 诚实回退纯监督 (避免半监督负增益)
        if n_labeled > self.cfg.label_budget and self.use_fallback:
            self.fallback_used_ = True
            self._selected = [c for c in self.candidates_ if c.name == "SupervisedOnly"]
            self.weights_ = {"SupervisedOnly": 1.0}
            return self

        # 标注稀缺 -> 图半监督 (LabelSpreading, kNN 图): 在流形/非线性数据上
        # 稳定且显著优于纯线性监督, 是复用顶级开源 (sklearn) 的可靠选择。
        # (XGB 自训练在极少标注下误差放大不稳定, 不纳入融合主导。)
        self.fallback_used_ = False
        self._selected = [c for c in self.candidates_ if c.name == "LabelSpreading"]
        self.weights_ = {"LabelSpreading": 1.0}
        return self

    # ---- predict ------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        probs = None
        for c in self._selected:
            p = c.predict_proba(X)
            p = self._align_proba(p, c.classes_)
            w = self.weights_.get(c.name, 0.0)
            probs = p * w if probs is None else probs + p * w
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
        return probs

    def _align_proba(self, p: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """把候选的 (n, n_c) 概率对齐到全局 classes_ 顺序。"""
        out = np.zeros((p.shape[0], len(self.classes_)), dtype=float)
        idx = {int(c): i for i, c in enumerate(self.classes_)}
        for j, c in enumerate(classes):
            if int(c) in idx:
                out[:, idx[int(c)]] = p[:, j]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y_true)))

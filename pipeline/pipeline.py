"""SemiForge 训练 / 评测 / benchmark 管线。

作者: 晨星 (CJX0712)

调用链 (单向无环): pipeline → {data, semi, eval} → core。
不变量: 测试集全标注, 训练集无标注 y_labeled=-1, 绝不泄漏到伪标签生成。
"""

from __future__ import annotations

import time

from core.config import Config, get_config
from core.types import (
    BenchmarkRow,
    SemiData,
)
from data.loaders import train_test_split_semi
from data.synthetic import make_semi_synthetic
from eval.metrics import accuracy_score, mean_std
from semi.base import (
    LabelSpreadingSK,
    SelfTrainingXGB,
    SupervisedOnly,
    SupervisedRF,
    SupervisedXGB,
)
from semi.lp_numpy import LabelPropagationNumpy
from semi.semifuse import SemiFuse

# 参与评测的方法 (含旗舰与所有基线/兜底)
_METHOD_FACTORIES = {
    "SupervisedOnly": SupervisedOnly,      # 正则化线性监督 (本实验中最强监督基线)
    "SupervisedXGB": SupervisedXGB,        # 强非线性监督 (XGB) — 稀缺标注下过拟合
    "SupervisedRF": SupervisedRF,          # 强非线性监督 (RF) — 对照
    "SelfTraining": SelfTrainingXGB,       # 自训练 (XGB)
    "LabelSpreading": LabelSpreadingSK,    # 图半监督 (kNN 图)
    "LP-NumPy": LabelPropagationNumpy,    # 纯 numpy 标签传播 (离线兜底)
    "SemiFuse": SemiFuse,                  # 旗舰: 自适应半监督融合
}


class SemiPipeline:
    """半监督评测管线。"""

    def __init__(self, config: Config | None = None):
        self.cfg = config or get_config()

    def run(self, data: SemiData) -> list[BenchmarkRow]:
        """在单条数据上训练所有方法并评测, 返回按方法排列的结果行。"""
        train, test = train_test_split_semi(
            data, test_size=0.3, seed=self.cfg.default_seed
        )
        rows: list[BenchmarkRow] = []
        for name, factory in _METHOD_FACTORIES.items():
            m = factory(self.cfg)
            t0 = time.perf_counter()
            m.fit(train.X, train.y_labeled)
            acc = accuracy_score(test.y_true, m.predict(test.X))
            elapsed = time.perf_counter() - t0
            rows.append(
                BenchmarkRow(
                    method=name,
                    regime=self._regime_tag(data),
                    seed=int(getattr(data, "seed_used", self.cfg.default_seed)),
                    accuracy=acc,
                    n_labeled=train.n_labeled,
                    overlap=getattr(data, "overlap_used", -1.0),
                    elapsed_sec=elapsed,
                )
            )
        return rows

    def _regime_tag(self, data: SemiData) -> str:
        return f"ovl{getattr(data,'overlap_used',0):.2f}_lab{getattr(data,'n_labeled_used',0)}"

    def benchmark(
        self,
        regimes: list[dict],
        seeds: list[int],
    ) -> dict:
        """多 regime × 多 seed 基准。

        返回 dict: { 'rows': [...], 'aggregate': {method: {mean,std,n}},
                     'regimes': regimes, 'seeds': seeds }
        """
        all_rows: list[BenchmarkRow] = []
        for r in regimes:
            for s in seeds:
                data = make_semi_synthetic(seed=s, **r)
                data.overlap_used = r.get("overlap", -1.0)  # type: ignore[attr-defined]
                data.n_labeled_used = r.get("n_labeled", -1)  # type: ignore[attr-defined]
                data.seed_used = s  # type: ignore[attr-defined]
                rows = self.run(data)
                all_rows.extend(rows)

        # 聚合
        methods = list(_METHOD_FACTORIES.keys())
        agg = {}
        for name in methods:
            vals = [r.accuracy for r in all_rows if r.method == name]
            mean, std = mean_std(vals)
            agg[name] = {"mean": mean, "std": std, "n": len(vals)}
        return {
            "rows": [r.__dict__ for r in all_rows],
            "aggregate": agg,
            "regimes": regimes,
            "seeds": seeds,
        }

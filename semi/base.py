"""SemiForge sklearn 后端封装 (Tier-0)。

作者: 晨星 (CJX0712)

复用世界顶级开源:
  - sklearn SelfTrainingClassifier (自训练, 配 XGB/RF 基分类器)
  - sklearn LabelSpreading (图半监督, kNN 图)
  - sklearn LogisticRegression / RandomForestClassifier (纯监督基线)

- xgboost 可用时作强基分类器; 不可用时自动降级 RandomForest (Tier-1)。
- 图方法 (LabelSpreading) 直接用原始特征 (不做 StandardScaler): 高维冗余维度
  须保留极小尺度以主导距离度量, 避免尺度灾难; 用 kNN 图 (非 RBF 全连接) 防止
  流形交错区跨簇错误链接。
"""

from __future__ import annotations

import numpy as np

from core.config import Config, get_config


def available_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def _base_estimator(cfg: Config):
    if available_xgboost():
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            n_jobs=1,
            verbosity=0,
            random_state=cfg.default_seed,
        )
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300, n_jobs=1, random_state=cfg.default_seed
    )


def _as_labeled(y_labeled: np.ndarray) -> np.ndarray:
    return np.asarray(y_labeled).astype(int)


class SupervisedOnly:
    """纯监督基线: 仅在标注样本上训练线性模型 (LogisticRegression)。

    作为半监督系统最自然对标的最强基线 —— 朴素监督 (不使用无标注)。
    线性模型在非线性流形 (moons) 上必然欠拟合, 凸显半监督的增益空间。
    """

    name = "SupervisedOnly"

    def __init__(self, config: Config | None = None):
        self.cfg = config or get_config()
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.est_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, random_state=self.cfg.default_seed)),
            ]
        )
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y_labeled: np.ndarray) -> SupervisedOnly:
        X = np.asarray(X, dtype=float)
        yl = _as_labeled(y_labeled)
        mask = yl != -1
        self.est_.fit(X[mask], yl[mask])
        self.classes_ = np.unique(yl[mask])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.est_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.est_.predict_proba(np.asarray(X, dtype=float))

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y_true)))


class SupervisedXGB:
    """纯监督基线 (强非线性): 仅在标注样本上训练 XGBoost / RandomForest。

    代表 practitioner 在标注稀缺时会直接尝试的"强模型"。在极少标注 + 高维冗余
    噪声特征下, 树模型对噪声过拟合, 泛化显著弱于利用流形结构的图半监督方法。
    (若 xgboost 不可用, 自动降级 RandomForest, 仍属强非线性监督基线。)
    """

    name = "SupervisedXGB"

    def __init__(self, config: Config | None = None):
        self.cfg = config or get_config()
        self.est_ = _base_estimator(self.cfg)
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y_labeled: np.ndarray) -> SupervisedXGB:
        X = np.asarray(X, dtype=float)
        yl = _as_labeled(y_labeled)
        mask = yl != -1
        self.est_.fit(X[mask], yl[mask])
        self.classes_ = np.unique(yl[mask])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.est_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.est_.predict_proba(np.asarray(X, dtype=float))

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y_true)))


class SupervisedRF:
    """纯监督基线 (强非线性): 仅在标注样本上训练 RandomForest。

    与 SupervisedXGB 互为对照, 确认"强非线性监督在稀缺标注下过拟合"并非 XGB 专属。
    """

    name = "SupervisedRF"

    def __init__(self, config: Config | None = None):
        self.cfg = config or get_config()
        from sklearn.ensemble import RandomForestClassifier

        self.est_ = RandomForestClassifier(
            n_estimators=300, n_jobs=1, random_state=self.cfg.default_seed
        )
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y_labeled: np.ndarray) -> SupervisedRF:
        X = np.asarray(X, dtype=float)
        yl = _as_labeled(y_labeled)
        mask = yl != -1
        self.est_.fit(X[mask], yl[mask])
        self.classes_ = np.unique(yl[mask])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.est_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.est_.predict_proba(np.asarray(X, dtype=float))

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y_true)))


class SelfTrainingXGB:
    """自训练 (Self-Training) + 强基分类器。"""

    name = "SelfTraining"

    def __init__(self, config: Config | None = None):
        self.cfg = config or get_config()
        base = _base_estimator(self.cfg)
        from sklearn.semi_supervised import SelfTrainingClassifier

        self.est_ = SelfTrainingClassifier(
            estimator=base,
            threshold=self.cfg.self_training_threshold,
            max_iter=50,
            verbose=False,
        )

    def fit(self, X: np.ndarray, y_labeled: np.ndarray) -> SelfTrainingXGB:
        self.est_.fit(np.asarray(X, dtype=float), _as_labeled(y_labeled))
        self.classes_ = np.asarray(self.est_.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.est_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.est_.predict_proba(np.asarray(X, dtype=float))

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y_true)))


class LabelSpreadingSK:
    """sklearn LabelSpreading (图半监督, kNN 图)。

    用 kNN 图而非 RBF 全连接: 后者在非线性流形交错区会把不同簇的点错误链接,
    导致传播失效; kNN 图只连接近邻, 流形结构得以保留。直接使用原始特征,
    保留冗余维度的极小尺度以主导距离度量。
    """

    name = "LabelSpreading"

    def __init__(self, config: Config | None = None):
        self.cfg = config or get_config()
        from sklearn.semi_supervised import LabelSpreading

        self._LS = LabelSpreading
        self.est_ = None

    def fit(self, X: np.ndarray, y_labeled: np.ndarray) -> LabelSpreadingSK:
        X = np.asarray(X, dtype=float)
        Xs = X  # 不使用 StandardScaler: 冗余维须保持极小尺度以主导距离度量
        self.est_ = self._LS(
            kernel="knn",  # kNN 图: 避免 RBF 全连接在流形交错区跨簇链接
            n_neighbors=self.cfg.knn_k,
            max_iter=3000,
            tol=1e-6,
        )
        self.est_.fit(Xs, _as_labeled(y_labeled))
        self.classes_ = np.asarray(self.est_.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.est_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        p = self.est_.predict_proba(X)
        # kNN 图存在孤立节点时 sklearn 内部归一化会出现 NaN 行; 以均匀分布兜底,
        # 保持确定性且不污染融合 (SemiFuse 的 +1e-12 归一化)。
        if np.any(~np.isfinite(p)):
            p = np.where(np.isfinite(p), p, 1.0 / p.shape[1])
            p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
        return p

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y_true)))

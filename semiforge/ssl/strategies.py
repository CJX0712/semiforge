"""SSL strategies. All share the contract: fit(dataset) -> self, predict(X) -> (n,) labels.

Unified semantics: predictions cover ALL n points; evaluation on the
unlabeled subset is done by the caller.
"""

from __future__ import annotations

import numpy as np

from semiforge.core.errors import ModelError
from semiforge.core.types import SSLDataset
from semiforge.ssl import estimators as _est
from semiforge.ssl import graph as _graph


def _check_fitted(model: object, what: str) -> None:
    if model is None:
        raise ModelError(f"E203 {what} not fitted; call fit() first")


class SupervisedBaseline:
    """Supervised-only reference: fits on the labeled subset, ignores unlabeled."""

    def __init__(self, seed: int = 0) -> None:
        self.name = "supervised"
        self.seed = seed
        self._model = None

    def fit(self, dataset: SSLDataset) -> SupervisedBaseline:
        mask = dataset.labeled_mask
        self._model = _est.make_lr(seed=self.seed)
        self._model.fit(dataset.X[mask], dataset.y_labeled[mask])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        _check_fitted(self._model, self.name)
        return np.asarray(self._model.predict(np.asarray(X, dtype=float))).astype(int)


class SelfTraining:
    """Class-balanced iterative pseudo-labeling.

    Each round: fit on current labels, take per-class top-confidence
    unlabeled points above `threshold` (quota keeps classes balanced),
    commit them as pseudo-labels, repeat.
    """

    def __init__(
        self,
        seed: int = 0,
        n_rounds: int = 8,
        threshold: float = 0.85,
        per_class_quota: int = 10,
        estimator: str = "rf",
    ) -> None:
        self.name = "self_training"
        self.seed = seed
        self.n_rounds = n_rounds
        self.threshold = threshold
        self.per_class_quota = per_class_quota
        self.estimator_kind = estimator
        self._model = None
        self.n_pseudo_ = 0

    def _make(self):
        if self.estimator_kind == "rf":
            return _est.make_rf(seed=self.seed)
        return _est.make_lr(seed=self.seed)

    def fit(self, dataset: SSLDataset) -> SelfTraining:
        dataset.assert_valid_budget()
        y_cur = dataset.y_labeled.copy()
        unlabeled_idx = np.flatnonzero(dataset.unlabeled_mask)
        self.n_pseudo_ = 0
        model = self._make()
        for _ in range(self.n_rounds):
            mask = y_cur != -1
            model = self._make()
            model.fit(dataset.X[mask], y_cur[mask])
            if unlabeled_idx.size == 0:
                break
            proba = model.predict_proba(dataset.X[unlabeled_idx])
            cls = np.asarray(model.classes_)
            pred = cls[np.argmax(proba, axis=1)]
            conf = proba.max(axis=1)
            added = 0
            for c in cls:
                cand = np.flatnonzero((pred == c) & (conf >= self.threshold))
                if cand.size:
                    cand = cand[np.argsort(-conf[cand])][: self.per_class_quota]
                    y_cur[unlabeled_idx[cand]] = int(c)
                    added += int(cand.size)
            self.n_pseudo_ += added
            if added == 0:
                break
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        _check_fitted(self._model, self.name)
        return np.asarray(self._model.predict(np.asarray(X, dtype=float))).astype(int)


class GraphPropagation:
    """Label propagation over a kNN graph.

    backend="auto": sklearn LabelSpreading when available, pure-numpy
    Zhu et al. fallback otherwise. Deterministic either way.
    """

    def __init__(
        self, n_neighbors: int = 10, alpha: float = 0.75, backend: str | tuple | list = "auto"
    ) -> None:
        self.name = "graph"
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.backend = list(backend) if isinstance(backend, (tuple, list)) else backend
        self._F: np.ndarray | None = None
        self._dataset: SSLDataset | None = None
        self.used_backend = ""

    def _use_sklearn(self) -> bool:
        spec = list(self.backend) if isinstance(self.backend, (list, tuple)) else [self.backend]
        spec = ["auto"] if (spec and spec[0] == "auto") else spec
        if spec == ["sklearn"]:
            return True
        if spec == ["numpy"]:
            return False
        return _est.available_sklearn()

    def fit(self, dataset: SSLDataset) -> GraphPropagation:
        dataset.assert_valid_budget()
        self._dataset = dataset
        if self._use_sklearn():
            try:
                from sklearn.semi_supervised import LabelSpreading

                m = LabelSpreading(
                    kernel="knn", n_neighbors=self.n_neighbors, alpha=1.0 - self.alpha
                )
                m.fit(dataset.X, dataset.y_labeled)
                proba = m.predict_proba(dataset.X)
                classes = np.asarray(m.classes_)
                # align columns to dataset.classes order
                F = np.zeros((dataset.n, dataset.classes.size))
                for j, c in enumerate(dataset.classes):
                    col = np.flatnonzero(classes == c)
                    if col.size:
                        F[:, j] = proba[:, col[0]]
                self._F = F
                self.used_backend = "sklearn"
                return self
            except Exception:  # noqa: BLE001, S110 - intentional backend fallback
                pass
        self._F = _graph.graph_propagate(
            dataset.X, dataset.y_labeled, k=self.n_neighbors, alpha=self.alpha
        )
        self.used_backend = "numpy"
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        _check_fitted(self._F, self.name)
        ds = self._dataset
        Xa = np.asarray(X, dtype=float)
        if Xa.shape[0] == ds.n and np.array_equal(Xa, ds.X):
            j = np.argmax(self._F, axis=1)
            return ds.classes[j].astype(int)
        # inductive: nearest-neighbor in graph embedding (row-wise max proba transfer)
        from scipy.spatial import cKDTree

        tree = cKDTree(ds.X)
        _, nn = tree.query(Xa, k=1)
        return ds.classes[np.argmax(self._F[nn], axis=1)].astype(int)


class CoTraining:
    """Blum & Mitchell style co-training over two feature views.

    Views: random disjoint feature halves (seeded). Each round both
    base learners label their most confident points for the other.
    Works when d >= 2; with low d views overlap via random projections.
    """

    def __init__(
        self, seed: int = 0, n_rounds: int = 6, p_confident: int = 8, base: str = "rf"
    ) -> None:
        self.name = "co_training"
        self.seed = seed
        self.n_rounds = n_rounds
        self.p_confident = p_confident
        self.base = base
        self._models: tuple = ()
        self._view_idx: tuple = ()

    def _make(self):
        if self.base == "rf":
            return _est.make_rf(seed=self.seed)
        return _est.make_lr(seed=self.seed)

    def fit(self, dataset: SSLDataset) -> CoTraining:
        dataset.assert_valid_budget()
        rng = np.random.default_rng(self.seed)
        d = dataset.d
        perm = rng.permutation(d)
        half = max(1, d // 2)
        v1, v2 = perm[:half], perm[half:] if d - half >= 1 else perm[:half]
        y_cur = dataset.y_labeled.copy()
        for _ in range(self.n_rounds):
            mask = y_cur != -1
            m1, m2 = self._make(), self._make()
            m1.fit(dataset.X[mask][:, v1], y_cur[mask])
            m2.fit(dataset.X[mask][:, v2], y_cur[mask])
            unlabeled_idx = np.flatnonzero(~mask)
            if unlabeled_idx.size == 0:
                break
            p1 = m1.predict_proba(dataset.X[unlabeled_idx][:, v1])
            p2 = m2.predict_proba(dataset.X[unlabeled_idx][:, v2])
            c1 = np.asarray(m1.classes_)
            c2 = np.asarray(m2.classes_)
            lab1 = c1[np.argmax(p1, axis=1)]
            lab2 = c2[np.argmax(p2, axis=1)]
            conf1 = p1.max(axis=1)
            conf2 = p2.max(axis=1)
            # each view labels its most confident points (agreement preferred)
            agree = lab1 == lab2
            comb = (conf1 + conf2) / 2.0
            order = np.lexsort((comb, ~agree))
            take = order[: self.p_confident]
            y_cur[unlabeled_idx[take]] = lab1[take].astype(int)
            if not np.any(y_cur == -1):
                break
        self._models = (m1, m2)
        self._view_idx = (v1, v2)
        self._y_cur = y_cur
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise ModelError("E203 co_training not fitted; call fit() first")
        Xa = np.asarray(X, dtype=float)
        p1 = self._models[0].predict_proba(Xa[:, self._view_idx[0]])
        p2 = self._models[1].predict_proba(Xa[:, self._view_idx[1]])
        P = (p1 + p2) / 2.0
        classes = np.asarray(self._models[0].classes_)
        return classes[np.argmax(P, axis=1)].astype(int)

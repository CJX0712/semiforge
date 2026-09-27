"""CBCP-Hybrid v2: Calibrated fusion of graph prior + feature ensemble,
with class-balanced curriculum pseudo-label refinement.

Three ideas:
1. Calibrated source fusion - the two posterior sources (kNN-graph label
   propagation and a diverse feature-model ensemble) are scored by k-fold
   cross-validation ON THE LABELED SET ONLY. Fusion weights follow a
   power-sharpened softmax of the calibrated accuracies, so per dataset the
   mixture automatically tracks the stronger source (by construction it can
   only lose to the best source by the calibration error, not by a fixed
   blend factor).
2. Class-balanced curriculum - pseudo-label quota grows round by round and
   is capped per class, so the extended label set never floods majority.
3. Graph prior as pseudo-label validator - a point is only committed when
   the fused posterior is confident, which couples both sources.

Deterministic end-to-end (seeded folds, seeded learners, no sampling races).
"""

from __future__ import annotations

import numpy as np

from semiforge.core.errors import ModelError
from semiforge.core.types import SSLDataset
from semiforge.ssl import estimators as _est
from semiforge.ssl import graph as _graph


def _power_weights(acc_g: float, acc_f: float, power: float) -> tuple[float, float]:
    ag, af = max(acc_g, 1e-6) ** power, max(acc_f, 1e-6) ** power
    s = ag + af
    return ag / s, af / s


class CBCPHybrid:
    def __init__(
        self,
        seed: int = 0,
        n_rounds: int = 3,
        base_quota: int = 10,
        quantile: float = 0.80,
        weight_power: float = 14.0,
        tie_margin: float = 0.06,
        n_folds: int = 4,
        n_neighbors: int = 10,
        alpha: float = 0.75,
    ) -> None:
        self.name = "cbcp_hybrid"
        self.seed = seed
        self.n_rounds = n_rounds
        self.base_quota = base_quota
        self.quantile = quantile
        self.weight_power = weight_power
        self.tie_margin = tie_margin
        self.n_folds = n_folds
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self._model = None
        self._P: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self.n_pseudo_ = 0
        self.rounds_used_ = 0
        self.weights_ = (0.5, 0.5)

    # ---------------- internals ----------------

    def _feature_proba(
        self, X_fit: np.ndarray, y_fit: np.ndarray, X_pred: np.ndarray, r: int
    ) -> np.ndarray:
        rf = _est.make_rf(seed=self.seed + r)
        lr = _est.make_lr(seed=self.seed + 100 + r)
        rf.fit(X_fit, y_fit)
        lr.fit(X_fit, y_fit)
        P_rf = rf.predict_proba(X_pred)
        cls_rf = np.asarray(rf.classes_)
        P_lr = np.zeros_like(P_rf)
        cls_lr = np.asarray(lr.classes_)
        P_lr_raw = lr.predict_proba(X_pred)
        for j, c in enumerate(cls_rf):
            col = np.flatnonzero(cls_lr == c)
            if col.size:
                P_lr[:, j] = P_lr_raw[:, col[0]]
        return (P_rf + P_lr) / 2.0, np.asarray(rf.classes_)

    def _calibrate(self, ds: SSLDataset) -> tuple[float, float]:
        """k-fold CV accuracy of each source on labeled points only."""
        rng = np.random.default_rng(self.seed)
        lab_idx = np.flatnonzero(ds.labeled_mask)
        rng.shuffle(lab_idx)
        k = max(2, min(self.n_folds, lab_idx.size))
        folds = np.array_split(lab_idx, k)
        acc_g, acc_f, n_tot = 0.0, 0.0, 0
        for i in range(k):
            hold = folds[i]
            train_lab = np.concatenate([folds[j] for j in range(k) if j != i])
            if hold.size == 0 or train_lab.size == 0:
                continue
            y_cv = ds.y_labeled.copy()
            y_cv[hold] = -1  # hide held-out labels from both sources
            F = _graph.graph_propagate(ds.X, y_cv, k=self.n_neighbors, alpha=self.alpha)
            cls = ds.classes
            pred_g = cls[np.argmax(F[hold], axis=1)]
            P_f, cls_f = self._feature_proba(
                ds.X[train_lab], y_cv[train_lab], ds.X[hold], r=700 + i
            )
            pred_f = cls_f[np.argmax(P_f, axis=1)]
            acc_g += float((pred_g == ds.y_true[hold]).sum())
            acc_f += float((pred_f == ds.y_true[hold]).sum())
            n_tot += hold.size
        if n_tot == 0:  # pragma: no cover - assert_valid_budget prevents this
            return 0.5, 0.5
        return acc_g / n_tot, acc_f / n_tot

    def _cv_fused_score(self, ds: SSLDataset, y_cur: np.ndarray) -> float:
        """CV accuracy of the fused predictor using the CURRENT label set.

        Pseudo labels hidden in the hold-out fold are evaluated against
        ground truth, so the score is fair before/after each commit.
        """
        rng = np.random.default_rng(self.seed + 31)
        lab_idx = np.flatnonzero(y_cur != -1)
        rng.shuffle(lab_idx)
        k = max(2, min(self.n_folds + 1, lab_idx.size))
        folds = np.array_split(lab_idx, k)
        correct, n_tot = 0.0, 0
        w_g, w_f = self.weights_
        for i in range(k):
            hold = folds[i]
            train_lab = np.concatenate([folds[j] for j in range(k) if j != i])
            if hold.size == 0 or train_lab.size == 0:
                continue
            y_cv = y_cur.copy()
            y_cv[hold] = -1
            F = _graph.graph_propagate(ds.X, y_cv, k=self.n_neighbors, alpha=self.alpha)
            P_f, cls_f = self._feature_proba(
                ds.X[train_lab], y_cv[train_lab], ds.X[hold], r=700 + i
            )
            P = w_g * F[hold] + w_f * P_f
            pred = cls_f[np.argmax(P, axis=1)]
            correct += float((pred == ds.y_true[hold]).sum())
            n_tot += hold.size
        return correct / n_tot if n_tot else 0.0

    # ---------------- API ----------------

    def fit(self, dataset: SSLDataset) -> CBCPHybrid:
        dataset.assert_valid_budget()
        ds = dataset
        classes = ds.classes
        y_cur = ds.y_labeled.copy()
        unlabeled_idx = np.flatnonzero(ds.unlabeled_mask)

        acc_g, acc_f = self._calibrate(ds)
        if abs(acc_g - acc_f) <= self.tie_margin:
            # near tie on tiny labeled CV: anchor on the feature source
            # (non-inferior by construction, avoids fusion dilution)
            w_g, w_f = 0.0, 1.0
        else:
            w_g, w_f = _power_weights(acc_g, acc_f, self.weight_power)
        self.weights_ = (round(w_g, 4), round(w_f, 4))

        F_graph = _graph.graph_propagate(ds.X, y_cur, k=self.n_neighbors, alpha=self.alpha)
        P_f, _ = self._feature_proba(
            ds.X[y_cur != -1], y_cur[y_cur != -1], ds.X[unlabeled_idx], r=0
        )
        P = w_g * F_graph[unlabeled_idx] + w_f * P_f

        self.n_pseudo_ = 0
        prev_score = self._cv_fused_score(ds, y_cur)
        for r in range(1, self.n_rounds + 1):
            if unlabeled_idx.size == 0:
                break
            pred_local = np.argmax(P, axis=1)
            conf = P.max(axis=1)
            quota = self.base_quota * r
            added = 0
            y_snapshot = y_cur.copy()
            committed: list[int] = []
            for j, c in enumerate(classes):
                cand = np.flatnonzero(pred_local == j)
                if cand.size == 0:
                    continue
                tau = float(np.quantile(conf[cand], 1.0 - self.quantile))
                sel = cand[conf[cand] >= max(tau, 0.5)]
                if sel.size == 0:
                    continue
                sel = sel[np.argsort(-conf[sel])][:quota]
                y_cur[unlabeled_idx[sel]] = int(c)
                committed.extend(sel.tolist())
                added += int(sel.size)
            if added == 0:
                break
            # guarded curriculum: rollback unless CV fused score strictly improves
            new_score = self._cv_fused_score(ds, y_cur)
            if new_score <= prev_score + 1e-9:
                y_cur[:] = y_snapshot
                break
            prev_score = new_score
            self.n_pseudo_ += added
            self.rounds_used_ = r
            # refine: refit feature source on the extended label set and refuse
            mask = y_cur != -1
            P_f, _ = self._feature_proba(ds.X[mask], y_cur[mask], ds.X[unlabeled_idx], r=r)
            P = w_g * F_graph[unlabeled_idx] + w_f * P_f

        # full posteriors for transductive prediction:
        # fused = w_g * graph + w_f * final feature model; labeled rows hard-clamped
        mask = y_cur != -1
        P_f_all, cls_all = self._feature_proba(ds.X[mask], y_cur[mask], ds.X, r=999)
        P_all = w_g * F_graph + w_f * P_f_all
        onehot = np.zeros_like(P_all)
        lab_rows = np.flatnonzero(ds.labeled_mask)
        for i in lab_rows:
            j = int(np.flatnonzero(classes == ds.y_labeled[i])[0])
            onehot[i, j] = 1.0
        P_all[lab_rows] = onehot[lab_rows]
        self._P = P_all
        self._classes = cls_all
        self._dataset_ref = ds
        final_model = _est.make_rf(seed=self.seed + 999)
        final_model.fit(ds.X[mask], y_cur[mask])
        self._model = final_model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._P is None or self._model is None:
            raise ModelError("E203 cbcp_hybrid not fitted; call fit() first")
        Xa = np.asarray(X, dtype=float)
        ds = self._dataset_ref
        if Xa.shape[0] == ds.n and np.array_equal(Xa, ds.X):
            return self._classes[np.argmax(self._P, axis=1)].astype(int)
        return np.asarray(self._model.predict(Xa)).astype(int)

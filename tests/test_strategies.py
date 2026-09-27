"""Strategy tests: contract, determinism, offline fallback, and the key
hard invariant - SSL strategies must beat the supervised baseline when
manifold structure is informative."""

import numpy as np
import pytest

from semiforge.data.synthetic import make_blobs, make_circle, make_moons
from semiforge.eval.benchmark import run_strategy
from semiforge.ssl import estimators as est
from semiforge.ssl.graph import graph_propagate
from semiforge.ssl.registry import ALL_STRATEGIES, get_strategy, list_strategies


@pytest.mark.parametrize("name", sorted(ALL_STRATEGIES))
def test_contract(name):
    ds = make_moons(n=300, seed=0)
    m = get_strategy(name, seed=0)
    m.fit(ds)
    preds = m.predict(ds.X)
    assert preds.shape == (ds.n,)
    assert np.isin(preds, ds.classes).all()


@pytest.mark.parametrize("name", ["graph", "cbcp_hybrid"])
def test_determinism(name):
    preds = []
    for _ in range(2):
        ds = make_moons(n=250, seed=3)
        m = get_strategy(name, seed=5)
        m.fit(ds)
        preds.append(m.predict(ds.X))
    assert np.array_equal(preds[0], preds[1])


def test_graph_fallback_beats_supervised_on_circle():
    """Hard invariant: propagation on the circle manifold must beat supervised
    (which sees only 5% labels)."""
    ds = make_circle(n=500, frac_labeled=0.05, seed=7)
    row_base = run_strategy(ds, "supervised", seed=0)
    row_graph = run_strategy(ds, "graph", seed=0, backend="numpy")
    assert row_graph.accuracy > row_base.accuracy + 0.05


def test_hybrid_beats_supervised_on_moons():
    ds = make_moons(n=500, frac_labeled=0.05, seed=11)
    row_base = run_strategy(ds, "supervised", seed=0)
    row_hyb = run_strategy(ds, "cbcp_hybrid", seed=0)
    assert row_hyb.accuracy > row_base.accuracy + 0.03


def test_numpy_lr_fallback_sane():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.5, (80, 2)), rng.normal(3, 0.5, (80, 2))])
    y = np.array([0] * 80 + [1] * 80)
    m = est.NumpyLogisticRegression(n_iter=300).fit(X, y)
    acc = (m.predict(X) == y).mean()
    assert acc > 0.95
    P = m.predict_proba(X)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)


def test_numpy_lr_not_fitted_raises():
    m = est.NumpyLogisticRegression()
    with pytest.raises(RuntimeError, match="E201"):
        m.predict(np.zeros((3, 2)))


def test_graph_backend_numpy_forced(monkeypatch):
    monkeypatch.setattr(est, "available_sklearn", lambda: False)
    ds = make_moons(n=200, seed=1)
    m = get_strategy("graph", seed=0, backend="auto")
    m.fit(ds)
    assert m.used_backend == "numpy"


def test_self_training_commits_pseudo_labels():
    ds = make_blobs(n=400, frac_labeled=0.05, seed=2)
    m = get_strategy("self_training", seed=0)
    m.fit(ds)
    assert m.n_pseudo_ > 0


def test_graph_inductive_path():
    ds = make_moons(n=200, seed=4)
    m = get_strategy("graph", seed=0)
    m.fit(ds)
    X_new = ds.X[:10] + 0.01
    preds = m.predict(X_new)
    assert preds.shape == (10,)


def test_unknown_strategy_raises():
    from semiforge.core.errors import RegistryError

    with pytest.raises(RegistryError, match="E402"):
        get_strategy("nope")
    assert set(list_strategies()) == set(ALL_STRATEGIES)


def test_graph_math_invariant():
    """Labeled rows stay clamped; unlabeled rows are proper distributions."""
    ds = make_moons(n=150, seed=8)
    F = graph_propagate(ds.X, ds.y_labeled, k=8)
    unl = ds.unlabeled_mask
    assert np.allclose(F[unl].sum(axis=1), 1.0, atol=1e-6)
    lab = ds.labeled_mask
    for i in np.flatnonzero(lab):
        assert F[i].argmax() == ds.y_labeled[i]

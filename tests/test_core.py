"""Core tests: types, errors, config."""

import numpy as np
import pytest

from semiforge.core.config import Config
from semiforge.core.errors import ModelError, RegistryError
from semiforge.core.types import SSLDataset, StrategyResult


def test_dataset_valid():
    X = np.random.default_rng(0).normal(size=(50, 3))
    y = np.repeat([0, 1], 25)
    yl = y.copy()
    yl[10:45] = -1
    ds = SSLDataset(X, yl, y)
    assert ds.n == 50 and ds.d == 3
    assert ds.labeled_mask.sum() == 15
    assert ds.unlabeled_mask.sum() == 35


def test_dataset_rejects_bad_shapes():
    with pytest.raises(ValueError):
        SSLDataset(np.zeros((5,)), np.zeros(5), np.zeros(5))


def test_dataset_rejects_unknown_labels():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        SSLDataset(X, np.array([0, 5] + [0] * 8), np.array([0, 1] + [0] * 8))


def test_budget_checks():
    X = np.zeros((10, 2))
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="E104"):
        SSLDataset(X, np.full(10, -1), y).assert_valid_budget()
    yl = np.array([0, -1] * 5)  # class 1 unlabeled
    with pytest.raises(ValueError, match="E105"):
        SSLDataset(X, yl, y).assert_valid_budget()


def test_strategy_result_casts_int():
    r = StrategyResult(strategy="x", preds=np.array([0.0, 1.0, 1.0]))
    assert r.preds.dtype.kind == "i"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("ENV_SEMIFORGE_SEED", "7")
    monkeypatch.setenv("ENV_SEMIFORGE_N_HPO_TRIALS", "3")
    cfg = Config.from_env()
    assert cfg.seed == 7 and cfg.n_hpo_trials == 3


def test_errors_have_codes():
    assert ModelError("x").code == "E200"
    assert RegistryError("x").code == "E400"

"""Data tests: generators, budgets, determinism."""

import numpy as np
import pytest

from semiforge.data.loaders import available_digits, load_digits_ssl
from semiforge.data.synthetic import GENERATORS, make_circle, make_moons


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_generator_contract(name):
    ds = GENERATORS[name](n=300, seed=123)
    assert ds.n == 300
    assert ds.labeled_mask.sum() >= ds.classes.size
    ds.assert_valid_budget()
    # labeled labels agree with ground truth
    lm = ds.labeled_mask
    assert np.array_equal(ds.y_labeled[lm], ds.y_true[lm])


def test_determinism():
    a, b = make_moons(seed=42), make_moons(seed=42)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y_labeled, b.y_labeled)


def test_budget_fraction_respected():
    ds = make_circle(n=400, frac_labeled=0.1, seed=0)
    assert 0.03 < ds.labeled_mask.mean() <= 0.25


@pytest.mark.skipif(not available_digits(), reason="sklearn unavailable")
def test_digits_loader():
    ds = load_digits_ssl(n=500, frac_labeled=0.1, seed=7)
    assert ds.X.shape[1] == 64
    assert ds.classes.size == 10
    ds.assert_valid_budget()
    assert ds.X.max() <= 1.0

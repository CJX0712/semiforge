"""数据与切分不变量测试 (作者: 晨星)。

覆盖 synthetic.make_semi_synthetic 的三条不变量 + 可复现性, 以及
loaders.train_test_split_semi 的防泄漏保证。
"""

import numpy as np
import pytest

from core.config import get_config
from core.errors import DataError
from core.seed import set_all
from data.loaders import train_test_split_semi
from data.synthetic import make_semi_synthetic

cfg = get_config()
set_all(cfg.default_seed)


def _gaussian():
    return make_semi_synthetic(
        n_total=300, n_labeled=30, n_features=10, n_classes=3,
        overlap=0.3, dgp="gaussian", seed=7,
    )


def _moons():
    return make_semi_synthetic(
        n_total=300, n_labeled=12, n_features=12, n_classes=2,
        overlap=0.25, dgp="moons", seed=7,
    )


def test_invariant_label_count():
    for d in (_gaussian(), _moons()):
        n_unlabeled = int(np.sum(d.y_labeled == -1))
        assert n_unlabeled == d.n_samples - d.n_labeled
        # 每类至少标注数
        for c in range(d.n_classes):
            assert np.sum(d.y_labeled == c) >= 3


def test_invariant_class_coverage():
    for d in (_gaussian(), _moons()):
        assert set(np.unique(d.y_true).tolist()) == set(range(d.n_classes))


def test_make_semi_reproducible():
    a = make_semi_synthetic(n_total=200, n_labeled=20, n_features=8,
                            n_classes=2, overlap=0.4, dgp="moons", seed=99)
    b = make_semi_synthetic(n_total=200, n_labeled=20, n_features=8,
                            n_classes=2, overlap=0.4, dgp="moons", seed=99)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y_true, b.y_true)
    assert np.array_equal(a.y_labeled, b.y_labeled)


def test_make_semi_different_seed_differs():
    a = make_semi_synthetic(seed=1, n_total=200, n_labeled=20, dgp="moons")
    b = make_semi_synthetic(seed=2, n_total=200, n_labeled=20, dgp="moons")
    assert not np.array_equal(a.X, b.X)


def test_split_no_leakage():
    d = _moons()
    train, test = train_test_split_semi(d, test_size=0.3, seed=cfg.default_seed)
    n = d.n_samples
    # 测试集全标注
    assert np.all(test.y_labeled == test.y_true)
    # 训练集包含无标注 (防泄漏)
    assert np.any(train.y_labeled == -1)
    # 训练/测试样本数之和等于总数 (setdiff1d 保证不交)
    assert train.n_samples + test.n_samples == n
    # 测试集无 -1
    assert not np.any(test.y_labeled == -1)


def test_split_reproducible():
    d = _moons()
    t1, _ = train_test_split_semi(d, test_size=0.3, seed=cfg.default_seed)
    t2, _ = train_test_split_semi(d, test_size=0.3, seed=cfg.default_seed)
    assert np.array_equal(t1.X, t2.X)
    assert np.array_equal(t1.y_labeled, t2.y_labeled)


@pytest.mark.parametrize("bad", [
    {"n_labeled": 1000},  # 超过 n_total
    {"overlap": -1.0},    # 非法 overlap
    {"dgp": "unknown"},   # 非法 dgp
])
def test_make_semi_rejects_invalid(bad):
    kwargs = {
        "n_total": 100, "n_labeled": 10, "n_features": 5, "n_classes": 2,
        "overlap": 0.5, "dgp": "gaussian", "seed": 1,
    }
    kwargs.update(bad)
    with pytest.raises(DataError):
        make_semi_synthetic(**kwargs)

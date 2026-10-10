"""确定性控制测试 (作者: 晨星)。

验证 core.seed.set_all 提供可复现的随机源: 同 seed 两次抽取逐位一致,
不同 seed 产生不同流。
"""

import numpy as np

from core.seed import rng, set_all


def test_set_all_reproducible():
    set_all(12345)
    a = np.random.default_rng(rng().spawn(1)[0].integers(0, 2**31 - 1)).random(5)
    set_all(12345)
    b = np.random.default_rng(rng().spawn(1)[0].integers(0, 2**31 - 1)).random(5)
    assert np.array_equal(a, b)


def test_set_all_different_seeds_differ():
    set_all(1)
    x = np.random.standard_normal(10)
    set_all(2)
    y = np.random.standard_normal(10)
    assert not np.allclose(x, y)


def test_rng_deterministic_from_seed():
    g1 = rng()
    g2 = rng()
    assert np.array_equal(g1.standard_normal(4), g2.standard_normal(4))


def test_set_all_returns_seed():
    assert set_all(42) == 42

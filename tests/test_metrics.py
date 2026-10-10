"""评估指标测试 (作者: 晨星)。

验证手写 accuracy_score 与 sklearn 一致, 以及 mean_std 的正确性。
"""

import numpy as np
from sklearn.metrics import accuracy_score as sk_acc

from core.seed import set_all
from eval.metrics import accuracy_score, mean_std

set_all(0)


def test_accuracy_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=200)
    y_pred = rng.integers(0, 3, size=200)
    assert abs(accuracy_score(y_true, y_pred) - sk_acc(y_true, y_pred)) < 1e-12


def test_accuracy_perfect():
    y = np.array([0, 1, 2, 0, 1])
    assert accuracy_score(y, y) == 1.0


def test_mean_std():
    vals = [0.5, 0.6, 0.7, 0.8]
    m, s = mean_std(vals)
    assert abs(m - 0.65) < 1e-9
    assert abs(s - np.std(vals, ddof=0)) < 1e-9


def test_mean_std_empty():
    m, s = mean_std([])
    assert m == 0.0 and s == 0.0

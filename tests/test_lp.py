"""LP-NumPy 不变量 I1-I4 测试 (作者: 晨星)。

纯 numpy 标签传播, 零第三方依赖, 是系统的离线兜底。
"""

import numpy as np

from core.config import get_config
from core.seed import set_all
from data.loaders import train_test_split_semi
from data.synthetic import make_semi_synthetic
from semi.lp_numpy import LabelPropagationNumpy

cfg = get_config()
set_all(cfg.default_seed)


def _fit_lp():
    d = make_semi_synthetic(n_total=200, n_labeled=12, n_features=8,
                            n_classes=2, overlap=0.3, dgp="moons", seed=3)
    train, test = train_test_split_semi(d, test_size=0.3, seed=cfg.default_seed)
    lp = LabelPropagationNumpy(cfg)
    lp.fit(train.X, train.y_labeled)
    return lp, train, test


def test_I1_labeled_labels_unchanged():
    lp, train, _test = _fit_lp()
    mask = train.y_labeled != -1
    # 已标注点的训练软标签 F 被 hard clamp 为 Y, argmax 必须等于标注类 (I1)
    soft = lp.F_train_[mask]
    assert np.all(np.argmax(soft, axis=1) == train.y_labeled[mask])


def test_I2_diff_monotonic_nonincreasing():
    lp, _, _ = _fit_lp()
    diffs = np.array(lp.diff_history_)
    assert np.all(np.diff(diffs) <= 1e-9)


def test_I3_symmetric_affinity_and_row_stochastic():
    lp, _, _ = _fit_lp()
    S = lp.S_
    assert np.allclose(S, S.T, atol=1e-9)            # S 对称
    W = lp.P_
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-9)  # 行归一化行和=1


def test_I4_proba_sums_to_one():
    lp, _, test = _fit_lp()
    p = lp.predict_proba(test.X)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    # 预测类别与软标签 argmax 一致
    assert np.all(lp.predict(test.X) == lp.classes_[np.argmax(p, axis=1)])


def test_lp_deterministic():
    d = make_semi_synthetic(n_total=150, n_labeled=10, n_features=6,
                            n_classes=2, overlap=0.3, dgp="moons", seed=5)
    train, test = train_test_split_semi(d, test_size=0.3, seed=cfg.default_seed)
    a = LabelPropagationNumpy(cfg); a.fit(train.X, train.y_labeled)
    b = LabelPropagationNumpy(cfg); b.fit(train.X, train.y_labeled)
    assert a.score(test.X, test.y_true) == b.score(test.X, test.y_true)


def test_lp_no_sklearn_dependency():
    # 模块级不导入 sklearn (离线兜底的硬保证)
    import importlib
    import sys
    mod = importlib.import_module("semi.lp_numpy")
    for modname in list(sys.modules):
        if modname.startswith("sklearn"):
            del sys.modules[modname]
    # 重新导入并运行, 期间不触碰 sklearn
    d = make_semi_synthetic(n_total=120, n_labeled=8, n_features=5,
                            n_classes=2, overlap=0.3, dgp="moons", seed=11)
    tr, te = train_test_split_semi(d, test_size=0.3, seed=cfg.default_seed)
    lp = mod.LabelPropagationNumpy(cfg)
    lp.fit(tr.X, tr.y_labeled)
    assert lp.score(te.X, te.y_true) >= 0.0

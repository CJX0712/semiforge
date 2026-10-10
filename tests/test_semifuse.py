"""SemiFuse 旗舰路由 / 诚实回退 / 防泄漏测试 (作者: 晨星)。"""

import numpy as np

from core.config import get_config
from core.seed import set_all
from data.loaders import train_test_split_semi
from data.synthetic import make_semi_synthetic
from semi.base import SupervisedOnly
from semi.semifuse import SemiFuse

cfg = get_config()
set_all(cfg.default_seed)


def _split(n_labeled=12, n_total=200, overlap=0.3):
    d = make_semi_synthetic(n_total=n_total, n_labeled=n_labeled,
                            n_features=8, n_classes=2, overlap=overlap,
                            dgp="moons", seed=6)
    return train_test_split_semi(d, test_size=0.3, seed=cfg.default_seed)


def test_100pct_labeled_equals_supervised():
    # 标注率 100% → SemiFuse 必须 == 纯监督 (无泄漏, 无半监督增益空间)
    train, test = _split(n_labeled=20, n_total=40)
    # 手动把训练集全部标注 (真实标签), 模拟全标注场景
    train = type(train)(X=train.X, y_labeled=train.y_true.copy(),
                        y_true=train.y_true, n_classes=train.n_classes)
    assert np.all(train.y_labeled != -1)
    fuse = SemiFuse(cfg)
    fuse.fit(train.X, train.y_labeled)
    sup = SupervisedOnly(cfg)
    sup.fit(train.X, train.y_labeled)
    assert fuse.fallback_used_ is True
    assert np.array_equal(fuse.predict(test.X), sup.predict(test.X))
    assert fuse.score(test.X, test.y_true) == sup.score(test.X, test.y_true)


def test_scarce_labeled_routes_to_graph():
    # 标注稀缺 (< label_budget) → 选用图半监督 LabelSpreading
    train, _test = _split(n_labeled=12)
    fuse = SemiFuse(cfg)
    fuse.fit(train.X, train.y_labeled)
    assert fuse.fallback_used_ is False
    assert [c.name for c in fuse._selected] == ["LabelSpreading"]


def test_abundant_labeled_fallback_to_supervised():
    # 标注充足 (> label_budget) → 诚实回退纯监督
    train, _test = _split(n_labeled=60, n_total=400)
    fuse = SemiFuse(cfg)
    fuse.fit(train.X, train.y_labeled)
    assert fuse.fallback_used_ is True
    assert [c.name for c in fuse._selected] == ["SupervisedOnly"]


def test_no_routing_ablation_selects_selftraining():
    train, _test = _split(n_labeled=12)
    fuse = SemiFuse(cfg, use_routing=False)
    fuse.fit(train.X, train.y_labeled)
    assert [c.name for c in fuse._selected] == ["SelfTraining"]


def test_semifuse_beats_or_matches_xgb_supervised_scarce():
    # 稀缺区 SemiFuse 应 >= 纯 XGB 监督 (强非线性监督过拟合); 验证方向性
    train, test = _split(n_labeled=12)
    from semi.base import SupervisedXGB
    xgb = SupervisedXGB(cfg); xgb.fit(train.X, train.y_labeled)
    fuse = SemiFuse(cfg); fuse.fit(train.X, train.y_labeled)
    assert fuse.score(test.X, test.y_true) >= xgb.score(test.X, test.y_true)


def test_semifuse_no_leakage_in_pipeline():
    # 训练集含无标注 (-1), SemiFuse 仅用 y_labeled; 预测不依赖测试标签
    train, test = _split(n_labeled=12)
    assert np.any(train.y_labeled == -1)
    fuse = SemiFuse(cfg)
    fuse.fit(train.X, train.y_labeled)
    p1 = fuse.predict(test.X)
    # 即便把测试标签随机扰动, 预测也不变 (未泄漏)
    assert np.array_equal(p1, fuse.predict(test.X))

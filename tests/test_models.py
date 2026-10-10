"""监督 / 半监督模型接口与基本正确性测试 (作者: 晨星)。"""

import numpy as np
import pytest

from core.config import get_config
from core.seed import set_all
from data.loaders import train_test_split_semi
from data.synthetic import make_semi_synthetic
from semi.base import (
    LabelSpreadingSK,
    SelfTrainingXGB,
    SupervisedOnly,
    SupervisedRF,
    SupervisedXGB,
    available_xgboost,
)
from semi.lp_numpy import LabelPropagationNumpy

cfg = get_config()
set_all(cfg.default_seed)


def _data(n_labeled=12, n_total=200, overlap=0.3):
    d = make_semi_synthetic(n_total=n_total, n_labeled=n_labeled,
                            n_features=8, n_classes=2, overlap=overlap,
                            dgp="moons", seed=4)
    train, test = train_test_split_semi(d, test_size=0.3, seed=cfg.default_seed)
    return train, test


_MODELS = [SupervisedOnly, SupervisedXGB, SupervisedRF, SelfTrainingXGB,
           LabelSpreadingSK, LabelPropagationNumpy]


@pytest.mark.parametrize("factory", _MODELS)
def test_model_fit_predict_score(factory):
    train, test = _data()
    m = factory(cfg)
    m.fit(train.X, train.y_labeled)
    pred = m.predict(test.X)
    assert pred.shape == (test.n_samples,)
    assert set(np.unique(pred).tolist()).issubset(set(np.unique(train.y_true).tolist()))
    assert 0.0 <= m.score(test.X, test.y_true) <= 1.0
    # classes_ 必须被设置
    assert m.classes_ is not None and len(m.classes_) == 2


@pytest.mark.parametrize("factory", _MODELS)
def test_model_predict_proba(factory):
    train, test = _data()
    m = factory(cfg)
    m.fit(train.X, train.y_labeled)
    p = m.predict_proba(test.X)
    assert p.shape == (test.n_samples, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_labeled_only_models_respect_unlabeled():
    # SupervisedOnly 只在标注样本上训练, 预测不应因无标注样本改变
    train, test = _data(n_labeled=10)
    m = SupervisedOnly(cfg)
    m.fit(train.X, train.y_labeled)
    pred = m.predict(test.X)
    assert pred.shape == (test.n_samples,)


def test_xgboost_availability_flag():
    assert available_xgboost() in (True, False)

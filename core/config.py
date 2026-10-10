"""SemiForge 配置: 环境变量覆盖 + schema 校验。

作者: 晨星 (CJX0712)

约定: 所有环境变量以 SEMIFORGE_ 前缀, 仅用于覆盖默认值;
不存在时使用内建默认值。get_config() 返回经校验的字典。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .errors import ConfigError

_PREFIX = "SEMIFORGE_"


@dataclass
class Config:
    """运行配置 (所有字段带默认值, 可被环境变量覆盖)。"""

    default_seed: int = 20261010
    n_folds_cv: int = 3
    lp_max_iter: int = 1000
    lp_tol: float = 1e-5
    lp_alpha: float = 0.99  # 标签传播: 已标注点保留权重
    lp_gamma: float = 10.0  # RBF 核带宽参数 (兜底; 核方法默认用 median heuristic 自适应)
    knn_k: int = 5  # 图构造邻居数 (LabelSpreading kNN 图)
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 4
    self_training_threshold: float = 0.75
    router_topk: int = 2  # SemiFuse 加权集成取 top-k 候选
    label_budget: int = 25  # 标注预算阈值: 标注数 > 此值走纯监督, 否则半监督融合
    extra: dict = field(default_factory=dict)


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


_TYPE_PARSERS = {
    int: lambda v: int(v),
    float: lambda v: float(v),
    bool: _parse_bool,
}


def _fields():
    import dataclasses

    return {f.name: f for f in dataclasses.fields(Config) if f.name != "extra"}


def get_config(overrides: dict | None = None) -> Config:
    """合并默认值 / 环境变量 / 显式覆盖, 并做基础校验。"""
    cfg = Config()
    fields = _fields()

    # 环境变量覆盖
    for name, f in fields.items():
        env_key = _PREFIX + name.upper()
        if env_key in os.environ:
            try:
                parser = _TYPE_PARSERS[f.type]
                setattr(cfg, name, parser(os.environ[env_key]))
            except (ValueError, TypeError) as e:
                raise ConfigError(f"环境变量 {env_key} 解析失败: {e}") from e

    # 显式覆盖
    if overrides:
        for k, v in overrides.items():
            if k not in fields:
                cfg.extra[k] = v
            else:
                setattr(cfg, k, v)

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.n_folds_cv < 2:
        raise ConfigError("n_folds_cv 必须 >= 2")
    if not (0.0 < cfg.lp_alpha < 1.0):
        raise ConfigError("lp_alpha 必须严格在 (0, 1)")
    if cfg.lp_gamma <= 0:
        raise ConfigError("lp_gamma 必须 > 0")
    if cfg.xgb_n_estimators <= 0:
        raise ConfigError("xgb_n_estimators 必须 > 0")
    if cfg.router_topk < 1:
        raise ConfigError("router_topk 必须 >= 1")

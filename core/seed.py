"""SemiForge 全局确定性控制。

作者: 晨星 (CJX0712)

唯一入口 core.seed.set_all(seed) 一次性设齐 numpy / random / 本地 RNG,
保证 demo 两次运行核心指标逐位一致。
"""

from __future__ import annotations

import random as _random

import numpy as np

_SEED: int | None = None


def set_all(seed: int) -> int:
    """一次性设齐所有随机源。返回实际生效的 seed (供记录)。

    覆盖: numpy (default_rng + 旧 RandomState 兜底) / 标准库 random /
    本模块维护的独立 Generator。不触碰可能不存在的 torch 等。
    """
    global _SEED
    _SEED = int(seed)
    _random.seed(_SEED)
    np.random.seed(_SEED)
    return _SEED


def get_seed() -> int | None:
    return _SEED


def rng() -> np.random.Generator:
    """返回一个由当前全局 seed 派生的确定性生成器。

    注意: 为避免多次调用产生相关流, 调用方应持有返回的 Generator 实例。
    """
    base = _SEED if _SEED is not None else 0
    return np.random.default_rng(np.random.SeedSequence([base]))

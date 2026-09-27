"""Strategy registry."""

from __future__ import annotations

from collections.abc import Callable

from semiforge.core.errors import RegistryError
from semiforge.core.types import SSLDataset
from semiforge.ssl.hybrid import CBCPHybrid
from semiforge.ssl.strategies import (
    CoTraining,
    GraphPropagation,
    SelfTraining,
    SupervisedBaseline,
)

Factories: dict[str, Callable[..., object]] = {}


def register(name: str) -> Callable:
    def deco(factory: Callable[..., object]) -> Callable[..., object]:
        if name in Factories:
            raise RegistryError(f"E401 strategy '{name}' already registered")
        Factories[name] = factory
        return factory

    return deco


@register("supervised")
def _supervised(dataset: SSLDataset | None = None, seed: int = 0, **kw):
    return SupervisedBaseline(seed=seed)


@register("self_training")
def _self_training(dataset: SSLDataset | None = None, seed: int = 0, **kw):
    return SelfTraining(seed=seed, **kw)


@register("graph")
def _graph(dataset: SSLDataset | None = None, seed: int = 0, **kw):
    return GraphPropagation(**kw)


@register("co_training")
def _co_training(dataset: SSLDataset | None = None, seed: int = 0, **kw):
    return CoTraining(seed=seed, **kw)


@register("cbcp_hybrid")
def _hybrid(dataset: SSLDataset | None = None, seed: int = 0, **kw):
    return CBCPHybrid(seed=seed, **kw)


ALL_STRATEGIES = tuple(Factories.keys())
FLAGSHIP = "cbcp_hybrid"
BASELINE = "supervised"


def get_strategy(name: str, **kwargs):
    if name not in Factories:
        raise RegistryError(f"E402 unknown strategy '{name}'. Available: {sorted(Factories)}")
    return Factories[name](**kwargs)


def list_strategies() -> list[str]:
    return sorted(Factories)

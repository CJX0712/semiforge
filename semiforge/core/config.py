"""Runtime configuration with ENV_SEMIFORGE_* overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(f"ENV_SEMIFORGE_{key}")
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(f"ENV_SEMIFORGE_{key}")
    return float(raw) if raw else default


@dataclass
class Config:
    seed: int = 42
    n_neighbors: int = 10
    graph_alpha: float = 0.75
    n_hpo_trials: int = 30
    hpo_timeout: int = 60

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            seed=_env_int("SEED", 42),
            n_neighbors=_env_int("N_NEIGHBORS", 10),
            graph_alpha=_env_float("GRAPH_ALPHA", 0.75),
            n_hpo_trials=_env_int("N_HPO_TRIALS", 30),
            hpo_timeout=_env_int("HPO_TIMEOUT", 60),
        )

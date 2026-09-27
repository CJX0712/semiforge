"""Optuna HPO for the CBCP-Hybrid flagship (TPE, deterministic seed).

Objective: mean unlabeled accuracy across the given datasets.
timeout <= 0 skips tuning and returns current defaults (fallback branch).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from semiforge.eval.benchmark import run_strategy
from semiforge.ssl.registry import FLAGSHIP


def tune_flagship(
    make_datasets: Callable[[], list], n_trials: int = 20, timeout: int = 60, seed: int = 42
) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if timeout <= 0 or n_trials <= 0:
        return {"status": "skipped", "params": {}, "best_value": None}

    datasets = make_datasets()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_rounds": trial.suggest_int("n_rounds", 2, 6),
            "base_quota": trial.suggest_int("base_quota", 4, 16),
            "quantile": trial.suggest_float("quantile", 0.6, 0.95),
            "weight_power": trial.suggest_float("weight_power", 6.0, 20.0),
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 20),
            "alpha": trial.suggest_float("alpha", 0.5, 0.95),
        }
        scores = []
        for ds in datasets:
            row = run_strategy(ds, FLAGSHIP, seed=seed, **params)
            scores.append(row.accuracy)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return {
        "status": "ok",
        "params": dict(study.best_params),
        "best_value": float(study.best_value),
    }

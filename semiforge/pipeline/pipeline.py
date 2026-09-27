"""End-to-end pipeline: datasets -> (optional HPO) -> benchmark -> JSON."""

from __future__ import annotations

import json
from pathlib import Path

from semiforge.core.config import Config
from semiforge.data.loaders import available_digits, load_digits_ssl
from semiforge.data.synthetic import GENERATORS
from semiforge.eval.benchmark import aggregate_gain, format_table
from semiforge.hpo.tune import tune_flagship
from semiforge.ssl.registry import ALL_STRATEGIES, FLAGSHIP


def make_datasets(names: list[str], seed: int = 42) -> list:
    datasets = []
    for nm in names:
        if nm == "digits":
            if available_digits():
                datasets.append(load_digits_ssl(seed=seed))
            continue
        if nm not in GENERATORS:
            from semiforge.core.errors import DataError

            raise DataError(f"E107 unknown dataset '{nm}'. Available: {sorted(GENERATORS)}")
        datasets.append(GENERATORS[nm](seed=seed))
    return datasets


class SemiForgePipeline:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()

    def run(
        self,
        dataset_names: list[str] | None = None,
        strategies: list[str] | None = None,
        hpo: bool = False,
        out_path: str | None = None,
    ) -> dict:
        cfg = self.config
        ds_names = dataset_names or ["moons", "blobs", "circle", "spiral", "highdim"]
        strategies = list(strategies) if strategies else list(ALL_STRATEGIES)
        if "digits" in ds_names and not available_digits():
            ds_names = [d for d in ds_names if d != "digits"]

        hpo_result: dict = {"status": "skipped", "params": {}, "best_value": None}
        extra_kw: dict = {}
        if hpo:
            hpo_result = tune_flagship(
                lambda: make_datasets(["moons", "circle"], seed=cfg.seed),
                n_trials=cfg.n_hpo_trials,
                timeout=cfg.hpo_timeout,
                seed=cfg.seed,
            )
            if hpo_result["status"] == "ok":
                extra_kw = hpo_result["params"]

        datasets = make_datasets(ds_names, seed=cfg.seed)
        # flagship gets tuned/extra params; others stay at defaults
        rows = []
        for ds in datasets:
            for s in strategies:
                kw = dict(extra_kw) if s == FLAGSHIP else {}
                from semiforge.eval.benchmark import run_strategy

                rows.append(run_strategy(ds, s, seed=cfg.seed, **kw))

        gains = aggregate_gain(rows, FLAGSHIP, "supervised")
        result = {
            "config": {
                "seed": cfg.seed,
                "datasets": ds_names,
                "strategies": strategies,
                "hpo": hpo_result,
            },
            "rows": [r.as_dict() for r in rows],
            "flagship_gain_vs_supervised": {k: round(v, 4) for k, v in gains.items()},
        }
        if out_path:
            Path(out_path).write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        self.last_result_ = result
        self.last_table_ = format_table(rows)
        return result


def print_report(result: dict) -> str:
    hdr = f"{'dataset':<10}{'strategy':<14}{'acc':>8}{'macro_f1':>10}{'secs':>8}"
    lines = ["=== SemiForge benchmark ===", hdr]
    for r in result["rows"]:
        lines.append(
            f"{r['dataset']:<10}{r['strategy']:<14}{r['accuracy']:>8.4f}"
            f"{r['macro_f1']:>10.4f}{r['fit_seconds']:>8.2f}"
        )
    lines.append(f"flagship gain vs supervised: {result['flagship_gain_vs_supervised']}")
    return "\n".join(lines)

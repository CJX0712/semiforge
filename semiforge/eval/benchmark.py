"""Benchmark runner: strategies x datasets -> BenchmarkRow list."""

from __future__ import annotations

import time

from semiforge.core.types import BenchmarkRow, SSLDataset
from semiforge.eval.metrics import evaluate
from semiforge.ssl.registry import get_strategy


def run_strategy(ds: SSLDataset, name: str, seed: int = 0, **kw) -> BenchmarkRow:
    model = get_strategy(name, seed=seed, **kw)
    t0 = time.perf_counter()
    model.fit(ds)
    elapsed = time.perf_counter() - t0
    preds = model.predict(ds.X)
    unl = ds.unlabeled_mask
    m = evaluate(ds.y_true[unl], preds[unl])
    extra: dict = {}
    if hasattr(model, "used_backend"):
        extra["backend"] = model.used_backend
    if hasattr(model, "n_pseudo_"):
        extra["n_pseudo"] = int(model.n_pseudo_)
    return BenchmarkRow(
        dataset=ds.name,
        strategy=name,
        accuracy=m["accuracy"],
        macro_f1=m["macro_f1"],
        n_labeled=int(ds.labeled_mask.sum()),
        n_unlabeled=int(unl.sum()),
        fit_seconds=elapsed,
        extra=extra,
    )


def run_benchmark(
    datasets: list[SSLDataset], strategies: list[str], seed: int = 0, **shared_kw
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for ds in datasets:
        for s in strategies:
            rows.append(run_strategy(ds, s, seed=seed, **shared_kw))
    return rows


def format_table(rows: list[BenchmarkRow]) -> str:
    """Fixed-width table (CLI-safe alignment)."""
    headers = ["dataset", "strategy", "acc", "macro_f1", "secs", "note"]
    header = (
        f"{headers[0]:<10}{headers[1]:<14}{headers[2]:>8}{headers[3]:>10}"
        f"{headers[4]:>8}  {headers[5]}"
    )
    lines = [header]
    for r in rows:
        note = ""
        if "backend" in r.extra:
            note += f"backend={r.extra['backend']} "
        if "n_pseudo" in r.extra:
            note += f"pseudo={r.extra['n_pseudo']}"
        lines.append(
            f"{r.dataset:<10}{r.strategy:<14}{r.accuracy:>8.4f}"
            f"{r.macro_f1:>10.4f}{r.fit_seconds:>8.2f}  {note}"
        )
    return "\n".join(lines)


def aggregate_gain(rows: list[BenchmarkRow], flagship: str, baseline: str) -> dict[str, float]:
    """Per-dataset accuracy gain of flagship over supervised baseline."""
    gains: dict[str, float] = {}
    acc: dict[tuple[str, str], float] = {}
    for r in rows:
        acc[(r.dataset, r.strategy)] = r.accuracy
    for (ds, s), a in acc.items():
        if s == flagship and (ds, baseline) in acc:
            gains[ds] = a - acc[(ds, baseline)]
    return gains

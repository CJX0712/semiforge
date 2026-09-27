"""Pipeline / benchmark / HPO / CLI tests."""

import json

import pytest

from semiforge.core.config import Config
from semiforge.data.synthetic import make_moons
from semiforge.eval.benchmark import aggregate_gain, run_benchmark
from semiforge.hpo.tune import tune_flagship
from semiforge.pipeline.pipeline import SemiForgePipeline, make_datasets
from semiforge.ssl.registry import FLAGSHIP


def test_run_benchmark_small():
    ds = make_moons(n=200, seed=0)
    rows = run_benchmark([ds], ["supervised", "graph"], seed=0)
    assert len(rows) == 2
    d = rows[0].as_dict()
    assert {"dataset", "strategy", "accuracy", "macro_f1"} <= set(d)


def test_aggregate_gain():
    ds = make_moons(n=250, seed=1)
    rows = run_benchmark([ds], ["supervised", FLAGSHIP], seed=0)
    gains = aggregate_gain(rows, FLAGSHIP, "supervised")
    assert "moons" in gains


def test_make_datasets_unknown_raises():
    from semiforge.core.errors import DataError

    with pytest.raises(DataError, match="E107"):
        make_datasets(["bogus"])


def test_pipeline_end_to_end(tmp_path):
    cfg = Config(seed=42, n_hpo_trials=2, hpo_timeout=30)
    pipe = SemiForgePipeline(cfg)
    out = tmp_path / "bench.json"
    result = pipe.run(
        dataset_names=["moons", "circle"],
        strategies=["supervised", "graph", FLAGSHIP],
        hpo=False,
        out_path=str(out),
    )
    assert len(result["rows"]) == 6
    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["config"]["seed"] == 42
    assert set(parsed["flagship_gain_vs_supervised"]) == {"moons", "circle"}


def test_hpo_fallback_branch():
    res = tune_flagship(lambda: [make_moons(n=120, seed=0)], n_trials=2, timeout=0)
    assert res["status"] == "skipped"


def test_hpo_small_run():
    res = tune_flagship(lambda: [make_moons(n=150, seed=0)], n_trials=2, timeout=60, seed=0)
    assert res["status"] == "ok"
    assert res["best_value"] is not None and 0.0 < res["best_value"] <= 1.0


def test_cli_run(tmp_path, capsys):
    from semiforge.cli import main

    out = tmp_path / "cli.json"
    code = main(["run", "--datasets", "moons", "--out", str(out)])
    assert code == 0
    captured = capsys.readouterr()
    assert "cbcp_hybrid" in captured.out
    assert out.exists()

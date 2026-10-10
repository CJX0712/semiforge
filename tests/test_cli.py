"""CLI 冒烟 + 管线确定性测试 (作者: 晨星)。"""

import numpy as np

import cli as cli_mod
from core.config import get_config
from core.seed import set_all
from pipeline.pipeline import SemiPipeline

cfg = get_config()
set_all(cfg.default_seed)


def test_cli_run_smoke(capsys):
    rc = cli_mod.main(["run", "--n_labeled", "10", "--seeds", "2",
                       "--overlap", "0.2", "--n_total", "120"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SemiFuse" in out
    assert "LabelSpreading" in out


def test_pipeline_benchmark_deterministic():
    pipe = SemiPipeline(cfg)
    regimes = [{"dgp": "moons", "overlap": 0.2, "n_labeled": 10,
                "n_total": 150, "n_features": 10, "label_noise": 0.0,
                "n_classes": 2}]
    b1 = pipe.benchmark(regimes, [1, 2])
    b2 = pipe.benchmark(regimes, [1, 2])
    a1 = sorted([r["accuracy"] for r in b1["rows"]])
    a2 = sorted([r["accuracy"] for r in b2["rows"]])
    assert np.allclose(a1, a2, atol=0.0)


def test_demo_reproducible(tmp_path):
    # 跑两次 demo 子流程, 比对 accuracy 行 (排除计时/时间戳)
    import examples.run_demo as demo

    def run_once():
        rep = demo.main()
        return sorted([r["accuracy"] for r in rep["benchmark"]["rows"]])

    a = run_once()
    b = run_once()
    assert np.allclose(a, b, atol=0.0)
    # determinism 字段应 True
    assert demo.determinism_check(SemiPipeline(cfg), demo.REGIMES[0], cfg)["passed"]

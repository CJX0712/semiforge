#!/usr/bin/env python
"""SemiForge 命令行入口。

作者: 晨星 (CJX0712)

用法:
  python cli.py demo         # 跑端到端基准 (benchmark.json)
  python cli.py run --help   # 自定义基准参数
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from core.config import get_config
from core.seed import set_all
from pipeline.pipeline import SemiPipeline
from semi.base import available_xgboost


def cmd_demo(args) -> int:
    from examples.run_demo import main

    main()
    return 0


def cmd_run(args) -> int:
    cfg = get_config(overrides={"default_seed": args.seed})
    set_all(args.seed)
    pipe = SemiPipeline(cfg)
    regimes = [
        {"overlap": args.overlap, "n_labeled": args.n_labeled,
         "n_total": args.n_total, "n_features": 20, "label_noise": 0.0, "n_classes": 2}
    ]
    bench = pipe.benchmark(regimes, list(range(1, args.seeds + 1)))
    print(f"xgboost_available={available_xgboost()}")
    print(f"{'method':<16}{'mean':>8}{'std':>8}")
    for name, a in bench["aggregate"].items():
        print(f"{name:<16}{a['mean']:>8.4f}{a['std']:>8.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semiforge", description="SemiForge 半监督学习系统 (作者: 晨星)")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="跑端到端基准")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="自定义单 regime 基准")
    r.add_argument("--overlap", type=float, default=0.5)
    r.add_argument("--n_labeled", type=int, default=20)
    r.add_argument("--n_total", type=int, default=500)
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--seed", type=int, default=20261010)
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

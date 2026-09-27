"""SemiForge CLI."""

from __future__ import annotations

import argparse
import sys

from semiforge.core.config import Config
from semiforge.data.synthetic import GENERATORS
from semiforge.pipeline.pipeline import SemiForgePipeline
from semiforge.ssl.registry import list_strategies


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="semiforge", description="SemiForge: modular semi-supervised learning"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the SSL benchmark")
    r.add_argument(
        "--datasets",
        nargs="*",
        default=["moons", "blobs", "circle"],
        help=f"choices: {sorted(GENERATORS) + ['digits']}",
    )
    r.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help=f"choices: {list_strategies()} (default: all)",
    )
    r.add_argument("--hpo", action="store_true", help="tune flagship with Optuna first")
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--hpo-trials", type=int, default=None)
    r.add_argument("--hpo-timeout", type=int, default=None)
    r.add_argument("--out", type=str, default="benchmark.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        cfg = Config.from_env()
        if args.seed is not None:
            cfg.seed = args.seed
        if args.hpo_trials is not None:
            cfg.n_hpo_trials = args.hpo_trials
        if args.hpo_timeout is not None:
            cfg.hpo_timeout = args.hpo_timeout
        pipe = SemiForgePipeline(cfg)
        result = pipe.run(
            dataset_names=args.datasets,
            strategies=args.strategies,
            hpo=args.hpo,
            out_path=args.out,
        )
        print(pipe.last_table_)
        print(f"\nflagship gain vs supervised: {result['flagship_gain_vs_supervised']}")
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

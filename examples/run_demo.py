"""End-to-end demo: 5 synthetic datasets + digits -> benchmark.json.

Run:  python examples/run_demo.py
Output: prints the benchmark table, writes benchmark.json next to this file.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semiforge.pipeline.pipeline import SemiForgePipeline


def main() -> None:
    pipe = SemiForgePipeline()
    result = pipe.run(
        dataset_names=["moons", "blobs", "circle", "spiral", "highdim", "digits"],
        hpo=False,
        out_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.json"),
    )
    print(pipe.last_table_)
    print(f"\nflagship gain vs supervised: {result['flagship_gain_vs_supervised']}")
    n_rows = len(result["rows"])
    print(f"{n_rows} benchmark rows written.")


if __name__ == "__main__":
    main()

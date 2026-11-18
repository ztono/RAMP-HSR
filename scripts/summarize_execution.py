import argparse
from collections import Counter
import json
from pathlib import Path
from ramp_hsr.execution_metrics import (
    TRANSITIONS,
    transition_metrics,
    repository_bootstrap,
)
from scripts.evaluate_operating_points import read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    records = read_records(args.records)
    results = {}
    for setting in sorted({row["model_key"] for row in records}):
        rows = [row for row in records if row["model_key"] == setting]
        counts = Counter((row["transition"] for row in rows))
        results[setting] = {
            "metrics": transition_metrics({key: counts[key] for key in TRANSITIONS}),
            "bootstrap": repository_bootstrap(
                rows, replicates=args.bootstrap_replicates, seed=args.seed
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

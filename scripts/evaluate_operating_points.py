import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from ramp_hsr.operating_points import (
    _fit_points,
    _predict,
    binary_metrics,
    repository_bootstrap_intervals,
)


def read_records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arrays(rows):
    columns = {
        "labels": "base_error",
        "composite": "neuron_chain_composite_risk",
        "pair": "neuron_chain_pair_risk",
        "msp": "msp_risk",
        "entropy": "entropy_risk",
        "margin": "margin_risk",
    }
    if not rows:
        raise ValueError("records must be non-empty")
    return {
        name: np.asarray(
            [row[key] for row in rows], dtype=bool if name == "labels" else np.float64
        )
        for (name, key) in columns.items()
    }


def evaluate(calibration, test, *, replicates=10000, seed=13):
    (fit, heldout) = (arrays(calibration), arrays(test))
    points = _fit_points(fit)
    predictions = [_predict(point, heldout) for point in points]
    intervals = repository_bootstrap_intervals(
        heldout["labels"],
        predictions,
        np.asarray([row["repository"] for row in test], dtype=object),
        replicates=replicates,
        seed=seed,
    )
    for point, predicted, interval in zip(points, predictions, intervals):
        point["test_metrics"] = binary_metrics(heldout["labels"], predicted)
        point["test_repository_bootstrap"] = interval
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    result = evaluate(
        read_records(args.calibration),
        read_records(args.test),
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"points": result}, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()

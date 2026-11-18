from __future__ import annotations
from pathlib import Path
import sys
from typing import Any
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
TRANSITIONS = ("Pass→Pass", "Pass→Fail", "Fail→Pass", "Fail→Fail")


def transition_metrics(counts: dict[str, int]) -> dict[str, Any]:
    """Compute conditional damage/recovery and paired net Pass@1."""
    missing = set(TRANSITIONS) - set(counts)
    if missing:
        raise ValueError(f"missing transitions: {sorted(missing)}")
    pp = int(counts["Pass→Pass"])
    pf = int(counts["Pass→Fail"])
    fp = int(counts["Fail→Pass"])
    ff = int(counts["Fail→Fail"])
    n = pp + pf + fp + ff
    base_pass = pp + pf
    base_fail = fp + ff
    editor_pass = pp + fp
    return {
        "n": n,
        "transitions": {name: int(counts[name]) for name in TRANSITIONS},
        "base_pass_count": base_pass,
        "base_fail_count": base_fail,
        "editor_pass_count": editor_pass,
        "base_pass_at_1": base_pass / n if n else None,
        "editor_pass_at_1": editor_pass / n if n else None,
        "conditional_damage": pf / base_pass if base_pass else None,
        "conditional_recovery": fp / base_fail if base_fail else None,
        "delta_pass_at_1": (fp - pf) / n if n else None,
        "net_pass_count": fp - pf,
    }


def _metrics_from_count_array(values: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized PP/PF/FP/FF rows to the E5 statistics."""
    counts = np.asarray(values, dtype=np.float64)
    (pp, pf, fp, ff) = (counts[..., index] for index in range(4))
    n = pp + pf + fp + ff
    base_pass = pp + pf
    base_fail = fp + ff
    editor_pass = pp + fp
    return {
        "base_pass_at_1": np.divide(base_pass, n, out=np.zeros_like(n), where=n > 0),
        "editor_pass_at_1": np.divide(
            editor_pass, n, out=np.zeros_like(n), where=n > 0
        ),
        "conditional_damage": np.divide(
            pf, base_pass, out=np.full_like(n, np.nan), where=base_pass > 0
        ),
        "conditional_recovery": np.divide(
            fp, base_fail, out=np.full_like(n, np.nan), where=base_fail > 0
        ),
        "delta_pass_at_1": np.divide(fp - pf, n, out=np.zeros_like(n), where=n > 0),
    }


def repository_bootstrap(
    records: list[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    repositories = sorted({str(row["repository"]) for row in records})
    repo_index = {repo: index for (index, repo) in enumerate(repositories)}
    per_repo = np.zeros((len(repositories), 4), dtype=np.int64)
    transition_index = {name: index for (index, name) in enumerate(TRANSITIONS)}
    for row in records:
        per_repo[
            repo_index[str(row["repository"])], transition_index[str(row["transition"])]
        ] += 1
    point = _metrics_from_count_array(per_repo.sum(axis=0, keepdims=True))
    if replicates <= 0:
        return {
            "replicates": 0,
            "unit": "repository",
            "repositories": len(repositories),
            "point": {name: float(value[0]) for (name, value) in point.items()},
            "intervals": {},
        }
    rng = np.random.default_rng(int(seed))
    weights = rng.multinomial(
        len(repositories),
        np.full(len(repositories), 1.0 / len(repositories)),
        size=int(replicates),
    )
    bootstrap_counts = weights @ per_repo
    metrics = _metrics_from_count_array(bootstrap_counts)
    intervals: dict[str, Any] = {}
    for name, values in metrics.items():
        finite = values[np.isfinite(values)]
        intervals[name] = {
            "low": float(np.quantile(finite, 0.025)) if finite.size else None,
            "high": float(np.quantile(finite, 0.975)) if finite.size else None,
            "finite_replicates": int(finite.size),
        }
    return {
        "replicates": int(replicates),
        "unit": "repository",
        "repositories": len(repositories),
        "seed": int(seed),
        "point": {name: float(value[0]) for (name, value) in point.items()},
        "intervals": intervals,
    }

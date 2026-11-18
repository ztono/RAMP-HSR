from __future__ import annotations
import math
from pathlib import Path
import sys
from typing import Any
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
QUANTILE_COUNT = 201
TARGET_FPRS = (0.01, 0.05, 0.1)
ORIGINAL_COMPOSITE_THRESHOLD = 2.0
ORIGINAL_PAIR_THRESHOLD = 1.5


def _ratio(left: int | float, right: int | float) -> float | None:
    return float(left) / float(right) if right else None


def binary_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Return the fixed-label risk-detection metrics used in E4."""
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(predicted, dtype=bool)
    if y.ndim != 1 or p.shape != y.shape:
        raise ValueError("labels and predictions must be same-length vectors")
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    recall = _ratio(tp, tp + fn)
    fpr = _ratio(fp, fp + tn)
    precision = _ratio(tp, tp + fp)
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    lr_plus_infinite = bool(recall not in (None, 0.0) and fpr == 0.0)
    return {
        "n": int(y.size),
        "positive": int(np.sum(y)),
        "negative": int(np.sum(~y)),
        "prevalence": float(y.mean()) if y.size else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "coverage": float(p.mean()) if p.size else None,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "lr_plus": (
            float(recall) / float(fpr)
            if recall is not None and fpr not in (None, 0.0)
            else None
        ),
        "lr_plus_infinite": lr_plus_infinite,
        "mcc": float(tp * tn - fp * fn) / denominator if denominator else 0.0,
    }


def quantile_thresholds(
    values: np.ndarray, *, quantile_count: int = QUANTILE_COUNT
) -> np.ndarray:
    """Build deterministic >= candidates from training-score quantiles."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not scores.size:
        raise ValueError("threshold construction requires a non-empty vector")
    if not bool(np.isfinite(scores).all()):
        raise ValueError("threshold construction received non-finite scores")
    if quantile_count < 2:
        raise ValueError("quantile_count must be at least two")
    quantiles = np.linspace(0.0, 1.0, int(quantile_count))
    try:
        finite = np.quantile(scores, quantiles, method="nearest")
    except TypeError:
        finite = np.quantile(scores, quantiles, interpolation="nearest")
    inactive = np.nextafter(float(np.max(scores)), np.inf)
    return np.asarray(sorted({*finite.tolist(), inactive}), dtype=np.float64)


def _metric_rank(metrics: dict[str, Any], objective: str) -> tuple[float, ...]:
    recall = float(metrics.get("recall") or 0.0)
    fpr = float(metrics.get("fpr") or 0.0)
    mcc = float(metrics.get("mcc") or 0.0)
    precision = float(metrics.get("precision") or 0.0)
    coverage = float(metrics.get("coverage") or 0.0)
    if objective == "recall_at_fpr":
        return (recall, -fpr, mcc, precision, -coverage)
    if objective == "max_mcc":
        return (mcc, recall, -fpr, precision, -coverage)
    raise KeyError(objective)


def select_single_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    objective: str,
    target_fpr: float | None = None,
    quantile_count: int = QUANTILE_COUNT,
) -> dict[str, Any]:
    """Select one monotone risk threshold using training rows only."""
    risk = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if risk.shape != y.shape or risk.ndim != 1:
        raise ValueError("score/label shape mismatch")
    best: tuple[tuple[float, ...], dict[str, Any]] | None = None
    candidates = quantile_thresholds(risk, quantile_count=quantile_count)
    for threshold in candidates:
        metrics = binary_metrics(y, risk >= float(threshold))
        if objective == "recall_at_fpr":
            if target_fpr is None:
                raise ValueError("target_fpr is required")
            if float(metrics["fpr"] or 0.0) > float(target_fpr) + 1e-12:
                continue
        rank = (*_metric_rank(metrics, objective), float(threshold))
        payload = {
            "risk_threshold": float(threshold),
            "candidate_count": int(candidates.size),
            "metrics": metrics,
        }
        if best is None or rank > best[0]:
            best = (rank, payload)
    if best is None:
        raise RuntimeError("no feasible single-score operating point")
    return best[1]


def select_route_thresholds(
    composite: np.ndarray,
    pair: np.ndarray,
    labels: np.ndarray,
    *,
    objective: str,
    target_fpr: float | None = None,
    quantile_count: int = QUANTILE_COUNT,
) -> dict[str, Any]:
    """Select a Composite-OR-Pair point from training quantile candidates."""
    comp = np.asarray(composite, dtype=np.float64)
    pair_score = np.asarray(pair, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if comp.shape != pair_score.shape or comp.shape != y.shape or comp.ndim != 1:
        raise ValueError("route score/label shape mismatch")
    comp_candidates = quantile_thresholds(comp, quantile_count=quantile_count)
    pair_candidates = quantile_thresholds(pair_score, quantile_count=quantile_count)
    pair_active = pair_score[None, :] >= pair_candidates[:, None]
    best: tuple[tuple[float, ...], dict[str, Any]] | None = None
    for comp_threshold in comp_candidates:
        comp_active = comp >= float(comp_threshold)
        for index, pair_threshold in enumerate(pair_candidates):
            active = comp_active | pair_active[index]
            metrics = binary_metrics(y, active)
            if objective == "recall_at_fpr":
                if target_fpr is None:
                    raise ValueError("target_fpr is required")
                if float(metrics["fpr"] or 0.0) > float(target_fpr) + 1e-12:
                    continue
            rank = (
                *_metric_rank(metrics, objective),
                float(comp_threshold),
                float(pair_threshold),
            )
            payload = {
                "composite_threshold": float(comp_threshold),
                "pair_threshold": float(pair_threshold),
                "composite_candidate_count": int(comp_candidates.size),
                "pair_candidate_count": int(pair_candidates.size),
                "joint_candidate_count": int(
                    comp_candidates.size * pair_candidates.size
                ),
                "metrics": metrics,
            }
            if best is None or rank > best[0]:
                best = (rank, payload)
    if best is None:
        raise RuntimeError("no feasible route operating point")
    return best[1]


def _raw_threshold(detector: str, risk_threshold: float) -> dict[str, Any]:
    if detector == "msp":
        return {
            "display_name": "MSP",
            "risk_transform": "1 - MSP",
            "raw_operator": "<=",
            "raw_threshold": 1.0 - float(risk_threshold),
        }
    if detector == "entropy":
        return {
            "display_name": "Entropy",
            "risk_transform": "Entropy",
            "raw_operator": ">=",
            "raw_threshold": float(risk_threshold),
        }
    if detector == "margin":
        return {
            "display_name": "Margin",
            "risk_transform": "-Margin",
            "raw_operator": "<=",
            "raw_threshold": -float(risk_threshold),
        }
    raise KeyError(detector)


def _fit_points(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    labels = arrays["labels"]
    points: list[dict[str, Any]] = []
    original_metrics = binary_metrics(
        labels,
        (arrays["composite"] >= ORIGINAL_COMPOSITE_THRESHOLD)
        | (arrays["pair"] >= ORIGINAL_PAIR_THRESHOLD),
    )
    points.append(
        {
            "detector": "risk_route",
            "calibration_rule": "original_fixed",
            "selection_objective": "none; historical fixed operating point",
            "composite_threshold": ORIGINAL_COMPOSITE_THRESHOLD,
            "pair_threshold": ORIGINAL_PAIR_THRESHOLD,
            "calibration_metrics": original_metrics,
        }
    )
    objectives = [
        (f"fpr_budget_{int(alpha * 100)}pct", "recall_at_fpr", alpha)
        for alpha in TARGET_FPRS
    ] + [("maximum_mcc", "max_mcc", None)]
    for rule_name, objective, target_fpr in objectives:
        selected_route = select_route_thresholds(
            arrays["composite"],
            arrays["pair"],
            labels,
            objective=objective,
            target_fpr=target_fpr,
        )
        points.append(
            {
                "detector": "risk_route",
                "calibration_rule": rule_name,
                "selection_objective": (
                    "maximum recall subject to calibration FPR budget; ties use lower FPR, higher MCC, higher precision, lower coverage"
                    if target_fpr is not None
                    else "maximum calibration MCC; ties use higher recall then lower FPR"
                ),
                "target_fpr": target_fpr,
                "composite_threshold": selected_route["composite_threshold"],
                "pair_threshold": selected_route["pair_threshold"],
                "candidate_audit": {
                    key: selected_route[key]
                    for key in (
                        "composite_candidate_count",
                        "pair_candidate_count",
                        "joint_candidate_count",
                    )
                },
                "calibration_metrics": selected_route["metrics"],
            }
        )
        for detector in ("msp", "entropy", "margin"):
            selected = select_single_threshold(
                arrays[detector], labels, objective=objective, target_fpr=target_fpr
            )
            raw_rule = _raw_threshold(detector, selected["risk_threshold"])
            points.append(
                {
                    "detector": detector,
                    "calibration_rule": rule_name,
                    "selection_objective": (
                        "maximum recall subject to calibration FPR budget; ties use lower FPR, higher MCC, higher precision, lower coverage"
                        if target_fpr is not None
                        else "maximum calibration MCC; ties use higher recall then lower FPR"
                    ),
                    "target_fpr": target_fpr,
                    "risk_threshold": selected["risk_threshold"],
                    "raw_rule": raw_rule,
                    "candidate_audit": {"candidate_count": selected["candidate_count"]},
                    "calibration_metrics": selected["metrics"],
                }
            )
    return points


def _predict(point: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    detector = str(point["detector"])
    if detector == "risk_route":
        return (arrays["composite"] >= float(point["composite_threshold"])) | (
            arrays["pair"] >= float(point["pair_threshold"])
        )
    return arrays[detector] >= float(point["risk_threshold"])


def _metrics_from_counts(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized TP/FP/FN/TN rows to metric arrays."""
    values = np.asarray(counts, dtype=np.float64)
    (tp, fp, fn, tn) = (values[..., index] for index in range(4))
    total = tp + fp + fn + tn
    positive = tp + fn
    negative = fp + tn
    active = tp + fp
    recall = np.divide(tp, positive, out=np.zeros_like(tp), where=positive > 0)
    fpr = np.divide(fp, negative, out=np.zeros_like(fp), where=negative > 0)
    precision = np.divide(tp, active, out=np.zeros_like(tp), where=active > 0)
    coverage = np.divide(active, total, out=np.zeros_like(tp), where=total > 0)
    denominator = np.sqrt(active * positive * negative * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0
    )
    lr_plus = np.divide(recall, fpr, out=np.full_like(recall, np.nan), where=fpr > 0)
    return {
        "coverage": coverage,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "lr_plus": lr_plus,
        "mcc": mcc,
    }


def repository_bootstrap_intervals(
    labels: np.ndarray,
    predictions: list[np.ndarray],
    repositories: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Shared repository bootstrap for all operating points in one setting."""
    y = np.asarray(labels, dtype=bool)
    repos = np.asarray(repositories, dtype=object)
    if y.shape != repos.shape:
        raise ValueError("bootstrap labels/repositories shape mismatch")
    (unique_repos, inverse) = np.unique(repos, return_inverse=True)
    point_count = len(predictions)
    per_repo = np.zeros((len(unique_repos), point_count, 4), dtype=np.int64)
    for point_index, predicted in enumerate(predictions):
        p = np.asarray(predicted, dtype=bool)
        if p.shape != y.shape:
            raise ValueError("bootstrap prediction shape mismatch")
        categories = np.stack((y & p, ~y & p, y & ~p, ~y & ~p), axis=1)
        for category in range(4):
            np.add.at(
                per_repo[:, point_index, category],
                inverse,
                categories[:, category].astype(np.int64),
            )
    if replicates <= 0:
        return [{} for _ in predictions]
    rng = np.random.default_rng(int(seed))
    collected = {
        name: np.empty((replicates, point_count), dtype=np.float64)
        for name in ("coverage", "precision", "recall", "fpr", "lr_plus", "mcc")
    }
    batch_size = 100
    probabilities = np.full(len(unique_repos), 1.0 / len(unique_repos))
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        weights = rng.multinomial(len(unique_repos), probabilities, size=stop - start)
        counts = np.einsum("br,rpc->bpc", weights, per_repo, optimize=True)
        metrics = _metrics_from_counts(counts)
        for name, values in metrics.items():
            collected[name][start:stop] = values
    output: list[dict[str, Any]] = []
    for point_index in range(point_count):
        intervals: dict[str, Any] = {
            "replicates": int(replicates),
            "unit": "repository",
            "repositories": int(len(unique_repos)),
            "seed": int(seed),
        }
        for name, values in collected.items():
            column = values[:, point_index]
            finite = column[np.isfinite(column)]
            intervals[name] = {
                "low": float(np.quantile(finite, 0.025)) if finite.size else None,
                "high": float(np.quantile(finite, 0.975)) if finite.size else None,
                "finite_replicates": int(finite.size),
            }
        output.append(intervals)
    return output

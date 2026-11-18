from __future__ import annotations
import hashlib
import math
from pathlib import Path
from typing import Any
import numpy as np
from tqdm import tqdm
from ramp_hsr.io import discover_sample_files, read_jsonl_streaming, write_json

ANALYSIS_VERSION = "error-neuron-analysis-v4"
LABEL_MODES = ("base_error", "base_first_token_error", "base_correct")
NEGATIVE_MODES = ("base_correct", "base_error")
CHAIN_SCORE_MODES = ("reference_correct", "runtime_zscore")
DEFAULT_MAX_SAMPLES = 20000
DEFAULT_MAX_PER_CLASS = 10000
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_NEURONS = 20
DEFAULT_CHAIN_NODE_Z_THRESHOLD = 1.0
DEFAULT_CHAIN_MIN_ACTIVE_FRACTION = 0.5
DEFAULT_AUTO_CHAIN_MIN_LENGTH = 2
DEFAULT_AUTO_CHAIN_TOP_NEURONS = 128
DEFAULT_CHAIN_BREAKDOWN_MIN_SAMPLES = 16
DEFAULT_CHAIN_BREAKDOWN_MAX_GROUPS = 8
DEFAULT_PAIRWISE_CHAIN_TOP_K = 8
DEFAULT_PAIRWISE_CHAIN_MAX_PAIRS = 16
DEFAULT_ROUTE_ENSEMBLE_SIZES = (2, 3, 5, 10)
DEFAULT_ROUTE_SCORE_ACTIVE_FRACTION = 0.2
DEFAULT_RANKED_CHAIN_GLOBAL_SIZES = (4, 8, 16)
DEFAULT_RANKED_CHAIN_PER_LAYER_COUNTS = (1, 2, 3)


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if scores.size == 0 or labels.size == 0 or scores.size != labels.size:
        return None
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = np.arange(1, scores.size + 1, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[sorted_labels == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    return float(auc)


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def _stable_shuffled_labels(labels: np.ndarray, key: str) -> np.ndarray:
    if labels.size == 0:
        return labels
    seed = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return labels[rng.permutation(labels.size)]


def _fixed_count_active_mask(scores: np.ndarray, active_count: int) -> np.ndarray:
    active_count = max(0, min(int(active_count), int(scores.size)))
    mask = np.zeros(scores.size, dtype=bool)
    if active_count <= 0:
        return mask
    order = np.argsort(scores, kind="mergesort")
    mask[order[-active_count:]] = True
    return mask


def _risk_ratio_for_mask(labels: np.ndarray, active_mask: np.ndarray) -> dict[str, Any]:
    active_count = int(active_mask.sum())
    inactive_mask = ~active_mask
    inactive_count = int(inactive_mask.sum())
    positive_active = int(labels[active_mask].sum()) if active_count else 0
    positive_inactive = int(labels[inactive_mask].sum()) if inactive_count else 0
    positive_rate_active = positive_active / active_count if active_count > 0 else None
    positive_rate_inactive = (
        positive_inactive / inactive_count if inactive_count > 0 else None
    )
    return {
        "active_count": active_count,
        "inactive_count": inactive_count,
        "positive_rate_when_active": positive_rate_active,
        "positive_rate_when_inactive": positive_rate_inactive,
        "risk_ratio": (
            positive_rate_active / positive_rate_inactive
            if isinstance(positive_rate_active, (int, float))
            and isinstance(positive_rate_inactive, (int, float))
            and (positive_rate_inactive > 0.0)
            else None
        ),
        "odds_ratio": _odds_ratio(
            positive_active, active_count, positive_inactive, inactive_count
        ),
    }


def _summarize_scored_route(
    *,
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    sample_ids: list[str],
    active_mask: np.ndarray | None = None,
    active_fraction: float = DEFAULT_ROUTE_SCORE_ACTIVE_FRACTION,
) -> dict[str, Any]:
    if scores.size == 0 or labels.size == 0 or scores.size != labels.size:
        return {"name": name, "score_auc": None, "route_auc": None}
    if active_mask is None:
        active_count = int(round(scores.size * max(0.0, min(1.0, active_fraction))))
        active_mask = _fixed_count_active_mask(scores, active_count)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)
    route_risk = _risk_ratio_for_mask(labels, active_mask)
    random_scores = np.asarray(
        [
            _stable_unit_interval(f"{name}|{sample_id or index}|random")
            for (index, sample_id) in enumerate(sample_ids)
        ],
        dtype=np.float64,
    )
    shuffled_labels = _stable_shuffled_labels(labels, f"{name}|risk_shuffle")
    random_risk = _risk_ratio_for_mask(
        labels, _fixed_count_active_mask(random_scores, int(active_mask.sum()))
    )
    shuffled_risk = _risk_ratio_for_mask(shuffled_labels, active_mask)
    score_auc = _roc_auc(scores, labels)
    return {
        "name": name,
        "score_auc": score_auc,
        "route_auc": score_auc,
        "random_route_auc": _roc_auc(random_scores, labels),
        "shuffled_label_route_auc": _roc_auc(
            scores, _stable_shuffled_labels(labels, f"{name}|auc_shuffle")
        ),
        "active_count": route_risk.get("active_count"),
        "inactive_count": route_risk.get("inactive_count"),
        "active_rate": (
            float(route_risk.get("active_count", 0)) / float(labels.size)
            if labels.size > 0
            else None
        ),
        "positive_rate_when_active": route_risk.get("positive_rate_when_active"),
        "positive_rate_when_inactive": route_risk.get("positive_rate_when_inactive"),
        "risk_ratio": route_risk.get("risk_ratio"),
        "odds_ratio": route_risk.get("odds_ratio"),
        "random_route_risk_ratio": random_risk.get("risk_ratio"),
        "random_route_positive_rate_when_active": random_risk.get(
            "positive_rate_when_active"
        ),
        "random_route_positive_rate_when_inactive": random_risk.get(
            "positive_rate_when_inactive"
        ),
        "random_route_odds_ratio": random_risk.get("odds_ratio"),
        "shuffled_label_risk_ratio": shuffled_risk.get("risk_ratio"),
        "shuffled_label_positive_rate_when_active": shuffled_risk.get(
            "positive_rate_when_active"
        ),
        "shuffled_label_positive_rate_when_inactive": shuffled_risk.get(
            "positive_rate_when_inactive"
        ),
        "shuffled_label_odds_ratio": shuffled_risk.get("odds_ratio"),
        "score_mean_when_positive": (
            float(scores[labels == 1].mean()) if np.any(labels == 1) else None
        ),
        "score_mean_when_negative": (
            float(scores[labels == 0].mean()) if np.any(labels == 0) else None
        ),
    }


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "n/a"


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _odds_ratio(
    active_count: int, total_count: int, ref_active_count: int, ref_total_count: int
) -> float | None:
    if total_count <= 0 or ref_total_count <= 0:
        return None
    inactive_count = max(0, total_count - active_count)
    ref_inactive_count = max(0, ref_total_count - ref_active_count)
    odds = (active_count + 0.5) / (inactive_count + 0.5)
    ref_odds = (ref_active_count + 0.5) / (ref_inactive_count + 0.5)
    if ref_odds <= 0.0:
        return None
    return float(odds / ref_odds)


def _normalize_completion(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _sample_id(record: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(record.get("repo_id", "")),
            str(record.get("file_path", "")),
            str(record.get("line", "")),
            str(record.get("col", "")),
            str(record.get("target", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _record_language(record: dict[str, Any]) -> str:
    existing = record.get("language")
    if isinstance(existing, str) and existing:
        return existing
    suffix = Path(str(record.get("file_path") or "")).suffix.lower()
    if suffix == ".py":
        return "python"
    return suffix.lstrip(".") or "unknown"


def _record_metric(record: dict[str, Any], key: str) -> float | None:
    uncertainty = record.get("uncertainty")
    if isinstance(uncertainty, dict) and key in uncertainty:
        return _safe_float(uncertainty.get(key))
    return _safe_float(record.get(key))


def _sample_hash_fraction(record: dict[str, Any]) -> float:
    digest = hashlib.sha1(_sample_id(record).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def _base_error_label(record: dict[str, Any]) -> int | None:
    token_id_flag = record.get("base_top1_equals_target_token_id")
    if record.get("target_is_single_token") is True and isinstance(token_id_flag, bool):
        return 0 if token_id_flag else 1
    token_flag = record.get("base_top1_equals_target_token")
    if isinstance(token_flag, bool):
        return 0 if token_flag else 1
    if "base_prediction" not in record or "target" not in record:
        if isinstance(record.get("base_correct"), bool):
            return 0 if record["base_correct"] else 1
        return None
    return (
        0
        if _normalize_completion(record.get("base_prediction"))
        == _normalize_completion(record.get("target"))
        else 1
    )


def _base_first_token_error_label(record: dict[str, Any]) -> int | None:
    base_top1_token_id = record.get("base_top1_token_id")
    target_first_token_id = record.get("target_first_token_id")
    if base_top1_token_id is None or target_first_token_id is None:
        return _base_error_label(record)
    try:
        base_id = int(base_top1_token_id)
        target_id = int(target_first_token_id)
    except (TypeError, ValueError):
        return _base_error_label(record)
    return 0 if base_id == target_id else 1


def _extract_label(
    record: dict[str, Any],
    *,
    label_mode: str = "base_error",
    negative_mode: str = "base_correct",
) -> int | None:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label_mode: {label_mode}")
    if negative_mode not in NEGATIVE_MODES:
        raise ValueError(f"Unsupported negative_mode: {negative_mode}")
    if label_mode == "base_error":
        return _base_error_label(record)
    if label_mode == "base_first_token_error":
        return _base_first_token_error_label(record)
    if label_mode == "base_correct":
        base_error = _base_error_label(record)
        return None if base_error is None else 1 - int(base_error)
    return None


def _extract_baseline_risks(record: dict[str, Any]) -> dict[str, float]:
    uncertainty = record.get("uncertainty") or {}
    risks: dict[str, float] = {}
    margin = _safe_float(uncertainty.get("margin", record.get("margin")))
    if margin is not None:
        risks["margin"] = -margin
    for key in ("entropy", "layer_disagreement", "representation_variance"):
        value = _safe_float(uncertainty.get(key, record.get(key)))
        if value is not None:
            risks[key] = value
    return risks


def _normalize_chain_direction(value: Any) -> str | None:
    if value is None:
        return None
    direction = str(value).strip().lower()
    if direction in {"+", "pos", "positive"}:
        return "positive"
    if direction in {"-", "neg", "negative"}:
        return "negative"
    if direction in {"abs", "absolute"}:
        return "absolute"
    raise ValueError(f"Invalid chain node direction '{value}', expected +, -, or abs")


def _parse_chain_node(text: str) -> dict[str, Any]:
    parts = [part.strip() for part in str(text).split(":") if part.strip()]
    if len(parts) not in {2, 3}:
        raise ValueError(
            f"Invalid chain node '{text}', expected layer:dim or layer:dim:direction"
        )
    node = {"layer_index": int(parts[0]), "dimension": int(parts[1])}
    if len(parts) == 3:
        node["direction"] = _normalize_chain_direction(parts[2])
    return node


def _parse_chain_spec(spec: str, ordinal: int) -> dict[str, Any]:
    raw = str(spec).strip()
    if not raw:
        raise ValueError("Chain spec cannot be empty")
    if "=" in raw:
        (name, nodes_text) = raw.split("=", 1)
        chain_name = name.strip() or f"chain_{ordinal}"
    else:
        chain_name = f"chain_{ordinal}"
        nodes_text = raw
    nodes = [
        _parse_chain_node(node_text)
        for node_text in nodes_text.split(",")
        if node_text.strip()
    ]
    if not nodes:
        raise ValueError(f"Chain spec '{spec}' did not contain any nodes")
    return {"name": chain_name, "nodes": nodes, "source": "cli"}


def _load_chain_file(path: str | Path) -> list[dict[str, Any]]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    chains = payload.get("chains") if isinstance(payload, dict) else payload
    if not isinstance(chains, list):
        raise ValueError(
            f"Invalid chain file at {path}, expected list or {{'chains': [...]}}"
        )
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(chains, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid chain entry at index {index}, expected object")
        name = str(item.get("name") or f"chain_{index}")
        raw_nodes = item.get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ValueError(f"Chain '{name}' must define a non-empty nodes list")
        nodes = []
        for node in raw_nodes:
            if isinstance(node, dict):
                parsed_node: dict[str, Any] = {
                    "layer_index": int(node["layer_index"]),
                    "dimension": int(node["dimension"]),
                }
                if node.get("direction") is not None:
                    parsed_node["direction"] = _normalize_chain_direction(
                        node.get("direction")
                    )
                nodes.append(parsed_node)
            else:
                nodes.append(_parse_chain_node(str(node)))
        parsed.append(
            {"name": name, "nodes": nodes, "source": f"file:{Path(path).name}"}
        )
    return parsed


def _iter_records(input_path: str | Path):
    for sample_file in discover_sample_files(input_path):
        for record in read_jsonl_streaming(sample_file):
            yield (sample_file, record)


def select_records_for_error_neuron_analysis(
    input_path: str | Path,
    *,
    label_mode: str = "base_error",
    negative_mode: str = "base_correct",
    max_samples: int = DEFAULT_MAX_SAMPLES,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    sample_rate: float = 1.0,
    sample_hash_min: float = 0.0,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    counts = {"negative": 0, "positive": 0}
    files: list[str] = []
    skipped_missing_label = 0
    skipped_missing_prefix = 0
    seen = 0
    for sample_file, record in _iter_records(input_path):
        if not files or files[-1] != str(sample_file):
            files.append(str(sample_file))
        seen += 1
        label = _extract_label(
            record, label_mode=label_mode, negative_mode=negative_mode
        )
        if label is None:
            skipped_missing_label += 1
            continue
        prefix = record.get("prefix")
        if not isinstance(prefix, str):
            skipped_missing_prefix += 1
            continue
        hash_fraction = _sample_hash_fraction(record)
        if hash_fraction < max(0.0, float(sample_hash_min)):
            continue
        if sample_rate < 1.0 and hash_fraction > sample_rate:
            continue
        bucket = "positive" if label == 1 else "negative"
        if max_per_class > 0 and counts[bucket] >= max_per_class:
            if (
                max_per_class > 0
                and counts["negative"] >= max_per_class
                and (counts["positive"] >= max_per_class)
            ):
                break
            continue
        if max_samples > 0 and len(selected) >= max_samples:
            break
        selected.append(
            {
                "sample_id": _sample_id(record),
                "repo_id": record.get("repo_id"),
                "file_path": record.get("file_path"),
                "language": _record_language(record),
                "prefix": prefix,
                "prefix_length": len(prefix),
                "target": record.get("target"),
                "base_prediction": record.get("base_prediction"),
                "label": label,
                "label_mode": label_mode,
                "negative_mode": negative_mode,
                "target_token_ids": record.get("target_token_ids"),
                "target_first_token_id": record.get("target_first_token_id"),
                "target_is_single_token": record.get("target_is_single_token"),
                "target_match_normalization": record.get("target_match_normalization"),
                "base_top1_token_id": record.get("base_top1_token_id"),
                "base_top1_equals_target_token": record.get(
                    "base_top1_equals_target_token"
                ),
                "base_top1_equals_target_whitespace_normalized": record.get(
                    "base_top1_equals_target_whitespace_normalized"
                ),
                "base_top1_equals_target_token_id": record.get(
                    "base_top1_equals_target_token_id"
                ),
                "base_top1_equals_target_raw_text": record.get(
                    "base_top1_equals_target_raw_text"
                ),
                "base_model_name": record.get("base_model_name"),
                "baseline_risks": _extract_baseline_risks(record),
                "entropy": _record_metric(record, "entropy"),
                "margin": _record_metric(record, "margin"),
            }
        )
        if label_mode == "base_correct":
            selected[-1]["error_semantic_bucket"] = (
                "base_correct" if label == 1 else "base_error"
            )
        else:
            selected[-1]["error_semantic_bucket"] = (
                "correct" if label == 0 else "base_error"
            )
        counts[bucket] += 1
    correct_count = (
        counts["positive"] if label_mode == "base_correct" else counts["negative"]
    )
    wrong_count = (
        counts["negative"] if label_mode == "base_correct" else counts["positive"]
    )
    return {
        "selected_records": selected,
        "files": files,
        "seen_records": seen,
        "selected_count": len(selected),
        "label_mode": label_mode,
        "negative_mode": negative_mode,
        "sample_hash_min": float(sample_hash_min),
        "sample_rate": float(sample_rate),
        "negative_count": counts["negative"],
        "positive_count": counts["positive"],
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "skipped_missing_label": skipped_missing_label,
        "skipped_missing_prefix": skipped_missing_prefix,
    }


def _empty_group_stats(num_layers: int, hidden_dim: int) -> dict[str, Any]:
    shape = (num_layers, hidden_dim)
    return {
        "count": 0,
        "signed_sum": np.zeros(shape, dtype=np.float64),
        "signed_sumsq": np.zeros(shape, dtype=np.float64),
        "abs_sum": np.zeros(shape, dtype=np.float64),
        "abs_sumsq": np.zeros(shape, dtype=np.float64),
    }


def _update_group_stats(group: dict[str, Any], vectors: np.ndarray) -> None:
    if vectors.size == 0:
        return
    group["count"] += int(vectors.shape[0])
    group["signed_sum"] += vectors.sum(axis=0)
    group["signed_sumsq"] += np.square(vectors).sum(axis=0)
    abs_vectors = np.abs(vectors)
    group["abs_sum"] += abs_vectors.sum(axis=0)
    group["abs_sumsq"] += np.square(abs_vectors).sum(axis=0)


def _means_and_vars(group: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    count = max(1, int(group["count"]))
    sums = group[f"{key}_sum"]
    sumsq = group[f"{key}_sumsq"]
    mean = sums / count
    var = np.maximum(sumsq / count - np.square(mean), 0.0)
    return (mean, var)


def _compute_neuron_rows(
    layer_indices: list[int], correct_group: dict[str, Any], wrong_group: dict[str, Any]
) -> list[dict[str, Any]]:
    (correct_signed_mean, correct_signed_var) = _means_and_vars(correct_group, "signed")
    (wrong_signed_mean, wrong_signed_var) = _means_and_vars(wrong_group, "signed")
    (correct_abs_mean, correct_abs_var) = _means_and_vars(correct_group, "abs")
    (wrong_abs_mean, wrong_abs_var) = _means_and_vars(wrong_group, "abs")
    correct_count = max(1, int(correct_group["count"]))
    wrong_count = max(1, int(wrong_group["count"]))
    pooled_abs_std = np.sqrt(
        np.maximum(
            (
                correct_abs_var * max(correct_count - 1, 0)
                + wrong_abs_var * max(wrong_count - 1, 0)
            )
            / max(correct_count + wrong_count - 2, 1),
            1e-12,
        )
    )
    rows: list[dict[str, Any]] = []
    for layer_pos, layer_idx in enumerate(layer_indices):
        hidden_dim = correct_abs_mean.shape[1]
        for dim_idx in range(hidden_dim):
            abs_mean_correct = float(correct_abs_mean[layer_pos, dim_idx])
            abs_mean_wrong = float(wrong_abs_mean[layer_pos, dim_idx])
            abs_gap = abs_mean_wrong - abs_mean_correct
            signed_mean_correct = float(correct_signed_mean[layer_pos, dim_idx])
            signed_mean_wrong = float(wrong_signed_mean[layer_pos, dim_idx])
            signed_gap = signed_mean_wrong - signed_mean_correct
            abs_std = float(pooled_abs_std[layer_pos, dim_idx])
            abs_cohen_d = abs_gap / abs_std if abs_std > 1e-12 else 0.0
            ratio = abs_mean_wrong / max(abs_mean_correct, 1e-08)
            rows.append(
                {
                    "layer_index": int(layer_idx),
                    "layer_position": int(layer_pos),
                    "dimension": int(dim_idx),
                    "abs_mean_correct": abs_mean_correct,
                    "abs_mean_wrong": abs_mean_wrong,
                    "abs_mean_gap": abs_gap,
                    "abs_cohen_d": abs_cohen_d,
                    "abs_ratio_wrong_vs_correct": ratio,
                    "signed_mean_correct": signed_mean_correct,
                    "signed_mean_wrong": signed_mean_wrong,
                    "signed_mean_gap": signed_gap,
                    "pooled_abs_std": abs_std,
                }
            )
    rows.sort(
        key=lambda row: (
            row["abs_cohen_d"],
            row["abs_mean_gap"],
            row["abs_ratio_wrong_vs_correct"],
        ),
        reverse=True,
    )
    return rows


def _build_neuron_reference(
    layer_indices: list[int], correct_group: dict[str, Any], wrong_group: dict[str, Any]
) -> dict[str, Any]:
    (correct_signed_mean, correct_signed_var) = _means_and_vars(correct_group, "signed")
    (wrong_signed_mean, wrong_signed_var) = _means_and_vars(wrong_group, "signed")
    (correct_abs_mean, correct_abs_var) = _means_and_vars(correct_group, "abs")
    (wrong_abs_mean, wrong_abs_var) = _means_and_vars(wrong_group, "abs")
    correct_count = max(1, int(correct_group["count"]))
    wrong_count = max(1, int(wrong_group["count"]))
    pooled_signed_std = np.sqrt(
        np.maximum(
            (
                correct_signed_var * max(correct_count - 1, 0)
                + wrong_signed_var * max(wrong_count - 1, 0)
            )
            / max(correct_count + wrong_count - 2, 1),
            1e-12,
        )
    )
    pooled_abs_std = np.sqrt(
        np.maximum(
            (
                correct_abs_var * max(correct_count - 1, 0)
                + wrong_abs_var * max(wrong_count - 1, 0)
            )
            / max(correct_count + wrong_count - 2, 1),
            1e-12,
        )
    )
    layer_to_position = {int(layer): idx for (idx, layer) in enumerate(layer_indices)}
    return {
        "layer_indices": [int(layer) for layer in layer_indices],
        "layer_to_position": layer_to_position,
        "correct_signed_mean": correct_signed_mean,
        "wrong_signed_mean": wrong_signed_mean,
        "pooled_signed_std": pooled_signed_std,
        "correct_abs_mean": correct_abs_mean,
        "wrong_abs_mean": wrong_abs_mean,
        "pooled_abs_std": pooled_abs_std,
    }


def _chunked(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_hidden_state_cache(
    cache_path: str | Path | None,
) -> dict[str, dict[str, Any]]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Hidden-state cache does not exist: {path}")
    files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    cache: dict[str, dict[str, Any]] = {}
    for file_path in files:
        with np.load(file_path, allow_pickle=True) as payload:
            if not {"sample_ids", "layers", "hidden"}.issubset(payload.files):
                continue
            sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
            layers = [int(value) for value in payload["layers"].tolist()]
            hidden = np.asarray(payload["hidden"], dtype=np.float32)
            top1_tokens = (
                [str(value) for value in payload["top1_tokens"].tolist()]
                if "top1_tokens" in payload.files
                else [""] * len(sample_ids)
            )
            for idx, sample_id in enumerate(sample_ids):
                cache[sample_id] = {
                    "layers": layers,
                    "hidden": hidden[idx],
                    "top1_token": top1_tokens[idx] if idx < len(top1_tokens) else "",
                    "source": str(file_path),
                }
    return cache


def _attach_hidden_state_cache(
    records: list[dict[str, Any]],
    *,
    cache_path: str | Path | None,
    layer_indices: list[int] | None,
) -> dict[str, Any]:
    if cache_path is None:
        return {
            "path": None,
            "loaded_entries": 0,
            "attached_count": 0,
            "missing_count": len(records),
            "missing_sample_ids": [],
        }
    cache = _load_hidden_state_cache(cache_path)
    requested_layers = [int(layer) for layer in layer_indices or []]
    attached = 0
    missing_ids: list[str] = []
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        row = cache.get(sample_id)
        if row is None:
            missing_ids.append(sample_id)
            continue
        cached_layers = [int(layer) for layer in row["layers"]]
        if requested_layers and (
            not set(requested_layers).issubset(set(cached_layers))
        ):
            missing_ids.append(sample_id)
            continue
        record["_cached_layer_indices"] = cached_layers
        record["_cached_layer_matrix"] = np.asarray(row["hidden"], dtype=np.float32)
        record["_cached_top1_token"] = str(row.get("top1_token") or "")
        attached += 1
    return {
        "path": str(Path(cache_path).resolve()),
        "loaded_entries": len(cache),
        "attached_count": attached,
        "missing_count": len(records) - attached,
        "missing_sample_ids": missing_ids[:10],
    }


def _records_have_hidden_cache(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all(
        (
            "_cached_layer_matrix" in record and "_cached_layer_indices" in record
            for record in records
        )
    )


def _cached_result_for_record(
    record: dict[str, Any], layer_indices: list[int] | None
) -> dict[str, Any]:
    matrix = np.asarray(record["_cached_layer_matrix"], dtype=np.float32)
    cached_layers = [int(layer) for layer in record["_cached_layer_indices"]]
    requested_layers = [int(layer) for layer in layer_indices or cached_layers]
    layer_to_pos = {int(layer): idx for (idx, layer) in enumerate(cached_layers)}
    layer_states = {
        int(layer): matrix[layer_to_pos[int(layer)]]
        for layer in requested_layers
        if int(layer) in layer_to_pos
    }
    top1 = str(record.get("_cached_top1_token") or "")
    return {
        "layer_hidden_states": layer_states,
        "top_k_tokens": [(top1, 1.0)] if top1 else [],
    }


def _analyze_records_batch(
    batch_records: list[dict[str, Any]],
    llm: Any | None,
    *,
    layer_indices: list[int] | None,
    max_length: int,
    k: int = 2,
) -> list[dict[str, Any]]:
    if _records_have_hidden_cache(batch_records):
        return [
            _cached_result_for_record(record, layer_indices) for record in batch_records
        ]
    if llm is None:
        raise ValueError(
            "Missing hidden-state cache and no model is available for fallback"
        )
    return llm.analyze_prefix_batch(
        [record["prefix"] for record in batch_records],
        k=k,
        sample_layers=layer_indices or None,
        max_length=max_length,
    )


def _prepare_batch_vectors(
    batch_records: list[dict[str, Any]],
    batch_results: list[dict[str, Any]],
    layer_indices: list[int] | None,
) -> tuple[np.ndarray, list[int], list[str], int]:
    resolved_layers = list(layer_indices or [])
    if not resolved_layers:
        for result in batch_results:
            layer_states = result.get("layer_hidden_states") or {}
            if layer_states:
                resolved_layers = sorted((int(key) for key in layer_states))
                break
    if not resolved_layers:
        raise ValueError("No sampled layer hidden states were returned by the model")
    vectors = []
    predicted_top1: list[str] = []
    mismatch_count = 0
    for record, result in zip(batch_records, batch_results):
        layer_states = result.get("layer_hidden_states") or {}
        matrix = np.asarray(
            [
                np.asarray(layer_states[layer], dtype=np.float64)
                for layer in resolved_layers
            ],
            dtype=np.float64,
        )
        vectors.append(matrix)
        top_k_tokens = result.get("top_k_tokens") or []
        top1 = str(top_k_tokens[0][0]) if top_k_tokens else ""
        predicted_top1.append(top1)
        if top1 != str(record.get("base_prediction") or ""):
            mismatch_count += 1
    return (np.stack(vectors, axis=0), resolved_layers, predicted_top1, mismatch_count)


def _baseline_auc_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int8)
    keys = sorted(
        {key for record in records for key in record.get("baseline_risks") or {}}
    )
    summary: dict[str, float] = {}
    for key in keys:
        values = [
            _safe_float((record.get("baseline_risks") or {}).get(key))
            for record in records
        ]
        mask = np.asarray([value is not None for value in values], dtype=bool)
        if not np.any(mask):
            continue
        auc = _roc_auc(
            np.asarray(
                [float(value) for value in values if value is not None],
                dtype=np.float64,
            ),
            labels[mask],
        )
        if auc is not None:
            summary[key] = float(auc)
    return summary


def _auto_discover_chains(
    neuron_rows: list[dict[str, Any]], *, min_length: int, top_neurons: int
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in neuron_rows[: max(1, int(top_neurons))]:
        grouped.setdefault(int(row["dimension"]), []).append(row)
    chains: list[dict[str, Any]] = []
    for dimension, rows in grouped.items():
        unique_rows = sorted(
            rows, key=lambda row: (int(row["layer_index"]), -float(row["abs_cohen_d"]))
        )
        if len(unique_rows) < max(2, int(min_length)):
            continue
        directions = [
            "positive" if float(row.get("signed_mean_gap", 0.0)) >= 0.0 else "negative"
            for row in unique_rows
        ]
        direction_signature = "".join(
            ("+" if d == "positive" else "-" for d in directions)
        )
        chains.append(
            {
                "name": "dim_"
                + str(dimension)
                + "_layers_"
                + "_".join((str(int(row["layer_index"])) for row in unique_rows)),
                "nodes": [
                    {
                        "layer_index": int(row["layer_index"]),
                        "dimension": int(row["dimension"]),
                        "direction": (
                            "positive"
                            if float(row.get("signed_mean_gap", 0.0)) >= 0.0
                            else "negative"
                        ),
                    }
                    for row in unique_rows
                ],
                "source": "auto_dimension_chain",
                "direction_signature": direction_signature,
            }
        )
    chains.sort(key=lambda chain: (len(chain["nodes"]), chain["name"]), reverse=True)
    return chains


def _node_from_neuron_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_index": int(row["layer_index"]),
        "dimension": int(row["dimension"]),
        "direction": (
            "positive" if float(row.get("signed_mean_gap", 0.0)) >= 0.0 else "negative"
        ),
    }


def _direction_signature(nodes: list[dict[str, Any]]) -> str:
    signature = []
    for node in nodes:
        direction = str(node.get("direction") or "absolute").lower()
        if direction == "positive":
            signature.append("+")
        elif direction == "negative":
            signature.append("-")
        else:
            signature.append("a")
    return "".join(signature)


def _auto_discover_ranked_node_chains(
    neuron_rows: list[dict[str, Any]],
    *,
    layer_indices: list[int],
    global_sizes: list[int],
    per_layer_counts: list[int],
    top_neurons: int,
) -> list[dict[str, Any]]:
    """Build opt-in single-chain candidates from the strongest discovery neurons.

    These chains are not ensembles: each candidate is one fixed node sequence that
    is scored with the same chain scorer as hand-authored chains. The selection is
    univariate and deterministic on the discovery split, then the exact same chain
    can be scored on held-out data.
    """
    ranked_rows = neuron_rows[: max(1, int(top_neurons))]
    chains: list[dict[str, Any]] = []
    seen_global_sizes: set[int] = set()
    for size in global_sizes:
        resolved_size = max(1, int(size))
        if resolved_size in seen_global_sizes or len(ranked_rows) < resolved_size:
            continue
        seen_global_sizes.add(resolved_size)
        nodes = [_node_from_neuron_row(row) for row in ranked_rows[:resolved_size]]
        chains.append(
            {
                "name": f"ranked_top{resolved_size}_global_chain",
                "nodes": nodes,
                "source": "auto_ranked_global_chain",
                "direction_signature": _direction_signature(nodes),
            }
        )
    rows_by_layer: dict[int, list[dict[str, Any]]] = {
        int(layer_idx): [] for layer_idx in layer_indices
    }
    for row in neuron_rows:
        layer_idx = int(row["layer_index"])
        if layer_idx in rows_by_layer:
            rows_by_layer[layer_idx].append(row)
    seen_per_layer_counts: set[int] = set()
    for count in per_layer_counts:
        resolved_count = max(1, int(count))
        if resolved_count in seen_per_layer_counts:
            continue
        seen_per_layer_counts.add(resolved_count)
        nodes: list[dict[str, Any]] = []
        for layer_idx in layer_indices:
            layer_rows = rows_by_layer.get(int(layer_idx), [])
            nodes.extend(
                (_node_from_neuron_row(row) for row in layer_rows[:resolved_count])
            )
        if len(nodes) < max(1, len(layer_indices)):
            continue
        chains.append(
            {
                "name": f"ranked_top{resolved_count}_per_layer_chain",
                "nodes": nodes,
                "source": "auto_ranked_per_layer_chain",
                "direction_signature": _direction_signature(nodes),
            }
        )
    return chains


def _resolve_chain_definitions(
    *,
    chain_specs: list[str] | None,
    chain_file: str | Path | None,
    auto_chains: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    for index, spec in enumerate(chain_specs or [], start=1):
        chains.append(_parse_chain_spec(spec, index))
    if chain_file:
        chains.extend(_load_chain_file(chain_file))
    if not chains:
        chains.extend(auto_chains)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[tuple[int, int, str], ...]]] = set()
    for chain in chains:
        nodes = tuple(
            sorted(
                (
                    (
                        int(node["layer_index"]),
                        int(node["dimension"]),
                        str(node.get("direction") or "auto"),
                    )
                    for node in chain.get("nodes", [])
                ),
                key=lambda item: (item[0], item[1], item[2]),
            )
        )
        key = (str(chain.get("name") or ""), nodes)
        if not nodes or key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "name": str(chain.get("name") or f"chain_{len(deduped) + 1}"),
                "nodes": [
                    {
                        "layer_index": layer_idx,
                        "dimension": dim_idx,
                        "direction": direction if direction != "auto" else None,
                    }
                    for (layer_idx, dim_idx, direction) in nodes
                ],
                "source": str(chain.get("source") or "unknown"),
                "direction_signature": str(chain.get("direction_signature") or ""),
            }
        )
    return deduped


def _score_chain_for_sample(
    layer_states: dict[int, Any],
    chain: dict[str, Any],
    reference: dict[str, Any],
    *,
    node_z_threshold: float,
    min_active_fraction: float,
    score_mode: str = "reference_correct",
) -> dict[str, Any] | None:
    resolved_mode = str(score_mode or "reference_correct")
    if resolved_mode not in CHAIN_SCORE_MODES:
        raise ValueError(f"Unsupported chain score mode: {score_mode}")
    layer_to_position = reference["layer_to_position"]
    correct_signed_mean = reference["correct_signed_mean"]
    pooled_signed_std = reference["pooled_signed_std"]
    correct_abs_mean = reference["correct_abs_mean"]
    pooled_abs_std = reference["pooled_abs_std"]
    z_scores: list[float] = []
    active_nodes = 0
    for node in chain.get("nodes", []):
        layer_idx = int(node["layer_index"])
        dim_idx = int(node["dimension"])
        if layer_idx not in layer_states:
            return None
        vector = np.asarray(layer_states[layer_idx], dtype=np.float64)
        if dim_idx < 0 or dim_idx >= vector.shape[0]:
            return None
        direction = str(node.get("direction") or "absolute").lower()
        value = float(vector[dim_idx])
        if resolved_mode == "runtime_zscore":
            signed_mean = float(np.mean(vector))
            signed_std = float(np.std(vector))
            signed_denom = signed_std if signed_std > 1e-12 else 1.0
            if direction == "positive":
                z = (value - signed_mean) / signed_denom
            elif direction == "negative":
                z = (signed_mean - value) / signed_denom
            else:
                abs_vector = np.abs(vector)
                abs_mean = float(np.mean(abs_vector))
                abs_std = float(np.std(abs_vector))
                abs_denom = abs_std if abs_std > 1e-12 else 1.0
                z = (abs(value) - abs_mean) / abs_denom
        else:
            if layer_idx not in layer_to_position:
                return None
            pos = int(layer_to_position[layer_idx])
            if direction == "positive":
                baseline = float(correct_signed_mean[pos, dim_idx])
                std = float(pooled_signed_std[pos, dim_idx])
                denom = std if std > 1e-12 else 1.0
                z = (value - baseline) / denom
            elif direction == "negative":
                baseline = float(correct_signed_mean[pos, dim_idx])
                std = float(pooled_signed_std[pos, dim_idx])
                denom = std if std > 1e-12 else 1.0
                z = (baseline - value) / denom
            else:
                baseline = float(correct_abs_mean[pos, dim_idx])
                std = float(pooled_abs_std[pos, dim_idx])
                denom = std if std > 1e-12 else 1.0
                z = (abs(value) - baseline) / denom
        z_scores.append(z)
        if z >= node_z_threshold:
            active_nodes += 1
    if not z_scores:
        return None
    active_fraction = active_nodes / len(z_scores)
    required_fraction = max(0.0, min(1.0, float(min_active_fraction)))
    if resolved_mode == "runtime_zscore":
        score = float(np.mean([max(0.0, score) for score in z_scores]))
    else:
        score = float(np.mean(z_scores))
    return {
        "score": score,
        "active_nodes": int(active_nodes),
        "node_count": int(len(z_scores)),
        "active_fraction": float(active_fraction),
        "active": bool(active_fraction >= required_fraction),
        "score_mode": resolved_mode,
    }


def _summarize_chain_events(
    chain: dict[str, Any], events: list[dict[str, Any]], *, overall_error_rate: float
) -> dict[str, Any]:
    if not events:
        return {
            "name": chain["name"],
            "source": chain.get("source"),
            "nodes": chain.get("nodes", []),
            "active_count": 0,
            "score_auc": None,
        }
    labels = np.asarray([int(event["label"]) for event in events], dtype=np.int8)
    scores = np.asarray([float(event["score"]) for event in events], dtype=np.float64)
    active_mask = np.asarray([bool(event["active"]) for event in events], dtype=bool)
    inactive_mask = ~active_mask
    active_count = int(active_mask.sum())
    inactive_count = int(inactive_mask.sum())
    wrong_active = int(labels[active_mask].sum()) if active_count else 0
    wrong_inactive = int(labels[inactive_mask].sum()) if inactive_count else 0
    error_rate_active = wrong_active / active_count if active_count > 0 else None
    error_rate_inactive = (
        wrong_inactive / inactive_count if inactive_count > 0 else None
    )
    score_auc = _roc_auc(scores, labels)
    random_scores = np.asarray(
        [
            _stable_unit_interval(
                f"{chain.get('name', 'chain')}|{event.get('sample_id', index)}|random"
            )
            for (index, event) in enumerate(events)
        ],
        dtype=np.float64,
    )
    random_route_auc = _roc_auc(random_scores, labels)
    random_route_risk = _risk_ratio_for_mask(
        labels, _fixed_count_active_mask(random_scores, active_count)
    )
    shuffled_label_route_auc = _roc_auc(
        scores, _stable_shuffled_labels(labels, f"{chain.get('name', 'chain')}|shuffle")
    )
    shuffled_route_risk = _risk_ratio_for_mask(
        _stable_shuffled_labels(labels, f"{chain.get('name', 'chain')}|risk_shuffle"),
        active_mask,
    )
    recall_on_errors = (
        wrong_active / int(labels.sum()) if int(labels.sum()) > 0 else None
    )
    precision_when_active = error_rate_active
    risk_ratio_vs_overall = (
        error_rate_active / overall_error_rate
        if isinstance(error_rate_active, (int, float)) and overall_error_rate > 0.0
        else None
    )
    risk_ratio_active_vs_inactive = (
        error_rate_active / error_rate_inactive
        if isinstance(error_rate_active, (int, float))
        and isinstance(error_rate_inactive, (int, float))
        and (error_rate_inactive > 0.0)
        else None
    )
    odds_ratio_active_vs_inactive = _odds_ratio(
        wrong_active, active_count, wrong_inactive, inactive_count
    )
    return {
        "name": chain["name"],
        "source": chain.get("source"),
        "nodes": chain.get("nodes", []),
        "node_count": len(chain.get("nodes", [])),
        "active_count": active_count,
        "inactive_count": inactive_count,
        "active_rate": active_count / len(events),
        "error_rate_when_active": error_rate_active,
        "error_rate_when_inactive": error_rate_inactive,
        "positive_rate_when_active": error_rate_active,
        "positive_rate_when_inactive": error_rate_inactive,
        "error_lift_vs_inactive": (
            error_rate_active - error_rate_inactive
            if isinstance(error_rate_active, (int, float))
            and isinstance(error_rate_inactive, (int, float))
            else None
        ),
        "positive_lift_vs_inactive": (
            error_rate_active - error_rate_inactive
            if isinstance(error_rate_active, (int, float))
            and isinstance(error_rate_inactive, (int, float))
            else None
        ),
        "risk_ratio_vs_overall": risk_ratio_vs_overall,
        "risk_ratio_active_vs_inactive": risk_ratio_active_vs_inactive,
        "risk_ratio": risk_ratio_active_vs_inactive,
        "odds_ratio_active_vs_inactive": odds_ratio_active_vs_inactive,
        "odds_ratio": odds_ratio_active_vs_inactive,
        "recall_on_errors": recall_on_errors,
        "precision_when_active": precision_when_active,
        "score_auc": score_auc,
        "route_auc": score_auc,
        "random_route_auc": random_route_auc,
        "random_route_risk_ratio": random_route_risk.get("risk_ratio"),
        "random_route_positive_rate_when_active": random_route_risk.get(
            "positive_rate_when_active"
        ),
        "random_route_positive_rate_when_inactive": random_route_risk.get(
            "positive_rate_when_inactive"
        ),
        "random_route_odds_ratio": random_route_risk.get("odds_ratio"),
        "shuffled_label_route_auc": shuffled_label_route_auc,
        "shuffled_label_risk_ratio": shuffled_route_risk.get("risk_ratio"),
        "shuffled_label_positive_rate_when_active": shuffled_route_risk.get(
            "positive_rate_when_active"
        ),
        "shuffled_label_positive_rate_when_inactive": shuffled_route_risk.get(
            "positive_rate_when_inactive"
        ),
        "shuffled_label_odds_ratio": shuffled_route_risk.get("odds_ratio"),
        "score_mean_when_wrong": (
            float(scores[labels == 1].mean()) if np.any(labels == 1) else None
        ),
        "score_mean_when_correct": (
            float(scores[labels == 0].mean()) if np.any(labels == 0) else None
        ),
        "active_fraction_mean_when_wrong": (
            float(
                np.mean(
                    [
                        float(event["active_fraction"])
                        for event in events
                        if int(event["label"]) == 1
                    ]
                )
            )
            if np.any(labels == 1)
            else None
        ),
        "active_fraction_mean_when_correct": (
            float(
                np.mean(
                    [
                        float(event["active_fraction"])
                        for event in events
                        if int(event["label"]) == 0
                    ]
                )
            )
            if np.any(labels == 0)
            else None
        ),
    }


def _summarize_chain_events_by_semantics(
    events: list[dict[str, Any]], *, min_samples: int, max_groups: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        group_key = str(event.get("error_semantic_bucket") or "unknown")
        grouped.setdefault(group_key, []).append(event)
    correct_events = grouped.get("correct", [])
    correct_count = len(correct_events)
    correct_active_count = sum(
        (1 for event in correct_events if bool(event.get("active")))
    )
    correct_active_rate = _rate(correct_active_count, correct_count)
    rows: list[tuple[int, float, int, str, dict[str, Any]]] = []
    for bucket, bucket_events in grouped.items():
        if len(bucket_events) < max(1, int(min_samples)):
            continue
        count = len(bucket_events)
        active_count = sum((1 for event in bucket_events if bool(event.get("active"))))
        active_rate = _rate(active_count, count)
        lift = (
            active_rate - correct_active_rate
            if isinstance(active_rate, (int, float))
            and isinstance(correct_active_rate, (int, float))
            else None
        )
        if bucket == "correct":
            enrichment = 1.0
            odds_ratio = 1.0
        else:
            enrichment = (
                active_rate / correct_active_rate
                if isinstance(active_rate, (int, float))
                and isinstance(correct_active_rate, (int, float))
                and (correct_active_rate > 0.0)
                else None
            )
            odds_ratio = _odds_ratio(
                active_count, count, correct_active_count, correct_count
            )
        row = {
            "count": count,
            "active_count": active_count,
            "active_rate_in_bucket": active_rate,
            "active_rate_in_correct": correct_active_rate,
            "active_lift_vs_correct": lift,
            "enrichment_vs_correct": enrichment,
            "odds_ratio_vs_correct": odds_ratio,
        }
        sort_enrichment = (
            float(enrichment) if isinstance(enrichment, (int, float)) else -1000000000.0
        )
        rows.append(
            (
                0 if bucket == "correct" else 1,
                -sort_enrichment,
                -active_count,
                bucket,
                row,
            )
        )
    summaries: dict[str, Any] = {}
    for _, _, _, bucket, row in sorted(rows)[: max(1, int(max_groups))]:
        summaries[bucket] = row
    return summaries


def _build_route_ensemble_specs(
    chain_summaries: list[dict[str, Any]],
    *,
    sizes: tuple[int, ...] = DEFAULT_ROUTE_ENSEMBLE_SIZES,
) -> list[dict[str, Any]]:
    ranked_names = [
        str(row["name"])
        for row in chain_summaries
        if isinstance(row.get("name"), str)
        and isinstance(row.get("route_auc", row.get("score_auc")), (int, float))
    ]
    specs: list[dict[str, Any]] = []
    for size in sizes:
        if len(ranked_names) < int(size):
            continue
        chain_names = ranked_names[: int(size)]
        specs.append(
            {
                "name": f"top{int(size)}_chain_mean_score",
                "source": "discovery_top_chain_auc",
                "chain_names": chain_names,
                "active_fraction": DEFAULT_ROUTE_SCORE_ACTIVE_FRACTION,
            }
        )
    return specs


def _summarize_route_ensembles(
    record_chain_states: list[dict[str, Any]], specs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        chain_names = [
            str(name)
            for name in spec.get("chain_names", [])
            if isinstance(name, str) and name
        ]
        if not chain_names:
            continue
        scores: list[float] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        active_any: list[bool] = []
        for record_state in record_chain_states:
            score_map = record_state.get("score_map") or {}
            active_map = record_state.get("active_map") or {}
            values = [
                float(score_map[name])
                for name in chain_names
                if isinstance(score_map.get(name), (int, float))
            ]
            if not values:
                continue
            scores.append(float(np.mean(values)))
            labels.append(int(record_state["label"]))
            sample_ids.append(str(record_state.get("sample_id") or ""))
            active_any.append(any((bool(active_map.get(name)) for name in chain_names)))
        if not scores:
            continue
        summary = _summarize_scored_route(
            name=str(spec.get("name") or "route_ensemble"),
            scores=np.asarray(scores, dtype=np.float64),
            labels=np.asarray(labels, dtype=np.int8),
            sample_ids=sample_ids,
            active_mask=np.asarray(active_any, dtype=bool),
            active_fraction=float(
                spec.get("active_fraction", DEFAULT_ROUTE_SCORE_ACTIVE_FRACTION)
            ),
        )
        summary["source"] = str(spec.get("source") or "route_ensemble")
        summary["chain_names"] = chain_names
        summary["chain_count"] = len(chain_names)
        rows.append(summary)
    rows.sort(
        key=lambda row: (
            -1000000000.0 if row.get("route_auc") is None else -float(row["route_auc"]),
            (
                -1000000000.0
                if row.get("risk_ratio") is None
                else -float(row["risk_ratio"])
            ),
        )
    )
    return rows


def _pair_route_summary(
    record_chain_states: list[dict[str, Any]], *, left_name: str, right_name: str
) -> dict[str, Any] | None:
    scores: list[float] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    active_both: list[bool] = []
    for record_state in record_chain_states:
        score_map = record_state.get("score_map") or {}
        active_map = record_state.get("active_map") or {}
        if not isinstance(score_map.get(left_name), (int, float)):
            continue
        if not isinstance(score_map.get(right_name), (int, float)):
            continue
        scores.append(float(np.mean([score_map[left_name], score_map[right_name]])))
        labels.append(int(record_state["label"]))
        sample_ids.append(str(record_state.get("sample_id") or ""))
        active_both.append(
            bool(active_map.get(left_name, False))
            and bool(active_map.get(right_name, False))
        )
    if not scores:
        return None
    return _summarize_scored_route(
        name=f"{left_name}+{right_name}",
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int8),
        sample_ids=sample_ids,
        active_mask=np.asarray(active_both, dtype=bool),
    )


def _summarize_pair_interactions(
    chain_summaries: list[dict[str, Any]],
    record_chain_states: list[dict[str, Any]],
    *,
    top_k: int,
    max_pairs: int,
    breakdown_min_samples: int,
    breakdown_max_groups: int,
    pair_specs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not chain_summaries or not record_chain_states:
        return []
    if pair_specs is None:
        chain_order = [
            row["name"]
            for row in chain_summaries[: max(2, int(top_k))]
            if isinstance(row.get("name"), str)
        ]
        if len(chain_order) < 2:
            return []
        pair_name_specs = [
            {"left_chain": chain_order[left_idx], "right_chain": chain_order[right_idx]}
            for left_idx in range(len(chain_order))
            for right_idx in range(left_idx + 1, len(chain_order))
        ]
    else:
        pair_name_specs = [
            {
                "left_chain": str(spec.get("left_chain")),
                "right_chain": str(spec.get("right_chain")),
            }
            for spec in pair_specs
            if spec.get("left_chain") and spec.get("right_chain")
        ]
    rows: list[dict[str, Any]] = []
    for spec in pair_name_specs:
        left_name = str(spec["left_chain"])
        right_name = str(spec["right_chain"])
        both_labels: list[int] = []
        left_only_labels: list[int] = []
        right_only_labels: list[int] = []
        neither_labels: list[int] = []
        for record_state in record_chain_states:
            label = int(record_state["label"])
            active_map = record_state.get("active_map", {})
            left_active = bool(active_map.get(left_name, False))
            right_active = bool(active_map.get(right_name, False))
            if left_active and right_active:
                both_labels.append(label)
            elif left_active:
                left_only_labels.append(label)
            elif right_active:
                right_only_labels.append(label)
            else:
                neither_labels.append(label)
        if not both_labels:
            continue

        def error_rate(values: list[int]) -> float | None:
            return float(np.mean(values)) if values else None

        both_error = error_rate(both_labels)
        neither_error = error_rate(neither_labels)
        score_summary = (
            _pair_route_summary(
                record_chain_states, left_name=left_name, right_name=right_name
            )
            or {}
        )
        rows.append(
            {
                "left_chain": left_name,
                "right_chain": right_name,
                "name": f"{left_name}+{right_name}",
                "source": "pair_route",
                "both_active_count": len(both_labels),
                "both_active_rate": len(both_labels) / len(record_chain_states),
                "error_rate_when_both_active": both_error,
                "error_rate_when_left_only": error_rate(left_only_labels),
                "error_rate_when_right_only": error_rate(right_only_labels),
                "error_rate_when_neither_active": neither_error,
                "error_lift_vs_neither": (
                    both_error - neither_error
                    if isinstance(both_error, (int, float))
                    and isinstance(neither_error, (int, float))
                    else None
                ),
                "risk_ratio_vs_neither": (
                    both_error / neither_error
                    if isinstance(both_error, (int, float))
                    and isinstance(neither_error, (int, float))
                    and (neither_error > 0.0)
                    else None
                ),
                **{
                    key: value
                    for (key, value) in score_summary.items()
                    if key
                    not in {
                        "name",
                        "source",
                        "active_count",
                        "inactive_count",
                        "active_rate",
                    }
                },
            }
        )
    rows.sort(
        key=lambda row: (
            (
                -1000000000.0
                if row.get("error_lift_vs_neither") is None
                else -float(row["error_lift_vs_neither"])
            ),
            -int(row.get("both_active_count", 0)),
        )
    )
    return rows[: max(1, int(max_pairs))]


def _analyze_chain_activations(
    records: list[dict[str, Any]],
    llm: Any,
    *,
    layer_indices: list[int],
    chains: list[dict[str, Any]],
    reference: dict[str, Any],
    batch_size: int,
    max_length: int,
    node_z_threshold: float,
    min_active_fraction: float,
    chain_score_mode: str,
    breakdown_min_samples: int,
    breakdown_max_groups: int,
    pairwise_chain_top_k: int,
    pairwise_chain_max_pairs: int,
    route_ensemble_specs: list[dict[str, Any]] | None = None,
    pair_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not chains:
        return {
            "chains": [],
            "route_ensembles": [],
            "route_ensemble_specs": [],
            "pair_interactions": [],
            "pair_specs": [],
        }
    chain_events: dict[str, list[dict[str, Any]]] = {
        chain["name"]: [] for chain in chains
    }
    record_chain_states: list[dict[str, Any]] = []
    for batch_records in tqdm(
        list(_chunked(records, batch_size)),
        desc="Scoring chains",
        unit="batch",
        dynamic_ncols=True,
        disable=not records,
    ):
        results = _analyze_records_batch(
            batch_records, llm, layer_indices=layer_indices, max_length=max_length, k=2
        )
        for record, result in zip(batch_records, results):
            layer_states = result.get("layer_hidden_states") or {}
            active_map: dict[str, bool] = {}
            score_map: dict[str, float] = {}
            for chain in chains:
                event = _score_chain_for_sample(
                    layer_states,
                    chain,
                    reference,
                    node_z_threshold=node_z_threshold,
                    min_active_fraction=min_active_fraction,
                    score_mode=chain_score_mode,
                )
                if event is None:
                    continue
                event["label"] = int(record["label"])
                event["error_semantic_bucket"] = str(
                    record.get("error_semantic_bucket") or "unknown"
                )
                event["sample_id"] = str(record.get("sample_id") or "")
                chain_events[chain["name"]].append(event)
                active_map[chain["name"]] = bool(event["active"])
                score_map[chain["name"]] = float(event["score"])
            record_chain_states.append(
                {
                    "sample_id": str(record.get("sample_id") or ""),
                    "label": int(record["label"]),
                    "active_map": active_map,
                    "score_map": score_map,
                }
            )
    overall_error_rate = (
        float(np.mean([int(record["label"]) for record in records])) if records else 0.0
    )
    summaries = []
    for chain in chains:
        summary = _summarize_chain_events(
            chain,
            chain_events.get(chain["name"], []),
            overall_error_rate=overall_error_rate,
        )
        summary["direction_signature"] = str(chain.get("direction_signature") or "")
        summary["by_error_semantics"] = _summarize_chain_events_by_semantics(
            chain_events.get(chain["name"], []),
            min_samples=breakdown_min_samples,
            max_groups=breakdown_max_groups,
        )
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            -1000000000.0 if row.get("score_auc") is None else -float(row["score_auc"]),
            (
                -1000000000.0
                if row.get("error_lift_vs_inactive") is None
                else -float(row["error_lift_vs_inactive"])
            ),
            -int(row.get("active_count", 0)),
        )
    )
    resolved_ensemble_specs = route_ensemble_specs or _build_route_ensemble_specs(
        summaries
    )
    route_ensembles = _summarize_route_ensembles(
        record_chain_states, resolved_ensemble_specs
    )
    pair_interactions = _summarize_pair_interactions(
        summaries,
        record_chain_states,
        top_k=pairwise_chain_top_k,
        max_pairs=pairwise_chain_max_pairs,
        breakdown_min_samples=breakdown_min_samples,
        breakdown_max_groups=breakdown_max_groups,
        pair_specs=pair_specs,
    )
    resolved_pair_specs = pair_specs or [
        {
            "left_chain": row["left_chain"],
            "right_chain": row["right_chain"],
            "name": row.get("name") or f"{row['left_chain']}+{row['right_chain']}",
            "source": "discovery_pair_interaction",
        }
        for row in pair_interactions
    ]
    return {
        "chains": summaries,
        "route_ensembles": route_ensembles,
        "route_ensemble_specs": resolved_ensemble_specs,
        "pair_interactions": pair_interactions,
        "pair_specs": resolved_pair_specs,
    }


def _score_records_with_top_neurons(
    records: list[dict[str, Any]],
    llm: Any,
    *,
    layer_indices: list[int],
    top_neurons: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    if not top_neurons:
        return {"score_auc": None, "top_neuron_count": 0}
    grouped: dict[int, list[tuple[int, float, float]]] = {}
    for row in top_neurons:
        grouped.setdefault(int(row["layer_index"]), []).append(
            (
                int(row["dimension"]),
                float(row["abs_mean_correct"]),
                float(row["pooled_abs_std"]),
            )
        )
    scores: list[float] = []
    labels: list[int] = []
    for batch_records in tqdm(
        list(_chunked(records, batch_size)),
        desc="Scoring top neurons",
        unit="batch",
        dynamic_ncols=True,
        disable=not records,
    ):
        results = _analyze_records_batch(
            batch_records, llm, layer_indices=layer_indices, max_length=max_length, k=2
        )
        for record, result in zip(batch_records, results):
            layer_states = result.get("layer_hidden_states") or {}
            neuron_scores: list[float] = []
            for layer_idx, entries in grouped.items():
                vector = np.abs(np.asarray(layer_states[layer_idx], dtype=np.float64))
                for dim_idx, correct_mean, pooled_std in entries:
                    denom = pooled_std if pooled_std > 1e-12 else 1.0
                    neuron_scores.append(
                        (float(vector[dim_idx]) - correct_mean) / denom
                    )
            if not neuron_scores:
                continue
            scores.append(float(np.mean(neuron_scores)))
            labels.append(int(record["label"]))
    score_auc = (
        _roc_auc(
            np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int8)
        )
        if scores
        else None
    )
    return {
        "score_auc": score_auc,
        "top_neuron_count": len(top_neurons),
        "score_mean": float(np.mean(scores)) if scores else None,
        "score_std": float(np.std(scores)) if scores else None,
    }


def _class_balance_from_records(
    records: list[dict[str, Any]], *, label_mode: str, negative_mode: str
) -> dict[str, Any]:
    negative_count = sum((1 for record in records if int(record["label"]) == 0))
    positive_count = sum((1 for record in records if int(record["label"]) == 1))
    total = negative_count + positive_count
    return {
        "negative_count": int(negative_count),
        "positive_count": int(positive_count),
        "correct_count": int(negative_count),
        "wrong_count": int(positive_count),
        "label_mode": label_mode,
        "negative_mode": negative_mode,
        "error_rate": float(positive_count) / float(total) if total > 0 else None,
        "positive_rate": float(positive_count) / float(total) if total > 0 else None,
    }


def _best_numeric_entry(values: dict[str, Any]) -> tuple[str | None, float | None]:
    numeric = [
        (str(key), float(value))
        for (key, value) in values.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not numeric:
        return (None, None)
    return max(numeric, key=lambda item: item[1])


def _paper_claim_check(section: dict[str, Any]) -> dict[str, Any]:
    (baseline_name, baseline_auc) = _best_numeric_entry(
        section.get("baseline_uncertainty_auc")
        or section.get("baseline_error_auc")
        or {}
    )
    chains = [
        row
        for row in section.get("chain_activation_analysis") or []
        if isinstance(row.get("route_auc", row.get("score_auc")), (int, float))
    ]
    ensemble_routes = [
        {**row, "route_kind": "ensemble"}
        for row in section.get("route_ensemble_analysis") or []
        if isinstance(row.get("route_auc", row.get("score_auc")), (int, float))
    ]
    pair_routes = [
        {**row, "route_kind": "pair"}
        for row in section.get("chain_pair_interactions") or []
        if isinstance(row.get("route_auc", row.get("score_auc")), (int, float))
    ]
    chain_routes = [{**row, "route_kind": "chain"} for row in chains]
    all_routes = chain_routes + ensemble_routes + pair_routes
    best_chain = max(
        chains,
        key=lambda row: float(row.get("route_auc", row.get("score_auc"))),
        default=None,
    )
    best_route = max(
        all_routes,
        key=lambda row: float(row.get("route_auc", row.get("score_auc"))),
        default=None,
    )
    top_neuron_auc = (section.get("neuron_score") or {}).get("score_auc")
    best_chain_auc = (
        float(best_chain.get("route_auc", best_chain.get("score_auc")))
        if best_chain is not None
        else None
    )
    best_chain_risk = (
        float(best_chain.get("risk_ratio"))
        if best_chain is not None
        and isinstance(best_chain.get("risk_ratio"), (int, float))
        else None
    )
    best_chain_random_risk = (
        float(best_chain.get("random_route_risk_ratio"))
        if best_chain is not None
        and isinstance(best_chain.get("random_route_risk_ratio"), (int, float))
        else None
    )
    best_route_auc = (
        float(best_route.get("route_auc", best_route.get("score_auc")))
        if best_route is not None
        else None
    )
    best_route_risk = (
        float(best_route.get("risk_ratio", best_route.get("risk_ratio_vs_neither")))
        if best_route is not None
        and isinstance(
            best_route.get("risk_ratio", best_route.get("risk_ratio_vs_neither")),
            (int, float),
        )
        else None
    )
    best_route_random_risk = (
        float(best_route.get("random_route_risk_ratio"))
        if best_route is not None
        and isinstance(best_route.get("random_route_risk_ratio"), (int, float))
        else None
    )
    return {
        "best_uncertainty_baseline": baseline_name,
        "best_uncertainty_auc": baseline_auc,
        "top_neuron_route_auc": (
            float(top_neuron_auc) if isinstance(top_neuron_auc, (int, float)) else None
        ),
        "top_neuron_route_auc_exceeds_best_uncertainty": (
            bool(float(top_neuron_auc) > baseline_auc)
            if isinstance(top_neuron_auc, (int, float))
            and isinstance(baseline_auc, (int, float))
            else None
        ),
        "best_chain_name": best_chain.get("name") if best_chain else None,
        "best_chain_route_auc": best_chain_auc,
        "best_chain_route_auc_exceeds_best_uncertainty": (
            bool(best_chain_auc > baseline_auc)
            if isinstance(best_chain_auc, (int, float))
            and isinstance(baseline_auc, (int, float))
            else None
        ),
        "best_chain_risk_ratio": best_chain_risk,
        "best_chain_random_risk_ratio": best_chain_random_risk,
        "best_chain_risk_ratio_over_random": (
            best_chain_risk / best_chain_random_risk
            if isinstance(best_chain_risk, (int, float))
            and isinstance(best_chain_random_risk, (int, float))
            and (best_chain_random_risk > 0.0)
            else None
        ),
        "best_chain_risk_exceeds_random": (
            bool(best_chain_risk > best_chain_random_risk)
            if isinstance(best_chain_risk, (int, float))
            and isinstance(best_chain_random_risk, (int, float))
            else None
        ),
        "best_route_kind": best_route.get("route_kind") if best_route else None,
        "best_route_name": best_route.get("name") if best_route else None,
        "best_route_auc": best_route_auc,
        "best_route_auc_exceeds_best_uncertainty": (
            bool(best_route_auc > baseline_auc)
            if isinstance(best_route_auc, (int, float))
            and isinstance(baseline_auc, (int, float))
            else None
        ),
        "best_route_risk_ratio": best_route_risk,
        "best_route_random_risk_ratio": best_route_random_risk,
        "best_route_risk_ratio_over_random": (
            best_route_risk / best_route_random_risk
            if isinstance(best_route_risk, (int, float))
            and isinstance(best_route_random_risk, (int, float))
            and (best_route_random_risk > 0.0)
            else None
        ),
        "best_route_risk_exceeds_random": (
            bool(best_route_risk > best_route_random_risk)
            if isinstance(best_route_risk, (int, float))
            and isinstance(best_route_random_risk, (int, float))
            else None
        ),
    }


def _score_fixed_route_analysis(
    records: list[dict[str, Any]],
    llm: Any,
    *,
    layer_indices: list[int],
    chain_definitions: list[dict[str, Any]],
    neuron_reference: dict[str, Any],
    top_neurons: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    chain_node_z_threshold: float,
    chain_min_active_fraction: float,
    chain_score_mode: str,
    chain_breakdown_min_samples: int,
    chain_breakdown_max_groups: int,
    pairwise_chain_top_k: int,
    pairwise_chain_max_pairs: int,
    route_ensemble_specs: list[dict[str, Any]] | None = None,
    pair_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    neuron_score = _score_records_with_top_neurons(
        records,
        llm,
        layer_indices=layer_indices,
        top_neurons=top_neurons,
        batch_size=batch_size,
        max_length=max_length,
    )
    chain_analysis = _analyze_chain_activations(
        records,
        llm,
        layer_indices=layer_indices,
        chains=chain_definitions,
        reference=neuron_reference,
        batch_size=batch_size,
        max_length=max_length,
        node_z_threshold=float(chain_node_z_threshold),
        min_active_fraction=float(chain_min_active_fraction),
        chain_score_mode=chain_score_mode,
        breakdown_min_samples=max(1, int(chain_breakdown_min_samples)),
        breakdown_max_groups=max(1, int(chain_breakdown_max_groups)),
        pairwise_chain_top_k=max(2, int(pairwise_chain_top_k)),
        pairwise_chain_max_pairs=max(1, int(pairwise_chain_max_pairs)),
        route_ensemble_specs=route_ensemble_specs,
        pair_specs=pair_specs,
    )
    return {
        "baseline_error_auc": _baseline_auc_summary(records),
        "baseline_uncertainty_auc": _baseline_auc_summary(records),
        "neuron_score": neuron_score,
        "chain_activation_analysis": chain_analysis.get("chains", []),
        "route_ensemble_analysis": chain_analysis.get("route_ensembles", []),
        "chain_pair_interactions": chain_analysis.get("pair_interactions", []),
    }


def analyze_error_associated_neurons(
    input_path: str | Path,
    *,
    model_name: str | None = None,
    device: str | None = None,
    sample_layers: list[int] | None = None,
    label_mode: str = "base_error",
    negative_mode: str = "base_correct",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = 512,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    sample_rate: float = 1.0,
    sample_hash_min: float = 0.0,
    top_neurons: int = DEFAULT_TOP_NEURONS,
    chain_specs: list[str] | None = None,
    chain_file: str | Path | None = None,
    chain_node_z_threshold: float = DEFAULT_CHAIN_NODE_Z_THRESHOLD,
    chain_min_active_fraction: float = DEFAULT_CHAIN_MIN_ACTIVE_FRACTION,
    chain_score_mode: str = "reference_correct",
    auto_chain_min_length: int = DEFAULT_AUTO_CHAIN_MIN_LENGTH,
    auto_chain_top_neurons: int = DEFAULT_AUTO_CHAIN_TOP_NEURONS,
    include_ranked_node_chains: bool = False,
    ranked_chain_global_sizes: list[int] | None = None,
    ranked_chain_per_layer_counts: list[int] | None = None,
    chain_breakdown_min_samples: int = DEFAULT_CHAIN_BREAKDOWN_MIN_SAMPLES,
    chain_breakdown_max_groups: int = DEFAULT_CHAIN_BREAKDOWN_MAX_GROUPS,
    pairwise_chain_top_k: int = DEFAULT_PAIRWISE_CHAIN_TOP_K,
    pairwise_chain_max_pairs: int = DEFAULT_PAIRWISE_CHAIN_MAX_PAIRS,
    heldout_input_path: str | Path | None = None,
    heldout_max_samples: int | None = None,
    heldout_max_per_class: int | None = None,
    heldout_sample_rate: float | None = None,
    heldout_sample_hash_min: float | None = None,
    hidden_cache_path: str | Path | None = None,
    heldout_hidden_cache_path: str | Path | None = None,
    require_hidden_cache: bool = False,
    llm: Any | None = None,
) -> dict[str, Any]:
    if chain_score_mode not in CHAIN_SCORE_MODES:
        raise ValueError(f"Unsupported chain_score_mode: {chain_score_mode}")
    selection = select_records_for_error_neuron_analysis(
        input_path,
        label_mode=label_mode,
        negative_mode=negative_mode,
        max_samples=max_samples,
        max_per_class=max_per_class,
        sample_rate=sample_rate,
        sample_hash_min=sample_hash_min,
    )
    records = selection["selected_records"]
    if not records:
        raise ValueError("No usable prepared samples were selected for neuron analysis")
    hidden_cache_status = _attach_hidden_state_cache(
        records, cache_path=hidden_cache_path, layer_indices=sample_layers
    )
    if require_hidden_cache and hidden_cache_status["missing_count"] > 0:
        raise ValueError(
            f"Train hidden-state cache is incomplete: {hidden_cache_status['missing_count']} selected records are missing"
        )
    resolved_model_name = model_name or next(
        (
            record.get("base_model_name")
            for record in records
            if isinstance(record.get("base_model_name"), str)
            and record.get("base_model_name")
        ),
        "Qwen/Qwen2.5-Coder-1.5B",
    )
    if llm is None and (not _records_have_hidden_cache(records)):
        from ramp_hsr.runtime import QwenCoderLLM

        llm = QwenCoderLLM(model_name=resolved_model_name, device=device)
    layer_indices = list(sample_layers or [])
    correct_group: dict[str, Any] | None = None
    wrong_group: dict[str, Any] | None = None
    base_prediction_mismatches = 0
    analyzed_batches = 0
    for batch_records in tqdm(
        list(_chunked(records, batch_size)),
        desc="Analyzing hidden states",
        unit="batch",
        dynamic_ncols=True,
        disable=not records,
    ):
        batch_results = _analyze_records_batch(
            batch_records,
            llm,
            layer_indices=layer_indices or None,
            max_length=max_length,
            k=2,
        )
        (batch_vectors, resolved_layers, _, mismatch_count) = _prepare_batch_vectors(
            batch_records, batch_results, layer_indices or None
        )
        if not layer_indices:
            layer_indices = resolved_layers
        if correct_group is None or wrong_group is None:
            correct_group = _empty_group_stats(
                num_layers=batch_vectors.shape[1], hidden_dim=batch_vectors.shape[2]
            )
            wrong_group = _empty_group_stats(
                num_layers=batch_vectors.shape[1], hidden_dim=batch_vectors.shape[2]
            )
        labels = np.asarray(
            [record["label"] for record in batch_records], dtype=np.int8
        )
        wrong_mask = labels == 1
        correct_mask = labels == 0
        _update_group_stats(correct_group, batch_vectors[correct_mask])
        _update_group_stats(wrong_group, batch_vectors[wrong_mask])
        base_prediction_mismatches += mismatch_count
        analyzed_batches += 1
    assert correct_group is not None and wrong_group is not None
    neuron_reference = _build_neuron_reference(
        layer_indices, correct_group, wrong_group
    )
    neuron_rows = _compute_neuron_rows(layer_indices, correct_group, wrong_group)
    top_rows = neuron_rows[: max(1, int(top_neurons))]
    auto_chains = _auto_discover_chains(
        neuron_rows,
        min_length=max(2, int(auto_chain_min_length)),
        top_neurons=max(1, int(auto_chain_top_neurons)),
    )
    if include_ranked_node_chains:
        auto_chains.extend(
            _auto_discover_ranked_node_chains(
                neuron_rows,
                layer_indices=layer_indices,
                global_sizes=list(
                    ranked_chain_global_sizes
                    if ranked_chain_global_sizes is not None
                    else DEFAULT_RANKED_CHAIN_GLOBAL_SIZES
                ),
                per_layer_counts=list(
                    ranked_chain_per_layer_counts
                    if ranked_chain_per_layer_counts is not None
                    else DEFAULT_RANKED_CHAIN_PER_LAYER_COUNTS
                ),
                top_neurons=max(1, int(auto_chain_top_neurons)),
            )
        )
    chain_definitions = _resolve_chain_definitions(
        chain_specs=chain_specs, chain_file=chain_file, auto_chains=auto_chains
    )
    per_layer: dict[str, Any] = {}
    for layer_idx in layer_indices:
        layer_rows = [
            row for row in neuron_rows if row["layer_index"] == int(layer_idx)
        ]
        per_layer[str(layer_idx)] = {
            "top_wrong_associated_neurons": layer_rows[: max(1, min(10, top_neurons))],
            "max_abs_cohen_d": (
                float(layer_rows[0]["abs_cohen_d"]) if layer_rows else None
            ),
            "max_abs_gap": float(layer_rows[0]["abs_mean_gap"]) if layer_rows else None,
        }
    neuron_score = _score_records_with_top_neurons(
        records,
        llm,
        layer_indices=layer_indices,
        top_neurons=top_rows,
        batch_size=batch_size,
        max_length=max_length,
    )
    chain_analysis = _analyze_chain_activations(
        records,
        llm,
        layer_indices=layer_indices,
        chains=chain_definitions,
        reference=neuron_reference,
        batch_size=batch_size,
        max_length=max_length,
        node_z_threshold=float(chain_node_z_threshold),
        min_active_fraction=float(chain_min_active_fraction),
        chain_score_mode=chain_score_mode,
        breakdown_min_samples=max(1, int(chain_breakdown_min_samples)),
        breakdown_max_groups=max(1, int(chain_breakdown_max_groups)),
        pairwise_chain_top_k=max(2, int(pairwise_chain_top_k)),
        pairwise_chain_max_pairs=max(1, int(pairwise_chain_max_pairs)),
    )
    baseline_auc = _baseline_auc_summary(records)
    heldout_validation = None
    if heldout_input_path is not None:
        heldout_selection = select_records_for_error_neuron_analysis(
            heldout_input_path,
            label_mode=label_mode,
            negative_mode=negative_mode,
            max_samples=(
                max_samples if heldout_max_samples is None else int(heldout_max_samples)
            ),
            max_per_class=(
                max_per_class
                if heldout_max_per_class is None
                else int(heldout_max_per_class)
            ),
            sample_rate=(
                sample_rate
                if heldout_sample_rate is None
                else float(heldout_sample_rate)
            ),
            sample_hash_min=(
                sample_hash_min
                if heldout_sample_hash_min is None
                else float(heldout_sample_hash_min)
            ),
        )
        heldout_records = heldout_selection["selected_records"]
        if not heldout_records:
            raise ValueError(
                "No usable held-out samples were selected for route validation"
            )
        heldout_hidden_cache_status = _attach_hidden_state_cache(
            heldout_records,
            cache_path=heldout_hidden_cache_path,
            layer_indices=layer_indices,
        )
        if require_hidden_cache and heldout_hidden_cache_status["missing_count"] > 0:
            raise ValueError(
                f"Held-out hidden-state cache is incomplete: {heldout_hidden_cache_status['missing_count']} selected records are missing"
            )
        if llm is None and (not _records_have_hidden_cache(heldout_records)):
            from ramp_hsr.runtime import QwenCoderLLM

            llm = QwenCoderLLM(model_name=resolved_model_name, device=device)
        heldout_scores = _score_fixed_route_analysis(
            heldout_records,
            llm,
            layer_indices=layer_indices,
            chain_definitions=chain_definitions,
            neuron_reference=neuron_reference,
            top_neurons=top_rows,
            batch_size=batch_size,
            max_length=max_length,
            chain_node_z_threshold=float(chain_node_z_threshold),
            chain_min_active_fraction=float(chain_min_active_fraction),
            chain_score_mode=chain_score_mode,
            chain_breakdown_min_samples=max(1, int(chain_breakdown_min_samples)),
            chain_breakdown_max_groups=max(1, int(chain_breakdown_max_groups)),
            pairwise_chain_top_k=max(2, int(pairwise_chain_top_k)),
            pairwise_chain_max_pairs=max(1, int(pairwise_chain_max_pairs)),
            route_ensemble_specs=chain_analysis.get("route_ensemble_specs", []),
            pair_specs=chain_analysis.get("pair_specs", []),
        )
        heldout_validation = {
            "input": {
                "path": str(Path(heldout_input_path).resolve()),
                "max_samples": (
                    int(max_samples)
                    if heldout_max_samples is None
                    else int(heldout_max_samples)
                ),
                "max_per_class": (
                    int(max_per_class)
                    if heldout_max_per_class is None
                    else int(heldout_max_per_class)
                ),
                "sample_rate": (
                    float(sample_rate)
                    if heldout_sample_rate is None
                    else float(heldout_sample_rate)
                ),
                "sample_hash_min": (
                    float(sample_hash_min)
                    if heldout_sample_hash_min is None
                    else float(heldout_sample_hash_min)
                ),
                "chain_score_mode": chain_score_mode,
                "hidden_cache": heldout_hidden_cache_status,
            },
            "selection": {
                key: value
                for (key, value) in heldout_selection.items()
                if key != "selected_records"
            },
            "class_balance": _class_balance_from_records(
                heldout_records, label_mode=label_mode, negative_mode=negative_mode
            ),
            **heldout_scores,
        }
        heldout_validation["paper_claim_checks"] = _paper_claim_check(
            heldout_validation
        )
    report = {
        "analysis_version": ANALYSIS_VERSION,
        "input": {
            "path": str(Path(input_path).resolve()),
            "model_name": resolved_model_name,
            "device": device,
            "sample_layers": layer_indices,
            "label_mode": label_mode,
            "negative_mode": negative_mode,
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "tokenizer_context_policy": (
                llm.tokenizer_context_policy(max_length=max_length)
                if hasattr(llm, "tokenizer_context_policy")
                else None
            ),
            "max_samples": int(max_samples),
            "max_per_class": int(max_per_class),
            "sample_rate": float(sample_rate),
            "sample_hash_min": float(sample_hash_min),
            "top_neurons": int(top_neurons),
            "chain_node_z_threshold": float(chain_node_z_threshold),
            "chain_min_active_fraction": float(chain_min_active_fraction),
            "chain_score_mode": chain_score_mode,
            "auto_chain_min_length": int(auto_chain_min_length),
            "auto_chain_top_neurons": int(auto_chain_top_neurons),
            "include_ranked_node_chains": bool(include_ranked_node_chains),
            "ranked_chain_global_sizes": list(
                ranked_chain_global_sizes
                if ranked_chain_global_sizes is not None
                else DEFAULT_RANKED_CHAIN_GLOBAL_SIZES
            ),
            "ranked_chain_per_layer_counts": list(
                ranked_chain_per_layer_counts
                if ranked_chain_per_layer_counts is not None
                else DEFAULT_RANKED_CHAIN_PER_LAYER_COUNTS
            ),
            "chain_breakdown_min_samples": int(chain_breakdown_min_samples),
            "chain_breakdown_max_groups": int(chain_breakdown_max_groups),
            "pairwise_chain_top_k": int(pairwise_chain_top_k),
            "pairwise_chain_max_pairs": int(pairwise_chain_max_pairs),
            "heldout_input_path": (
                str(Path(heldout_input_path).resolve())
                if heldout_input_path is not None
                else None
            ),
            "hidden_cache": hidden_cache_status,
            "heldout_hidden_cache_path": (
                str(Path(heldout_hidden_cache_path).resolve())
                if heldout_hidden_cache_path is not None
                else None
            ),
            "require_hidden_cache": bool(require_hidden_cache),
        },
        "selection": {
            key: value
            for (key, value) in selection.items()
            if key != "selected_records"
        },
        "base_prediction_replay": {
            "mismatch_count": int(base_prediction_mismatches),
            "mismatch_rate": (
                float(base_prediction_mismatches) / float(len(records))
                if records
                else None
            ),
            "analyzed_batches": int(analyzed_batches),
        },
        "class_balance": {
            "negative_count": int(correct_group["count"]),
            "positive_count": int(wrong_group["count"]),
            "correct_count": int(correct_group["count"]),
            "wrong_count": int(wrong_group["count"]),
            "label_mode": label_mode,
            "negative_mode": negative_mode,
            "error_rate": (
                float(wrong_group["count"])
                / float(correct_group["count"] + wrong_group["count"])
                if correct_group["count"] + wrong_group["count"] > 0
                else None
            ),
            "positive_rate": (
                float(wrong_group["count"])
                / float(correct_group["count"] + wrong_group["count"])
                if correct_group["count"] + wrong_group["count"] > 0
                else None
            ),
        },
        "baseline_error_auc": baseline_auc,
        "baseline_uncertainty_auc": baseline_auc,
        "neuron_score": neuron_score,
        "top_wrong_associated_neurons": top_rows,
        "chain_activation_analysis": chain_analysis.get("chains", []),
        "route_ensemble_analysis": chain_analysis.get("route_ensembles", []),
        "route_ensemble_specs": chain_analysis.get("route_ensemble_specs", []),
        "chain_pair_interactions": chain_analysis.get("pair_interactions", []),
        "pair_specs": chain_analysis.get("pair_specs", []),
        "by_layer": per_layer,
    }
    if heldout_validation is not None:
        report["heldout_validation"] = heldout_validation
    report["paper_claim_checks"] = _paper_claim_check(report)
    return report


def render_error_neuron_analysis_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    input_info = report.get("input", {})
    selection = report.get("selection", {})
    replay = report.get("base_prediction_replay", {})
    baseline_auc = report.get("baseline_error_auc", {})
    neuron_score = report.get("neuron_score", {})
    top_rows = report.get("top_wrong_associated_neurons", [])
    chain_rows = report.get("chain_activation_analysis", [])
    ensemble_rows = report.get("route_ensemble_analysis", [])
    pair_rows = report.get("chain_pair_interactions", [])
    lines.append("# Error-Associated Neuron Analysis")
    lines.append("")
    lines.append(f"- Input: `{input_info.get('path', 'unknown')}`")
    lines.append(f"- Model: `{input_info.get('model_name', 'unknown')}`")
    lines.append(f"- Sample layers: `{input_info.get('sample_layers', [])}`")
    lines.append(f"- Label mode: `{input_info.get('label_mode', 'base_error')}`")
    lines.append(
        f"- Negative mode: `{input_info.get('negative_mode', 'base_correct')}`"
    )
    lines.append(
        f"- Selected samples: `{selection.get('selected_count', 0)}` (negative=`{selection.get('negative_count', selection.get('correct_count', 0))}`, positive=`{selection.get('positive_count', selection.get('wrong_count', 0))}`)"
    )
    lines.append(f"- Base replay mismatch rate: `{_fmt(replay.get('mismatch_rate'))}`")
    lines.append("")
    if baseline_auc:
        lines.append("## Baselines")
        lines.append("")
        for key, value in sorted(
            baseline_auc.items(), key=lambda item: item[1], reverse=True
        ):
            lines.append(f"- `{key}` error AUC: `{_fmt(value)}`")
        lines.append("")
    if neuron_score:
        lines.append("## Neuron Score")
        lines.append("")
        lines.append(
            f"- Top-neuron composite error AUC: `{_fmt(neuron_score.get('score_auc'))}`"
        )
        lines.append(f"- Top neurons used: `{neuron_score.get('top_neuron_count', 0)}`")
        lines.append("")
    claim_checks = report.get("paper_claim_checks") or {}
    if claim_checks:
        lines.append("## Paper Claim Checks")
        lines.append("")
        lines.append(
            f"- Best uncertainty baseline: `{claim_checks.get('best_uncertainty_baseline')}` AUC `{_fmt(claim_checks.get('best_uncertainty_auc'))}`"
        )
        lines.append(
            f"- Best chain: `{claim_checks.get('best_chain_name')}` route AUC `{_fmt(claim_checks.get('best_chain_route_auc'))}`, risk ratio `{_fmt(claim_checks.get('best_chain_risk_ratio'))}`, random risk `{_fmt(claim_checks.get('best_chain_random_risk_ratio'))}`"
        )
        lines.append(
            f"- Best chain risk > random: `{claim_checks.get('best_chain_risk_exceeds_random')}`; best chain AUC > best uncertainty: `{claim_checks.get('best_chain_route_auc_exceeds_best_uncertainty')}`"
        )
        lines.append(
            f"- Best route overall: `{claim_checks.get('best_route_name')}` ({claim_checks.get('best_route_kind')}) AUC `{_fmt(claim_checks.get('best_route_auc'))}`, risk `{_fmt(claim_checks.get('best_route_risk_ratio'))}`, random risk `{_fmt(claim_checks.get('best_route_random_risk_ratio'))}`"
        )
        lines.append(
            f"- Best route risk > random: `{claim_checks.get('best_route_risk_exceeds_random')}`; best route AUC > best uncertainty: `{claim_checks.get('best_route_auc_exceeds_best_uncertainty')}`"
        )
        lines.append(
            f"- Top-neuron route AUC > best uncertainty: `{claim_checks.get('top_neuron_route_auc_exceeds_best_uncertainty')}`"
        )
        lines.append("")
    if top_rows:
        lines.append("## Top Wrong-Associated Neurons")
        lines.append("")
        lines.append(
            "| Rank | Layer | Dim | Abs d | Abs Gap | Wrong/Correct Ratio | Signed Gap |"
        )
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for rank, row in enumerate(top_rows, start=1):
            lines.append(
                f"| {rank} | {row['layer_index']} | {row['dimension']} | {_fmt(row['abs_cohen_d'])} | {_fmt(row['abs_mean_gap'])} | {_fmt(row['abs_ratio_wrong_vs_correct'])} | {_fmt(row['signed_mean_gap'])} |"
            )
        lines.append("")
    if chain_rows:
        lines.append("## Chain Activation Analysis")
        lines.append("")
        lines.append(
            "| Chain | Nodes | Dir | Source | Route AUC | Random AUC | Shuffled AUC | P(pos active) | P(pos inactive) | Lift | Risk Ratio | Random Risk | Shuffled Risk | Odds Ratio | Active Rate |"
        )
        lines.append(
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in chain_rows[:20]:
            lines.append(
                f"| `{row['name']}` | {row.get('node_count', 0)} | `{row.get('direction_signature', '') or 'n/a'}` | `{row.get('source', 'n/a')}` | {_fmt(row.get('route_auc', row.get('score_auc')))} | {_fmt(row.get('random_route_auc'))} | {_fmt(row.get('shuffled_label_route_auc'))} | {_fmt(row.get('positive_rate_when_active', row.get('error_rate_when_active')))} | {_fmt(row.get('positive_rate_when_inactive', row.get('error_rate_when_inactive')))} | {_fmt(row.get('positive_lift_vs_inactive', row.get('error_lift_vs_inactive')))} | {_fmt(row.get('risk_ratio'))} | {_fmt(row.get('random_route_risk_ratio'))} | {_fmt(row.get('shuffled_label_risk_ratio'))} | {_fmt(row.get('odds_ratio'))} | {_fmt(row.get('active_rate'))} |"
            )
        lines.append("")
        for row in chain_rows[:5]:
            by_semantics = row.get("by_error_semantics") or {}
            if by_semantics:
                lines.append(f"### {row['name']} By Error Semantics")
                lines.append("")
                lines.append(
                    "| Semantic Bucket | Count | Active | Active Rate | Correct Active Rate | Enrichment | Odds Ratio |"
                )
                lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
                for bucket, bucket_row in by_semantics.items():
                    lines.append(
                        f"| `{bucket}` | {bucket_row.get('count', 0)} | {bucket_row.get('active_count', 0)} | {_fmt(bucket_row.get('active_rate_in_bucket'))} | {_fmt(bucket_row.get('active_rate_in_correct'))} | {_fmt(bucket_row.get('enrichment_vs_correct'))} | {_fmt(bucket_row.get('odds_ratio_vs_correct'))} |"
                    )
                lines.append("")
    if ensemble_rows:
        lines.append("## Route Ensemble Analysis")
        lines.append("")
        lines.append(
            "| Ensemble | Chains | Route AUC | Random AUC | Shuffled AUC | P(pos active) | P(pos inactive) | Risk Ratio | Random Risk | Shuffled Risk | Active Rate |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in ensemble_rows[:20]:
            lines.append(
                f"| `{row['name']}` | {row.get('chain_count', 0)} | {_fmt(row.get('route_auc', row.get('score_auc')))} | {_fmt(row.get('random_route_auc'))} | {_fmt(row.get('shuffled_label_route_auc'))} | {_fmt(row.get('positive_rate_when_active'))} | {_fmt(row.get('positive_rate_when_inactive'))} | {_fmt(row.get('risk_ratio'))} | {_fmt(row.get('random_route_risk_ratio'))} | {_fmt(row.get('shuffled_label_risk_ratio'))} | {_fmt(row.get('active_rate'))} |"
            )
        lines.append("")
    heldout = report.get("heldout_validation") or {}
    heldout_chains = heldout.get("chain_activation_analysis") or []
    heldout_ensembles = heldout.get("route_ensemble_analysis") or []
    heldout_pairs = heldout.get("chain_pair_interactions") or []
    if heldout:
        heldout_input = heldout.get("input") or {}
        heldout_selection = heldout.get("selection") or {}
        heldout_baseline = heldout.get("baseline_uncertainty_auc") or {}
        heldout_neuron_score = heldout.get("neuron_score") or {}
        lines.append("## Held-Out Validation")
        lines.append("")
        lines.append(f"- Held-out input: `{heldout_input.get('path', 'unknown')}`")
        lines.append(
            f"- Held-out selected samples: `{heldout_selection.get('selected_count', 0)}` (negative=`{heldout_selection.get('negative_count', heldout_selection.get('correct_count', 0))}`, positive=`{heldout_selection.get('positive_count', heldout_selection.get('wrong_count', 0))}`)"
        )
        if heldout_baseline:
            best_baseline = max(heldout_baseline.items(), key=lambda item: item[1])
            lines.append(
                f"- Best held-out uncertainty baseline: `{best_baseline[0]}` AUC `{_fmt(best_baseline[1])}`"
            )
        if heldout_neuron_score:
            lines.append(
                f"- Held-out top-neuron composite AUC: `{_fmt(heldout_neuron_score.get('score_auc'))}`"
            )
        heldout_claims = heldout.get("paper_claim_checks") or {}
        if heldout_claims:
            lines.append(
                f"- Held-out best chain: `{heldout_claims.get('best_chain_name')}` route AUC `{_fmt(heldout_claims.get('best_chain_route_auc'))}`, risk ratio `{_fmt(heldout_claims.get('best_chain_risk_ratio'))}`, random risk `{_fmt(heldout_claims.get('best_chain_random_risk_ratio'))}`"
            )
            lines.append(
                f"- Held-out best route overall: `{heldout_claims.get('best_route_name')}` ({heldout_claims.get('best_route_kind')}) AUC `{_fmt(heldout_claims.get('best_route_auc'))}`, risk `{_fmt(heldout_claims.get('best_route_risk_ratio'))}`, random risk `{_fmt(heldout_claims.get('best_route_random_risk_ratio'))}`"
            )
            lines.append(
                f"- Held-out chain risk > random: `{heldout_claims.get('best_chain_risk_exceeds_random')}`; held-out chain AUC > best uncertainty: `{heldout_claims.get('best_chain_route_auc_exceeds_best_uncertainty')}`; held-out best route AUC > best uncertainty: `{heldout_claims.get('best_route_auc_exceeds_best_uncertainty')}`; held-out top-neuron route AUC > best uncertainty: `{heldout_claims.get('top_neuron_route_auc_exceeds_best_uncertainty')}`"
            )
        lines.append("")
        if heldout_chains:
            lines.append(
                "| Chain | Route AUC | Random AUC | Shuffled AUC | P(pos active) | P(pos inactive) | Risk Ratio | Random Risk | Shuffled Risk | Active Rate |"
            )
            lines.append(
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            )
            for row in heldout_chains[:20]:
                lines.append(
                    f"| `{row['name']}` | {_fmt(row.get('route_auc', row.get('score_auc')))} | {_fmt(row.get('random_route_auc'))} | {_fmt(row.get('shuffled_label_route_auc'))} | {_fmt(row.get('positive_rate_when_active', row.get('error_rate_when_active')))} | {_fmt(row.get('positive_rate_when_inactive', row.get('error_rate_when_inactive')))} | {_fmt(row.get('risk_ratio'))} | {_fmt(row.get('random_route_risk_ratio'))} | {_fmt(row.get('shuffled_label_risk_ratio'))} | {_fmt(row.get('active_rate'))} |"
                )
            lines.append("")
        if heldout_ensembles:
            lines.append("### Held-Out Route Ensembles")
            lines.append("")
            lines.append(
                "| Ensemble | Chains | Route AUC | Random AUC | Shuffled AUC | P(pos active) | P(pos inactive) | Risk Ratio | Random Risk | Shuffled Risk | Active Rate |"
            )
            lines.append(
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            )
            for row in heldout_ensembles[:20]:
                lines.append(
                    f"| `{row['name']}` | {row.get('chain_count', 0)} | {_fmt(row.get('route_auc', row.get('score_auc')))} | {_fmt(row.get('random_route_auc'))} | {_fmt(row.get('shuffled_label_route_auc'))} | {_fmt(row.get('positive_rate_when_active'))} | {_fmt(row.get('positive_rate_when_inactive'))} | {_fmt(row.get('risk_ratio'))} | {_fmt(row.get('random_route_risk_ratio'))} | {_fmt(row.get('shuffled_label_risk_ratio'))} | {_fmt(row.get('active_rate'))} |"
                )
            lines.append("")
        if heldout_pairs:
            lines.append("### Held-Out Pair Interactions")
            lines.append("")
            lines.append(
                "| Left | Right | Route AUC | Random AUC | Both Active | Error Both | Error Neither | Lift | Risk Ratio | Random Risk |"
            )
            lines.append(
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            )
            for row in heldout_pairs[:20]:
                lines.append(
                    f"| `{row['left_chain']}` | `{row['right_chain']}` | {_fmt(row.get('route_auc', row.get('score_auc')))} | {_fmt(row.get('random_route_auc'))} | {row.get('both_active_count', 0)} | {_fmt(row.get('error_rate_when_both_active'))} | {_fmt(row.get('error_rate_when_neither_active'))} | {_fmt(row.get('error_lift_vs_neither'))} | {_fmt(row.get('risk_ratio_vs_neither'))} | {_fmt(row.get('random_route_risk_ratio'))} |"
                )
            lines.append("")
    if pair_rows:
        lines.append("## Pair Interactions")
        lines.append("")
        lines.append(
            "| Left | Right | Route AUC | Random AUC | Both Active | Error Both | Error Neither | Lift | Risk Ratio | Random Risk |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in pair_rows[:20]:
            lines.append(
                f"| `{row['left_chain']}` | `{row['right_chain']}` | {_fmt(row.get('route_auc', row.get('score_auc')))} | {_fmt(row.get('random_route_auc'))} | {row.get('both_active_count', 0)} | {_fmt(row.get('error_rate_when_both_active'))} | {_fmt(row.get('error_rate_when_neither_active'))} | {_fmt(row.get('error_lift_vs_neither'))} | {_fmt(row.get('risk_ratio_vs_neither'))} | {_fmt(row.get('random_route_risk_ratio'))} |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_error_neuron_outputs(
    report: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "error_neuron_analysis.json"
    md_path = out_dir / "error_neuron_analysis.md"
    write_json(report, json_path)
    md_path.write_text(render_error_neuron_analysis_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}

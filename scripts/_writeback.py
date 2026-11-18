from __future__ import annotations
from collections import defaultdict
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluate_continuation import (
    _append_token_text,
    _extract_runtime_features_from_context_ids_batch,
    _greedy_generate_token_ids_from_context_batch,
    _last_added_python_name_piece,
    _predict_repair_tokens_batch,
    _runtime_sample_from_features,
)
from ramp_hsr.runtime_helpers import OfflineEvaluator
from ramp_hsr.sample_features import (
    RISK_ACTIVE_COMPOSITE_THRESHOLD,
    RISK_ACTIVE_PAIR_THRESHOLD,
)

GAP_BUCKETS = (("0-3", 0, 3), ("4-15", 4, 15), ("16-63", 16, 63), (">=64", 64, None))


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _threshold_active(composite: float, pair: float) -> bool:
    return (
        composite >= RISK_ACTIVE_COMPOSITE_THRESHOLD
        or pair >= RISK_ACTIVE_PAIR_THRESHOLD
    )


def _feature_float(features: dict[str, Any], key: str) -> float:
    raw = features.get(key)
    if raw is None:
        uncertainty = features.get("uncertainty")
        if isinstance(uncertainty, dict):
            raw = uncertainty.get(key)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _gap_bucket(gap: int) -> str:
    for label, lo, hi in GAP_BUCKETS:
        if gap >= lo and (hi is None or gap <= hi):
            return label
    return "unknown"


def _reference_identifier_step_list(sample: dict[str, Any], steps: int) -> list[int]:
    target_pieces = [
        str(piece) for piece in sample["_rollout_target_token_pieces"][:steps]
    ]
    gold_prefix = str(sample["_rollout_prefix"])
    event_steps: list[int] = []
    for step, piece in enumerate(target_pieces):
        is_identifier = _last_added_python_name_piece(gold_prefix, piece)
        if is_identifier:
            event_steps.append(step)
        gold_prefix += piece
    return event_steps


def _extract_and_predict(
    *,
    evaluator: OfflineEvaluator,
    repair_moe,
    norm,
    lm_head,
    tokenizer: Any,
    device: str,
    route_schema: tuple[str, ...],
    code_schema: tuple[str, ...],
    hidden_state_interface: str,
    items: list[dict[str, Any]],
    max_context_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not items:
        return ([], [])
    features = _extract_runtime_features_from_context_ids_batch(
        evaluator=evaluator,
        samples=[item["sample"] for item in items],
        prefixes=[str(item["prefix"]) for item in items],
        context_id_rows=[
            [int(token_id) for token_id in item["context_ids"]] for item in items
        ],
        max_context_tokens=max_context_tokens,
    )
    runtime_samples = [
        _runtime_sample_from_features(
            item["sample"],
            prefix=str(item["prefix"]),
            features=feature,
            generated_prefix=str(item.get("generated_prefix") or ""),
        )
        for (item, feature) in zip(items, features, strict=True)
    ]
    predictions = _predict_repair_tokens_batch(
        repair_moe=repair_moe,
        norm=norm,
        lm_head=lm_head,
        tokenizer=tokenizer,
        runtime_samples=runtime_samples,
        device=device,
        route_feature_schema=route_schema,
        code_feature_schema=code_schema,
        runtime_gate_mode="always_repair",
        gate_risk_threshold=0.0,
        gate_margin_threshold=0.0,
        gate_confidence_threshold=0.0,
        hidden_state_interface=hidden_state_interface,
    )
    return (features, predictions)


def _prediction_token_piece(
    *, tokenizer: Any, context_ids: list[int], prediction: dict[str, Any]
) -> tuple[int | None, str, list[int]]:
    token_id_raw = prediction.get("prediction_token_id")
    token_id = int(token_id_raw) if token_id_raw is not None else None
    (token_text, next_context_ids) = _append_token_text(
        tokenizer,
        list(context_ids),
        token_id,
        fallback_text=str(prediction.get("prediction") or ""),
    )
    return (token_id, token_text, next_context_ids)


def _length_bucket(length: int, quantum: int = 64) -> int:
    length = max(0, int(length))
    if length == 0:
        return 0
    quantum = max(1, int(quantum))
    return int(math.ceil(length / quantum) * quantum)


def _length_summary(lengths: list[int], *, bucket_quantum: int = 64) -> dict[str, Any]:
    nonzero = [int(length) for length in lengths if int(length) > 0]
    if not nonzero:
        return {
            "nonzero_rows": 0,
            "max": 0,
            "mean": 0.0,
            "bucket_quantum": int(bucket_quantum),
            "buckets": {},
        }
    buckets: dict[str, int] = defaultdict(int)
    for length in nonzero:
        buckets[str(_length_bucket(length, quantum=bucket_quantum))] += 1
    return {
        "nonzero_rows": len(nonzero),
        "max": max(nonzero),
        "mean": round(sum(nonzero) / len(nonzero), 3),
        "bucket_quantum": int(bucket_quantum),
        "buckets": dict(sorted(buckets.items(), key=lambda item: int(item[0]))),
    }


def _greedy_generate_token_ids_from_context_batch_by_length(
    llm: Any,
    context_rows: list[list[int]],
    *,
    max_new_tokens_by_row: list[int],
    max_length: int,
    batch_size: int,
    bucket_quantum: int = 64,
) -> list[list[int]]:
    if len(context_rows) != len(max_new_tokens_by_row):
        raise ValueError("context_rows and max_new_tokens_by_row length mismatch")
    rows: list[list[int]] = [[] for _ in context_rows]
    groups: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(max_new_tokens_by_row):
        bucket = _length_bucket(int(length), quantum=bucket_quantum)
        if bucket > 0:
            groups[bucket].append(index)
    for bucket, indices in sorted(groups.items()):
        for start in range(0, len(indices), max(1, int(batch_size))):
            chunk_indices = indices[start : start + max(1, int(batch_size))]
            generated = _greedy_generate_token_ids_from_context_batch(
                llm,
                [
                    [int(token_id) for token_id in context_rows[index]]
                    for index in chunk_indices
                ],
                max_new_tokens=int(bucket),
                max_length=max_length,
            )
            for index, row in zip(chunk_indices, generated, strict=True):
                wanted = int(max_new_tokens_by_row[index])
                rows[index] = [int(token_id) for token_id in row[:wanted]]
    return rows

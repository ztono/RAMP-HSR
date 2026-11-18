from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluate_continuation import (
    _append_token_text,
    _collect_samples,
    _decode_single_token,
    _encode_text,
    _extract_runtime_features_from_context_ids_batch,
    _fallback_context_ids,
)
from scripts._writeback import (
    _extract_and_predict,
    _feature_float,
    _gap_bucket,
    _greedy_generate_token_ids_from_context_batch_by_length,
    _length_summary,
    _prediction_token_piece,
    _reference_identifier_step_list,
    _safe_rate,
    _threshold_active,
)
from ramp_hsr.checkpoints import load_editor
from scripts._rollout_features import (
    _base_prediction_token_id,
    _resolve_norm_and_lm_head,
    _resolve_repair_feature_schema,
)
from ramp_hsr.runtime_helpers import OfflineEvaluator, _take_evenly

GAP_BUCKET_ORDER = ("0-3", "4-15", "16-63", ">=64")
ROUTE_SCORE_PREFIXES = ("neuron_chain_score_", "neuron_chain_pair_score_")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _risk(feature: dict[str, Any]) -> dict[str, Any]:
    composite = _feature_float(feature, "neuron_chain_composite_risk")
    pair = _feature_float(feature, "neuron_chain_pair_risk")
    return {
        "composite": composite,
        "pair": pair,
        "active": _threshold_active(composite, pair),
    }


def _route_score_vector(feature: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    sources = [feature]
    uncertainty = feature.get("uncertainty")
    if isinstance(uncertainty, dict):
        sources.append(uncertainty)
    for source in sources:
        for key, raw in source.items():
            if not any(
                (str(key).startswith(prefix) for prefix in ROUTE_SCORE_PREFIXES)
            ):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values[str(key)] = value
    if values:
        return values
    return {
        "neuron_chain_composite_risk": _feature_float(
            feature, "neuron_chain_composite_risk"
        ),
        "neuron_chain_pair_risk": _feature_float(feature, "neuron_chain_pair_risk"),
    }


def _top_routes(route_vector: dict[str, float], k: int) -> list[str]:
    return [
        key
        for (key, _value) in sorted(
            route_vector.items(), key=lambda item: (-float(item[1]), item[0])
        )[: max(1, int(k))]
    ]


def _route_pattern_delta(
    left: dict[str, Any], right: dict[str, Any], *, k: int
) -> dict[str, Any]:
    left_vector = _route_score_vector(left)
    right_vector = _route_score_vector(right)
    left_top = _top_routes(left_vector, k)
    right_top = _top_routes(right_vector, k)
    top1_changed = int(bool(left_top and right_top and (left_top[0] != right_top[0])))
    denominator = max(1, min(max(1, int(k)), max(len(left_top), len(right_top))))
    turnover = 1.0 - len(set(left_top) & set(right_top)) / denominator
    keys = sorted(set(left_vector) | set(right_vector))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for key in keys:
        left_value = float(left_vector.get(key, 0.0))
        right_value = float(right_vector.get(key, 0.0))
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0.0 and right_norm <= 0.0:
        cosine_distance = 0.0
    elif left_norm <= 0.0 or right_norm <= 0.0:
        cosine_distance = 1.0
    else:
        cosine = dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
        cosine_distance = 1.0 - max(-1.0, min(1.0, cosine))
    return {
        "top1_changed": top1_changed,
        "turnover_at_k": float(turnover),
        "cosine_distance": float(cosine_distance),
        "vector_dim": len(keys),
        "left_top1": left_top[0] if left_top else None,
        "right_top1": right_top[0] if right_top else None,
        "left_topk": left_top,
        "right_topk": right_top,
    }


def _token_id_list(values: list[int] | tuple[int, ...] | None) -> list[int]:
    if values is None:
        return []
    return [int(value) for value in values]


def _empty_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "sum_gap": 0.0,
        "first_pre_comp": 0.0,
        "first_post_comp": 0.0,
        "first_delta_comp": 0.0,
        "first_pre_pair": 0.0,
        "first_post_pair": 0.0,
        "first_delta_pair": 0.0,
        "first_pre_active": 0,
        "first_post_active": 0,
        "base_second_pre_comp": 0.0,
        "base_second_pre_pair": 0.0,
        "base_second_pre_active": 0,
        "base_second_post_comp": 0.0,
        "base_second_post_pair": 0.0,
        "base_second_post_active": 0,
        "first_only_second_post_comp": 0.0,
        "first_only_second_post_pair": 0.0,
        "first_only_second_post_active": 0,
        "second_pre_comp": 0.0,
        "second_post_comp": 0.0,
        "second_delta_comp": 0.0,
        "second_pre_pair": 0.0,
        "second_post_pair": 0.0,
        "second_delta_pair": 0.0,
        "second_pre_active": 0,
        "second_post_active": 0,
        "first_writeback_delta_comp": 0.0,
        "first_writeback_delta_pair": 0.0,
        "first_writeback_post_delta_comp": 0.0,
        "first_writeback_post_delta_pair": 0.0,
        "second_repair_post_delta_comp": 0.0,
        "second_repair_post_delta_pair": 0.0,
        "second_base_wrong": 0,
        "second_repair_correct": 0,
        "second_fix": 0,
        "second_damage": 0,
        "second_changed": 0,
        "base_vs_first_route_compared": 0,
        "base_vs_first_top1_changed": 0,
        "base_vs_first_turnover_at_k": 0.0,
        "base_vs_first_cosine_distance": 0.0,
        "base_vs_first_vector_dim": 0.0,
        "first_vs_second_route_compared": 0,
        "first_vs_second_top1_changed": 0,
        "first_vs_second_turnover_at_k": 0.0,
        "first_vs_second_cosine_distance": 0.0,
        "first_vs_second_vector_dim": 0.0,
    }


def _update_stats(
    stats: dict[str, Any],
    *,
    gap: int,
    first_pre: dict[str, Any],
    first_post: dict[str, Any],
    base_second_pre: dict[str, Any],
    base_second_post: dict[str, Any],
    first_only_second_post: dict[str, Any],
    second_pre: dict[str, Any],
    second_post: dict[str, Any],
    second_base_token_id: int | None,
    second_repair_token_id: int | None,
    second_target_token_id: int,
    route_turnover_k: int = 5,
) -> None:
    first_pre_risk = _risk(first_pre)
    first_post_risk = _risk(first_post)
    base_second_pre_risk = _risk(base_second_pre)
    base_second_post_risk = _risk(base_second_post)
    first_only_second_post_risk = _risk(first_only_second_post)
    second_pre_risk = _risk(second_pre)
    second_post_risk = _risk(second_post)
    base_correct = second_base_token_id is not None and int(
        second_base_token_id
    ) == int(second_target_token_id)
    repair_correct = second_repair_token_id is not None and int(
        second_repair_token_id
    ) == int(second_target_token_id)
    changed = (
        second_base_token_id is not None
        and second_repair_token_id is not None
        and (int(second_base_token_id) != int(second_repair_token_id))
    )
    stats["count"] += 1
    stats["sum_gap"] += float(gap)
    for prefix, pre_risk, post_risk in (
        ("first", first_pre_risk, first_post_risk),
        ("second", second_pre_risk, second_post_risk),
    ):
        stats[f"{prefix}_pre_comp"] += float(pre_risk["composite"])
        stats[f"{prefix}_post_comp"] += float(post_risk["composite"])
        stats[f"{prefix}_delta_comp"] += float(post_risk["composite"]) - float(
            pre_risk["composite"]
        )
        stats[f"{prefix}_pre_pair"] += float(pre_risk["pair"])
        stats[f"{prefix}_post_pair"] += float(post_risk["pair"])
        stats[f"{prefix}_delta_pair"] += float(post_risk["pair"]) - float(
            pre_risk["pair"]
        )
        stats[f"{prefix}_pre_active"] += int(bool(pre_risk["active"]))
        stats[f"{prefix}_post_active"] += int(bool(post_risk["active"]))
    stats["base_second_pre_comp"] += float(base_second_pre_risk["composite"])
    stats["base_second_pre_pair"] += float(base_second_pre_risk["pair"])
    stats["base_second_pre_active"] += int(bool(base_second_pre_risk["active"]))
    stats["base_second_post_comp"] += float(base_second_post_risk["composite"])
    stats["base_second_post_pair"] += float(base_second_post_risk["pair"])
    stats["base_second_post_active"] += int(bool(base_second_post_risk["active"]))
    stats["first_only_second_post_comp"] += float(
        first_only_second_post_risk["composite"]
    )
    stats["first_only_second_post_pair"] += float(first_only_second_post_risk["pair"])
    stats["first_only_second_post_active"] += int(
        bool(first_only_second_post_risk["active"])
    )
    stats["first_writeback_delta_comp"] += float(second_pre_risk["composite"]) - float(
        base_second_pre_risk["composite"]
    )
    stats["first_writeback_delta_pair"] += float(second_pre_risk["pair"]) - float(
        base_second_pre_risk["pair"]
    )
    stats["first_writeback_post_delta_comp"] += float(
        first_only_second_post_risk["composite"]
    ) - float(base_second_post_risk["composite"])
    stats["first_writeback_post_delta_pair"] += float(
        first_only_second_post_risk["pair"]
    ) - float(base_second_post_risk["pair"])
    stats["second_repair_post_delta_comp"] += float(
        second_post_risk["composite"]
    ) - float(first_only_second_post_risk["composite"])
    stats["second_repair_post_delta_pair"] += float(second_post_risk["pair"]) - float(
        first_only_second_post_risk["pair"]
    )
    stats["second_base_wrong"] += int(not base_correct)
    stats["second_repair_correct"] += int(repair_correct)
    stats["second_fix"] += int(not base_correct and repair_correct)
    stats["second_damage"] += int(base_correct and (not repair_correct))
    stats["second_changed"] += int(changed)
    base_vs_first = _route_pattern_delta(
        base_second_pre, second_pre, k=int(route_turnover_k)
    )
    first_vs_second = _route_pattern_delta(
        first_only_second_post, second_post, k=int(route_turnover_k)
    )
    for prefix, delta in (
        ("base_vs_first", base_vs_first),
        ("first_vs_second", first_vs_second),
    ):
        stats[f"{prefix}_route_compared"] += 1
        stats[f"{prefix}_top1_changed"] += int(delta["top1_changed"])
        stats[f"{prefix}_turnover_at_k"] += float(delta["turnover_at_k"])
        stats[f"{prefix}_cosine_distance"] += float(delta["cosine_distance"])
        stats[f"{prefix}_vector_dim"] += float(delta["vector_dim"])


def _event_record(
    *,
    sample_index: int,
    initial_step: int,
    second_step: int,
    first_pre: dict[str, Any],
    first_post: dict[str, Any],
    base_second_pre: dict[str, Any],
    base_second_post: dict[str, Any],
    first_only_second_post: dict[str, Any],
    second_pre: dict[str, Any],
    second_post: dict[str, Any],
    first_base_token_id: int | None,
    first_repair_token_id: int | None,
    first_target_token_id: int,
    second_base_token_id: int | None,
    second_repair_token_id: int | None,
    second_target_token_id: int,
    base_branch_token_ids_to_second_pre: list[int] | None = None,
    base_branch_token_ids_to_second_post: list[int] | None = None,
    first_only_branch_token_ids_to_second_pre: list[int] | None = None,
    first_only_branch_token_ids_to_second_post: list[int] | None = None,
    first_plus_second_branch_token_ids_to_second_post: list[int] | None = None,
    route_turnover_k: int = 5,
) -> dict[str, Any]:
    gap = max(0, int(second_step) - int(initial_step) - 1)
    first_pre_risk = _risk(first_pre)
    first_post_risk = _risk(first_post)
    base_second_pre_risk = _risk(base_second_pre)
    base_second_post_risk = _risk(base_second_post)
    first_only_second_post_risk = _risk(first_only_second_post)
    second_pre_risk = _risk(second_pre)
    second_post_risk = _risk(second_post)
    second_base_correct = second_base_token_id is not None and int(
        second_base_token_id
    ) == int(second_target_token_id)
    second_repair_correct = second_repair_token_id is not None and int(
        second_repair_token_id
    ) == int(second_target_token_id)
    second_changed = (
        second_base_token_id is not None
        and second_repair_token_id is not None
        and (int(second_base_token_id) != int(second_repair_token_id))
    )
    base_vs_first = _route_pattern_delta(
        base_second_pre, second_pre, k=int(route_turnover_k)
    )
    first_vs_second = _route_pattern_delta(
        first_only_second_post, second_post, k=int(route_turnover_k)
    )
    return {
        "sample_index": int(sample_index),
        "initial_step": int(initial_step),
        "second_step": int(second_step),
        "gap": int(gap),
        "first_base_token_id": (
            int(first_base_token_id) if first_base_token_id is not None else None
        ),
        "first_repair_token_id": (
            int(first_repair_token_id) if first_repair_token_id is not None else None
        ),
        "first_target_token_id": int(first_target_token_id),
        "first_pre_comp": round(float(first_pre_risk["composite"]), 6),
        "first_post_comp": round(float(first_post_risk["composite"]), 6),
        "first_delta_comp": round(
            float(first_post_risk["composite"]) - float(first_pre_risk["composite"]), 6
        ),
        "first_pre_pair": round(float(first_pre_risk["pair"]), 6),
        "first_post_pair": round(float(first_post_risk["pair"]), 6),
        "first_delta_pair": round(
            float(first_post_risk["pair"]) - float(first_pre_risk["pair"]), 6
        ),
        "first_pre_active": int(bool(first_pre_risk["active"])),
        "first_post_active": int(bool(first_post_risk["active"])),
        "base_second_pre_comp": round(float(base_second_pre_risk["composite"]), 6),
        "base_second_pre_pair": round(float(base_second_pre_risk["pair"]), 6),
        "base_second_pre_active": int(bool(base_second_pre_risk["active"])),
        "base_second_post_comp": round(float(base_second_post_risk["composite"]), 6),
        "base_second_post_pair": round(float(base_second_post_risk["pair"]), 6),
        "base_second_post_active": int(bool(base_second_post_risk["active"])),
        "first_only_second_post_comp": round(
            float(first_only_second_post_risk["composite"]), 6
        ),
        "first_only_second_post_pair": round(
            float(first_only_second_post_risk["pair"]), 6
        ),
        "first_only_second_post_active": int(
            bool(first_only_second_post_risk["active"])
        ),
        "first_writeback_delta_comp": round(
            float(second_pre_risk["composite"])
            - float(base_second_pre_risk["composite"]),
            6,
        ),
        "first_writeback_delta_pair": round(
            float(second_pre_risk["pair"]) - float(base_second_pre_risk["pair"]), 6
        ),
        "first_writeback_post_delta_comp": round(
            float(first_only_second_post_risk["composite"])
            - float(base_second_post_risk["composite"]),
            6,
        ),
        "first_writeback_post_delta_pair": round(
            float(first_only_second_post_risk["pair"])
            - float(base_second_post_risk["pair"]),
            6,
        ),
        "second_repair_post_delta_comp": round(
            float(second_post_risk["composite"])
            - float(first_only_second_post_risk["composite"]),
            6,
        ),
        "second_repair_post_delta_pair": round(
            float(second_post_risk["pair"])
            - float(first_only_second_post_risk["pair"]),
            6,
        ),
        "second_base_token_id": (
            int(second_base_token_id) if second_base_token_id is not None else None
        ),
        "second_repair_token_id": (
            int(second_repair_token_id) if second_repair_token_id is not None else None
        ),
        "second_target_token_id": int(second_target_token_id),
        "second_pre_comp": round(float(second_pre_risk["composite"]), 6),
        "second_post_comp": round(float(second_post_risk["composite"]), 6),
        "second_delta_comp": round(
            float(second_post_risk["composite"]) - float(second_pre_risk["composite"]),
            6,
        ),
        "second_pre_pair": round(float(second_pre_risk["pair"]), 6),
        "second_post_pair": round(float(second_post_risk["pair"]), 6),
        "second_delta_pair": round(
            float(second_post_risk["pair"]) - float(second_pre_risk["pair"]), 6
        ),
        "second_pre_active": int(bool(second_pre_risk["active"])),
        "second_post_active": int(bool(second_post_risk["active"])),
        "second_base_wrong": int(not second_base_correct),
        "second_repair_correct": int(second_repair_correct),
        "second_fix": int(not second_base_correct and second_repair_correct),
        "second_damage": int(second_base_correct and (not second_repair_correct)),
        "second_changed": int(second_changed),
        "base_branch_token_ids_to_second_pre": _token_id_list(
            base_branch_token_ids_to_second_pre
        ),
        "base_branch_token_ids_to_second_post": _token_id_list(
            base_branch_token_ids_to_second_post
        ),
        "first_only_branch_token_ids_to_second_pre": _token_id_list(
            first_only_branch_token_ids_to_second_pre
        ),
        "first_only_branch_token_ids_to_second_post": _token_id_list(
            first_only_branch_token_ids_to_second_post
        ),
        "first_plus_second_branch_token_ids_to_second_post": _token_id_list(
            first_plus_second_branch_token_ids_to_second_post
        ),
        "route_turnover_k": int(route_turnover_k),
        "base_vs_first_top1_route_changed": int(base_vs_first["top1_changed"]),
        "base_vs_first_turnover_at_k": round(float(base_vs_first["turnover_at_k"]), 6),
        "base_vs_first_cosine_distance": round(
            float(base_vs_first["cosine_distance"]), 6
        ),
        "base_vs_first_vector_dim": int(base_vs_first["vector_dim"]),
        "base_vs_first_left_top1_route": base_vs_first["left_top1"],
        "base_vs_first_right_top1_route": base_vs_first["right_top1"],
        "base_vs_first_left_topk_routes": base_vs_first["left_topk"],
        "base_vs_first_right_topk_routes": base_vs_first["right_topk"],
        "first_vs_second_top1_route_changed": int(first_vs_second["top1_changed"]),
        "first_vs_second_turnover_at_k": round(
            float(first_vs_second["turnover_at_k"]), 6
        ),
        "first_vs_second_cosine_distance": round(
            float(first_vs_second["cosine_distance"]), 6
        ),
        "first_vs_second_vector_dim": int(first_vs_second["vector_dim"]),
        "first_vs_second_left_top1_route": first_vs_second["left_top1"],
        "first_vs_second_right_top1_route": first_vs_second["right_top1"],
        "first_vs_second_left_topk_routes": first_vs_second["left_topk"],
        "first_vs_second_right_topk_routes": first_vs_second["right_topk"],
    }


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = int(stats.get("count", 0))
    out = {"count": count}
    if count <= 0:
        return {
            **out,
            "mean_gap": 0.0,
            "first_delta_comp": 0.0,
            "first_delta_pair": 0.0,
            "base_second_pre_comp": 0.0,
            "base_second_pre_pair": 0.0,
            "base_second_pre_active": 0.0,
            "base_second_post_comp": 0.0,
            "base_second_post_pair": 0.0,
            "base_second_post_active": 0.0,
            "first_only_second_post_comp": 0.0,
            "first_only_second_post_pair": 0.0,
            "first_only_second_post_active": 0.0,
            "first_writeback_delta_comp": 0.0,
            "first_writeback_delta_pair": 0.0,
            "first_writeback_post_delta_comp": 0.0,
            "first_writeback_post_delta_pair": 0.0,
            "second_repair_post_delta_comp": 0.0,
            "second_repair_post_delta_pair": 0.0,
            "second_delta_comp": 0.0,
            "second_delta_pair": 0.0,
            "second_fix_rate": 0.0,
            "second_damage_rate": 0.0,
            "second_changed_rate": 0.0,
            "base_vs_first_top1_change_rate": 0.0,
            "base_vs_first_turnover_at_k": 0.0,
            "base_vs_first_cosine_distance": 0.0,
            "base_vs_first_vector_dim": 0.0,
            "first_vs_second_top1_change_rate": 0.0,
            "first_vs_second_turnover_at_k": 0.0,
            "first_vs_second_cosine_distance": 0.0,
            "first_vs_second_vector_dim": 0.0,
        }
    out["mean_gap"] = round(float(stats["sum_gap"]) / count, 6)
    for prefix in ("first", "second"):
        for key in (
            "pre_comp",
            "post_comp",
            "delta_comp",
            "pre_pair",
            "post_pair",
            "delta_pair",
        ):
            out[f"{prefix}_{key}"] = round(float(stats[f"{prefix}_{key}"]) / count, 6)
        out[f"{prefix}_pre_active"] = _safe_rate(stats[f"{prefix}_pre_active"], count)
        out[f"{prefix}_post_active"] = _safe_rate(stats[f"{prefix}_post_active"], count)
    out["base_second_pre_comp"] = round(float(stats["base_second_pre_comp"]) / count, 6)
    out["base_second_pre_pair"] = round(float(stats["base_second_pre_pair"]) / count, 6)
    out["base_second_pre_active"] = _safe_rate(stats["base_second_pre_active"], count)
    out["base_second_post_comp"] = round(
        float(stats["base_second_post_comp"]) / count, 6
    )
    out["base_second_post_pair"] = round(
        float(stats["base_second_post_pair"]) / count, 6
    )
    out["base_second_post_active"] = _safe_rate(stats["base_second_post_active"], count)
    out["first_only_second_post_comp"] = round(
        float(stats["first_only_second_post_comp"]) / count, 6
    )
    out["first_only_second_post_pair"] = round(
        float(stats["first_only_second_post_pair"]) / count, 6
    )
    out["first_only_second_post_active"] = _safe_rate(
        stats["first_only_second_post_active"], count
    )
    out["first_writeback_delta_comp"] = round(
        float(stats["first_writeback_delta_comp"]) / count, 6
    )
    out["first_writeback_delta_pair"] = round(
        float(stats["first_writeback_delta_pair"]) / count, 6
    )
    out["first_writeback_post_delta_comp"] = round(
        float(stats["first_writeback_post_delta_comp"]) / count, 6
    )
    out["first_writeback_post_delta_pair"] = round(
        float(stats["first_writeback_post_delta_pair"]) / count, 6
    )
    out["second_repair_post_delta_comp"] = round(
        float(stats["second_repair_post_delta_comp"]) / count, 6
    )
    out["second_repair_post_delta_pair"] = round(
        float(stats["second_repair_post_delta_pair"]) / count, 6
    )
    out["second_base_error_rate"] = _safe_rate(stats["second_base_wrong"], count)
    out["second_repair_acc"] = _safe_rate(stats["second_repair_correct"], count)
    out["second_fix_rate"] = _safe_rate(stats["second_fix"], count)
    out["second_damage_rate"] = _safe_rate(stats["second_damage"], count)
    out["second_changed_rate"] = _safe_rate(stats["second_changed"], count)
    out["second_fix_count"] = int(stats["second_fix"])
    out["second_damage_count"] = int(stats["second_damage"])
    out["second_changed_count"] = int(stats["second_changed"])
    for prefix in ("base_vs_first", "first_vs_second"):
        compared = int(stats.get(f"{prefix}_route_compared", 0))
        out[f"{prefix}_route_compared"] = compared
        out[f"{prefix}_top1_change_rate"] = _safe_rate(
            stats.get(f"{prefix}_top1_changed", 0), compared
        )
        out[f"{prefix}_turnover_at_k"] = _safe_rate(
            float(stats.get(f"{prefix}_turnover_at_k", 0.0)), compared
        )
        out[f"{prefix}_cosine_distance"] = _safe_rate(
            float(stats.get(f"{prefix}_cosine_distance", 0.0)), compared
        )
        out[f"{prefix}_vector_dim"] = _safe_rate(
            float(stats.get(f"{prefix}_vector_dim", 0.0)), compared
        )
    return out


def _batch_extract_features(
    *, evaluator: OfflineEvaluator, items: list[dict[str, Any]], max_context_tokens: int
) -> list[dict[str, Any]]:
    return _extract_runtime_features_from_context_ids_batch(
        evaluator=evaluator,
        samples=[item["sample"] for item in items],
        prefixes=[str(item["prefix"]) for item in items],
        context_id_rows=[
            [int(token_id) for token_id in item["context_ids"]] for item in items
        ],
        max_context_tokens=max_context_tokens,
    )


def _choose_second_identifier_step(
    *,
    identifier_steps: list[int],
    initial_step: int,
    generation_steps: int,
    future_horizon: int,
    anchor_search_window: int,
    rng: random.Random,
) -> tuple[int | None, dict[str, Any]]:
    max_future_step = min(
        int(generation_steps) - 1, int(initial_step) + max(1, int(future_horizon))
    )
    if max_future_step <= int(initial_step):
        return (None, {"reason": "no_future_window"})
    future_identifiers = [
        int(step)
        for step in identifier_steps
        if int(initial_step) < int(step) <= max_future_step
    ]
    if not future_identifiers:
        return (None, {"reason": "no_future_identifier_in_horizon"})
    anchor = int(initial_step) + rng.randint(1, max_future_step - int(initial_step))
    window = max(0, int(anchor_search_window))
    lo = max(int(initial_step) + 1, anchor - window)
    hi = min(max_future_step, anchor + window)
    candidates = [step for step in future_identifiers if lo <= step <= hi]
    if not candidates:
        return (
            None,
            {
                "reason": "no_identifier_near_anchor",
                "anchor": anchor,
                "lo": lo,
                "hi": hi,
            },
        )
    chosen = min(candidates, key=lambda step: (abs(step - anchor), step))
    return (
        chosen,
        {
            "reason": "selected",
            "anchor": anchor,
            "lo": lo,
            "hi": hi,
            "candidate_count": len(candidates),
            "gap": int(chosen) - int(initial_step) - 1,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--schema-manifest")
    parser.add_argument("--base-model")
    parser.add_argument("--neuron-risk-config")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--sample-shard-index", type=int, default=0)
    parser.add_argument("--sample-shard-count", type=int, default=1)
    parser.add_argument("--prefilter-oversample-factor", type=int, default=8)
    parser.add_argument("--sample-selection", choices=("first", "even"), default="even")
    parser.add_argument("--generation-steps", type=int, default=384)
    parser.add_argument(
        "--future-horizon",
        type=int,
        default=128,
        help="Maximum number of future tokens considered after the first repair.",
    )
    parser.add_argument(
        "--anchor-search-window",
        type=int,
        default=8,
        help="Uniformly sample a future anchor inside --future-horizon, then use the nearest reference identifier within +/- this many tokens.",
    )
    parser.add_argument(
        "--route-turnover-k",
        type=int,
        default=5,
        help="Top-k route set size used for route-pattern turnover diagnostics.",
    )
    parser.add_argument("--target-window-chars", type=int, default=4096)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--repair-batch-size", type=int, default=64)
    parser.add_argument("--generation-length-bucket-quantum", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--no-save-events",
        action="store_true",
        help="Do not store per-attempt raw event records in the output JSON.",
    )
    args = parser.parse_args()
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    base_model_name = args.base_model or str(checkpoint.get("base_model") or "")
    if not base_model_name:
        raise ValueError("--base-model is required when checkpoint lacks base_model")
    if args.neuron_risk_config:
        os.environ["SRLCC_NEURON_RISK_CONFIG"] = str(args.neuron_risk_config)
    from transformers import AutoTokenizer
    from ramp_hsr.tokenizer_context import (
        assert_tokenizer_context_policy,
        configure_tokenizer_context_policy,
        tokenizer_context_policy_metadata,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=False, local_files_only=True
    )
    configure_tokenizer_context_policy(tokenizer)
    assert_tokenizer_context_policy(tokenizer, context="two-repair gap risk")
    shard_count = max(1, int(args.sample_shard_count))
    shard_index = int(args.sample_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"--sample-shard-index must satisfy 0 <= index < count, got {shard_index}/{shard_count}"
        )
    per_shard_max = 0
    if int(args.max_samples) > 0:
        per_shard_max = max(1, math.ceil(int(args.max_samples) / shard_count))
    collect_max = per_shard_max
    if collect_max > 0 and str(args.sample_selection) != "even":
        collect_max *= max(1, int(args.prefilter_oversample_factor))
    (samples, input_metadata) = _collect_samples(
        args.samples_file,
        tokenizer=tokenizer,
        steps=int(args.generation_steps),
        target_window_chars=int(args.target_window_chars),
        max_samples=collect_max,
        sample_selection=str(args.sample_selection),
        sample_shard_index=int(args.sample_shard_index),
        sample_num_shards=int(args.sample_shard_count),
    )
    raw_identifier_steps_by_sample = [
        _reference_identifier_step_list(sample, int(args.generation_steps))
        for sample in samples
    ]
    rng = random.Random(int(args.seed))
    selected_second_steps_by_index: dict[int, int] = {}
    selection_stats: Counter[str] = Counter()
    selection_debug: Counter[str] = Counter()
    for index, identifier_steps in enumerate(raw_identifier_steps_by_sample):
        if not identifier_steps or int(identifier_steps[0]) != 0:
            selection_stats["skip_initial_not_identifier"] += 1
            continue
        (second_step, meta) = _choose_second_identifier_step(
            identifier_steps=identifier_steps,
            initial_step=0,
            generation_steps=int(args.generation_steps),
            future_horizon=int(args.future_horizon),
            anchor_search_window=int(args.anchor_search_window),
            rng=rng,
        )
        selection_stats[str(meta.get("reason") or "unknown")] += 1
        if second_step is None:
            continue
        selected_second_steps_by_index[index] = int(second_step)
        selection_debug[f"gap_bucket_{_gap_bucket(int(meta.get('gap', 0)))}"] += 1
    eligible_indices = sorted(selected_second_steps_by_index)
    if int(args.max_samples) > 0 and len(eligible_indices) > int(args.max_samples):
        selected_positions = _take_evenly(
            list(range(len(eligible_indices))), int(args.max_samples)
        )
        eligible_indices = [eligible_indices[pos] for pos in selected_positions]
    samples = [samples[index] for index in eligible_indices]
    second_steps = [selected_second_steps_by_index[index] for index in eligible_indices]
    if per_shard_max > 0 and len(samples) > per_shard_max:
        selected_positions = _take_evenly(list(range(len(samples))), per_shard_max)
        samples = [samples[pos] for pos in selected_positions]
        second_steps = [second_steps[pos] for pos in selected_positions]
    shard_stats = {
        "sample_shard_index": shard_index,
        "sample_shard_count": shard_count,
        "pre_shard_count": len(eligible_indices),
        "post_shard_count": len(samples),
        "collector_applied_shard": True,
        "per_shard_max_samples": per_shard_max,
    }
    timings["collect_samples_sec"] = round(time.perf_counter() - t_start, 3)
    (repair_moe, repair_checkpoint) = load_editor(args.checkpoint, args.device)
    hidden_state_interface = str(
        repair_checkpoint.get("hidden_state_interface", "post_norm")
    )
    (route_schema, code_schema) = _resolve_repair_feature_schema(
        checkpoint=repair_checkpoint, manifest_path=args.schema_manifest
    )
    evaluator = OfflineEvaluator(
        long_sequence_rollout_model_name=base_model_name,
        long_sequence_rollout_device=args.device,
        long_sequence_rollout_batch_size=max(1, int(args.rollout_batch_size)),
    )
    (llm, _) = evaluator._get_long_sequence_rollout_llm(samples)
    (norm, lm_head) = _resolve_norm_and_lm_head(llm)
    initial_context_ids_by_sample: list[list[int]] = []
    prefixes: list[str] = []
    for sample in samples:
        prefix = str(sample["_rollout_prefix"])
        prefixes.append(prefix)
        context_ids = _encode_text(tokenizer, prefix)
        if not context_ids:
            context_ids = _fallback_context_ids(tokenizer)
        initial_context_ids_by_sample.append(context_ids)
    initial_generation_lengths = [0 for _ in samples]
    timings["initial_base_generation_sec"] = 0.0
    initial_items: list[dict[str, Any]] = []
    for sample, initial_context_ids in zip(
        samples, initial_context_ids_by_sample, strict=True
    ):
        initial_items.append(
            {
                "sample": sample,
                "prefix": str(sample["_rollout_prefix"]),
                "generated_prefix": "",
                "context_ids": list(initial_context_ids),
            }
        )
    initial_success_records: list[dict[str, Any]] = []
    initial_raw = Counter()
    t0 = time.perf_counter()
    for start in tqdm(
        range(0, len(initial_items), max(1, int(args.repair_batch_size))),
        desc="Initial repair",
        unit="batch",
        dynamic_ncols=True,
    ):
        chunk_items = initial_items[start : start + max(1, int(args.repair_batch_size))]
        (first_pre_features, predictions) = _extract_and_predict(
            evaluator=evaluator,
            repair_moe=repair_moe,
            norm=norm,
            lm_head=lm_head,
            tokenizer=tokenizer,
            device=args.device,
            route_schema=route_schema,
            code_schema=code_schema,
            hidden_state_interface=hidden_state_interface,
            items=chunk_items,
            max_context_tokens=int(args.max_context_tokens),
        )
        for offset, (first_pre_feature, prediction) in enumerate(
            zip(first_pre_features, predictions, strict=True)
        ):
            sample_index = start + offset
            sample = samples[sample_index]
            initial_step = 0
            target_id = int(sample["_rollout_target_token_ids"][initial_step])
            base_token_id = _base_prediction_token_id(first_pre_feature)
            repair_raw = prediction.get("prediction_token_id")
            repair_token_id = int(repair_raw) if repair_raw is not None else None
            base_correct = base_token_id == target_id
            repair_correct = repair_token_id == target_id
            initial_raw["candidate_count"] += 1
            initial_raw["base_wrong"] += int(not base_correct)
            initial_raw["repair_correct"] += int(repair_correct)
            initial_raw["initial_success"] += int(not base_correct and repair_correct)
            if base_correct or not repair_correct:
                continue
            (token_id, token_text, next_context_ids) = _prediction_token_piece(
                tokenizer=tokenizer,
                context_ids=list(chunk_items[offset]["context_ids"]),
                prediction=prediction,
            )
            if token_id is None:
                continue
            initial_success_records.append(
                {
                    "sample_index": sample_index,
                    "initial_step": initial_step,
                    "second_step": int(second_steps[sample_index]),
                    "first_pre_feature": first_pre_feature,
                    "first_base_token_id": (
                        int(base_token_id) if base_token_id is not None else None
                    ),
                    "first_target_token_id": int(target_id),
                    "first_repair_token_id": int(token_id),
                    "first_repair_piece": token_text,
                    "first_post_context_ids": next_context_ids,
                    "first_post_prefix": str(chunk_items[offset]["prefix"])
                    + token_text,
                }
            )
    timings["initial_repair_sec"] = round(time.perf_counter() - t0, 3)
    if not initial_success_records:
        report = {
            "enabled": True,
            "processed_samples": len(samples),
            "initial": dict(initial_raw),
            "second_attempt_count": 0,
            "summary": _finalize_stats(_empty_stats()),
            "by_gap_bucket": {
                bucket: _finalize_stats(_empty_stats()) for bucket in GAP_BUCKET_ORDER
            },
            "by_gap_exact": {},
            "events": [] if not args.no_save_events else None,
            "timings_sec": timings,
            "input": {
                **input_metadata,
                "collect_max_samples": collect_max,
                "eligible_after_identifier_filter": len(eligible_indices),
                "sample_shard": shard_stats,
                "second_selection": {
                    "future_horizon": int(args.future_horizon),
                    "anchor_search_window": int(args.anchor_search_window),
                    "stats": dict(selection_stats),
                    "gap_buckets": dict(selection_debug),
                },
            },
            "config": _jsonable(vars(args)),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.no_save_events:
            events_jsonl = args.output.with_name(f"{args.output.stem}.events.jsonl")
            events_jsonl.write_text("", encoding="utf-8")
            report["events_jsonl"] = str(events_jsonl)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    t0 = time.perf_counter()
    first_post_items = [
        {
            "sample": samples[int(record["sample_index"])],
            "prefix": str(record["first_post_prefix"]),
            "context_ids": list(record["first_post_context_ids"]),
        }
        for record in initial_success_records
    ]
    first_post_features = _batch_extract_features(
        evaluator=evaluator,
        items=first_post_items,
        max_context_tokens=int(args.max_context_tokens),
    )
    for record, feature in zip(
        initial_success_records, first_post_features, strict=True
    ):
        record["first_post_feature"] = feature
    timings["first_post_feature_sec"] = round(time.perf_counter() - t0, 3)
    base_continuation_lengths = [
        max(0, int(record["second_step"]) - int(record["initial_step"]))
        for record in initial_success_records
    ]
    t0 = time.perf_counter()
    base_continuation_ids = _greedy_generate_token_ids_from_context_batch_by_length(
        llm,
        [
            list(initial_context_ids_by_sample[int(record["sample_index"])])
            for record in initial_success_records
        ],
        max_new_tokens_by_row=base_continuation_lengths,
        max_length=int(args.max_context_tokens),
        batch_size=int(args.rollout_batch_size),
        bucket_quantum=int(args.generation_length_bucket_quantum),
    )
    timings["base_no_first_continuation_generation_sec"] = round(
        time.perf_counter() - t0, 3
    )
    base_second_pre_items: list[dict[str, Any]] = []
    for record, generated_ids in zip(
        initial_success_records, base_continuation_ids, strict=True
    ):
        sample = samples[int(record["sample_index"])]
        current_context = list(
            initial_context_ids_by_sample[int(record["sample_index"])]
        )
        record["base_branch_token_ids_to_second_pre"] = _token_id_list(generated_ids)
        pieces: list[str] = []
        for token_id in generated_ids:
            (piece, current_context) = _append_token_text(
                tokenizer,
                current_context,
                int(token_id),
                fallback_text=_decode_single_token(tokenizer, int(token_id)),
            )
            pieces.append(piece)
        record["base_second_pre_context_ids"] = current_context
        record["base_second_pre_prefix"] = str(sample["_rollout_prefix"]) + "".join(
            pieces
        )
        base_second_pre_items.append(
            {
                "sample": sample,
                "prefix": str(record["base_second_pre_prefix"]),
                "generated_prefix": str(record["base_second_pre_prefix"])[
                    len(str(sample["_rollout_prefix"])) :
                ],
                "context_ids": list(current_context),
            }
        )
    t0 = time.perf_counter()
    base_second_pre_features = _batch_extract_features(
        evaluator=evaluator,
        items=base_second_pre_items,
        max_context_tokens=int(args.max_context_tokens),
    )
    timings["base_second_pre_feature_sec"] = round(time.perf_counter() - t0, 3)
    base_second_post_items: list[dict[str, Any]] = []
    for record, pre_feature in zip(
        initial_success_records, base_second_pre_features, strict=True
    ):
        sample = samples[int(record["sample_index"])]
        base_token_id = _base_prediction_token_id(pre_feature)
        (token_text, next_context_ids) = _append_token_text(
            tokenizer,
            list(record["base_second_pre_context_ids"]),
            base_token_id,
            fallback_text=(
                _decode_single_token(tokenizer, int(base_token_id))
                if base_token_id is not None
                else ""
            ),
        )
        base_post_ids = list(record.get("base_branch_token_ids_to_second_pre") or [])
        if base_token_id is not None:
            base_post_ids.append(int(base_token_id))
        record["base_branch_token_ids_to_second_post"] = base_post_ids
        record["base_second_post_context_ids"] = next_context_ids
        record["base_second_post_prefix"] = (
            str(record["base_second_pre_prefix"]) + token_text
        )
        base_second_post_items.append(
            {
                "sample": sample,
                "prefix": str(record["base_second_post_prefix"]),
                "generated_prefix": str(record["base_second_post_prefix"])[
                    len(str(sample["_rollout_prefix"])) :
                ],
                "context_ids": list(next_context_ids),
            }
        )
    t0 = time.perf_counter()
    base_second_post_features = _batch_extract_features(
        evaluator=evaluator,
        items=base_second_post_items,
        max_context_tokens=int(args.max_context_tokens),
    )
    timings["base_second_post_feature_sec"] = round(time.perf_counter() - t0, 3)
    continuation_lengths = [
        max(0, int(record["second_step"]) - int(record["initial_step"]) - 1)
        for record in initial_success_records
    ]
    t0 = time.perf_counter()
    continuation_ids = _greedy_generate_token_ids_from_context_batch_by_length(
        llm,
        [list(record["first_post_context_ids"]) for record in initial_success_records],
        max_new_tokens_by_row=continuation_lengths,
        max_length=int(args.max_context_tokens),
        batch_size=int(args.rollout_batch_size),
        bucket_quantum=int(args.generation_length_bucket_quantum),
    )
    timings["committed_continuation_generation_sec"] = round(
        time.perf_counter() - t0, 3
    )
    second_items: list[dict[str, Any]] = []
    for record, generated_ids in zip(
        initial_success_records, continuation_ids, strict=True
    ):
        current_context = list(record["first_post_context_ids"])
        first_only_pre_ids = [int(record["first_repair_token_id"])]
        first_only_pre_ids.extend(_token_id_list(generated_ids))
        record["first_only_branch_token_ids_to_second_pre"] = first_only_pre_ids
        pieces: list[str] = []
        for token_id in generated_ids:
            (piece, current_context) = _append_token_text(
                tokenizer,
                current_context,
                int(token_id),
                fallback_text=_decode_single_token(tokenizer, int(token_id)),
            )
            pieces.append(piece)
        record["second_pre_context_ids"] = current_context
        record["second_pre_prefix"] = str(record["first_post_prefix"]) + "".join(pieces)
        second_items.append(
            {
                "sample": samples[int(record["sample_index"])],
                "prefix": str(record["second_pre_prefix"]),
                "generated_prefix": str(record["second_pre_prefix"])[
                    len(str(samples[int(record["sample_index"])]["_rollout_prefix"])) :
                ],
                "context_ids": list(current_context),
            }
        )
    second_pre_features: list[dict[str, Any]] = []
    second_predictions: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for start in tqdm(
        range(0, len(second_items), max(1, int(args.repair_batch_size))),
        desc="Second repair",
        unit="batch",
        dynamic_ncols=True,
    ):
        chunk = second_items[start : start + max(1, int(args.repair_batch_size))]
        (features, predictions) = _extract_and_predict(
            evaluator=evaluator,
            repair_moe=repair_moe,
            norm=norm,
            lm_head=lm_head,
            tokenizer=tokenizer,
            device=args.device,
            route_schema=route_schema,
            code_schema=code_schema,
            hidden_state_interface=hidden_state_interface,
            items=chunk,
            max_context_tokens=int(args.max_context_tokens),
        )
        second_pre_features.extend(features)
        second_predictions.extend(predictions)
    timings["second_repair_sec"] = round(time.perf_counter() - t0, 3)
    first_only_second_post_items: list[dict[str, Any]] = []
    for record, pre_feature in zip(
        initial_success_records, second_pre_features, strict=True
    ):
        sample = samples[int(record["sample_index"])]
        base_token_id = _base_prediction_token_id(pre_feature)
        (token_text, next_context_ids) = _append_token_text(
            tokenizer,
            list(record["second_pre_context_ids"]),
            base_token_id,
            fallback_text=(
                _decode_single_token(tokenizer, int(base_token_id))
                if base_token_id is not None
                else ""
            ),
        )
        first_only_post_ids = list(
            record.get("first_only_branch_token_ids_to_second_pre") or []
        )
        if base_token_id is not None:
            first_only_post_ids.append(int(base_token_id))
        record["first_only_branch_token_ids_to_second_post"] = first_only_post_ids
        record["first_only_second_post_context_ids"] = next_context_ids
        record["first_only_second_post_prefix"] = (
            str(record["second_pre_prefix"]) + token_text
        )
        first_only_second_post_items.append(
            {
                "sample": sample,
                "prefix": str(record["first_only_second_post_prefix"]),
                "generated_prefix": str(record["first_only_second_post_prefix"])[
                    len(str(sample["_rollout_prefix"])) :
                ],
                "context_ids": list(next_context_ids),
            }
        )
    t0 = time.perf_counter()
    first_only_second_post_features = _batch_extract_features(
        evaluator=evaluator,
        items=first_only_second_post_items,
        max_context_tokens=int(args.max_context_tokens),
    )
    timings["first_only_second_post_feature_sec"] = round(time.perf_counter() - t0, 3)
    second_post_items: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    second_repair_token_ids: list[int] = []
    for index, (record, prediction) in enumerate(
        zip(initial_success_records, second_predictions, strict=True)
    ):
        (token_id, token_text, next_context_ids) = _prediction_token_piece(
            tokenizer=tokenizer,
            context_ids=list(record["second_pre_context_ids"]),
            prediction=prediction,
        )
        if token_id is None:
            continue
        branch_post_ids = list(
            record.get("first_only_branch_token_ids_to_second_pre") or []
        )
        branch_post_ids.append(int(token_id))
        record["first_plus_second_branch_token_ids_to_second_post"] = branch_post_ids
        kept_indices.append(index)
        second_repair_token_ids.append(int(token_id))
        second_post_items.append(
            {
                "sample": samples[int(record["sample_index"])],
                "prefix": str(record["second_pre_prefix"]) + token_text,
                "context_ids": next_context_ids,
            }
        )
    t0 = time.perf_counter()
    second_post_features = _batch_extract_features(
        evaluator=evaluator,
        items=second_post_items,
        max_context_tokens=int(args.max_context_tokens),
    )
    timings["second_post_feature_sec"] = round(time.perf_counter() - t0, 3)
    summary = _empty_stats()
    by_gap_bucket = {bucket: _empty_stats() for bucket in GAP_BUCKET_ORDER}
    by_gap_bucket["unknown"] = _empty_stats()
    by_gap_exact: defaultdict[int, dict[str, Any]] = defaultdict(_empty_stats)
    events: list[dict[str, Any]] = []
    for out_index, source_index in enumerate(kept_indices):
        record = initial_success_records[source_index]
        sample = samples[int(record["sample_index"])]
        second_step = int(record["second_step"])
        second_target_id = int(sample["_rollout_target_token_ids"][second_step])
        second_base_token_id = _base_prediction_token_id(
            second_pre_features[source_index]
        )
        second_repair_token_id = int(second_repair_token_ids[out_index])
        gap = max(0, second_step - int(record["initial_step"]) - 1)
        kwargs = {
            "gap": gap,
            "first_pre": record["first_pre_feature"],
            "first_post": record["first_post_feature"],
            "base_second_pre": base_second_pre_features[source_index],
            "base_second_post": base_second_post_features[source_index],
            "first_only_second_post": first_only_second_post_features[source_index],
            "second_pre": second_pre_features[source_index],
            "second_post": second_post_features[out_index],
            "second_base_token_id": second_base_token_id,
            "second_repair_token_id": second_repair_token_id,
            "second_target_token_id": second_target_id,
            "route_turnover_k": int(args.route_turnover_k),
        }
        _update_stats(summary, **kwargs)
        _update_stats(by_gap_bucket[_gap_bucket(gap)], **kwargs)
        _update_stats(by_gap_exact[int(gap)], **kwargs)
        if not args.no_save_events:
            events.append(
                _event_record(
                    sample_index=int(record["sample_index"]),
                    initial_step=int(record["initial_step"]),
                    second_step=second_step,
                    first_pre=record["first_pre_feature"],
                    first_post=record["first_post_feature"],
                    base_second_pre=base_second_pre_features[source_index],
                    base_second_post=base_second_post_features[source_index],
                    first_only_second_post=first_only_second_post_features[
                        source_index
                    ],
                    second_pre=second_pre_features[source_index],
                    second_post=second_post_features[out_index],
                    first_base_token_id=record.get("first_base_token_id"),
                    first_repair_token_id=record.get("first_repair_token_id"),
                    first_target_token_id=int(record["first_target_token_id"]),
                    second_base_token_id=second_base_token_id,
                    second_repair_token_id=second_repair_token_id,
                    second_target_token_id=second_target_id,
                    base_branch_token_ids_to_second_pre=record.get(
                        "base_branch_token_ids_to_second_pre"
                    ),
                    base_branch_token_ids_to_second_post=record.get(
                        "base_branch_token_ids_to_second_post"
                    ),
                    first_only_branch_token_ids_to_second_pre=record.get(
                        "first_only_branch_token_ids_to_second_pre"
                    ),
                    first_only_branch_token_ids_to_second_post=record.get(
                        "first_only_branch_token_ids_to_second_post"
                    ),
                    first_plus_second_branch_token_ids_to_second_post=record.get(
                        "first_plus_second_branch_token_ids_to_second_post"
                    ),
                    route_turnover_k=int(args.route_turnover_k),
                )
            )
    finalized_by_gap = {
        bucket: _finalize_stats(stats)
        for (bucket, stats) in by_gap_bucket.items()
        if int(stats.get("count", 0)) > 0 or bucket in GAP_BUCKET_ORDER
    }
    finalized_by_gap_exact = {
        str(gap): _finalize_stats(stats)
        for (gap, stats) in sorted(by_gap_exact.items())
    }
    report = {
        "enabled": True,
        "mode": "two_repair_random_future_identifier_gap_risk",
        "model_name": base_model_name,
        "processed_samples": len(samples),
        "initial": {
            **dict(initial_raw),
            "initial_success_rate_on_candidates": _safe_rate(
                initial_raw["initial_success"], initial_raw["candidate_count"]
            ),
            "initial_success_rate_on_base_wrong": _safe_rate(
                initial_raw["initial_success"], initial_raw["base_wrong"]
            ),
        },
        "second_attempt_count": len(kept_indices),
        "summary": _finalize_stats(summary),
        "by_gap_bucket": finalized_by_gap,
        "by_gap_exact": finalized_by_gap_exact,
        "events": events if not args.no_save_events else None,
        "base_generation_length_summary": _length_summary(
            initial_generation_lengths,
            bucket_quantum=int(args.generation_length_bucket_quantum),
        ),
        "continuation_length_summary": _length_summary(
            continuation_lengths,
            bucket_quantum=int(args.generation_length_bucket_quantum),
        ),
        "input": {
            **input_metadata,
            "collect_max_samples": collect_max,
            "eligible_after_identifier_filter": len(eligible_indices),
            "sample_shard": shard_stats,
            "second_selection": {
                "future_horizon": int(args.future_horizon),
                "anchor_search_window": int(args.anchor_search_window),
                "stats": dict(selection_stats),
                "gap_buckets": dict(selection_debug),
            },
        },
        "config": {
            **_jsonable(vars(args)),
            "tokenizer_context_policy": tokenizer_context_policy_metadata(tokenizer),
        },
        "timings_sec": {**timings, "total": round(time.perf_counter() - t_start, 3)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_save_events:
        events_jsonl = args.output.with_name(f"{args.output.stem}.events.jsonl")
        with open(events_jsonl, "w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        report["events_jsonl"] = str(events_jsonl)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "processed_samples": report["processed_samples"],
                "initial_success": report["initial"]["initial_success"],
                "second_attempt_count": report["second_attempt_count"],
                "summary": report["summary"],
                "timings_sec": report["timings_sec"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

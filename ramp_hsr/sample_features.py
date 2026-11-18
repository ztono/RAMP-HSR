from __future__ import annotations
import math
from typing import Any
from ramp_hsr.routes import (
    load_neuron_risk_policy,
    neuron_chain_feature_keys,
    neuron_pair_feature_keys,
)

RISK_ACTIVE_COMPOSITE_THRESHOLD = 1.0
RISK_ACTIVE_PAIR_THRESHOLD = 0.7


def _float_feature(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _top_two_scores(
    scores: list[tuple[str, float]]
) -> tuple[tuple[str, float] | None, float]:
    if not scores:
        return (None, 0.0)
    ordered = sorted(scores, key=lambda item: item[1], reverse=True)
    dominant = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    return (dominant, second_score)


def _risk_route_profile_features(uncertainty: dict[str, Any]) -> dict[str, float]:
    policy = load_neuron_risk_policy()
    composite_risk = _float_feature(uncertainty.get("neuron_chain_composite_risk"))
    pair_risk = _float_feature(uncertainty.get("neuron_chain_pair_risk"))
    max_risk = max(composite_risk, pair_risk)
    profile: dict[str, float] = {
        "risk_route_max_risk": max_risk,
        "risk_route_dominant_chain_score": 0.0,
        "risk_route_second_chain_score": 0.0,
        "risk_route_chain_score_margin": 0.0,
        "risk_route_active_chain_count": 0.0,
        "risk_route_dominant_pair_score": 0.0,
        "risk_route_second_pair_score": 0.0,
        "risk_route_pair_score_margin": 0.0,
        "risk_route_active_pair_count": 0.0,
    }
    if policy is None:
        return profile
    chain_scores: list[tuple[str, float]] = []
    chain_active_count = 0
    for chain in policy.chains:
        keys = neuron_chain_feature_keys(chain.name)
        score = _float_feature(uncertainty.get(keys["score"]))
        chain_scores.append((str(chain.name), score))
        chain_active_count += int(bool(_float_feature(uncertainty.get(keys["active"]))))
    pair_scores: list[tuple[str, float]] = []
    pair_active_count = 0
    for pair in policy.pairs:
        keys = neuron_pair_feature_keys(pair.name)
        score = _float_feature(uncertainty.get(keys["score"]))
        pair_scores.append((str(pair.name), score))
        pair_active_count += int(bool(_float_feature(uncertainty.get(keys["active"]))))
    (dominant_chain, second_chain_score) = _top_two_scores(chain_scores)
    (dominant_pair, second_pair_score) = _top_two_scores(pair_scores)
    dominant_chain_name = (
        dominant_chain[0] if dominant_chain and dominant_chain[1] > 0.0 else ""
    )
    dominant_pair_name = (
        dominant_pair[0] if dominant_pair and dominant_pair[1] > 0.0 else ""
    )
    dominant_chain_score = dominant_chain[1] if dominant_chain_name else 0.0
    dominant_pair_score = dominant_pair[1] if dominant_pair_name else 0.0
    profile.update(
        {
            "risk_route_dominant_chain_score": dominant_chain_score,
            "risk_route_second_chain_score": second_chain_score,
            "risk_route_chain_score_margin": max(
                0.0, dominant_chain_score - second_chain_score
            ),
            "risk_route_active_chain_count": float(chain_active_count),
            "risk_route_dominant_pair_score": dominant_pair_score,
            "risk_route_second_pair_score": second_pair_score,
            "risk_route_pair_score_margin": max(
                0.0, dominant_pair_score - second_pair_score
            ),
            "risk_route_active_pair_count": float(pair_active_count),
        }
    )
    for chain in policy.chains:
        suffix = neuron_chain_feature_keys(chain.name)["score"].removeprefix(
            "neuron_chain_score_"
        )
        profile[f"risk_route_dominant_chain_{suffix}"] = float(
            str(chain.name) == dominant_chain_name
        )
    for pair in policy.pairs:
        suffix = neuron_pair_feature_keys(pair.name)["score"].removeprefix(
            "neuron_chain_pair_score_"
        )
        profile[f"risk_route_dominant_pair_{suffix}"] = float(
            str(pair.name) == dominant_pair_name
        )
    return profile


def _feature_vector(sample: dict[str, Any], fields: tuple[str, ...]) -> list[float]:
    uncertainty = sample.get("uncertainty") or {}
    route_profile = _risk_route_profile_features(uncertainty)
    return [
        _float_feature(
            route_profile[field] if field in route_profile else uncertainty.get(field)
        )
        for field in fields
    ]

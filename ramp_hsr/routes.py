from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEURON_RISK_POLICY_PATH = REPO_ROOT / "configs" / "neuron_risk.active.json"
AGGREGATE_NEURON_RISK_KEYS = (
    "neuron_chain_max_risk",
    "neuron_chain_mean_risk",
    "neuron_chain_active_fraction",
    "neuron_chain_pair_risk",
    "neuron_chain_pair_active_fraction",
    "neuron_chain_composite_risk",
)


@dataclass(frozen=True)
class NeuronRiskNode:
    layer_index: int
    dimension: int
    direction: str


@dataclass(frozen=True)
class NeuronRiskChain:
    name: str
    family: str
    weight: float
    nodes: tuple[NeuronRiskNode, ...]


@dataclass(frozen=True)
class NeuronRiskPair:
    name: str
    left: str
    right: str
    family: str
    weight: float


@dataclass(frozen=True)
class NeuronRiskPolicy:
    name: str
    path: str
    node_z_threshold: float
    chain_active_fraction: float
    chains: tuple[NeuronRiskChain, ...]
    pairs: tuple[NeuronRiskPair, ...]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _feature_slug(name: str) -> str:
    slug = re.sub("[^0-9A-Za-z]+", "_", str(name or "").strip())
    return slug.strip("_").lower()


def neuron_risk_feature_spec(
    policy: NeuronRiskPolicy | None = None,
) -> dict[str, tuple[str, ...]]:
    resolved = policy or load_neuron_risk_policy()
    numeric = list(AGGREGATE_NEURON_RISK_KEYS)
    boolean: list[str] = []
    if resolved is None:
        return {"numeric": tuple(numeric), "bool": tuple(boolean)}
    for chain in resolved.chains:
        suffix = _feature_slug(chain.name)
        numeric.extend(
            [f"neuron_chain_score_{suffix}", f"neuron_chain_active_fraction_{suffix}"]
        )
        boolean.append(f"neuron_chain_active_{suffix}")
    for pair in resolved.pairs:
        suffix = _feature_slug(pair.name)
        numeric.append(f"neuron_chain_pair_score_{suffix}")
        boolean.append(f"neuron_chain_pair_active_{suffix}")
    return {"numeric": tuple(numeric), "bool": tuple(boolean)}


def neuron_chain_feature_keys(chain_name: str) -> dict[str, str]:
    suffix = _feature_slug(chain_name)
    return {
        "score": f"neuron_chain_score_{suffix}",
        "active_fraction": f"neuron_chain_active_fraction_{suffix}",
        "active": f"neuron_chain_active_{suffix}",
    }


def neuron_pair_feature_keys(pair_name: str) -> dict[str, str]:
    suffix = _feature_slug(pair_name)
    return {
        "score": f"neuron_chain_pair_score_{suffix}",
        "active": f"neuron_chain_pair_active_{suffix}",
    }


def neuron_risk_feature_names(
    policy: NeuronRiskPolicy | None = None,
) -> tuple[str, ...]:
    spec = neuron_risk_feature_spec(policy)
    return tuple([*spec["numeric"], *spec["bool"]])


def neuron_risk_default_metrics(
    policy: NeuronRiskPolicy | None = None,
) -> dict[str, float]:
    return {key: 0.0 for key in neuron_risk_feature_names(policy)}


def _parse_node(raw: Any) -> NeuronRiskNode:
    if isinstance(raw, dict):
        return NeuronRiskNode(
            layer_index=int(raw["layer_index"]),
            dimension=int(raw["dimension"]),
            direction=str(raw.get("direction") or "absolute").lower(),
        )
    parts = [part.strip() for part in str(raw).split(":") if part.strip()]
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid neuron risk node '{raw}'")
    return NeuronRiskNode(
        layer_index=int(parts[0]),
        dimension=int(parts[1]),
        direction=(
            {
                "+": "positive",
                "pos": "positive",
                "positive": "positive",
                "-": "negative",
                "neg": "negative",
                "negative": "negative",
                "abs": "absolute",
                "absolute": "absolute",
            }.get(parts[2].lower(), parts[2].lower())
            if len(parts) == 3
            else "absolute"
        ),
    )


def load_neuron_risk_policy(path: str | Path | None = None) -> NeuronRiskPolicy | None:
    env_path = os.getenv("SRLCC_NEURON_RISK_CONFIG")
    chosen_path = Path(path or env_path or DEFAULT_NEURON_RISK_POLICY_PATH)
    return _load_neuron_risk_policy_cached(str(chosen_path))


@lru_cache(maxsize=4)
def _load_neuron_risk_policy_cached(path: str) -> NeuronRiskPolicy | None:
    chosen_path = Path(path)
    if not chosen_path.exists():
        return None
    payload = json.loads(chosen_path.read_text(encoding="utf-8"))
    chains = tuple(
        (
            NeuronRiskChain(
                name=str(item["name"]),
                family=str(item.get("family") or "identifier"),
                weight=_safe_float(item.get("weight"), 1.0),
                nodes=tuple((_parse_node(node) for node in item.get("nodes", []))),
            )
            for item in payload.get("chains", [])
        )
    )
    pairs = tuple(
        (
            NeuronRiskPair(
                name=str(item["name"]),
                left=str(item["left"]),
                right=str(item["right"]),
                family=str(item.get("family") or "identifier"),
                weight=_safe_float(item.get("weight"), 1.0),
            )
            for item in payload.get("pairs", [])
        )
    )
    return NeuronRiskPolicy(
        name=str(payload.get("name") or chosen_path.stem),
        path=str(chosen_path),
        node_z_threshold=_safe_float(payload.get("node_z_threshold"), 1.0),
        chain_active_fraction=_safe_float(payload.get("chain_active_fraction"), 0.5),
        chains=chains,
        pairs=pairs,
    )


def _layer_stats(vector: np.ndarray) -> dict[str, float]:
    signed_mean = float(np.mean(vector))
    signed_std = float(np.std(vector))
    abs_vector = np.abs(vector)
    abs_mean = float(np.mean(abs_vector))
    abs_std = float(np.std(abs_vector))
    return {
        "signed_mean": signed_mean,
        "signed_std": signed_std if signed_std > 1e-06 else 1.0,
        "abs_mean": abs_mean,
        "abs_std": abs_std if abs_std > 1e-06 else 1.0,
    }


def _node_risk_z(vector: np.ndarray, *, dimension: int, direction: str) -> float | None:
    if dimension < 0 or dimension >= vector.shape[0]:
        return None
    stats = _layer_stats(vector)
    value = float(vector[dimension])
    direction_text = str(direction or "absolute").lower()
    if direction_text == "positive":
        return (value - stats["signed_mean"]) / stats["signed_std"]
    if direction_text == "negative":
        return (stats["signed_mean"] - value) / stats["signed_std"]
    return (abs(value) - stats["abs_mean"]) / stats["abs_std"]


def compute_neuron_risk_metrics(
    layer_hidden_states: dict[int, Any] | None,
    *,
    policy: NeuronRiskPolicy | None = None,
) -> dict[str, float]:
    policy = policy or load_neuron_risk_policy()
    if policy is None or not layer_hidden_states or (not policy.chains):
        return neuron_risk_default_metrics(policy)
    chain_metrics: dict[str, dict[str, float | bool]] = {}
    for chain in policy.chains:
        z_scores: list[float] = []
        active_nodes = 0
        for node in chain.nodes:
            raw_vector = layer_hidden_states.get(node.layer_index)
            if raw_vector is None:
                continue
            vector = np.asarray(raw_vector, dtype=np.float64)
            z_score = _node_risk_z(
                vector, dimension=node.dimension, direction=node.direction
            )
            if z_score is None:
                continue
            z_scores.append(float(z_score))
            if z_score >= policy.node_z_threshold:
                active_nodes += 1
        if not z_scores:
            chain_metrics[chain.name] = {
                "score": 0.0,
                "weighted_score": 0.0,
                "active_fraction": 0.0,
                "active": False,
            }
            continue
        positive_scores = [max(0.0, score) for score in z_scores]
        raw_score = float(sum(positive_scores) / len(positive_scores))
        weighted_score = raw_score * float(chain.weight)
        active_fraction = float(active_nodes) / float(len(z_scores))
        chain_metrics[chain.name] = {
            "score": raw_score,
            "weighted_score": weighted_score,
            "active_fraction": active_fraction,
            "active": active_fraction >= float(policy.chain_active_fraction),
        }
    weighted_scores = [
        float(metrics["weighted_score"]) for metrics in chain_metrics.values()
    ]
    max_risk = max(weighted_scores, default=0.0)
    mean_risk = (
        float(sum(weighted_scores) / len(weighted_scores)) if weighted_scores else 0.0
    )
    active_fraction = (
        float(
            sum(
                (1 for metrics in chain_metrics.values() if bool(metrics.get("active")))
            )
        )
        / float(len(chain_metrics))
        if chain_metrics
        else 0.0
    )
    pair_scores: list[float] = []
    pair_active_count = 0
    pair_metrics: dict[str, dict[str, float | bool]] = {}
    for pair in policy.pairs:
        left = chain_metrics.get(pair.left)
        right = chain_metrics.get(pair.right)
        if left is None or right is None:
            continue
        synergy = min(
            float(left.get("active_fraction", 0.0)),
            float(right.get("active_fraction", 0.0)),
        )
        pair_score = (
            0.5
            * (
                float(left.get("weighted_score", 0.0))
                + float(right.get("weighted_score", 0.0))
            )
            * synergy
            * float(pair.weight)
        )
        pair_scores.append(pair_score)
        pair_active = bool(left.get("active")) and bool(right.get("active"))
        pair_metrics[pair.name] = {"score": pair_score, "active": pair_active}
        if pair_active:
            pair_active_count += 1
    pair_risk = max(pair_scores, default=0.0)
    pair_active_fraction = (
        float(pair_active_count) / float(len(pair_scores)) if pair_scores else 0.0
    )
    composite_risk = (
        0.5 * max_risk
        + 0.2 * mean_risk
        + 0.2 * pair_risk
        + 0.05 * active_fraction
        + 0.05 * pair_active_fraction
    )
    metrics = {
        "neuron_chain_max_risk": round(max(0.0, max_risk), 4),
        "neuron_chain_mean_risk": round(max(0.0, mean_risk), 4),
        "neuron_chain_active_fraction": round(max(0.0, active_fraction), 4),
        "neuron_chain_pair_risk": round(max(0.0, pair_risk), 4),
        "neuron_chain_pair_active_fraction": round(max(0.0, pair_active_fraction), 4),
        "neuron_chain_composite_risk": round(max(0.0, composite_risk), 4),
    }
    for chain in policy.chains:
        chain_metric = chain_metrics.get(chain.name) or {}
        keys = neuron_chain_feature_keys(chain.name)
        metrics[keys["score"]] = round(
            max(0.0, float(chain_metric.get("weighted_score", 0.0) or 0.0)), 4
        )
        metrics[keys["active_fraction"]] = round(
            max(0.0, float(chain_metric.get("active_fraction", 0.0) or 0.0)), 4
        )
        metrics[keys["active"]] = float(bool(chain_metric.get("active", False)))
    for pair in policy.pairs:
        pair_metric = pair_metrics.get(pair.name) or {}
        keys = neuron_pair_feature_keys(pair.name)
        metrics[keys["score"]] = round(
            max(0.0, float(pair_metric.get("score", 0.0) or 0.0)), 4
        )
        metrics[keys["active"]] = float(bool(pair_metric.get("active", False)))
    return metrics

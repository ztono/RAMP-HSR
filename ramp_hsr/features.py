from __future__ import annotations
import math
from typing import Any
from ramp_hsr.routes import compute_neuron_risk_metrics, neuron_risk_feature_names
from ramp_hsr.tokenizer_context import validate_max_context_length
from ramp_hsr.runtime import QwenCoderLLM

FEATURE_EXTRACTOR_VERSION = "uncertainty-extractor"
TARGET_MATCH_NORMALIZATION = "whitespace_strip"


def _sanitize_uncertainty_value(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def compute_margin(logits: list[float]) -> float:
    """Compute the margin between top-1 and top-2 logits.

    Low margin indicates uncertainty.

    Args:
        logits: Logit values.

    Returns:
        Margin value (top1 - top2).
    """
    if len(logits) < 2:
        return 0.0
    sorted_logits = sorted(logits, reverse=True)
    return sorted_logits[0] - sorted_logits[1]


def compute_entropy(logits: list[float], temperature: float = 1.0) -> float:
    """Compute the entropy of the softmax distribution.

    High entropy indicates uncertainty.

    Args:
        logits: Logit values.
        temperature: Softmax temperature.

    Returns:
        Entropy value.
    """
    scaled_logits = [logit / temperature for logit in logits]
    max_logit = max(scaled_logits)
    exp_logits = [math.exp(logit - max_logit) for logit in scaled_logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]
    entropy = 0.0
    for p in probs:
        if p > 1e-10:
            entropy -= p * math.log(p)
    return entropy


def compute_layer_disagreement(layer_hidden_states: dict[int, list[float]]) -> float:
    """Compute disagreement between different layer representations.

    High disagreement indicates semantic instability.

    Args:
        layer_hidden_states: Dictionary mapping layer index to hidden state.

    Returns:
        Layer disagreement score (average pairwise distance).
    """
    if len(layer_hidden_states) < 2:
        return 0.0
    layers = list(layer_hidden_states.values())
    total_dist = 0.0
    count = 0
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            dist = _cosine_distance(layers[i], layers[j])
            total_dist += dist
            count += 1
    return total_dist / count if count > 0 else 0.0


def compute_representation_variance(
    layer_hidden_states: dict[int, list[float]]
) -> float:
    """Compute variance of representations across layers.

    High variance indicates internal activation instability.

    Args:
        layer_hidden_states: Dictionary mapping layer index to hidden state.

    Returns:
        Mean per-dimension variance across layers.
    """
    if len(layer_hidden_states) < 2:
        return 0.0
    layers = [
        [_sanitize_uncertainty_value(x) for x in layer]
        for layer in layer_hidden_states.values()
    ]
    n_layers = len(layers)
    n_dims = len(layers[0])
    total_var = 0.0
    for d in range(n_dims):
        mean_d = sum((layer[d] for layer in layers)) / n_layers
        var_d = sum(((layer[d] - mean_d) ** 2 for layer in layers)) / n_layers
        total_var += var_d
    return total_var / n_dims


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine distance (1 - cosine_similarity).
    """
    dot_product = sum((x * y for (x, y) in zip(a, b)))
    norm_a = math.sqrt(sum((x * x for x in a)))
    norm_b = math.sqrt(sum((x * x for x in b)))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot_product / (norm_a * norm_b)
    return 1.0 - max(-1.0, min(1.0, similarity))


_STAT_KEYS = (
    "margin",
    "entropy",
    "max_softmax_probability",
    "energy_score",
    "sampling_disagreement_probability",
    "layer_disagreement",
    "representation_variance",
    *neuron_risk_feature_names(),
)


def _analyze_batch(
    llm_interface: QwenCoderLLM,
    prefixes: list[str],
    top_k: int,
    *,
    max_length: int = 512,
) -> list[dict[str, Any]]:
    max_length = validate_max_context_length(max_length, context="feature extraction")
    fast_batch_fn = getattr(llm_interface, "analyze_prefix_batch_fast", None)
    if callable(fast_batch_fn):
        return fast_batch_fn(prefixes, k=top_k, max_length=max_length)
    return llm_interface.analyze_prefix_batch(prefixes, k=top_k, max_length=max_length)


def _base_topk_tokens(items: list[Any] | None) -> list[str]:
    tokens: list[str] = []
    for item in items or []:
        if isinstance(item, (list, tuple)) and item:
            token = item[0]
        else:
            token = item
        if token is None:
            continue
        token_str = str(token)
        if token_str:
            tokens.append(token_str)
    return tokens


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return 0.0
    return coerced


def _round_optional_float(value: Any, digits: int = 4) -> float | None:
    coerced = _coerce_optional_float(value)
    if coerced is None:
        return None
    return round(coerced, digits)


def _flatten_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        token_ids: list[int] = []
        for item in value:
            token_ids.extend(_flatten_token_ids(item))
        return token_ids
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _encode_target_token(llm_interface: QwenCoderLLM, target: str) -> dict[str, Any]:
    tokenizer = getattr(llm_interface, "tokenizer", None)
    token_ids: list[int] = []
    target_text = "" if target is None else str(target)
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        token_ids = _flatten_token_ids(
            tokenizer.encode(target_text, add_special_tokens=False)
        )
    elif callable(tokenizer):
        encoded = tokenizer(target_text, add_special_tokens=False)
        input_ids = getattr(encoded, "input_ids", None)
        if input_ids is None and isinstance(encoded, dict):
            input_ids = encoded.get("input_ids")
        token_ids = _flatten_token_ids(input_ids)
    return {
        "target_token_ids": token_ids,
        "target_first_token_id": token_ids[0] if token_ids else None,
        "target_is_single_token": len(token_ids) == 1,
    }


def _normalize_whitespace_match(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_top_k_details(
    analysis: dict[str, Any], top_k_tokens: list[Any]
) -> list[dict[str, Any]]:
    raw_details = analysis.get("top_k_details")
    details: list[dict[str, Any]] = []
    if isinstance(raw_details, list) and raw_details:
        for idx, raw in enumerate(raw_details):
            if not isinstance(raw, dict):
                continue
            token = raw.get("token")
            prob = raw.get("prob")
            if idx < len(top_k_tokens) and isinstance(top_k_tokens[idx], (list, tuple)):
                token = top_k_tokens[idx][0] if token is None else token
                if len(top_k_tokens[idx]) > 1 and prob is None:
                    prob = top_k_tokens[idx][1]
            details.append(
                {
                    "rank": _coerce_optional_int(raw.get("rank")) or idx + 1,
                    "token_id": _coerce_optional_int(raw.get("token_id")),
                    "token": "" if token is None else str(token),
                    "prob": _round_optional_float(prob),
                    "logprob": _round_optional_float(raw.get("logprob")),
                    "logit": _round_optional_float(raw.get("logit")),
                }
            )
        return details
    for idx, item in enumerate(top_k_tokens):
        if isinstance(item, (list, tuple)) and item:
            token = item[0]
            prob = item[1] if len(item) > 1 else None
        else:
            token = item
            prob = None
        details.append(
            {
                "rank": idx + 1,
                "token_id": None,
                "token": "" if token is None else str(token),
                "prob": _round_optional_float(prob),
                "logprob": None,
                "logit": None,
            }
        )
    return details


def _target_match_features(
    target: str | None,
    target_features: dict[str, Any],
    base_topk_details: list[dict[str, Any]],
    base_topk: list[list[Any]],
) -> dict[str, Any]:
    target_first_token_id = target_features["target_first_token_id"]
    base_top1_token_id = (
        base_topk_details[0].get("token_id") if base_topk_details else None
    )
    target_text = "" if target is None else str(target)
    normalized_target = _normalize_whitespace_match(target_text)
    base_top1_equals_target_token_id = (
        target_first_token_id is not None
        and base_top1_token_id == target_first_token_id
    )
    base_top1_tokens = _base_topk_tokens(base_topk[:1])
    base_top1_token_text = base_top1_tokens[0] if base_top1_tokens else ""
    base_top1_equals_target_raw_text = bool(
        base_top1_tokens and base_top1_token_text == target_text
    )
    base_top1_equals_target_whitespace_normalized = bool(
        base_top1_tokens
        and _normalize_whitespace_match(base_top1_token_text) == normalized_target
    )
    return {
        **target_features,
        "target_match_normalization": TARGET_MATCH_NORMALIZATION,
        "base_top1_token_id": base_top1_token_id,
        "base_top1_equals_target_token": base_top1_equals_target_whitespace_normalized,
        "base_top1_equals_target_text": base_top1_equals_target_whitespace_normalized,
        "base_top1_equals_target_whitespace_normalized": base_top1_equals_target_whitespace_normalized,
        "base_top1_equals_target_token_id": base_top1_equals_target_token_id,
        "base_top1_equals_target_raw_text": base_top1_equals_target_raw_text,
        "text_match_disagrees_token_id_match": base_top1_equals_target_whitespace_normalized
        != base_top1_equals_target_token_id,
    }


def _target_token_features(
    llm_interface: QwenCoderLLM,
    target: str | None,
    base_topk_details: list[dict[str, Any]],
    base_topk: list[list[Any]],
) -> dict[str, Any]:
    target_features = _encode_target_token(
        llm_interface, "" if target is None else str(target)
    )
    return _target_match_features(target, target_features, base_topk_details, base_topk)


def _features_from_analysis(
    analysis: dict[str, Any],
    llm_interface: QwenCoderLLM,
    top_k: int,
    *,
    target: str | None = None,
) -> dict[str, Any]:
    top_k_tokens = analysis["top_k_tokens"]
    logits = analysis.get("logits")
    layer_states = analysis.get("layer_hidden_states") or {}
    uncertainty = {
        key: round(
            _sanitize_uncertainty_value(
                (analysis.get("uncertainty") or {}).get(key, 0.0)
            ),
            4,
        )
        for key in _STAT_KEYS
    }
    if logits is not None:
        uncertainty["margin"] = round(
            _sanitize_uncertainty_value(compute_margin(logits)), 4
        )
        uncertainty["entropy"] = round(
            _sanitize_uncertainty_value(compute_entropy(logits)), 4
        )
    if layer_states:
        uncertainty["layer_disagreement"] = round(
            _sanitize_uncertainty_value(compute_layer_disagreement(layer_states)), 4
        )
        uncertainty["representation_variance"] = round(
            _sanitize_uncertainty_value(compute_representation_variance(layer_states)),
            4,
        )
        neuron_risks = compute_neuron_risk_metrics(layer_states)
        for key, value in neuron_risks.items():
            uncertainty[key] = round(_sanitize_uncertainty_value(value), 4)
    base_topk = [[token, round(float(prob), 4)] for (token, prob) in top_k_tokens]
    base_topk_details = _normalize_top_k_details(analysis, top_k_tokens)
    features = {
        "base_prediction": top_k_tokens[0][0] if top_k_tokens else "",
        "base_topk": base_topk,
        "base_topk_details": base_topk_details,
        "uncertainty": uncertainty,
        "base_model_name": llm_interface.model_name,
        "tokenizer_name": getattr(
            llm_interface.tokenizer, "name_or_path", llm_interface.model_name
        ),
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
    }
    if target is not None:
        features.update(
            _target_token_features(llm_interface, target, base_topk_details, base_topk)
        )
    return features

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from ramp_hsr.runtime_helpers import OfflineEvaluator


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_repair_feature_schema(*, checkpoint, manifest_path=None):
    if (
        checkpoint.get("condition_on") != "hidden"
        or int(checkpoint.get("route_dim", -1))
        or int(checkpoint.get("code_dim", -1))
    ):
        raise ValueError("RAMP-HSR requires hidden-only input")
    return ((), ())


def _base_prediction_token_id(sample: dict[str, Any]) -> int | None:
    details = sample.get("base_topk_details")
    if isinstance(details, list) and details:
        first = details[0]
        if isinstance(first, dict):
            token_id = _coerce_optional_int(first.get("token_id"))
            if token_id is not None:
                return token_id
    return _coerce_optional_int(sample.get("base_top1_token_id"))


def _decode_single_token(tokenizer: Any, token_id: int) -> str:
    if hasattr(tokenizer, "batch_decode"):
        decoded = tokenizer.batch_decode(
            [[int(token_id)]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded[0]) if decoded else ""
    return str(tokenizer.decode([int(token_id)]))


def _resolve_norm_and_lm_head(llm: Any) -> tuple[torch.nn.Module, torch.nn.Module]:
    model = getattr(llm, "model", None)
    norm = getattr(getattr(model, "model", None), "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if norm is None or lm_head is None:
        raise RuntimeError("Unable to resolve base-model norm/lm_head from rollout LLM")
    return (norm, lm_head)


def _cached_fp32_tensor(
    module: torch.nn.Module, name: str, tensor: torch.Tensor
) -> torch.Tensor:
    cache_name = f"_srlcc_rollout_{name}_fp32"
    cached = getattr(module, cache_name, None)
    if (
        not torch.is_tensor(cached)
        or cached.device != tensor.device
        or cached.shape != tensor.shape
    ):
        cached = tensor.detach().to(device=tensor.device, dtype=torch.float32)
        setattr(module, cache_name, cached)
    return cached


def _repair_logits_fp32(
    *,
    norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    repaired_h: torch.Tensor,
    hidden_state_interface: str = "post_norm",
) -> torch.Tensor:
    """Compute repair logits in fp32 without mutating the rollout base model.

    QwenCoderLLM keeps the base model in fp16 on CUDA. Cached h_last tensors are
    post-final-norm for current HF causal-LM outputs, so the default path applies
    the LM head directly. The pre_norm option is kept only for older manifests.
    """
    h = repaired_h.float()
    if hidden_state_interface == "post_norm":
        decoded_h = h
    elif hidden_state_interface == "pre_norm":
        norm_weight = getattr(norm, "weight", None)
        eps = getattr(norm, "variance_epsilon", None)
        if torch.is_tensor(norm_weight) and eps is not None:
            weight = _cached_fp32_tensor(norm, "weight", norm_weight)
            variance = h.pow(2).mean(dim=-1, keepdim=True)
            decoded_h = h * torch.rsqrt(variance + float(eps))
            decoded_h = decoded_h * weight
        else:
            try:
                norm_dtype = next(norm.parameters()).dtype
            except StopIteration:
                norm_dtype = h.dtype
            decoded_h = norm(repaired_h.to(dtype=norm_dtype)).float()
    else:
        raise ValueError(
            f"Unsupported hidden_state_interface={hidden_state_interface!r}; expected 'post_norm' or 'pre_norm'"
        )
    head_weight = getattr(lm_head, "weight", None)
    if not torch.is_tensor(head_weight):
        return lm_head(decoded_h.to(dtype=next(lm_head.parameters()).dtype)).float()
    weight = _cached_fp32_tensor(lm_head, "weight", head_weight)
    bias_tensor = getattr(lm_head, "bias", None)
    bias = (
        _cached_fp32_tensor(lm_head, "bias", bias_tensor)
        if torch.is_tensor(bias_tensor)
        else None
    )
    return F.linear(decoded_h, weight, bias)


def _pad_rollout_context_ids(
    *, tokenizer: Any, context_id_rows: list[list[int]], max_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    fallback_id = None
    for token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id):
        if token_id is not None:
            fallback_id = int(token_id)
            break
    if fallback_id is None:
        fallback_id = 0
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.bos_token_id
    if pad_id is None:
        pad_id = fallback_id
    pad_id = int(pad_id)
    rows: list[list[int]] = []
    max_length = max(1, int(max_length))
    for context_ids in context_id_rows:
        row = [int(token_id) for token_id in context_ids if token_id is not None]
        if not row:
            row = [fallback_id]
        rows.append(row[-max_length:])
    batch_len = max((len(row) for row in rows))
    padding_side = str(getattr(tokenizer, "padding_side", "left") or "left")
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for row in rows:
        pad_len = batch_len - len(row)
        if padding_side == "right":
            padded.append([*row, *[pad_id] * pad_len])
            masks.append([*(1 for _ in row), *[0] * pad_len])
        else:
            padded.append([*[pad_id] * pad_len, *row])
            masks.append([*[0] * pad_len, *(1 for _ in row)])
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def _analyze_rollout_context_ids_batch(
    llm: Any, context_id_rows: list[list[int]], *, top_k: int, max_length: int
) -> list[dict[str, Any]]:
    if not context_id_rows:
        return []
    from ramp_hsr.routes import load_neuron_risk_policy, neuron_risk_default_metrics
    from ramp_hsr.runtime import _batch_neuron_chain_risks

    torch_mod = getattr(llm, "_torch", torch)
    tokenizer = llm.tokenizer
    (input_ids, attention_mask) = _pad_rollout_context_ids(
        tokenizer=tokenizer,
        context_id_rows=context_id_rows,
        max_length=max_length,
        device=llm.device,
    )
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch_mod.arange(input_ids.shape[0], device=input_ids.device)
    use_last_position_only = bool(torch_mod.all(attention_mask[:, -1] == 1).item())
    with torch_mod.inference_mode():
        outputs = llm._forward_batched_model(
            input_ids,
            attention_mask,
            output_hidden_states=True,
            use_last_position_only=use_last_position_only,
        )
    if outputs.logits.shape[1] == 1:
        logits = outputs.logits[:, 0, :]
    else:
        logits = outputs.logits[batch_indices, seq_lengths, :]
    log_probs = torch_mod.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    top_k_val = min(int(top_k), probs.shape[-1])
    (top_k_probs, top_k_indices) = torch_mod.topk(probs, top_k_val, dim=-1)
    top_k_log_probs = torch_mod.gather(log_probs, dim=-1, index=top_k_indices)
    top_k_logits = torch_mod.gather(logits, dim=-1, index=top_k_indices)
    top2_val = min(2, logits.shape[-1])
    (top2_logits, _) = torch_mod.topk(logits, top2_val, dim=-1)
    if top2_val >= 2:
        margins = top2_logits[:, 0] - top2_logits[:, 1]
    else:
        margins = torch_mod.zeros(logits.shape[0], device=logits.device)
    entropies = -(probs * log_probs).sum(dim=-1)
    sample_layers = llm._resolve_sample_layers(None)
    hidden_states = outputs.hidden_states
    if use_last_position_only:
        final_hidden = hidden_states[-1][:, -1, :]
    else:
        final_hidden = hidden_states[-1][batch_indices, seq_lengths, :]
    layer_vectors: list[Any] = []
    for layer_idx in sample_layers:
        if layer_idx >= len(hidden_states):
            continue
        if use_last_position_only:
            hidden = hidden_states[layer_idx][:, -1, :]
        else:
            hidden = hidden_states[layer_idx][batch_indices, seq_lengths, :]
        layer_vectors.append(hidden.float())
    if len(layer_vectors) >= 2:
        stacked = torch_mod.stack(layer_vectors, dim=0)
        nonfinite_hidden_mask = ~torch_mod.isfinite(stacked)
        hidden_nonfinite_per_sample = nonfinite_hidden_mask.any(dim=0).any(dim=-1)
        stacked = torch_mod.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
        representation_variances = stacked.var(dim=0, unbiased=False).mean(dim=-1)
        normalized_layers = torch_mod.nn.functional.normalize(
            stacked, p=2, dim=-1, eps=1e-12
        )
        similarities = torch_mod.einsum(
            "lbd,mbd->blm", normalized_layers, normalized_layers
        )
        upper = torch_mod.triu_indices(
            similarities.shape[1],
            similarities.shape[2],
            offset=1,
            device=similarities.device,
        )
        layer_disagreements = (1.0 - similarities[:, upper[0], upper[1]]).mean(dim=-1)
        neuron_chain_risks = _batch_neuron_chain_risks(
            torch=torch_mod, stacked_layers=stacked, sample_layers=sample_layers
        )
    else:
        representation_variances = torch_mod.zeros(
            logits.shape[0], device=logits.device
        )
        layer_disagreements = torch_mod.zeros(logits.shape[0], device=logits.device)
        hidden_nonfinite_per_sample = torch_mod.zeros(
            logits.shape[0], dtype=torch_mod.bool, device=logits.device
        )
        neuron_chain_risks = {
            key: torch_mod.zeros(logits.shape[0], device=logits.device)
            for key in neuron_risk_default_metrics(load_neuron_risk_policy())
        }
    top_k_probs_cpu = top_k_probs.detach().cpu().tolist()
    top_k_indices_cpu = top_k_indices.detach().cpu().tolist()
    top_k_log_probs_cpu = top_k_log_probs.detach().cpu().tolist()
    top_k_logits_cpu = top_k_logits.detach().cpu().tolist()
    margins_cpu = margins.detach().cpu().tolist()
    entropies_cpu = entropies.detach().cpu().tolist()
    layer_disagreements_cpu = layer_disagreements.detach().cpu().tolist()
    representation_variances_cpu = representation_variances.detach().cpu().tolist()
    neuron_chain_risks_cpu = {
        key: value.detach().cpu().tolist()
        for (key, value) in neuron_chain_risks.items()
    }
    hidden_nonfinite_cpu = hidden_nonfinite_per_sample.detach().cpu().tolist()
    flat_top_k_ids = [token_id for row in top_k_indices_cpu for token_id in row]
    flat_top_k_tokens = llm._decode_token_ids(flat_top_k_ids)
    results: list[dict[str, Any]] = []
    token_offset = 0
    for i, (indices_row, probs_row) in enumerate(
        zip(top_k_indices_cpu, top_k_probs_cpu)
    ):
        row_tokens = flat_top_k_tokens[token_offset : token_offset + len(indices_row)]
        token_offset += len(indices_row)
        top_k_tokens = [
            (token, float(prob)) for (token, prob) in zip(row_tokens, probs_row)
        ]
        top_k_details = llm._build_top_k_details(
            token_ids=[int(token_id) for token_id in indices_row],
            tokens=row_tokens,
            probs=[float(prob) for prob in probs_row],
            logprobs=[float(value) for value in top_k_log_probs_cpu[i]],
            logits=[float(value) for value in top_k_logits_cpu[i]],
        )
        raw_uncertainty = {
            "margin": float(margins_cpu[i]),
            "entropy": float(entropies_cpu[i]),
            "layer_disagreement": float(layer_disagreements_cpu[i]),
            "representation_variance": float(representation_variances_cpu[i]),
        }
        for key, values in neuron_chain_risks_cpu.items():
            raw_uncertainty[key] = float(values[i])
        (uncertainty, diagnostics) = llm._sanitize_uncertainty_dict(raw_uncertainty)
        diagnostics["nonfinite_hidden_states"] = bool(hidden_nonfinite_cpu[i])
        results.append(
            {
                "top_k_tokens": top_k_tokens,
                "top_k_details": top_k_details,
                "uncertainty": uncertainty,
                "uncertainty_diagnostics": diagnostics,
                "_h_last_tensor": final_hidden[i].detach().cpu().float(),
            }
        )
    return results


def _extract_rollout_features_with_hidden(
    evaluator: OfflineEvaluator,
    cache_keys: list[tuple[int, ...]],
    template_sample: dict[str, Any],
    episode: list[dict[str, Any]],
    *,
    context_ids_by_key: dict[tuple[int, ...], list[int]] | None = None,
    prefix_by_key: dict[tuple[int, ...], str] | None = None,
    max_length: int = 512,
) -> dict[tuple[int, ...], dict[str, Any]]:
    if not cache_keys:
        return {}
    unique_keys: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for key in cache_keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
    (llm, _) = evaluator._get_long_sequence_rollout_llm(episode)
    from ramp_hsr.features import _analyze_batch, _features_from_analysis

    top_k = len(template_sample.get("base_topk", []) or [])
    if top_k <= 0:
        top_k = 20
    batch_size = max(1, int(evaluator.long_sequence_rollout_batch_size))
    context_max_length = max(1, int(max_length))
    features_by_key: dict[tuple[int, ...], dict[str, Any]] = {}
    for start in range(0, len(unique_keys), batch_size):
        batch_keys = unique_keys[start : start + batch_size]
        if context_ids_by_key is not None and all(
            (key in context_ids_by_key for key in batch_keys)
        ):
            analyses = _analyze_rollout_context_ids_batch(
                llm,
                [context_ids_by_key[key] for key in batch_keys],
                top_k=top_k,
                max_length=context_max_length,
            )
        else:
            batch_prefixes = [
                str((prefix_by_key or {}).get(key) or "") for key in batch_keys
            ]
            analyses = _analyze_batch(
                llm, batch_prefixes, top_k, max_length=context_max_length
            )
        for key, analysis in zip(batch_keys, analyses, strict=True):
            features = _features_from_analysis(analysis, llm, top_k, target=None)
            h_last_tensor = analysis.get("_h_last_tensor")
            if h_last_tensor is not None:
                features["_h_last_tensor"] = h_last_tensor
            features_by_key[key] = features
    missing_hidden_keys = [
        key
        for key in unique_keys
        if features_by_key.get(key, {}).get("_h_last_tensor") is None
    ]
    if not missing_hidden_keys:
        return features_by_key
    if context_ids_by_key is not None and all(
        (key in context_ids_by_key for key in missing_hidden_keys)
    ):
        missing_analyses = _analyze_rollout_context_ids_batch(
            llm,
            [context_ids_by_key[key] for key in missing_hidden_keys],
            top_k=top_k,
            max_length=context_max_length,
        )
        last_hidden_batch = [
            analysis.get("_h_last_tensor") for analysis in missing_analyses
        ]
    else:
        last_hidden_batch = llm.extract_last_hidden_states_batch(
            [str((prefix_by_key or {}).get(key) or "") for key in missing_hidden_keys],
            max_length=max_length,
        )
    for key, h_last in zip(missing_hidden_keys, last_hidden_batch, strict=True):
        feature_row = features_by_key.get(key)
        if feature_row is not None:
            feature_row["_h_last_tensor"] = h_last
    return features_by_key


def _named_feature_value(
    values: list[float], names: tuple[str, ...], name: str, default: float = 0.0
) -> float:
    try:
        index = names.index(name)
    except ValueError:
        return default
    if index >= len(values):
        return default
    try:
        return float(values[index])
    except (TypeError, ValueError):
        return default


def _runtime_route_gate_score(
    route_values: list[float], route_feature_schema: tuple[str, ...]
) -> float:
    names = (
        "risk_route_max_risk",
        "neuron_chain_composite_risk",
        "neuron_chain_pair_risk",
        "risk_route_dominant_chain_score",
        "risk_route_dominant_pair_score",
    )
    return max(
        (
            _named_feature_value(route_values, route_feature_schema, name)
            for name in names
        )
    )


def _repair_confidence(logits: torch.Tensor) -> float:
    if logits.numel() <= 0:
        return 0.0
    probs = torch.softmax(logits.float(), dim=-1)
    return float(probs.max(dim=-1).values.item())


def _runtime_gate_decision(*, mode, **diagnostic_values):
    if mode != "always_repair":
        raise ValueError("Only the evaluated always-repair policy is released")
    return (True, "always_repair")

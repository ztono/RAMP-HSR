from __future__ import annotations
import argparse
from collections import Counter
import io
import json
import keyword
import math
import os
from pathlib import Path
import sys
import tokenize
from typing import Any
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from ramp_hsr.checkpoints import load_editor
from scripts._rollout_features import (
    _base_prediction_token_id,
    _decode_single_token,
    _extract_rollout_features_with_hidden,
    _named_feature_value,
    _pad_rollout_context_ids,
    _repair_confidence,
    _repair_logits_fp32,
    _resolve_norm_and_lm_head,
    _resolve_repair_feature_schema,
    _runtime_gate_decision,
    _runtime_route_gate_score,
)
from ramp_hsr.runtime_helpers import (
    OfflineEvaluator,
    _calc_codebleu,
    _build_completion_metric_context,
    _compute_completion_metrics_with_context,
    _contains_generation_special_token,
    _empty_metric_agg,
    _empty_validity_agg,
    _metric_deltas,
    _summarize_metric_agg,
    _summarize_validity_agg,
    _take_evenly,
    _update_metric_agg,
    _update_validity_agg,
    evaluate_python_validity,
)
from ramp_hsr.sample_features import _feature_vector
from ramp_hsr.io import (
    discover_sample_files,
    is_sha256,
    load_file_content_map,
    read_jsonl_streaming,
    resolve_full_file_content,
    write_json,
)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _line_start_offsets(text: str) -> list[int]:
    offsets = [0]
    for index, char in enumerate(text):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def _python_name_token_spans(text: str) -> list[tuple[int, int]]:
    offsets = _line_start_offsets(text)
    spans: list[tuple[int, int]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type != tokenize.NAME or keyword.iskeyword(token.string):
                continue
            (start_line, start_col) = token.start
            (end_line, end_col) = token.end
            if start_line <= 0 or end_line <= 0:
                continue
            if start_line > len(offsets) or end_line > len(offsets):
                continue
            start = offsets[start_line - 1] + start_col
            end = offsets[end_line - 1] + end_col
            if start < end:
                spans.append((start, end))
    except (IndentationError, SyntaxError, SystemError, tokenize.TokenError):
        return spans
    return spans


def _piece_core_span(piece: str, absolute_start: int) -> tuple[int, int, str] | None:
    text = str(piece)
    left = len(text) - len(text.lstrip())
    right = len(text.rstrip())
    if left >= right:
        return None
    core = text[left:right]
    return (absolute_start + left, absolute_start + right, core)


def _last_added_python_name_piece(prefix: str, added_text: str) -> bool:
    """Return whether the newest generated text ends in a Python NAME token."""
    if not added_text:
        return False
    prefix = str(prefix)
    added_text = str(added_text)
    added_start = len(prefix)
    added_end = added_start + len(added_text)
    core_span = _piece_core_span(added_text, added_start)
    if core_span is None:
        return False
    (_core_start, core_end, _core) = core_span
    window_chars = int(os.environ.get("SRLCC_ROLLOUT_NAME_WINDOW_CHARS", "2048"))
    window_start = max(0, len(prefix) - max(128, window_chars))
    local_text = prefix[window_start:] + added_text
    local_added_start = added_start - window_start
    local_added_end = added_end - window_start
    local_core_end = core_end - window_start
    spans = _python_name_token_spans(local_text)
    for span_start, span_end in spans:
        if span_end != local_core_end:
            continue
        if span_start < local_added_end and local_added_start < span_end:
            return True
    return False


def _decode_token_pieces(tokenizer: Any, token_ids: list[int]) -> list[str]:
    if not token_ids:
        return []
    if hasattr(tokenizer, "batch_decode"):
        decoded = tokenizer.batch_decode(
            [[int(token_id)] for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return [str(piece) for piece in decoded]
    return [str(tokenizer.decode([int(token_id)])) for token_id in token_ids]


def _encode_text(
    tokenizer: Any, text: str, *, max_context_chars: int = 8192
) -> list[int]:
    if max_context_chars > 0 and len(text) > max_context_chars:
        text = text[-max_context_chars:]
    try:
        raw = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        raw = tokenizer.encode(text)
    if not isinstance(raw, list):
        return []
    return [int(token_id) for token_id in raw]


def _fallback_context_ids(tokenizer: Any) -> list[int]:
    for token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id):
        if token_id is not None:
            return [int(token_id)]
    return [0]


def _select_next_token_logits(
    *, logits: torch.Tensor, attention_mask: torch.Tensor, torch_mod: Any
) -> torch.Tensor:
    if logits.shape[1] == 1:
        return logits[:, 0, :]
    if bool(torch_mod.all(attention_mask[:, -1] == 1).item()):
        return logits[:, -1, :]
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch_mod.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, seq_lengths, :]


def _decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        )
    except TypeError:
        return str(tokenizer.decode(token_ids))


def _append_token_text(
    tokenizer: Any, context_ids: list[int], token_id: int | None, *, fallback_text: str
) -> tuple[str, list[int]]:
    """Return the text added by token_id in the current tokenizer context."""
    if token_id is None:
        return (str(fallback_text), context_ids)
    next_context_ids = context_ids + [int(token_id)]
    try:
        prefix_text = _decode_ids(tokenizer, context_ids)
        full_text = _decode_ids(tokenizer, next_context_ids)
    except Exception:
        return (str(fallback_text), next_context_ids)
    if full_text.startswith(prefix_text):
        added_text = full_text[len(prefix_text) :]
        if added_text:
            return (added_text, next_context_ids)
    return (str(fallback_text), next_context_ids)


def _greedy_generate_token_ids_from_context_batch(
    llm: Any,
    context_id_rows: list[list[int]],
    *,
    max_new_tokens: int,
    max_length: int = 512,
) -> list[list[int]]:
    if max_new_tokens <= 0:
        return [[] for _ in context_id_rows]
    if not context_id_rows:
        return []
    tokenizer = llm.tokenizer
    torch_mod = llm._torch
    sanitized_rows = [
        [int(token_id) for token_id in row] if row else _fallback_context_ids(tokenizer)
        for row in context_id_rows
    ]
    (input_ids, attention_mask) = _pad_rollout_context_ids(
        tokenizer=tokenizer,
        context_id_rows=sanitized_rows,
        max_length=max_length,
        device=llm.device,
    )
    generated_steps: list[torch.Tensor] = []
    past_key_values = None
    with torch_mod.inference_mode():
        current_input_ids = input_ids
        for step in range(max_new_tokens):
            outputs = llm.model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = _select_next_token_logits(
                logits=outputs.logits,
                attention_mask=attention_mask,
                torch_mod=torch_mod,
            )
            next_ids = torch_mod.argmax(logits, dim=-1)
            generated_steps.append(next_ids.detach())
            past_key_values = outputs.past_key_values
            current_input_ids = next_ids.unsqueeze(-1)
            if step + 1 < max_new_tokens:
                attention_mask = torch_mod.cat(
                    [
                        attention_mask,
                        torch_mod.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
    if not generated_steps:
        return [[] for _ in sanitized_rows]
    generated_tensor = torch_mod.stack(generated_steps, dim=1)
    return [
        [int(token_id) for token_id in row]
        for row in generated_tensor.detach().cpu().tolist()
    ]


def _encode_suffix_target(
    tokenizer: Any, suffix: str, *, steps: int, initial_chars: int
) -> tuple[list[int], list[str], str, str | None]:
    if steps <= 0 or not suffix:
        return ([], [], "", "empty_suffix")
    window = max(initial_chars, steps * 8)
    encoded: list[int] = []
    while True:
        try:
            raw = tokenizer.encode(
                suffix[: min(len(suffix), window)], add_special_tokens=False
            )
        except Exception as exc:
            return ([], [], "", f"tokenize_error:{type(exc).__name__}")
        encoded = [int(token_id) for token_id in raw] if isinstance(raw, list) else []
        if len(encoded) >= steps or window >= len(suffix):
            break
        window = min(len(suffix), window * 2)
    if len(encoded) < steps:
        return ([], [], "", "short_suffix_tokenization")
    target_ids = encoded[:steps]
    pieces = _decode_token_pieces(tokenizer, target_ids)
    target_text = "".join(pieces)
    if not target_text or not suffix.startswith(target_text):
        try:
            decoded = tokenizer.decode(
                target_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = tokenizer.decode(target_ids)
        except Exception:
            decoded = ""
        target_text = str(decoded)
    if not target_text or not suffix.startswith(target_text):
        return ([], [], "", "target_text_mismatch")
    return (target_ids, pieces, target_text, None)


def _runtime_sample_from_features(
    template_sample: dict[str, Any],
    *,
    prefix: str,
    features: dict[str, Any],
    generated_prefix: str,
) -> dict[str, Any]:
    runtime_sample = dict(template_sample)
    runtime_sample.update(dict(features))
    for field in (
        "slot",
        "slot_type",
        "slot_multiplier",
        "position_kind",
        "syntax_state",
        "visible_symbols",
        "param_symbols",
        "local_var_symbols",
        "import_candidates",
        "member_candidates",
        "attribute_candidates",
        "struct_punct_candidates",
        "template_candidates",
        "oracle_action",
        "proxy_gate",
        "oracle_gate",
        "gate_label_source",
        "oracle_symbol",
        "target_family",
    ):
        runtime_sample.pop(field, None)
    runtime_sample["prefix"] = prefix
    runtime_sample["absolute_position"] = len(prefix)
    runtime_sample["line"] = prefix.count("\n") + 1
    runtime_sample["col"] = len(prefix.rsplit("\n", 1)[-1]) + 1
    runtime_sample["full_file_content"] = prefix
    return runtime_sample


def _context_key(context_ids: list[int], max_context_tokens: int) -> tuple[int, ...]:
    if max_context_tokens > 0:
        context_ids = context_ids[-max_context_tokens:]
    return tuple((int(token_id) for token_id in context_ids))


def _extract_runtime_features_batch(
    *,
    evaluator: OfflineEvaluator,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    prefixes: list[str],
    max_context_tokens: int,
) -> dict[str, dict[str, Any]]:
    if not samples:
        return {}
    keys: list[tuple[int, ...]] = []
    context_ids_by_key: dict[tuple[int, ...], list[int]] = {}
    prefix_by_key: dict[tuple[int, ...], str] = {}
    prefix_key: dict[str, tuple[int, ...]] = {}
    for prefix in prefixes:
        context_ids = _encode_text(tokenizer, prefix)
        if not context_ids:
            context_ids = _fallback_context_ids(tokenizer)
        key = _context_key(context_ids, max_context_tokens)
        keys.append(key)
        context_ids_by_key[key] = list(context_ids)
        prefix_by_key[key] = prefix
        prefix_key[prefix] = key
    features_by_key = _extract_rollout_features_with_hidden(
        evaluator,
        keys,
        samples[0],
        samples,
        context_ids_by_key=context_ids_by_key,
        prefix_by_key=prefix_by_key,
        max_length=max_context_tokens,
    )
    return {prefix: features_by_key[prefix_key[prefix]] for prefix in prefixes}


def _extract_runtime_features_from_context_ids_batch(
    *,
    evaluator: OfflineEvaluator,
    samples: list[dict[str, Any]],
    prefixes: list[str],
    context_id_rows: list[list[int]],
    max_context_tokens: int,
) -> list[dict[str, Any]]:
    if not samples:
        return []
    if len(samples) != len(prefixes) or len(samples) != len(context_id_rows):
        raise ValueError(
            "samples, prefixes, and context_id_rows must have the same length"
        )
    keys: list[tuple[int, ...]] = []
    context_ids_by_key: dict[tuple[int, ...], list[int]] = {}
    prefix_by_key: dict[tuple[int, ...], str] = {}
    for prefix, context_ids in zip(prefixes, context_id_rows, strict=True):
        key = _context_key(context_ids, max_context_tokens)
        keys.append(key)
        context_ids_by_key[key] = list(context_ids)
        prefix_by_key[key] = prefix
    features_by_key = _extract_rollout_features_with_hidden(
        evaluator,
        keys,
        samples[0],
        samples,
        context_ids_by_key=context_ids_by_key,
        prefix_by_key=prefix_by_key,
        max_length=max_context_tokens,
    )
    return [features_by_key[key] for key in keys]


def _base_next_token_from_features(features: dict[str, Any]) -> tuple[int | None, str]:
    token_id = _base_prediction_token_id(features)
    token_text = str(features.get("base_prediction") or "")
    return (token_id, token_text)


def _predict_repair_tokens_batch(
    *,
    repair_moe,
    norm,
    lm_head,
    tokenizer: Any,
    runtime_samples: list[dict[str, Any]],
    device: str,
    route_feature_schema: tuple[str, ...],
    code_feature_schema: tuple[str, ...],
    runtime_gate_mode: str,
    gate_risk_threshold: float,
    gate_margin_threshold: float,
    gate_confidence_threshold: float,
    hidden_state_interface: str,
) -> list[dict[str, Any]]:
    if route_feature_schema or code_feature_schema:
        raise ValueError(
            "RAMP-HSR prediction accepts only hidden states; feature schemas must be empty"
        )
    if not runtime_samples:
        return []
    h_last_rows = []
    route_rows = []
    code_rows = []
    for runtime_sample in runtime_samples:
        h_last = runtime_sample.get("_h_last_tensor")
        if h_last is None:
            raise ValueError(
                "Runtime sample is missing _h_last_tensor for BP-HSR rollout"
            )
        route_values = _feature_vector(runtime_sample, route_feature_schema)
        code_values = _feature_vector(runtime_sample, code_feature_schema)
        if torch.is_tensor(h_last):
            h_tensor = h_last.detach().to(device=device, dtype=torch.float32)
        else:
            h_tensor = torch.tensor(h_last, dtype=torch.float32, device=device)
        if h_tensor.dim() == 1:
            h_tensor = h_tensor.unsqueeze(0)
        h_last_rows.append(h_tensor.squeeze(0))
        route_rows.append(route_values)
        code_rows.append(code_values)
    h_last_tensor = torch.stack(h_last_rows, dim=0)
    repaired_h = repair_moe(h_last_tensor)
    repaired_logits = _repair_logits_fp32(
        norm=norm,
        lm_head=lm_head,
        repaired_h=repaired_h,
        hidden_state_interface=hidden_state_interface,
    )
    raw_token_ids = [
        int(token_id)
        for token_id in repaired_logits.argmax(dim=-1).detach().cpu().tolist()
    ]
    raw_token_texts = _decode_token_pieces(tokenizer, raw_token_ids)
    confidences = [
        _repair_confidence(repaired_logits[index : index + 1])
        for index in range(repaired_logits.shape[0])
    ]
    predictions: list[dict[str, Any]] = []
    for index, runtime_sample in enumerate(runtime_samples):
        route_values = route_rows[index]
        code_values = code_rows[index]
        token_id = raw_token_ids[index]
        token_text = raw_token_texts[index]
        base_prediction = str(runtime_sample.get("base_prediction") or "")
        base_token_id = _base_prediction_token_id(runtime_sample)
        route_score = _runtime_route_gate_score(route_values, route_feature_schema)
        base_margin = _named_feature_value(code_values, code_feature_schema, "margin")
        confidence = confidences[index]
        if _contains_generation_special_token(token_text):
            predictions.append(
                {
                    "prediction": base_prediction,
                    "prediction_token_id": base_token_id,
                    "raw_repair_prediction": token_text,
                    "raw_repair_token_id": token_id,
                    "repair_used": False,
                    "gate_reason": "special_token_fallback",
                    "route_score": route_score,
                    "base_margin": base_margin,
                    "repair_confidence": confidence,
                }
            )
            continue
        (repair_used, gate_reason) = _runtime_gate_decision(
            mode=runtime_gate_mode,
            route_score=route_score,
            base_margin=base_margin,
            repair_confidence=confidence,
            risk_threshold=gate_risk_threshold,
            margin_threshold=gate_margin_threshold,
            confidence_threshold=gate_confidence_threshold,
        )
        predictions.append(
            {
                "prediction": token_text if repair_used else base_prediction,
                "prediction_token_id": token_id if repair_used else base_token_id,
                "raw_repair_prediction": token_text,
                "raw_repair_token_id": token_id,
                "repair_used": repair_used,
                "gate_reason": gate_reason,
                "route_score": route_score,
                "base_margin": base_margin,
                "repair_confidence": confidence,
            }
        )
    return predictions


def _sample_id(sample: dict[str, Any], index: int) -> str:
    raw = sample.get("sample_id") or sample.get("id")
    if raw:
        return str(raw)
    return f"{sample.get('repo_id', '')}:{sample.get('file_path', '')}:{sample.get('absolute_position', '')}:{index}"


def _resolve_rollout_full_file_content(
    sample: dict[str, Any],
    *,
    fc_map: dict[str, str],
    file_path_cache: dict[str, str | None],
) -> tuple[str | None, str]:
    resolved = resolve_full_file_content(sample, fc_map)
    absolute_position = sample.get("absolute_position")
    if (
        isinstance(resolved, str)
        and (not is_sha256(resolved))
        and isinstance(absolute_position, int)
        and (0 <= absolute_position < len(resolved))
    ):
        return (resolved, "sample_or_fc_map")
    file_path = sample.get("file_path")
    if isinstance(file_path, str) and file_path:
        if file_path not in file_path_cache:
            path = Path(file_path)
            try:
                file_path_cache[file_path] = (
                    path.read_text(encoding="utf-8", errors="replace")
                    if path.exists()
                    else None
                )
            except OSError:
                file_path_cache[file_path] = None
        cached = file_path_cache.get(file_path)
        if isinstance(cached, str):
            return (cached, "file_path")
    if isinstance(resolved, str):
        return (resolved, "unresolved_sample")
    return (None, "missing")


def _count_jsonl_records(sample_files: list[Path]) -> int:
    total = 0
    for sample_file in sample_files:
        with open(sample_file, "r", encoding="utf-8") as handle:
            total += sum((1 for line in handle if line.strip()))
    return total


def _iter_selected_jsonl_records(
    sample_files: list[Path], selected_global_indices: set[int]
):
    if not selected_global_indices:
        return
    max_selected = max(selected_global_indices)
    global_index = 0
    for sample_file in sample_files:
        with open(sample_file, "r", encoding="utf-8") as handle:
            local_index = 0
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if global_index > max_selected:
                    return
                if global_index in selected_global_indices:
                    yield (global_index, sample_file, local_index, json.loads(stripped))
                global_index += 1
                local_index += 1


def _prepare_rollout_sample(
    sample: dict[str, Any],
    *,
    index: int,
    tokenizer: Any,
    steps: int,
    target_window_chars: int,
    fc_map: dict[str, str],
    file_path_cache: dict[str, str | None],
    stats: Counter,
) -> dict[str, Any] | None:
    (full_file_content, content_source) = _resolve_rollout_full_file_content(
        sample, fc_map=fc_map, file_path_cache=file_path_cache
    )
    stats[f"context_source_{content_source}"] += 1
    absolute_position = sample.get("absolute_position")
    if not isinstance(full_file_content, str) or not isinstance(absolute_position, int):
        stats["missing_context"] += 1
        return None
    if absolute_position < 0 or absolute_position >= len(full_file_content):
        stats["bad_position"] += 1
        return None
    suffix = full_file_content[absolute_position:]
    (target_ids, target_pieces, target_text, skip_reason) = _encode_suffix_target(
        tokenizer, suffix, steps=steps, initial_chars=target_window_chars
    )
    if skip_reason is not None:
        stats[f"skip_{skip_reason}"] += 1
        return None
    row = dict(sample)
    row["_rollout_sample_id"] = _sample_id(sample, index)
    row["_rollout_prefix"] = full_file_content[:absolute_position]
    row["_rollout_suffix"] = suffix
    row["_rollout_full_file_sha256"] = sample.get("full_file_sha256")
    row["_rollout_target_token_ids"] = target_ids
    row["_rollout_target_token_pieces"] = target_pieces
    row["_rollout_target_text"] = target_text
    stats["eligible_samples"] += 1
    return row


def _collect_samples(
    samples_file: str | Path,
    *,
    tokenizer: Any,
    steps: int,
    target_window_chars: int,
    max_samples: int,
    sample_selection: str,
    sample_shard_index: int = 0,
    sample_num_shards: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_num_shards = max(1, int(sample_num_shards))
    sample_shard_index = int(sample_shard_index)
    if sample_shard_index < 0 or sample_shard_index >= sample_num_shards:
        raise ValueError(
            f"--sample-shard-index must satisfy 0 <= index < num_shards, got {sample_shard_index}/{sample_num_shards}"
        )
    sample_files = discover_sample_files(samples_file)
    fc_map = load_file_content_map(sample_files)
    file_path_cache: dict[str, str | None] = {}
    candidates: list[dict[str, Any]] = []
    stats = Counter()
    stop_early = False
    if sample_selection == "even" and max_samples > 0:
        total_records = _count_jsonl_records(sample_files)
        stats["raw_records"] = total_records
        for oversample_factor in (2, 4, 8, 16, 32):
            global_target_count = min(
                total_records, max_samples * sample_num_shards * oversample_factor
            )
            global_selected = _take_evenly(
                list(range(total_records)), global_target_count
            )
            shard_selected = global_selected[sample_shard_index::sample_num_shards]
            selected_indices = set(shard_selected)
            target_count = len(selected_indices)
            stats["preselected_records"] = target_count
            stats["global_preselected_records"] = global_target_count
            stats["preselect_oversample_factor"] = oversample_factor
            candidates = []
            file_path_cache.clear()
            stats_for_pass = Counter()
            stats_for_pass["raw_records"] = total_records
            stats_for_pass["preselected_records"] = target_count
            stats_for_pass["global_preselected_records"] = global_target_count
            stats_for_pass["preselect_oversample_factor"] = oversample_factor
            iterator = _iter_selected_jsonl_records(sample_files, selected_indices)
            for global_index, _sample_file, local_index, sample in tqdm(
                iterator,
                total=target_count,
                desc=f"Collect rollout samples x{oversample_factor}",
                unit="rows",
                dynamic_ncols=True,
            ):
                stats_for_pass["read_samples"] += 1
                row = _prepare_rollout_sample(
                    sample,
                    index=local_index if len(sample_files) > 1 else global_index,
                    tokenizer=tokenizer,
                    steps=steps,
                    target_window_chars=target_window_chars,
                    fc_map=fc_map,
                    file_path_cache=file_path_cache,
                    stats=stats_for_pass,
                )
                if row is not None:
                    candidates.append(row)
            stats = stats_for_pass
            if len(candidates) >= max_samples or global_target_count >= total_records:
                break
        if len(candidates) > max_samples:
            selected = _take_evenly(list(range(len(candidates))), max_samples)
            candidates = [candidates[index] for index in selected]
        stats["selected_samples"] = len(candidates)
        return (
            candidates,
            {
                "sample_files": [str(path) for path in sample_files],
                "file_content_map_entries": len(fc_map),
                "file_path_cache_entries": len(file_path_cache),
                "sample_selection": sample_selection,
                "sample_shard_index": sample_shard_index,
                "sample_num_shards": sample_num_shards,
                "scan_stopped_early": False,
                "stats": dict(stats),
            },
        )
    for sample_file in sample_files:
        for index, sample in enumerate(read_jsonl_streaming(sample_file)):
            stats["read_samples"] += 1
            if (
                sample_num_shards > 1
                and (stats["read_samples"] - 1) % sample_num_shards
                != sample_shard_index
            ):
                continue
            row = _prepare_rollout_sample(
                sample,
                index=index,
                tokenizer=tokenizer,
                steps=steps,
                target_window_chars=target_window_chars,
                fc_map=fc_map,
                file_path_cache=file_path_cache,
                stats=stats,
            )
            if row is None:
                continue
            candidates.append(row)
            if (
                sample_selection == "first"
                and max_samples > 0
                and (len(candidates) >= max_samples)
            ):
                stop_early = True
                break
        if stop_early:
            break
    stats["selected_samples"] = len(candidates)
    return (
        candidates,
        {
            "sample_files": [str(path) for path in sample_files],
            "file_content_map_entries": len(fc_map),
            "file_path_cache_entries": len(file_path_cache),
            "sample_selection": sample_selection,
            "sample_shard_index": sample_shard_index,
            "sample_num_shards": sample_num_shards,
            "scan_stopped_early": stop_early,
            "stats": dict(stats),
        },
    )


def _empty_strategy_agg() -> dict[str, Any]:
    return {
        "samples": 0,
        "sequence_exact": 0,
        "sequence_prefix_compatible": 0,
        "changed_sequence": 0,
        "first_token_correct": 0,
        "first_token_changed": 0,
        "first_token_fix": 0,
        "first_token_damage": 0,
        "first_token_strict_evaluable": 0,
        "first_token_strict_correct": 0,
        "first_token_strict_changed": 0,
        "first_token_strict_fix": 0,
        "first_token_strict_damage": 0,
        "step_count": 0,
        "step_correct": 0,
        "step_changed": 0,
        "step_fix": 0,
        "step_damage": 0,
        "step_token_id_evaluable": 0,
        "step_buckets": [],
        "completion_metrics": _empty_metric_agg(),
        "python_validity": _empty_validity_agg(),
    }


def _text_prefix_compatible(pred_text: str, target_text: str) -> bool:
    if pred_text == target_text:
        return True
    if not pred_text or not target_text:
        return False
    return target_text.startswith(pred_text) or pred_text.startswith(target_text)


def _update_strategy_agg(
    agg: dict[str, Any],
    *,
    prediction_pieces: list[str],
    prediction_text: str,
    target_pieces: list[str],
    target_text: str,
    base_pieces: list[str],
    base_text: str,
    sample: dict[str, Any],
    enable_codebleu: bool,
    completion_metrics: dict[str, float] | None = None,
    prediction_ids: list[int | None] | None = None,
    target_ids: list[int] | None = None,
    base_ids: list[int | None] | None = None,
) -> None:
    agg["samples"] += 1
    sequence_exact = prediction_text == target_text
    sequence_prefix_compatible = _text_prefix_compatible(prediction_text, target_text)
    agg["sequence_exact"] += int(sequence_exact)
    agg["sequence_prefix_compatible"] += int(sequence_prefix_compatible)
    agg["changed_sequence"] += int(prediction_text != base_text)
    first_pred_piece = prediction_pieces[0] if prediction_pieces else ""
    first_base_piece = base_pieces[0] if base_pieces else ""
    first_correct = _text_prefix_compatible(first_pred_piece, target_text)
    base_first_correct = _text_prefix_compatible(first_base_piece, target_text)
    first_fix = not base_first_correct and first_correct
    first_damage = base_first_correct and (not first_correct)
    agg["first_token_correct"] += int(first_correct)
    agg["first_token_changed"] += int(first_pred_piece != first_base_piece)
    agg["first_token_fix"] += int(first_fix)
    agg["first_token_damage"] += int(first_damage)
    if prediction_ids and target_ids and base_ids:
        pred_id = prediction_ids[0]
        target_id = target_ids[0]
        base_id = base_ids[0]
        if pred_id is not None and base_id is not None:
            strict_correct = int(pred_id) == int(target_id)
            strict_base_correct = int(base_id) == int(target_id)
            agg["first_token_strict_evaluable"] += 1
            agg["first_token_strict_correct"] += int(strict_correct)
            agg["first_token_strict_changed"] += int(int(pred_id) != int(base_id))
            agg["first_token_strict_fix"] += int(
                not strict_base_correct and strict_correct
            )
            agg["first_token_strict_damage"] += int(
                strict_base_correct and (not strict_correct)
            )
    if completion_metrics is None:
        context = _build_completion_metric_context(target_text, sample)
        completion_metrics = _compute_completion_metrics_with_context(
            prediction_text, enable_codebleu=enable_codebleu, context=context
        )
    _update_metric_agg(agg["completion_metrics"], completion_metrics)
    suffix = str(sample.get("_rollout_suffix") or "")
    reconstructed = None
    if suffix.startswith(target_text):
        reconstructed = (
            str(sample.get("_rollout_prefix") or "")
            + prediction_text
            + suffix[len(target_text) :]
        )
    _update_validity_agg(
        agg["python_validity"], evaluate_python_validity(reconstructed)
    )


def _summarize_strategy_agg(agg: dict[str, Any]) -> dict[str, Any]:
    samples = int(agg["samples"])
    step_count = int(agg.get("step_count", 0))
    step_evaluable = int(agg.get("step_token_id_evaluable", 0))
    first_token_strict_evaluable = int(agg.get("first_token_strict_evaluable", 0))
    per_step = []
    for bucket in agg.get("step_buckets", []):
        bucket_evaluable = int(bucket.get("token_id_evaluable", 0))
        per_step.append(
            {
                "step": int(bucket.get("step", len(per_step))),
                "count": int(bucket.get("count", 0)),
                "token_id_evaluable": bucket_evaluable,
                "accuracy": _safe_rate(int(bucket.get("correct", 0)), bucket_evaluable),
                "changed_rate": _safe_rate(
                    int(bucket.get("changed", 0)), bucket_evaluable
                ),
                "fix_count": int(bucket.get("fix", 0)),
                "damage_count": int(bucket.get("damage", 0)),
                "net_gain": int(bucket.get("fix", 0)) - int(bucket.get("damage", 0)),
                "fix_rate": _safe_rate(int(bucket.get("fix", 0)), bucket_evaluable),
                "damage_rate": _safe_rate(
                    int(bucket.get("damage", 0)), bucket_evaluable
                ),
            }
        )
    return {
        "sample_count": samples,
        "metric_level": "text",
        "sequence_exact_count": int(agg["sequence_exact"]),
        "sequence_exact_rate": _safe_rate(int(agg["sequence_exact"]), samples),
        "sequence_prefix_compatible_count": int(agg["sequence_prefix_compatible"]),
        "sequence_prefix_compatible_rate": _safe_rate(
            int(agg["sequence_prefix_compatible"]), samples
        ),
        "changed_sequence_count": int(agg["changed_sequence"]),
        "changed_sequence_rate": _safe_rate(int(agg["changed_sequence"]), samples),
        "first_token_correct_count": int(agg["first_token_correct"]),
        "first_token_text_compatible_rate": _safe_rate(
            int(agg["first_token_correct"]), samples
        ),
        "first_token_changed_count": int(agg["first_token_changed"]),
        "first_token_changed_rate": _safe_rate(
            int(agg["first_token_changed"]), samples
        ),
        "first_token_fix_count": int(agg["first_token_fix"]),
        "first_token_damage_count": int(agg["first_token_damage"]),
        "first_token_net_gain": int(agg["first_token_fix"])
        - int(agg["first_token_damage"]),
        "first_token_match_definition": "text-prefix-compatible: first generated text equals, prefixes, or is prefixed by the target continuation text",
        "first_token_strict_evaluable_count": first_token_strict_evaluable,
        "first_token_strict_correct_count": int(
            agg.get("first_token_strict_correct", 0)
        ),
        "first_token_strict_accuracy": _safe_rate(
            int(agg.get("first_token_strict_correct", 0)), first_token_strict_evaluable
        ),
        "first_token_strict_changed_count": int(
            agg.get("first_token_strict_changed", 0)
        ),
        "first_token_strict_changed_rate": _safe_rate(
            int(agg.get("first_token_strict_changed", 0)), first_token_strict_evaluable
        ),
        "first_token_strict_fix_count": int(agg.get("first_token_strict_fix", 0)),
        "first_token_strict_damage_count": int(agg.get("first_token_strict_damage", 0)),
        "first_token_strict_net_gain": int(agg.get("first_token_strict_fix", 0))
        - int(agg.get("first_token_strict_damage", 0)),
        "first_token_strict_fix_rate": _safe_rate(
            int(agg.get("first_token_strict_fix", 0)), first_token_strict_evaluable
        ),
        "first_token_strict_damage_rate": _safe_rate(
            int(agg.get("first_token_strict_damage", 0)), first_token_strict_evaluable
        ),
        "first_token_strict_match_definition": "token-id exact match at step 0",
        "step_count": step_count,
        "step_token_id_evaluable": step_evaluable,
        "step_accuracy": _safe_rate(int(agg.get("step_correct", 0)), step_evaluable),
        "step_changed_rate": _safe_rate(
            int(agg.get("step_changed", 0)), step_evaluable
        ),
        "step_fix_count": int(agg.get("step_fix", 0)),
        "step_damage_count": int(agg.get("step_damage", 0)),
        "step_net_gain": int(agg.get("step_fix", 0)) - int(agg.get("step_damage", 0)),
        "step_fix_rate": _safe_rate(int(agg.get("step_fix", 0)), step_evaluable),
        "step_damage_rate": _safe_rate(int(agg.get("step_damage", 0)), step_evaluable),
        "step_metric": "token_id_strict",
        "per_step": per_step,
        "completion_metrics": _summarize_metric_agg(agg["completion_metrics"]),
        "python_validity": _summarize_validity_agg(agg["python_validity"]),
    }


def _step_bucket(agg: dict[str, Any], step_index: int) -> dict[str, Any]:
    buckets = agg.setdefault("step_buckets", [])
    while len(buckets) <= step_index:
        buckets.append(
            {
                "step": len(buckets),
                "count": 0,
                "token_id_evaluable": 0,
                "correct": 0,
                "changed": 0,
                "fix": 0,
                "damage": 0,
            }
        )
    return buckets[step_index]


def _update_step_agg(
    agg: dict[str, Any],
    *,
    prediction_ids: list[int | None],
    target_ids: list[int],
    base_ids: list[int | None],
) -> None:
    for pred_id, target_id, base_id in zip(
        prediction_ids, target_ids, base_ids, strict=False
    ):
        bucket = _step_bucket(agg, int(agg["step_count"]) % max(1, len(target_ids)))
        bucket["count"] += 1
        agg["step_count"] += 1
        if pred_id is None:
            continue
        agg["step_token_id_evaluable"] += 1
        bucket["token_id_evaluable"] += 1
        pred_correct = int(pred_id) == int(target_id)
        base_correct = base_id is not None and int(base_id) == int(target_id)
        agg["step_correct"] += int(pred_correct)
        agg["step_changed"] += int(base_id is not None and int(pred_id) != int(base_id))
        agg["step_fix"] += int(not base_correct and pred_correct)
        agg["step_damage"] += int(base_correct and (not pred_correct))
        bucket["correct"] += int(pred_correct)
        bucket["changed"] += int(base_id is not None and int(pred_id) != int(base_id))
        bucket["fix"] += int(not base_correct and pred_correct)
        bucket["damage"] += int(base_correct and (not pred_correct))


def _empty_repair_prediction_diag() -> dict[str, Any]:
    return {"count": 0, "step_buckets": []}


def _repair_prediction_diag_bucket(
    diag: dict[str, Any], step_index: int
) -> dict[str, Any]:
    buckets = diag.setdefault("step_buckets", [])
    while len(buckets) <= step_index:
        buckets.append(
            {
                "step": len(buckets),
                "count": 0,
                "prediction_evaluable": 0,
                "raw_prediction_evaluable": 0,
                "correct": 0,
                "fix": 0,
                "damage": 0,
                "raw_correct": 0,
                "raw_fix": 0,
                "raw_damage": 0,
                "repair_used": 0,
                "repair_used_prediction_evaluable": 0,
                "repair_used_base_correct": 0,
                "repair_used_correct": 0,
                "repair_used_fix": 0,
                "repair_used_damage": 0,
                "repair_used_raw_prediction_evaluable": 0,
                "repair_used_raw_correct": 0,
                "repair_used_raw_fix": 0,
                "repair_used_raw_damage": 0,
                "special_fallback": 0,
                "route_score_sum": 0.0,
                "route_score_count": 0,
                "base_margin_sum": 0.0,
                "base_margin_count": 0,
                "repair_confidence_sum": 0.0,
                "repair_confidence_count": 0,
            }
        )
    return buckets[step_index]


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _update_repair_prediction_diag(
    diag: dict[str, Any],
    *,
    repair_predictions: list[dict[str, Any]],
    target_ids: list[int],
    base_ids: list[int | None],
) -> None:
    for step_index, repair_prediction in enumerate(repair_predictions):
        if step_index >= len(target_ids) or step_index >= len(base_ids):
            break
        bucket = _repair_prediction_diag_bucket(diag, step_index)
        bucket["count"] += 1
        diag["count"] += 1
        route_score = _maybe_float(repair_prediction.get("route_score"))
        if route_score is not None:
            bucket["route_score_sum"] += route_score
            bucket["route_score_count"] += 1
        base_margin = _maybe_float(repair_prediction.get("base_margin"))
        if base_margin is not None:
            bucket["base_margin_sum"] += base_margin
            bucket["base_margin_count"] += 1
        repair_confidence = _maybe_float(repair_prediction.get("repair_confidence"))
        if repair_confidence is not None:
            bucket["repair_confidence_sum"] += repair_confidence
            bucket["repair_confidence_count"] += 1
        repair_used = bool(repair_prediction.get("repair_used"))
        bucket["repair_used"] += int(repair_used)
        bucket["special_fallback"] += int(
            repair_prediction.get("gate_reason") == "special_token_fallback"
        )
        target_id = int(target_ids[step_index])
        base_id = base_ids[step_index]
        base_correct = base_id is not None and int(base_id) == target_id
        if repair_used and base_id is not None:
            bucket["repair_used_base_correct"] += int(base_correct)
        pred_id = _maybe_int(repair_prediction.get("prediction_token_id"))
        if pred_id is not None:
            bucket["prediction_evaluable"] += 1
            pred_correct = pred_id == target_id
            bucket["correct"] += int(pred_correct)
            bucket["fix"] += int(not base_correct and pred_correct)
            bucket["damage"] += int(base_correct and (not pred_correct))
            if repair_used:
                bucket["repair_used_prediction_evaluable"] += 1
                bucket["repair_used_correct"] += int(pred_correct)
                bucket["repair_used_fix"] += int(not base_correct and pred_correct)
                bucket["repair_used_damage"] += int(base_correct and (not pred_correct))
        raw_id = _maybe_int(repair_prediction.get("raw_repair_token_id"))
        if raw_id is not None:
            bucket["raw_prediction_evaluable"] += 1
            raw_correct = raw_id == target_id
            bucket["raw_correct"] += int(raw_correct)
            bucket["raw_fix"] += int(not base_correct and raw_correct)
            bucket["raw_damage"] += int(base_correct and (not raw_correct))
            if repair_used:
                bucket["repair_used_raw_prediction_evaluable"] += 1
                bucket["repair_used_raw_correct"] += int(raw_correct)
                bucket["repair_used_raw_fix"] += int(not base_correct and raw_correct)
                bucket["repair_used_raw_damage"] += int(
                    base_correct and (not raw_correct)
                )


def _mean_from_bucket(bucket: dict[str, Any], sum_key: str, count_key: str) -> float:
    count = int(bucket.get(count_key, 0))
    if count <= 0:
        return 0.0
    return round(float(bucket.get(sum_key, 0.0)) / count, 4)


def _summarize_repair_prediction_diag(diag: dict[str, Any]) -> dict[str, Any]:
    per_step: list[dict[str, Any]] = []
    for bucket in diag.get("step_buckets", []):
        count = int(bucket.get("count", 0))
        pred_eval = int(bucket.get("prediction_evaluable", 0))
        raw_eval = int(bucket.get("raw_prediction_evaluable", 0))
        fix = int(bucket.get("fix", 0))
        damage = int(bucket.get("damage", 0))
        raw_fix = int(bucket.get("raw_fix", 0))
        raw_damage = int(bucket.get("raw_damage", 0))
        repair_used = int(bucket.get("repair_used", 0))
        used_eval = int(bucket.get("repair_used_prediction_evaluable", 0))
        used_fix = int(bucket.get("repair_used_fix", 0))
        used_damage = int(bucket.get("repair_used_damage", 0))
        used_raw_eval = int(bucket.get("repair_used_raw_prediction_evaluable", 0))
        used_raw_fix = int(bucket.get("repair_used_raw_fix", 0))
        used_raw_damage = int(bucket.get("repair_used_raw_damage", 0))
        per_step.append(
            {
                "step": int(bucket.get("step", len(per_step))),
                "count": count,
                "prediction_evaluable": pred_eval,
                "repair_used": repair_used,
                "repair_used_rate": _safe_rate(repair_used, count),
                "special_fallback_rate": _safe_rate(
                    int(bucket.get("special_fallback", 0)), count
                ),
                "mean_route_score": _mean_from_bucket(
                    bucket, "route_score_sum", "route_score_count"
                ),
                "mean_base_margin": _mean_from_bucket(
                    bucket, "base_margin_sum", "base_margin_count"
                ),
                "mean_repair_confidence": _mean_from_bucket(
                    bucket, "repair_confidence_sum", "repair_confidence_count"
                ),
                "accuracy": _safe_rate(int(bucket.get("correct", 0)), pred_eval),
                "fix_rate": _safe_rate(fix, pred_eval),
                "damage_rate": _safe_rate(damage, pred_eval),
                "net_gain": fix - damage,
                "net_rate": _safe_rate(fix - damage, pred_eval),
                "raw_accuracy": _safe_rate(int(bucket.get("raw_correct", 0)), raw_eval),
                "raw_fix_rate": _safe_rate(raw_fix, raw_eval),
                "raw_damage_rate": _safe_rate(raw_damage, raw_eval),
                "raw_net_gain": raw_fix - raw_damage,
                "raw_net_rate": _safe_rate(raw_fix - raw_damage, raw_eval),
                "repair_used_only": {
                    "count": repair_used,
                    "prediction_evaluable": used_eval,
                    "base_accuracy": _safe_rate(
                        int(bucket.get("repair_used_base_correct", 0)), repair_used
                    ),
                    "repair_accuracy": _safe_rate(
                        int(bucket.get("repair_used_correct", 0)), used_eval
                    ),
                    "fix_rate": _safe_rate(used_fix, used_eval),
                    "damage_rate": _safe_rate(used_damage, used_eval),
                    "net_gain": used_fix - used_damage,
                    "net_rate": _safe_rate(used_fix - used_damage, used_eval),
                    "raw_repair_accuracy": _safe_rate(
                        int(bucket.get("repair_used_raw_correct", 0)), used_raw_eval
                    ),
                    "raw_fix_rate": _safe_rate(used_raw_fix, used_raw_eval),
                    "raw_damage_rate": _safe_rate(used_raw_damage, used_raw_eval),
                    "raw_net_gain": used_raw_fix - used_raw_damage,
                    "raw_net_rate": _safe_rate(
                        used_raw_fix - used_raw_damage, used_raw_eval
                    ),
                },
            }
        )
    used_total = sum(
        (int(bucket.get("repair_used", 0)) for bucket in diag.get("step_buckets", []))
    )
    used_eval_total = sum(
        (
            int(bucket.get("repair_used_prediction_evaluable", 0))
            for bucket in diag.get("step_buckets", [])
        )
    )
    used_fix_total = sum(
        (
            int(bucket.get("repair_used_fix", 0))
            for bucket in diag.get("step_buckets", [])
        )
    )
    used_damage_total = sum(
        (
            int(bucket.get("repair_used_damage", 0))
            for bucket in diag.get("step_buckets", [])
        )
    )
    used_raw_eval_total = sum(
        (
            int(bucket.get("repair_used_raw_prediction_evaluable", 0))
            for bucket in diag.get("step_buckets", [])
        )
    )
    used_raw_fix_total = sum(
        (
            int(bucket.get("repair_used_raw_fix", 0))
            for bucket in diag.get("step_buckets", [])
        )
    )
    used_raw_damage_total = sum(
        (
            int(bucket.get("repair_used_raw_damage", 0))
            for bucket in diag.get("step_buckets", [])
        )
    )
    total_count = int(diag.get("count", 0))
    return {
        "count": total_count,
        "repair_used_only": {
            "count": used_total,
            "rate": _safe_rate(used_total, total_count),
            "prediction_evaluable": used_eval_total,
            "base_accuracy": _safe_rate(
                sum(
                    (
                        int(bucket.get("repair_used_base_correct", 0))
                        for bucket in diag.get("step_buckets", [])
                    )
                ),
                used_total,
            ),
            "repair_accuracy": _safe_rate(
                sum(
                    (
                        int(bucket.get("repair_used_correct", 0))
                        for bucket in diag.get("step_buckets", [])
                    )
                ),
                used_eval_total,
            ),
            "fix_rate": _safe_rate(used_fix_total, used_eval_total),
            "damage_rate": _safe_rate(used_damage_total, used_eval_total),
            "net_gain": used_fix_total - used_damage_total,
            "net_rate": _safe_rate(used_fix_total - used_damage_total, used_eval_total),
            "raw_repair_accuracy": _safe_rate(
                sum(
                    (
                        int(bucket.get("repair_used_raw_correct", 0))
                        for bucket in diag.get("step_buckets", [])
                    )
                ),
                used_raw_eval_total,
            ),
            "raw_fix_rate": _safe_rate(used_raw_fix_total, used_raw_eval_total),
            "raw_damage_rate": _safe_rate(used_raw_damage_total, used_raw_eval_total),
            "raw_net_gain": used_raw_fix_total - used_raw_damage_total,
            "raw_net_rate": _safe_rate(
                used_raw_fix_total - used_raw_damage_total, used_raw_eval_total
            ),
        },
        "per_step": per_step,
    }


def _empty_first_step_group_agg() -> dict[str, Any]:
    return {
        "samples": 0,
        "first_base_correct": 0,
        "first_repair_correct": 0,
        "first_changed": 0,
        "continuation_steps": 0,
        "base_continuation_correct": 0,
        "repair_continuation_correct": 0,
        "continuation_fix": 0,
        "continuation_damage": 0,
        "sequence_exact_base": 0,
        "sequence_exact_repair": 0,
        "sequence_prefix_base": 0,
        "sequence_prefix_repair": 0,
        "changed_sequence": 0,
    }


def _first_step_group_name(
    *, base_ids: list[int | None], repair_ids: list[int | None], target_ids: list[int]
) -> str:
    if not target_ids or not base_ids or (not repair_ids):
        return "missing_first_token"
    target_id = int(target_ids[0])
    base_id = base_ids[0]
    repair_id = repair_ids[0]
    if base_id is None or repair_id is None:
        return "missing_first_token"
    base_correct = int(base_id) == target_id
    repair_correct = int(repair_id) == target_id
    if base_correct and repair_correct:
        return "both_correct"
    if not base_correct and repair_correct:
        return "repair_fix"
    if base_correct and (not repair_correct):
        return "repair_damage"
    if int(base_id) == int(repair_id):
        return "both_wrong_same"
    return "both_wrong_changed"


def _update_first_step_group_agg(
    groups: dict[str, dict[str, Any]],
    *,
    base_ids: list[int | None],
    repair_ids: list[int | None],
    target_ids: list[int],
    base_text: str,
    repair_text: str,
    target_text: str,
) -> None:
    group_name = _first_step_group_name(
        base_ids=base_ids, repair_ids=repair_ids, target_ids=target_ids
    )
    agg = groups.setdefault(group_name, _empty_first_step_group_agg())
    agg["samples"] += 1
    if target_ids and base_ids and repair_ids:
        target_id = int(target_ids[0])
        base_id = base_ids[0]
        repair_id = repair_ids[0]
        base_correct = base_id is not None and int(base_id) == target_id
        repair_correct = repair_id is not None and int(repair_id) == target_id
        agg["first_base_correct"] += int(base_correct)
        agg["first_repair_correct"] += int(repair_correct)
        agg["first_changed"] += int(
            base_id is not None
            and repair_id is not None
            and (int(base_id) != int(repair_id))
        )
    for base_id, repair_id, target_id in zip(
        base_ids[1:], repair_ids[1:], target_ids[1:], strict=False
    ):
        if base_id is None or repair_id is None:
            continue
        agg["continuation_steps"] += 1
        base_correct = int(base_id) == int(target_id)
        repair_correct = int(repair_id) == int(target_id)
        agg["base_continuation_correct"] += int(base_correct)
        agg["repair_continuation_correct"] += int(repair_correct)
        agg["continuation_fix"] += int(not base_correct and repair_correct)
        agg["continuation_damage"] += int(base_correct and (not repair_correct))
    base_exact = base_text == target_text
    repair_exact = repair_text == target_text
    base_prefix = _text_prefix_compatible(base_text, target_text)
    repair_prefix = _text_prefix_compatible(repair_text, target_text)
    agg["sequence_exact_base"] += int(base_exact)
    agg["sequence_exact_repair"] += int(repair_exact)
    agg["sequence_prefix_base"] += int(base_prefix)
    agg["sequence_prefix_repair"] += int(repair_prefix)
    agg["changed_sequence"] += int(base_text != repair_text)


def _summarize_first_step_groups(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered_names = [
        "repair_fix",
        "repair_damage",
        "both_correct",
        "both_wrong_changed",
        "both_wrong_same",
        "missing_first_token",
    ]
    ordered_names.extend((name for name in sorted(groups) if name not in ordered_names))
    summary: dict[str, Any] = {}
    for name in ordered_names:
        if name not in groups:
            continue
        agg = groups[name]
        samples = int(agg.get("samples", 0))
        continuation_steps = int(agg.get("continuation_steps", 0))
        cont_fix = int(agg.get("continuation_fix", 0))
        cont_damage = int(agg.get("continuation_damage", 0))
        summary[name] = {
            "sample_count": samples,
            "sample_rate": _safe_rate(
                samples, sum((int(g.get("samples", 0)) for g in groups.values()))
            ),
            "first_base_accuracy": _safe_rate(
                int(agg.get("first_base_correct", 0)), samples
            ),
            "first_repair_accuracy": _safe_rate(
                int(agg.get("first_repair_correct", 0)), samples
            ),
            "first_changed_rate": _safe_rate(int(agg.get("first_changed", 0)), samples),
            "continuation_step_count": continuation_steps,
            "base_continuation_accuracy": _safe_rate(
                int(agg.get("base_continuation_correct", 0)), continuation_steps
            ),
            "repair_continuation_accuracy": _safe_rate(
                int(agg.get("repair_continuation_correct", 0)), continuation_steps
            ),
            "continuation_fix_count": cont_fix,
            "continuation_damage_count": cont_damage,
            "continuation_net_gain": cont_fix - cont_damage,
            "continuation_fix_rate": _safe_rate(cont_fix, continuation_steps),
            "continuation_damage_rate": _safe_rate(cont_damage, continuation_steps),
            "base_sequence_exact_rate": _safe_rate(
                int(agg.get("sequence_exact_base", 0)), samples
            ),
            "repair_sequence_exact_rate": _safe_rate(
                int(agg.get("sequence_exact_repair", 0)), samples
            ),
            "base_sequence_prefix_rate": _safe_rate(
                int(agg.get("sequence_prefix_base", 0)), samples
            ),
            "repair_sequence_prefix_rate": _safe_rate(
                int(agg.get("sequence_prefix_repair", 0)), samples
            ),
            "changed_sequence_rate": _safe_rate(
                int(agg.get("changed_sequence", 0)), samples
            ),
        }
    return summary


def _empty_target_type_group_agg() -> dict[str, int]:
    return {
        "steps": 0,
        "base_correct": 0,
        "repair_correct": 0,
        "changed": 0,
        "fix": 0,
        "damage": 0,
        "first_steps": 0,
        "first_base_correct": 0,
        "first_repair_correct": 0,
    }


def _update_target_type_groups(
    groups: dict[str, dict[str, int]],
    *,
    prefix: str,
    target_pieces: list[str],
    target_ids: list[int],
    base_ids: list[int | None],
    repair_ids: list[int | None],
) -> None:
    gold_prefix = str(prefix)
    for step_idx, (target_piece, target_id, base_id, repair_id) in enumerate(
        zip(target_pieces, target_ids, base_ids, repair_ids, strict=False)
    ):
        target_is_name = _last_added_python_name_piece(gold_prefix, str(target_piece))
        group_name = "target_python_name" if target_is_name else "target_non_name"
        agg = groups.setdefault(group_name, _empty_target_type_group_agg())
        gold_prefix += str(target_piece)
        if base_id is None or repair_id is None:
            continue
        agg["steps"] += 1
        base_correct = int(base_id) == int(target_id)
        repair_correct = int(repair_id) == int(target_id)
        changed = int(base_id) != int(repair_id)
        agg["base_correct"] += int(base_correct)
        agg["repair_correct"] += int(repair_correct)
        agg["changed"] += int(changed)
        agg["fix"] += int(not base_correct and repair_correct)
        agg["damage"] += int(base_correct and (not repair_correct))
        if step_idx == 0:
            agg["first_steps"] += 1
            agg["first_base_correct"] += int(base_correct)
            agg["first_repair_correct"] += int(repair_correct)


def _summarize_target_type_groups(groups: dict[str, dict[str, int]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    total_steps = sum((int(group.get("steps", 0)) for group in groups.values()))
    for group_name in ("target_python_name", "target_non_name"):
        if group_name not in groups:
            continue
        agg = groups[group_name]
        steps = int(agg.get("steps", 0))
        first_steps = int(agg.get("first_steps", 0))
        fix = int(agg.get("fix", 0))
        damage = int(agg.get("damage", 0))
        summary[group_name] = {
            "step_count": steps,
            "step_rate": _safe_rate(steps, total_steps),
            "base_accuracy": _safe_rate(int(agg.get("base_correct", 0)), steps),
            "repair_accuracy": _safe_rate(int(agg.get("repair_correct", 0)), steps),
            "changed_rate": _safe_rate(int(agg.get("changed", 0)), steps),
            "fix_count": fix,
            "damage_count": damage,
            "net_gain": fix - damage,
            "fix_rate": _safe_rate(fix, steps),
            "damage_rate": _safe_rate(damage, steps),
            "first_step_count": first_steps,
            "first_base_accuracy": _safe_rate(
                int(agg.get("first_base_correct", 0)), first_steps
            ),
            "first_repair_accuracy": _safe_rate(
                int(agg.get("first_repair_correct", 0)), first_steps
            ),
        }
    return summary


def _evaluate(
    *,
    samples: list[dict[str, Any]],
    checkpoint_path: Path,
    base_model_name: str,
    device: str,
    steps: int,
    rollout_batch_size: int,
    max_context_tokens: int,
    enable_codebleu: bool,
    schema_manifest: str | Path | None,
    save_examples: int,
    report_horizons: list[int] | None = None,
) -> dict[str, Any]:
    horizons = sorted(set((int(value) for value in report_horizons or [steps])))
    if not horizons or any((value <= 0 or value > steps for value in horizons)):
        raise ValueError(
            f"report horizons must be positive and no larger than --generation-steps={steps}: {horizons}"
        )
    if steps not in horizons:
        raise ValueError(
            "--report-horizons must include --generation-steps so the backward-compatible top-level report is defined"
        )
    (repair_moe, checkpoint) = load_editor(checkpoint_path, device)
    hidden_state_interface = str(checkpoint.get("hidden_state_interface", "post_norm"))
    (route_schema, code_schema) = _resolve_repair_feature_schema(
        checkpoint=checkpoint, manifest_path=schema_manifest
    )
    evaluator = OfflineEvaluator(
        long_sequence_rollout_model_name=base_model_name,
        long_sequence_rollout_device=device,
        long_sequence_rollout_batch_size=rollout_batch_size,
    )
    (llm, rollout_model_name) = evaluator._get_long_sequence_rollout_llm(samples)
    tokenizer = llm.tokenizer
    (norm, lm_head) = _resolve_norm_and_lm_head(llm)
    aggs_by_horizon = {
        horizon: {
            "base": _empty_strategy_agg(),
            "bp_hsr_first_token_then_base": _empty_strategy_agg(),
        }
        for horizon in horizons
    }
    first_step_groups: dict[str, dict[str, Any]] = {}
    target_type_groups: dict[str, dict[str, int]] = {}
    repair_prediction_diag = _empty_repair_prediction_diag()
    examples: list[dict[str, Any]] = []
    repair_used_count = 0
    repair_special_fallback_count = 0
    batch_size = max(1, int(rollout_batch_size))
    progress = tqdm(
        total=len(samples),
        desc="First-token BP-HSR rollout",
        unit="samples",
        dynamic_ncols=True,
    )
    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start : batch_start + batch_size]
        prefixes = [str(sample["_rollout_prefix"]) for sample in batch_samples]
        with torch.inference_mode():
            initial_features_by_prefix = _extract_runtime_features_batch(
                evaluator=evaluator,
                tokenizer=tokenizer,
                samples=batch_samples,
                prefixes=prefixes,
                max_context_tokens=max_context_tokens,
            )
            runtime_samples = [
                _runtime_sample_from_features(
                    sample,
                    prefix=prefix,
                    features=initial_features_by_prefix[prefix],
                    generated_prefix="",
                )
                for (sample, prefix) in zip(batch_samples, prefixes, strict=True)
            ]
            base_initial_predictions = [
                _base_next_token_from_features(runtime_sample)
                for runtime_sample in runtime_samples
            ]
            repair_predictions = _predict_repair_tokens_batch(
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
        per_sample_state: list[dict[str, Any]] = []
        for sample, prefix, base_initial, repair_prediction in zip(
            batch_samples,
            prefixes,
            base_initial_predictions,
            repair_predictions,
            strict=True,
        ):
            (base_token_id, base_token_text) = base_initial
            bp_token_text = str(repair_prediction.get("prediction") or "")
            bp_token_id_raw = repair_prediction.get("prediction_token_id")
            bp_token_id = int(bp_token_id_raw) if bp_token_id_raw is not None else None
            repair_used = bool(repair_prediction.get("repair_used"))
            repair_used_count += int(repair_used)
            repair_special_fallback_count += int(
                repair_prediction.get("gate_reason") == "special_token_fallback"
            )
            initial_context_ids = _encode_text(tokenizer, prefix)
            if not initial_context_ids:
                initial_context_ids = _fallback_context_ids(tokenizer)
            base_context_ids = list(initial_context_ids)
            bp_context_ids = list(initial_context_ids)
            (base_token_text, base_context_ids) = _append_token_text(
                tokenizer,
                base_context_ids,
                base_token_id,
                fallback_text=base_token_text,
            )
            (bp_token_text, bp_context_ids) = _append_token_text(
                tokenizer, bp_context_ids, bp_token_id, fallback_text=bp_token_text
            )
            state = {
                "sample": sample,
                "prefix": prefix,
                "repair_prediction": repair_prediction,
                "repair_used": repair_used,
                "base_context_ids": base_context_ids,
                "bp_context_ids": bp_context_ids,
                "base_ids": [base_token_id],
                "bp_ids": [bp_token_id],
                "base_pieces": [base_token_text],
                "bp_pieces": [bp_token_text],
            }
            per_sample_state.append(state)
        remaining_steps = max(0, steps - 1)
        if remaining_steps > 0:
            base_context_rows = [
                [int(token_id) for token_id in state["base_context_ids"]]
                for state in per_sample_state
            ]
            bp_context_rows = [
                [int(token_id) for token_id in state["bp_context_ids"]]
                for state in per_sample_state
            ]
            continuation_ids = _greedy_generate_token_ids_from_context_batch(
                llm,
                [*base_context_rows, *bp_context_rows],
                max_new_tokens=remaining_steps,
                max_length=max_context_tokens,
            )
            split = len(base_context_rows)
            base_continuations = continuation_ids[:split]
            bp_continuations = continuation_ids[split:]
            for state, base_ids_next, bp_ids_next in zip(
                per_sample_state, base_continuations, bp_continuations, strict=True
            ):
                base_context_ids = state["base_context_ids"]
                bp_context_ids = state["bp_context_ids"]
                for base_next_id, bp_next_id in zip(
                    base_ids_next, bp_ids_next, strict=True
                ):
                    base_token_id = int(base_next_id)
                    bp_token_id = int(bp_next_id)
                    (base_token_text, base_context_ids) = _append_token_text(
                        tokenizer,
                        base_context_ids,
                        base_token_id,
                        fallback_text=_decode_single_token(tokenizer, base_token_id),
                    )
                    (bp_token_text, bp_context_ids) = _append_token_text(
                        tokenizer,
                        bp_context_ids,
                        bp_token_id,
                        fallback_text=_decode_single_token(tokenizer, bp_token_id),
                    )
                    state["base_ids"].append(base_token_id)
                    state["base_pieces"].append(base_token_text)
                    state["bp_ids"].append(bp_token_id)
                    state["bp_pieces"].append(bp_token_text)
                state["base_context_ids"] = base_context_ids
                state["bp_context_ids"] = bp_context_ids
        for state in per_sample_state:
            sample = state["sample"]
            prefix = state["prefix"]
            target_ids = [
                int(token_id) for token_id in sample["_rollout_target_token_ids"]
            ]
            target_pieces = [
                str(piece) for piece in sample["_rollout_target_token_pieces"]
            ]
            base_ids = state["base_ids"]
            bp_ids = state["bp_ids"]
            base_pieces = state["base_pieces"]
            bp_pieces = state["bp_pieces"]
            repair_prediction = state["repair_prediction"]
            repair_used = bool(state["repair_used"])
            base_text = "".join(base_pieces)
            bp_text = "".join(bp_pieces)
            target_text = str(sample["_rollout_target_text"])
            suffix = str(sample.get("_rollout_suffix") or "")
            for horizon in horizons:
                horizon_target_ids = target_ids[:horizon]
                horizon_target_pieces = target_pieces[:horizon]
                horizon_target_text = "".join(horizon_target_pieces)
                if not horizon_target_text or not suffix.startswith(
                    horizon_target_text
                ):
                    horizon_target_text = _decode_ids(tokenizer, horizon_target_ids)
                horizon_base_ids = base_ids[:horizon]
                horizon_bp_ids = bp_ids[:horizon]
                horizon_base_pieces = base_pieces[:horizon]
                horizon_bp_pieces = bp_pieces[:horizon]
                horizon_base_text = "".join(horizon_base_pieces)
                horizon_bp_text = "".join(horizon_bp_pieces)
                horizon_aggs = aggs_by_horizon[horizon]
                _update_strategy_agg(
                    horizon_aggs["base"],
                    prediction_pieces=horizon_base_pieces,
                    prediction_text=horizon_base_text,
                    target_pieces=horizon_target_pieces,
                    target_text=horizon_target_text,
                    base_pieces=horizon_base_pieces,
                    base_text=horizon_base_text,
                    sample=sample,
                    enable_codebleu=enable_codebleu,
                    prediction_ids=horizon_base_ids,
                    target_ids=horizon_target_ids,
                    base_ids=horizon_base_ids,
                )
                _update_step_agg(
                    horizon_aggs["base"],
                    prediction_ids=horizon_base_ids,
                    target_ids=horizon_target_ids,
                    base_ids=horizon_base_ids,
                )
                _update_strategy_agg(
                    horizon_aggs["bp_hsr_first_token_then_base"],
                    prediction_pieces=horizon_bp_pieces,
                    prediction_text=horizon_bp_text,
                    target_pieces=horizon_target_pieces,
                    target_text=horizon_target_text,
                    base_pieces=horizon_base_pieces,
                    base_text=horizon_base_text,
                    sample=sample,
                    enable_codebleu=enable_codebleu,
                    prediction_ids=horizon_bp_ids,
                    target_ids=horizon_target_ids,
                    base_ids=horizon_base_ids,
                )
                _update_step_agg(
                    horizon_aggs["bp_hsr_first_token_then_base"],
                    prediction_ids=horizon_bp_ids,
                    target_ids=horizon_target_ids,
                    base_ids=horizon_base_ids,
                )
            _update_first_step_group_agg(
                first_step_groups,
                base_ids=base_ids,
                repair_ids=bp_ids,
                target_ids=target_ids,
                base_text=base_text,
                repair_text=bp_text,
                target_text=target_text,
            )
            _update_target_type_groups(
                target_type_groups,
                prefix=prefix,
                target_pieces=target_pieces,
                target_ids=target_ids,
                base_ids=base_ids,
                repair_ids=bp_ids,
            )
            _update_repair_prediction_diag(
                repair_prediction_diag,
                repair_predictions=[repair_prediction],
                target_ids=target_ids,
                base_ids=base_ids,
            )
            if len(examples) < save_examples:
                examples.append(
                    {
                        "sample_id": sample.get("_rollout_sample_id"),
                        "prefix_tail": prefix[-160:],
                        "target_text": target_text,
                        "base_text": base_text,
                        "bp_hsr_text": bp_text,
                        "target_token_ids": target_ids,
                        "base_token_ids": base_ids,
                        "bp_hsr_token_ids": bp_ids,
                        "first_token": {
                            "target": {
                                "token_id": target_ids[0],
                                "text": _decode_single_token(tokenizer, target_ids[0]),
                            },
                            "base": {"token_id": base_ids[0], "text": base_pieces[0]},
                            "bp_hsr": {
                                "token_id": bp_ids[0],
                                "text": bp_pieces[0],
                                "raw_repair_token_id": repair_prediction.get(
                                    "raw_repair_token_id"
                                ),
                                "raw_repair_text": repair_prediction.get(
                                    "raw_repair_prediction"
                                ),
                                "repair_used": repair_used,
                                "gate_reason": repair_prediction.get("gate_reason"),
                                "route_score": repair_prediction.get("route_score"),
                                "base_margin": repair_prediction.get("base_margin"),
                                "repair_confidence": repair_prediction.get(
                                    "repair_confidence"
                                ),
                            },
                        },
                    }
                )
        progress.update(len(batch_samples))
    progress.close()
    horizon_reports: dict[str, Any] = {}
    for horizon in horizons:
        horizon_strategies = {
            key: _summarize_strategy_agg(agg)
            for (key, agg) in aggs_by_horizon[horizon].items()
        }
        horizon_base_metrics = horizon_strategies["base"]["completion_metrics"]
        horizon_bp_metrics = horizon_strategies["bp_hsr_first_token_then_base"][
            "completion_metrics"
        ]
        horizon_completion_metrics = {
            "available_optional_metrics": {"codebleu": bool(enable_codebleu)},
            "base": horizon_base_metrics,
            "bp_hsr_first_token_then_base": horizon_bp_metrics,
            "bp_hsr_first_token_then_base_minus_base": _metric_deltas(
                horizon_bp_metrics, horizon_base_metrics
            ),
        }
        horizon_python_validity = {
            "base": horizon_strategies["base"]["python_validity"],
            "bp_hsr_first_token_then_base": horizon_strategies[
                "bp_hsr_first_token_then_base"
            ]["python_validity"],
        }
        horizon_python_validity["bp_hsr_first_token_then_base_minus_base"] = {
            key: round(
                float(
                    horizon_python_validity["bp_hsr_first_token_then_base"].get(
                        key, 0.0
                    )
                )
                - float(horizon_python_validity["base"].get(key, 0.0)),
                4,
            )
            for key in (
                "ast_parse_rate",
                "compile_success_rate",
                "indentation_error_rate",
            )
        }
        horizon_reports[str(horizon)] = {
            "generation_steps": horizon,
            "processed_sample_count": len(samples),
            "strategies": horizon_strategies,
            "completion_metrics": horizon_completion_metrics,
            "python_validity": horizon_python_validity,
        }
    max_horizon_report = horizon_reports[str(steps)]
    strategies = max_horizon_report["strategies"]
    completion_metrics = max_horizon_report["completion_metrics"]
    python_validity = max_horizon_report["python_validity"]
    return {
        "enabled": True,
        "mode": "first_token_bp_hsr_then_base",
        "method_name": "BP-HSR first-token rollout",
        "repair_strategy_key": "bp_hsr_first_token_then_base",
        "model_name": rollout_model_name,
        "generation_steps": steps,
        "report_horizons": horizons,
        "horizon_reports": horizon_reports,
        "max_context_tokens": int(max_context_tokens),
        "processed_sample_count": len(samples),
        "total_generated_tokens": len(samples) * steps,
        "runtime_context": "token_id_context_after_initial_prefix_no_gt_suffix",
        "runtime_target_features": False,
        "metric_level": "text_and_token_step",
        "token_ids": "strict_step_metrics",
        "first_token_policy": {
            "bp_hsr_applied_steps": [0],
            "base_continuation_steps": [1, steps - 1] if steps > 1 else [],
            "repair_used_count": repair_used_count,
            "repair_used_rate": _safe_rate(repair_used_count, len(samples)),
            "special_fallback_count": repair_special_fallback_count,
        },
        "first_step_outcome_groups": _summarize_first_step_groups(first_step_groups),
        "target_token_type_groups": _summarize_target_type_groups(target_type_groups),
        "repair_prediction_diagnostics": _summarize_repair_prediction_diag(
            repair_prediction_diag
        ),
        "strategies": strategies,
        "completion_metrics": completion_metrics,
        "python_validity": python_validity,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--schema-manifest")
    parser.add_argument("--generation-steps", type=int, default=32)
    parser.add_argument(
        "--report-horizons",
        default="",
        help="Comma-separated prefix horizons to summarize from one generated trajectory. Supported for first_token; the maximum value must equal --generation-steps. Empty reports only --generation-steps.",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--sample-selection", choices=("first", "even"), default="first"
    )
    parser.add_argument("--sample-shard-index", type=int, default=0)
    parser.add_argument("--sample-num-shards", type=int, default=1)
    parser.add_argument("--target-window-chars", type=int, default=1024)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--save-examples", type=int, default=0)
    codebleu_group = parser.add_mutually_exclusive_group()
    codebleu_group.add_argument(
        "--enable-codebleu", dest="enable_codebleu", action="store_true", default=True
    )
    codebleu_group.add_argument(
        "--disable-codebleu", dest="enable_codebleu", action="store_false"
    )
    args = parser.parse_args()
    if args.generation_steps <= 0:
        raise ValueError("--generation-steps must be positive")
    if args.enable_codebleu and _calc_codebleu is None:
        raise RuntimeError(
            "--enable-codebleu was requested, but the codebleu package is not available in this Python environment"
        )
    report_horizons = [
        int(value.strip())
        for value in str(args.report_horizons).split(",")
        if value.strip()
    ] or [int(args.generation_steps)]
    (_, checkpoint) = load_editor(args.checkpoint, "cpu")
    base_model_name = args.base_model or str(checkpoint.get("base_model") or "")
    if not base_model_name:
        raise ValueError("--base-model is required when checkpoint lacks base_model")
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
    assert_tokenizer_context_policy(tokenizer, context="first-token rollout tokenizer")
    (samples, input_metadata) = _collect_samples(
        args.samples_file,
        tokenizer=tokenizer,
        steps=int(args.generation_steps),
        target_window_chars=int(args.target_window_chars),
        max_samples=int(args.max_samples),
        sample_selection=str(args.sample_selection),
        sample_shard_index=int(args.sample_shard_index),
        sample_num_shards=int(args.sample_num_shards),
    )
    report = _evaluate(
        samples=samples,
        checkpoint_path=args.checkpoint,
        base_model_name=base_model_name,
        device=args.device,
        steps=int(args.generation_steps),
        rollout_batch_size=int(args.rollout_batch_size),
        max_context_tokens=int(args.max_context_tokens),
        enable_codebleu=bool(args.enable_codebleu),
        schema_manifest=args.schema_manifest,
        save_examples=int(args.save_examples),
        report_horizons=report_horizons,
    )
    report["input"] = {
        "samples_file": str(args.samples_file),
        "checkpoint": str(args.checkpoint),
        "schema_manifest": str(args.schema_manifest or ""),
        "max_samples": int(args.max_samples),
        "sample_selection": str(args.sample_selection),
        "sample_shard_index": int(args.sample_shard_index),
        "sample_num_shards": int(args.sample_num_shards),
        "target_window_chars": int(args.target_window_chars),
        **input_metadata,
    }
    report["config"] = {
        "base_model": base_model_name,
        "device": args.device,
        "generation_steps": int(args.generation_steps),
        "report_horizons": report_horizons,
        "rollout_batch_size": int(args.rollout_batch_size),
        "max_context_tokens": int(args.max_context_tokens),
        "tokenizer_context_policy": tokenizer_context_policy_metadata(
            tokenizer, max_length=int(args.max_context_tokens)
        ),
        "enable_codebleu": bool(args.enable_codebleu),
        "neuron_risk_config": os.environ.get("SRLCC_NEURON_RISK_CONFIG"),
    }
    report["checkpoint"] = {
        "variant": checkpoint.get("variant", "full"),
        "method_name": checkpoint.get("method_name"),
        "feature_mode": checkpoint.get("feature_mode", "full"),
        "hidden_dim": int(checkpoint.get("hidden_dim", 0)),
        "route_dim": int(checkpoint.get("route_dim", 0)),
        "code_dim": int(checkpoint.get("code_dim", 0)),
        "num_experts": int(checkpoint.get("num_experts", 0)),
        "rank": int(checkpoint.get("rank", 0)),
        "gate_top_k": int(checkpoint.get("gate_top_k", 2)),
        "expert_activation": checkpoint.get("expert_activation", "gelu"),
        "fixed_alpha": bool(checkpoint.get("fixed_alpha", False)),
        "loss_config": checkpoint.get("loss_config", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, args.output)
    repair_key = str(report.get("repair_strategy_key") or "")
    base_summary = report.get("strategies", {}).get("base", {})
    repair_summary = report.get("strategies", {}).get(repair_key, {})
    delta_summary = report.get("completion_metrics", {}).get(
        f"{repair_key}_minus_base", {}
    )
    stdout_summary = {
        "output": str(args.output),
        "method_name": report.get("method_name"),
        "processed_sample_count": report.get("processed_sample_count"),
        "generation_steps": report.get("generation_steps"),
        "repair_strategy_key": repair_key,
        "base_first_token_accuracy": base_summary.get(
            "first_token_text_compatible_rate"
        ),
        "repair_first_token_accuracy": repair_summary.get(
            "first_token_text_compatible_rate"
        ),
        "repair_used_rate": (
            report.get("every_step_policy", {})
            or report.get("first_token_policy", {})
            or report.get("base_prefix_probe_policy", {})
        ).get("repair_used_rate"),
        "completion_delta": {
            "em": delta_summary.get("em"),
            "codebleu": delta_summary.get("codebleu"),
            "tok_f1": delta_summary.get("tok_f1"),
        },
    }
    print(json.dumps(stdout_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from ramp_hsr.checkpoints import load_editor
from scripts._model_io import _load_model, _model_dtype, _pad_batch
from ramp_hsr.identifiers import (
    EVALUATOR_VERSION,
    decode_generated_addition,
    evaluate_anchored_identifier,
)

CURSOR_PROTOCOLS = {
    "original": "reviewer3-e1-same-original-cursor-v2",
    "identifier_start": "reviewer3-e1-paired-cursor-identifier-start-v1",
    "native_internal": "reviewer3-e1-paired-cursor-native-internal-v1",
}
PROTOCOL = CURSOR_PROTOCOLS["original"]
BRANCHES = ("base", "editor_first_token_then_base")
ROUTE_METRICS = (
    "neuron_chain_max_risk",
    "neuron_chain_mean_risk",
    "neuron_chain_active_fraction",
    "neuron_chain_pair_risk",
    "neuron_chain_pair_active_fraction",
    "neuron_chain_composite_risk",
)


def _experiment_protocol(cursor_condition: str, max_context_tokens: int) -> str:
    if cursor_condition == "original" and max_context_tokens != 512:
        return f"reviewer3-e2-paired-context-k{max_context_tokens}-v1"
    return CURSOR_PROTOCOLS[cursor_condition]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            yield value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def _finite(value: torch.Tensor) -> float:
    result = float(value.detach().float().cpu().item())
    if not math.isfinite(result):
        raise ValueError("model inference produced a non-finite scalar")
    return result


def _route_layers(policy: Any) -> list[int]:
    layers = sorted(
        {int(node.layer_index) for chain in policy.chains for node in chain.nodes}
    )
    if not layers:
        raise ValueError("frozen route policy contains no sampled layers")
    return layers


def _empty_prefix_fallback_id(tokenizer: Any) -> int:
    for value in (tokenizer.bos_token_id, tokenizer.eos_token_id):
        if value is not None:
            return int(value)
    return 0


def _prepare_records(
    *,
    manifest: Path,
    reference_audit: Path | None,
    tokenizer: Any,
    setting: str,
    language: str,
    max_context_tokens: int,
    max_samples: int,
    shard_index: int,
    num_shards: int,
    protocol: str = PROTOCOL,
    cursor_condition: str = "original",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    fallback_id = _empty_prefix_fallback_id(tokenizer)
    manifest_rows = _read_jsonl_gz(manifest)
    if reference_audit is None:
        paired_rows = ((row, None) for row in manifest_rows)
    else:
        paired_rows = zip(manifest_rows, _read_jsonl_gz(reference_audit), strict=True)
    for manifest_row, reference_row in paired_rows:
        sample_index = int(manifest_row["e1_sample_index"])
        sample_id = str(manifest_row["canonical_sample_id"])
        if reference_row is not None:
            if sample_index != int(reference_row["e1_sample_index"]):
                raise ValueError("manifest/reference-audit row order mismatch")
            if sample_id != str(reference_row["canonical_sample_id"]):
                raise ValueError("manifest/reference-audit occurrence mismatch")
        if sample_index % num_shards != shard_index:
            continue
        if max_samples > 0 and len(records) >= max_samples:
            break
        manifest_condition = str(manifest_row.get("cursor_condition") or "original")
        if manifest_condition != cursor_condition:
            raise ValueError(
                f"{sample_id}: manifest cursor condition {manifest_condition!r} != requested {cursor_condition!r}"
            )
        prefix = str(manifest_row["prefix"])
        full_prefix_ids = [
            int(value) for value in tokenizer.encode(prefix, add_special_tokens=False)
        ]
        independent_suffix_ids = [
            int(value)
            for value in tokenizer.encode(
                str(manifest_row["remaining_identifier_suffix"]),
                add_special_tokens=False,
            )
        ]
        if not independent_suffix_ids:
            raise ValueError(f"empty suffix reference for {sample_id}")
        reference_target_mode = (
            str(
                reference_row.get("reference_target_mode")
                or "independent_remaining_suffix"
            )
            if reference_row is not None
            else "independent_remaining_suffix"
        )
        if reference_target_mode == "independent_remaining_suffix":
            reference_ids = independent_suffix_ids
        elif reference_target_mode == "continuous_source_next_token":
            assert reference_row is not None
            reference_ids = [
                int(value) for value in reference_row["reference_token_ids"]
            ]
            if not reference_ids:
                raise ValueError(f"empty continuous reference for {sample_id}")
        else:
            raise ValueError(f"unknown reference target mode {reference_target_mode!r}")
        if reference_row is None:
            old_reference = int(reference_ids[0])
            reference_path_category = "not_repeated_for_native_full_e1b"
            character_reconstruction_exact = None
            same_as_continuous_token_path = None
            reference_failure_cause = None
        else:
            reference_path = reference_row["reference_path"]
            if len(full_prefix_ids) != int(reference_path["prefix_token_count"]):
                raise ValueError(f"prefix token-count mismatch for {sample_id}")
            old_reference = int(reference_row["old_reference_first_token_id"])
            reference_path_category = str(reference_path["category"])
            character_reconstruction_exact = bool(
                reference_path["character_reconstruction_exact"]
            )
            same_as_continuous_token_path = bool(
                reference_path["same_as_continuous_token_path"]
            )
            reference_failure_cause = reference_path.get("failure_cause")
        if reference_ids[0] != old_reference:
            raise ValueError(f"old reference token mismatch for {sample_id}")
        used_fallback = not full_prefix_ids
        model_prefix_ids = full_prefix_ids or [fallback_id]
        context_ids = model_prefix_ids[-max_context_tokens:]
        decoded_window = tokenizer.decode(
            context_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        records.append(
            {
                "protocol": protocol,
                "evaluator_version": EVALUATOR_VERSION,
                "setting": setting,
                "language": language,
                "e1_sample_index": sample_index,
                "canonical_sample_id": sample_id,
                "repository": str(manifest_row["repo_id"]),
                "file": str(manifest_row["file_path"]),
                "full_file_sha256": str(manifest_row["full_file_sha256"]),
                "occurrence": {
                    "identifier_span_start": int(
                        manifest_row["identifier_start_absolute_position"]
                    ),
                    "identifier_span_end": int(
                        manifest_row["identifier_start_absolute_position"]
                    )
                    + len(str(manifest_row["target_identifier"])),
                    "original_character_cursor": int(
                        manifest_row["original_cursor_absolute_position"]
                    ),
                    "evaluation_character_cursor": int(
                        manifest_row.get("evaluation_cursor_absolute_position")
                        if manifest_row.get("evaluation_cursor_absolute_position")
                        is not None
                        else manifest_row["original_cursor_absolute_position"]
                    ),
                    "cursor_condition": cursor_condition,
                    "target_identifier": str(manifest_row["target_identifier"]),
                    "existing_identifier_prefix": str(
                        manifest_row["existing_identifier_prefix"]
                    ),
                    "remaining_identifier_suffix": str(
                        manifest_row["remaining_identifier_suffix"]
                    ),
                    "cursor_kind": str(manifest_row["cursor_kind"]),
                    "target_split_offset": int(manifest_row["target_split_offset"]),
                },
                "input": {
                    "full_character_prefix_count": len(prefix),
                    "full_prefix_token_count": len(full_prefix_ids),
                    "model_input_used_empty_prefix_fallback": used_fallback,
                    "model_input_token_count": len(context_ids),
                    "left_truncated_token_count": max(
                        0, len(model_prefix_ids) - len(context_ids)
                    ),
                    "token_ids": context_ids,
                    "token_ids_sha256": _sha256_ids(context_ids),
                    "decoded_window_with_special_tokens": str(decoded_window),
                },
                "old_reference": {
                    "first_token_id": old_reference,
                    "reference_target_mode": reference_target_mode,
                    "reference_token_path_ids": reference_ids,
                    "independent_suffix_token_ids": independent_suffix_ids,
                    "suffix_token_ids": reference_ids,
                    "reference_path_category": reference_path_category,
                    "character_reconstruction_exact": character_reconstruction_exact,
                    "same_as_continuous_token_path": same_as_continuous_token_path,
                    "failure_cause": reference_failure_cause,
                },
            }
        )
    return records


@torch.inference_mode()
def _score_first_tokens(
    *,
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    editor: torch.nn.Module,
    sample_layers: list[int],
    device: str,
    batch_size: int,
    composite_threshold: float,
    pair_threshold: float,
) -> None:
    from ramp_hsr.runtime import _batch_neuron_chain_risks

    order = sorted(
        range(len(records)),
        key=lambda index: (len(records[index]["input"]["token_ids"]), index),
    )
    lm_dtype = _model_dtype(model.lm_head)
    started = time.monotonic()
    for offset in range(0, len(order), batch_size):
        indices = order[offset : offset + batch_size]
        sequences = [records[index]["input"]["token_ids"] for index in indices]
        (input_ids, attention_mask, position_ids, _) = _pad_batch(
            sequences, pad_token_id=int(tokenizer.pad_token_id), device=device
        )
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        rows = torch.arange(len(indices), device=device)
        positions = torch.full(
            (len(indices),),
            int(input_ids.shape[1]) - 1,
            dtype=torch.long,
            device=device,
        )
        final_hidden = outputs.last_hidden_state[rows, positions].float()
        base_logits = model.lm_head(final_hidden.to(dtype=lm_dtype)).float()
        edited_hidden = editor(final_hidden).float()
        edited_logits = model.lm_head(edited_hidden.to(dtype=lm_dtype)).float()
        base_log_probs = F.log_softmax(base_logits, dim=-1)
        base_probs = base_log_probs.exp()
        base_top2 = torch.topk(base_logits, k=2, dim=-1)
        base_ids = base_logits.argmax(dim=-1)
        edited_ids = edited_logits.argmax(dim=-1)
        target_ids = torch.tensor(
            [records[index]["old_reference"]["first_token_id"] for index in indices],
            dtype=torch.long,
            device=device,
        )
        sampled = torch.stack(
            [
                outputs.hidden_states[layer][rows, positions].float()
                for layer in sample_layers
            ],
            dim=0,
        )
        if not bool(torch.isfinite(sampled).all().item()):
            raise ValueError("route hidden states contain NaN or infinity")
        route = _batch_neuron_chain_risks(
            torch=torch, stacked_layers=sampled, sample_layers=sample_layers
        )
        for local, index in enumerate(indices):
            composite = _finite(route["neuron_chain_composite_risk"][local])
            pair = _finite(route["neuron_chain_pair_risk"][local])
            base_id = int(base_ids[local].item())
            edited_id = int(edited_ids[local].item())
            target_id = int(target_ids[local].item())
            records[index]["first_token"] = {
                "base_token_id": base_id,
                "editor_token_id": edited_id,
                "base_token_decoded": tokenizer.decode(
                    [base_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "editor_token_decoded": tokenizer.decode(
                    [edited_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "base_old_token_exact": base_id == target_id,
                "editor_old_token_exact": edited_id == target_id,
                "base_confidence": {
                    "maximum_softmax_probability": _finite(base_probs[local].max()),
                    "entropy": _finite(
                        -(base_probs[local] * base_log_probs[local]).sum()
                    ),
                    "logit_margin": _finite(
                        base_top2.values[local, 0] - base_top2.values[local, 1]
                    ),
                    "target_log_probability": _finite(base_log_probs[local, target_id]),
                },
                "route": {
                    **{key: _finite(route[key][local]) for key in ROUTE_METRICS},
                    "composite_threshold": float(composite_threshold),
                    "pair_threshold": float(pair_threshold),
                    "active": bool(
                        composite >= composite_threshold or pair >= pair_threshold
                    ),
                },
            }
        processed = min(offset + len(indices), len(order))
        if processed == len(order) or processed % max(batch_size * 25, 1) == 0:
            elapsed = max(time.monotonic() - started, 1e-09)
            print(
                json.dumps(
                    {
                        "event": "e1_first_token_progress",
                        "processed": processed,
                        "total": len(order),
                        "records_per_second": processed / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def _truncate_at_first_special(
    token_ids: Sequence[int], special_token_ids: set[int]
) -> tuple[list[int], int | None]:
    values = [int(value) for value in token_ids]
    for index, value in enumerate(values):
        if value in special_token_ids:
            return (values[: index + 1], value)
    return (values, None)


def _evaluate_generation(
    *,
    tokenizer: Any,
    record: dict[str, Any],
    generated_token_ids: Sequence[int],
    special_token_ids: set[int],
    horizon: int,
    final_cap: bool,
) -> dict[str, Any]:
    (token_ids, terminal_token_id) = _truncate_at_first_special(
        generated_token_ids, special_token_ids
    )
    decoded = decode_generated_addition(
        tokenizer=tokenizer,
        context_token_ids=record["input"]["token_ids"],
        generated_token_ids=token_ids,
    )
    occurrence = record["occurrence"]
    common = {
        "target_identifier": occurrence["target_identifier"],
        "existing_identifier_prefix": occurrence["existing_identifier_prefix"],
        "generated_text": decoded["added_text"],
        "language": record["language"],
        "cursor_kind": occurrence["cursor_kind"],
        "terminal_boundary_observed": terminal_token_id is not None,
        "generation_reached_cap": bool(final_cap and terminal_token_id is None),
    }
    tolerant = evaluate_anchored_identifier(
        **common, allow_identifier_start_horizontal_whitespace=True
    )
    strict = evaluate_anchored_identifier(
        **common, allow_identifier_start_horizontal_whitespace=False
    )
    raw_with_special = tokenizer.decode(
        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return {
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "generation_horizon_used": int(horizon),
        "terminal_special_token_id": terminal_token_id,
        "raw_generated_decode_with_special_tokens": str(raw_with_special),
        "decoded_generation": decoded,
        "complete_identifier_tolerant_start": tolerant,
        "complete_identifier_strict_start": strict,
    }


def _generation_needs_more(result: dict[str, Any]) -> bool:
    if result["terminal_special_token_id"] is not None:
        return False
    scored = result["complete_identifier_tolerant_start"]
    if bool(scored["right_boundary_observed"]):
        return False
    if bool(scored["identifier_emitted"]):
        return True
    generated = str(scored["generated_text"])
    if generated == "":
        return True
    return bool(
        scored["cursor_kind"] == "identifier_start" and generated.strip(" \t") == ""
    )


def _generation_stop_reason(result: dict[str, Any]) -> str:
    scored = result["complete_identifier_tolerant_start"]
    if result["terminal_special_token_id"] is not None:
        return "terminal_special_token"
    if bool(scored["textual_right_boundary_observed"]):
        return "lexical_right_boundary"
    if bool(scored["generation_reached_cap"]):
        return "generation_cap_without_identifier_boundary"
    if not bool(scored["identifier_emitted"]):
        return "no_identifier_at_target_position"
    return "unresolved_identifier_prefix"


@torch.inference_mode()
def _generate_to_horizon(
    *,
    tokenizer: Any,
    model: Any,
    requests: list[dict[str, Any]],
    device: str,
    batch_size: int,
    horizon: int,
    generated_prefixes: dict[tuple[int, str], list[int]] | None = None,
) -> dict[tuple[int, str], list[int]]:
    if horizon <= 1:
        return {
            (int(request["record_index"]), str(request["branch"])): [
                int(request["first_token_id"])
            ]
            for request in requests
        }
    results: dict[tuple[int, str], list[int]] = {}

    def committed_prefix(request: dict[str, Any]) -> list[int]:
        key = (int(request["record_index"]), str(request["branch"]))
        if generated_prefixes is None:
            return [int(request["first_token_id"])]
        prefix = [int(value) for value in generated_prefixes[key]]
        if not prefix or prefix[0] != int(request["first_token_id"]):
            raise RuntimeError(f"invalid committed generation prefix for {key}")
        return prefix

    order = sorted(
        requests,
        key=lambda item: (
            len(committed_prefix(item)),
            len(item["context_ids"]),
            int(item["record_index"]),
            str(item["branch"]),
        ),
    )
    pad_id = int(tokenizer.pad_token_id)
    started = time.monotonic()
    for offset in range(0, len(order), batch_size):
        batch = order[offset : offset + batch_size]
        prefixes = [committed_prefix(request) for request in batch]
        prefix_lengths = {len(prefix) for prefix in prefixes}
        if len(prefix_lengths) != 1:
            raise RuntimeError("mixed committed-prefix lengths in generation batch")
        committed_length = next(iter(prefix_lengths))
        if committed_length >= horizon:
            raise RuntimeError(
                f"committed prefix length {committed_length} reaches horizon {horizon}"
            )
        sequences = [
            [*request["context_ids"], *prefix]
            for (request, prefix) in zip(batch, prefixes, strict=True)
        ]
        width = max((len(sequence) for sequence in sequences))
        input_ids = torch.full(
            (len(sequences), width), pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, -len(sequence) :] = torch.tensor(
                sequence, dtype=torch.long, device=device
            )
            attention_mask[row, -len(sequence) :] = 1
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=horizon - committed_length,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        continuation = generated[:, width:].detach().cpu().tolist()
        for request, prefix, tail in zip(batch, prefixes, continuation, strict=True):
            key = (int(request["record_index"]), str(request["branch"]))
            results[key] = [*prefix, *[int(value) for value in tail]]
        processed = min(offset + len(batch), len(order))
        if processed == len(order) or processed % max(batch_size * 25, 1) == 0:
            elapsed = max(time.monotonic() - started, 1e-09)
            print(
                json.dumps(
                    {
                        "event": "e1_generation_progress",
                        "horizon": horizon,
                        "processed_requests": processed,
                        "total_requests": len(order),
                        "requests_per_second": processed / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return results


def _run_generations(
    *,
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
    probe_horizon: int,
    primary_cap: int,
    extension_cap: int,
    branch_selection: dict[int, set[str]] | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    first_token_ids = {
        "base": "base_token_id",
        "editor_first_token_then_base": "editor_token_id",
    }
    requests: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        for branch, field in first_token_ids.items():
            if branch_selection is not None and branch not in branch_selection.get(
                index, set()
            ):
                continue
            requests.append(
                {
                    "record_index": index,
                    "branch": branch,
                    "context_ids": record["input"]["token_ids"],
                    "first_token_id": int(record["first_token"][field]),
                }
            )
    final: dict[tuple[int, str], dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for request in requests:
        key = (int(request["record_index"]), str(request["branch"]))
        result = _evaluate_generation(
            tokenizer=tokenizer,
            record=records[key[0]],
            generated_token_ids=[int(request["first_token_id"])],
            special_token_ids=special_ids,
            horizon=1,
            final_cap=False,
        )
        if _generation_needs_more(result):
            pending.append(request)
        else:
            result["stop_reason"] = _generation_stop_reason(result)
            final[key] = result
    previous: dict[tuple[int, str], list[int]] = {}
    horizons = []
    for value in (probe_horizon, primary_cap, extension_cap):
        if value > 1 and value not in horizons:
            horizons.append(value)
    for horizon in horizons:
        if not pending:
            break
        generated = _generate_to_horizon(
            tokenizer=tokenizer,
            model=model,
            requests=pending,
            device=device,
            batch_size=batch_size,
            horizon=horizon,
            generated_prefixes=previous if previous else None,
        )
        next_pending: list[dict[str, Any]] = []
        for request in pending:
            key = (int(request["record_index"]), str(request["branch"]))
            ids = generated[key]
            if key in previous and ids[: len(previous[key])] != previous[key]:
                raise RuntimeError(
                    f"greedy generation prefix changed across horizons for {key}"
                )
            previous[key] = ids
            final_cap = horizon == extension_cap
            result = _evaluate_generation(
                tokenizer=tokenizer,
                record=records[key[0]],
                generated_token_ids=ids,
                special_token_ids=special_ids,
                horizon=horizon,
                final_cap=final_cap,
            )
            if _generation_needs_more(result) and (not final_cap):
                next_pending.append(request)
            else:
                result["stop_reason"] = _generation_stop_reason(result)
                final[key] = result
        print(
            json.dumps(
                {
                    "event": "e1_generation_horizon_complete",
                    "horizon": horizon,
                    "evaluated": len(pending),
                    "remaining_unresolved": len(next_pending),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        pending = next_pending
    if pending:
        raise RuntimeError("generation schedule ended with unresolved requests")
    if len(final) != len(requests):
        raise RuntimeError("generation left missing branch outputs")
    return final


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    tb = [bool(row["first_token"]["base_old_token_exact"]) for row in rows]
    te = [bool(row["first_token"]["editor_old_token_exact"]) for row in rows]
    cb = [
        bool(
            row["base"]["complete_identifier_tolerant_start"][
                "complete_identifier_exact"
            ]
        )
        for row in rows
    ]
    ce = [
        bool(
            row["editor_first_token_then_base"]["complete_identifier_tolerant_start"][
                "complete_identifier_exact"
            ]
        )
        for row in rows
    ]
    token_fix = sum((not left and right for (left, right) in zip(tb, te, strict=True)))
    token_damage = sum(
        (left and (not right) for (left, right) in zip(tb, te, strict=True))
    )
    id_fix = sum((not left and right for (left, right) in zip(cb, ce, strict=True)))
    id_damage = sum(
        (left and (not right) for (left, right) in zip(cb, ce, strict=True))
    )
    token_wrong = total - sum(tb)
    token_correct = sum(tb)
    id_wrong = total - sum(cb)
    id_correct = sum(cb)
    old_fix_mapping = Counter()
    for base_token, editor_token, base_id, editor_id in zip(
        tb, te, cb, ce, strict=True
    ):
        if not base_token and editor_token:
            old_fix_mapping[
                f"base_id_{('correct' if base_id else 'wrong')}__editor_id_{('correct' if editor_id else 'wrong')}"
            ] += 1
    id_fixes_not_old_token_fixes = sum(
        (
            not base_id and editor_id and (not (not base_token and editor_token))
            for (base_token, editor_token, base_id, editor_id) in zip(
                tb, te, cb, ce, strict=True
            )
        )
    )
    base_id_correct_token_wrong = sum(
        (
            not base_token and base_id
            for (base_token, base_id) in zip(tb, cb, strict=True)
        )
    )
    outcomes: dict[str, Counter[str]] = {branch: Counter() for branch in BRANCHES}
    horizons: dict[str, Counter[str]] = {branch: Counter() for branch in BRANCHES}
    for row in rows:
        for branch in BRANCHES:
            payload = row[branch]
            outcomes[branch][
                str(payload["complete_identifier_tolerant_start"]["outcome"])
            ] += 1
            horizons[branch][str(payload["generation_horizon_used"])] += 1
    strict_base = sum(
        (
            bool(
                row["base"]["complete_identifier_strict_start"][
                    "complete_identifier_exact"
                ]
            )
            for row in rows
        )
    )
    strict_editor = sum(
        (
            bool(
                row["editor_first_token_then_base"]["complete_identifier_strict_start"][
                    "complete_identifier_exact"
                ]
            )
            for row in rows
        )
    )
    return {
        "samples": total,
        "base_token_exact_count": sum(tb),
        "base_token_exact_rate": _rate(sum(tb), total),
        "editor_token_exact_count": sum(te),
        "editor_token_exact_rate": _rate(sum(te), total),
        "token_error_recovery_count": token_fix,
        "token_error_recovery_rate": _rate(token_fix, token_wrong),
        "token_damage_count": token_damage,
        "token_damage_rate": _rate(token_damage, token_correct),
        "base_identifier_exact_count": sum(cb),
        "base_identifier_exact_rate": _rate(sum(cb), total),
        "editor_identifier_exact_count": sum(ce),
        "editor_identifier_exact_rate": _rate(sum(ce), total),
        "identifier_error_recovery_count": id_fix,
        "identifier_error_recovery_rate": _rate(id_fix, id_wrong),
        "identifier_damage_count": id_damage,
        "identifier_damage_rate": _rate(id_damage, id_correct),
        "identifier_net_gain_count": id_fix - id_damage,
        "identifier_net_gain_rate": _rate(id_fix - id_damage, total),
        "strict_identifier_start_sensitivity": {
            "base_identifier_exact_count": strict_base,
            "base_identifier_exact_rate": _rate(strict_base, total),
            "editor_identifier_exact_count": strict_editor,
            "editor_identifier_exact_rate": _rate(strict_editor, total),
        },
        "old_token_error_recovery_mapping_to_complete_id": dict(
            sorted(old_fix_mapping.items())
        ),
        "base_id_correct_despite_old_token_wrong_count": base_id_correct_token_wrong,
        "base_id_correct_given_old_token_wrong": _rate(
            base_id_correct_token_wrong, token_wrong
        ),
        "complete_id_fixes_not_captured_by_old_token_fix_count": id_fixes_not_old_token_fixes,
        "complete_id_fixes_not_captured_by_old_token_fix_rate": _rate(
            id_fixes_not_old_token_fixes, id_fix
        ),
        "outcome_counts": {
            branch: dict(sorted(values.items()))
            for (branch, values) in outcomes.items()
        },
        "generation_horizon_counts": {
            branch: dict(sorted(values.items(), key=lambda item: int(item[0])))
            for (branch, values) in horizons.items()
        },
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_cursor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_cursor[str(row["occurrence"]["cursor_kind"])].append(row)
    decode_fallbacks = {
        branch: sum(
            (
                row[branch]["decoded_generation"]["decode_method"]
                == "standalone_fallback"
                for row in records
            )
        )
        for branch in BRANCHES
    }
    return {
        "denominator_policy": "Every frozen cohort occurrence remains in the denominator; no-ID, special-token, and generation-censored outputs count as incorrect.",
        "main": _summary_for_rows(records),
        "by_original_cursor_kind": {
            cursor: _summary_for_rows(rows)
            for (cursor, rows) in sorted(by_cursor.items())
        },
        "route_active_count": sum(
            (bool(row["first_token"]["route"]["active"]) for row in records)
        ),
        "decode_standalone_fallback_counts": decode_fallbacks,
        "reference_path_categories": dict(
            sorted(
                Counter(
                    (row["old_reference"]["reference_path_category"] for row in records)
                ).items()
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--language", choices=("python", "typescript"), required=True)
    parser.add_argument(
        "--cursor-condition", choices=tuple(CURSOR_PROTOCOLS), default="original"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--risk-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--first-token-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--probe-horizon", type=int, default=8)
    parser.add_argument("--primary-generation-cap", type=int, default=64)
    parser.add_argument("--extension-generation-cap", type=int, default=256)
    parser.add_argument("--composite-threshold", type=float, default=2.0)
    parser.add_argument("--pair-threshold", type=float, default=1.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--reuse-first-token-cache", action="store_true")
    args = parser.parse_args(argv)
    protocol = _experiment_protocol(
        str(args.cursor_condition), int(args.max_context_tokens)
    )
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    if (
        min(
            args.first_token_batch_size,
            args.generation_batch_size,
            args.max_context_tokens,
            args.probe_horizon,
        )
        <= 0
    ):
        raise ValueError("batch sizes, context length, and horizons must be positive")
    if (
        not args.probe_horizon
        <= args.primary_generation_cap
        <= args.extension_generation_cap
    ):
        raise ValueError("generation horizons must be nondecreasing")
    os.environ["SRLCC_NEURON_RISK_CONFIG"] = str(
        args.risk_config.expanduser().resolve()
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(13)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "first_token_scores.jsonl.gz"
    started = time.monotonic()
    reused_first_token_cache = False
    cached_row_protocols: list[str] = []
    from ramp_hsr.routes import _load_neuron_risk_policy_cached, load_neuron_risk_policy

    _load_neuron_risk_policy_cached.cache_clear()
    policy = load_neuron_risk_policy(args.risk_config.expanduser().resolve())
    if policy is None:
        raise ValueError("risk policy failed to load")
    sample_layers = _route_layers(policy)
    (tokenizer, model) = _load_model(
        args.model.expanduser().resolve(), str(args.device)
    )
    (editor, checkpoint) = load_editor(
        args.checkpoint.expanduser().resolve(), str(args.device)
    )
    if (
        str(checkpoint.get("condition_on")) != "hidden"
        or int(checkpoint.get("route_dim", -1)) != 0
        or int(checkpoint.get("code_dim", -1)) != 0
        or (str(checkpoint.get("hidden_state_interface")) != "post_norm")
    ):
        raise ValueError("E1 requires the frozen pure-hidden post-norm editor")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if args.reuse_first_token_cache and cache_path.is_file():
        records = list(_read_jsonl_gz(cache_path))
        if not records or any((row.get("setting") != args.setting for row in records)):
            raise ValueError("first-token cache setting mismatch")
        if any(
            (
                str(row.get("occurrence", {}).get("cursor_condition") or "original")
                != str(args.cursor_condition)
                for row in records
            )
        ):
            raise ValueError("first-token cache cursor-condition mismatch")
        reused_first_token_cache = True
        cached_row_protocols = sorted({str(row.get("protocol")) for row in records})
        for record in records:
            record["protocol"] = protocol
        print(
            json.dumps({"event": "e1_reused_first_token_cache", "rows": len(records)}),
            flush=True,
        )
    else:
        records = _prepare_records(
            manifest=args.manifest.expanduser().resolve(),
            reference_audit=(
                args.reference_audit.expanduser().resolve()
                if args.reference_audit is not None
                else None
            ),
            tokenizer=tokenizer,
            setting=str(args.setting),
            language=str(args.language),
            max_context_tokens=int(args.max_context_tokens),
            max_samples=max(0, int(args.max_samples)),
            shard_index=int(args.shard_index),
            num_shards=int(args.num_shards),
            protocol=protocol,
            cursor_condition=str(args.cursor_condition),
        )
        if not records:
            raise ValueError("selected E1 shard is empty")
        _score_first_tokens(
            records=records,
            tokenizer=tokenizer,
            model=model,
            editor=editor,
            sample_layers=sample_layers,
            device=str(args.device),
            batch_size=int(args.first_token_batch_size),
            composite_threshold=float(args.composite_threshold),
            pair_threshold=float(args.pair_threshold),
        )
        _write_jsonl_gz(cache_path, records)
    generated = _run_generations(
        records=records,
        tokenizer=tokenizer,
        model=model,
        device=str(args.device),
        batch_size=int(args.generation_batch_size),
        probe_horizon=int(args.probe_horizon),
        primary_cap=int(args.primary_generation_cap),
        extension_cap=int(args.extension_generation_cap),
    )
    resolved_model_path = args.model.expanduser().resolve()
    resolved_checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = _sha256_file(resolved_checkpoint_path)
    for index, record in enumerate(records):
        for branch in BRANCHES:
            record[branch] = generated[index, branch]
        record["model_and_tokenizer"] = {
            "model_path": str(resolved_model_path),
            "model_class": type(model).__name__,
            "model_dtype": str(_model_dtype(model)),
            "tokenizer_path": str(resolved_model_path),
            "tokenizer_class": type(tokenizer).__name__,
            "checkpoint_sha256": checkpoint_sha256,
        }
    predictions_path = output_dir / "predictions.jsonl.gz"
    _write_jsonl_gz(predictions_path, records)
    summary = _summarize(records)
    model_weight_path = resolved_model_path / "model.safetensors"
    report = {
        "protocol": protocol,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "setting": str(args.setting),
        "language": str(args.language),
        "samples": len(records),
        "summary": summary,
        "contract": {
            "cursor_condition": str(args.cursor_condition),
            "same_original_character_cursor": args.cursor_condition == "original",
            "same_source_identifier_occurrence": True,
            "causal_prefix_only": True,
            "prompt_context_tokens": int(args.max_context_tokens),
            "base_generation": "frozen Base greedy open vocabulary",
            "editor_generation": "frozen editor selects first token once; frozen Base generates every later token",
            "identifier_token_vocabulary_constraint": False,
            "teacher_forcing_after_cursor": False,
            "stop_at_reference_identifier_length": False,
            "identifier_start_horizontal_whitespace_main": "ASCII space and tab only",
            "strict_no_whitespace_start_sensitivity_reported": True,
            "newline_or_indentation_skip": False,
            "search_later_output_for_reference": False,
            "right_identifier_boundary_required": True,
            "primary_new_token_cap": int(args.primary_generation_cap),
            "unclosed_extension_new_token_cap": int(args.extension_generation_cap),
            "probe_horizon_is_compute_optimization_only": int(args.probe_horizon),
            "generation_extension": "unresolved generations continue from the previously committed prefix; earlier greedy tokens are never regenerated",
            "all_failures_remain_in_denominator": True,
        },
        "route": {
            "risk_config": str(args.risk_config.expanduser().resolve()),
            "risk_config_sha256": _sha256_file(args.risk_config.expanduser().resolve()),
            "sample_layers": sample_layers,
            "composite_threshold": float(args.composite_threshold),
            "pair_threshold": float(args.pair_threshold),
            "route_does_not_gate_editor_in_e1": True,
        },
        "checkpoint": {
            "path": str(resolved_checkpoint_path),
            "sha256": checkpoint_sha256,
            **{
                key: checkpoint.get(key)
                for key in (
                    "variant",
                    "method_name",
                    "rank",
                    "hidden_dim",
                    "route_dim",
                    "code_dim",
                    "condition_on",
                    "hidden_state_interface",
                    "normalize_expert_input",
                )
            },
        },
        "model": {
            "path": str(resolved_model_path),
            "class": type(model).__name__,
            "dtype": str(_model_dtype(model)),
            "weight_sha256": (
                _sha256_file(model_weight_path) if model_weight_path.is_file() else None
            ),
            "max_position_embeddings": int(
                getattr(model.config, "max_position_embeddings", 0)
            ),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "padding_side": str(tokenizer.padding_side),
            "truncation_side": str(tokenizer.truncation_side),
            "special_token_ids": [int(value) for value in tokenizer.all_special_ids],
        },
        "inputs": {
            "manifest": str(args.manifest.expanduser().resolve()),
            "manifest_sha256": _sha256_file(args.manifest.expanduser().resolve()),
            "reference_audit": (
                str(args.reference_audit.expanduser().resolve())
                if args.reference_audit is not None
                else None
            ),
            "reference_audit_sha256": (
                _sha256_file(args.reference_audit.expanduser().resolve())
                if args.reference_audit is not None
                else None
            ),
        },
        "selection": {
            "max_samples": int(args.max_samples),
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "reused_first_token_cache": reused_first_token_cache,
            "cached_row_protocols_before_prediction_relabel": cached_row_protocols,
        },
        "compute": {
            "device": str(args.device),
            "first_token_batch_size": int(args.first_token_batch_size),
            "generation_batch_size": int(args.generation_batch_size),
        },
        "outputs": {
            "first_token_scores": str(cache_path),
            "first_token_scores_sha256": _sha256_file(cache_path),
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256_file(predictions_path),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output_dir / "worker_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

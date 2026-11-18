from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from scripts.evaluate_identifiers import (
    BRANCHES,
    _generate_to_horizon,
    _route_layers,
    _score_first_tokens,
    _sha256_file,
    _sha256_ids,
    _truncate_at_first_special,
)
from ramp_hsr.checkpoints import load_editor
from scripts._model_io import _load_model, _model_dtype
from ramp_hsr.identifiers import (
    EVALUATOR_VERSION,
    decode_generated_addition,
    evaluate_anchored_identifier,
)

INPUT_PROTOCOL = "reviewer3-e3-public-protocol-inputs-v1"
PROTOCOLS = {
    "benchmark_original_prefix": "reviewer3-e3-benchmark-original-prefix-v1",
    "target_aligned_prefix": "reviewer3-e3-target-aligned-prefix-v1",
}


def _read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prepare_records(
    *,
    manifest: Path,
    setting: str,
    language: str,
    condition: str,
    max_samples: int,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    protocol = PROTOCOLS[condition]
    for row in _read_jsonl_gz(manifest):
        if str(row.get("protocol")) != INPUT_PROTOCOL:
            raise ValueError("unexpected E3 input protocol")
        if str(row.get("setting")) != setting:
            raise ValueError("setting manifest mismatch")
        if str(row.get("language")) != language:
            raise ValueError("language manifest mismatch")
        sample_index = int(row["e3_sample_index"])
        if sample_index % num_shards != shard_index:
            continue
        if max_samples > 0 and len(records) >= max_samples:
            break
        input_block = dict(row["input_conditions"][condition])
        token_ids = [int(value) for value in input_block["token_ids"]]
        if not token_ids:
            raise ValueError(f"{row['canonical_sample_id']}: empty model input")
        if _sha256_ids(token_ids) != str(input_block["token_ids_sha256"]):
            raise ValueError(f"{row['canonical_sample_id']}: token hash mismatch")
        reference_block = dict(row["reference_conditions"][condition])
        target_first_token = int(reference_block["first_token_id"])
        aligned_reference = row["reference_conditions"]["target_aligned_prefix"]
        reference_path = aligned_reference.get("reference_path", {})
        records.append(
            {
                "protocol": protocol,
                "evaluator_version": EVALUATOR_VERSION,
                "setting": setting,
                "language": language,
                "e3_sample_index": sample_index,
                "canonical_sample_id": str(row["canonical_sample_id"]),
                "dataset": str(row["dataset"]),
                "source_index": int(row["source_index"]),
                "repository": str(row["repository"]),
                "file": str(row["file"]),
                "full_reference_sha256": str(row["full_reference_sha256"]),
                "occurrence": {
                    "protocol_condition": condition,
                    "target_identifier": str(row["target_identifier"]),
                    "existing_identifier_prefix": "",
                    "cursor_kind": "identifier_start",
                    "reference_leading_text": str(row["reference_leading_text"]),
                    "reference_leading_text_sha256": str(
                        row["reference_leading_text_sha256"]
                    ),
                    "reference_characters_added_in_target_aligned": int(
                        row["reference_characters_added"]
                    ),
                    "leading_text_category": str(row["leading_text_category"]),
                    "target_identifier_start_in_target": int(
                        row["target_identifier_start_in_target"]
                    ),
                    "target_identifier_end_in_target": int(
                        row["target_identifier_end_in_target"]
                    ),
                },
                "input": input_block,
                "old_reference": {
                    "first_token_id": target_first_token,
                    "prediction_object": str(reference_block["prediction_object"]),
                    "reference_path_category": str(
                        reference_path.get("category") or "not_applicable"
                    ),
                    "character_reconstruction_exact": reference_path.get(
                        "character_reconstruction_exact"
                    ),
                    "same_as_continuous_token_path": reference_path.get(
                        "same_as_continuous_token_path"
                    ),
                    "failure_cause": reference_path.get("failure_cause"),
                },
                "protocol_comparison": {
                    "token_reference_same": bool(row["protocol_token_reference_same"]),
                    "input_window_same": bool(row["protocol_input_window_same"]),
                    "historical_target_aligned_input": row[
                        "historical_target_aligned_input"
                    ],
                },
            }
        )
    if not records:
        raise ValueError("selected E3 shard is empty")
    return records


def _first_character_difference(left: str, right: str) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _blank_identifier_result(
    *,
    target_identifier: str,
    generated_text: str,
    language: str,
    terminal_boundary_observed: bool,
    generation_reached_cap: bool,
    outcome: str,
) -> dict[str, Any]:
    result = evaluate_anchored_identifier(
        target_identifier=target_identifier,
        existing_identifier_prefix="",
        generated_text="",
        language=language,
        cursor_kind="identifier_start",
        allow_identifier_start_horizontal_whitespace=True,
        terminal_boundary_observed=terminal_boundary_observed,
        generation_reached_cap=generation_reached_cap,
    )
    result["generated_text"] = generated_text
    result["outcome"] = outcome
    result["complete_identifier_exact"] = False
    return result


def score_generated_text(
    *,
    record: dict[str, Any],
    generated_text: str,
    terminal_boundary_observed: bool,
    generation_reached_cap: bool,
) -> dict[str, Any]:
    """Score generated characters at the predeclared target coordinate."""
    condition = str(record["occurrence"]["protocol_condition"])
    target_identifier = str(record["occurrence"]["target_identifier"])
    language = str(record["language"])
    leading = str(record["occurrence"]["reference_leading_text"])
    if condition == "target_aligned_prefix":
        suffix = generated_text
        target_reached = True
        leading_exact = True
        leading_status = "provided_in_model_input"
    elif condition == "benchmark_original_prefix":
        if len(generated_text) < len(leading) and leading.startswith(generated_text):
            return {
                "target_reached": False,
                "pre_identifier_reference_exact": None,
                "pre_identifier_status": "matching_prefix_target_not_reached",
                "first_pre_identifier_difference": None,
                "generated_pre_identifier_text": generated_text,
                "identifier_generated_text": "",
                "complete_identifier_tolerant_start": _blank_identifier_result(
                    target_identifier=target_identifier,
                    generated_text=generated_text,
                    language=language,
                    terminal_boundary_observed=terminal_boundary_observed,
                    generation_reached_cap=generation_reached_cap,
                    outcome="target_identifier_not_reached",
                ),
                "complete_identifier_strict_start": _blank_identifier_result(
                    target_identifier=target_identifier,
                    generated_text=generated_text,
                    language=language,
                    terminal_boundary_observed=terminal_boundary_observed,
                    generation_reached_cap=generation_reached_cap,
                    outcome="target_identifier_not_reached",
                ),
            }
        generated_leading = generated_text[: len(leading)]
        if generated_leading != leading:
            difference = _first_character_difference(generated_leading, leading)
            return {
                "target_reached": len(generated_text) >= len(leading),
                "pre_identifier_reference_exact": False,
                "pre_identifier_status": "reference_prefix_mismatch",
                "first_pre_identifier_difference": difference,
                "generated_pre_identifier_text": generated_leading,
                "identifier_generated_text": "",
                "complete_identifier_tolerant_start": _blank_identifier_result(
                    target_identifier=target_identifier,
                    generated_text=generated_text,
                    language=language,
                    terminal_boundary_observed=terminal_boundary_observed,
                    generation_reached_cap=generation_reached_cap,
                    outcome="pre_identifier_reference_mismatch",
                ),
                "complete_identifier_strict_start": _blank_identifier_result(
                    target_identifier=target_identifier,
                    generated_text=generated_text,
                    language=language,
                    terminal_boundary_observed=terminal_boundary_observed,
                    generation_reached_cap=generation_reached_cap,
                    outcome="pre_identifier_reference_mismatch",
                ),
            }
        suffix = generated_text[len(leading) :]
        target_reached = True
        leading_exact = True
        leading_status = "generated_reference_prefix_exact"
    else:
        raise ValueError(f"unsupported E3 condition: {condition}")
    common = {
        "target_identifier": target_identifier,
        "existing_identifier_prefix": "",
        "generated_text": suffix,
        "language": language,
        "cursor_kind": "identifier_start",
        "terminal_boundary_observed": terminal_boundary_observed,
        "generation_reached_cap": generation_reached_cap,
    }
    tolerant = evaluate_anchored_identifier(
        **common, allow_identifier_start_horizontal_whitespace=True
    )
    strict = evaluate_anchored_identifier(
        **common, allow_identifier_start_horizontal_whitespace=False
    )
    return {
        "target_reached": target_reached,
        "pre_identifier_reference_exact": leading_exact,
        "pre_identifier_status": leading_status,
        "first_pre_identifier_difference": None,
        "generated_pre_identifier_text": (
            leading if condition == "benchmark_original_prefix" else ""
        ),
        "identifier_generated_text": suffix,
        "complete_identifier_tolerant_start": tolerant,
        "complete_identifier_strict_start": strict,
    }


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
    scored = score_generated_text(
        record=record,
        generated_text=str(decoded["added_text"]),
        terminal_boundary_observed=terminal_token_id is not None,
        generation_reached_cap=bool(final_cap and terminal_token_id is None),
    )
    return {
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "generation_horizon_used": int(horizon),
        "terminal_special_token_id": terminal_token_id,
        "raw_generated_decode_with_special_tokens": tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "decoded_generation": decoded,
        **scored,
    }


def _generation_needs_more(result: dict[str, Any]) -> bool:
    if result["terminal_special_token_id"] is not None:
        return False
    if result["pre_identifier_status"] == "reference_prefix_mismatch":
        return False
    if not bool(result["target_reached"]):
        return not bool(
            result["complete_identifier_tolerant_start"]["generation_reached_cap"]
        )
    scored = result["complete_identifier_tolerant_start"]
    if bool(scored["right_boundary_observed"]):
        return False
    if bool(scored["generation_reached_cap"]):
        return False
    if bool(scored["identifier_emitted"]):
        return True
    text = str(scored["generated_text"])
    return text == "" or text.strip(" \t") == ""


def _stop_reason(result: dict[str, Any]) -> str:
    if result["terminal_special_token_id"] is not None:
        return "terminal_special_token"
    if result["pre_identifier_status"] == "reference_prefix_mismatch":
        return "pre_identifier_reference_mismatch"
    if not bool(result["target_reached"]):
        return "generation_cap_before_target_identifier"
    scored = result["complete_identifier_tolerant_start"]
    if bool(scored["textual_right_boundary_observed"]):
        return "lexical_right_boundary"
    if bool(scored["generation_reached_cap"]):
        return "generation_cap_without_identifier_boundary"
    if not bool(scored["identifier_emitted"]):
        return "no_identifier_at_target_position"
    return "unresolved_identifier_prefix"


@torch.inference_mode()
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
) -> dict[tuple[int, str], dict[str, Any]]:
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    first_fields = {
        "base": "base_token_id",
        "editor_first_token_then_base": "editor_token_id",
    }
    requests: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        for branch, field in first_fields.items():
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
            result["stop_reason"] = _stop_reason(result)
            final[key] = result
    previous: dict[tuple[int, str], list[int]] = {}
    horizons: list[int] = []
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
                raise RuntimeError(f"greedy prefix changed across horizons for {key}")
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
                result["stop_reason"] = _stop_reason(result)
                final[key] = result
        print(
            json.dumps(
                {
                    "event": "e3_generation_horizon_complete",
                    "horizon": horizon,
                    "evaluated": len(pending),
                    "remaining_unresolved": len(next_pending),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        pending = next_pending
    if pending or len(final) != len(requests):
        raise RuntimeError("E3 generation schedule left unresolved requests")
    return final


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _summary_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    base_token = sum((bool(row["first_token"]["base_old_token_exact"]) for row in rows))
    editor_token = sum(
        (bool(row["first_token"]["editor_old_token_exact"]) for row in rows)
    )
    base_id = sum(
        (
            bool(
                row["base"]["complete_identifier_tolerant_start"][
                    "complete_identifier_exact"
                ]
            )
            for row in rows
        )
    )
    editor_id = sum(
        (
            bool(
                row["editor_first_token_then_base"][
                    "complete_identifier_tolerant_start"
                ]["complete_identifier_exact"]
            )
            for row in rows
        )
    )
    id_fix = sum(
        (
            not bool(
                row["base"]["complete_identifier_tolerant_start"][
                    "complete_identifier_exact"
                ]
            )
            and bool(
                row["editor_first_token_then_base"][
                    "complete_identifier_tolerant_start"
                ]["complete_identifier_exact"]
            )
            for row in rows
        )
    )
    id_damage = sum(
        (
            bool(
                row["base"]["complete_identifier_tolerant_start"][
                    "complete_identifier_exact"
                ]
            )
            and (
                not bool(
                    row["editor_first_token_then_base"][
                        "complete_identifier_tolerant_start"
                    ]["complete_identifier_exact"]
                )
            )
            for row in rows
        )
    )
    outcomes: dict[str, Counter[str]] = {
        branch: Counter(
            (
                str(row[branch]["complete_identifier_tolerant_start"]["outcome"])
                for row in rows
            )
        )
        for branch in BRANCHES
    }
    pre_identifier: dict[str, Counter[str]] = {
        branch: Counter((str(row[branch]["pre_identifier_status"]) for row in rows))
        for branch in BRANCHES
    }
    horizons: dict[str, Counter[int]] = {
        branch: Counter((int(row[branch]["generation_horizon_used"]) for row in rows))
        for branch in BRANCHES
    }
    return {
        "N": n,
        "base_token_correct": base_token,
        "editor_token_correct": editor_token,
        "base_token_em": _safe_rate(base_token, n),
        "editor_token_em": _safe_rate(editor_token, n),
        "base_identifier_correct": base_id,
        "editor_identifier_correct": editor_id,
        "base_identifier_em": _safe_rate(base_id, n),
        "editor_identifier_em": _safe_rate(editor_id, n),
        "identifier_fix": id_fix,
        "identifier_damage": id_damage,
        "identifier_error_recovery": _safe_rate(id_fix, n - base_id),
        "identifier_damage_rate": _safe_rate(id_damage, base_id),
        "identifier_net": _safe_rate(editor_id - base_id, n),
        "route_active": sum(
            (bool(row["first_token"]["route"]["active"]) for row in rows)
        ),
        "outcome_counts": {
            branch: dict(sorted(counter.items()))
            for (branch, counter) in outcomes.items()
        },
        "pre_identifier_status_counts": {
            branch: dict(sorted(counter.items()))
            for (branch, counter) in pre_identifier.items()
        },
        "generation_horizon_counts": {
            branch: {str(key): value for (key, value) in sorted(counter.items())}
            for (branch, counter) in horizons.items()
        },
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_dataset[str(row["dataset"])].append(row)
    return {
        "overall": _summary_rows(records),
        "by_dataset": {
            dataset: _summary_rows(rows)
            for (dataset, rows) in sorted(by_dataset.items())
        },
        "route_active_count": sum(
            (bool(row["first_token"]["route"]["active"]) for row in records)
        ),
        "denominator_policy": "Every predeclared cohort row remains in the denominator. Prefix mismatch, no-ID, special-token, and generation-censored outcomes are incorrect and are also reported by reason.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--language", choices=("python", "typescript"), required=True)
    parser.add_argument("--condition", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--risk-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--first-token-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=64)
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
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    if (
        min(args.first_token_batch_size, args.generation_batch_size, args.probe_horizon)
        <= 0
    ):
        raise ValueError("batch sizes and horizons must be positive")
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
    protocol = PROTOCOLS[str(args.condition)]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "first_token_scores.jsonl.gz"
    started = time.monotonic()
    from ramp_hsr.routes import _load_neuron_risk_policy_cached, load_neuron_risk_policy

    _load_neuron_risk_policy_cached.cache_clear()
    risk_path = args.risk_config.expanduser().resolve()
    policy = load_neuron_risk_policy(risk_path)
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
        raise ValueError("E3 requires the frozen pure-hidden post-norm editor")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    reused = False
    if args.reuse_first_token_cache and cache_path.is_file():
        records = list(_read_jsonl_gz(cache_path))
        if not records or any(
            (str(row.get("protocol")) != protocol for row in records)
        ):
            raise ValueError("E3 first-token cache protocol mismatch")
        reused = True
    else:
        records = _prepare_records(
            manifest=args.manifest.expanduser().resolve(),
            setting=str(args.setting),
            language=str(args.language),
            condition=str(args.condition),
            max_samples=max(0, int(args.max_samples)),
            shard_index=int(args.shard_index),
            num_shards=int(args.num_shards),
        )
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
    resolved_model = args.model.expanduser().resolve()
    resolved_checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha = _sha256_file(resolved_checkpoint)
    for index, record in enumerate(records):
        for branch in BRANCHES:
            record[branch] = generated[index, branch]
        record["model_and_tokenizer"] = {
            "model_path": str(resolved_model),
            "model_class": type(model).__name__,
            "model_dtype": str(_model_dtype(model)),
            "tokenizer_class": type(tokenizer).__name__,
            "checkpoint_sha256": checkpoint_sha,
        }
    predictions_path = output_dir / "predictions.jsonl.gz"
    _write_jsonl_gz(predictions_path, records)
    model_weight = resolved_model / "model.safetensors"
    manifest_path = args.manifest.expanduser().resolve()
    report = {
        "status": "complete",
        "protocol": protocol,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "setting": str(args.setting),
        "language": str(args.language),
        "condition": str(args.condition),
        "samples": len(records),
        "summary": _summarize(records),
        "contract": {
            "same_pre_outcome_common_cohort": True,
            "benchmark_original_prefix": args.condition == "benchmark_original_prefix",
            "target_aligned_prefix": args.condition == "target_aligned_prefix",
            "complete_id_target_coordinate": "first non-keyword identifier in the benchmark target, fixed before generation",
            "original_prefix_success_requires_exact_pre_identifier_text": True,
            "target_aligned_reference_characters": "only benchmark_target[:identifier_start]; target identifier excluded",
            "prompt_context_tokens": 512,
            "base_generation": "frozen Base greedy open vocabulary",
            "editor_generation": "frozen editor selects generated step 0 once; frozen Base generates all later tokens",
            "runtime_gate": "always edit; frozen route is recorded but does not gate",
            "teacher_forcing": False,
            "identifier_vocabulary_constraint": False,
            "search_later_output_for_reference": False,
            "stop_at_reference_identifier_length": False,
            "primary_new_token_cap": int(args.primary_generation_cap),
            "unclosed_extension_new_token_cap": int(args.extension_generation_cap),
            "generation_extension": "continue from the committed greedy prefix; never regenerate earlier tokens",
            "all_failures_remain_in_denominator": True,
        },
        "route": {
            "risk_config": str(risk_path),
            "risk_config_sha256": _sha256_file(risk_path),
            "sample_layers": sample_layers,
            "composite_threshold": float(args.composite_threshold),
            "pair_threshold": float(args.pair_threshold),
            "route_does_not_gate_editor": True,
        },
        "checkpoint": {
            "path": str(resolved_checkpoint),
            "sha256": checkpoint_sha,
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
                )
            },
        },
        "model": {
            "path": str(resolved_model),
            "class": type(model).__name__,
            "dtype": str(_model_dtype(model)),
            "weight_sha256": (
                _sha256_file(model_weight) if model_weight.is_file() else None
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
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
        },
        "selection": {
            "max_samples": int(args.max_samples),
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "reused_first_token_cache": reused,
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

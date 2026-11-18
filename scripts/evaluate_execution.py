from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import keyword
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    position = 0
    for part in text.splitlines(True):
        position += len(part)
        offsets.append(position)
    return offsets


def _first_nonkeyword_python_name(text: str) -> tuple[int, str, tuple[int, int]] | None:
    offsets = _line_offsets(text)
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type != tokenize.NAME or keyword.iskeyword(token.string):
                continue
            (row, column) = token.start
            if row <= 0 or row > len(offsets):
                return None
            position = offsets[row - 1] + column
            if 0 <= position < len(text):
                return (position, token.string, token.start)
    except (IndentationError, SyntaxError, SystemError, tokenize.TokenError):
        return None
    return None


def _round_robin_fill_types(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if count <= 0 or count >= len(rows):
        return list(rows)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("fill_type") or "unknown")].append(row)
    ordered_keys = sorted(buckets)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < count:
        added = False
        for key in ordered_keys:
            bucket = buckets[key]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        cursor += 1
    return selected


def command_prepare(args: argparse.Namespace) -> None:
    requested_repos = (
        {item.strip() for item in args.repos.split(",") if item.strip()}
        if args.repos
        else None
    )
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    source_pass = 0
    with args.input.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            repo_name = str(row.get("repo_name") or "")
            if requested_repos is not None and repo_name not in requested_repos:
                continue
            if args.require_source_pass and (not bool(row.get("is_pass"))):
                continue
            source_pass += 1
            middle = str(row.get("middle_code") or "")
            first = _first_nonkeyword_python_name(middle)
            if first is None:
                continue
            (offset, name, start) = first
            prepared = dict(row)
            prepared["_source_index"] = source_index
            prepared["_first_identifier_offset"] = offset
            prepared["_first_identifier"] = name
            prepared["_first_identifier_start"] = list(start)
            prepared["_reference_scaffold"] = middle[:offset]
            candidates[repo_name].append(prepared)
    selected: list[dict[str, Any]] = []
    for repo_name in sorted(candidates):
        selected.extend(
            _round_robin_fill_types(candidates[repo_name], args.max_per_repo)
        )
    selected.sort(key=lambda row: int(row["_source_index"]))
    for manifest_index, row in enumerate(selected):
        row["_manifest_index"] = manifest_index
        row["_task_id"] = f"execrepobench:{int(row['_source_index'])}"
    _write_jsonl(args.output, selected)
    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "raw_rows": total,
        "rows_passing_source_filter": source_pass,
        "selected_rows": len(selected),
        "selected_by_repo": {
            repo: sum((1 for row in selected if row["repo_name"] == repo))
            for repo in sorted(candidates)
        },
        "require_source_pass": bool(args.require_source_pass),
        "max_per_repo": int(args.max_per_repo),
        "protocol": "Reference scaffold before the first Python tokenize.NAME non-keyword is supplied identically to Base and RAMP; the edited token and frozen-base continuation are written back for repository execution.",
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _truncate_prompt(
    prompt: str, max_num_tokens: int, tokenizer: Any, side: str
) -> str:
    tokens = tokenizer.tokenize(prompt)
    if len(tokens) <= max_num_tokens:
        return prompt
    if side == "left":
        tokens = tokens[-max_num_tokens:]
    elif side == "right":
        tokens = tokens[:max_num_tokens]
    else:
        raise ValueError(f"Unknown truncation side: {side}")
    return tokenizer.convert_tokens_to_string(tokens)


def _build_official_qwen_prompt(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_context_tokens: int,
    max_generation_tokens: int,
    max_tokens: int,
) -> tuple[str, dict[str, int | str]]:
    context_code: list[str] = []
    context_names: list[str] = []
    context_token_count = 0
    for file_name, file_code in row.get("context_code") or []:
        file_code = str(file_code)
        current_count = len(tokenizer.tokenize(file_code))
        if current_count + context_token_count < max_context_tokens:
            context_token_count += current_count
            context_code.append(f"##{file_name}##:\n{file_code}")
            context_names.append(str(file_name))
        else:
            available = max(0, max_context_tokens - context_token_count)
            partial = _truncate_prompt(
                f"##{file_name}##:\n{file_code}", available, tokenizer, "right"
            )
            partial = "\n".join(partial.split("\n")[:-1])
            context_code.append(partial)
            context_names.append(str(file_name))
            break
    context_code.reverse()
    context_names.reverse()
    max_in_file_tokens = max_tokens - max_generation_tokens - max_context_tokens
    scaffold = str(row["_reference_scaffold"])
    prefix = str(row.get("prefix_code") or "") + scaffold
    suffix = str(row.get("suffix_code") or "")
    prefix = _truncate_prompt(prefix, max_in_file_tokens // 2, tokenizer, "left")
    suffix = _truncate_prompt(suffix, max_in_file_tokens // 2, tokenizer, "right")
    repo_content = ""
    for context, file_name in zip(context_code, context_names, strict=True):
        repo_content += f"\n<|file_sep|>{file_name}\n{context}"
    repo_context = (
        f"<|repo_name|>{row['repo_name']}"
        + repo_content
        + f"\n<|file_sep|>{row['file_name']}\n"
    )
    prompt = (
        f"\n{repo_context}<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>\n"
    )
    return (
        prompt,
        {
            "prompt_policy": "official_fim",
            "context_code_tokens_before_wrappers": context_token_count,
            "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
        },
    )


def _build_causal_prefix_prompt(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_context_tokens: int,
    max_generation_tokens: int,
    max_tokens: int,
) -> tuple[str, dict[str, int | str]]:
    """Reproduce the legacy public first-identifier causal-prefix protocol."""
    scaffold = str(row["_reference_scaffold"])
    prompt = str(row.get("prefix_code") or "") + scaffold
    prompt_tokens_before = len(tokenizer.encode(prompt, add_special_tokens=False))
    max_prompt_tokens = max(
        1,
        min(
            int(max_context_tokens),
            int(max_tokens) - max(1, int(max_generation_tokens)),
        ),
    )
    prompt = _truncate_prompt(prompt, max_prompt_tokens, tokenizer, "left")
    return (
        prompt,
        {
            "prompt_policy": "causal_prefix",
            "context_code_tokens_before_wrappers": 0,
            "prompt_tokens_before_truncation": prompt_tokens_before,
            "causal_context_token_limit": int(max_context_tokens),
            "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
        },
    )


def _load_pure_hidden_repair(checkpoint_path, device):
    from ramp_hsr.checkpoints import load_editor

    (editor, checkpoint) = load_editor(checkpoint_path, device)
    contract = {
        key: checkpoint[key]
        for key in (
            "condition_on",
            "route_dim",
            "code_dim",
            "hidden_state_interface",
            "normalize_expert_input",
        )
    }
    return (editor, checkpoint, contract)


def _generate_from_first_tokens(
    *,
    model: Any,
    prompt_ids: Any,
    first_token_ids: list[int],
    max_generation_tokens: int,
    stop_token_ids: list[int],
) -> tuple[dict[int, list[int]], dict[int, int | None]]:
    import torch

    unique_ids = list(dict.fromkeys((int(item) for item in first_token_ids)))
    results: dict[int, list[int]] = {}
    terminal_stops: dict[int, int | None] = {}
    stop_ids = list(dict.fromkeys((int(item) for item in stop_token_ids)))
    stop_set = set(stop_ids)
    active = [item for item in unique_ids if item not in stop_set]
    for item in unique_ids:
        if item in stop_set:
            results[item] = []
            terminal_stops[item] = item
    if not active:
        return (results, terminal_stops)
    rows = [
        torch.cat(
            [
                prompt_ids[0],
                torch.tensor(
                    [token_id], dtype=prompt_ids.dtype, device=prompt_ids.device
                ),
            ]
        )
        for token_id in active
    ]
    input_ids = torch.stack(rows, dim=0)
    if max_generation_tokens <= 1:
        generated = input_ids
    else:
        attention_mask = torch.ones_like(input_ids)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_generation_tokens - 1,
            eos_token_id=stop_ids,
            pad_token_id=stop_ids[0],
            use_cache=True,
        )
    prefix_length = int(prompt_ids.shape[1])
    for row_index, token_id in enumerate(active):
        token_ids = [
            int(value)
            for value in generated[row_index, prefix_length:].detach().cpu().tolist()
        ]
        terminal_stop = (
            token_ids[-1] if token_ids and token_ids[-1] in stop_set else None
        )
        if terminal_stop is not None:
            token_ids = token_ids[:-1]
        results[token_id] = token_ids
        terminal_stops[token_id] = terminal_stop
    return (results, terminal_stops)


def command_generate(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    rows = list(_read_jsonl(args.manifest))
    rows = [
        row
        for row in rows
        if int(row["_manifest_index"]) % args.num_shards == args.shard_index
    ]
    completed: set[str] = set()
    if args.output.exists() and args.resume:
        completed = {str(row["_task_id"]) for row in _read_jsonl(args.output)}
    elif args.output.exists():
        args.output.unlink()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=False, local_files_only=True
    )
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    (repair, checkpoint, checkpoint_contract) = _load_pure_hidden_repair(
        args.checkpoint, args.device
    )
    if int(checkpoint["hidden_dim"]) != int(model.config.hidden_size):
        raise ValueError("Checkpoint/model hidden dimensions do not match")
    lm_head_weight_fp32 = model.lm_head.weight.detach().float()
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    special_ids = set((int(value) for value in tokenizer.all_special_ids))
    eos_token_id = int(tokenizer.eos_token_id)
    repository_control_ids = [
        int(tokenizer.convert_tokens_to_ids(token))
        for token in REPOSITORY_CONTROL_TOKENS
    ]
    stop_token_ids = (
        [eos_token_id]
        if args.stop_policy == "eos_only"
        else [eos_token_id, *repository_control_ids]
    )
    generation_special_ids = special_ids | set(repository_control_ids)
    for ordinal, row in enumerate(rows, start=1):
        task_id = str(row["_task_id"])
        if task_id in completed:
            continue
        started = time.monotonic()
        scaffold = str(row["_reference_scaffold"])
        reference_continuation = str(row["middle_code"])[len(scaffold) :]
        reference_token_ids = tokenizer.encode(
            reference_continuation, add_special_tokens=False
        )
        if not reference_token_ids:
            raise ValueError(f"Empty reference continuation for {task_id}")
        if args.generation_budget_policy == "fixed_max":
            generation_budget_tokens = int(args.max_generation_tokens)
        elif args.generation_budget_policy == "reference_target_length":
            generation_budget_tokens = len(reference_token_ids)
            if int(args.max_generation_tokens) > 0:
                generation_budget_tokens = min(
                    generation_budget_tokens, int(args.max_generation_tokens)
                )
        else:
            raise ValueError(
                f"Unsupported generation budget policy: {args.generation_budget_policy}"
            )
        if generation_budget_tokens <= 0:
            raise ValueError(f"Non-positive generation budget for {task_id}")
        if args.prompt_policy == "official_fim":
            (prompt, prompt_stats) = _build_official_qwen_prompt(
                tokenizer,
                row,
                max_context_tokens=args.max_context_tokens,
                max_generation_tokens=args.max_generation_tokens,
                max_tokens=args.max_tokens,
            )
        elif args.prompt_policy == "causal_prefix":
            (prompt, prompt_stats) = _build_causal_prefix_prompt(
                tokenizer,
                row,
                max_context_tokens=args.max_context_tokens,
                max_generation_tokens=args.max_generation_tokens,
                max_tokens=args.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported prompt policy: {args.prompt_policy}")
        if args.prompt_policy == "causal_prefix" and (not prompt):
            print(
                f"[{args.shard_index}:{ordinal}/{len(rows)}] {task_id} skipped=empty_causal_prompt",
                flush=True,
            )
            continue
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = encoded.input_ids.to(args.device)
        if int(prompt_ids.shape[1]) > int(model.config.max_position_embeddings):
            raise ValueError(
                f"Prompt exceeds model context: {prompt_ids.shape[1]} > {model.config.max_position_embeddings} for {task_id}"
            )
        with torch.inference_mode():
            outputs = model(
                input_ids=prompt_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            base_token_id = int(outputs.logits[0, -1].argmax().item())
            h_last = outputs.hidden_states[-1][:, -1, :].float()
            repaired_h = repair(h_last)
            repaired_logits = functional.linear(repaired_h, lm_head_weight_fp32)
            raw_ramp_token_id = int(repaired_logits[0].argmax().item())
            ramp_special_fallback = raw_ramp_token_id in generation_special_ids
            ramp_token_id = (
                base_token_id if ramp_special_fallback else raw_ramp_token_id
            )
            (generations, terminal_stops) = _generate_from_first_tokens(
                model=model,
                prompt_ids=prompt_ids,
                first_token_ids=[base_token_id, ramp_token_id],
                max_generation_tokens=generation_budget_tokens,
                stop_token_ids=stop_token_ids,
            )
        base_ids = generations[base_token_id]
        ramp_ids = generations[ramp_token_id]
        base_special_events = [
            {
                "position": position,
                "token_id": token_id,
                "token": tokenizer.convert_ids_to_tokens(token_id),
            }
            for (position, token_id) in enumerate(base_ids)
            if token_id in special_ids
        ]
        ramp_special_events = [
            {
                "position": position,
                "token_id": token_id,
                "token": tokenizer.convert_ids_to_tokens(token_id),
            }
            for (position, token_id) in enumerate(ramp_ids)
            if token_id in special_ids
        ]
        base_continuation = tokenizer.decode(
            base_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        ramp_continuation = tokenizer.decode(
            ramp_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result = {
            "_task_id": task_id,
            "_source_index": int(row["_source_index"]),
            "_manifest_index": int(row["_manifest_index"]),
            "repo_name": row["repo_name"],
            "file_name": row["file_name"],
            "fill_type": row.get("fill_type"),
            "prefix_code": row["prefix_code"],
            "middle_code": row["middle_code"],
            "suffix_code": row["suffix_code"],
            "reference_scaffold": scaffold,
            "first_identifier": row["_first_identifier"],
            "first_identifier_offset": int(row["_first_identifier_offset"]),
            "reference_first_token_id": int(reference_token_ids[0]),
            "reference_first_token_text": tokenizer.decode(
                [int(reference_token_ids[0])],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "reference_generation_tokens": len(reference_token_ids),
            "generation_budget_tokens": generation_budget_tokens,
            "target_length_truncated": generation_budget_tokens
            < len(reference_token_ids),
            "base_first_token_id": base_token_id,
            "base_first_token_text": tokenizer.decode(
                [base_token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "raw_ramp_first_token_id": raw_ramp_token_id,
            "raw_ramp_first_token_text": tokenizer.decode(
                [raw_ramp_token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "ramp_first_token_id": ramp_token_id,
            "ramp_special_fallback": ramp_special_fallback,
            "first_token_changed": ramp_token_id != base_token_id,
            "base_generated_continuation": base_continuation,
            "ramp_generated_continuation": ramp_continuation,
            "base_generated_middle_code": scaffold + base_continuation,
            "ramp_generated_middle_code": scaffold + ramp_continuation,
            "base_generation_tokens": len(base_ids),
            "ramp_generation_tokens": len(ramp_ids),
            "base_terminal_stop_token_id": terminal_stops[base_token_id],
            "ramp_terminal_stop_token_id": terminal_stops[ramp_token_id],
            "base_generated_special_tokens": base_special_events,
            "ramp_generated_special_tokens": ramp_special_events,
            "prompt_sha256": _sha256_text(prompt),
            "prompt_stats": prompt_stats,
            "generation_seconds": time.monotonic() - started,
            "generation_contract": {
                "model": str(args.model.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_contract": checkpoint_contract,
                "intervention": "first_identifier_token_only_then_frozen_base_greedy",
                "prompt_prefix_includes_reference_scaffold": True,
                "runtime_gate": "always_repair_with_special_token_fallback",
                "max_context_tokens": args.max_context_tokens,
                "max_generation_tokens": args.max_generation_tokens,
                "generation_budget_policy": args.generation_budget_policy,
                "max_tokens": args.max_tokens,
                "dtype": args.dtype,
                "prompt_policy": args.prompt_policy,
                "stop_policy": args.stop_policy,
                "stop_token_ids": stop_token_ids,
            },
        }
        _append_jsonl(args.output, result)
        print(
            f"[{args.shard_index}:{ordinal}/{len(rows)}] {task_id} changed={result['first_token_changed']} seconds={result['generation_seconds']:.2f}",
            flush=True,
        )


def command_merge(args: argparse.Namespace) -> None:
    rows: dict[str, dict[str, Any]] = {}
    for path in args.inputs:
        for row in _read_jsonl(path):
            task_id = str(row["_task_id"])
            if task_id in rows:
                raise ValueError(f"Duplicate generated task: {task_id}")
            rows[task_id] = row
    ordered = sorted(rows.values(), key=lambda row: int(row["_manifest_index"]))
    _write_jsonl(args.output, ordered)
    print(json.dumps({"merged": len(ordered), "output": str(args.output)}, indent=2))


REPOSITORY_CONTROL_TOKENS = (
    "<|file_sep|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|fim_pad|>",
    "<|repo_name|>",
)


def _truncate_at_repository_control(text: str) -> tuple[str, dict[str, Any] | None]:
    hits = [(text.find(token), token) for token in REPOSITORY_CONTROL_TOKENS]
    hits = [(position, token) for (position, token) in hits if position >= 0]
    if not hits:
        return (text, None)
    (position, token) = min(hits)
    return (text[:position], {"token": token, "character_offset": position})


def command_control_stop(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    counts = {"base": 0, "ramp": 0}
    for source in _read_jsonl(args.input):
        row = dict(source)
        scaffold = str(row["reference_scaffold"])
        for method in ("base", "ramp"):
            key = f"{method}_generated_continuation"
            (truncated, event) = _truncate_at_repository_control(str(row[key]))
            row[f"{method}_pre_control_stop_continuation"] = row[key]
            row[key] = truncated
            row[f"{method}_generated_middle_code"] = scaffold + truncated
            row[f"{method}_repository_control_stop"] = event
            counts[method] += event is not None
        contract = dict(row.get("generation_contract") or {})
        contract["stop_policy"] = (
            "truncate_before_first_repository_control_token; post-hoc equivalent to token-ID stopping because greedy tokens before the stop are unchanged"
        )
        contract["repository_control_tokens"] = list(REPOSITORY_CONTROL_TOKENS)
        row["generation_contract"] = contract
        rows.append(row)
    _write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "base_rows_truncated": counts["base"],
                "ramp_rows_truncated": counts["ramp"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


def command_build_runnable(args: argparse.Namespace) -> None:
    passed_repos: set[str] = set()
    audit_sources: list[str] = []
    for audit_path in args.repo_audits:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        passed_repos.update((str(item) for item in payload["summary"]["passed_repos"]))
        audit_sources.append(str(audit_path.resolve()))
    selected: list[dict[str, Any]] = []
    excluded_repo_rows = 0
    reconstruction_mismatches: list[str] = []
    for source in _read_jsonl(args.manifest):
        if str(source["repo_name"]) not in passed_repos:
            excluded_repo_rows += 1
            continue
        source_file = args.repos_dir / str(source["file_name"]).lstrip("/")
        reconstructed = (
            str(source["prefix_code"])
            + str(source["middle_code"])
            + str(source["suffix_code"])
        )
        if (
            not source_file.is_file()
            or source_file.read_text(encoding="utf-8") != reconstructed
        ):
            reconstruction_mismatches.append(str(source["_task_id"]))
            continue
        row = dict(source)
        row["_manifest_index"] = len(selected)
        selected.append(row)
    _write_jsonl(args.output, selected)
    by_repo = {
        repo: sum((1 for row in selected if row["repo_name"] == repo))
        for repo in sorted(passed_repos)
    }
    summary = {
        "manifest": str(args.manifest.resolve()),
        "output": str(args.output.resolve()),
        "repo_audits": audit_sources,
        "passed_repos": sorted(passed_repos),
        "passed_repo_count": len(passed_repos),
        "selected_rows": len(selected),
        "selected_by_repo": by_repo,
        "excluded_nonpassing_repo_rows": excluded_repo_rows,
        "excluded_reconstruction_mismatches": reconstruction_mismatches,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _python_for_repo(args: argparse.Namespace, repo_name: str) -> Path:
    if args.python_map:
        mapping = json.loads(args.python_map.read_text(encoding="utf-8"))
        if repo_name not in mapping:
            raise KeyError(f"No Python executable mapped for repo {repo_name}")
        return Path(mapping[repo_name]).resolve()
    return args.python.resolve()


def _run_process(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: int
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        (stdout, stderr) = process.communicate(timeout=timeout)
        return {
            "pass": process.returncode == 0,
            "returncode": process.returncode,
            "timeout": False,
            "seconds": time.monotonic() - started,
            "stdout_tail": stdout.decode("utf-8", errors="replace")[-8000:],
            "stderr_tail": stderr.decode("utf-8", errors="replace")[-8000:],
        }
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            (stdout, stderr) = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            (stdout, stderr) = process.communicate()
        return {
            "pass": False,
            "returncode": None,
            "timeout": True,
            "seconds": time.monotonic() - started,
            "stdout_tail": stdout.decode("utf-8", errors="replace")[-8000:],
            "stderr_tail": stderr.decode("utf-8", errors="replace")[-8000:],
        }


def _execute_middle(
    *,
    row: dict[str, Any],
    middle_code: str,
    repos_dir: Path,
    python_executable: Path,
    timeout: int,
) -> dict[str, Any]:
    repo_name = str(row["repo_name"])
    source_repo = repos_dir / repo_name
    if not source_repo.is_dir():
        raise FileNotFoundError(f"Missing ExecRepoBench repo: {source_repo}")
    if not python_executable.is_file():
        raise FileNotFoundError(f"Missing Python executable: {python_executable}")
    relative = Path(str(row["file_name"]).lstrip("/"))
    if not relative.parts or relative.parts[0] != repo_name or ".." in relative.parts:
        raise ValueError(f"Unsafe or unexpected masked file path: {relative}")
    code = str(row["prefix_code"]) + middle_code + str(row["suffix_code"])
    with tempfile.TemporaryDirectory(prefix="execrepobench_") as temporary:
        root = Path(temporary)
        copied_repo = root / repo_name
        shutil.copytree(source_repo, copied_repo, symlinks=True)
        masked_file = root / relative
        masked_file.parent.mkdir(parents=True, exist_ok=True)
        masked_file.write_text(code, encoding="utf-8")
        env = os.environ.copy()
        env["PATH"] = str(python_executable.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONNOUSERSITE"] = "1"
        env["TZ"] = "UTC"
        import_roots = [copied_repo]
        if (copied_repo / "src").is_dir():
            import_roots.insert(0, copied_repo / "src")
        env["PYTHONPATH"] = os.pathsep.join((str(path) for path in import_roots))
        compile_result = _run_process(
            [str(python_executable), "-m", "py_compile", str(masked_file)],
            cwd=copied_repo,
            env=env,
            timeout=min(timeout, 60),
        )
        if compile_result["pass"]:
            test_result = _run_process(
                [str(python_executable), "evaluate_repo.py"],
                cwd=copied_repo,
                env=env,
                timeout=timeout,
            )
        else:
            test_result = {
                "pass": False,
                "returncode": None,
                "timeout": False,
                "seconds": 0.0,
                "stdout_tail": "",
                "stderr_tail": "Skipped because py_compile failed.",
            }
    return {
        "compile": compile_result,
        "tests": test_result,
        "pass": bool(compile_result["pass"] and test_result["pass"]),
        "code_sha256": _sha256_text(code),
        "python": str(python_executable),
    }


def _paired_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if record["reference"]["pass"]]
    base_pass = sum((bool(record["base"]["pass"]) for record in eligible))
    ramp_pass = sum((bool(record["ramp"]["pass"]) for record in eligible))
    base_fail_ramp_pass = sum(
        (not record["base"]["pass"] and record["ramp"]["pass"] for record in eligible)
    )
    base_pass_ramp_fail = sum(
        (record["base"]["pass"] and (not record["ramp"]["pass"]) for record in eligible)
    )
    both_pass = sum(
        (record["base"]["pass"] and record["ramp"]["pass"] for record in eligible)
    )
    neither_pass = sum(
        (
            not record["base"]["pass"] and (not record["ramp"]["pass"])
            for record in eligible
        )
    )
    denominator = len(eligible)
    return {
        "generated_rows": len(records),
        "reference_verified_rows": denominator,
        "reference_invalid_excluded": len(records) - denominator,
        "base_pass_count": base_pass,
        "ramp_pass_count": ramp_pass,
        "base_pass_at_1": base_pass / denominator if denominator else None,
        "ramp_pass_at_1": ramp_pass / denominator if denominator else None,
        "base_fail_to_ramp_pass": base_fail_ramp_pass,
        "base_pass_to_ramp_fail": base_pass_ramp_fail,
        "both_pass": both_pass,
        "neither_pass": neither_pass,
        "net_pass_transitions": base_fail_ramp_pass - base_pass_ramp_fail,
        "base_compile_pass_at_1": (
            sum((record["base"]["compile"]["pass"] for record in eligible))
            / denominator
            if denominator
            else None
        ),
        "ramp_compile_pass_at_1": (
            sum((record["ramp"]["compile"]["pass"] for record in eligible))
            / denominator
            if denominator
            else None
        ),
        "first_token_changed": sum(
            (bool(record["generation"]["first_token_changed"]) for record in eligible)
        ),
        "protocol": "Paired first-identifier reference scaffold; one always-on pure-hidden RAMP token intervention; frozen-base greedy continuation; repository tests scored only where reference code passes locally.",
    }


def _has_timeout(result: dict[str, Any]) -> bool:
    return any(
        (bool(result.get(stage, {}).get("timeout")) for stage in ("compile", "tests"))
    )


def _transition(base: dict[str, Any], ramp: dict[str, Any]) -> str:
    if not base["pass"] and ramp["pass"]:
        return "BaseFail->RAMPPass"
    if base["pass"] and (not ramp["pass"]):
        return "BasePass->RAMPFail"
    if base["pass"] and ramp["pass"]:
        return "BothPass"
    return "NeitherPass"


def command_audit_repos(args: argparse.Namespace) -> None:
    requested_repos = (
        {item.strip() for item in args.repos.split(",") if item.strip()}
        if args.repos
        else None
    )
    excluded_repos = {
        item.strip() for item in args.exclude_repos.split(",") if item.strip()
    }
    representatives: dict[str, dict[str, Any]] = {}
    mismatch_rows: list[str] = []
    for row in _read_jsonl(args.manifest):
        repo_name = str(row["repo_name"])
        if repo_name in excluded_repos:
            continue
        if requested_repos is not None and repo_name not in requested_repos:
            continue
        if repo_name in representatives:
            continue
        source_file = args.repos_dir / str(row["file_name"]).lstrip("/")
        reconstructed = (
            str(row["prefix_code"]) + str(row["middle_code"]) + str(row["suffix_code"])
        )
        if (
            not source_file.is_file()
            or source_file.read_text(encoding="utf-8") != reconstructed
        ):
            mismatch_rows.append(str(row["_task_id"]))
            continue
        representatives[repo_name] = row
    jobs: dict[Any, str] = {}
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for repo_name, row in representatives.items():
            future = pool.submit(
                _execute_middle,
                row=row,
                middle_code=str(row["middle_code"]),
                repos_dir=args.repos_dir,
                python_executable=_python_for_repo(args, repo_name),
                timeout=args.timeout,
            )
            jobs[future] = repo_name
        for future in as_completed(jobs):
            repo_name = jobs[future]
            try:
                results[repo_name] = future.result()
            except Exception as exc:
                results[repo_name] = {
                    "compile": {"pass": False},
                    "tests": {"pass": False},
                    "pass": False,
                    "infrastructure_error": f"{type(exc).__name__}: {exc}",
                }
            print(
                f"{repo_name}: {('PASS' if results[repo_name]['pass'] else 'FAIL')}",
                flush=True,
            )
    passed = sorted((repo for (repo, result) in results.items() if result["pass"]))
    failed = sorted((repo for (repo, result) in results.items() if not result["pass"]))
    payload = {
        "summary": {
            "repos_represented": len(representatives),
            "repos_passed": len(passed),
            "repos_failed": len(failed),
            "passed_repos": passed,
            "failed_repos": failed,
            "candidate_rows_skipped_before_exact_representative": mismatch_rows,
        },
        "manifest": str(args.manifest.resolve()),
        "repos_dir": str(args.repos_dir.resolve()),
        "python": str(args.python.resolve()) if not args.python_map else None,
        "timeout_seconds": args.timeout,
        "results": {repo: results[repo] for repo in sorted(results)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload["summary"], indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    generations = [
        row
        for row in _read_jsonl(args.generations)
        if int(row["_manifest_index"]) % args.num_shards == args.shard_index
    ]
    if args.discordant_from:
        previous = json.loads(args.discordant_from.read_text(encoding="utf-8"))
        discordant_task_ids = {
            str(record["_task_id"])
            for record in previous["records"]
            if bool(record["base"]["pass"]) != bool(record["ramp"]["pass"])
        }
        generations = [
            row for row in generations if str(row["_task_id"]) in discordant_task_ids
        ]
    if args.timeouts_from:
        previous = json.loads(args.timeouts_from.read_text(encoding="utf-8"))
        timeout_task_ids = {
            str(record["_task_id"])
            for record in previous["records"]
            if _has_timeout(record["base"]) or _has_timeout(record["ramp"])
        }
        generations = [
            row for row in generations if str(row["_task_id"]) in timeout_task_ids
        ]
    reference_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    if args.trusted_reference_audit:
        audit = json.loads(args.trusted_reference_audit.read_text(encoding="utf-8"))
        passed_repos = set((str(item) for item in audit["summary"]["passed_repos"]))
        for row in generations:
            key = (
                str(row["repo_name"]),
                str(row["file_name"]),
                _sha256_text(
                    str(row["prefix_code"])
                    + str(row["middle_code"])
                    + str(row["suffix_code"])
                ),
            )
            if key in reference_cache:
                continue
            source_file = args.repos_dir / str(row["file_name"]).lstrip("/")
            reconstructed = (
                str(row["prefix_code"])
                + str(row["middle_code"])
                + str(row["suffix_code"])
            )
            exact = (
                source_file.is_file()
                and source_file.read_text(encoding="utf-8") == reconstructed
            )
            passed = str(row["repo_name"]) in passed_repos and exact
            reference_cache[key] = {
                "compile": {"pass": passed},
                "tests": {"pass": passed},
                "pass": passed,
                "verification": "exact_reconstruction_plus_pristine_repo_audit",
                "repo_audit": str(args.trusted_reference_audit.resolve()),
                "exact_reconstruction": exact,
            }
    else:
        reference_jobs: dict[Any, tuple[str, str, str]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in generations:
                key = (
                    str(row["repo_name"]),
                    str(row["file_name"]),
                    _sha256_text(
                        str(row["prefix_code"])
                        + str(row["middle_code"])
                        + str(row["suffix_code"])
                    ),
                )
                if key in reference_cache or key in reference_jobs.values():
                    continue
                future = pool.submit(
                    _execute_middle,
                    row=row,
                    middle_code=str(row["middle_code"]),
                    repos_dir=args.repos_dir,
                    python_executable=_python_for_repo(args, str(row["repo_name"])),
                    timeout=args.timeout,
                )
                reference_jobs[future] = key
            for future in as_completed(reference_jobs):
                key = reference_jobs[future]
                try:
                    reference_cache[key] = future.result()
                except Exception as exc:
                    reference_cache[key] = {
                        "compile": {"pass": False},
                        "tests": {"pass": False},
                        "pass": False,
                        "infrastructure_error": f"{type(exc).__name__}: {exc}",
                    }
    method_jobs: dict[Any, tuple[int, str]] = {}
    method_results: dict[tuple[int, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(generations):
            for method, key in (
                ("base", "base_generated_middle_code"),
                ("ramp", "ramp_generated_middle_code"),
            ):
                if (
                    method == "ramp"
                    and row["ramp_generated_middle_code"]
                    == row["base_generated_middle_code"]
                ):
                    continue
                future = pool.submit(
                    _execute_middle,
                    row=row,
                    middle_code=str(row[key]),
                    repos_dir=args.repos_dir,
                    python_executable=_python_for_repo(args, str(row["repo_name"])),
                    timeout=args.timeout,
                )
                method_jobs[future] = (index, method)
        completed_jobs = 0
        for future in as_completed(method_jobs):
            key = method_jobs[future]
            try:
                method_results[key] = future.result()
            except Exception as exc:
                method_results[key] = {
                    "compile": {"pass": False},
                    "tests": {"pass": False},
                    "pass": False,
                    "infrastructure_error": f"{type(exc).__name__}: {exc}",
                }
            completed_jobs += 1
            if completed_jobs % 10 == 0 or completed_jobs == len(method_jobs):
                print(
                    f"[evaluation {args.shard_index}/{args.num_shards}] {completed_jobs}/{len(method_jobs)} method executions",
                    flush=True,
                )
    records: list[dict[str, Any]] = []
    for index, row in enumerate(generations):
        reference_key = (
            str(row["repo_name"]),
            str(row["file_name"]),
            _sha256_text(
                str(row["prefix_code"])
                + str(row["middle_code"])
                + str(row["suffix_code"])
            ),
        )
        base = method_results[index, "base"]
        ramp = (
            base
            if row["ramp_generated_middle_code"] == row["base_generated_middle_code"]
            else method_results[index, "ramp"]
        )
        records.append(
            {
                "_task_id": row["_task_id"],
                "_source_index": row["_source_index"],
                "repo_name": row["repo_name"],
                "file_name": row["file_name"],
                "fill_type": row.get("fill_type"),
                "generation": {
                    "first_identifier": row["first_identifier"],
                    "base_first_token_id": row["base_first_token_id"],
                    "ramp_first_token_id": row["ramp_first_token_id"],
                    "first_token_changed": row["first_token_changed"],
                    "prompt_sha256": row["prompt_sha256"],
                },
                "reference": reference_cache[reference_key],
                "base": base,
                "ramp": ramp,
                "transition": _transition(base, ramp),
            }
        )
    payload = {
        "summary": _paired_summary(records),
        "generations": str(args.generations.resolve()),
        "repos_dir": str(args.repos_dir.resolve()),
        "timeout_seconds": args.timeout,
        "workers": args.workers,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "trusted_reference_audit": (
            str(args.trusted_reference_audit.resolve())
            if args.trusted_reference_audit
            else None
        ),
        "discordant_from": (
            str(args.discordant_from.resolve()) if args.discordant_from else None
        ),
        "timeouts_from": (
            str(args.timeouts_from.resolve()) if args.timeouts_from else None
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload["summary"], indent=2))


def command_merge_evaluations(args: argparse.Namespace) -> None:
    records_by_task: dict[str, dict[str, Any]] = {}
    inputs: list[str] = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        inputs.append(str(path.resolve()))
        for record in payload["records"]:
            task_id = str(record["_task_id"])
            if task_id in records_by_task:
                raise ValueError(f"Duplicate evaluated task: {task_id}")
            records_by_task[task_id] = record
    records = sorted(
        records_by_task.values(), key=lambda record: int(record["_source_index"])
    )
    output = {"summary": _paired_summary(records), "inputs": inputs, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output["summary"], indent=2))


def command_overlay_rechecks(args: argparse.Namespace) -> None:
    """Overlay low-concurrency rechecks on a primary paired evaluation."""
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    records = primary["records"]
    records_by_task: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = str(record["_task_id"])
        if task_id in records_by_task:
            raise ValueError(f"Duplicate primary task: {task_id}")
        records_by_task[task_id] = record
    source_audits: list[dict[str, Any]] = []
    unique_rechecked: set[str] = set()
    source_record_count = 0
    pair_status_agreements = 0
    transition_agreements = 0
    for path in args.rechecks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_seen: set[str] = set()
        source_pair_agreements = 0
        source_transition_agreements = 0
        for rechecked in payload["records"]:
            task_id = str(rechecked["_task_id"])
            if task_id in source_seen:
                raise ValueError(f"Duplicate recheck task in {path}: {task_id}")
            source_seen.add(task_id)
            if task_id not in records_by_task:
                raise ValueError(f"Recheck task absent from primary: {task_id}")
            original = records_by_task[task_id]
            for key in ("_source_index", "repo_name", "file_name"):
                if rechecked[key] != original[key]:
                    raise ValueError(f"Recheck metadata mismatch for {task_id}: {key}")
            for key in (
                "base_first_token_id",
                "ramp_first_token_id",
                "first_token_changed",
                "prompt_sha256",
            ):
                if rechecked["generation"][key] != original["generation"][key]:
                    raise ValueError(
                        f"Recheck generation mismatch for {task_id}: {key}"
                    )
            if bool(rechecked["reference"]["pass"]) != bool(
                original["reference"]["pass"]
            ):
                raise ValueError(f"Recheck reference mismatch for {task_id}")
            original_pair = (
                bool(original["base"]["pass"]),
                bool(original["ramp"]["pass"]),
            )
            rechecked_pair = (
                bool(rechecked["base"]["pass"]),
                bool(rechecked["ramp"]["pass"]),
            )
            pair_agrees = original_pair == rechecked_pair
            transition_agrees = original["transition"] == rechecked["transition"]
            source_pair_agreements += int(pair_agrees)
            source_transition_agreements += int(transition_agrees)
            pair_status_agreements += int(pair_agrees)
            transition_agreements += int(transition_agrees)
            source_record_count += 1
            unique_rechecked.add(task_id)
            original["base"] = rechecked["base"]
            original["ramp"] = rechecked["ramp"]
            original["transition"] = _transition(rechecked["base"], rechecked["ramp"])
        source_audits.append(
            {
                "path": str(path.resolve()),
                "records": len(source_seen),
                "pair_status_agreement_with_input_state": source_pair_agreements,
                "transition_agreement_with_input_state": source_transition_agreements,
            }
        )
    output = {
        "summary": _paired_summary(records),
        "primary": str(args.primary.resolve()),
        "recheck_audit": {
            "source_record_count": source_record_count,
            "unique_rechecked_task_count": len(unique_rechecked),
            "pair_status_agreement_count": pair_status_agreements,
            "transition_agreement_count": transition_agreements,
            "sources": source_audits,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output["summary"], indent=2))
    print(json.dumps(output["recheck_audit"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--repos", default="")
    prepare.add_argument("--max-per-repo", type=int, default=8)
    prepare.add_argument(
        "--require-source-pass", action=argparse.BooleanOptionalAction, default=True
    )
    prepare.set_defaults(func=command_prepare)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--device", default="cuda:0")
    generate.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--num-shards", type=int, default=1)
    generate.add_argument("--max-context-tokens", type=int, default=8192)
    generate.add_argument("--max-generation-tokens", type=int, default=1024)
    generate.add_argument(
        "--generation-budget-policy",
        choices=("fixed_max", "reference_target_length"),
        default="fixed_max",
    )
    generate.add_argument("--max-tokens", type=int, default=32768)
    generate.add_argument(
        "--prompt-policy",
        choices=("official_fim", "causal_prefix"),
        default="official_fim",
    )
    generate.add_argument(
        "--stop-policy", choices=("eos_only", "repository_control"), default="eos_only"
    )
    generate.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    generate.set_defaults(func=command_generate)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(func=command_merge)
    control_stop = subparsers.add_parser("control-stop")
    control_stop.add_argument("--input", type=Path, required=True)
    control_stop.add_argument("--output", type=Path, required=True)
    control_stop.set_defaults(func=command_control_stop)
    build_runnable = subparsers.add_parser("build-runnable")
    build_runnable.add_argument("--manifest", type=Path, required=True)
    build_runnable.add_argument("--repos-dir", type=Path, required=True)
    build_runnable.add_argument("--repo-audits", type=Path, nargs="+", required=True)
    build_runnable.add_argument("--output", type=Path, required=True)
    build_runnable.set_defaults(func=command_build_runnable)
    audit_repos = subparsers.add_parser("audit-repos")
    audit_repos.add_argument("--manifest", type=Path, required=True)
    audit_repos.add_argument("--repos-dir", type=Path, required=True)
    audit_repos.add_argument("--output", type=Path, required=True)
    audit_repos.add_argument("--python", type=Path, default=Path(sys.executable))
    audit_repos.add_argument("--python-map", type=Path)
    audit_repos.add_argument("--workers", type=int, default=8)
    audit_repos.add_argument("--timeout", type=int, default=240)
    audit_repos.add_argument("--repos", default="")
    audit_repos.add_argument("--exclude-repos", default="")
    audit_repos.set_defaults(func=command_audit_repos)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--generations", type=Path, required=True)
    evaluate.add_argument("--repos-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--python", type=Path, default=Path(sys.executable))
    evaluate.add_argument("--python-map", type=Path)
    evaluate.add_argument("--workers", type=int, default=8)
    evaluate.add_argument("--timeout", type=int, default=240)
    evaluate.add_argument("--shard-index", type=int, default=0)
    evaluate.add_argument("--num-shards", type=int, default=1)
    evaluate.add_argument("--trusted-reference-audit", type=Path)
    evaluate_selection = evaluate.add_mutually_exclusive_group()
    evaluate_selection.add_argument("--discordant-from", type=Path)
    evaluate_selection.add_argument("--timeouts-from", type=Path)
    evaluate.set_defaults(func=command_evaluate)
    merge_evaluations = subparsers.add_parser("merge-evaluations")
    merge_evaluations.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_evaluations.add_argument("--output", type=Path, required=True)
    merge_evaluations.set_defaults(func=command_merge_evaluations)
    overlay_rechecks = subparsers.add_parser("overlay-rechecks")
    overlay_rechecks.add_argument("--primary", type=Path, required=True)
    overlay_rechecks.add_argument("--rechecks", type=Path, nargs="+", required=True)
    overlay_rechecks.add_argument("--output", type=Path, required=True)
    overlay_rechecks.set_defaults(func=command_overlay_rechecks)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

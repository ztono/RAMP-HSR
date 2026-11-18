from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import statistics
from typing import Any, Iterator, Sequence
from transformers import AutoTokenizer
from scripts._prepare_public_first_identifier_benchmarks import _first_nonkeyword_name
from ramp_hsr.identifiers import EVALUATOR_VERSION, audit_reference_continuation_path

PROTOCOL = "reviewer3-e3-public-protocol-inputs-v1"
COHORT_PROTOCOL = "reviewer3-e3-common-public-cohort-v1"
MAX_CONTEXT_TOKENS = 512
DATASETS = ("crosscodeeval_python", "repobench_python", "crosscodeeval_typescript")
SETTINGS = {
    "qwen_python": ("python", "qwen"),
    "starcoder2_python": ("python", "starcoder2"),
    "qwen_typescript": ("typescript", "qwen"),
    "starcoder2_typescript": ("typescript", "starcoder2"),
}
_WORKER_TOKENIZERS: dict[str, Any] = {}
_WORKER_HISTORICAL: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_ids(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "sha256": _sha256_file(resolved),
    }


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def _iter_shards(root: Path, language_or_setting: str) -> Iterator[dict[str, Any]]:
    shard_root = root / language_or_setting / "shards"
    paths = sorted(shard_root.glob("input.shard_*_of_7.jsonl"))
    if len(paths) != 7:
        raise FileNotFoundError(
            f"expected seven input shards under {shard_root}, found {len(paths)}"
        )
    for path in paths:
        yield from _iter_jsonl(path)


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return dict(value) if isinstance(value, dict) else {}


def _dataset(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    value = metadata.get("source_dataset") or row.get("dataset")
    if not value:
        raise ValueError("row lacks source_dataset")
    return str(value)


def _source_index(row: dict[str, Any]) -> int:
    metadata = _metadata(row)
    value = metadata.get("source_index")
    if value is None:
        raise ValueError("row lacks metadata.source_index")
    return int(value)


def _language_for_dataset(dataset: str) -> str:
    return "typescript" if dataset.endswith("typescript") else "python"


def _repo_id(row: dict[str, Any], dataset: str) -> str:
    metadata = _metadata(row)
    value = (
        row.get("repo_id")
        or metadata.get("repository")
        or metadata.get("repo_name")
        or dataset
    )
    return str(value)


def _file_path(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    return str(row.get("file_path") or metadata.get("file") or "")


def _leading_category(value: str) -> str:
    if value == "":
        return "identifier_at_benchmark_endpoint"
    if value.strip(" \t") == "":
        return "horizontal_whitespace_only"
    if "\n" in value or "\r" in value:
        return "contains_newline"
    return "contains_nonwhitespace_reference_code"


def canonicalize_original_row(
    row: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert one original-prefix row without consulting model outcomes."""
    dataset = _dataset(row)
    if dataset not in DATASETS:
        return (None, "dataset_out_of_scope")
    language = _language_for_dataset(dataset)
    prompt = str(row.get("prompt") or "")
    target = str(row.get("target") or "")
    if not prompt:
        return (None, "empty_original_prefix")
    if not target:
        return (None, "empty_target")
    found = _first_nonkeyword_name(target, language=language)
    if found is None:
        return (None, "no_first_nonkeyword_identifier")
    (cut, identifier, token_start) = found
    if cut < 0 or cut + len(identifier) > len(target):
        return (None, "invalid_identifier_span")
    leading = target[:cut]
    identifier_slice = target[cut : cut + len(identifier)]
    if identifier_slice != identifier:
        return (None, "identifier_span_text_mismatch")
    source_index = _source_index(row)
    sample_id = f"{dataset}:{source_index}"
    right_context = str(row.get("right_context") or "")
    full_reference = prompt + target + right_context
    return (
        {
            "protocol": COHORT_PROTOCOL,
            "canonical_sample_id": sample_id,
            "dataset": dataset,
            "language": language,
            "source_index": source_index,
            "benchmark_sample_id": str(row.get("sample_id") or sample_id),
            "repository": _repo_id(row, dataset),
            "file": _file_path(row),
            "original_prefix": prompt,
            "benchmark_target": target,
            "right_context": right_context,
            "original_prefix_sha256": _sha256_text(prompt),
            "benchmark_target_sha256": _sha256_text(target),
            "full_reference_sha256": _sha256_text(full_reference),
            "target_identifier": identifier,
            "target_identifier_start_in_target": int(cut),
            "target_identifier_end_in_target": int(cut + len(identifier)),
            "target_identifier_token_start": [int(token_start[0]), int(token_start[1])],
            "reference_leading_text": leading,
            "reference_leading_text_sha256": _sha256_text(leading),
            "reference_characters_added": len(leading),
            "leading_text_category": _leading_category(leading),
            "original_endpoint_absolute_character": len(prompt),
            "target_aligned_endpoint_absolute_character": len(prompt) + len(leading),
            "target_identifier_absolute_character": len(prompt) + len(leading),
            "target_aligned_prefix_sha256": _sha256_text(prompt + leading),
            "target_aligned_adds_identifier_characters": False,
            "target_aligned_adds_only_pre_identifier_reference": True,
        },
        None,
    )


def _load_historical_first_inputs(
    root: Path, setting: str
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _iter_shards(root, setting):
        dataset = _dataset(row)
        if dataset not in DATASETS:
            continue
        if _language_for_dataset(dataset) != SETTINGS[setting][0]:
            continue
        key = (dataset, _source_index(row))
        if key in result:
            raise ValueError(f"duplicate historical first-id input {setting}/{key}")
        metadata = _metadata(row)
        result[key] = {
            "prompt": str(row.get("prompt") or ""),
            "target": str(row.get("target") or ""),
            "first_identifier": str(metadata.get("first_identifier") or ""),
            "cut_offset": metadata.get("cut_offset"),
            "record_id": metadata.get("record_id"),
            "prompt_protocol": metadata.get("prompt_protocol"),
            "prompt_audit_variant": metadata.get("repobench_prompt_audit_variant"),
        }
    return result


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]


def _empty_prefix_fallback_id(tokenizer: Any) -> int:
    for value in (tokenizer.bos_token_id, tokenizer.eos_token_id):
        if value is not None:
            return int(value)
    return 0


def _input_window(tokenizer: Any, prefix: str) -> dict[str, Any]:
    full_ids = _token_ids(tokenizer, prefix)
    used_fallback = not full_ids
    model_ids = full_ids or [_empty_prefix_fallback_id(tokenizer)]
    context_ids = model_ids[-MAX_CONTEXT_TOKENS:]
    return {
        "full_character_count": len(prefix),
        "full_prefix_token_count": len(full_ids),
        "empty_prefix_fallback": used_fallback,
        "left_truncated_token_count": max(0, len(model_ids) - len(context_ids)),
        "token_ids": context_ids,
        "token_ids_sha256": _sha256_ids(context_ids),
        "decoded_window_with_special_tokens": tokenizer.decode(
            context_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
    }


def setting_record(
    canonical: dict[str, Any],
    *,
    e3_sample_index: int,
    setting: str,
    tokenizer: Any,
    historical: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create exact model/tokenizer inputs for one common cohort row."""
    original_prefix = str(canonical["original_prefix"])
    leading = str(canonical["reference_leading_text"])
    aligned_prefix = original_prefix + leading
    target = str(canonical["benchmark_target"])
    identifier = str(canonical["target_identifier"])
    original_target_ids = _token_ids(tokenizer, target)
    aligned_target_ids = _token_ids(tokenizer, identifier)
    if not original_target_ids or not aligned_target_ids:
        raise ValueError(f"{canonical['canonical_sample_id']}: empty token target")
    original_window = _input_window(tokenizer, original_prefix)
    aligned_window = _input_window(tokenizer, aligned_prefix)
    reference_path = audit_reference_continuation_path(
        tokenizer=tokenizer,
        full_character_prefix=aligned_prefix,
        remaining_identifier_suffix=identifier,
    )
    if historical is None:
        historical_audit: dict[str, Any] = {
            "present": False,
            "prompt_exact": False,
            "effective_512_token_window_exact": False,
            "identifier_exact": False,
            "target_text_exact": False,
        }
    else:
        historical_prompt = str(historical["prompt"])
        prompt_exact = historical_prompt == aligned_prefix
        if prompt_exact:
            historical_window_ids = aligned_window["token_ids"]
        else:
            historical_window_ids = _input_window(tokenizer, historical_prompt)[
                "token_ids"
            ]
        historical_audit = {
            "present": True,
            "prompt_exact": prompt_exact,
            "effective_512_token_window_exact": list(historical_window_ids)
            == list(aligned_window["token_ids"]),
            "identifier_exact": str(historical["first_identifier"]) == identifier,
            "target_text_exact": str(historical["target"])
            == target[int(canonical["target_identifier_start_in_target"]) :],
            "historical_prompt_sha256": _sha256_text(historical_prompt),
            "historical_prompt_character_count": len(historical_prompt),
            "historical_record_id": historical.get("record_id"),
            "historical_cut_offset": historical.get("cut_offset"),
            "historical_prompt_protocol": historical.get("prompt_protocol"),
            "historical_prompt_audit_variant": historical.get("prompt_audit_variant"),
        }
    return {
        "protocol": PROTOCOL,
        "evaluator_version": EVALUATOR_VERSION,
        "setting": setting,
        "language": canonical["language"],
        "e3_sample_index": int(e3_sample_index),
        "canonical_sample_id": canonical["canonical_sample_id"],
        "dataset": canonical["dataset"],
        "source_index": int(canonical["source_index"]),
        "repository": canonical["repository"],
        "file": canonical["file"],
        "full_reference_sha256": canonical["full_reference_sha256"],
        "target_identifier": identifier,
        "reference_leading_text": leading,
        "reference_leading_text_sha256": canonical["reference_leading_text_sha256"],
        "reference_characters_added": int(canonical["reference_characters_added"]),
        "leading_text_category": canonical["leading_text_category"],
        "target_identifier_start_in_target": int(
            canonical["target_identifier_start_in_target"]
        ),
        "target_identifier_end_in_target": int(
            canonical["target_identifier_end_in_target"]
        ),
        "input_conditions": {
            "benchmark_original_prefix": original_window,
            "target_aligned_prefix": aligned_window,
        },
        "reference_conditions": {
            "benchmark_original_prefix": {
                "prediction_object": "first token at benchmark-provided endpoint",
                "first_token_id": int(original_target_ids[0]),
                "independent_target_token_ids": original_target_ids,
            },
            "target_aligned_prefix": {
                "prediction_object": "first token and complete identifier at target-aligned endpoint",
                "first_token_id": int(aligned_target_ids[0]),
                "independent_identifier_token_ids": aligned_target_ids,
                "reference_path": reference_path,
            },
        },
        "protocol_token_reference_same": int(original_target_ids[0])
        == int(aligned_target_ids[0]),
        "protocol_input_window_same": original_window["token_ids"]
        == aligned_window["token_ids"],
        "historical_target_aligned_input": historical_audit,
    }


def _setting_record_worker(
    canonical: dict[str, Any], e3_sample_index: int, setting: str, tokenizer_family: str
) -> dict[str, Any]:
    """Fork-worker entry point; all mutable work remains sample local."""
    key = (str(canonical["dataset"]), int(canonical["source_index"]))
    return setting_record(
        canonical,
        e3_sample_index=e3_sample_index,
        setting=setting,
        tokenizer=_WORKER_TOKENIZERS[tokenizer_family],
        historical=_WORKER_HISTORICAL[setting].get(key),
    )


class _GzipJsonlWriter:

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temp = path.with_suffix(path.suffix + ".tmp")
        self.handle: Any = None
        self.count = 0

    def __enter__(self) -> "_GzipJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.temp, "wt", encoding="utf-8", compresslevel=3)
        return self

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        self.handle.write("\n")
        self.count += 1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.close()
        if exc_type is None:
            os.replace(self.temp, self.path)
        elif self.temp.exists():
            self.temp.unlink()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = round((len(ordered) - 1) * fraction)
        return int(ordered[index])

    return {
        "count": len(values),
        "min": int(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": int(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }


def run(
    *,
    original_root: Path,
    historical_first_root: Path,
    qwen_tokenizer_path: Path,
    starcoder2_tokenizer_path: Path,
    output_root: Path,
    workers: int = 8,
    pending_rows: int = 64,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if pending_rows < 1:
        raise ValueError("pending_rows must be positive")
    tokenizers = {
        "qwen": AutoTokenizer.from_pretrained(
            qwen_tokenizer_path, local_files_only=True, trust_remote_code=False
        ),
        "starcoder2": AutoTokenizer.from_pretrained(
            starcoder2_tokenizer_path, local_files_only=True, trust_remote_code=False
        ),
    }
    for tokenizer in tokenizers.values():
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    historical = {
        setting: _load_historical_first_inputs(historical_first_root, setting)
        for setting in SETTINGS
    }
    _WORKER_TOKENIZERS.clear()
    _WORKER_TOKENIZERS.update(tokenizers)
    _WORKER_HISTORICAL.clear()
    _WORKER_HISTORICAL.update(historical)
    counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    leading_categories: dict[str, Counter[str]] = defaultdict(Counter)
    leading_lengths: dict[str, list[int]] = defaultdict(list)
    setting_audit: dict[str, Counter[str]] = {
        setting: Counter() for setting in SETTINGS
    }
    selected_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[tuple[str, int]] = set()
    cohort_path = output_root / "common_cohort.jsonl.gz"
    unmapped_path = output_root / "unmapped_original_rows.jsonl.gz"
    setting_paths = {
        setting: output_root / setting / "manifest.jsonl.gz" for setting in SETTINGS
    }
    with _GzipJsonlWriter(cohort_path) as cohort_writer, _GzipJsonlWriter(
        unmapped_path
    ) as unmapped_writer, _GzipJsonlWriter(
        setting_paths["qwen_python"]
    ) as qwen_py_writer, _GzipJsonlWriter(
        setting_paths["starcoder2_python"]
    ) as star_py_writer, _GzipJsonlWriter(
        setting_paths["qwen_typescript"]
    ) as qwen_ts_writer, _GzipJsonlWriter(
        setting_paths["starcoder2_typescript"]
    ) as star_ts_writer, ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork")
    ) as executor:
        writers = {
            "qwen_python": qwen_py_writer,
            "starcoder2_python": star_py_writer,
            "qwen_typescript": qwen_ts_writer,
            "starcoder2_typescript": star_ts_writer,
        }
        global_index = 0

        def commit_record_bundle(
            canonical: dict[str, Any], futures: list[tuple[str, Future[dict[str, Any]]]]
        ) -> None:
            """Write one row after its tokenizer jobs finish, preserving source order."""
            cohort_writer.write(canonical)
            dataset = str(canonical["dataset"])
            dataset_counts[dataset] += 1
            category = str(canonical["leading_text_category"])
            leading_categories[dataset][category] += 1
            leading_lengths[dataset].append(
                int(canonical["reference_characters_added"])
            )
            category_example = f"leading:{category}"
            if len(selected_examples[category_example]) < 3:
                selected_examples[category_example].append(
                    {
                        "canonical_sample_id": canonical["canonical_sample_id"],
                        "dataset": dataset,
                        "reference_leading_text": canonical["reference_leading_text"],
                        "target_identifier": canonical["target_identifier"],
                        "selection_rule": "first three cohort rows in deterministic source order; selected from input fields before inference",
                    }
                )
            for setting, future in futures:
                record = future.result()
                writers[setting].write(record)
                audit = setting_audit[setting]
                audit["rows"] += 1
                audit["protocol_token_reference_same"] += int(
                    record["protocol_token_reference_same"]
                )
                audit["protocol_input_window_same"] += int(
                    record["protocol_input_window_same"]
                )
                hist_audit = record["historical_target_aligned_input"]
                audit["historical_present"] += int(hist_audit["present"])
                audit["historical_prompt_exact"] += int(hist_audit["prompt_exact"])
                audit["historical_effective_window_exact"] += int(
                    hist_audit["effective_512_token_window_exact"]
                )
                audit["historical_identifier_exact"] += int(
                    hist_audit["identifier_exact"]
                )
                audit["historical_target_text_exact"] += int(
                    hist_audit["target_text_exact"]
                )
                path_category = record["reference_conditions"]["target_aligned_prefix"][
                    "reference_path"
                ]["category"]
                audit[f"reference_path:{path_category}"] += 1
                for flag, example_category in (
                    (
                        hist_audit["present"]
                        and (not hist_audit["prompt_exact"])
                        and hist_audit["effective_512_token_window_exact"],
                        f"{setting}:historical_prompt_diff_effective_window_same",
                    ),
                    (
                        hist_audit["present"]
                        and (not hist_audit["effective_512_token_window_exact"]),
                        f"{setting}:historical_effective_window_diff",
                    ),
                    (
                        not record["protocol_token_reference_same"],
                        f"{setting}:protocol_reference_token_diff",
                    ),
                ):
                    if flag and len(selected_examples[example_category]) < 3:
                        selected_examples[example_category].append(
                            {
                                "canonical_sample_id": canonical["canonical_sample_id"],
                                "dataset": dataset,
                                "reference_leading_text": canonical[
                                    "reference_leading_text"
                                ],
                                "target_identifier": canonical["target_identifier"],
                                "selection_rule": "first three matching input-only rows in deterministic source order",
                            }
                        )

        for language in ("python", "typescript"):
            pending: deque[
                tuple[dict[str, Any], list[tuple[str, Future[dict[str, Any]]]]]
            ] = deque()
            for raw_row in _iter_shards(original_root, language):
                dataset = _dataset(raw_row)
                if dataset not in DATASETS:
                    continue
                key = (dataset, _source_index(raw_row))
                if key in seen_keys:
                    raise ValueError(f"duplicate original benchmark row {key}")
                seen_keys.add(key)
                (canonical, reason) = canonicalize_original_row(raw_row)
                if canonical is None:
                    skip_counts[str(reason)] += 1
                    unmapped_writer.write(
                        {
                            "protocol": PROTOCOL,
                            "dataset": dataset,
                            "source_index": key[1],
                            "reason": reason,
                            "benchmark_sample_id": raw_row.get("sample_id"),
                            "repository": _repo_id(raw_row, dataset),
                            "file": _file_path(raw_row),
                        }
                    )
                    continue
                canonical["e3_sample_index"] = global_index
                row_futures: list[tuple[str, Future[dict[str, Any]]]] = []
                for setting, (setting_language, tokenizer_family) in SETTINGS.items():
                    if setting_language != language:
                        continue
                    future = executor.submit(
                        _setting_record_worker,
                        canonical,
                        global_index,
                        setting,
                        tokenizer_family,
                    )
                    row_futures.append((setting, future))
                pending.append((canonical, row_futures))
                global_index += 1
                if len(pending) >= pending_rows:
                    commit_record_bundle(*pending.popleft())
            while pending:
                commit_record_bundle(*pending.popleft())
        counts["cohort_rows"] = global_index
    language_expected = {
        "python": sum(
            (
                value
                for (dataset, value) in dataset_counts.items()
                if _language_for_dataset(dataset) == "python"
            )
        ),
        "typescript": sum(
            (
                value
                for (dataset, value) in dataset_counts.items()
                if _language_for_dataset(dataset) == "typescript"
            )
        ),
    }
    for setting, (language, _) in SETTINGS.items():
        if setting_audit[setting]["rows"] != language_expected[language]:
            raise AssertionError(f"{setting}: incomplete setting manifest")
    report = {
        "status": "complete",
        "protocol": PROTOCOL,
        "cohort_protocol": COHORT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "scope": list(DATASETS),
        "eligibility": "Defined only from archived benchmark prompt/target text: nonempty original prefix and target, with a lexically located first non-keyword identifier. No model prediction, route activation, or editor outcome participates in selection.",
        "cohort_rows": int(counts["cohort_rows"]),
        "dataset_rows": dict(sorted(dataset_counts.items())),
        "unmapped_rows": int(sum(skip_counts.values())),
        "unmapped_reason_counts": dict(sorted(skip_counts.items())),
        "leading_text": {
            dataset: {
                "category_counts": dict(sorted(leading_categories[dataset].items())),
                "added_character_distribution": _distribution(leading_lengths[dataset]),
            }
            for dataset in DATASETS
        },
        "settings": {
            setting: {
                **dict(sorted(counter.items())),
                "historical_missing": int(
                    counter["rows"] - counter["historical_present"]
                ),
            }
            for (setting, counter) in setting_audit.items()
        },
        "contract": {
            "original_condition": "archived benchmark-provided prefix",
            "target_aligned_condition": "the same prefix plus exactly target[:first_identifier_start]",
            "reference_identifier_in_model_input": False,
            "future_after_identifier_in_model_input": False,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "truncation_side": "left",
            "tokenizer_swap": False,
            "same_text_and_sample_ids_across_models": True,
            "historical_inputs_do_not_control_new_cohort_eligibility": True,
            "parallel_tokenization_workers": int(workers),
            "parallel_pending_rows": int(pending_rows),
            "parallel_backend": "forked_process_pool",
            "parallel_results_committed_in_source_order": True,
        },
        "tokenizers": {
            "qwen": {
                "requested_path": str(qwen_tokenizer_path.resolve()),
                "class": type(tokenizers["qwen"]).__name__,
                "vocab_size": len(tokenizers["qwen"]),
            },
            "starcoder2": {
                "requested_path": str(starcoder2_tokenizer_path.resolve()),
                "class": type(tokenizers["starcoder2"]).__name__,
                "vocab_size": len(tokenizers["starcoder2"]),
            },
        },
        "inputs": {
            "original_root": str(original_root.resolve()),
            "historical_first_root": str(historical_first_root.resolve()),
        },
        "outputs": {
            "common_cohort": _fingerprint(cohort_path),
            "unmapped_original_rows": _fingerprint(unmapped_path),
            "setting_manifests": {
                setting: _fingerprint(path) for (setting, path) in setting_paths.items()
            },
        },
        "pre_outcome_examples": dict(sorted(selected_examples.items())),
    }
    _write_json(output_root / "input_audit.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--historical-first-root", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer", type=Path, required=True)
    parser.add_argument("--starcoder2-tokenizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pending-rows", type=int, default=64)
    args = parser.parse_args(argv)
    report = run(
        original_root=args.original_root.expanduser().resolve(),
        historical_first_root=args.historical_first_root.expanduser().resolve(),
        qwen_tokenizer_path=args.qwen_tokenizer.expanduser().resolve(),
        starcoder2_tokenizer_path=args.starcoder2_tokenizer.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        workers=args.workers,
        pending_rows=args.pending_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

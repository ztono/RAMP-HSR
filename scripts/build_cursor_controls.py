from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence
from transformers import AutoTokenizer
from ramp_hsr.identifiers import EVALUATOR_VERSION, audit_reference_continuation_path

PROTOCOL = "reviewer3-e1-paired-cursor-inputs-v1"
CONDITION_PROTOCOLS = {
    "original": "reviewer3-e1-same-original-cursor-v2",
    "identifier_start": "reviewer3-e1-paired-cursor-identifier-start-v1",
    "native_internal": "reviewer3-e1-paired-cursor-native-internal-v1",
}
MODEL_KEYS = ("qwen", "starcoder2")
SETTING_BY_LANGUAGE = {
    "python": {"qwen": "qwen_python", "starcoder2": "starcoder2_python"},
    "typescript": {"qwen": "qwen_typescript", "starcoder2": "starcoder2_typescript"},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_ids(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _read_jsonl_gzip(path: Path) -> Iterable[dict[str, Any]]:
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


def _write_jsonl_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def _load_fixed_rows(manifest: Path, id_path: Path) -> list[dict[str, Any]]:
    fixed_ids = [
        line.strip()
        for line in id_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(fixed_ids) != 5000 or len(set(fixed_ids)) != len(fixed_ids):
        raise ValueError(f"expected 5,000 unique fixed IDs in {id_path}")
    wanted = set(fixed_ids)
    found = {
        str(row["canonical_sample_id"]): row
        for row in _read_jsonl_gzip(manifest)
        if str(row["canonical_sample_id"]) in wanted
    }
    missing = [sample_id for sample_id in fixed_ids if sample_id not in found]
    if missing:
        raise ValueError(
            f"missing {len(missing)} fixed occurrences; first={missing[:3]}"
        )
    return [found[sample_id] for sample_id in fixed_ids]


def _load_required_contents(sidecar: Path, required_hashes: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    with sidecar.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_hash = str(row.get("sha256") or "").lower()
            if source_hash not in required_hashes:
                continue
            content = row.get("content")
            if not isinstance(content, str):
                raise ValueError(f"non-string content at {sidecar}:{line_number}")
            if _sha256_text(content) != source_hash:
                raise ValueError(f"content hash mismatch at {sidecar}:{line_number}")
            prior = result.get(source_hash)
            if prior is not None and prior != content:
                raise ValueError(f"SHA-256 collision for {source_hash}")
            result[source_hash] = content
    missing = required_hashes.difference(result)
    if missing:
        raise ValueError(
            f"sidecar lacks {len(missing)} required contents; first={sorted(missing)[:3]}"
        )
    return result


def _encode_continuous(
    tokenizer: Any, source: str
) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        source,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for (start, end) in encoded["offset_mapping"]]
    if len(token_ids) != len(offsets):
        raise ValueError("token IDs and offset mapping have different lengths")
    return (token_ids, offsets)


def native_internal_boundaries(
    *,
    tokenizer: Any,
    source: str,
    token_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    identifier_start: int,
    identifier: str,
) -> dict[int, dict[str, Any]]:
    """Return exact causal-prefix-compatible boundaries keyed by split offset."""
    identifier_end = identifier_start + len(identifier)
    candidates: dict[int, dict[str, Any]] = {}
    for token_index in range(len(offsets) - 1):
        (left_start, left_end) = offsets[token_index]
        (right_start, right_end) = offsets[token_index + 1]
        if not identifier_start < left_end < identifier_end:
            continue
        if (
            right_start != left_end
            or left_start >= left_end
            or right_start >= right_end
        ):
            continue
        evaluation_prefix = source[:left_end]
        causal_prefix_ids = [
            int(value)
            for value in tokenizer.encode(evaluation_prefix, add_special_tokens=False)
        ]
        continuous_prefix_ids = [int(value) for value in token_ids[: token_index + 1]]
        if causal_prefix_ids != continuous_prefix_ids:
            continue
        if any((end > left_end for (_start, end) in offsets[: token_index + 1])):
            raise ValueError("continuous prefix contains a token crossing the cursor")
        candidates[left_end - identifier_start] = {
            "character_cursor": left_end,
            "continuous_next_token_index": token_index + 1,
            "continuous_previous_token_id": int(token_ids[token_index]),
            "continuous_next_token_id": int(token_ids[token_index + 1]),
            "continuous_previous_token_offset": [left_start, left_end],
            "continuous_next_token_offset": [right_start, right_end],
            "prefix_token_count": len(causal_prefix_ids),
            "prefix_token_ids_sha256": _sha256_ids(causal_prefix_ids),
            "causal_prefix_equals_continuous_prefix_ids": True,
            "no_token_crosses_cursor": True,
        }
    return candidates


def choose_common_boundary(
    qwen_offsets: set[int], starcoder2_offsets: set[int], original_split_offset: int
) -> int | None:
    common = qwen_offsets & starcoder2_offsets
    if not common:
        return None
    return min(common, key=lambda offset: (abs(offset - original_split_offset), offset))


def _line_and_column(prefix: str) -> tuple[int, int]:
    return (prefix.count("\n") + 1, len(prefix.rsplit("\n", 1)[-1]))


def transform_cursor_row(
    row: dict[str, Any],
    *,
    source: str,
    split_offset: int,
    cursor_condition: str,
    condition_index: int,
    native_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identifier = str(row["target_identifier"])
    identifier_start = int(row["identifier_start_absolute_position"])
    if not 0 <= split_offset < len(identifier):
        raise ValueError(f"invalid split {split_offset} for {identifier!r}")
    evaluation_cursor = identifier_start + split_offset
    prefix = source[:evaluation_cursor]
    (line, column) = _line_and_column(prefix)
    result = {
        **row,
        "e1_cursor_protocol": PROTOCOL,
        "cursor_condition": cursor_condition,
        "source_original_cursor_absolute_position": int(
            row["original_cursor_absolute_position"]
        ),
        "evaluation_cursor_absolute_position": evaluation_cursor,
        "absolute_position": evaluation_cursor,
        "prefix": prefix,
        "target_split_offset": split_offset,
        "target": identifier[split_offset:],
        "remaining_identifier_suffix": identifier[split_offset:],
        "existing_identifier_prefix": identifier[:split_offset],
        "cursor_kind": (
            "identifier_start" if split_offset == 0 else "identifier_internal"
        ),
        "target_length": len(identifier) - split_offset,
        "line": line,
        "col": column,
        "e1_sample_index": condition_index,
        "original_e1_sample_index": int(row["e1_sample_index"]),
    }
    if native_metadata is not None:
        result["native_boundary_audit"] = native_metadata
    return result


def _make_independent_reference(
    *, setting: str, language: str, sample: dict[str, Any], tokenizer: Any
) -> dict[str, Any]:
    result = audit_reference_continuation_path(
        tokenizer=tokenizer,
        full_character_prefix=str(sample["prefix"]),
        remaining_identifier_suffix=str(sample["remaining_identifier_suffix"]),
    )
    suffix_ids = [int(value) for value in result["suffix_token_ids"]]
    if not suffix_ids:
        raise ValueError(
            f"empty independent reference for {sample['canonical_sample_id']}"
        )
    return {
        "protocol": "reviewer3-e1-cursor-reference-audit-v1",
        "setting": setting,
        "language": language,
        "cursor_condition": str(sample["cursor_condition"]),
        "e1_sample_index": int(sample["e1_sample_index"]),
        "canonical_sample_id": str(sample["canonical_sample_id"]),
        "repo_id": str(sample["repo_id"]),
        "file_path": str(sample["file_path"]),
        "full_file_sha256": str(sample["full_file_sha256"]),
        "target_identifier": str(sample["target_identifier"]),
        "existing_identifier_prefix": str(sample["existing_identifier_prefix"]),
        "remaining_identifier_suffix": str(sample["remaining_identifier_suffix"]),
        "cursor_kind": str(sample["cursor_kind"]),
        "reference_target_mode": "independent_remaining_suffix",
        "reference_token_ids": suffix_ids,
        "old_reference_first_token_id": suffix_ids[0],
        "reference_path": result,
    }


def _make_native_reference(
    *,
    setting: str,
    language: str,
    sample: dict[str, Any],
    tokenizer: Any,
    source: str,
    token_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    identifier_end = int(sample["identifier_start_absolute_position"]) + len(
        str(sample["target_identifier"])
    )
    next_index = int(boundary["continuous_next_token_index"])
    completion_end_index = None
    for token_index in range(next_index, len(offsets)):
        if offsets[token_index][1] >= identifier_end:
            completion_end_index = token_index
            break
    if completion_end_index is None:
        raise ValueError(
            f"identifier end is absent from token offsets for {sample['canonical_sample_id']}"
        )
    reference_ids = [
        int(value) for value in token_ids[next_index : completion_end_index + 1]
    ]
    if not reference_ids:
        raise ValueError(f"empty native reference for {sample['canonical_sample_id']}")
    prefix_ids = [
        int(value)
        for value in tokenizer.encode(str(sample["prefix"]), add_special_tokens=False)
    ]
    continuous_prefix_ids = [int(value) for value in token_ids[:next_index]]
    if prefix_ids != continuous_prefix_ids:
        raise ValueError("selected native prefix no longer matches continuous IDs")
    path_ids = [*prefix_ids, *reference_ids]
    covered_end = int(offsets[completion_end_index][1])
    decoded = tokenizer.decode(
        path_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    intended = source[:covered_end]
    reconstruction_exact = decoded == intended
    path = {
        "category": (
            "native_continuous_path_reconstructs_characters"
            if reconstruction_exact
            else "native_continuous_path_tokenizer_normalization"
        ),
        "failure_cause": (
            None
            if reconstruction_exact
            else "tokenizer_normalization_of_continuous_source"
        ),
        "prefix_token_count": len(prefix_ids),
        "suffix_token_ids": reference_ids,
        "continuous_identifier_completion_token_ids": reference_ids,
        "continuous_next_token_id": reference_ids[0],
        "continuous_next_token_index": next_index,
        "continuous_completion_end_token_index": completion_end_index,
        "continuous_completion_covered_character_end": covered_end,
        "character_reconstruction_exact": reconstruction_exact,
        "same_as_continuous_token_path": True,
        "causal_prefix_equals_continuous_prefix_ids": True,
        "no_input_token_crosses_cursor": True,
        "decoded_path_sha256": _sha256_text(decoded),
        "intended_path_sha256": _sha256_text(intended),
    }
    return {
        "protocol": "reviewer3-e1-cursor-reference-audit-v1",
        "setting": setting,
        "language": language,
        "cursor_condition": "native_internal",
        "e1_sample_index": int(sample["e1_sample_index"]),
        "canonical_sample_id": str(sample["canonical_sample_id"]),
        "repo_id": str(sample["repo_id"]),
        "file_path": str(sample["file_path"]),
        "full_file_sha256": str(sample["full_file_sha256"]),
        "target_identifier": str(sample["target_identifier"]),
        "existing_identifier_prefix": str(sample["existing_identifier_prefix"]),
        "remaining_identifier_suffix": str(sample["remaining_identifier_suffix"]),
        "cursor_kind": "identifier_internal",
        "reference_target_mode": "continuous_source_next_token",
        "reference_token_ids": reference_ids,
        "independent_suffix_token_ids": [
            int(value)
            for value in tokenizer.encode(
                str(sample["remaining_identifier_suffix"]), add_special_tokens=False
            )
        ],
        "old_reference_first_token_id": reference_ids[0],
        "reference_path": path,
    }


def _subset_existing_references(
    *, path: Path, fixed_ids: list[str]
) -> list[dict[str, Any]]:
    wanted = set(fixed_ids)
    found = {
        str(row["canonical_sample_id"]): row
        for row in _read_jsonl_gzip(path)
        if str(row["canonical_sample_id"]) in wanted
    }
    if set(found) != wanted:
        raise ValueError(f"existing reference audit does not cover fixed IDs: {path}")
    result = []
    for index, sample_id in enumerate(fixed_ids):
        row = dict(found[sample_id])
        row["e1_sample_index"] = index
        row["cursor_condition"] = "original"
        row["reference_target_mode"] = "independent_remaining_suffix"
        row["reference_token_ids"] = list(row["reference_path"]["suffix_token_ids"])
        result.append(row)
    return result


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def build_language(
    *,
    language: str,
    cohort_dir: Path,
    reference_root: Path,
    sidecar: Path,
    tokenizers: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    manifest = cohort_dir / language / "e1_original_cursor.jsonl.gz"
    fixed_id_path = cohort_dir / language / "e1_cursor_comparison_occurrence_ids.txt"
    original_rows = _load_fixed_rows(manifest, fixed_id_path)
    fixed_ids = [str(row["canonical_sample_id"]) for row in original_rows]
    required_hashes = {str(row["full_file_sha256"]).lower() for row in original_rows}
    contents = _load_required_contents(sidecar, required_hashes)
    transformed_original: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    native_rows_by_original_index: dict[int, dict[str, Any]] = {}
    native_references_by_original_index: dict[int, dict[str, dict[str, Any]]] = {}
    ineligible_by_original_index: dict[int, dict[str, Any]] = {}
    availability = Counter()
    relation_to_original = Counter()
    chosen_offsets = Counter()
    rows_by_hash: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for original_index, row in enumerate(original_rows):
        source_hash = str(row["full_file_sha256"]).lower()
        source = contents[source_hash]
        identifier = str(row["target_identifier"])
        identifier_start = int(row["identifier_start_absolute_position"])
        original_cursor = int(row["original_cursor_absolute_position"])
        if source[identifier_start : identifier_start + len(identifier)] != identifier:
            raise ValueError(f"source target mismatch for {row['canonical_sample_id']}")
        if source[:original_cursor] != str(row["prefix"]):
            raise ValueError(f"source prefix mismatch for {row['canonical_sample_id']}")
        transformed_original.append(
            transform_cursor_row(
                row,
                source=source,
                split_offset=int(row["target_split_offset"]),
                cursor_condition="original",
                condition_index=original_index,
            )
        )
        start_rows.append(
            transform_cursor_row(
                row,
                source=source,
                split_offset=0,
                cursor_condition="identifier_start",
                condition_index=original_index,
            )
        )
        rows_by_hash[source_hash].append((original_index, row))
    for source_hash, file_rows in rows_by_hash.items():
        source = contents[source_hash]
        encoded_by_model = {
            model_key: _encode_continuous(tokenizer, source)
            for (model_key, tokenizer) in tokenizers.items()
        }
        for original_index, row in file_rows:
            identifier = str(row["target_identifier"])
            identifier_start = int(row["identifier_start_absolute_position"])
            boundaries_by_model: dict[str, dict[int, dict[str, Any]]] = {}
            for model_key, tokenizer in tokenizers.items():
                (token_ids, offsets) = encoded_by_model[model_key]
                boundaries = native_internal_boundaries(
                    tokenizer=tokenizer,
                    source=source,
                    token_ids=token_ids,
                    offsets=offsets,
                    identifier_start=identifier_start,
                    identifier=identifier,
                )
                boundaries_by_model[model_key] = boundaries
                availability[f"{model_key}_has_internal_boundary"] += bool(boundaries)
                availability[f"original_cursor_is_{model_key}_native_boundary"] += (
                    int(row["target_split_offset"]) in boundaries
                )
            chosen = choose_common_boundary(
                set(boundaries_by_model["qwen"]),
                set(boundaries_by_model["starcoder2"]),
                int(row["target_split_offset"]),
            )
            if chosen is None:
                missing = [
                    model_key
                    for model_key in MODEL_KEYS
                    if not boundaries_by_model[model_key]
                ]
                if missing:
                    reason = "no_internal_native_boundary_for_" + "_and_".join(missing)
                else:
                    reason = "no_common_character_boundary_between_tokenizers"
                ineligible_by_original_index[original_index] = {
                    "protocol": PROTOCOL,
                    "language": language,
                    "canonical_sample_id": str(row["canonical_sample_id"]),
                    "original_e1_sample_index": int(row["e1_sample_index"]),
                    "repository": str(row["repo_id"]),
                    "file": str(row["file_path"]),
                    "full_file_sha256": source_hash,
                    "target_identifier": identifier,
                    "original_split_offset": int(row["target_split_offset"]),
                    "qwen_native_split_offsets": sorted(boundaries_by_model["qwen"]),
                    "starcoder2_native_split_offsets": sorted(
                        boundaries_by_model["starcoder2"]
                    ),
                    "reason": reason,
                }
                availability[f"ineligible_{reason}"] += 1
                continue
            common_offsets = sorted(
                set(boundaries_by_model["qwen"])
                & set(boundaries_by_model["starcoder2"])
            )
            metadata = {
                "selection_rule": "common Qwen/StarCoder2 continuous-source internal token boundary closest to original split; lower split offset breaks ties",
                "selected_split_offset": chosen,
                "common_candidate_split_offsets": common_offsets,
                "per_model": {
                    model_key: boundaries_by_model[model_key][chosen]
                    for model_key in MODEL_KEYS
                },
            }
            native_row = transform_cursor_row(
                row,
                source=source,
                split_offset=chosen,
                cursor_condition="native_internal",
                condition_index=original_index,
                native_metadata=metadata,
            )
            native_rows_by_original_index[original_index] = native_row
            chosen_offsets[str(chosen)] += 1
            original_split = int(row["target_split_offset"])
            relation_to_original[
                (
                    "same"
                    if chosen == original_split
                    else "earlier" if chosen < original_split else "later"
                )
            ] += 1
            per_model_references = {}
            for model_key, tokenizer in tokenizers.items():
                (token_ids, offsets) = encoded_by_model[model_key]
                setting = SETTING_BY_LANGUAGE[language][model_key]
                per_model_references[model_key] = _make_native_reference(
                    setting=setting,
                    language=language,
                    sample=native_row,
                    tokenizer=tokenizer,
                    source=source,
                    token_ids=token_ids,
                    offsets=offsets,
                    boundary=boundaries_by_model[model_key][chosen],
                )
            native_references_by_original_index[original_index] = per_model_references
            availability["common_internal_boundary"] += 1
    native_rows = [
        native_rows_by_original_index[index]
        for index in range(len(original_rows))
        if index in native_rows_by_original_index
    ]
    ineligible_rows = [
        ineligible_by_original_index[index]
        for index in range(len(original_rows))
        if index in ineligible_by_original_index
    ]
    if len(native_rows) + len(ineligible_rows) != len(original_rows):
        raise AssertionError("native eligibility ledger is incomplete")
    for index, row in enumerate(native_rows):
        row["e1_sample_index"] = index
    language_dir = output_dir / language
    manifests = {
        "original": language_dir / "manifests" / "original.jsonl.gz",
        "identifier_start": language_dir / "manifests" / "identifier_start.jsonl.gz",
        "native_internal": language_dir / "manifests" / "native_internal.jsonl.gz",
    }
    _write_jsonl_gzip(manifests["original"], transformed_original)
    _write_jsonl_gzip(manifests["identifier_start"], start_rows)
    _write_jsonl_gzip(manifests["native_internal"], native_rows)
    ineligible_path = language_dir / "native_internal_ineligible.jsonl.gz"
    _write_jsonl_gzip(ineligible_path, ineligible_rows)
    reference_artifacts: dict[str, dict[str, Any]] = defaultdict(dict)
    for model_key, tokenizer in tokenizers.items():
        setting = SETTING_BY_LANGUAGE[language][model_key]
        existing_path = reference_root / setting / "reference_path_rows.jsonl.gz"
        original_references = _subset_existing_references(
            path=existing_path, fixed_ids=fixed_ids
        )
        start_references = [
            _make_independent_reference(
                setting=setting, language=language, sample=row, tokenizer=tokenizer
            )
            for row in start_rows
        ]
        native_references = [
            native_references_by_original_index[index][model_key]
            for index in range(len(original_rows))
            if index in native_references_by_original_index
        ]
        for index, row in enumerate(native_references):
            row["e1_sample_index"] = index
        by_condition = {
            "original": original_references,
            "identifier_start": start_references,
            "native_internal": native_references,
        }
        for condition, rows in by_condition.items():
            path = language_dir / "references" / condition / f"{setting}.jsonl.gz"
            _write_jsonl_gzip(path, rows)
            reference_artifacts[condition][setting] = _artifact(path, len(rows))
    return {
        "language": language,
        "fixed_occurrences": len(original_rows),
        "fixed_occurrence_ids": _artifact(fixed_id_path, len(fixed_ids)),
        "source_manifest": _artifact(manifest),
        "source_content_sidecar": _artifact(sidecar),
        "unique_source_files": len(required_hashes),
        "all_source_prefix_and_target_coordinates_verified": True,
        "selection_uses_model_outcomes_or_route": False,
        "native_boundary_selection": "intersection of exact continuous-source token boundaries for the normal Qwen and StarCoder2 tokenizer pair; nearest to original cursor, with the lower offset breaking ties",
        "native_eligible_common_occurrences": len(native_rows),
        "native_ineligible_occurrences": len(ineligible_rows),
        "native_availability_counts": dict(sorted(availability.items())),
        "selected_native_boundary_vs_original": dict(
            sorted(relation_to_original.items())
        ),
        "selected_native_split_offset_counts": dict(
            sorted(chosen_offsets.items(), key=lambda item: int(item[0]))
        ),
        "manifests": {
            condition: _artifact(path, len(rows))
            for (condition, path, rows) in (
                ("original", manifests["original"], transformed_original),
                ("identifier_start", manifests["identifier_start"], start_rows),
                ("native_internal", manifests["native_internal"], native_rows),
            )
        },
        "native_ineligible_ledger": _artifact(ineligible_path, len(ineligible_rows)),
        "references": dict(reference_artifacts),
    }


def run(
    *,
    cohort_dir: Path,
    reference_root: Path,
    canonical_root: Path,
    qwen_tokenizer_path: Path,
    starcoder2_tokenizer_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    tokenizer_paths = {
        "qwen": qwen_tokenizer_path,
        "starcoder2": starcoder2_tokenizer_path,
    }
    tokenizers = {
        model_key: AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        for (model_key, path) in tokenizer_paths.items()
    }
    tokenizer_metadata = {}
    for model_key, tokenizer in tokenizers.items():
        original_model_max_length = int(tokenizer.model_max_length)
        tokenizer.model_max_length = 10**30
        tokenizer_metadata[model_key] = {
            "path": str(tokenizer_paths[model_key].resolve()),
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "configured_model_max_length": original_model_max_length,
            "fast_offset_mapping": bool(getattr(tokenizer, "is_fast", False)),
        }
        if not bool(getattr(tokenizer, "is_fast", False)):
            raise ValueError(f"{model_key} tokenizer lacks offset mapping")
    languages = {}
    for language in ("python", "typescript"):
        languages[language] = build_language(
            language=language,
            cohort_dir=cohort_dir,
            reference_root=reference_root,
            sidecar=canonical_root
            / "languages"
            / language
            / "selected_full_file_content.fc.jsonl",
            tokenizers=tokenizers,
            output_dir=output_dir,
        )
    report = {
        "protocol": PROTOCOL,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "condition_protocols": CONDITION_PROTOCOLS,
        "tokenizers": tokenizer_metadata,
        "constraints": [
            "the 5,000 occurrence IDs were fixed before E1 outcome generation",
            "identifier-start and native cursors use the same source occurrence",
            "native boundaries come from continuous full-source tokenization",
            "the model input ends exactly at the selected boundary",
            "no token crosses the native cursor and no right-side character enters input",
            "native next-token references come from the continuous token sequence",
            "ineligible native-boundary occurrences remain in an explicit ledger",
            "selection never reads Base/Editor correctness or route activation",
        ],
        "languages": languages,
    }
    _write_json(output_dir / "cursor_input_audit.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer", type=Path, required=True)
    parser.add_argument("--starcoder2-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(
        cohort_dir=args.cohort_dir.expanduser().resolve(),
        reference_root=args.reference_root.expanduser().resolve(),
        canonical_root=args.canonical_root.expanduser().resolve(),
        qwen_tokenizer_path=args.qwen_tokenizer.expanduser().resolve(),
        starcoder2_tokenizer_path=args.starcoder2_tokenizer.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

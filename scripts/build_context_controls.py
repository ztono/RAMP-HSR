from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Sequence
from transformers import AutoTokenizer

PROTOCOL = "reviewer3-e2-context-inputs-and-evidence-v1"
SETTING_ORDER = (
    "qwen_python",
    "starcoder2_python",
    "qwen_typescript",
    "starcoder2_typescript",
)
TOKENIZER_KEY = {
    "qwen_python": "qwen",
    "qwen_typescript": "qwen",
    "starcoder2_python": "starcoder2",
    "starcoder2_typescript": "starcoder2",
}
WINDOWS = (512, 2048, 8192)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_gzip(path: Path) -> Iterable[dict[str, Any]]:
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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def _write_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _load_manifest(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(_read_gzip(path))
    index = {str(row["canonical_sample_id"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"duplicate canonical IDs in {path}")
    return (rows, index)


def _load_reference_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(_read_gzip(path))
    result = {str(row["canonical_sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate reference IDs in {path}")
    return result


def _load_binding_probe(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["canonical_sample_id"])
            if sample_id in result:
                raise ValueError(f"duplicate binding probe ID {sample_id}")
            result[sample_id] = row
    return result


def _load_contents(sidecar: Path, required: set[str]) -> dict[str, str]:
    result = {}
    with sidecar.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_hash = str(row.get("sha256") or "").lower()
            if source_hash not in required:
                continue
            content = row.get("content")
            if not isinstance(content, str):
                raise ValueError(f"non-string source at {sidecar}:{line_number}")
            if _sha256_text(content) != source_hash:
                raise ValueError(f"source SHA mismatch at {sidecar}:{line_number}")
            result[source_hash] = content
    missing = required.difference(result)
    if missing:
        raise ValueError(f"missing {len(missing)} source files from {sidecar}")
    return result


def _identifier_continuation(character: str, language: str) -> bool:
    if not character:
        return False
    if language == "typescript":
        return bool(re.fullmatch("[A-Za-z0-9_$]", character))
    return (
        character == "_" or character.isalnum() or bool(re.fullmatch("\\w", character))
    )


def prior_text_occurrences(
    source: str, identifier: str, target_start: int, language: str
) -> list[dict[str, int]]:
    spans = []
    for match in re.finditer(re.escape(identifier), source[:target_start]):
        (start, end) = match.span()
        left = source[start - 1] if start else ""
        right = source[end] if end < len(source) else ""
        if _identifier_continuation(left, language) or _identifier_continuation(
            right, language
        ):
            continue
        spans.append({"start": start, "end": end})
    return spans


def _window_start_char(
    offsets: Sequence[tuple[int, int]], token_count: int, window: int
) -> int:
    if token_count <= window:
        return 0
    index = token_count - window
    for start, end in offsets[index:]:
        if end > start:
            return int(start)
    raise ValueError("window contains no token with a positive character span")


def _visibility_group(
    spans: Sequence[dict[str, Any]], starts: dict[int, int]
) -> tuple[str, dict[int, int]]:
    visible_counts = {
        window: sum((int(span["start"]) >= char_start for span in spans))
        for (window, char_start) in starts.items()
    }
    if visible_counts[512]:
        group = "Visible-512"
    elif visible_counts[2048]:
        group = "Restored-long"
    elif spans:
        group = "Still-missing"
    else:
        group = "Unresolved / not applicable"
    return (group, visible_counts)


def _binding_evidence(
    *,
    sample: dict[str, Any],
    probe: dict[str, Any] | None,
    window_starts: dict[int, int],
) -> dict[str, Any]:
    if probe is None:
        return {
            "group": "Unresolved / not applicable",
            "reason": "no_conservative_local_binding_record",
            "reliable_local_use_subset": False,
        }
    identifier = str(sample["target_identifier"])
    identifier_start = int(sample["identifier_start_absolute_position"])
    if (
        str(probe["target_name"]) != identifier
        or int(probe["target_position"]) != identifier_start
    ):
        raise ValueError(
            f"binding record target mismatch for {sample['canonical_sample_id']}"
        )
    candidate = probe["candidates"][int(probe["correct_candidate_index"])]
    spans = [
        {"start": int(span["start"]), "end": int(span["end"]), "role": span["role"]}
        for span in candidate["prior_representation_offsets"]
        if int(span["end"]) <= identifier_start
    ]
    if not spans:
        raise ValueError("conservative binding record contains no prior evidence")
    (group, visible_counts) = _visibility_group(spans, window_starts)
    return {
        "group": group,
        "reason": None,
        "reliable_local_use_subset": True,
        "analysis_protocol": str(probe["protocol"]),
        "symbol_kind": str(candidate["kind"]),
        "target_is_direct_load": True,
        "prior_binding_evidence_spans": spans,
        "prior_binding_evidence_count": len(spans),
        "visible_evidence_counts": {str(k): v for (k, v) in visible_counts.items()},
        "static_restrictions": probe.get("static_restrictions"),
    }


def _setting_language(setting: str) -> str:
    return "typescript" if setting.endswith("typescript") else "python"


def run(
    *,
    cohort_dir: Path,
    reference_root: Path,
    canonical_root: Path,
    qwen_tokenizer_path: Path,
    starcoder2_tokenizer_path: Path,
    python_binding_probe_path: Path | None,
    typescript_binding_probe_path: Path | None,
    binding_source_archive: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    tokenizer_paths = {
        "qwen": qwen_tokenizer_path,
        "starcoder2": starcoder2_tokenizer_path,
    }
    tokenizers = {
        key: AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        for (key, path) in tokenizer_paths.items()
    }
    tokenizer_metadata = {}
    for key, tokenizer in tokenizers.items():
        configured_limit = int(tokenizer.model_max_length)
        tokenizer.model_max_length = 10**30
        tokenizer_metadata[key] = {
            "path": str(tokenizer_paths[key].resolve()),
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "configured_model_max_length": configured_limit,
        }
    manifests_by_language = {}
    indices_by_language = {}
    candidate_ids_by_setting = {}
    references_by_setting = {}
    union_ids_by_language = {"python": set(), "typescript": set()}
    for language in ("python", "typescript"):
        manifest_path = cohort_dir / language / "e1_original_cursor.jsonl.gz"
        (rows, index) = _load_manifest(manifest_path)
        manifests_by_language[language] = (manifest_path, rows)
        indices_by_language[language] = index
    for setting in SETTING_ORDER:
        language = _setting_language(setting)
        id_path = reference_root / setting / "long_context_candidate_ids.txt"
        candidate_ids = [
            line.strip()
            for line in id_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(candidate_ids) > 5000 or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"invalid pre-fixed candidate list for {setting}")
        if any(
            (
                sample_id not in indices_by_language[language]
                for sample_id in candidate_ids
            )
        ):
            raise ValueError(f"{setting}: candidate outside frozen E1 cohort")
        candidate_ids_by_setting[setting] = (id_path, candidate_ids)
        union_ids_by_language[language].update(candidate_ids)
        references_by_setting[setting] = _load_reference_index(
            reference_root / setting / "reference_path_rows.jsonl.gz"
        )
    contents_by_language = {}
    binding_subset_artifacts = {}
    for language in ("python", "typescript"):
        required_hashes = {
            str(indices_by_language[language][sample_id]["full_file_sha256"]).lower()
            for sample_id in union_ids_by_language[language]
        }
        source_sidecar = (
            canonical_root
            / "languages"
            / language
            / "selected_full_file_content.fc.jsonl"
        )
        contents = _load_contents(source_sidecar, required_hashes)
        contents_by_language[language] = contents
        union_rows = sorted(
            (
                indices_by_language[language][sample_id]
                for sample_id in union_ids_by_language[language]
            ),
            key=lambda row: (
                int(row["canonical_index"]),
                str(row["canonical_sample_id"]),
            ),
        )
        subset_dir = output_dir / "binding_subset" / language
        manifest_out = subset_dir / "canonical_candidates.jsonl"
        sidecar_out = subset_dir / "full_file_content.fc.jsonl"
        _write_jsonl(manifest_out, union_rows)
        _write_jsonl(
            sidecar_out,
            (
                {"sha256": source_hash, "content": contents[source_hash]}
                for source_hash in sorted(contents)
            ),
        )
        binding_subset_artifacts[language] = {
            "candidate_occurrences": len(union_rows),
            "unique_source_files": len(contents),
            "canonical_candidates": _artifact(manifest_out, len(union_rows)),
            "full_file_content": _artifact(sidecar_out, len(contents)),
            "source_sidecar": _artifact(source_sidecar),
        }
    binding_probes = {
        "python": _load_binding_probe(python_binding_probe_path),
        "typescript": _load_binding_probe(typescript_binding_probe_path),
    }
    setting_reports = {}
    all_group_counts = {}
    for setting_index, setting in enumerate(SETTING_ORDER):
        language = _setting_language(setting)
        tokenizer = tokenizers[TOKENIZER_KEY[setting]]
        (id_path, candidate_ids) = candidate_ids_by_setting[setting]
        manifest_rows = []
        reference_rows = []
        evidence_rows = []
        textual_counts = Counter()
        binding_counts = Counter()
        for condition_index, sample_id in enumerate(candidate_ids):
            original = indices_by_language[language][sample_id]
            sample = dict(original)
            sample["source_e1_sample_index"] = int(original["e1_sample_index"])
            sample["e1_sample_index"] = condition_index
            sample["e2_protocol"] = PROTOCOL
            sample["cursor_condition"] = "original"
            sample["evaluation_cursor_absolute_position"] = int(
                original["original_cursor_absolute_position"]
            )
            manifest_rows.append(sample)
            reference = dict(references_by_setting[setting][sample_id])
            reference["e1_sample_index"] = condition_index
            reference["cursor_condition"] = "original"
            reference["reference_target_mode"] = "independent_remaining_suffix"
            reference["reference_token_ids"] = list(
                reference["reference_path"]["suffix_token_ids"]
            )
            reference_rows.append(reference)
            source_hash = str(sample["full_file_sha256"]).lower()
            source = contents_by_language[language][source_hash]
            cursor = int(sample["original_cursor_absolute_position"])
            identifier_start = int(sample["identifier_start_absolute_position"])
            identifier = str(sample["target_identifier"])
            if source[:cursor] != str(sample["prefix"]):
                raise ValueError(f"causal prefix mismatch for {sample_id}")
            if (
                source[identifier_start : identifier_start + len(identifier)]
                != identifier
            ):
                raise ValueError(f"target source mismatch for {sample_id}")
            encoded = tokenizer(
                str(sample["prefix"]),
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            token_ids = [int(value) for value in encoded["input_ids"]]
            offsets = [(int(a), int(b)) for (a, b) in encoded["offset_mapping"]]
            expected_count = int(reference["reference_path"]["prefix_token_count"])
            if len(token_ids) != expected_count or len(token_ids) <= 512:
                raise ValueError(f"long-context token-count mismatch for {sample_id}")
            window_starts = {
                window: _window_start_char(offsets, len(token_ids), window)
                for window in WINDOWS
            }
            text_spans = prior_text_occurrences(
                source, identifier, identifier_start, language
            )
            (text_group, text_visible) = _visibility_group(text_spans, window_starts)
            textual_counts[text_group] += 1
            binding = _binding_evidence(
                sample=sample,
                probe=binding_probes[language].get(sample_id),
                window_starts=window_starts,
            )
            binding_counts[str(binding["group"])] += 1
            evidence_rows.append(
                {
                    "protocol": PROTOCOL,
                    "setting": setting,
                    "language": language,
                    "e1_sample_index": condition_index,
                    "source_e1_sample_index": int(original["e1_sample_index"]),
                    "canonical_sample_id": sample_id,
                    "repository": str(sample["repo_id"]),
                    "file": str(sample["file_path"]),
                    "full_file_sha256": source_hash,
                    "target_identifier": identifier,
                    "identifier_start_absolute_position": identifier_start,
                    "original_character_cursor": cursor,
                    "full_prefix_token_count": len(token_ids),
                    "window_start_character_offsets": {
                        str(k): v for (k, v) in window_starts.items()
                    },
                    "prior_text_occurrence": {
                        "terminology": "exact prior source-text occurrence with identifier character boundaries; not a binding claim",
                        "group": text_group,
                        "prior_spans": text_spans,
                        "prior_occurrence_count": len(text_spans),
                        "visible_occurrence_counts": {
                            str(k): v for (k, v) in text_visible.items()
                        },
                    },
                    "conservative_binding": binding,
                    "causal_only": all(
                        (
                            int(span["end"]) <= identifier_start
                            for span in binding.get("prior_binding_evidence_spans", [])
                        )
                    ),
                }
            )
        setting_dir = output_dir / setting
        manifest_path = setting_dir / "manifest.jsonl.gz"
        reference_path = setting_dir / "reference_path_rows.jsonl.gz"
        evidence_path = setting_dir / "evidence_rows.jsonl.gz"
        _write_gzip(manifest_path, manifest_rows)
        _write_gzip(reference_path, reference_rows)
        _write_gzip(evidence_path, evidence_rows)
        reliable = sum(
            (
                count
                for (group, count) in binding_counts.items()
                if group != "Unresolved / not applicable"
            )
        )
        truncated_reliable = (
            binding_counts["Restored-long"] + binding_counts["Still-missing"]
        )
        restored_fraction = (
            binding_counts["Restored-long"] / truncated_reliable
            if truncated_reliable
            else None
        )
        all_group_counts[setting] = dict(binding_counts)
        setting_reports[setting] = {
            "language": language,
            "samples": len(manifest_rows),
            "repositories": len({str(row["repo_id"]) for row in manifest_rows}),
            "selection": "pre-fixed E1 rows with Tok(full original character prefix) > 512, in frozen E1 cohort order, capped at 5,000",
            "model_max_position_embeddings": (
                32768 if TOKENIZER_KEY[setting] == "qwen" else 16384
            ),
            "context_windows": [512, 2048],
            "candidate_ids": _artifact(id_path, len(candidate_ids)),
            "manifest": _artifact(manifest_path, len(manifest_rows)),
            "reference_audit": _artifact(reference_path, len(reference_rows)),
            "evidence": _artifact(evidence_path, len(evidence_rows)),
            "prior_text_occurrence_group_counts": dict(sorted(textual_counts.items())),
            "conservative_binding_group_counts": dict(sorted(binding_counts.items())),
            "reliable_binding_samples": reliable,
            "binding_evidence_missing_at_512": truncated_reliable,
            "fraction_of_512_missing_binding_evidence_restored_by_2048": restored_fraction,
        }
    trigger_settings = []
    for setting, groups in all_group_counts.items():
        missing = groups.get("Restored-long", 0) + groups.get("Still-missing", 0)
        if missing and groups.get("Restored-long", 0) / missing < 0.5:
            trigger_settings.append(setting)
    report = {
        "protocol": PROTOCOL,
        "status": (
            "complete"
            if python_binding_probe_path and typescript_binding_probe_path
            else "binding_probe_pending"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_uses_model_outcomes_or_route": False,
        "frozen_conditions": {
            "cursor": "original E1 character cursor",
            "model_and_editor_checkpoints": "unchanged",
            "generation_and_identifier_scoring": "same as E1",
            "only_intervention": "left causal model-input window K=512 versus K=2048",
        },
        "tokenizers": tokenizer_metadata,
        "binding_probe_sources": {
            "python": (
                _artifact(python_binding_probe_path)
                if python_binding_probe_path
                else None
            ),
            "typescript": (
                _artifact(typescript_binding_probe_path)
                if typescript_binding_probe_path
                else None
            ),
            "python_source_archive": (
                _artifact(binding_source_archive) if binding_source_archive else None
            ),
            "scope": "conservative same-function local direct-use subset; absence of a record is Unresolved/not applicable, never evidence of no binding",
        },
        "binding_subsets": binding_subset_artifacts,
        "settings": setting_reports,
        "longer_context_trigger_rule": "Before reading model outcomes, add K=8192 only for a setting where K=2048 restores fewer than half of reliably parsed binding-evidence cases that are missing at K=512.",
        "settings_triggering_k8192": trigger_settings,
    }
    _write_json(output_dir / "e2_input_audit.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer", type=Path, required=True)
    parser.add_argument("--starcoder2-tokenizer", type=Path, required=True)
    parser.add_argument("--python-binding-probe", type=Path)
    parser.add_argument("--typescript-binding-probe", type=Path)
    parser.add_argument("--python-binding-source-archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(
        cohort_dir=args.cohort_dir.expanduser().resolve(),
        reference_root=args.reference_root.expanduser().resolve(),
        canonical_root=args.canonical_root.expanduser().resolve(),
        qwen_tokenizer_path=args.qwen_tokenizer.expanduser().resolve(),
        starcoder2_tokenizer_path=args.starcoder2_tokenizer.expanduser().resolve(),
        python_binding_probe_path=(
            args.python_binding_probe.expanduser().resolve()
            if args.python_binding_probe
            else None
        ),
        typescript_binding_probe_path=(
            args.typescript_binding_probe.expanduser().resolve()
            if args.typescript_binding_probe
            else None
        ),
        binding_source_archive=(
            args.python_binding_source_archive.expanduser().resolve()
            if args.python_binding_source_archive
            else None
        ),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

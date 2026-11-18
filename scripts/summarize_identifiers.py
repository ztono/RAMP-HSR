from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from scripts.audit_isolation import normalize_repo_id

PROTOCOL = "reviewer3-e1-same-original-cursor-v2"
SUMMARY_PROTOCOL = "reviewer3-e1-same-cursor-summary-v1"
SETTING_ORDER = (
    "qwen_python",
    "starcoder2_python",
    "qwen_typescript",
    "starcoder2_typescript",
)
DISPLAY_NAMES = {
    "qwen_python": "Qwen / Python",
    "starcoder2_python": "StarCoder2 / Python",
    "qwen_typescript": "Qwen / TypeScript",
    "starcoder2_typescript": "StarCoder2 / TypeScript",
}
COUNT_FIELDS = (
    "samples",
    "base_token_correct",
    "editor_token_correct",
    "token_fix",
    "token_damage",
    "base_identifier_correct",
    "editor_identifier_correct",
    "identifier_fix",
    "identifier_damage",
    "strict_base_identifier_correct",
    "strict_editor_identifier_correct",
    "base_identifier_correct_and_base_token_wrong",
    "identifier_fix_not_old_token_fix",
    "old_token_fix",
    "old_fix_base_id_wrong_editor_id_correct",
    "old_fix_base_id_correct_editor_id_correct",
    "old_fix_base_id_wrong_editor_id_wrong",
    "old_fix_base_id_correct_editor_id_wrong",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _count_jsonl_gzip(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum((bool(line.strip()) for line in handle))


def expected_samples_from_cohort_audit(
    path: Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Derive census denominators from an independently written cohort audit."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("E1 cohort audit is not complete")
    language_counts: dict[str, int] = {}
    for language in ("python", "typescript"):
        item = payload.get("languages", {}).get(language)
        if not isinstance(item, dict):
            raise ValueError(f"E1 cohort audit is missing {language}")
        selected = int(item.get("selected_samples", 0))
        if selected <= 0:
            raise ValueError(f"E1 cohort audit has no selected {language} rows")
        if not bool(item.get("selection_is_outcome_independent")):
            raise ValueError(f"E1 {language} cohort selection is outcome-dependent")
        if int(item.get("post_selection_train_repository_overlap", -1)) != 0:
            raise ValueError(f"E1 {language} cohort has training-repository overlap")
        if int(item.get("post_selection_train_file_hash_overlap", -1)) != 0:
            raise ValueError(f"E1 {language} cohort has training-file overlap")
        language_counts[language] = selected
    expected = {
        f"{model}_{language}": language_counts[language]
        for language in ("python", "typescript")
        for model in ("qwen", "starcoder2")
    }
    provenance = {
        "source": "completed cohort audit",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "protocol": payload.get("protocol"),
        "selection_constraints": payload.get("selection_constraints", []),
        "selected_samples_by_language": language_counts,
        "outcome_independent_and_post_e6_zero_overlap_verified": True,
    }
    return (expected, provenance)


def expected_samples_from_native_manifests(
    root: Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Derive independent full-Test denominators from native manifests."""
    language_counts: dict[str, int] = {}
    manifests: dict[str, Any] = {}
    for language in ("python", "typescript"):
        summary_path = root / language / "manifest_summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"native {language} manifest is not complete")
        if (
            payload.get("selection")
            != "all rows from every supplied original Test shard"
        ):
            raise ValueError(f"native {language} manifest is not a full-Test census")
        if bool(payload.get("repository_or_file_deduplication")):
            raise ValueError(f"native {language} manifest unexpectedly deduplicates")
        samples = int(payload.get("samples", 0))
        if samples <= 0:
            raise ValueError(f"native {language} manifest is empty")
        combined = payload["combined_manifest"]
        manifest_path = Path(combined["path"])
        if _sha256(manifest_path) != str(combined["sha256"]):
            raise ValueError(f"native {language} combined manifest hash mismatch")
        language_counts[language] = samples
        manifests[language] = {
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": str(combined["sha256"]),
            "samples": samples,
            "cursor_counts": payload.get("cursor_counts"),
        }
    expected = {
        f"{model}_{language}": language_counts[language]
        for language in ("python", "typescript")
        for model in ("qwen", "starcoder2")
    }
    return (
        expected,
        {
            "source": "independent native full-Test manifests",
            "manifest_root": str(root.resolve()),
            "selected_samples_by_language": language_counts,
            "cross_model_intersection_required": False,
            "repository_or_file_deduplication": False,
            "manifests": manifests,
        },
    )


def _labels(row: dict[str, Any]) -> tuple[bool, bool, bool, bool, bool, bool]:
    token_base = bool(row["first_token"]["base_old_token_exact"])
    token_editor = bool(row["first_token"]["editor_old_token_exact"])
    id_base = bool(
        row["base"]["complete_identifier_tolerant_start"]["complete_identifier_exact"]
    )
    id_editor = bool(
        row["editor_first_token_then_base"]["complete_identifier_tolerant_start"][
            "complete_identifier_exact"
        ]
    )
    strict_id_base = bool(
        row["base"]["complete_identifier_strict_start"]["complete_identifier_exact"]
    )
    strict_id_editor = bool(
        row["editor_first_token_then_base"]["complete_identifier_strict_start"][
            "complete_identifier_exact"
        ]
    )
    return (
        token_base,
        token_editor,
        id_base,
        id_editor,
        strict_id_base,
        strict_id_editor,
    )


def counts_for_rows(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    counts = Counter({field: 0 for field in COUNT_FIELDS})
    for row in rows:
        (tb, te, cb, ce, strict_cb, strict_ce) = _labels(row)
        token_fix = not tb and te
        identifier_fix = not cb and ce
        counts["samples"] += 1
        counts["base_token_correct"] += tb
        counts["editor_token_correct"] += te
        counts["token_fix"] += token_fix
        counts["token_damage"] += tb and (not te)
        counts["base_identifier_correct"] += cb
        counts["editor_identifier_correct"] += ce
        counts["identifier_fix"] += identifier_fix
        counts["identifier_damage"] += cb and (not ce)
        counts["strict_base_identifier_correct"] += strict_cb
        counts["strict_editor_identifier_correct"] += strict_ce
        counts["base_identifier_correct_and_base_token_wrong"] += cb and (not tb)
        counts["identifier_fix_not_old_token_fix"] += identifier_fix and (not token_fix)
        counts["old_token_fix"] += token_fix
        if token_fix:
            counts[
                f"old_fix_base_id_{('correct' if cb else 'wrong')}_editor_id_{('correct' if ce else 'wrong')}"
            ] += 1
    return np.asarray([counts[field] for field in COUNT_FIELDS], dtype=np.int64)


def _as_count_dict(values: np.ndarray) -> dict[str, int]:
    return {
        field: int(value) for (field, value) in zip(COUNT_FIELDS, values, strict=True)
    }


def metrics_from_counts(values: np.ndarray) -> dict[str, float | None]:
    c = _as_count_dict(values)

    def rate(numerator: int, denominator: int) -> float | None:
        return float(numerator) / float(denominator) if denominator else None

    n = c["samples"]
    token_fail = n - c["base_token_correct"]
    identifier_fail = n - c["base_identifier_correct"]
    return {
        "base_token_em": rate(c["base_token_correct"], n),
        "editor_token_em": rate(c["editor_token_correct"], n),
        "token_error_recovery": rate(c["token_fix"], token_fail),
        "token_damage": rate(c["token_damage"], c["base_token_correct"]),
        "token_net": rate(c["editor_token_correct"] - c["base_token_correct"], n),
        "base_id_em": rate(c["base_identifier_correct"], n),
        "editor_id_em": rate(c["editor_identifier_correct"], n),
        "id_error_recovery": rate(c["identifier_fix"], identifier_fail),
        "id_damage": rate(c["identifier_damage"], c["base_identifier_correct"]),
        "id_net": rate(
            c["editor_identifier_correct"] - c["base_identifier_correct"], n
        ),
        "strict_base_id_em": rate(c["strict_base_identifier_correct"], n),
        "strict_editor_id_em": rate(c["strict_editor_identifier_correct"], n),
        "base_id_correct_given_base_token_wrong": rate(
            c["base_identifier_correct_and_base_token_wrong"], token_fail
        ),
        "id_fixes_not_captured_by_old_token_fix": rate(
            c["identifier_fix_not_old_token_fix"], c["identifier_fix"]
        ),
    }


def equal_cursor_standardization(
    rows_by_cursor: dict[str, Sequence[dict[str, Any]]]
) -> dict[str, Any]:
    """Return a 50:50 start/internal sensitivity result from weighted counts."""
    required = ("identifier_start", "identifier_internal")
    if any((not rows_by_cursor.get(cursor) for cursor in required)):
        raise ValueError("50:50 standardization requires both cursor strata")
    normalized = {
        cursor: counts_for_rows(rows_by_cursor[cursor]).astype(np.float64)
        / float(len(rows_by_cursor[cursor]))
        for cursor in required
    }
    weighted = 0.5 * normalized[required[0]] + 0.5 * normalized[required[1]]
    c = {
        field: float(value)
        for (field, value) in zip(COUNT_FIELDS, weighted, strict=True)
    }

    def rate(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator else None

    n = c["samples"]
    token_fail = n - c["base_token_correct"]
    identifier_fail = n - c["base_identifier_correct"]
    metrics = {
        "base_token_em": rate(c["base_token_correct"], n),
        "editor_token_em": rate(c["editor_token_correct"], n),
        "token_error_recovery": rate(c["token_fix"], token_fail),
        "token_damage": rate(c["token_damage"], c["base_token_correct"]),
        "token_net": rate(c["editor_token_correct"] - c["base_token_correct"], n),
        "base_id_em": rate(c["base_identifier_correct"], n),
        "editor_id_em": rate(c["editor_identifier_correct"], n),
        "id_error_recovery": rate(c["identifier_fix"], identifier_fail),
        "id_damage": rate(c["identifier_damage"], c["base_identifier_correct"]),
        "id_net": rate(
            c["editor_identifier_correct"] - c["base_identifier_correct"], n
        ),
    }
    return {
        "label": "50:50 identifier-start/identifier-internal sensitivity analysis",
        "population_claim": False,
        "weights": {"identifier_start": 0.5, "identifier_internal": 0.5},
        "method": "normalize each stratum's confusion counts by its stratum N, apply the 0.5 weights, sum weighted counts, then form conditional rates; the two stratum rates are never averaged directly",
        "weighted_confusion_counts_per_standardized_sample": c,
        "point_estimates": metrics,
    }


def repository_cluster_bootstrap(
    rows: Sequence[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    by_repository: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        repository = normalize_repo_id(row.get("repository"))
        if not repository:
            raise ValueError("missing normalized repository identity")
        by_repository[repository].append(row)
    repositories = sorted(by_repository)
    if not repositories:
        raise ValueError("cannot bootstrap an empty cohort")
    matrix = np.stack(
        [counts_for_rows(by_repository[repository]) for repository in repositories]
    )
    total_counts = matrix.sum(axis=0)
    points = metrics_from_counts(total_counts)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in points}
    remaining = int(iterations)
    while remaining:
        batch = min(512, remaining)
        indices = rng.integers(0, len(repositories), size=(batch, len(repositories)))
        sampled = matrix[indices].sum(axis=1)
        for count_row in sampled:
            for metric, value in metrics_from_counts(count_row).items():
                if value is not None and math.isfinite(value):
                    draws[metric].append(float(value))
        remaining -= batch
    return {
        "method": "repository-cluster paired percentile bootstrap; repositories are sampled with replacement and every occurrence plus its paired Base/Editor labels is retained; conditional rates are computed from each replicate's aggregate confusion counts",
        "iterations": int(iterations),
        "seed": int(seed),
        "repository_count": len(repositories),
        "counts": _as_count_dict(total_counts),
        "point_estimates": points,
        "ci95": {
            metric: (
                [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
                if values
                else None
            )
            for (metric, values) in draws.items()
        },
    }


def _outcome_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for branch in ("base", "editor_first_token_then_base"):
        outcomes = Counter(
            (
                str(row[branch]["complete_identifier_tolerant_start"]["outcome"])
                for row in rows
            )
        )
        horizons = Counter(
            (str(row[branch]["generation_horizon_used"]) for row in rows)
        )
        payload[branch] = {
            "outcomes": dict(sorted(outcomes.items())),
            "generation_horizons": dict(
                sorted(horizons.items(), key=lambda item: int(item[0]))
            ),
            "standalone_decode_fallbacks": sum(
                (
                    row[branch]["decoded_generation"]["decode_method"]
                    == "standalone_fallback"
                    for row in rows
                )
            ),
        }
    return payload


def _validate_setting(
    setting: str,
    setting_dir: Path,
    expected_samples: dict[str, int] | None = None,
    expected_protocol: str = PROTOCOL,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    summary_path = setting_dir / "worker_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"{setting}: worker is not complete")
    if summary.get("protocol") != expected_protocol:
        raise ValueError(f"{setting}: unexpected protocol {summary.get('protocol')!r}")
    if summary.get("setting") != setting:
        raise ValueError(f"{setting}: worker summary setting mismatch")
    predictions_path = setting_dir / "predictions.jsonl.gz"
    first_tokens_path = setting_dir / "first_token_scores.jsonl.gz"
    prediction_sha = _sha256(predictions_path)
    first_token_sha = _sha256(first_tokens_path)
    if prediction_sha != summary["outputs"]["predictions_sha256"]:
        raise ValueError(f"{setting}: predictions SHA-256 mismatch")
    if first_token_sha != summary["outputs"]["first_token_scores_sha256"]:
        raise ValueError(f"{setting}: first-token SHA-256 mismatch")
    rows = _read_jsonl_gzip(predictions_path)
    expected = (
        int(summary["samples"])
        if expected_samples is None
        else expected_samples[setting]
    )
    if len(rows) != expected or int(summary["samples"]) != expected:
        raise ValueError(
            f"{setting}: expected {expected} rows; predictions={len(rows)}, summary={summary['samples']}"
        )
    first_token_rows = _count_jsonl_gzip(first_tokens_path)
    if first_token_rows != expected:
        raise ValueError(
            f"{setting}: expected {expected} first-token rows, got {first_token_rows}"
        )
    sample_ids = [str(row["canonical_sample_id"]) for row in rows]
    if len(set(sample_ids)) != expected:
        raise ValueError(f"{setting}: duplicate canonical sample IDs")
    if any((row.get("setting") != setting for row in rows)):
        raise ValueError(f"{setting}: per-row setting mismatch")
    if any((row.get("protocol") != expected_protocol for row in rows)):
        raise ValueError(f"{setting}: per-row protocol mismatch")
    count_values = counts_for_rows(rows)
    point = metrics_from_counts(count_values)
    worker_main = summary["summary"]["main"]
    worker_comparison = {
        "base_token_em": worker_main["base_token_exact_rate"],
        "editor_token_em": worker_main["editor_token_exact_rate"],
        "base_id_em": worker_main["base_identifier_exact_rate"],
        "editor_id_em": worker_main["editor_identifier_exact_rate"],
        "id_error_recovery": worker_main["identifier_error_recovery_rate"],
        "id_damage": worker_main["identifier_damage_rate"],
        "id_net": worker_main["identifier_net_gain_rate"],
    }
    for metric, expected_value in worker_comparison.items():
        if point[metric] != expected_value:
            raise ValueError(
                f"{setting}: recomputed {metric}={point[metric]!r} != worker summary {expected_value!r}"
            )
    integrity = {
        "status": "pass",
        "worker_summary_sha256": _sha256(summary_path),
        "predictions_sha256": prediction_sha,
        "predictions_rows": len(rows),
        "first_token_scores_sha256": first_token_sha,
        "first_token_scores_rows": first_token_rows,
        "unique_canonical_sample_ids": len(set(sample_ids)),
        "expected_samples": expected,
        "worker_point_estimates_recomputed_exactly": True,
    }
    return (summary, rows, integrity)


def _pairing_key(row: dict[str, Any]) -> tuple[Any, ...]:
    occurrence = row["occurrence"]
    return (
        row["canonical_sample_id"],
        row["repository"],
        row["file"],
        row["full_file_sha256"],
        occurrence["identifier_span_start"],
        occurrence["identifier_span_end"],
        occurrence["original_character_cursor"],
        occurrence["target_identifier"],
        occurrence["existing_identifier_prefix"],
        occurrence["remaining_identifier_suffix"],
        occurrence["cursor_kind"],
    )


def _verify_cross_model_pairing(
    rows_by_setting: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for language in ("python", "typescript"):
        left_name = f"qwen_{language}"
        right_name = f"starcoder2_{language}"
        left = rows_by_setting[left_name]
        right = rows_by_setting[right_name]
        if len(left) != len(right):
            raise ValueError(f"{language}: cross-model cohort length mismatch")
        mismatches = [
            index
            for (index, (left_row, right_row)) in enumerate(
                zip(left, right, strict=True)
            )
            if _pairing_key(left_row) != _pairing_key(right_row)
        ]
        if mismatches:
            raise ValueError(
                f"{language}: cross-model pairing mismatch at {mismatches[:10]}"
            )
        result[language] = {
            "paired_occurrences": len(left),
            "pairing_mismatches": 0,
            "same_source_file_identifier_and_original_cursor": True,
        }
    return result


def _fmt_rate(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}%"


def _fmt_ci(interval: list[float] | None) -> str:
    if interval is None:
        return "--"
    return f"[{100.0 * interval[0]:.2f}, {100.0 * interval[1]:.2f}]"


def _render_markdown(report: dict[str, Any]) -> str:
    independent_settings = (
        report.get("cross_model_pairing", {}).get("required") is False
    )
    canonical_paired = (
        str(report.get("cohort", {}).get("source")) == "frozen canonical paired subset"
    )
    max_context_tokens = int(report.get("max_context_tokens", 512))
    lines = [
        "# Reviewer 3 E1: same-cursor complete-identifier generation",
        "",
        f"Every setting uses the frozen original checkpoint, original character cursor, and a maximum {max_context_tokens}-token causal window. Base and Editor are paired within each occurrence: Editor selects only the first token, and all later tokens come from the frozen Base model. No-ID and capped generations remain in the denominator.",
        "",
    ]
    cohort_counts = report.get("cohort", {}).get("selected_samples_by_language")
    if cohort_counts:
        if independent_settings:
            denominator_text = f"Denominators come from the independent, non-deduplicated native full-Test manifests: Python {int(cohort_counts['python']):,} and TypeScript {int(cohort_counts['typescript']):,} occurrences per model. This E1-B census uses no cross-model intersection and no duplicate-file exclusion."
        elif canonical_paired:
            denominator_text = f"Denominators come from the frozen canonical paired subset: Python {int(cohort_counts['python']):,} and TypeScript {int(cohort_counts['typescript']):,} occurrences per model. Within each language, both models use the same source file, identifier occurrence, character cursor, prefix, and target."
        else:
            denominator_text = f"Denominators come from the completed outcome-independent cohort audit: Python {int(cohort_counts['python']):,} and TypeScript {int(cohort_counts['typescript']):,} paired occurrences per model, after zero-overlap E6 checks."
        lines.extend([denominator_text, ""])
    lines.extend(
        [
            (
                "## Main independent-setting results"
                if independent_settings
                else "## Main paired results"
            ),
            "",
            "| Setting | N | Base token EM | Editor token EM | Base ID EM | Editor ID EM | ID ErrRec | ID Dmg | ID Net |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTING_ORDER:
        item = report["settings"][setting]
        point = item["main"]["point_estimates"]
        lines.append(
            f"| {DISPLAY_NAMES[setting]} | {item['samples']:,} | {_fmt_rate(point['base_token_em'])} | {_fmt_rate(point['editor_token_em'])} | {_fmt_rate(point['base_id_em'])} | {_fmt_rate(point['editor_id_em'])} | {_fmt_rate(point['id_error_recovery'])} | {_fmt_rate(point['id_damage'])} | {_fmt_rate(point['id_net'])} |"
        )
    lines.extend(
        [
            "",
            "All uncertainty intervals below are 10,000-replicate paired-Base/Editor repository-cluster percentile bootstrap intervals (percentage points).",
            "",
            "| Setting | Base ID EM 95% CI | Editor ID EM 95% CI | ID ErrRec 95% CI | ID Dmg 95% CI | ID Net 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTING_ORDER:
        ci = report["settings"][setting]["main"]["ci95"]
        lines.append(
            f"| {DISPLAY_NAMES[setting]} | {_fmt_ci(ci['base_id_em'])} | {_fmt_ci(ci['editor_id_em'])} | {_fmt_ci(ci['id_error_recovery'])} | {_fmt_ci(ci['id_damage'])} | {_fmt_ci(ci['id_net'])} |"
        )
    lines.extend(
        [
            "",
            "## Original-cursor strata",
            "",
            "These are descriptive strata under the original sampling composition; conditional rates use aggregated confusion counts within each stratum.",
            "",
            "| Setting | Cursor | N | Base ID EM | Editor ID EM | ID ErrRec | ID Dmg | ID Net |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTING_ORDER:
        for cursor in ("identifier_start", "identifier_internal"):
            item = report["settings"][setting]["by_original_cursor_kind"][cursor]
            point = item["point_estimates"]
            lines.append(
                f"| {DISPLAY_NAMES[setting]} | {cursor} | {item['counts']['samples']:,} | {_fmt_rate(point['base_id_em'])} | {_fmt_rate(point['editor_id_em'])} | {_fmt_rate(point['id_error_recovery'])} | {_fmt_rate(point['id_damage'])} | {_fmt_rate(point['id_net'])} |"
            )
    lines.extend(
        [
            "",
            "## 50:50 cursor-composition sensitivity",
            "",
            "This standardized summary is a sensitivity analysis, not an estimate of real IDE cursor frequency. Conditional rates are formed only after weighting and aggregating confusion counts.",
            "",
            "| Setting | Base ID EM | Editor ID EM | ID ErrRec | ID Dmg | ID Net |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTING_ORDER:
        point = report["settings"][setting]["equal_cursor_50_50"]["point_estimates"]
        lines.append(
            f"| {DISPLAY_NAMES[setting]} | {_fmt_rate(point['base_id_em'])} | {_fmt_rate(point['editor_id_em'])} | {_fmt_rate(point['id_error_recovery'])} | {_fmt_rate(point['id_damage'])} | {_fmt_rate(point['id_net'])} |"
        )
    lines.extend(
        [
            "",
            "## What the old token-level fixes represent",
            "",
            "Rows here condition on old-token Base wrong and Editor correct. They are differences between evaluation standards, not automatically tokenization artifacts.",
            "",
            "| Setting | Base ID wrong → Editor ID correct | Base ID correct → Editor ID correct | Base ID wrong → Editor ID wrong | Base ID correct → Editor ID wrong | P(Base ID correct | Base token wrong) | ID fixes missed by old token fix |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTING_ORDER:
        main = report["settings"][setting]["main"]
        c = main["counts"]
        p = main["point_estimates"]
        lines.append(
            f"| {DISPLAY_NAMES[setting]} | {c['old_fix_base_id_wrong_editor_id_correct']:,} | {c['old_fix_base_id_correct_editor_id_correct']:,} | {c['old_fix_base_id_wrong_editor_id_wrong']:,} | {c['old_fix_base_id_correct_editor_id_wrong']:,} | {_fmt_rate(p['base_id_correct_given_base_token_wrong'])} | {c['identifier_fix_not_old_token_fix']:,} ({_fmt_rate(p['id_fixes_not_captured_by_old_token_fix'])}) |"
        )
    if independent_settings:
        integrity_text = "Strict identifier-start scoring (no horizontal-whitespace skip), all failure categories, generation horizons, hashes, and full interval sets are retained in `combined_summary.json`. Integrity is checked within each complete setting; no cross-model pairing assertion is used."
    else:
        integrity_text = "Strict identifier-start scoring (no horizontal-whitespace skip), all failure categories, generation horizons, hashes, and full interval sets are retained in `combined_summary.json`. The integrity check also confirms exact cross-model occurrence/cursor pairing within language."
    lines.extend(["", integrity_text, ""])
    return "\n".join(lines)


def _write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "setting",
        "display_name",
        "stratum",
        "samples",
        "repositories",
        "metric",
        "estimate",
        "ci95_low",
        "ci95_high",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for setting in SETTING_ORDER:
            item = report["settings"][setting]
            groups = {"main": item["main"], **item["by_original_cursor_kind"]}
            for stratum, group in groups.items():
                for metric, estimate in group["point_estimates"].items():
                    interval = group["ci95"][metric]
                    writer.writerow(
                        {
                            "setting": setting,
                            "display_name": DISPLAY_NAMES[setting],
                            "stratum": stratum,
                            "samples": group["counts"]["samples"],
                            "repositories": group["repository_count"],
                            "metric": metric,
                            "estimate": estimate,
                            "ci95_low": interval[0] if interval else None,
                            "ci95_high": interval[1] if interval else None,
                        }
                    )
    os.replace(temporary, path)


def run(
    *,
    input_dir: Path,
    output_dir: Path,
    iterations: int,
    seed: int,
    cohort_audit: Path | None = None,
    native_manifest_root: Path | None = None,
    require_cross_model_pairing: bool = True,
) -> dict[str, Any]:
    if cohort_audit is not None and native_manifest_root is not None:
        raise ValueError("choose either cohort audit or native manifest root")
    if native_manifest_root is not None:
        (expected_samples, cohort_provenance) = expected_samples_from_native_manifests(
            native_manifest_root
        )
    elif cohort_audit is None:
        expected_samples = None
        cohort_provenance: dict[str, Any] = {
            "source": "input worker summaries",
        }
    else:
        (expected_samples, cohort_provenance) = expected_samples_from_cohort_audit(
            cohort_audit
        )
    summaries: dict[str, dict[str, Any]] = {}
    rows_by_setting: dict[str, list[dict[str, Any]]] = {}
    integrity: dict[str, dict[str, Any]] = {}
    for setting in SETTING_ORDER:
        (summary, rows, validation) = _validate_setting(
            setting, input_dir / setting, expected_samples=expected_samples
        )
        summaries[setting] = summary
        rows_by_setting[setting] = rows
        integrity[setting] = validation
    cross_model_pairing = (
        _verify_cross_model_pairing(rows_by_setting)
        if require_cross_model_pairing
        else {"required": False, "status": "not_used_for_independent_setting_summaries"}
    )
    report: dict[str, Any] = {
        "protocol": SUMMARY_PROTOCOL,
        "source_protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {
            "iterations": int(iterations),
            "base_seed": int(seed),
            "confidence_interval": "two-sided 95% percentile",
        },
        "denominator_policy": "Every frozen-cohort occurrence remains in every unconditional denominator; No-ID, special-token, and generation-censored outputs are incorrect. ErrRec and Dmg use complete-ID Base-fail and Base-pass denominators, respectively.",
        "cohort": cohort_provenance,
        "integrity": integrity,
        "cross_model_pairing": cross_model_pairing,
        "settings": {},
    }
    for setting_index, setting in enumerate(SETTING_ORDER):
        rows = rows_by_setting[setting]
        by_cursor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_cursor[str(row["occurrence"]["cursor_kind"])].append(row)
        setting_seed = seed + setting_index * 10000
        item = {
            "samples": len(rows),
            "worker_elapsed_seconds": summaries[setting]["elapsed_seconds"],
            "model": summaries[setting]["model"],
            "tokenizer": summaries[setting]["tokenizer"],
            "checkpoint": summaries[setting]["checkpoint"],
            "route": summaries[setting]["route"],
            "main": repository_cluster_bootstrap(
                rows, iterations=iterations, seed=setting_seed
            ),
            "by_original_cursor_kind": {
                cursor: repository_cluster_bootstrap(
                    cursor_rows,
                    iterations=iterations,
                    seed=setting_seed + cursor_index + 1,
                )
                for (cursor_index, (cursor, cursor_rows)) in enumerate(
                    sorted(by_cursor.items())
                )
            },
            "equal_cursor_50_50": equal_cursor_standardization(by_cursor),
            "failure_and_generation_counts": _outcome_counts(rows),
            "route_active_count": sum(
                (bool(row["first_token"]["route"]["active"]) for row in rows)
            ),
            "reference_path_categories": dict(
                sorted(
                    Counter(
                        (
                            row["old_reference"]["reference_path_category"]
                            for row in rows
                        )
                    ).items()
                )
            ),
        }
        report["settings"][setting] = item
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "combined_summary.json"
    markdown_path = output_dir / "combined_summary.md"
    csv_path = output_dir / "combined_metrics.csv"
    _atomic_write(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, _render_markdown(report))
    _write_csv(report, csv_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--cohort-audit",
        type=Path,
        help="Optional completed cohort audit from which exact per-language census denominators and E6 isolation checks are derived.",
    )
    parser.add_argument(
        "--native-manifest-root",
        type=Path,
        help="Root containing python/typescript manifest_summary.json files for independent, non-deduplicated full-Test setting summaries.",
    )
    parser.add_argument(
        "--no-cross-model-pairing",
        action="store_true",
        help="Summarize each setting independently without a pairing assertion.",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.input_dir
    report = run(
        input_dir=args.input_dir,
        output_dir=output_dir,
        iterations=max(1, int(args.bootstrap_iterations)),
        seed=int(args.seed),
        cohort_audit=(
            args.cohort_audit.expanduser().resolve()
            if args.cohort_audit is not None
            else None
        ),
        native_manifest_root=(
            args.native_manifest_root.expanduser().resolve()
            if args.native_manifest_root is not None
            else None
        ),
        require_cross_model_pairing=not bool(args.no_cross_model_pairing),
    )
    compact = {
        setting: report["settings"][setting]["main"]["point_estimates"]
        for setting in SETTING_ORDER
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

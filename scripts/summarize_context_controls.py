from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence
from scripts.summarize_cursor_controls import paired_condition_bootstrap
from scripts.summarize_identifiers import (
    DISPLAY_NAMES,
    SETTING_ORDER,
    repository_cluster_bootstrap,
)

PROTOCOL = "reviewer3-e2-paired-context-summary-v1"
BINDING_GROUPS = (
    "Visible-512",
    "Restored-long",
    "Still-missing",
    "Unresolved / not applicable",
)
TEXT_GROUPS = (
    "Visible-512",
    "Restored-long",
    "Still-missing",
    "Unresolved / not applicable",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["canonical_sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate canonical sample IDs")
    return result


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _branch_exact(row: dict[str, Any], branch: str) -> bool:
    return bool(
        row[branch]["complete_identifier_tolerant_start"]["complete_identifier_exact"]
    )


def _transitions(
    rows_512: Sequence[dict[str, Any]], rows_long: Sequence[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    result = {}
    for branch in ("base", "editor_first_token_then_base"):
        counts = Counter()
        for left, right in zip(rows_512, rows_long, strict=True):
            left_ok = _branch_exact(left, branch)
            right_ok = _branch_exact(right, branch)
            counts[
                f"{('correct' if left_ok else 'wrong')}_to_{('correct' if right_ok else 'wrong')}"
            ] += 1
        result[branch] = {
            key: counts[key]
            for key in (
                "wrong_to_wrong",
                "wrong_to_correct",
                "correct_to_wrong",
                "correct_to_correct",
            )
        }
    return result


def _summarize_pair(
    rows_512: Sequence[dict[str, Any]],
    rows_long: Sequence[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if len(rows_512) != len(rows_long) or not rows_512:
        raise ValueError("context pair must be nonempty and aligned")
    for left, right in zip(rows_512, rows_long, strict=True):
        if left["canonical_sample_id"] != right["canonical_sample_id"]:
            raise ValueError("context pair occurrence mismatch")
        if left["repository"] != right["repository"]:
            raise ValueError("context pair repository mismatch")
    return {
        "samples": len(rows_512),
        "k512": repository_cluster_bootstrap(
            rows_512, iterations=iterations, seed=seed
        ),
        "k2048": repository_cluster_bootstrap(
            rows_long, iterations=iterations, seed=seed + 1
        ),
        "paired_difference": paired_condition_bootstrap(
            {"k512": rows_512, "k2048": rows_long}, iterations=iterations, seed=seed + 2
        ),
        "complete_id_transitions": _transitions(rows_512, rows_long),
    }


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}%"


def _fmt_delta(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:+.2f} pp"


def _fmt_ci(value: list[float] | None) -> str:
    if value is None:
        return "--"
    return f"[{100.0 * value[0]:+.2f}, {100.0 * value[1]:+.2f}]"


def _result_row(setting: str, group: str, summary: dict[str, Any]) -> str:
    p512 = summary["k512"]["point_estimates"]
    plong = summary["k2048"]["point_estimates"]
    return f"| {DISPLAY_NAMES[setting]} | {group} | {summary['samples']:,} | {_fmt(p512['base_id_em'])} | {_fmt(plong['base_id_em'])} | {_fmt(p512['editor_id_em'])} | {_fmt(plong['editor_id_em'])} | {_fmt(p512['id_net'])} | {_fmt(plong['id_net'])} |"


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# Reviewer 3 E2: paired 512-versus-2048 context",
        "",
        "The source file, identifier occurrence, original character cursor, checkpoint, first-step editor, greedy generation, and complete-ID scorer are fixed. The only intervention is the amount of causal left context. Every K=2048 input has its K=512 input as an exact token-ID suffix.",
        "",
        "## Overall",
        "",
        "| Setting | Group | N | Base ID@512 | Base ID@2048 | Editor ID@512 | Editor ID@2048 | Net@512 | Net@2048 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTING_ORDER:
        lines.append(
            _result_row(
                setting,
                "All long-prefix candidates",
                report["settings"][setting]["overall"],
            )
        )
    lines.extend(
        [
            "",
            "## Conservative local-binding groups",
            "",
            "Binding groups are restricted to compiler/AST-resolved direct local uses with prior same-symbol evidence. `Unresolved / not applicable` does not mean that no declaration or binding exists.",
            "",
            "| Setting | Group | N | Base ID@512 | Base ID@2048 | Editor ID@512 | Editor ID@2048 | Net@512 | Net@2048 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTING_ORDER:
        groups = report["settings"][setting]["by_conservative_binding"]
        for group in BINDING_GROUPS:
            if group in groups:
                lines.append(_result_row(setting, group, groups[group]))
    lines.extend(
        [
            "",
            "## Prior source-text occurrence groups",
            "",
            "These groups use exact earlier text with identifier character boundaries; they are deliberately not described as binding evidence.",
            "",
            "| Setting | Group | N | Base ID@512 | Base ID@2048 | Editor ID@512 | Editor ID@2048 | Net@512 | Net@2048 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTING_ORDER:
        groups = report["settings"][setting]["by_prior_text_occurrence"]
        for group in TEXT_GROUPS:
            if group in groups:
                lines.append(_result_row(setting, group, groups[group]))
    lines.extend(
        [
            "",
            "## Overall paired changes",
            "",
            "Intervals are from 10,000 repository-cluster paired percentile bootstrap replicates.",
            "",
            "| Setting | Metric | K=2048 − K=512 | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for setting in SETTING_ORDER:
        paired = report["settings"][setting]["overall"]["paired_difference"]
        point = paired["point_differences"]["k2048_minus_k512"]
        ci = paired["difference_ci95"]["k2048_minus_k512"]
        for metric in ("base_id_em", "editor_id_em", "id_net"):
            lines.append(
                f"| {DISPLAY_NAMES[setting]} | {metric} | {_fmt_delta(point[metric])} | {_fmt_ci(ci[metric])} |"
            )
    lines.extend(
        [
            "",
            "K=8192 was not run because K=2048 restored all reliably parsed binding-evidence cases that were missing at K=512 in the three settings where such cases existed; the fourth had none missing. Full transition counts and all group-specific intervals are in JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "setting",
        "stratification",
        "group",
        "samples",
        "context",
        "metric",
        "estimate",
        "ci95_low",
        "ci95_high",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for setting in SETTING_ORDER:
            item = report["settings"][setting]
            sections = [("overall", "all", item["overall"])]
            sections.extend(
                (
                    ("conservative_binding", group, value)
                    for (group, value) in item["by_conservative_binding"].items()
                )
            )
            sections.extend(
                (
                    ("prior_text_occurrence", group, value)
                    for (group, value) in item["by_prior_text_occurrence"].items()
                )
            )
            for stratification, group, value in sections:
                for context in ("k512", "k2048"):
                    bootstrap = value[context]
                    for metric, estimate in bootstrap["point_estimates"].items():
                        interval = bootstrap["ci95"][metric]
                        writer.writerow(
                            {
                                "setting": setting,
                                "stratification": stratification,
                                "group": group,
                                "samples": value["samples"],
                                "context": context,
                                "metric": metric,
                                "estimate": estimate,
                                "ci95_low": interval[0] if interval else None,
                                "ci95_high": interval[1] if interval else None,
                            }
                        )
    os.replace(temporary, path)


def run(
    *,
    e1_root: Path,
    input_root: Path,
    long_root: Path,
    output_dir: Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    input_audit_path = input_root / "e2_input_audit.json"
    input_audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
    if input_audit["status"] != "complete":
        raise ValueError("E2 input/evidence audit is incomplete")
    if input_audit["settings_triggering_k8192"]:
        raise ValueError("K=8192 evidence trigger fired; do not summarize only K=2048")
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": int(iterations),
        "base_seed": int(seed),
        "input_audit": {
            "path": str(input_audit_path.resolve()),
            "sha256": _sha256(input_audit_path),
        },
        "k8192_triggered": False,
        "settings": {},
    }
    for setting_index, setting in enumerate(SETTING_ORDER):
        expected = int(input_audit["settings"][setting]["samples"])
        manifest_rows = _read(input_root / setting / "manifest.jsonl.gz")
        evidence_rows = _read(input_root / setting / "evidence_rows.jsonl.gz")
        if len(manifest_rows) != expected or len(evidence_rows) != expected:
            raise ValueError(f"{setting}: E2 input row-count mismatch")
        ordered_ids = [str(row["canonical_sample_id"]) for row in manifest_rows]
        if ordered_ids != [str(row["canonical_sample_id"]) for row in evidence_rows]:
            raise ValueError(f"{setting}: evidence order mismatch")
        e1_path = e1_root / setting / "predictions.jsonl.gz"
        long_path = long_root / setting / "predictions.jsonl.gz"
        e1_rows_all = _read(e1_path)
        long_rows = _read(long_path)
        e1_index = _index(e1_rows_all)
        long_index = _index(long_rows)
        if set(long_index) != set(ordered_ids):
            raise ValueError(f"{setting}: K=2048 occurrence set mismatch")
        rows_512 = [e1_index[sample_id] for sample_id in ordered_ids]
        rows_long = [long_index[sample_id] for sample_id in ordered_ids]
        nested_failures = 0
        occurrence_failures = 0
        for left, right in zip(rows_512, rows_long, strict=True):
            if right["input"]["token_ids"][-512:] != left["input"]["token_ids"]:
                nested_failures += 1
            left_occurrence = left["occurrence"]
            right_occurrence = right["occurrence"]
            keys = (
                "identifier_span_start",
                "identifier_span_end",
                "original_character_cursor",
                "target_identifier",
                "existing_identifier_prefix",
                "remaining_identifier_suffix",
            )
            if any((left_occurrence[key] != right_occurrence[key] for key in keys)):
                occurrence_failures += 1
        if nested_failures or occurrence_failures:
            raise ValueError(
                f"{setting}: nested={nested_failures}, occurrence={occurrence_failures}"
            )
        worker_path = long_root / setting / "worker_summary.json"
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        if (
            worker["protocol"] != "reviewer3-e2-paired-context-k2048-v1"
            or int(worker["contract"]["prompt_context_tokens"]) != 2048
            or int(worker["samples"]) != expected
        ):
            raise ValueError(f"{setting}: invalid K=2048 worker contract")
        original_worker = json.loads(
            (e1_root / setting / "worker_summary.json").read_text(encoding="utf-8")
        )
        if (
            worker["checkpoint"]["sha256"] != original_worker["checkpoint"]["sha256"]
            or worker["model"]["weight_sha256"]
            != original_worker["model"]["weight_sha256"]
        ):
            raise ValueError(f"{setting}: model/checkpoint changed across context")
        base_seed = seed + setting_index * 100000
        overall = _summarize_pair(
            rows_512, rows_long, iterations=iterations, seed=base_seed
        )
        evidence_by_id = _index(evidence_rows)
        grouped: dict[str, dict[str, list[str]]] = {
            "binding": {group: [] for group in BINDING_GROUPS},
            "text": {group: [] for group in TEXT_GROUPS},
        }
        for sample_id in ordered_ids:
            evidence = evidence_by_id[sample_id]
            grouped["binding"][str(evidence["conservative_binding"]["group"])].append(
                sample_id
            )
            grouped["text"][str(evidence["prior_text_occurrence"]["group"])].append(
                sample_id
            )

        def summarize_groups(kind: str, seed_offset: int) -> dict[str, Any]:
            result = {}
            for group_index, (group, sample_ids) in enumerate(grouped[kind].items()):
                if not sample_ids:
                    continue
                result[group] = _summarize_pair(
                    [e1_index[sample_id] for sample_id in sample_ids],
                    [long_index[sample_id] for sample_id in sample_ids],
                    iterations=iterations,
                    seed=base_seed + seed_offset + group_index * 10,
                )
            return result

        report["settings"][setting] = {
            "samples": expected,
            "integrity": {
                "status": "pass",
                "same_occurrence_and_cursor": True,
                "k512_is_exact_token_suffix_of_k2048": True,
                "nested_input_failures": nested_failures,
                "occurrence_mismatches": occurrence_failures,
                "same_model_weight_sha256": worker["model"]["weight_sha256"],
                "same_checkpoint_sha256": worker["checkpoint"]["sha256"],
                "k512_source_predictions": {
                    "path": str(e1_path.resolve()),
                    "sha256": _sha256(e1_path),
                    "evaluated_rows": expected,
                },
                "k2048_predictions": {
                    "path": str(long_path.resolve()),
                    "sha256": _sha256(long_path),
                    "rows": len(long_rows),
                },
                "k2048_worker_summary": {
                    "path": str(worker_path.resolve()),
                    "sha256": _sha256(worker_path),
                },
            },
            "overall": overall,
            "by_conservative_binding": summarize_groups("binding", 1000),
            "by_prior_text_occurrence": summarize_groups("text", 2000),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "context_summary.json"
    markdown_path = output_dir / "context_summary.md"
    csv_path = output_dir / "context_metrics.csv"
    _write(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write(markdown_path, _render(report))
    _write_csv(report, csv_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    report = run(
        e1_root=args.e1_root.expanduser().resolve(),
        input_root=args.input_root.expanduser().resolve(),
        long_root=args.long_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        iterations=max(1, int(args.bootstrap_iterations)),
        seed=int(args.seed),
    )
    compact = {
        setting: report["settings"][setting]["overall"]["paired_difference"][
            "point_differences"
        ]["k2048_minus_k512"]
        for setting in SETTING_ORDER
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

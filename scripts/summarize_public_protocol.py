from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence
from scripts.summarize_cursor_controls import paired_condition_bootstrap
from scripts.summarize_identifiers import repository_cluster_bootstrap

PROTOCOL = "reviewer3-e3-public-protocol-summary-v1"
SETTINGS = (
    "qwen_python",
    "starcoder2_python",
    "qwen_typescript",
    "starcoder2_typescript",
)
CONDITIONS = ("benchmark_original_prefix", "target_aligned_prefix")
DATASETS = ("crosscodeeval_python", "repobench_python", "crosscodeeval_typescript")
DISPLAY = {
    "qwen_python": "Qwen / Python",
    "starcoder2_python": "StarCoder2 / Python",
    "qwen_typescript": "Qwen / TypeScript",
    "starcoder2_typescript": "StarCoder2 / TypeScript",
}
DATASET_DISPLAY = {
    "crosscodeeval_python": "CrossCodeEval-Python",
    "repobench_python": "RepoBench-Python",
    "crosscodeeval_typescript": "CrossCodeEval-TypeScript",
}


def _read(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result = {"path": str(path.resolve()), "sha256": _sha256(path)}
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _write(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["canonical_sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate E3 canonical sample IDs")
    return result


def _align(
    rows_by_name: dict[str, Sequence[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    names = tuple(rows_by_name)
    if not names:
        raise ValueError("cannot align empty E3 condition set")
    indexes = {name: _index(rows) for (name, rows) in rows_by_name.items()}
    ids = set(indexes[names[0]])
    for name in names[1:]:
        if set(indexes[name]) != ids:
            raise ValueError(f"{name}: E3 sample identity mismatch")
    ordered = sorted(ids)
    result = {
        name: [indexes[name][sample_id] for sample_id in ordered] for name in names
    }
    for position, sample_id in enumerate(ordered):
        datasets = {str(result[name][position]["dataset"]) for name in names}
        repositories = {str(result[name][position]["repository"]) for name in names}
        if len(datasets) != 1 or len(repositories) != 1:
            raise ValueError(f"{sample_id}: paired source identity mismatch")
    return result


def _legacy_row(
    document: dict[str, Any], setting: str, dataset: str, condition: str
) -> dict[str, Any]:
    row = document["settings"][setting]["datasets"][dataset]
    if condition == "target_aligned_prefix":
        base = float(row["base_first_rerun"])
        editor = float(row["repair_first"])
    else:
        base = float(row["base_first"])
        editor = float(row["repair_first"])
    return {
        "N": int(row["N"]),
        "base_token_em": base,
        "editor_token_em": editor,
        "token_net": editor - base,
    }


def _validate_worker_pair(raw_root: Path, setting: str) -> dict[str, Any]:
    reports = {}
    for condition in CONDITIONS:
        path = raw_root / setting / condition / "worker_summary.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ValueError(f"incomplete E3 worker: {path}")
        if str(report.get("condition")) != condition:
            raise ValueError(f"condition mismatch: {path}")
        reports[condition] = (path, report)
    checkpoints = {
        str(report["checkpoint"]["sha256"]) for (_, report) in reports.values()
    }
    model_weights = {
        str(report["model"]["weight_sha256"]) for (_, report) in reports.values()
    }
    manifests = {
        str(report["inputs"]["manifest_sha256"]) for (_, report) in reports.values()
    }
    if len(checkpoints) != 1 or len(model_weights) != 1 or len(manifests) != 1:
        raise ValueError(
            f"{setting}: protocol workers did not freeze common inputs/model"
        )
    return {
        "same_checkpoint": True,
        "same_model_weights": True,
        "same_setting_manifest": True,
        "checkpoint_sha256": next(iter(checkpoints)),
        "model_weight_sha256": next(iter(model_weights)),
        "manifest_sha256": next(iter(manifests)),
        "worker_reports": {
            condition: _artifact(path) for (condition, (path, _)) in reports.items()
        },
    }


def _setting_summary(
    *, setting: str, prediction_root: Path, raw_root: Path, iterations: int, seed: int
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    raw_rows: dict[str, list[dict[str, Any]]] = {}
    artifacts: dict[str, Any] = {}
    for condition in CONDITIONS:
        path = prediction_root / setting / condition / "predictions.jsonl.gz"
        rows = _read(path)
        if any((str(row.get("setting")) != setting for row in rows)):
            raise ValueError(f"{setting}/{condition}: setting mismatch")
        if any(
            (
                str(row.get("occurrence", {}).get("protocol_condition")) != condition
                for row in rows
            )
        ):
            raise ValueError(f"{setting}/{condition}: protocol mismatch")
        raw_rows[condition] = rows
        artifacts[condition] = _artifact(path, len(rows))
    aligned = _align(raw_rows)
    dataset_results: dict[str, Any] = {}
    for dataset_index, dataset in enumerate(DATASETS):
        selected = {
            condition: [
                row for row in aligned[condition] if str(row["dataset"]) == dataset
            ]
            for condition in CONDITIONS
        }
        if not selected[CONDITIONS[0]]:
            continue
        selected = _align(selected)
        condition_results = {
            condition: repository_cluster_bootstrap(
                selected[condition],
                iterations=iterations,
                seed=seed + 1000 * dataset_index + 17 * condition_index,
            )
            for (condition_index, condition) in enumerate(CONDITIONS)
        }
        paired = paired_condition_bootstrap(
            selected, iterations=iterations, seed=seed + 5000 + dataset_index
        )
        dataset_results[dataset] = {
            "N": len(selected[CONDITIONS[0]]),
            "conditions": condition_results,
            "paired_protocol_bootstrap": paired,
        }
    overall_conditions = {
        condition: repository_cluster_bootstrap(
            aligned[condition],
            iterations=iterations,
            seed=seed + 20000 + condition_index,
        )
        for (condition_index, condition) in enumerate(CONDITIONS)
    }
    overall_pair = paired_condition_bootstrap(
        aligned, iterations=iterations, seed=seed + 25000
    )
    return (
        {
            "N": len(aligned[CONDITIONS[0]]),
            "integrity": _validate_worker_pair(raw_root, setting),
            "prediction_artifacts": artifacts,
            "datasets": dataset_results,
            "overall": {
                "conditions": overall_conditions,
                "paired_protocol_bootstrap": overall_pair,
            },
        },
        aligned,
    )


def _cross_model(
    all_rows: dict[str, dict[str, list[dict[str, Any]]]], *, iterations: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pairs = (
        ("python", "qwen_python", "starcoder2_python"),
        ("typescript", "qwen_typescript", "starcoder2_typescript"),
    )
    for language_index, (language, qwen, star) in enumerate(pairs):
        result[language] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            aligned = _align(
                {
                    "qwen": all_rows[qwen][condition],
                    "starcoder2": all_rows[star][condition],
                }
            )
            result[language][condition] = {
                "N": len(aligned["qwen"]),
                "paired_model_bootstrap": paired_condition_bootstrap(
                    aligned,
                    iterations=iterations,
                    seed=seed + 30000 + 1000 * language_index + condition_index,
                ),
            }
    return result


def _preselected_cases(
    input_audit: dict[str, Any], all_rows: dict[str, dict[str, list[dict[str, Any]]]]
) -> dict[str, Any]:
    indexes = {
        setting: {condition: _index(rows) for (condition, rows) in conditions.items()}
        for (setting, conditions) in all_rows.items()
    }
    output: dict[str, Any] = {}
    for category, cases in input_audit.get("pre_outcome_examples", {}).items():
        enriched = []
        for case in cases:
            sample_id = str(case["canonical_sample_id"])
            record: dict[str, Any] = dict(case)
            record["selection_uses_model_outcome"] = False
            record["model_outputs"] = {}
            for setting in SETTINGS:
                if sample_id not in indexes[setting][CONDITIONS[0]]:
                    continue
                record["model_outputs"][setting] = {}
                for condition in CONDITIONS:
                    row = indexes[setting][condition][sample_id]
                    record["model_outputs"][setting][condition] = {
                        "base_first_token": row["first_token"]["base_token_decoded"],
                        "editor_first_token": row["first_token"][
                            "editor_token_decoded"
                        ],
                        "base_generated_text": row["base"]["decoded_generation"][
                            "added_text"
                        ],
                        "editor_generated_text": row["editor_first_token_then_base"][
                            "decoded_generation"
                        ]["added_text"],
                        "base_extracted_identifier": row["base"][
                            "complete_identifier_tolerant_start"
                        ]["predicted_identifier"],
                        "editor_extracted_identifier": row[
                            "editor_first_token_then_base"
                        ]["complete_identifier_tolerant_start"]["predicted_identifier"],
                        "base_outcome": row["base"][
                            "complete_identifier_tolerant_start"
                        ]["outcome"],
                        "editor_outcome": row["editor_first_token_then_base"][
                            "complete_identifier_tolerant_start"
                        ]["outcome"],
                    }
            enriched.append(record)
        output[category] = enriched
    return output


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}%"


def _fmt_delta(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:+.2f} pp"


def _fmt_ci(interval: list[float] | None) -> str:
    if interval is None:
        return "--"
    return f"[{100.0 * interval[0]:+.2f}, {100.0 * interval[1]:+.2f}]"


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# Reviewer 3 E3: public protocol replication and paired audit",
        "",
        "The two protocols retain their own prediction task. The target-aligned condition appends only the predeclared reference characters before the first non-keyword identifier; it does not append the identifier. Same benchmark ID therefore does not imply the same model input or token target.",
        "",
        "## Input and endpoint audit",
        "",
        "| Dataset | Common N | Added chars, median / p95 / max | At original endpoint | Horizontal whitespace | Contains newline | Other reference code |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    input_audit = report["input_audit"]["content"]
    for dataset in DATASETS:
        block = input_audit["leading_text"][dataset]
        dist = block["added_character_distribution"]
        cats = block["category_counts"]
        lines.append(
            f"| {DATASET_DISPLAY[dataset]} | {input_audit['dataset_rows'][dataset]:,} | {dist['median']:.1f} / {dist['p95']} / {dist['max']} | {cats.get('identifier_at_benchmark_endpoint', 0):,} | {cats.get('horizontal_whitespace_only', 0):,} | {cats.get('contains_newline', 0):,} | {cats.get('contains_nonwhitespace_reference_code', 0):,} |"
        )
    lines.extend(
        [
            "",
            "## Archived-record reproduction",
            "",
            "These rows are recalculated from the archived per-sample reports, not copied from displayed manuscript decimals.",
            "",
            "| Setting | Dataset | Protocol | N | Base token EM | Editor token EM | Net |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for dataset in DATASETS:
            if dataset not in report["legacy_reproduction"]["rows"].get(setting, {}):
                continue
            for condition in CONDITIONS:
                row = report["legacy_reproduction"]["rows"][setting][dataset][condition]
                lines.append(
                    f"| {DISPLAY[setting]} | {DATASET_DISPLAY[dataset]} | {condition} | {row['N']:,} | {_fmt(row['base_token_em'])} | {_fmt(row['editor_token_em'])} | {_fmt_delta(row['token_net'])} |"
                )
    lines.extend(
        [
            "",
            "## New common-set complete-identifier evaluation",
            "",
            "For the original-prefix condition, ID correctness additionally requires exact generation of the fixed reference text before the target identifier. No later-output search is used.",
            "",
            "| Setting | Dataset | Protocol | N | Base token EM | Editor token EM | Base ID EM | Editor ID EM | ID ErrRec | ID Dmg | ID Net |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for dataset, block in report["settings"][setting]["datasets"].items():
            for condition in CONDITIONS:
                point = block["conditions"][condition]["point_estimates"]
                lines.append(
                    f"| {DISPLAY[setting]} | {DATASET_DISPLAY[dataset]} | {condition} | {block['N']:,} | {_fmt(point['base_token_em'])} | {_fmt(point['editor_token_em'])} | {_fmt(point['base_id_em'])} | {_fmt(point['editor_id_em'])} | {_fmt(point['id_error_recovery'])} | {_fmt(point['id_damage'])} | {_fmt(point['id_net'])} |"
                )
    lines.extend(
        [
            "",
            "## Paired protocol sensitivity",
            "",
            "Intervals use 10,000 repository-cluster paired bootstrap replicates.",
            "",
            "| Setting | Dataset | Target-aligned minus original ID Net | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    contrast = "target_aligned_prefix_minus_benchmark_original_prefix"
    for setting in SETTINGS:
        for dataset, block in report["settings"][setting]["datasets"].items():
            paired = block["paired_protocol_bootstrap"]
            value = paired["point_differences"][contrast]["id_net"]
            interval = paired["difference_ci95"][contrast]["id_net"]
            lines.append(
                f"| {DISPLAY[setting]} | {DATASET_DISPLAY[dataset]} | {_fmt_delta(value)} | {_fmt_ci(interval)} |"
            )
    lines.extend(
        [
            "",
            "Changing the endpoint also changes visible reference spelling and sometimes the token target. These paired differences establish protocol sensitivity and are not a tokenizer-artifact fraction.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    prediction_root: Path,
    raw_root: Path,
    input_audit_path: Path,
    harmonization_path: Path,
    legacy_first_path: Path,
    legacy_original_path: Path,
    output_dir: Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    input_audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
    harmonization = json.loads(harmonization_path.read_text(encoding="utf-8"))
    legacy_first = json.loads(legacy_first_path.read_text(encoding="utf-8"))
    legacy_original = json.loads(legacy_original_path.read_text(encoding="utf-8"))
    if (
        input_audit.get("status") != "complete"
        or harmonization.get("status") != "complete"
    ):
        raise ValueError("E3 input or harmonization audit is incomplete")
    report: dict[str, Any] = {
        "status": "complete",
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        "input_audit": {**_artifact(input_audit_path), "content": input_audit},
        "identical_input_harmonization": {
            **_artifact(harmonization_path),
            "content": harmonization,
        },
        "legacy_reproduction": {
            "first_identifier": _artifact(legacy_first_path),
            "original_prefix": _artifact(legacy_original_path),
            "rows": {},
        },
        "settings": {},
    }
    for setting in SETTINGS:
        report["legacy_reproduction"]["rows"][setting] = {}
        expected_datasets = (
            ("crosscodeeval_typescript",)
            if setting.endswith("typescript")
            else ("crosscodeeval_python", "repobench_python")
        )
        for dataset in expected_datasets:
            report["legacy_reproduction"]["rows"][setting][dataset] = {
                "benchmark_original_prefix": _legacy_row(
                    legacy_original, setting, dataset, "benchmark_original_prefix"
                ),
                "target_aligned_prefix": _legacy_row(
                    legacy_first, setting, dataset, "target_aligned_prefix"
                ),
            }
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for setting_index, setting in enumerate(SETTINGS):
        (setting_report, aligned_rows) = _setting_summary(
            setting=setting,
            prediction_root=prediction_root,
            raw_root=raw_root,
            iterations=iterations,
            seed=seed + 100000 * setting_index,
        )
        report["settings"][setting] = setting_report
        all_rows[setting] = aligned_rows
    report["cross_model_same_text_pairs"] = _cross_model(
        all_rows, iterations=iterations, seed=seed
    )
    cases = _preselected_cases(input_audit, all_rows)
    cases_path = output_dir / "preselected_cases.json"
    _write_json(
        cases_path,
        {
            "status": "complete",
            "selection": "Input-only categories and deterministic source order, fixed by the E3 input builder before inference.",
            "cases": cases,
        },
    )
    report["preselected_cases"] = _artifact(cases_path)
    _write_json(output_dir / "protocol_summary.json", report)
    _write(output_dir / "protocol_summary.md", _render(report) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--harmonization", type=Path, required=True)
    parser.add_argument("--legacy-first", type=Path, required=True)
    parser.add_argument("--legacy-original", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    report = run(
        prediction_root=args.prediction_root.expanduser().resolve(),
        raw_root=args.raw_root.expanduser().resolve(),
        input_audit_path=args.input_audit.expanduser().resolve(),
        harmonization_path=args.harmonization.expanduser().resolve(),
        legacy_first_path=args.legacy_first.expanduser().resolve(),
        legacy_original_path=args.legacy_original.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    print(
        json.dumps(
            {
                setting: {
                    dataset: {
                        condition: block["conditions"][condition]["point_estimates"][
                            "id_net"
                        ]
                        for condition in CONDITIONS
                    }
                    for (dataset, block) in setting_report["datasets"].items()
                }
                for (setting, setting_report) in report["settings"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

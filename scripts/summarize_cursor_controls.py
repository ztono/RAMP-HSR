from __future__ import annotations
import argparse
from collections import defaultdict
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
from scripts.summarize_identifiers import (
    DISPLAY_NAMES,
    SETTING_ORDER,
    counts_for_rows,
    metrics_from_counts,
    repository_cluster_bootstrap,
)

PROTOCOL = "reviewer3-e1-paired-cursor-summary-v1"
PAIR_METRICS = (
    "base_id_em",
    "editor_id_em",
    "id_error_recovery",
    "id_damage",
    "id_net",
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


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _index_unique(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["canonical_sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate canonical sample IDs")
    return result


def _align(
    rows_by_condition: dict[str, Sequence[dict[str, Any]]], ordered_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for condition, rows in rows_by_condition.items():
        index = _index_unique(rows)
        missing = [sample_id for sample_id in ordered_ids if sample_id not in index]
        if missing:
            raise ValueError(
                f"{condition}: missing {len(missing)} paired rows; first={missing[:3]}"
            )
        result[condition] = [index[sample_id] for sample_id in ordered_ids]
    for position, sample_id in enumerate(ordered_ids):
        repositories = {
            normalize_repo_id(rows[position]["repository"]) for rows in result.values()
        }
        if len(repositories) != 1:
            raise ValueError(f"repository mismatch for {sample_id}")
    return result


def paired_condition_bootstrap(
    rows_by_condition: dict[str, Sequence[dict[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    conditions = tuple(rows_by_condition)
    lengths = {len(rows) for rows in rows_by_condition.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("paired condition rows must have one nonzero length")
    by_repository: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {condition: [] for condition in conditions}
    )
    row_count = next(iter(lengths))
    for position in range(row_count):
        repositories = {
            normalize_repo_id(rows_by_condition[condition][position]["repository"])
            for condition in conditions
        }
        if len(repositories) != 1:
            raise ValueError("paired repository identity mismatch")
        repository = next(iter(repositories))
        for condition in conditions:
            by_repository[repository][condition].append(
                rows_by_condition[condition][position]
            )
    repositories = sorted(by_repository)
    matrices = {
        condition: np.stack(
            [
                counts_for_rows(by_repository[repository][condition])
                for repository in repositories
            ]
        )
        for condition in conditions
    }
    points = {
        condition: metrics_from_counts(matrix.sum(axis=0))
        for (condition, matrix) in matrices.items()
    }
    pair_names = [
        (left, right)
        for (left_index, left) in enumerate(conditions)
        for right in conditions[left_index + 1 :]
    ]
    point_differences = {
        f"{right}_minus_{left}": {
            metric: (
                float(points[right][metric]) - float(points[left][metric])
                if points[right][metric] is not None
                and points[left][metric] is not None
                else None
            )
            for metric in PAIR_METRICS
        }
        for (left, right) in pair_names
    }
    draws: dict[str, dict[str, list[float]]] = {
        pair: {metric: [] for metric in PAIR_METRICS} for pair in point_differences
    }
    rng = np.random.default_rng(seed)
    remaining = int(iterations)
    while remaining:
        batch = min(512, remaining)
        indices = rng.integers(0, len(repositories), size=(batch, len(repositories)))
        sampled = {
            condition: matrix[indices].sum(axis=1)
            for (condition, matrix) in matrices.items()
        }
        for row_index in range(batch):
            sampled_metrics = {
                condition: metrics_from_counts(values[row_index])
                for (condition, values) in sampled.items()
            }
            for left, right in pair_names:
                pair = f"{right}_minus_{left}"
                for metric in PAIR_METRICS:
                    left_value = sampled_metrics[left][metric]
                    right_value = sampled_metrics[right][metric]
                    if left_value is None or right_value is None:
                        continue
                    difference = float(right_value) - float(left_value)
                    if math.isfinite(difference):
                        draws[pair][metric].append(difference)
        remaining -= batch
    return {
        "method": "paired repository-cluster percentile bootstrap; each repository draw retains the same occurrence under every cursor condition",
        "iterations": int(iterations),
        "seed": int(seed),
        "samples": row_count,
        "repository_count": len(repositories),
        "condition_point_estimates": points,
        "point_differences": point_differences,
        "difference_ci95": {
            pair: {
                metric: (
                    [
                        float(np.percentile(values, 2.5)),
                        float(np.percentile(values, 97.5)),
                    ]
                    if values
                    else None
                )
                for (metric, values) in metric_draws.items()
            }
            for (pair, metric_draws) in draws.items()
        },
    }


def _condition_artifact(
    rows: Sequence[dict[str, Any]], path: Path, *, iterations: int, seed: int
) -> dict[str, Any]:
    return {
        "predictions": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "evaluated_rows": len(rows),
        },
        "repository_bootstrap": repository_cluster_bootstrap(
            rows, iterations=iterations, seed=seed
        ),
    }


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
        "# Reviewer 3 E1: paired cursor controls",
        "",
        "Identifier-start uses all 5,000 pre-fixed occurrences per language. Native-internal uses only occurrences with a shared continuous-source token boundary for both normal tokenizer/model pairs; all ineligible occurrences remain documented as N/A. Identical model inputs are anchored to the earlier original-cursor generation by an input-only rule.",
        "",
        "## All fixed occurrences: original versus identifier start",
        "",
        "| Setting | Condition | N | Base ID EM | Editor ID EM | ID ErrRec | ID Dmg | ID Net |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTING_ORDER:
        section = report["settings"][setting]["all_fixed_5000"]
        for condition in ("original", "identifier_start"):
            point = section["conditions"][condition]["repository_bootstrap"][
                "point_estimates"
            ]
            lines.append(
                f"| {DISPLAY_NAMES[setting]} | {condition} | 5,000 | {_fmt(point['base_id_em'])} | {_fmt(point['editor_id_em'])} | {_fmt(point['id_error_recovery'])} | {_fmt(point['id_damage'])} | {_fmt(point['id_net'])} |"
            )
    lines.extend(
        [
            "",
            "## Shared-native-boundary common subset",
            "",
            "Cross-condition comparisons below use exactly the same occurrence set. Token EM is not mixed across conditions because the native condition's reference is the continuous token sequence, whereas the other conditions retain their own reference construction.",
            "",
            "| Setting | Condition | N | Base ID EM | Editor ID EM | ID ErrRec | ID Dmg | ID Net |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTING_ORDER:
        section = report["settings"][setting]["native_common_subset"]
        for condition in ("original", "identifier_start", "native_internal"):
            point = section["conditions"][condition]["repository_bootstrap"][
                "point_estimates"
            ]
            lines.append(
                f"| {DISPLAY_NAMES[setting]} | {condition} | {section['samples']:,} | {_fmt(point['base_id_em'])} | {_fmt(point['editor_id_em'])} | {_fmt(point['id_error_recovery'])} | {_fmt(point['id_damage'])} | {_fmt(point['id_net'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired change in ID Net",
            "",
            "Deltas and 95% intervals are percentage points from 10,000 paired repository-cluster bootstrap replicates.",
            "",
            "| Setting | Common set | Contrast | ID Net delta | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for setting in SETTING_ORDER:
        setting_report = report["settings"][setting]
        for section_name, contrasts in (
            ("all_fixed_5000", ("identifier_start_minus_original",)),
            (
                "native_common_subset",
                (
                    "identifier_start_minus_original",
                    "native_internal_minus_original",
                    "native_internal_minus_identifier_start",
                ),
            ),
        ):
            paired = setting_report[section_name]["paired_bootstrap"]
            for contrast in contrasts:
                point = paired["point_differences"][contrast]["id_net"]
                interval = paired["difference_ci95"][contrast]["id_net"]
                lines.append(
                    f"| {DISPLAY_NAMES[setting]} | {section_name} | {contrast} | {_fmt_delta(point)} | {_fmt_ci(interval)} |"
                )
    lines.extend(
        [
            "",
            "Changing the cursor also changes the visible spelling cue. These contrasts therefore establish protocol sensitivity; they are not an estimate of a tokenization-artifact fraction.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "setting",
        "subset",
        "condition",
        "samples",
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
            for subset in ("all_fixed_5000", "native_common_subset"):
                section = report["settings"][setting][subset]
                for condition, artifact in section["conditions"].items():
                    bootstrap = artifact["repository_bootstrap"]
                    for metric, estimate in bootstrap["point_estimates"].items():
                        interval = bootstrap["ci95"][metric]
                        writer.writerow(
                            {
                                "setting": setting,
                                "subset": subset,
                                "condition": condition,
                                "samples": section["samples"],
                                "metric": metric,
                                "estimate": estimate,
                                "ci95_low": interval[0] if interval else None,
                                "ci95_high": interval[1] if interval else None,
                            }
                        )
    os.replace(temporary, path)


def run(
    *,
    original_root: Path,
    control_root: Path,
    input_root: Path,
    output_dir: Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    input_audit = json.loads(
        (input_root / "cursor_input_audit.json").read_text(encoding="utf-8")
    )
    harmonization = json.loads(
        (control_root / "identical_input_harmonization.json").read_text(
            encoding="utf-8"
        )
    )
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": int(iterations),
        "base_seed": int(seed),
        "input_audit": {
            "path": str((input_root / "cursor_input_audit.json").resolve()),
            "sha256": _sha256(input_root / "cursor_input_audit.json"),
            "native_eligible": {
                language: item["native_eligible_common_occurrences"]
                for (language, item) in input_audit["languages"].items()
            },
            "native_ineligible": {
                language: item["native_ineligible_occurrences"]
                for (language, item) in input_audit["languages"].items()
            },
        },
        "identical_input_anchor": {
            "path": str(
                (control_root / "identical_input_harmonization.json").resolve()
            ),
            "sha256": _sha256(control_root / "identical_input_harmonization.json"),
            "rule": harmonization["rule"],
        },
        "settings": {},
    }
    for setting_index, setting in enumerate(SETTING_ORDER):
        language = "typescript" if setting.endswith("typescript") else "python"
        original_all_path = original_root / setting / "predictions.jsonl.gz"
        original_all = _read(original_all_path)
        start_path = (
            control_root
            / "identifier_start"
            / setting
            / "predictions_harmonized.jsonl.gz"
        )
        native_path = (
            control_root
            / "native_internal"
            / setting
            / "predictions_harmonized.jsonl.gz"
        )
        start_rows = _read(start_path)
        native_rows = _read(native_path)
        if len(start_rows) != 5000:
            raise ValueError(f"{setting}: identifier-start count is not 5,000")
        expected_native = int(
            input_audit["languages"][language]["native_eligible_common_occurrences"]
        )
        if len(native_rows) != expected_native:
            raise ValueError(f"{setting}: native common count mismatch")
        original_index = _index_unique(original_all)
        start_ids = [str(row["canonical_sample_id"]) for row in start_rows]
        native_ids = [str(row["canonical_sample_id"]) for row in native_rows]
        original_fixed = [original_index[sample_id] for sample_id in start_ids]
        start_index = _index_unique(start_rows)
        original_common = [original_index[sample_id] for sample_id in native_ids]
        start_common = [start_index[sample_id] for sample_id in native_ids]
        all_fixed_aligned = _align(
            {"original": original_fixed, "identifier_start": start_rows}, start_ids
        )
        common_aligned = _align(
            {
                "original": original_common,
                "identifier_start": start_common,
                "native_internal": native_rows,
            },
            native_ids,
        )
        setting_seed = seed + setting_index * 100000
        all_fixed_conditions = {
            condition: _condition_artifact(
                rows,
                original_all_path if condition == "original" else start_path,
                iterations=iterations,
                seed=setting_seed + condition_index,
            )
            for (condition_index, (condition, rows)) in enumerate(
                all_fixed_aligned.items()
            )
        }
        common_conditions = {
            condition: _condition_artifact(
                rows,
                (
                    original_all_path
                    if condition == "original"
                    else start_path if condition == "identifier_start" else native_path
                ),
                iterations=iterations,
                seed=setting_seed + 10 + condition_index,
            )
            for (condition_index, (condition, rows)) in enumerate(
                common_aligned.items()
            )
        }
        report["settings"][setting] = {
            "language": language,
            "all_fixed_5000": {
                "samples": 5000,
                "conditions": all_fixed_conditions,
                "paired_bootstrap": paired_condition_bootstrap(
                    all_fixed_aligned, iterations=iterations, seed=setting_seed + 50
                ),
            },
            "native_common_subset": {
                "samples": expected_native,
                "conditions": common_conditions,
                "paired_bootstrap": paired_condition_bootstrap(
                    common_aligned, iterations=iterations, seed=setting_seed + 60
                ),
            },
            "harmonization_identity_audit": {
                condition: harmonization["settings"][condition][setting][
                    "identity_audit_counts"
                ]
                for condition in ("identifier_start", "native_internal")
            },
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cursor_comparison_summary.json"
    markdown_path = output_dir / "cursor_comparison_summary.md"
    csv_path = output_dir / "cursor_comparison_metrics.csv"
    _write(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write(markdown_path, _render(report))
    _write_csv(report, csv_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    report = run(
        original_root=args.original_root.expanduser().resolve(),
        control_root=args.control_root.expanduser().resolve(),
        input_root=args.input_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        iterations=max(1, int(args.bootstrap_iterations)),
        seed=int(args.seed),
    )
    print(
        json.dumps(
            {
                setting: report["settings"][setting]["native_common_subset"][
                    "paired_bootstrap"
                ]["point_differences"]
                for setting in SETTING_ORDER
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

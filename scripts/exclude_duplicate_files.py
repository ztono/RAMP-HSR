from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
from scripts.audit_isolation import normalize_repo_id
from scripts._eval_reference_substitution_identifier_recovery import summarize_rows

PROTOCOL = "reviewer3-e6-exact-file-remediation-v1"
SETTING_ORDER = (
    "qwen_python",
    "starcoder2_python",
    "qwen_typescript",
    "starcoder2_typescript",
)


def _read_identity_set(path: Path) -> set[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _read_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend((json.loads(line) for line in handle if line.strip()))
    rows.sort(key=lambda row: (int(row["probe_index"]), str(row["occurrence_id"])))
    occurrence_ids = [str(row["occurrence_id"]) for row in rows]
    if len(occurrence_ids) != len(set(occurrence_ids)):
        raise ValueError("duplicate occurrence IDs in recovery outputs")
    return rows


def split_rows_by_source_identity(
    rows: Sequence[dict[str, Any]],
    *,
    train_repo_ids: set[str],
    train_file_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        repo = normalize_repo_id(row.get("repo_id"))
        file_hash = str(row.get("full_file_sha256") or "").lower()
        reasons: list[str] = []
        if repo and repo in train_repo_ids:
            reasons.append("repository_overlap")
        if file_hash and file_hash in train_file_hashes:
            reasons.append("exact_file_sha256_overlap")
        if reasons:
            excluded.append({**row, "e6_exclusion_reasons": reasons})
        else:
            included.append(row)
    return (included, excluded)


def _counts(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    n = len(rows)
    base = np.asarray([bool(row["base"]["exact_match"]) for row in rows])
    editor = np.asarray(
        [bool(row["mlp_first_token_then_base"]["exact_match"]) for row in rows]
    )
    return np.asarray(
        [
            n,
            int(base.sum()),
            int(editor.sum()),
            int((~base & editor).sum()),
            int((base & ~editor).sum()),
            int((~base).sum()),
            int(base.sum()),
        ],
        dtype=np.int64,
    )


def _metrics_from_counts(counts: np.ndarray) -> dict[str, float | None]:
    (n, base_ok, editor_ok, fixed, damaged, base_fail, base_pass) = (
        int(value) for value in counts
    )

    def rate(num: int, den: int) -> float | None:
        return float(num) / float(den) if den else None

    return {
        "base_id_em": rate(base_ok, n),
        "editor_id_em": rate(editor_ok, n),
        "id_error_recovery": rate(fixed, base_fail),
        "id_damage": rate(damaged, base_pass),
        "id_net": rate(fixed - damaged, n),
    }


def repository_cluster_bootstrap(
    rows: Sequence[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_repo[normalize_repo_id(row.get("repo_id"))].append(row)
    repositories = sorted(by_repo)
    matrix = np.stack([_counts(by_repo[repo]) for repo in repositories], axis=0)
    point = _metrics_from_counts(matrix.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws_by_metric: dict[str, list[float]] = {key: [] for key in point}
    remaining = int(iterations)
    while remaining > 0:
        batch = min(512, remaining)
        sampled_indices = rng.integers(
            0, len(repositories), size=(batch, len(repositories))
        )
        sampled_counts = matrix[sampled_indices].sum(axis=1)
        for count_row in sampled_counts:
            values = _metrics_from_counts(count_row)
            for key, value in values.items():
                if value is not None:
                    draws_by_metric[key].append(float(value))
        remaining -= batch
    intervals = {
        key: (
            [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            if values
            else None
        )
        for (key, values) in draws_by_metric.items()
    }
    return {
        "method": "repository-cluster paired percentile bootstrap; repositories are sampled with replacement and all their occurrences are retained",
        "iterations": int(iterations),
        "seed": int(seed),
        "repository_count": len(repositories),
        "point_estimates": point,
        "ci95": intervals,
    }


def _compact_exclusion(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "occurrence_id": str(row["occurrence_id"]),
        "probe_index": int(row["probe_index"]),
        "repo_id": str(row["repo_id"]),
        "file_path": str(row["file_path"]),
        "full_file_sha256": str(row["full_file_sha256"]),
        "target_identifier": str(row["target_identifier"]),
        "e6_exclusion_reasons": list(row["e6_exclusion_reasons"]),
    }


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


def run(*, inputs, identity_dir, output_dir, iterations, seed):
    settings = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, setting in enumerate(SETTING_ORDER):
        if setting not in inputs:
            continue
        language = "typescript" if setting.endswith("typescript") else "python"
        train_repos = _read_identity_set(
            identity_dir / f"editor_train_{language}.normalized_repo_ids.txt.gz"
        )
        train_hashes = _read_identity_set(
            identity_dir / f"editor_train_{language}.full_file_sha256.txt.gz"
        )
        original = _read_rows([Path(value) for value in inputs[setting]])
        (clean, excluded) = split_rows_by_source_identity(
            original, train_repo_ids=train_repos, train_file_hashes=train_hashes
        )
        settings[setting] = {
            "original_samples": len(original),
            "excluded_samples": len(excluded),
            "clean_samples": len(clean),
            "original_summary": summarize_rows(original),
            "clean_summary": summarize_rows(clean),
            "clean_repository_bootstrap": repository_cluster_bootstrap(
                clean, iterations=iterations, seed=seed + index
            ),
        }
        _write_jsonl(output_dir / f"{setting}.clean.jsonl", clean)
        _write_jsonl(
            output_dir / f"{setting}.excluded_metadata.jsonl",
            (_compact_exclusion(row) for row in excluded),
        )
    report = {
        "protocol": PROTOCOL,
        "settings": settings,
        "selection_independent_of_outputs": True,
    }
    _write_json(output_dir / "clean_summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply output-independent repository/file duplicate exclusions."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="JSON mapping setting names to prediction-shard paths.",
    )
    parser.add_argument("--identity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    run(
        inputs=json.loads(args.inputs.read_text()),
        identity_dir=args.identity_dir,
        output_dir=args.output_dir,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

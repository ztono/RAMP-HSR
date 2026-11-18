from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata
from urllib.parse import unquote, urlparse
from typing import Any, Iterable, Sequence

HEX_SHA256 = re.compile("^[0-9a-fA-F]{64}$")
JSON_STRING_FIELD = {
    field_name: re.compile(
        b'"' + field_name.encode("ascii") + b'"\\s*:\\s*("(?:\\\\.|[^"\\\\])*")'
    )
    for field_name in (
        "repo_id",
        "file_path",
        "full_file_sha256",
        "occurrence_id",
        "canonical_sample_id",
    )
}


def normalize_repo_id(value: Any) -> str:
    """Return a stable, GitHub-compatible repository identity.

    Dataset identifiers normally use ``owner__repository``.  The additional
    URL/path handling prevents aliases such as a Git URL and a trailing
    ``.git`` spelling from hiding an overlap.  Case-folding is appropriate for
    GitHub owner/repository names, which are case-insensitive for identity.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return ""
    text = unquote(text).replace("\\", "/")
    if text.startswith("git@") and ":" in text:
        text = text.split(":", 1)[1]
    elif "://" in text:
        parsed = urlparse(text)
        text = parsed.path
    text = re.sub("/+", "/", text).strip("/")
    if text.lower().endswith(".git"):
        text = text[:-4]
    parts = [part for part in text.split("/") if part and part != "."]
    if len(parts) >= 2 and "__" not in text:
        text = f"{parts[-2]}__{parts[-1]}"
    elif parts:
        text = parts[-1]
    text = re.sub("\\s+", "", text)
    return text.casefold()


def _json_string_from_line(raw: bytes, field_name: str) -> str | None:
    match = JSON_STRING_FIELD[field_name].search(raw)
    if match is None:
        return None
    return str(json.loads(match.group(1)))


def _open_binary(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _path_split(path: str) -> str:
    normalized = str(path or "").replace("\\", "/").casefold()
    dataset_match = re.search("/(train|test|eval)/(?:python|typescript)/", normalized)
    if dataset_match:
        return "train" if dataset_match.group(1) == "train" else "test"
    return "unknown"


def _repo_relative_file_key(repo_id: str, file_path: str) -> str:
    """Build a path key resilient to different dataset mount prefixes."""
    repo = normalize_repo_id(repo_id)
    parts = list(PurePosixPath(str(file_path).replace("\\", "/")).parts)
    index = next(
        (idx for (idx, part) in enumerate(parts) if normalize_repo_id(part) == repo),
        None,
    )
    relative = "/".join(parts[index + 1 :]) if index is not None else parts[-1]
    return f"{repo}/{relative}"


@dataclass
class ArtifactSet:
    artifact_id: str
    source_kind: str
    scope: str = "exact_rows"
    paths: list[str] = field(default_factory=list)
    row_count: int = 0
    raw_repo_ids: set[str] = field(default_factory=set)
    repo_ids: set[str] = field(default_factory=set)
    file_hashes: set[str] = field(default_factory=set)
    file_paths: set[str] = field(default_factory=set)
    relative_file_keys: set[str] = field(default_factory=set)
    file_hash_by_relative_key: dict[str, str] = field(default_factory=dict)
    file_hash_row_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "test": 0, "unknown": 0}
    )
    notes: list[str] = field(default_factory=list)

    def merge(self, other: "ArtifactSet") -> None:
        self.paths.extend(other.paths)
        self.row_count += other.row_count
        self.raw_repo_ids.update(other.raw_repo_ids)
        self.repo_ids.update(other.repo_ids)
        self.file_hashes.update(other.file_hashes)
        self.file_paths.update(other.file_paths)
        self.relative_file_keys.update(other.relative_file_keys)
        for key, value in other.file_hash_by_relative_key.items():
            previous = self.file_hash_by_relative_key.setdefault(key, value)
            if previous != value:
                raise ValueError(
                    f"multiple source hashes for normalized path {key}: {previous} vs {value}"
                )
        for key, value in other.file_hash_row_counts.items():
            self.file_hash_row_counts[key] = self.file_hash_row_counts.get(
                key, 0
            ) + int(value)
        for key, value in other.split_counts.items():
            self.split_counts[key] = self.split_counts.get(key, 0) + int(value)
        self.notes.extend(other.notes)

    def inventory(self) -> dict[str, Any]:
        normalized_collisions: dict[str, set[str]] = {}
        for raw in self.raw_repo_ids:
            normalized_collisions.setdefault(normalize_repo_id(raw), set()).add(raw)
        collisions = {
            key: sorted(values)
            for (key, values) in normalized_collisions.items()
            if key and len(values) > 1
        }
        return {
            "artifact_id": self.artifact_id,
            "source_kind": self.source_kind,
            "scope": self.scope,
            "paths": sorted(set(self.paths)),
            "rows": self.row_count,
            "raw_repository_ids": len(self.raw_repo_ids),
            "normalized_repositories": len(self.repo_ids),
            "source_file_hashes": len(self.file_hashes),
            "rows_with_source_file_hash": sum(self.file_hash_row_counts.values()),
            "source_file_paths": len(self.file_paths),
            "split_row_counts": dict(sorted(self.split_counts.items())),
            "normalization_collision_count": len(collisions),
            "normalization_collision_examples": [
                {"normalized": key, "raw": values}
                for (key, values) in sorted(collisions.items())[:20]
            ],
            "repository_set_sha256": _set_digest(self.repo_ids),
            "file_hash_set_sha256": _set_digest(self.file_hashes),
            "notes": self.notes,
        }


def _set_digest(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scan_minimal_file(path_text: str) -> ArtifactSet:
    path = Path(path_text)
    artifact = ArtifactSet(artifact_id=path.name, source_kind="row_manifest")
    artifact.paths.append(str(path.resolve()))
    with _open_binary(path) as handle:
        for raw in handle:
            if not raw.strip():
                continue
            artifact.row_count += 1
            repo_raw = _json_string_from_line(raw, "repo_id") or ""
            repo = normalize_repo_id(repo_raw)
            if repo_raw:
                artifact.raw_repo_ids.add(repo_raw)
            if repo:
                artifact.repo_ids.add(repo)
            file_path = _json_string_from_line(raw, "file_path") or ""
            if file_path:
                artifact.file_paths.add(file_path)
                artifact.relative_file_keys.add(
                    _repo_relative_file_key(repo_raw, file_path)
                )
                artifact.split_counts[_path_split(file_path)] += 1
            else:
                artifact.split_counts["unknown"] += 1
            file_hash = (_json_string_from_line(raw, "full_file_sha256") or "").lower()
            if HEX_SHA256.fullmatch(file_hash):
                artifact.file_hashes.add(file_hash)
                artifact.file_hash_row_counts[file_hash] = (
                    artifact.file_hash_row_counts.get(file_hash, 0) + 1
                )
                if file_path:
                    key = _repo_relative_file_key(repo_raw, file_path)
                    previous = artifact.file_hash_by_relative_key.setdefault(
                        key, file_hash
                    )
                    if previous != file_hash:
                        raise ValueError(
                            f"multiple source hashes for normalized path {key}: {previous} vs {file_hash}"
                        )
    return artifact


def scan_paths(
    artifact_id: str,
    source_kind: str,
    paths: Sequence[Path],
    *,
    scope: str = "exact_rows",
    workers: int = 1,
) -> ArtifactSet:
    resolved = [path.expanduser().resolve() for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{artifact_id} misses files: {missing[:5]}")
    result = ArtifactSet(artifact_id=artifact_id, source_kind=source_kind, scope=scope)
    if workers > 1 and len(resolved) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(resolved))) as pool:
            for partial in pool.map(_scan_minimal_file, map(str, resolved)):
                result.merge(partial)
    else:
        for path in resolved:
            result.merge(_scan_minimal_file(str(path)))
    result.paths = sorted(set(result.paths))
    return result


def compare_sets(train: ArtifactSet, test: ArtifactSet) -> dict[str, Any]:
    shared_repos = train.repo_ids & test.repo_ids
    shared_hashes = train.file_hashes & test.file_hashes
    shared_paths = train.relative_file_keys & test.relative_file_keys
    return {
        "train_artifact": train.artifact_id,
        "test_artifact": test.artifact_id,
        "train_scope": train.scope,
        "train_repositories": len(train.repo_ids),
        "test_repositories": len(test.repo_ids),
        "shared_repositories": len(shared_repos),
        "shared_repository_examples": sorted(shared_repos)[:20],
        "train_file_hashes": len(train.file_hashes),
        "test_file_hashes": len(test.file_hashes),
        "identical_files": len(shared_hashes),
        "test_rows_in_identical_files": sum(
            (
                int(test.file_hash_row_counts.get(file_hash, 0))
                for file_hash in shared_hashes
            )
        ),
        "identical_file_sha256_examples": sorted(shared_hashes)[:20],
        "same_repo_relative_paths": len(shared_paths),
        "same_repo_relative_path_examples": sorted(shared_paths)[:20],
        "repository_disjoint": not shared_repos,
        "exact_file_disjoint": not shared_hashes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit repository and exact-file isolation from source-free metadata."
    )
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    train = scan_paths("train", "editor_train", args.train, workers=args.workers)
    test = scan_paths("test", "evaluation", args.test, workers=args.workers)
    comparison = compare_sets(train, test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "train": train.inventory(),
                "test": test.inventory(),
                "comparison": comparison,
            },
            indent=2,
        )
        + "\n"
    )
    return (
        0
        if comparison["repository_disjoint"] and comparison["exact_file_disjoint"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Reconstruct source-bearing samples locally from pinned upstream commits."""

from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from scripts.evaluate_operating_points import read_records


def reconstruct(row, repository, clones):
    commit = repository.get("commit_sha")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError(
            f"Missing verified commit SHA for {row['repo_id']}; no HEAD fallback is allowed"
        )
    relative = PurePosixPath(row["file_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source path must stay inside the upstream repository")
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(clones / row["repo_id"]),
            "show",
            f"{commit}:{relative.as_posix()}",
        ]
    )
    # Match Path.read_text(encoding='utf-8', errors='replace') in the sampler.
    source = (
        raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    )
    if hashlib.sha256(source.encode()).hexdigest() != row["full_file_sha256"]:
        raise ValueError("upstream source does not match the archived sample hash")
    start = int(row["identifier_start_absolute_position"])
    cursor = int(row["original_cursor_absolute_position"])
    end = start + int(row["identifier_length"])
    if not 0 <= start <= cursor < end <= len(source):
        raise ValueError("identifier/cursor offsets fall outside the source")
    prefix = source[:cursor]
    if (
        "prefix_sha256" in row
        and hashlib.sha256(prefix.encode()).hexdigest() != row["prefix_sha256"]
    ):
        raise ValueError("reconstructed prefix differs from the archived occurrence")
    identifier = source[start:end]
    return {
        **row,
        "file_path": str(clones / row["repo_id"] / relative.as_posix()),
        "prefix": prefix,
        "target": source[cursor:end],
        "target_identifier": identifier,
        "target_original": identifier,
        "target_original_absolute_position": start,
        "absolute_position": cursor,
        "existing_identifier_prefix": source[start:cursor],
        "remaining_identifier_suffix": source[cursor:end],
        "full_file_content": source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repositories", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--files",
        type=Path,
        help="File dictionary for compact full-pool metadata (data/files.jsonl.gz).",
    )
    parser.add_argument("--clones", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Local-only JSONL; reconstructed source must stay outside a release.",
    )
    args = parser.parse_args()
    repositories = {row["repo_id"]: row for row in read_records(args.repositories)}
    file_dictionary = (
        {row["file_id"]: row for row in read_records(args.files) if "file_id" in row}
        if args.files
        else {}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    opener = gzip.open if args.output.name.endswith(".gz") else open
    try:
        with opener(temporary, "wt", encoding="utf-8") as out:
            for row in read_records(args.samples):
                if "file_id" in row and "repo_id" not in row:
                    if row["file_id"] not in file_dictionary:
                        raise ValueError(
                            "compact metadata requires its complete --files dictionary"
                        )
                    row = {**file_dictionary[row["file_id"]], **row}
                    row["original_cursor_absolute_position"] = row["absolute_position"]
                    row["identifier_start_absolute_position"] = row[
                        "target_original_absolute_position"
                    ]
                repository = repositories[row["repo_id"]]
                out.write(
                    json.dumps(
                        reconstruct(row, repository, args.clones), ensure_ascii=False
                    )
                    + "\n"
                )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

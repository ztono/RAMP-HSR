from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object of the ensured directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl_streaming(filepath: str | Path):
    """Yield dictionaries from a JSONL file one at a time.

    Args:
        filepath: Input file path.

    Yields:
        Parsed JSON dictionaries.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def discover_sample_files(path: str | Path) -> list[Path]:
    """Return a sorted list of repo-level sample JSONL files.

    Accepts either:
      - A directory containing ``repos/*.jsonl`` (new layout)
      - A single JSONL file (legacy layout)

    Args:
        path: Directory or file path.

    Returns:
        Sorted list of Path objects.
    """
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        repos_dir = p / "repos"
        search_dir = repos_dir if repos_dir.is_dir() else p
        files = sorted(
            (f for f in search_dir.glob("*.jsonl") if not f.name.endswith(".fc.jsonl"))
        )
        if files:
            return files
    raise FileNotFoundError(f"No sample files found at {path}")


def write_json(data: Any, filepath: str | Path, indent: int | None = 2) -> None:
    """Write data to a JSON file.

    Args:
        data: Data to write (must be JSON serializable).
        filepath: Output file path.
        indent: JSON indentation level.
    """
    p = Path(filepath)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


_SHA256_RE = __import__("re").compile("^[0-9a-f]{64}$")


def is_sha256(s: str) -> bool:
    return bool(_SHA256_RE.match(s))


def load_file_content_map(sample_files: list[Path]) -> dict[str, str]:
    fc_map: dict[str, str] = {}
    for sf in sample_files:
        fc_path = Path(str(sf) + ".fc.jsonl")
        if not fc_path.exists():
            continue
        with open(fc_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sha = rec.get("sha256")
                content = rec.get("content")
                if sha and content is not None:
                    fc_map[sha] = content
    return fc_map


def resolve_full_file_content(
    sample: dict[str, Any], fc_map: dict[str, str] | None = None
) -> str | None:
    raw = sample.get("full_file_content")
    if raw is None:
        return None
    if isinstance(raw, str) and is_sha256(raw):
        if fc_map is None:
            return None
        return fc_map.get(raw)
    return raw

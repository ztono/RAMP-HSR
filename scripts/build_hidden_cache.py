from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterator
import numpy as np

FORMAT_NAME = "repair_hidden_memmap_v1"
FORMAT_VERSION = 1
ARRAY_SPECS = {
    "h_last": ("h_last.f32", np.dtype("<f4")),
    "target_token_id": ("target_token_id.int64", np.dtype("<i8")),
    "base_top1_token_id": ("base_top1_token_id.int64", np.dtype("<i8")),
    "base_correct": ("base_correct.uint8", np.dtype("u1")),
    "risk_active": ("risk_active.uint8", np.dtype("u1")),
    "low_risk": ("low_risk.uint8", np.dtype("u1")),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _model_provenance(model_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"model_name": model_name}
    model_path = Path(model_name)
    if not model_path.is_dir():
        return payload
    payload["resolved_path"] = str(model_path.resolve())
    config_path = model_path / "config.json"
    if config_path.is_file():
        payload["config"] = {
            **_fingerprint(config_path),
            "sha256": _sha256(config_path),
        }
    weight_metadata = (
        model_path
        / ".cache"
        / "huggingface"
        / "download"
        / "model.safetensors.metadata"
    )
    if weight_metadata.is_file():
        lines = weight_metadata.read_text(encoding="utf-8").splitlines()
        if lines:
            payload["revision"] = lines[0]
        if len(lines) > 1:
            payload["weight_sha256"] = lines[1]
    return payload


def _iter_nonempty_lines(handle) -> Iterator[str]:
    for line in handle:
        if line.strip():
            yield line


def _manifest_source_segments(
    manifest_path: Path,
) -> tuple[list[tuple[Path, int]], int, dict[str, Any]]:
    """Resolve archived raw shards while preserving their concatenated order."""
    resolved_manifest = manifest_path.resolve()
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError(f"source manifest has no shards: {resolved_manifest}")
    segments: list[tuple[Path, int]] = []
    provenance_shards = []
    total_rows = 0
    for index, item in enumerate(raw_shards):
        if not isinstance(item, dict):
            raise ValueError(f"invalid source shard {index}: {resolved_manifest}")
        path = Path(str(item.get("path") or ""))
        if not path.is_absolute():
            path = resolved_manifest.parent / path
        path = path.resolve()
        rows = int(item.get("samples", -1))
        if rows < 0 or not path.is_file():
            raise ValueError(f"invalid source shard {index}: path={path} rows={rows}")
        segments.append((path, rows))
        total_rows += rows
        provenance_shards.append(
            {"index": index, "rows": rows, "file": _fingerprint(path)}
        )
    declared_total = payload.get("emitted_samples")
    if declared_total is not None and int(declared_total) != total_rows:
        raise ValueError(
            f"source manifest total mismatch: {declared_total} != {total_rows}"
        )
    provenance = {
        "kind": "ordered_raw_shard_manifest_range",
        "manifest": _fingerprint(resolved_manifest),
        "manifest_emitted_samples": (
            int(declared_total) if declared_total is not None else None
        ),
        "shards": provenance_shards,
    }
    return (segments, total_rows, provenance)


def _iter_manifest_range(
    segments: list[tuple[Path, int]], *, start: int, count: int
) -> Iterator[str]:
    """Yield one contiguous range of the manifest's concatenated JSONL rows."""
    rows_to_skip = int(start)
    rows_remaining = int(count)
    for path, declared_rows in segments:
        if rows_remaining <= 0:
            break
        if rows_to_skip >= declared_rows:
            rows_to_skip -= declared_rows
            continue
        take = min(declared_rows - rows_to_skip, rows_remaining)
        observed = 0
        with path.open("r", encoding="utf-8") as handle:
            lines = _iter_nonempty_lines(handle)
            for _ in range(rows_to_skip):
                if next(lines, None) is None:
                    raise ValueError(
                        f"source shard ended before declared offset: {path}"
                    )
            for _ in range(take):
                line = next(lines, None)
                if line is None:
                    raise ValueError(
                        f"source shard ended before {declared_rows} rows: {path}"
                    )
                observed += 1
                yield line
        if observed != take:
            raise ValueError(f"source range underflow in {path}: {observed} != {take}")
        rows_remaining -= take
        rows_to_skip = 0
    if rows_remaining != 0:
        raise ValueError(
            f"source manifest range underflow: {rows_remaining} rows were unavailable"
        )


def _target_first_token_ids(tokenizer: Any, targets: list[str]) -> np.ndarray:
    encoded = tokenizer(targets, add_special_tokens=False, padding=False)
    token_rows = getattr(encoded, "input_ids", None)
    if token_rows is None and isinstance(encoded, dict):
        token_rows = encoded.get("input_ids")
    if token_rows is None or len(token_rows) != len(targets):
        raise ValueError("Tokenizer did not return one target-token row per sample")
    first_ids = []
    for target, token_ids in zip(targets, token_rows):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if not token_ids:
            raise ValueError(f"Target tokenizes to an empty sequence: {target!r}")
        first_ids.append(int(token_ids[0]))
    return np.asarray(first_ids, dtype=np.dtype("<i8"))


def _array_shapes(rows: int, hidden_dim: int) -> dict[str, tuple[int, ...]]:
    return {
        "h_last": (rows, hidden_dim),
        "target_token_id": (rows,),
        "base_top1_token_id": (rows,),
        "base_correct": (rows,),
        "risk_active": (rows,),
        "low_risk": (rows,),
    }


def _open_arrays(
    output_dir: Path, *, rows: int, hidden_dim: int, create: bool
) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    shapes = _array_shapes(rows, hidden_dim)
    for key, (filename, dtype) in ARRAY_SPECS.items():
        path = output_dir / filename
        mode = "w+" if create else "r+"
        arrays[key] = np.memmap(
            path, dtype=dtype, mode=mode, shape=shapes[key], order="C"
        )
    return arrays


def _flush_arrays(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def _expected_file_sizes(rows: int, hidden_dim: int) -> dict[str, int]:
    shapes = _array_shapes(rows, hidden_dim)
    return {
        key: int(np.prod(shapes[key], dtype=np.int64)) * int(dtype.itemsize)
        for (key, (_filename, dtype)) in ARRAY_SPECS.items()
    }


def _validate_files(output_dir: Path, *, rows: int, hidden_dim: int) -> None:
    expected = _expected_file_sizes(rows, hidden_dim)
    for key, (filename, _dtype) in ARRAY_SPECS.items():
        path = output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing packed array: {path}")
        actual_size = int(path.stat().st_size)
        if actual_size != expected[key]:
            raise ValueError(
                f"Packed array size mismatch for {path}: expected={expected[key]} actual={actual_size}"
            )


def extract_shard(args: argparse.Namespace) -> None:
    from ramp_hsr.runtime import QwenCoderLLM

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.json"
    expected_rows = int(args.expected_rows)
    source_start_row = int(args.source_start_row)
    if source_start_row < 0:
        raise ValueError("--source-start-row must be non-negative")
    input_file: Path | None = None
    source_segments: list[tuple[Path, int]] | None = None
    if args.input_manifest is not None:
        (source_segments, source_total_rows, source_fingerprint) = (
            _manifest_source_segments(args.input_manifest)
        )
        if source_start_row + expected_rows > source_total_rows:
            raise ValueError(
                f"requested source range exceeds manifest: start={source_start_row} rows={expected_rows} total={source_total_rows}"
            )
        source_fingerprint["source_start_row"] = source_start_row
        source_fingerprint["source_rows"] = expected_rows
    else:
        input_file = args.input_file.resolve()
        if source_start_row != 0:
            raise ValueError(
                "--source-start-row is supported only with --input-manifest"
            )
        source_fingerprint = _fingerprint(input_file)
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_files(
            output_dir,
            rows=int(metadata["rows"]),
            hidden_dim=int(metadata["hidden_dim"]),
        )
        if args.skip_existing:
            print(
                f"packed shard already complete: {output_dir} rows={metadata['rows']}",
                flush=True,
            )
            return
        raise FileExistsError(f"Packed shard is already complete: {output_dir}")
    print(
        f"loading model={args.base_model} device={args.device} shard={args.shard_index}",
        flush=True,
    )
    llm = QwenCoderLLM(model_name=args.base_model, device=args.device)
    hidden_dim = int(llm.model.config.hidden_size)
    extraction_contract = {
        "source": source_fingerprint,
        "expected_rows": expected_rows,
        "hidden_dim": hidden_dim,
        "base_model": _model_provenance(args.base_model),
        "language": args.language,
        "split": args.split,
        "shard_index": int(args.shard_index),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "tokenizer_padding_side": str(llm.tokenizer.padding_side),
        "tokenizer_truncation_side": str(llm.tokenizer.truncation_side),
        "hidden_state_interface": "post_norm",
        "feature_mode": "hidden_only",
    }
    rows_written = 0
    create_arrays = True
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("contract") != extraction_contract:
            raise ValueError(
                f"Cannot resume packed shard with a different contract: {output_dir}"
            )
        rows_written = int(progress.get("rows_written", 0))
        create_arrays = False
    elif any(((output_dir / spec[0]).exists() for spec in ARRAY_SPECS.values())):
        raise FileExistsError(
            f"Packed arrays exist without resumable progress metadata: {output_dir}"
        )
    arrays = _open_arrays(
        output_dir, rows=expected_rows, hidden_dim=hidden_dim, create=create_arrays
    )
    if create_arrays:
        _write_json_atomic(
            progress_path,
            {
                "format": FORMAT_NAME,
                "version": FORMAT_VERSION,
                "created_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "status": "in_progress",
                "rows_written": 0,
                "contract": extraction_contract,
            },
        )
    started = time.monotonic()
    interval_started = started
    interval_rows = rows_written
    batches_since_checkpoint = 0
    base_correct_count = int(
        np.asarray(arrays["base_correct"][:rows_written], dtype=np.int64).sum()
    )
    direct_handle = None
    if source_segments is not None:
        line_iterator = _iter_manifest_range(
            source_segments,
            start=source_start_row + rows_written,
            count=expected_rows - rows_written,
        )
    else:
        assert input_file is not None
        direct_handle = input_file.open("r", encoding="utf-8")
        line_iterator = _iter_nonempty_lines(direct_handle)
        for _ in range(rows_written):
            try:
                next(line_iterator)
            except StopIteration as exc:
                raise ValueError(
                    f"Source ended before resume row {rows_written}: {input_file}"
                ) from exc
    try:
        while rows_written < expected_rows:
            raw_batch = list(islice(line_iterator, int(args.batch_size)))
            if not raw_batch:
                raise ValueError(
                    f"Source ended early: expected={expected_rows} actual={rows_written}"
                )
            rows = [json.loads(line) for line in raw_batch]
            prefixes = [str(row.get("prefix") or "") for row in rows]
            targets = [str(row.get("target") or "") for row in rows]
            target_ids = _target_first_token_ids(llm.tokenizer, targets)
            extracted = llm.extract_hidden_repair_batch(
                prefixes, max_length=int(args.max_length)
            )
            h_last = extracted["h_last"].numpy()
            base_top1 = (
                extracted["base_top1_token_id"]
                .numpy()
                .astype(np.dtype("<i8"), copy=False)
            )
            batch_rows = len(rows)
            if h_last.shape != (batch_rows, hidden_dim):
                raise ValueError(
                    f"Hidden batch shape mismatch: expected={(batch_rows, hidden_dim)} actual={tuple(h_last.shape)}"
                )
            if base_top1.shape != (batch_rows,):
                raise ValueError(
                    f"Top-1 batch shape mismatch: expected={(batch_rows,)} actual={tuple(base_top1.shape)}"
                )
            end = rows_written + batch_rows
            if end > expected_rows:
                raise ValueError(
                    f"Source has more than expected {expected_rows} rows: {input_file}"
                )
            correct = (base_top1 == target_ids).astype(np.uint8, copy=False)
            arrays["h_last"][rows_written:end] = h_last
            arrays["target_token_id"][rows_written:end] = target_ids
            arrays["base_top1_token_id"][rows_written:end] = base_top1
            arrays["base_correct"][rows_written:end] = correct
            arrays["risk_active"][rows_written:end] = 0
            arrays["low_risk"][rows_written:end] = 0
            base_correct_count += int(correct.sum())
            rows_written = end
            batches_since_checkpoint += 1
            now = time.monotonic()
            should_checkpoint = (
                batches_since_checkpoint >= int(args.checkpoint_every_batches)
                or rows_written == expected_rows
            )
            if should_checkpoint:
                _flush_arrays(arrays)
                _write_json_atomic(
                    progress_path,
                    {
                        "format": FORMAT_NAME,
                        "version": FORMAT_VERSION,
                        "created_at_utc": json.loads(
                            progress_path.read_text(encoding="utf-8")
                        ).get("created_at_utc"),
                        "updated_at_utc": _utc_now(),
                        "status": "in_progress",
                        "rows_written": rows_written,
                        "base_correct_count": base_correct_count,
                        "contract": extraction_contract,
                    },
                )
                batches_since_checkpoint = 0
            if now - interval_started >= float(args.progress_interval_sec):
                rate = (rows_written - interval_rows) / max(
                    now - interval_started, 1e-09
                )
                remaining = expected_rows - rows_written
                eta = remaining / rate if rate > 0 else float("inf")
                print(
                    f"shard={args.shard_index} rows={rows_written}/{expected_rows} rate={rate:.1f}/s eta_sec={eta:.0f} base_correct={base_correct_count}",
                    flush=True,
                )
                interval_started = now
                interval_rows = rows_written
        if source_segments is None:
            extra_line = next(line_iterator, None)
            if extra_line is not None:
                raise ValueError(
                    f"Source has more than expected {expected_rows} rows: {input_file}"
                )
    finally:
        if direct_handle is not None:
            direct_handle.close()
    _flush_arrays(arrays)
    arrays.clear()
    _validate_files(output_dir, rows=expected_rows, hidden_dim=hidden_dim)
    file_metadata = {}
    for key, (filename, _dtype) in ARRAY_SPECS.items():
        path = output_dir / filename
        file_metadata[key] = {"path": filename, "bytes": int(path.stat().st_size)}
    elapsed = time.monotonic() - started
    metadata = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "created_at_utc": _utc_now(),
        "status": "complete",
        "rows": expected_rows,
        "hidden_dim": hidden_dim,
        "hidden_dtype": "float32",
        "source_manifest": source_fingerprint,
        "files": file_metadata,
        "base_correct_count": base_correct_count,
        "base_wrong_count": expected_rows - base_correct_count,
        "elapsed_sec": elapsed,
        "samples_per_second": expected_rows / elapsed if elapsed > 0 else 0.0,
        "risk_flag_semantics": "unavailable_for_hidden_only_cache; arrays_are_zero",
        "contract": extraction_contract,
    }
    _write_json_atomic(metadata_path, metadata)
    _write_json_atomic(
        progress_path,
        {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "created_at_utc": metadata["created_at_utc"],
            "updated_at_utc": _utc_now(),
            "status": "complete",
            "rows_written": expected_rows,
            "base_correct_count": base_correct_count,
            "contract": extraction_contract,
        },
    )
    print(json.dumps(metadata, indent=2), flush=True)


def finalize_root(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    root_metadata_path = output_dir / "metadata.json"
    if root_metadata_path.is_file() and args.skip_existing:
        print(root_metadata_path.read_text(encoding="utf-8"), end="")
        return
    shard_metadata = []
    total_rows = 0
    hidden_dims: set[int] = set()
    batch_sizes: set[int] = set()
    hidden_state_interfaces: set[str] = set()
    feature_modes: set[str] = set()
    contract_model_fingerprints: set[str] = set()
    base_correct_count = 0
    for shard_index in range(int(args.num_shards)):
        shard_dir = output_dir / f"shard_{shard_index}"
        metadata_path = shard_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing complete shard metadata: {metadata_path}")
        item = json.loads(metadata_path.read_text(encoding="utf-8"))
        if item.get("status") != "complete":
            raise ValueError(f"Packed shard is not complete: {metadata_path}")
        contract = item.get("contract") or {}
        expected_contract_values = {
            "language": args.language,
            "split": args.split,
            "max_length": int(args.max_length),
            "shard_index": shard_index,
        }
        contract_mismatches = {
            key: {"expected": expected, "actual": contract.get(key)}
            for (key, expected) in expected_contract_values.items()
            if contract.get(key) != expected
        }
        if contract_mismatches:
            raise ValueError(
                f"Packed shard contract mismatch in {metadata_path}: {contract_mismatches}"
            )
        batch_sizes.add(int(contract.get("batch_size", -1)))
        hidden_state_interfaces.add(str(contract.get("hidden_state_interface")))
        feature_modes.add(str(contract.get("feature_mode")))
        contract_model_fingerprints.add(
            json.dumps(contract.get("base_model"), sort_keys=True)
        )
        rows = int(item["rows"])
        hidden_dim = int(item["hidden_dim"])
        _validate_files(shard_dir, rows=rows, hidden_dim=hidden_dim)
        total_rows += rows
        hidden_dims.add(hidden_dim)
        base_correct_count += int(item.get("base_correct_count", 0))
        files = {
            key: {
                "path": str(Path(f"shard_{shard_index}") / value["path"]),
                "bytes": int(value["bytes"]),
            }
            for (key, value) in item["files"].items()
        }
        shard_metadata.append(
            {
                "rows": rows,
                "hidden_dim": hidden_dim,
                "hidden_dtype": "float32",
                "source_manifest": item["source_manifest"],
                "files": files,
                "base_correct_count": int(item.get("base_correct_count", 0)),
                "base_wrong_count": int(item.get("base_wrong_count", 0)),
                "elapsed_sec": float(item.get("elapsed_sec", 0.0)),
                "samples_per_second": float(item.get("samples_per_second", 0.0)),
            }
        )
    if total_rows != int(args.expected_total_rows):
        raise ValueError(
            f"Packed root row mismatch: expected={args.expected_total_rows} actual={total_rows}"
        )
    if len(hidden_dims) != 1:
        raise ValueError(f"Packed shards disagree on hidden width: {hidden_dims}")
    if len(batch_sizes) != 1 or min(batch_sizes) <= 0:
        raise ValueError(
            f"Packed shards disagree on extraction batch size: {batch_sizes}"
        )
    if hidden_state_interfaces != {"post_norm"}:
        raise ValueError(
            f"Packed shards disagree on hidden-state interface: {hidden_state_interfaces}"
        )
    if feature_modes != {"hidden_only"}:
        raise ValueError(f"Packed shards are not hidden-only: {feature_modes}")
    if len(contract_model_fingerprints) != 1:
        raise ValueError("Packed shards disagree on base-model provenance")
    stub_manifest = output_dir / "manifest.placeholder.jsonl"
    if not stub_manifest.exists():
        stub_manifest.touch()
    _write_json_atomic(
        stub_manifest.with_suffix(stub_manifest.suffix + ".summary.json"),
        {
            "rows": total_rows,
            "note": "Packed-only placeholder; sample rows live in the memmap arrays.",
        },
    )
    metadata = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "rows": total_rows,
        "hidden_dim": hidden_dims.pop(),
        "hidden_dtype": "float32",
        "feature_mode": "hidden_only",
        "hidden_state_interface": hidden_state_interfaces.pop(),
        "combined_manifest": None,
        "manifest_stub": str(stub_manifest.resolve()),
        "language": args.language,
        "split": args.split,
        "base_model": _model_provenance(args.base_model),
        "max_length": int(args.max_length),
        "extraction_batch_size": batch_sizes.pop(),
        "num_shards": int(args.num_shards),
        "base_correct_count": base_correct_count,
        "base_wrong_count": total_rows - base_correct_count,
        "risk_flag_semantics": "unavailable_for_hidden_only_cache; arrays_are_zero",
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "shards": shard_metadata,
    }
    _write_json_atomic(root_metadata_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract-shard")
    source = extract.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-file", type=Path)
    source.add_argument(
        "--input-manifest",
        type=Path,
        help="Raw-shard summary whose ordered shards form one logical source. Use with --source-start-row to create balanced compute ranges without copying or reordering samples.",
    )
    extract.add_argument("--source-start-row", type=int, default=0)
    extract.add_argument("--output-dir", required=True, type=Path)
    extract.add_argument("--expected-rows", required=True, type=int)
    extract.add_argument("--base-model", required=True)
    extract.add_argument("--language", required=True)
    extract.add_argument("--split", required=True)
    extract.add_argument("--shard-index", required=True, type=int)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--batch-size", type=int, default=32)
    extract.add_argument("--max-length", type=int, default=512)
    extract.add_argument("--checkpoint-every-batches", type=int, default=20)
    extract.add_argument("--progress-interval-sec", type=float, default=60.0)
    extract.add_argument("--skip-existing", action="store_true")
    extract.set_defaults(func=extract_shard)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-dir", required=True, type=Path)
    finalize.add_argument("--expected-total-rows", required=True, type=int)
    finalize.add_argument("--num-shards", required=True, type=int)
    finalize.add_argument("--base-model", required=True)
    finalize.add_argument("--language", required=True)
    finalize.add_argument("--split", required=True)
    finalize.add_argument("--max-length", type=int, default=512)
    finalize.add_argument("--skip-existing", action="store_true")
    finalize.set_defaults(func=finalize_root)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

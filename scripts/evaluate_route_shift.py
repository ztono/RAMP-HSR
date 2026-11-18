from __future__ import annotations
import argparse
import gc
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))
GROUPS = ("all", "recovered", "not_recovered", "damaged", "preserved")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path, *, with_hash: bool = False) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if with_hash:
        result["sha256"] = _sha256(resolved)
    return result


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _empty_group() -> dict[str, float | int]:
    return {
        "count": 0,
        "sum_pre_composite": 0.0,
        "sum_post_composite": 0.0,
        "sum_delta_composite": 0.0,
        "sumsq_delta_composite": 0.0,
        "sum_pre_pair": 0.0,
        "sum_post_pair": 0.0,
        "sum_delta_pair": 0.0,
        "sumsq_delta_pair": 0.0,
    }


def _new_raw_groups() -> dict[str, dict[str, float | int]]:
    return {name: _empty_group() for name in GROUPS}


def _update_group(
    group: dict[str, float | int],
    *,
    pre_composite: float,
    post_composite: float,
    pre_pair: float,
    post_pair: float,
) -> None:
    delta_composite = float(post_composite) - float(pre_composite)
    delta_pair = float(post_pair) - float(pre_pair)
    group["count"] = int(group["count"]) + 1
    group["sum_pre_composite"] = float(group["sum_pre_composite"]) + pre_composite
    group["sum_post_composite"] = float(group["sum_post_composite"]) + post_composite
    group["sum_delta_composite"] = float(group["sum_delta_composite"]) + delta_composite
    group["sumsq_delta_composite"] = (
        float(group["sumsq_delta_composite"]) + delta_composite * delta_composite
    )
    group["sum_pre_pair"] = float(group["sum_pre_pair"]) + pre_pair
    group["sum_post_pair"] = float(group["sum_post_pair"]) + post_pair
    group["sum_delta_pair"] = float(group["sum_delta_pair"]) + delta_pair
    group["sumsq_delta_pair"] = (
        float(group["sumsq_delta_pair"]) + delta_pair * delta_pair
    )


def _finalize_group(group: dict[str, float | int]) -> dict[str, Any]:
    count = int(group.get("count", 0))

    def mean(key: str) -> float:
        return round(float(group.get(key, 0.0)) / count, 6) if count else 0.0

    mean_delta_composite = mean("sum_delta_composite")
    mean_delta_pair = mean("sum_delta_pair")
    variance_composite = (
        float(group.get("sumsq_delta_composite", 0.0)) / count
        - mean_delta_composite * mean_delta_composite
        if count
        else 0.0
    )
    variance_pair = (
        float(group.get("sumsq_delta_pair", 0.0)) / count
        - mean_delta_pair * mean_delta_pair
        if count
        else 0.0
    )
    return {
        "count": count,
        "mean_pre_composite": mean("sum_pre_composite"),
        "mean_post_composite": mean("sum_post_composite"),
        "mean_delta_composite": mean_delta_composite,
        "std_delta_composite": round(math.sqrt(max(variance_composite, 0.0)), 6),
        "mean_pre_pair": mean("sum_pre_pair"),
        "mean_post_pair": mean("sum_post_pair"),
        "mean_delta_pair": mean_delta_pair,
        "std_delta_pair": round(math.sqrt(max(variance_pair, 0.0)), 6),
    }


def _feature_float(features: dict[str, Any], name: str) -> float:
    value = features.get(name)
    if value is None:
        uncertainty = features.get("uncertainty")
        if isinstance(uncertainty, dict):
            value = uncertainty.get(name)
    if value is None:
        for prefix in ("route", "code"):
            names = features.get(f"{prefix}_feature_names")
            values = features.get(f"{prefix}_features")
            if not isinstance(names, list) or not isinstance(values, list):
                continue
            try:
                index = names.index(name)
            except ValueError:
                continue
            if index < len(values):
                value = values[index]
                break
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Source row lacks finite route feature {name!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Source row has non-finite route feature {name!r}: {result}")
    return result


def _iter_nonempty_lines(path: Path, *, skip: int, take: int) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        lines = (line for line in handle if line.strip())
        for _ in range(skip):
            if next(lines, None) is None:
                raise ValueError(f"Source ended before skip={skip}: {path}")
        yielded = 0
        for line in islice(lines, take):
            yielded += 1
            yield line
        if yielded != take:
            raise ValueError(
                f"Source underflow in {path}: expected={take} actual={yielded}"
            )


def _iter_source_lines(
    descriptor: dict[str, Any], *, skip: int, take: int
) -> Iterator[str]:
    kind = str(descriptor.get("kind") or "")
    if kind != "ordered_raw_shard_manifest_range":
        path = Path(str(descriptor["path"])).resolve()
        stat = path.stat()
        if int(descriptor.get("size", stat.st_size)) != stat.st_size:
            raise ValueError(f"Source size changed: {path}")
        yield from _iter_nonempty_lines(path, skip=skip, take=take)
        return
    rows_to_skip = int(descriptor.get("source_start_row", 0)) + int(skip)
    rows_remaining = int(take)
    for item in descriptor.get("shards") or []:
        if rows_remaining <= 0:
            break
        declared_rows = int(item["rows"])
        if rows_to_skip >= declared_rows:
            rows_to_skip -= declared_rows
            continue
        file_info = item["file"]
        path = Path(str(file_info["path"])).resolve()
        local_take = min(declared_rows - rows_to_skip, rows_remaining)
        yield from _iter_nonempty_lines(path, skip=rows_to_skip, take=local_take)
        rows_remaining -= local_take
        rows_to_skip = 0
    if rows_remaining:
        raise ValueError(f"Source range underflow: missing {rows_remaining} rows")


def _open_array(
    packed_dir: Path,
    shard: dict[str, Any],
    key: str,
    *,
    dtype: str,
    shape: tuple[int, ...],
) -> np.memmap:
    path = packed_dir / str(shard["files"][key]["path"])
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(
            f"Packed array size mismatch for {path}: expected={expected} actual={path.stat().st_size}"
        )
    return np.memmap(path, dtype=np.dtype(dtype), mode="r", shape=shape)


def _target_first_ids(tokenizer: Any, rows: list[dict[str, Any]]) -> list[int]:
    targets = [str(row.get("target") or "") for row in rows]
    encoded = tokenizer(targets, add_special_tokens=False, padding=False)
    token_rows = getattr(encoded, "input_ids", None)
    if token_rows is None and isinstance(encoded, dict):
        token_rows = encoded.get("input_ids")
    if token_rows is None or len(token_rows) != len(rows):
        raise ValueError("Tokenizer did not return one target row per source row")
    result: list[int] = []
    for target, token_ids in zip(targets, token_rows, strict=True):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if not token_ids:
            raise ValueError(f"Target tokenizes to empty sequence: {target!r}")
        result.append(int(token_ids[0]))
    return result


def _contract(args: argparse.Namespace, metadata_path: Path) -> dict[str, Any]:
    contract = {
        "version": 2,
        "packed_dataset": str(args.packed_dataset.resolve()),
        "packed_metadata": _fingerprint(metadata_path, with_hash=True),
        "checkpoint": _fingerprint(args.checkpoint, with_hash=True),
        "base_model": str(Path(args.base_model).resolve()),
        "risk_config": _fingerprint(args.risk_config, with_hash=True),
        "shard_index": int(args.shard_index),
        "pre_route_source": str(args.pre_route_source),
        "pre_max_context_tokens": int(args.pre_max_context_tokens),
        "post_max_context_tokens": int(args.post_max_context_tokens),
        "batch_size": int(args.batch_size),
        "prediction_batch_size": int(args.prediction_batch_size),
        "max_rows": int(args.max_rows),
    }
    if args.pre_route_projection is not None:
        projection_metadata = args.pre_route_projection.resolve() / "metadata.json"
        contract["pre_route_projection"] = {
            "root": str(args.pre_route_projection.resolve()),
            "metadata": _fingerprint(projection_metadata, with_hash=True),
        }
    return contract


def _open_projected_pre_route(
    *,
    projection_root: Path,
    packed_dir: Path,
    packed_rows: int,
    shard_index: int,
    shard_rows: int,
    risk_config: Path,
) -> tuple[np.memmap, np.memmap]:
    projection_root = projection_root.resolve()
    metadata_path = projection_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "packed_route_projection_v1":
        raise ValueError(
            f"Unsupported pre-route projection format: {metadata.get('format')!r}"
        )
    projected_packed = Path(str(metadata.get("packed_dataset") or "")).resolve()
    if projected_packed != packed_dir:
        raise ValueError(
            f"Pre-route projection uses a different packed dataset: {projected_packed} != {packed_dir}"
        )
    if int(metadata.get("rows", -1)) != int(packed_rows):
        raise ValueError(
            f"Pre-route projection row count mismatch: {metadata.get('rows')} != {packed_rows}"
        )
    shards = metadata.get("shards") or []
    if shard_index < 0 or shard_index >= len(shards):
        raise ValueError(
            f"Pre-route projection lacks shard {shard_index}/{len(shards)}"
        )
    shard = shards[shard_index]
    if int(shard.get("index", -1)) != shard_index:
        raise ValueError("Pre-route projection shard index mismatch")
    if int(shard.get("rows", -1)) != shard_rows:
        raise ValueError(
            f"Pre-route projection shard rows mismatch: {shard.get('rows')} != {shard_rows}"
        )
    projected_policy = (shard.get("contract") or {}).get("risk_policy") or {}
    expected_policy_sha = _sha256(risk_config.resolve())
    if projected_policy.get("sha256") != expected_policy_sha:
        raise ValueError(
            f"Pre-route projection risk policy mismatch: {projected_policy.get('sha256')} != {expected_policy_sha}"
        )
    shard_root = projection_root / str(shard["path"])
    arrays = shard.get("arrays") or {}

    def open_feature(name: str) -> np.memmap:
        descriptor = arrays.get(name) or {}
        if descriptor.get("dtype") != "float32":
            raise ValueError(
                f"Unexpected projected dtype for {name}: {descriptor.get('dtype')!r}"
            )
        path = shard_root / str(descriptor["path"])
        expected_bytes = shard_rows * np.dtype("<f4").itemsize
        if int(descriptor.get("bytes", -1)) != expected_bytes:
            raise ValueError(
                f"Projected metadata size mismatch for {name}: {descriptor.get('bytes')} != {expected_bytes}"
            )
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Projected array size mismatch for {path}: {path.stat().st_size} != {expected_bytes}"
            )
        return np.memmap(path, dtype=np.dtype("<f4"), mode="r", shape=(shard_rows,))

    return (
        open_feature("neuron_chain_composite_risk"),
        open_feature("neuron_chain_pair_risk"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--risk-config", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=0,
        help="Sub-batch size used only for editor decoding. Zero inherits --batch-size. Set this to the main table-evaluation batch size while retaining a larger route-forward batch.",
    )
    parser.add_argument(
        "--pre-route-source", choices=("manifest", "live", "projection"), required=True
    )
    parser.add_argument(
        "--pre-route-projection",
        type=Path,
        help="Exactly aligned packed_route_projection_v1 root. Required when --pre-route-source=projection and forbidden otherwise.",
    )
    parser.add_argument("--pre-max-context-tokens", type=int, default=512)
    parser.add_argument("--post-max-context-tokens", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--checkpoint-every-batches", type=int, default=25)
    args = parser.parse_args()
    if args.prediction_batch_size == 0:
        args.prediction_batch_size = int(args.batch_size)
    if (
        args.batch_size <= 0
        or args.prediction_batch_size <= 0
        or args.checkpoint_every_batches <= 0
    ):
        raise ValueError("Batch sizes and checkpoint cadence must be positive")
    if args.pre_route_source == "projection":
        if args.pre_route_projection is None:
            raise ValueError(
                "--pre-route-projection is required for --pre-route-source=projection"
            )
    elif args.pre_route_projection is not None:
        raise ValueError(
            "--pre-route-projection is only valid for --pre-route-source=projection"
        )
    os.environ["SRLCC_NEURON_RISK_CONFIG"] = str(args.risk_config.resolve())
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from scripts.evaluate_continuation import (
        _decode_single_token,
        _encode_text,
        _extract_runtime_features_from_context_ids_batch,
        _fallback_context_ids,
    )
    from ramp_hsr.checkpoints import load_editor
    from ramp_hsr.training import _load_lm_head, _logits_from_repair_hidden
    from ramp_hsr.runtime_helpers import OfflineEvaluator

    packed_dir = args.packed_dataset.resolve()
    metadata_path = packed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "repair_hidden_memmap_v1":
        raise ValueError(f"Unsupported packed format: {metadata.get('format')!r}")
    shards = metadata.get("shards") or []
    if args.shard_index < 0 or args.shard_index >= len(shards):
        raise ValueError(f"Invalid shard index {args.shard_index}/{len(shards)}")
    shard = shards[int(args.shard_index)]
    shard_rows = int(shard["rows"])
    total_rows = (
        min(shard_rows, int(args.max_rows)) if args.max_rows > 0 else shard_rows
    )
    hidden_dim = int(metadata["hidden_dim"])
    source_descriptor = shard.get("source_manifest")
    if not isinstance(source_descriptor, dict):
        raise ValueError("Packed shard lacks an aligned source_manifest descriptor")
    h_last = _open_array(
        packed_dir, shard, "h_last", dtype="<f4", shape=(shard_rows, hidden_dim)
    )
    target_ids = _open_array(
        packed_dir, shard, "target_token_id", dtype="<i8", shape=(shard_rows,)
    )
    base_correct = _open_array(
        packed_dir, shard, "base_correct", dtype="u1", shape=(shard_rows,)
    )
    projected_pre_composite: np.memmap | None = None
    projected_pre_pair: np.memmap | None = None
    if args.pre_route_source == "projection":
        (projected_pre_composite, projected_pre_pair) = _open_projected_pre_route(
            projection_root=args.pre_route_projection,
            packed_dir=packed_dir,
            packed_rows=int(metadata["rows"]),
            shard_index=int(args.shard_index),
            shard_rows=shard_rows,
            risk_config=args.risk_config,
        )
    contract = _contract(args, metadata_path)
    contract["shard_rows"] = shard_rows
    contract["evaluated_rows"] = total_rows
    contract_sha = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    progress_path = args.output.with_name(f"{args.output.name}.progress.json")
    raw_groups = _new_raw_groups()
    rows_processed = 0
    resumed = False
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("contract_sha256") != contract_sha:
            raise ValueError(f"Resume contract mismatch: {progress_path}")
        rows_processed = int(progress.get("rows_processed", 0))
        raw_groups = progress.get("raw_groups") or _new_raw_groups()
        resumed = rows_processed > 0
    if rows_processed > total_rows:
        raise ValueError(f"Invalid resume offset {rows_processed}>{total_rows}")
    (repair_moe, repair_checkpoint) = load_editor(args.checkpoint, args.device)
    if str(repair_checkpoint.get("feature_mode")) != "hidden_only":
        raise ValueError("Route-shift rerun requires a hidden-only editor checkpoint")
    if int(repair_checkpoint.get("hidden_dim", -1)) != hidden_dim:
        raise ValueError("Checkpoint and packed hidden dimensions do not match")
    hidden_state_interface = str(
        repair_checkpoint.get("hidden_state_interface", "post_norm")
    )
    (head_holder, decode_norm, decode_lm_head) = _load_lm_head(
        args.base_model, args.device
    )
    del head_holder
    gc.collect()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    source_lines = _iter_source_lines(
        source_descriptor, skip=rows_processed, take=total_rows - rows_processed
    )
    first_raw_batch = list(
        islice(source_lines, min(args.batch_size, total_rows - rows_processed))
    )
    if total_rows > rows_processed and (not first_raw_batch):
        raise ValueError("Aligned source yielded no rows")
    first_rows = [json.loads(line) for line in first_raw_batch]
    evaluator = OfflineEvaluator(
        long_sequence_rollout_model_name=args.base_model,
        long_sequence_rollout_device=args.device,
        long_sequence_rollout_batch_size=int(args.batch_size),
    )
    (llm, _) = evaluator._get_long_sequence_rollout_llm(first_rows or [{"prefix": ""}])
    tokenizer = llm.tokenizer
    pending_batches: list[list[dict[str, Any]]] = []
    if first_rows:
        pending_batches.append(first_rows)

    def remaining_batches() -> Iterator[list[dict[str, Any]]]:
        while True:
            raw = list(islice(source_lines, int(args.batch_size)))
            if not raw:
                return
            yield [json.loads(line) for line in raw]

    started = time.monotonic()
    batches_since_checkpoint = 0
    progress_bar = tqdm(
        total=total_rows,
        initial=rows_processed,
        desc=f"post-writeback shard {args.shard_index}",
        unit="rows",
        dynamic_ncols=True,
        mininterval=5.0,
    )
    from itertools import chain

    iterator = chain(pending_batches, remaining_batches())
    with torch.inference_mode():
        for rows in iterator:
            start = rows_processed
            end = start + len(rows)
            if end > total_rows:
                raise ValueError(f"Source overflow: {end}>{total_rows}")
            observed_targets = _target_first_ids(tokenizer, rows)
            expected_targets = [int(value) for value in target_ids[start:end]]
            if observed_targets != expected_targets:
                mismatch = next(
                    (
                        index
                        for (index, (left, right)) in enumerate(
                            zip(observed_targets, expected_targets, strict=True)
                        )
                        if left != right
                    )
                )
                raise ValueError(
                    f"Packed/source target mismatch at local row {start + mismatch}: source={observed_targets[mismatch]} packed={expected_targets[mismatch]}"
                )
            predictions: list[int] = []
            for prediction_start in range(start, end, int(args.prediction_batch_size)):
                prediction_end = min(
                    prediction_start + int(args.prediction_batch_size), end
                )
                h_tensor = torch.from_numpy(
                    np.asarray(
                        h_last[prediction_start:prediction_end], dtype=np.float32
                    ).copy()
                ).to(args.device)
                empty = torch.empty(
                    (prediction_end - prediction_start, 0),
                    dtype=torch.float32,
                    device=args.device,
                )
                repaired = repair_moe(h_tensor)
                logits = _logits_from_repair_hidden(
                    repaired,
                    norm=decode_norm,
                    lm_head=decode_lm_head,
                    hidden_state_interface=hidden_state_interface,
                )
                predictions.extend(
                    (
                        int(value)
                        for value in logits.argmax(dim=-1).detach().cpu().tolist()
                    )
                )
                del h_tensor, empty, repaired, logits
            prefixes = [str(row.get("prefix") or "") for row in rows]
            context_rows: list[list[int]] = []
            post_context_rows: list[list[int]] = []
            post_prefixes: list[str] = []
            for prefix, prediction in zip(prefixes, predictions, strict=True):
                context = _encode_text(tokenizer, prefix)
                if not context:
                    context = _fallback_context_ids(tokenizer)
                context_rows.append(context)
                post_context_rows.append([*context, int(prediction)])
                post_prefixes.append(
                    prefix + _decode_single_token(tokenizer, int(prediction))
                )
            if args.pre_route_source == "live":
                pre_features = _extract_runtime_features_from_context_ids_batch(
                    evaluator=evaluator,
                    samples=rows,
                    prefixes=prefixes,
                    context_id_rows=context_rows,
                    max_context_tokens=int(args.pre_max_context_tokens),
                )
            elif args.pre_route_source == "projection":
                assert projected_pre_composite is not None
                assert projected_pre_pair is not None
                pre_features = [
                    {
                        "neuron_chain_composite_risk": float(composite),
                        "neuron_chain_pair_risk": float(pair),
                    }
                    for (composite, pair) in zip(
                        projected_pre_composite[start:end],
                        projected_pre_pair[start:end],
                        strict=True,
                    )
                ]
            else:
                pre_features = rows
            post_features = _extract_runtime_features_from_context_ids_batch(
                evaluator=evaluator,
                samples=rows,
                prefixes=post_prefixes,
                context_id_rows=post_context_rows,
                max_context_tokens=int(args.post_max_context_tokens),
            )
            for local, (pre, post, prediction, target) in enumerate(
                zip(
                    pre_features,
                    post_features,
                    predictions,
                    expected_targets,
                    strict=True,
                )
            ):
                pre_comp = _feature_float(pre, "neuron_chain_composite_risk")
                pre_pair = _feature_float(pre, "neuron_chain_pair_risk")
                post_comp = _feature_float(post, "neuron_chain_composite_risk")
                post_pair = _feature_float(post, "neuron_chain_pair_risk")
                is_base_correct = bool(base_correct[start + local])
                is_repair_correct = int(prediction) == int(target)
                outcome = (
                    "recovered"
                    if not is_base_correct and is_repair_correct
                    else (
                        "not_recovered"
                        if not is_base_correct
                        else "damaged" if not is_repair_correct else "preserved"
                    )
                )
                for group_name in ("all", outcome):
                    _update_group(
                        raw_groups[group_name],
                        pre_composite=pre_comp,
                        post_composite=post_comp,
                        pre_pair=pre_pair,
                        post_pair=post_pair,
                    )
            rows_processed = end
            batches_since_checkpoint += 1
            progress_bar.update(len(rows))
            if (
                batches_since_checkpoint >= int(args.checkpoint_every_batches)
                or rows_processed == total_rows
            ):
                _write_json_atomic(
                    progress_path,
                    {
                        "status": (
                            "in_progress" if rows_processed < total_rows else "complete"
                        ),
                        "contract_sha256": contract_sha,
                        "rows_processed": rows_processed,
                        "total_rows": total_rows,
                        "raw_groups": raw_groups,
                        "elapsed_sec_current_attempt": round(
                            time.monotonic() - started, 3
                        ),
                    },
                )
                batches_since_checkpoint = 0
    progress_bar.close()
    if rows_processed != total_rows:
        raise ValueError(f"Evaluation ended early: {rows_processed}!={total_rows}")
    finalized = {name: _finalize_group(raw_groups[name]) for name in GROUPS}
    recovered = finalized["recovered"]
    not_recovered = finalized["not_recovered"]
    compact = {
        "recovered_delta_composite": recovered["mean_delta_composite"],
        "recovered_delta_pair": recovered["mean_delta_pair"],
        "not_recovered_delta_composite": not_recovered["mean_delta_composite"],
        "not_recovered_delta_pair": not_recovered["mean_delta_pair"],
        "composite_gap": round(
            recovered["mean_delta_composite"] - not_recovered["mean_delta_composite"], 6
        ),
        "pair_gap": round(
            recovered["mean_delta_pair"] - not_recovered["mean_delta_pair"], 6
        ),
    }
    report = {
        "status": "complete",
        "mode": "packed_hidden_post_writeback_route_shift",
        "contract": contract,
        "contract_sha256": contract_sha,
        "resumed": resumed,
        "rows_evaluated": total_rows,
        "raw_groups": raw_groups,
        "groups": finalized,
        "compact_table": compact,
        "elapsed_sec_current_attempt": round(time.monotonic() - started, 3),
    }
    _write_json_atomic(args.output, report)
    print(json.dumps({"output": str(args.output), **compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()

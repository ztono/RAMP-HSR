from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from ramp_hsr.checkpoints import load_editor
from scripts._route_directions import (
    _extract_final_route_nodes,
    _normalize,
    _reference_margin,
    _route_grad_direction,
    _route_proxy_scores,
    _route_sparse_direction,
)
from ramp_hsr.training import _load_lm_head, _logits_from_repair_hidden
from ramp_hsr.sample_features import (
    RISK_ACTIVE_COMPOSITE_THRESHOLD,
    RISK_ACTIVE_PAIR_THRESHOLD,
)
from ramp_hsr.routes import load_neuron_risk_policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path, *, with_hash: bool = True) -> dict[str, Any]:
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _calibration_files(path: Path) -> list[Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    files = sorted(resolved.glob("shard_*.pt"))
    if not files and (resolved / "calibration").is_dir():
        files = sorted((resolved / "calibration").glob("shard_*.pt"))
    if not files:
        raise FileNotFoundError(f"No calibration shard_*.pt files under {resolved}")
    return files


def _load_calibration(
    path: Path,
    *,
    hidden_dim: int,
    device: str,
    knn_k: int,
    composite_threshold: float = RISK_ACTIVE_COMPOSITE_THRESHOLD,
    pair_threshold: float = RISK_ACTIVE_PAIR_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    files = _calibration_files(path)
    hidden_parts: list[torch.Tensor] = []
    correct_parts: list[torch.Tensor] = []
    composite_parts: list[torch.Tensor] = []
    pair_parts: list[torch.Tensor] = []
    cache_metadata: list[dict[str, Any]] = []
    calibration_source_splits: list[str] = []
    calibration_unknown_split_rows = 0
    expected_route_names: list[str] | None = None
    for file_path in files:
        payload = torch.load(file_path, map_location="cpu", weights_only=False)
        if "h_last" not in payload:
            raise ValueError(
                f"Calibration cache lacks h_last: {file_path}. Hydrate or recollect this cache before evaluation."
            )
        hidden = torch.as_tensor(payload["h_last"]).float()
        base_correct = torch.as_tensor(payload["base_correct"]).bool().reshape(-1)
        route = torch.as_tensor(payload["route_features"]).float()
        route_names = [str(name) for name in payload["route_feature_names"]]
        if expected_route_names is None:
            expected_route_names = route_names
        elif route_names != expected_route_names:
            raise ValueError("Calibration route feature schema differs across shards")
        try:
            composite_index = route_names.index("neuron_chain_composite_risk")
            pair_index = route_names.index("neuron_chain_pair_risk")
        except ValueError as exc:
            raise ValueError(
                f"Calibration cache lacks composite/pair route scores: {file_path}"
            ) from exc
        if hidden.ndim != 2 or int(hidden.shape[1]) != int(hidden_dim):
            raise ValueError(
                f"Calibration hidden shape mismatch in {file_path}: {tuple(hidden.shape)} vs (*,{hidden_dim})"
            )
        if not hidden.shape[0] == base_correct.shape[0] == route.shape[0]:
            raise ValueError(f"Calibration row-count mismatch in {file_path}")
        hidden_parts.append(hidden)
        correct_parts.append(base_correct)
        composite_parts.append(route[:, composite_index])
        pair_parts.append(route[:, pair_index])
        cache_metadata.append(_fingerprint(file_path))
        row_metadata = payload.get("metadata")
        if not isinstance(row_metadata, list) or len(row_metadata) != len(base_correct):
            calibration_unknown_split_rows += int(len(base_correct))
        else:
            for item in row_metadata:
                if not isinstance(item, dict):
                    calibration_unknown_split_rows += 1
                    continue
                explicit = item.get("split") or item.get("partition")
                source = str(item.get("source_feature_shard") or "")
                if isinstance(explicit, str) and explicit:
                    calibration_source_splits.append(explicit.lower())
                elif "train_feature_shards" in source or "/train." in source:
                    calibration_source_splits.append("train")
                elif "eval_feature_shards" in source or "/eval." in source:
                    calibration_source_splits.append("eval")
                elif "test_feature_shards" in source or "/test." in source:
                    calibration_source_splits.append("test")
                else:
                    calibration_unknown_split_rows += 1
    hidden = torch.cat(hidden_parts, dim=0)
    base_correct = torch.cat(correct_parts, dim=0)
    composite = torch.cat(composite_parts, dim=0)
    pair = torch.cat(pair_parts, dim=0)
    risk_active = (composite >= float(composite_threshold)) | (
        pair >= float(pair_threshold)
    )
    active_error = risk_active & ~base_correct
    inactive_correct = ~risk_active & base_correct
    if not bool(active_error.any()) or not bool(inactive_correct.any()):
        raise ValueError(
            "Calibration needs route-active base-wrong and route-inactive base-correct examples"
        )
    mu_active = hidden[active_error].mean(dim=0)
    mu_inactive = hidden[inactive_correct].mean(dim=0)
    global_direction = _normalize((mu_inactive - mu_active).reshape(1, -1)).squeeze(0)
    knn_matrix = _normalize(hidden[inactive_correct]).contiguous()
    if int(knn_matrix.shape[0]) < int(knn_k):
        raise ValueError(
            f"Only {knn_matrix.shape[0]} inactive-correct calibration rows for K={knn_k}"
        )
    audit = {
        "source": str(path.expanduser().resolve()),
        "files": cache_metadata,
        "rows": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[1]),
        "base_correct": int(base_correct.sum().item()),
        "base_wrong": int((~base_correct).sum().item()),
        "risk_active": int(risk_active.sum().item()),
        "risk_inactive": int((~risk_active).sum().item()),
        "active_error": int(active_error.sum().item()),
        "inactive_correct": int(inactive_correct.sum().item()),
        "risk_thresholds": {
            "composite_ge": float(composite_threshold),
            "pair_ge": float(pair_threshold),
        },
        "knn_k": int(knn_k),
        "inferred_source_splits": sorted(set(calibration_source_splits)),
        "unknown_source_split_rows": int(calibration_unknown_split_rows),
        "train_only_source_verified": bool(
            calibration_source_splits
            and set(calibration_source_splits) == {"train"}
            and (calibration_unknown_split_rows == 0)
        ),
    }
    return (global_direction.to(device), knn_matrix.to(device), audit)


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


def _matched_neighbor_direction(
    h: torch.Tensor, *, calibration: torch.Tensor, k: int
) -> torch.Tensor:
    normalized_h = _normalize(h.float())
    similarities = normalized_h @ calibration.transpose(0, 1)
    selected = similarities.topk(
        k=min(max(1, int(k)), int(calibration.shape[0])), dim=1, largest=True
    ).indices
    neighbor_mean = calibration[selected].mean(dim=1)
    return _normalize(neighbor_mean - normalized_h)


def _contract(
    args: argparse.Namespace, *, packed_metadata_path: Path, samples_path: Path
) -> dict[str, Any]:
    return {
        "format": "mlp_residual_transport_packed_shard_v1",
        "setting": str(args.setting),
        "packed_dataset": str(args.packed_dataset.resolve()),
        "packed_metadata": _fingerprint(packed_metadata_path),
        "checkpoint": _fingerprint(args.checkpoint),
        "base_model": str(Path(args.base_model).expanduser().resolve()),
        "risk_policy": _fingerprint(args.risk_policy),
        "calibration_cache": str(args.calibration_cache.expanduser().resolve()),
        "shard_index": int(args.shard_index),
        "batch_size": int(args.batch_size),
        "knn_k": int(args.knn_k),
        "max_rows": int(args.max_rows),
        "sample_arrays": str(samples_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--packed-dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--risk-policy", required=True, type=Path)
    parser.add_argument("--calibration-cache", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.knn_k <= 0 or args.max_rows < 0:
        raise ValueError("Batch size/K must be positive and max_rows nonnegative")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    started = time.monotonic()
    packed_dir = args.packed_dataset.expanduser().resolve()
    metadata_path = packed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "repair_hidden_memmap_v1":
        raise ValueError(f"Unsupported packed format: {metadata.get('format')!r}")
    shards = metadata.get("shards") or []
    if args.shard_index < 0 or args.shard_index >= len(shards):
        raise ValueError(f"Invalid shard {args.shard_index}/{len(shards)}")
    shard = shards[int(args.shard_index)]
    shard_rows = int(shard["rows"])
    total_rows = min(shard_rows, args.max_rows) if args.max_rows else shard_rows
    hidden_dim = int(metadata["hidden_dim"])
    global_offset = sum((int(item["rows"]) for item in shards[: args.shard_index]))
    h_last = _open_array(
        packed_dir, shard, "h_last", dtype="<f4", shape=(shard_rows, hidden_dim)
    )
    target_ids = _open_array(
        packed_dir, shard, "target_token_id", dtype="<i8", shape=(shard_rows,)
    )
    packed_top1 = _open_array(
        packed_dir, shard, "base_top1_token_id", dtype="<i8", shape=(shard_rows,)
    )
    base_correct_array = _open_array(
        packed_dir, shard, "base_correct", dtype="u1", shape=(shard_rows,)
    )
    (repair_moe, checkpoint) = load_editor(
        args.checkpoint.expanduser().resolve(), args.device
    )
    repair_moe.eval()
    if str(checkpoint.get("feature_mode")) != "hidden_only":
        raise ValueError("Residual transport requires a hidden-only checkpoint")
    if int(checkpoint.get("hidden_dim", -1)) != hidden_dim:
        raise ValueError("Checkpoint and packed hidden dimensions differ")
    if int(checkpoint.get("rank", -1)) != 256:
        raise ValueError(
            f"Expected the selected r=256 MLP, got rank={checkpoint.get('rank')}"
        )
    if str(checkpoint.get("variant")) != "hidden_single_expert":
        raise ValueError(f"Unexpected editor variant: {checkpoint.get('variant')!r}")
    hidden_state_interface = str(checkpoint.get("hidden_state_interface", "post_norm"))
    policy = load_neuron_risk_policy(args.risk_policy.expanduser().resolve())
    if policy is None:
        raise RuntimeError(f"Could not load risk policy {args.risk_policy}")
    (final_nodes, final_layer) = _extract_final_route_nodes(policy, hidden_dim)
    if not final_nodes:
        raise ValueError("Risk policy has no final-layer nodes in editor space")
    (global_direction, knn_calibration, calibration_audit) = _load_calibration(
        args.calibration_cache,
        hidden_dim=hidden_dim,
        device=args.device,
        knn_k=int(args.knn_k),
    )
    (holder, norm, lm_head) = _load_lm_head(args.base_model, args.device)
    del holder
    gc.collect()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    output_path = args.output.expanduser().resolve()
    samples_path = output_path.with_suffix(".samples.npz")
    contract = _contract(
        args, packed_metadata_path=metadata_path, samples_path=samples_path
    )
    chunks: dict[str, list[np.ndarray]] = {
        "global_index": [],
        "base_correct": [],
        "repaired_correct": [],
        "recovered": [],
        "damaged": [],
        "risk_drop": [],
        "margin_gain": [],
        "step_global_proto": [],
        "step_knn_pair": [],
        "step_route_grad": [],
        "step_route_sparse": [],
        "delta_norm": [],
        "repair_top1": [],
    }
    packed_label_mismatches = 0
    projected_base_top1_mismatches = 0
    batches = range(0, total_rows, int(args.batch_size))
    progress = tqdm(
        batches,
        total=(total_rows + int(args.batch_size) - 1) // int(args.batch_size),
        desc=f"transport {args.setting} shard {args.shard_index}",
        dynamic_ncols=True,
        mininterval=5.0,
    )
    for start in progress:
        end = min(start + int(args.batch_size), total_rows)
        h = torch.from_numpy(np.asarray(h_last[start:end], dtype=np.float32).copy()).to(
            args.device
        )
        target = torch.from_numpy(
            np.asarray(target_ids[start:end], dtype=np.int64).copy()
        ).to(args.device)
        expected_top1 = torch.from_numpy(
            np.asarray(packed_top1[start:end], dtype=np.int64).copy()
        ).to(args.device)
        base_correct = (
            torch.from_numpy(
                np.asarray(base_correct_array[start:end], dtype=np.uint8).copy()
            )
            .to(args.device)
            .bool()
        )
        packed_label_mismatches += int(
            (base_correct != expected_top1.eq(target)).sum().item()
        )
        with torch.no_grad():
            empty = torch.empty(
                (end - start, 0), dtype=torch.float32, device=args.device
            )
            repaired_h = repair_moe(h)
            delta = repaired_h - h
            delta_norm = delta.norm(dim=1)
            base_logits = _logits_from_repair_hidden(
                h,
                norm=norm,
                lm_head=lm_head,
                hidden_state_interface=hidden_state_interface,
            )
            repair_logits = _logits_from_repair_hidden(
                repaired_h,
                norm=norm,
                lm_head=lm_head,
                hidden_state_interface=hidden_state_interface,
            )
            projected_base_top1_mismatches += int(
                (base_logits.argmax(dim=1) != expected_top1).sum().item()
            )
            repair_top1 = repair_logits.argmax(dim=1)
            repaired_correct = repair_top1.eq(target)
            recovered = ~base_correct & repaired_correct
            damaged = base_correct & ~repaired_correct
            margin_gain = _reference_margin(
                repair_logits.float(), target
            ) - _reference_margin(base_logits.float(), target)
            (comp_pre, pair_pre, node_scores) = _route_proxy_scores(
                h, policy=policy, nodes=final_nodes
            )
            (comp_post, pair_post, _) = _route_proxy_scores(
                repaired_h, policy=policy, nodes=final_nodes
            )
            risk_drop = comp_pre + pair_pre - comp_post - pair_post
            global_batch = global_direction.reshape(1, -1).expand_as(delta)
            step_global = (delta * global_batch).sum(dim=1)
            knn_direction = _matched_neighbor_direction(
                h, calibration=knn_calibration, k=int(args.knn_k)
            )
            step_knn = (delta * knn_direction).sum(dim=1)
            sparse_direction = _route_sparse_direction(
                h, nodes=final_nodes, node_scores=node_scores
            )
            step_sparse = (delta * sparse_direction).sum(dim=1)
        with torch.enable_grad():
            grad_direction = _route_grad_direction(
                h.detach(), policy=policy, nodes=final_nodes
            )
        with torch.no_grad():
            step_grad = (delta * grad_direction).sum(dim=1)
        tensors = {
            "global_index": torch.arange(
                global_offset + start,
                global_offset + end,
                dtype=torch.int64,
                device=args.device,
            ),
            "base_correct": base_correct,
            "repaired_correct": repaired_correct,
            "recovered": recovered,
            "damaged": damaged,
            "risk_drop": risk_drop,
            "margin_gain": margin_gain,
            "step_global_proto": step_global,
            "step_knn_pair": step_knn,
            "step_route_grad": step_grad,
            "step_route_sparse": step_sparse,
            "delta_norm": delta_norm,
            "repair_top1": repair_top1,
        }
        for name, tensor in tensors.items():
            chunks[name].append(tensor.detach().cpu().numpy())
        del (
            h,
            target,
            expected_top1,
            base_correct,
            empty,
            repaired_h,
            delta,
            base_logits,
            repair_logits,
            grad_direction,
        )
    arrays = {name: np.concatenate(parts, axis=0) for (name, parts) in chunks.items()}
    if int(arrays["global_index"].shape[0]) != total_rows:
        raise AssertionError("Output row count mismatch")
    for name in (
        "risk_drop",
        "margin_gain",
        "step_global_proto",
        "step_knn_pair",
        "step_route_grad",
        "step_route_sparse",
        "delta_norm",
    ):
        if not bool(np.isfinite(arrays[name]).all()):
            raise ValueError(f"Non-finite values in {name}")
    base_correct_np = arrays["base_correct"].astype(bool)
    repaired_correct_np = arrays["repaired_correct"].astype(bool)
    report = {
        "status": "complete",
        "contract": contract,
        "rows_evaluated": total_rows,
        "shard_rows": shard_rows,
        "global_offset": global_offset,
        "global_end": global_offset + total_rows,
        "base_correct": int(base_correct_np.sum()),
        "base_wrong": int((~base_correct_np).sum()),
        "repaired_correct": int(repaired_correct_np.sum()),
        "recovered": int(arrays["recovered"].sum()),
        "damaged": int(arrays["damaged"].sum()),
        "packed_label_mismatches": int(packed_label_mismatches),
        "projected_base_top1_mismatches": int(projected_base_top1_mismatches),
        "calibration": calibration_audit,
        "route_proxy": {
            "policy": str(args.risk_policy.expanduser().resolve()),
            "final_layer": int(final_layer) if final_layer is not None else None,
            "final_node_count": len(final_nodes),
            "score": "final-layer composite proxy plus pair proxy",
        },
        "editor": {
            "variant": checkpoint.get("variant"),
            "rank": checkpoint.get("rank"),
            "total_trained_epochs": checkpoint.get("total_trained_epochs"),
            "hidden_state_interface": hidden_state_interface,
        },
        "sample_arrays": (
            _fingerprint(samples_path, with_hash=False)
            if samples_path.exists()
            else {"path": str(samples_path)}
        ),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    _write_npz_atomic(samples_path, arrays)
    report["sample_arrays"] = _fingerprint(samples_path)
    _write_json_atomic(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
import gc
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
from scripts._route_directions import EPS
from scripts.evaluate_residual_transport import (
    _fingerprint,
    _load_calibration,
    _matched_neighbor_direction,
    _open_array,
    _write_json_atomic,
)
from ramp_hsr.checkpoints import load_editor
from scripts._route_directions import _reference_margin
from scripts._projection import (
    _random_basis_like,
    _random_unit_like,
    orthonormalize_directions,
    project_residual,
)
from ramp_hsr.training import _load_lm_head, _logits_from_repair_hidden

VARIANT_ORDER = (
    "full",
    "no_region",
    "no_global_proto",
    "no_knn_pair",
    "energy_matched_no_region",
    "random_no_region",
)
ENERGY_VARIANT_TO_SOURCE = {
    "no_region": "region",
    "no_global_proto": "global_proto",
    "no_knn_pair": "knn_pair",
    "energy_matched_no_region": "region",
    "random_no_region": "random_region",
}


def _new_variant_stats() -> dict[str, float | int]:
    return {
        "correct": 0,
        "recovered": 0,
        "damaged": 0,
        "margin_gain_sum": 0.0,
        "margin_gain_count": 0,
    }


def _new_energy_stats() -> dict[str, float | int]:
    return {
        "sum": 0.0,
        "count": 0,
        "rank_sum": 0,
        "rank_count": 0,
        "rank_one": 0,
        "rank_two": 0,
        "rank_other": 0,
    }


def _accumulate_variant(
    stats: dict[str, float | int],
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    base_correct: torch.Tensor,
    margin_gain: torch.Tensor,
) -> None:
    correct = prediction.eq(target)
    recovered = ~base_correct & correct
    damaged = base_correct & ~correct
    finite = torch.isfinite(margin_gain)
    stats["correct"] = int(stats["correct"]) + int(correct.sum().item())
    stats["recovered"] = int(stats["recovered"]) + int(recovered.sum().item())
    stats["damaged"] = int(stats["damaged"]) + int(damaged.sum().item())
    stats["margin_gain_sum"] = float(stats["margin_gain_sum"]) + float(
        margin_gain[finite].double().sum().item()
    )
    stats["margin_gain_count"] = int(stats["margin_gain_count"]) + int(
        finite.sum().item()
    )


def _accumulate_energy(
    stats: dict[str, float | int],
    *,
    projection: torch.Tensor,
    delta: torch.Tensor,
    rank: torch.Tensor,
) -> None:
    energy = projection.double().pow(2).sum(dim=1) / delta.double().pow(2).sum(
        dim=1
    ).clamp_min(EPS)
    if not bool(torch.isfinite(energy).all()):
        raise ValueError("Non-finite removed-energy fraction")
    rank_i64 = rank.to(torch.int64)
    stats["sum"] = float(stats["sum"]) + float(energy.sum().item())
    stats["count"] = int(stats["count"]) + int(energy.numel())
    stats["rank_sum"] = int(stats["rank_sum"]) + int(rank_i64.sum().item())
    stats["rank_count"] = int(stats["rank_count"]) + int(rank_i64.numel())
    stats["rank_one"] = int(stats["rank_one"]) + int((rank_i64 == 1).sum().item())
    stats["rank_two"] = int(stats["rank_two"]) + int((rank_i64 == 2).sum().item())
    stats["rank_other"] = int(stats["rank_other"]) + int(
        ((rank_i64 != 1) & (rank_i64 != 2)).sum().item()
    )


def _decode(
    hidden: torch.Tensor,
    *,
    target: torch.Tensor,
    norm: Any,
    lm_head: Any,
    hidden_state_interface: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _logits_from_repair_hidden(
        hidden,
        norm=norm,
        lm_head=lm_head,
        hidden_state_interface=hidden_state_interface,
    ).float()
    return (logits.argmax(dim=1), _reference_margin(logits, target))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--packed-dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--calibration-cache", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=13)
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
        raise ValueError("Projection ablation requires a hidden-only checkpoint")
    if int(checkpoint.get("hidden_dim", -1)) != hidden_dim:
        raise ValueError("Checkpoint and packed hidden dimensions differ")
    if int(checkpoint.get("rank", -1)) != 256:
        raise ValueError(f"Expected r=256, got rank={checkpoint.get('rank')}")
    if str(checkpoint.get("variant")) != "hidden_single_expert":
        raise ValueError(f"Unexpected editor variant: {checkpoint.get('variant')!r}")
    hidden_state_interface = str(checkpoint.get("hidden_state_interface", "post_norm"))
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
    shard_seed = int(args.random_seed) + 1000003 * int(args.shard_index)
    energy_generator = torch.Generator(device=args.device)
    energy_generator.manual_seed(shard_seed + 17003)
    rank_generator = torch.Generator(device=args.device)
    rank_generator.manual_seed(shard_seed + 91009)
    variant_stats = {name: _new_variant_stats() for name in VARIANT_ORDER}
    energy_stats = {
        name: _new_energy_stats()
        for name in ("region", "global_proto", "knn_pair", "random_region")
    }
    packed_label_mismatches = 0
    projected_base_top1_mismatches = 0
    max_projection_orthogonality_error = 0.0
    batches = range(0, total_rows, int(args.batch_size))
    progress = tqdm(
        batches,
        total=(total_rows + int(args.batch_size) - 1) // int(args.batch_size),
        desc=f"projection {args.setting} shard {args.shard_index}",
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
            repaired_h = repair_moe(h).float()
            delta = repaired_h - h
            delta_norm = delta.norm(dim=1)
            (base_pred, base_margin) = _decode(
                h,
                target=target,
                norm=norm,
                lm_head=lm_head,
                hidden_state_interface=hidden_state_interface,
            )
            projected_base_top1_mismatches += int(
                (base_pred != expected_top1).sum().item()
            )
            global_batch = global_direction.reshape(1, -1).expand_as(delta)
            knn_direction = _matched_neighbor_direction(
                h, calibration=knn_calibration, k=int(args.knn_k)
            )
            (region_basis, region_mask) = orthonormalize_directions(
                torch.stack([global_batch, knn_direction], dim=1), eps=1e-08
            )
            region_projection = project_residual(delta, region_basis, region_mask)
            (global_basis, global_mask) = orthonormalize_directions(
                global_batch.unsqueeze(1), eps=1e-08
            )
            global_projection = project_residual(delta, global_basis, global_mask)
            (knn_basis, knn_mask) = orthonormalize_directions(
                knn_direction.unsqueeze(1), eps=1e-08
            )
            knn_projection = project_residual(delta, knn_basis, knn_mask)
            energy_unit = _random_unit_like(delta, generator=energy_generator)
            energy_norm = region_projection.norm(dim=1, keepdim=True).clamp_max(
                delta_norm.view(-1, 1)
            )
            energy_matched_projection = energy_unit * energy_norm
            (random_basis, random_mask) = _random_basis_like(
                region_basis, region_mask, generator=rank_generator, eps=1e-08
            )
            random_projection = project_residual(delta, random_basis, random_mask)
            variants = {
                "full": repaired_h,
                "no_region": h + delta - region_projection,
                "no_global_proto": h + delta - global_projection,
                "no_knn_pair": h + delta - knn_projection,
                "energy_matched_no_region": h + delta - energy_matched_projection,
                "random_no_region": h + delta - random_projection,
            }
            for variant_name in VARIANT_ORDER:
                (prediction, margin) = _decode(
                    variants[variant_name],
                    target=target,
                    norm=norm,
                    lm_head=lm_head,
                    hidden_state_interface=hidden_state_interface,
                )
                _accumulate_variant(
                    variant_stats[variant_name],
                    prediction=prediction,
                    target=target,
                    base_correct=base_correct,
                    margin_gain=margin - base_margin,
                )
            _accumulate_energy(
                energy_stats["region"],
                projection=region_projection,
                delta=delta,
                rank=region_mask.sum(dim=1),
            )
            _accumulate_energy(
                energy_stats["global_proto"],
                projection=global_projection,
                delta=delta,
                rank=global_mask.sum(dim=1),
            )
            _accumulate_energy(
                energy_stats["knn_pair"],
                projection=knn_projection,
                delta=delta,
                rank=knn_mask.sum(dim=1),
            )
            _accumulate_energy(
                energy_stats["random_region"],
                projection=random_projection,
                delta=delta,
                rank=random_mask.sum(dim=1),
            )
            remainder = delta - region_projection
            orthogonality = (region_projection * remainder).sum(
                dim=1
            ).abs() / delta_norm.clamp_min(1.0)
            max_projection_orthogonality_error = max(
                max_projection_orthogonality_error, float(orthogonality.max().item())
            )
    output_path = args.output.expanduser().resolve()
    report = {
        "status": "complete",
        "format": "mlp_route_region_projection_packed_shard_v1",
        "setting": str(args.setting),
        "rows_evaluated": int(total_rows),
        "shard_rows": int(shard_rows),
        "global_offset": int(global_offset),
        "global_end": int(global_offset + total_rows),
        "base_correct": int(
            np.asarray(base_correct_array[:total_rows], dtype=np.uint8).sum()
        ),
        "base_wrong": int(
            total_rows
            - np.asarray(base_correct_array[:total_rows], dtype=np.uint8).sum()
        ),
        "packed_label_mismatches": int(packed_label_mismatches),
        "projected_base_top1_mismatches": int(projected_base_top1_mismatches),
        "variant_stats": variant_stats,
        "energy_stats": energy_stats,
        "energy_variant_to_source": ENERGY_VARIANT_TO_SOURCE,
        "max_projection_orthogonality_error": float(max_projection_orthogonality_error),
        "calibration": calibration_audit,
        "editor": {
            "variant": checkpoint.get("variant"),
            "rank": checkpoint.get("rank"),
            "total_trained_epochs": checkpoint.get("total_trained_epochs"),
            "hidden_state_interface": hidden_state_interface,
        },
        "random_control": {
            "draws_per_sample": 1,
            "base_seed": int(args.random_seed),
            "shard_seed": int(shard_seed),
            "energy_matched_seed": int(shard_seed + 17003),
            "random_rank_seed": int(shard_seed + 91009),
            "random_rank_matches_region_mask": True,
        },
        "contract": {
            "packed_dataset": str(packed_dir),
            "packed_metadata": _fingerprint(metadata_path),
            "checkpoint": _fingerprint(args.checkpoint.expanduser().resolve()),
            "base_model": str(Path(args.base_model).expanduser().resolve()),
            "calibration_cache": str(args.calibration_cache.expanduser().resolve()),
            "shard_index": int(args.shard_index),
            "batch_size": int(args.batch_size),
            "knn_k": int(args.knn_k),
            "max_rows": int(args.max_rows),
            "evaluator": _fingerprint(Path(__file__).resolve()),
        },
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    _write_json_atomic(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

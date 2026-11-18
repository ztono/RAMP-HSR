from __future__ import annotations
import bisect
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from ramp_hsr.editor import RAMPHSR

REPAIR_LOSS_STAT_KEYS = ("ce", "margin", "preserve", "preserve_margin", "l2", "total")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: str
) -> None:
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(target)


def _tensor_from_row(
    row: dict[str, Any], key: str, *, base_dir: Path
) -> torch.Tensor | None:
    value = row.get(key)
    if value is not None:
        return torch.tensor(value, dtype=torch.float32)
    tensor_path = row.get("tensor_path") or row.get("hidden_state_path")
    if not tensor_path:
        return None
    tensor_path = Path(tensor_path)
    if not tensor_path.is_absolute():
        tensor_path = base_dir / tensor_path
    payload = torch.load(tensor_path, map_location="cpu")
    if isinstance(payload, dict) and key in payload:
        return payload[key].float()
    if key == "h_last" and isinstance(payload, torch.Tensor):
        return payload.float()
    return None


class PackedRepairTensorDataset(Dataset):
    """Lossless hidden-only dataset backed by a handful of raw memmaps."""

    FORMAT_NAME = "repair_hidden_memmap_v1"

    def __init__(
        self,
        packed_dir: Path,
        *,
        manifest_path: Path | None = None,
        feature_mode: str = "hidden_only",
    ):
        if feature_mode != "hidden_only":
            raise ValueError(
                "Packed repair tensors currently support only feature_mode=hidden_only"
            )
        self.logical_packed_dir = packed_dir.resolve()
        self.packed_dir = self.logical_packed_dir
        local_map_path = os.environ.get("RAMP_LOCAL_PACKED_DATASET_MAP", "").strip()
        if local_map_path:
            mapping_path = Path(local_map_path).expanduser().resolve()
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            mapped = mapping.get(str(self.logical_packed_dir))
            if mapped:
                self.packed_dir = Path(mapped).expanduser().resolve()
        metadata_path = self.packed_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing packed dataset metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format") != self.FORMAT_NAME:
            raise ValueError(
                f"Unsupported packed dataset format: {self.metadata.get('format')!r}"
            )
        if self.metadata.get("feature_mode") != "hidden_only":
            raise ValueError("Packed dataset is not marked hidden_only")
        if self.metadata.get("hidden_dtype") != "float32":
            raise ValueError("Packed dataset must preserve float32 hidden states")
        if manifest_path is not None:
            recorded = self.metadata.get("combined_manifest")
            if recorded:
                resolved_manifest = manifest_path.resolve()
                stat = resolved_manifest.stat()
                actual = {
                    "path": str(resolved_manifest),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
                if actual != recorded:
                    raise ValueError(
                        f"Packed dataset does not match the requested combined manifest: packed={recorded} actual={actual}"
                    )
        self.hidden_dim = int(self.metadata["hidden_dim"])
        self.shards: list[dict[str, Any]] = []
        self._ends: list[int] = []
        cumulative = 0
        for shard_metadata in self.metadata.get("shards") or []:
            rows = int(shard_metadata["rows"])
            files = shard_metadata["files"]
            arrays = {
                "h_last": np.memmap(
                    self.packed_dir / files["h_last"]["path"],
                    dtype=np.dtype("<f4"),
                    mode="c",
                    shape=(rows, self.hidden_dim),
                ),
                "target_token_id": np.memmap(
                    self.packed_dir / files["target_token_id"]["path"],
                    dtype=np.dtype("<i8"),
                    mode="c",
                    shape=(rows,),
                ),
                "base_top1_token_id": np.memmap(
                    self.packed_dir / files["base_top1_token_id"]["path"],
                    dtype=np.dtype("<i8"),
                    mode="c",
                    shape=(rows,),
                ),
                "base_correct": np.memmap(
                    self.packed_dir / files["base_correct"]["path"],
                    dtype=np.dtype("u1"),
                    mode="c",
                    shape=(rows,),
                ),
                "risk_active": np.memmap(
                    self.packed_dir / files["risk_active"]["path"],
                    dtype=np.dtype("u1"),
                    mode="c",
                    shape=(rows,),
                ),
                "low_risk": np.memmap(
                    self.packed_dir / files["low_risk"]["path"],
                    dtype=np.dtype("u1"),
                    mode="c",
                    shape=(rows,),
                ),
            }
            expected_sizes = {
                "h_last": rows * self.hidden_dim * 4,
                "target_token_id": rows * 8,
                "base_top1_token_id": rows * 8,
                "base_correct": rows,
                "risk_active": rows,
                "low_risk": rows,
            }
            for key, expected_size in expected_sizes.items():
                path = self.packed_dir / files[key]["path"]
                if path.stat().st_size != expected_size:
                    raise ValueError(
                        f"Packed array size mismatch for {path}: expected={expected_size} actual={path.stat().st_size}"
                    )
            self.shards.append({"rows": rows, "arrays": arrays})
            cumulative += rows
            self._ends.append(cumulative)
        if not self.shards or cumulative != int(self.metadata.get("rows", -1)):
            raise ValueError(
                f"Packed shard row counts do not match root metadata: shards={cumulative} metadata={self.metadata.get('rows')}"
            )
        self.rows = None

    def __len__(self) -> int:
        return int(self._ends[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        shard_start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        local_index = index - shard_start
        arrays = self.shards[shard_index]["arrays"]
        return {
            "sample_id": f"packed:{index}",
            "h_last": torch.from_numpy(arrays["h_last"][local_index]),
            "route_features": torch.empty(0, dtype=torch.float32),
            "code_features": torch.empty(0, dtype=torch.float32),
            "target_token_id": torch.tensor(
                int(arrays["target_token_id"][local_index]), dtype=torch.long
            ),
            "base_top1_token_id": torch.tensor(
                int(arrays["base_top1_token_id"][local_index]), dtype=torch.long
            ),
            "base_correct": torch.tensor(
                bool(arrays["base_correct"][local_index]), dtype=torch.bool
            ),
            "risk_active": torch.tensor(
                bool(arrays["risk_active"][local_index]), dtype=torch.bool
            ),
            "low_risk": torch.tensor(
                bool(arrays["low_risk"][local_index]), dtype=torch.bool
            ),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "h_last": torch.stack([item["h_last"] for item in batch], dim=0),
        "route_features": torch.stack(
            [item["route_features"] for item in batch], dim=0
        ),
        "code_features": torch.stack([item["code_features"] for item in batch], dim=0),
        "target_token_id": torch.stack(
            [item["target_token_id"] for item in batch], dim=0
        ),
        "base_top1_token_id": torch.stack(
            [item["base_top1_token_id"] for item in batch], dim=0
        ),
        "base_correct": torch.stack([item["base_correct"] for item in batch], dim=0),
        "risk_active": torch.stack([item["risk_active"] for item in batch], dim=0),
        "low_risk": torch.stack([item["low_risk"] for item in batch], dim=0),
    }


class DistributedGlobalBatchSampler(Sampler[list[int]]):
    """Deterministically split each global batch across all DDP ranks.

    This keeps an exact effective global batch size even when it is not
    divisible by the world size.  For example, global_batch_size=32 and
    num_replicas=7 yields fixed local batch sizes [5, 5, 5, 5, 4, 4, 4].
    Every rank has the same number of optimizer steps.
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        global_batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.global_batch_size = int(global_batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if self.global_batch_size < self.num_replicas:
            raise ValueError(
                "global_batch_size must be at least num_replicas so every rank receives one or more samples"
            )
        (base, remainder) = divmod(self.global_batch_size, self.num_replicas)
        self.local_batch_sizes = tuple(
            (
                base + (1 if replica < remainder else 0)
                for replica in range(self.num_replicas)
            )
        )
        self.local_batch_size = int(self.local_batch_sizes[self.rank])
        self.local_offset = int(sum(self.local_batch_sizes[: self.rank]))
        if self.drop_last:
            self.num_batches = self.dataset_size // self.global_batch_size
        else:
            self.num_batches = (
                self.dataset_size + self.global_batch_size - 1
            ) // self.global_batch_size
        self.total_size = self.num_batches * self.global_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        if self.drop_last:
            indices = indices[: self.total_size]
        elif len(indices) < self.total_size:
            padding_size = self.total_size - len(indices)
            repeats = (padding_size + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:padding_size])
        for batch_index in range(self.num_batches):
            global_start = batch_index * self.global_batch_size
            local_start = global_start + self.local_offset
            local_end = local_start + self.local_batch_size
            yield indices[local_start:local_end]

    def __len__(self) -> int:
        return self.num_batches


def _ddp_loss_scale(
    *, local_batch_size: int, global_batch_size: int, world_size: int
) -> float:
    """Correct DDP's equal-rank gradient average for unequal local batches."""
    return float(local_batch_size) * float(world_size) / float(global_batch_size)


def _resolve_cached_model_path(model_name: str) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from huggingface_hub import snapshot_download

    try:
        return str(snapshot_download(repo_id=model_name, local_files_only=True))
    except Exception as exc:
        raise RuntimeError(
            f"Base model '{model_name}' is not available in the local Hugging Face cache. Pre-download it or pass a local model directory."
        ) from exc


def _load_lm_head(model_name: str, device: str):
    head_cache_root = os.environ.get("RAMP_LOCAL_LM_HEAD_CACHE_ROOT", "").strip()
    if head_cache_root:
        logical_model_path = Path(model_name).expanduser().resolve()
        cache_path = Path(head_cache_root).expanduser() / (
            logical_model_path.name + ".lm_head.pt"
        )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing staged LM-head cache {cache_path}; run stage_lm_head_cache.py before launching distributed workers."
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError(f"Unsupported LM-head cache format: {cache_path}")
        source = payload.get("source") or {}
        weight_path = logical_model_path / "model.safetensors"
        weight_stat = weight_path.stat()
        checks = {
            "model_name": source.get("model_name") == logical_model_path.name,
            "weight_size": int(source.get("weight_size", -1)) == weight_stat.st_size,
            "weight_mtime_ns": int(source.get("weight_mtime_ns", -1))
            == weight_stat.st_mtime_ns,
        }
        if not all(checks.values()):
            raise ValueError(
                f"Staged LM-head cache does not match {logical_model_path}: {checks}"
            )
        hidden_size = int(payload["hidden_size"])
        vocab_size = int(payload["vocab_size"])
        lm_head_weight = payload["lm_head_weight"]
        if tuple(lm_head_weight.shape) != (vocab_size, hidden_size):
            raise ValueError(
                f"Invalid LM-head shape in {cache_path}: {tuple(lm_head_weight.shape)}"
            )
        lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        lm_head.weight = torch.nn.Parameter(lm_head_weight, requires_grad=False)
        norm_type = str(payload.get("norm_type", "rms_norm"))
        norm_eps = float(payload.get("norm_eps", 1e-06))
        if norm_type == "layer_norm":
            norm = torch.nn.LayerNorm(hidden_size, eps=norm_eps)
            norm.weight = torch.nn.Parameter(
                payload["norm_weight"], requires_grad=False
            )
            norm.bias = torch.nn.Parameter(payload["norm_bias"], requires_grad=False)
        else:

            class _CachedRMSNorm(torch.nn.Module):

                def __init__(self, weight: torch.Tensor, eps: float):
                    super().__init__()
                    self.weight = torch.nn.Parameter(weight, requires_grad=False)
                    self.variance_epsilon = float(eps)

                def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                    input_dtype = hidden_states.dtype
                    values = hidden_states.to(torch.float32)
                    variance = values.pow(2).mean(-1, keepdim=True)
                    values = values * torch.rsqrt(variance + self.variance_epsilon)
                    return self.weight * values.to(input_dtype)

            norm = _CachedRMSNorm(payload["norm_weight"], norm_eps)
        holder = torch.nn.Module()
        holder.model = torch.nn.Module()
        holder.model.norm = norm
        holder.lm_head = lm_head
        holder.to(device)
        holder.eval()
        return (holder, holder.model.norm, holder.lm_head)
    from transformers import AutoModelForCausalLM

    model_path = _resolve_cached_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    norm = model.model.norm
    lm_head = model.lm_head
    return (model, norm, lm_head)


def _logits_from_repair_hidden(
    hidden: torch.Tensor, *, norm, lm_head, hidden_state_interface: str
) -> torch.Tensor:
    """Decode cached repair hidden states with the matching LM-head interface."""
    if hidden_state_interface == "post_norm":
        return lm_head(hidden)
    if hidden_state_interface == "pre_norm":
        return lm_head(norm(hidden))
    raise ValueError(
        f"Unsupported hidden_state_interface={hidden_state_interface!r}; expected 'post_norm' or 'pre_norm'"
    )


def _repair_loss(
    *,
    h_last: torch.Tensor,
    repaired_h: torch.Tensor,
    norm,
    lm_head,
    target_token_id: torch.Tensor,
    base_top1_token_id: torch.Tensor,
    base_correct_mask: torch.Tensor,
    risk_active_mask: torch.Tensor,
    preserve_mask: torch.Tensor,
    repair_moe: RAMPHSR,
    lambda_margin: float,
    lambda_preserve: float,
    lambda_preserve_margin: float,
    lambda_l2: float,
    margin: float,
    preserve_margin: float,
    preserve_margin_risk_active_weight: float,
    hidden_state_interface: str,
    return_stats_tensor: bool = False,
) -> tuple[torch.Tensor, dict[str, float] | torch.Tensor]:
    with torch.no_grad():
        base_logits = _logits_from_repair_hidden(
            h_last,
            norm=norm,
            lm_head=lm_head,
            hidden_state_interface=hidden_state_interface,
        )
    repaired_logits = _logits_from_repair_hidden(
        repaired_h,
        norm=norm,
        lm_head=lm_head,
        hidden_state_interface=hidden_state_interface,
    )
    ce_loss = F.cross_entropy(repaired_logits.float(), target_token_id)
    valid_wrong = (base_top1_token_id >= 0) & (base_top1_token_id != target_token_id)
    target_logits = (
        repaired_logits.float().gather(1, target_token_id.view(-1, 1)).squeeze(1)
    )
    wrong_logits = (
        repaired_logits.float()
        .gather(1, base_top1_token_id.clamp_min(0).view(-1, 1))
        .squeeze(1)
    )
    margin_values = F.relu(margin - target_logits + wrong_logits)[valid_wrong]
    margin_loss = margin_values.sum() / valid_wrong.sum().clamp_min(1).to(
        margin_values.dtype
    )
    preserve_repaired_logits = repaired_logits[preserve_mask].float()
    preserve_base_logits = base_logits[preserve_mask].float()
    preserve_loss = F.kl_div(
        F.log_softmax(preserve_repaired_logits, dim=-1),
        F.softmax(preserve_base_logits, dim=-1),
        reduction="sum",
    ) / preserve_mask.sum().clamp_min(1).to(preserve_repaired_logits.dtype)
    preserve_margin_loss = repaired_logits.new_zeros(())
    preserve_margin_mask = base_correct_mask.bool() & (target_token_id >= 0)
    if lambda_preserve_margin > 0.0:
        preserve_repaired_logits = repaired_logits[preserve_margin_mask].float()
        preserve_target_ids = target_token_id[preserve_margin_mask]
        preserve_target_logits = preserve_repaired_logits.gather(
            1, preserve_target_ids.view(-1, 1)
        ).squeeze(1)
        preserve_wrong_logits = preserve_repaired_logits.clone()
        preserve_wrong_logits.scatter_(
            1,
            preserve_target_ids.view(-1, 1),
            torch.finfo(preserve_wrong_logits.dtype).min,
        )
        preserve_max_wrong_logits = preserve_wrong_logits.max(dim=1).values
        preserve_margin_per_row = F.relu(
            preserve_margin - preserve_target_logits + preserve_max_wrong_logits
        )
        weights = torch.ones_like(preserve_margin_per_row)
        preserve_risk_active = risk_active_mask[preserve_margin_mask].bool()
        if preserve_margin_risk_active_weight != 1.0:
            weights = torch.where(
                preserve_risk_active,
                torch.full_like(weights, float(preserve_margin_risk_active_weight)),
                weights,
            )
        preserve_margin_loss = (
            preserve_margin_per_row * weights
        ).sum() / weights.sum().clamp_min(1.0)
    residual_loss = repair_moe.residual_norm_loss()
    total = (
        ce_loss
        + lambda_margin * margin_loss
        + lambda_preserve * preserve_loss
        + lambda_preserve_margin * preserve_margin_loss
        + lambda_l2 * residual_loss
    )
    stat_values = (
        ce_loss,
        margin_loss,
        preserve_loss,
        preserve_margin_loss,
        residual_loss,
        total,
    )
    if return_stats_tensor:
        return (total, torch.stack([value.detach() for value in stat_values]))
    return (
        total,
        {
            key: float(value.detach().cpu())
            for (key, value) in zip(REPAIR_LOSS_STAT_KEYS, stat_values)
        },
    )


class RepairTensorDataset(Dataset):
    """Load locally reconstructed hidden-state rows; diagnostics are audit labels."""

    def __init__(self, manifest_path: Path, *, feature_mode="hidden_only", **kwargs):
        if feature_mode != "hidden_only":
            raise ValueError("Only the RAMP-HSR hidden-only interface is supported")
        self.base_dir = manifest_path.parent
        self.rows = _read_jsonl(manifest_path)
        if not self.rows:
            raise ValueError("empty repair manifest")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        h = _tensor_from_row(row, "h_last", base_dir=self.base_dir)
        target = row.get("target_token_id")
        if h is None or target is None or int(target) < 0:
            raise ValueError("each row requires h_last and a valid target_token_id")
        return {
            "sample_id": row.get("sample_id"),
            "h_last": h,
            "route_features": torch.empty(0),
            "code_features": torch.empty(0),
            "target_token_id": torch.tensor(int(target), dtype=torch.long),
            "base_top1_token_id": torch.tensor(
                int(row.get("base_top1_token_id", -1)), dtype=torch.long
            ),
            **{
                k: torch.tensor(bool(row.get(k)), dtype=torch.bool)
                for k in ("base_correct", "risk_active", "low_risk")
            },
        }

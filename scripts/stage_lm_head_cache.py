#!/usr/bin/env python3
"""Stage only the frozen norm/output head needed by packed editor runs.

The training and table evaluator operate on cached final hidden states, so
loading every transformer block on every DDP rank is unnecessary.  This tool
extracts the exact norm and tied/untied output matrix once to node-local
storage.  The logical model path remains in checkpoints and reports.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    config_path = model_dir / "config.json"
    weight_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Incomplete single-file model directory: {model_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight_stat = weight_path.stat()
    source = {
        "model_name": model_dir.name,
        "logical_path": str(model_dir),
        "weight_path": str(weight_path),
        "weight_size": int(weight_stat.st_size),
        "weight_mtime_ns": int(weight_stat.st_mtime_ns),
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / f"{model_dir.name}.lm_head.pt"
    manifest = cache_root / f"{model_dir.name}.lm_head.json"

    if output.is_file():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        if (
            int(existing.get("format_version", -1)) == 1
            and existing.get("source") == source
        ):
            print(json.dumps({"status": "reused", "output": str(output)}))
            return

    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        head_key = (
            "lm_head.weight"
            if "lm_head.weight" in keys
            else "model.embed_tokens.weight"
        )
        norm_weight_key = "model.norm.weight"
        if head_key not in keys or norm_weight_key not in keys:
            raise KeyError(
                f"Required tensors missing from {weight_path}: "
                f"head={head_key in keys}, norm={norm_weight_key in keys}"
            )
        lm_head_weight = handle.get_tensor(head_key).to(torch.float32).contiguous()
        norm_weight = handle.get_tensor(norm_weight_key).to(torch.float32).contiguous()
        norm_bias = (
            handle.get_tensor("model.norm.bias").to(torch.float32).contiguous()
            if "model.norm.bias" in keys
            else None
        )

    hidden_size = int(config["hidden_size"])
    vocab_size = int(config["vocab_size"])
    if tuple(lm_head_weight.shape) != (vocab_size, hidden_size):
        raise ValueError(
            f"Head shape {tuple(lm_head_weight.shape)} does not match "
            f"config {(vocab_size, hidden_size)}"
        )
    norm_type = str(config.get("norm_type", "rms_norm"))
    if norm_type == "layer_norm" and norm_bias is None:
        raise ValueError(f"LayerNorm bias missing from {weight_path}")
    norm_eps = float(config.get("norm_epsilon", config.get("rms_norm_eps", 1e-6)))

    payload = {
        "format_version": 1,
        "source": source,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "head_key": head_key,
        "norm_type": norm_type,
        "norm_eps": norm_eps,
        "lm_head_weight": lm_head_weight,
        "norm_weight": norm_weight,
        "norm_bias": norm_bias,
    }
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(output)
    _atomic_json(
        manifest,
        {
            "status": "complete",
            "output": str(output),
            "source": source,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "head_key": head_key,
            "norm_type": norm_type,
            "norm_eps": norm_eps,
            "output_bytes": output.stat().st_size,
        },
    )
    print(json.dumps({"status": "complete", "output": str(output)}))


if __name__ == "__main__":
    main()

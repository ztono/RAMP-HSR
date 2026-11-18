from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from ramp_hsr.editor import RAMPHSR

FORMAT = "ramp_hsr_mlp_v1"


def canonical_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Remove only the functionally inert single-expert compatibility gate.

    Refuse MoE, feature-conditioned, linear, unnormalized, and fixed-strength
    checkpoints rather than silently treating them as the paper's main method.
    """
    state = payload["state_dict"]
    if payload.get("format") == FORMAT:
        if payload.get("hidden_state_interface") != "post_norm":
            raise ValueError("RAMP-HSR requires post_norm hidden states")
        return state
    expected = {
        "num_experts": 1,
        "condition_on": "hidden",
        "normalize_expert_input": True,
        "expert_activation": "gelu",
        "fixed_alpha": False,
        "hidden_state_interface": "post_norm",
        "route_dim": 0,
        "code_dim": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"unsupported historical checkpoint: {key}={payload.get(key)!r}"
            )
    mapping = {
        "experts.0.down.weight": "down.weight",
        "experts.0.up.weight": "up.weight",
        "risk_strength.0.weight": "strength.0.weight",
        "risk_strength.0.bias": "strength.0.bias",
        "risk_strength.3.weight": "strength.2.weight",
        "risk_strength.3.bias": "strength.2.bias",
    }
    unexpected = [
        key for key in state if key not in mapping and (not key.startswith("gate."))
    ]
    if unexpected or not set(mapping).issubset(state):
        raise ValueError(f"unexpected historical state dictionary: {unexpected}")
    return {new: state[old] for (old, new) in mapping.items()}


def load_editor(
    checkpoint_path: Path, device: str = "cpu"
) -> tuple[RAMPHSR, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = canonical_state(payload)
    hidden_dim = int(payload["hidden_dim"])
    model = RAMPHSR(
        hidden_dim,
        int(state["down.weight"].shape[0]),
        int(state["strength.0.weight"].shape[0]),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {
        **payload,
        "hidden_dim": hidden_dim,
        "route_dim": 0,
        "code_dim": 0,
        "num_experts": 1,
        "gate_top_k": 1,
        "condition_on": "hidden",
        "feature_mode": "hidden_only",
        "normalize_expert_input": True,
        "variant": "hidden_single_expert",
        "method_name": "RAMP-HSR",
        "rank": model.rank,
        "feature_hidden_dim": model.strength_width,
        "hidden_state_interface": "post_norm",
        "route_feature_names": [],
        "code_feature_names": [],
    }
    return (model, metadata)


def editor_payload(model: RAMPHSR, **metadata) -> dict[str, Any]:
    return {
        **metadata,
        "format": FORMAT,
        "state_dict": model.state_dict(),
        "hidden_dim": model.hidden_dim,
        "rank": model.rank,
        "strength_width": model.strength_width,
        "hidden_state_interface": "post_norm",
    }

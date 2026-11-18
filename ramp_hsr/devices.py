from __future__ import annotations
import os


def _normalize_gpu_ids(value: str | None) -> list[str]:
    if value is None:
        return []
    normalized = [part.strip() for part in value.split(",")]
    return [part for part in normalized if part]


def visible_gpu_ids_from_env() -> list[str]:
    visible = _normalize_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible:
        return visible
    return _normalize_gpu_ids(os.environ.get("SRLCC_GPU_IDS"))


def normalize_torch_device(device: str | None) -> str | None:
    """Map a physical CUDA index into the current visible-device namespace."""
    if device is None:
        return None
    if device == "cuda":
        return device
    if not device.startswith("cuda:"):
        return device
    try:
        requested_idx = int(device.split(":", 1)[1])
    except ValueError:
        return device
    visible = visible_gpu_ids_from_env()
    if not visible:
        return device
    if 0 <= requested_idx < len(visible):
        return f"cuda:{requested_idx}"
    requested_physical = str(requested_idx)
    if requested_physical in visible:
        return f"cuda:{visible.index(requested_physical)}"
    return device

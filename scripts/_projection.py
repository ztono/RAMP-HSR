from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EPS = 1e-12


def orthonormalize_directions(
    directions: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched modified Gram-Schmidt.

    Args:
        directions: [batch, num_dirs, hidden_dim].
    Returns:
        (basis, mask), where basis has the same shape and mask is [batch, num_dirs].
    """
    if directions.ndim != 3:
        raise ValueError(f"directions must be [B,K,H], got {tuple(directions.shape)}")
    basis_cols: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for col in range(directions.shape[1]):
        vec = directions[:, col, :].float()
        vec_norm = vec.norm(dim=1, keepdim=True)
        valid = vec_norm.squeeze(1) >= float(eps)
        vec = torch.where(
            valid.view(-1, 1), vec / vec_norm.clamp_min(EPS), torch.zeros_like(vec)
        )
        for prev in basis_cols:
            coeff = (vec * prev).sum(dim=1, keepdim=True)
            vec = vec - coeff * prev
        norm = vec.norm(dim=1, keepdim=True)
        valid = valid & (norm.squeeze(1) >= float(eps))
        unit = torch.where(
            valid.view(-1, 1), vec / norm.clamp_min(EPS), torch.zeros_like(vec)
        )
        basis_cols.append(unit)
        masks.append(valid)
    if not basis_cols:
        empty = directions.new_zeros((directions.shape[0], 0, directions.shape[2]))
        return (empty, directions.new_zeros((directions.shape[0], 0), dtype=torch.bool))
    return (torch.stack(basis_cols, dim=1), torch.stack(masks, dim=1))


def project_residual(
    delta: torch.Tensor, basis: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if basis.numel() == 0:
        return torch.zeros_like(delta)
    coeff = (basis * delta.unsqueeze(1)).sum(dim=2)
    coeff = torch.where(mask, coeff, torch.zeros_like(coeff))
    projected = (coeff.unsqueeze(2) * basis).sum(dim=1)
    if torch.isfinite(projected).all():
        no_component = delta - projected
        dot = (projected * no_component).sum(dim=1).abs()
        tolerance = 0.002 * delta.norm(dim=1).clamp_min(1.0)
        if bool((dot > tolerance).any()):
            max_dot = float(dot.max().detach().cpu().item())
            print(f"[warn] projection orthogonality tolerance exceeded: {max_dot:.4e}")
    return projected


def _random_basis_like(
    basis: torch.Tensor, mask: torch.Tensor, *, generator: torch.Generator, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    random_dirs = torch.randn(
        basis.shape, device=basis.device, dtype=basis.dtype, generator=generator
    )
    (random_basis, random_mask) = orthonormalize_directions(random_dirs, eps=eps)
    return (random_basis, random_mask & mask)


def _random_unit_like(x: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return noise / noise.norm(dim=1, keepdim=True).clamp_min(EPS)

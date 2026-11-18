from __future__ import annotations
import torch
from torch import nn


class RAMPHSR(nn.Module):
    """Edit post-normalization hidden states before the frozen LM head.

    The route readout is independent of this module. Both trainable branches
    receive only parameter-free LayerNorm(h). The residual up-projection is
    initialized to zero, so a fresh editor is exactly the identity.
    """

    def __init__(self, hidden_dim: int, rank: int = 256, strength_width: int = 256):
        super().__init__()
        if min(hidden_dim, rank, strength_width) <= 0:
            raise ValueError("hidden dimension and branch widths must be positive")
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.strength_width = int(strength_width)
        self.hidden_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.strength = nn.Sequential(
            nn.Linear(hidden_dim, strength_width),
            nn.GELU(),
            nn.Linear(strength_width, 1),
        )
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.strength[-1].weight)
        nn.init.constant_(self.strength[-1].bias, -3.0)
        self._last_alpha = None
        self._last_delta = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim < 2 or h.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected [..., {self.hidden_dim}] hidden states")
        q = self.hidden_norm(h)
        delta = self.up(torch.nn.functional.gelu(self.down(q)))
        alpha = torch.sigmoid(self.strength(q))
        self._last_alpha = alpha
        self._last_delta = delta
        return h + alpha * delta

    def residual_norm_loss(self) -> torch.Tensor:
        if self._last_delta is None:
            return self.up.weight.new_zeros(())
        return self._last_delta.square().mean()

    def latest_diagnostics(self) -> dict:
        if self._last_alpha is None:
            return {}
        return {
            "alpha_mean": self._last_alpha.detach().mean().item(),
            "alpha_std": self._last_alpha.detach().std(unbiased=False).item(),
            "delta_norm_mean": self._last_delta.detach().norm(dim=-1).mean().item(),
        }

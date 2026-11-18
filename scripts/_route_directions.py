from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from ramp_hsr.routes import NeuronRiskPolicy

EPS = 1e-12


@dataclass(frozen=True)
class FinalRouteNode:
    chain_name: str
    layer_index: int
    dimension: int
    direction: str
    chain_weight: float

    @property
    def key(self) -> str:
        return (
            f"{self.chain_name}:l{self.layer_index}:d{self.dimension}:{self.direction}"
        )


def _direction_label(raw: str) -> str:
    text = str(raw or "absolute").lower()
    if text in {"+", "pos", "positive"}:
        return "positive"
    if text in {"-", "neg", "negative"}:
        return "negative"
    return "absolute"


def _safe_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x.norm(dim=dim, keepdim=True).clamp_min(EPS)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / _safe_norm(x, dim=-1)


def _reference_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_logits = logits.gather(1, target.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target.view(-1, 1), torch.finfo(masked.dtype).min)
    return target_logits - masked.max(dim=1).values


def _extract_final_route_nodes(
    policy: NeuronRiskPolicy, hidden_dim: int
) -> tuple[list[FinalRouteNode], int | None]:
    if not policy.chains:
        return ([], None)
    max_layer = max(
        (int(node.layer_index) for chain in policy.chains for node in chain.nodes)
    )
    nodes: list[FinalRouteNode] = []
    seen: set[tuple[str, int, int, str]] = set()
    for chain in policy.chains:
        for raw_node in chain.nodes:
            layer = int(raw_node.layer_index)
            dim = int(raw_node.dimension)
            direction = _direction_label(raw_node.direction)
            if layer != max_layer or dim < 0 or dim >= hidden_dim:
                continue
            key = (chain.name, layer, dim, direction)
            if key in seen:
                continue
            seen.add(key)
            nodes.append(
                FinalRouteNode(
                    chain_name=str(chain.name),
                    layer_index=layer,
                    dimension=dim,
                    direction=direction,
                    chain_weight=float(chain.weight),
                )
            )
    nodes.sort(key=lambda node: (node.chain_name, node.dimension, node.direction))
    return (nodes, max_layer)


def _hidden_stats(
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = h.mean(dim=-1, keepdim=True)
    std = h.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-06)
    abs_h = h.abs()
    abs_mean = abs_h.mean(dim=-1, keepdim=True)
    abs_std = abs_h.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-06)
    return (mean, std, abs_mean, abs_std)


def _node_z_scores(h: torch.Tensor, nodes: list[FinalRouteNode]) -> torch.Tensor:
    if not nodes:
        return h.new_zeros((h.shape[0], 0))
    (mean, std, abs_mean, abs_std) = _hidden_stats(h)
    columns: list[torch.Tensor] = []
    for node in nodes:
        value = h[:, int(node.dimension) : int(node.dimension) + 1]
        if node.direction == "positive":
            score = (value - mean) / std
        elif node.direction == "negative":
            score = (mean - value) / std
        else:
            score = (value.abs() - abs_mean) / abs_std
        columns.append(score.squeeze(1))
    return torch.stack(columns, dim=1)


def _route_proxy_scores(
    h: torch.Tensor, *, policy: NeuronRiskPolicy, nodes: list[FinalRouteNode]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuous final-layer route proxy used for post-state risk-drop audit.

    The full risk route uses several intermediate layers.  This proxy only uses
    the final-layer nodes that are comparable to the editor residual space.
    """
    if not nodes:
        zeros = h.new_zeros((h.shape[0],))
        return (zeros, zeros, h.new_zeros((h.shape[0], 0)))
    z = _node_z_scores(h, nodes)
    node_index_by_chain: dict[str, list[int]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        node_index_by_chain[node.chain_name].append(idx)
    chain_scores: dict[str, torch.Tensor] = {}
    for chain in policy.chains:
        indices = node_index_by_chain.get(chain.name, [])
        if not indices:
            continue
        score = F.relu(z[:, indices]).mean(dim=1) * float(chain.weight)
        chain_scores[chain.name] = score
    if not chain_scores:
        zeros = h.new_zeros((h.shape[0],))
        return (zeros, zeros, z)
    stacked = torch.stack(list(chain_scores.values()), dim=1)
    max_risk = stacked.max(dim=1).values
    mean_risk = stacked.mean(dim=1)
    pair_scores: list[torch.Tensor] = []
    for pair in policy.pairs:
        left = chain_scores.get(pair.left)
        right = chain_scores.get(pair.right)
        if left is None or right is None:
            continue
        synergy = torch.minimum(left, right)
        pair_scores.append(0.5 * (left + right) * synergy * float(pair.weight))
    pair_risk = (
        torch.stack(pair_scores, dim=1).max(dim=1).values
        if pair_scores
        else h.new_zeros((h.shape[0],))
    )
    composite = 0.5 * max_risk + 0.2 * mean_risk + 0.2 * pair_risk
    return (composite.clamp_min(0.0), pair_risk.clamp_min(0.0), z)


def _route_grad_direction(
    h: torch.Tensor, *, policy: NeuronRiskPolicy, nodes: list[FinalRouteNode]
) -> torch.Tensor:
    work = h.detach().clone().requires_grad_(True)
    (comp, pair, _) = _route_proxy_scores(work, policy=policy, nodes=nodes)
    score = (comp + pair).sum()
    grad = torch.autograd.grad(score, work, retain_graph=False, create_graph=False)[0]
    return _normalize(-grad.detach())


def _route_sparse_direction(
    h: torch.Tensor, *, nodes: list[FinalRouteNode], node_scores: torch.Tensor
) -> torch.Tensor:
    out = torch.zeros_like(h)
    if not nodes:
        return out
    for idx, node in enumerate(nodes):
        score = node_scores[:, idx].clamp_min(0.0)
        active = score > 0.0
        if not bool(active.any()):
            continue
        dim = int(node.dimension)
        if node.direction == "positive":
            out[active, dim] += -score[active]
        elif node.direction == "negative":
            out[active, dim] += score[active]
        else:
            out[active, dim] += (
                -torch.sign(h[active, dim]).clamp(min=-1.0, max=1.0) * score[active]
            )
    norms = out.norm(dim=1, keepdim=True)
    return torch.where(norms > EPS, out / norms.clamp_min(EPS), torch.zeros_like(out))

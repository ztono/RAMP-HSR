from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _node_key(node: Any, *, include_direction: bool) -> tuple[Any, ...]:
    if isinstance(node, dict):
        layer = int(node["layer_index"])
        dim = int(node["dimension"])
        direction = str(node.get("direction") or "absolute").lower()
    else:
        parts = [part.strip() for part in str(node).split(":") if part.strip()]
        if len(parts) < 2:
            raise ValueError(f"Invalid node spec: {node!r}")
        layer = int(parts[0])
        dim = int(parts[1])
        direction = parts[2].lower() if len(parts) >= 3 else "absolute"
    if direction in {"+", "pos"}:
        direction = "positive"
    elif direction in {"-", "neg"}:
        direction = "negative"
    elif direction in {"abs"}:
        direction = "absolute"
    return (layer, dim, direction) if include_direction else (layer, dim)


def _chain_nodes(
    chain: dict[str, Any], *, include_direction: bool
) -> set[tuple[Any, ...]]:
    return {
        _node_key(node, include_direction=include_direction)
        for node in chain.get("nodes", [])
    }


def _policy_nodes(
    policy: dict[str, Any], *, include_direction: bool
) -> set[tuple[Any, ...]]:
    nodes: set[tuple[Any, ...]] = set()
    for chain in policy.get("chains", []):
        nodes.update(_chain_nodes(chain, include_direction=include_direction))
    return nodes


def _jaccard(left: set[Any], right: set[Any]) -> float | None:
    if not left and (not right):
        return None
    return len(left & right) / len(left | right)


def _best_chain_matches(
    left_policy: dict[str, Any],
    right_policy: dict[str, Any],
    *,
    include_direction: bool,
) -> list[dict[str, Any]]:
    right_chains = list(right_policy.get("chains", []))
    rows: list[dict[str, Any]] = []
    for left_chain in left_policy.get("chains", []):
        left_nodes = _chain_nodes(left_chain, include_direction=include_direction)
        best: dict[str, Any] | None = None
        for right_chain in right_chains:
            right_nodes = _chain_nodes(right_chain, include_direction=include_direction)
            score = _jaccard(left_nodes, right_nodes)
            row = {
                "left_chain": left_chain.get("name"),
                "right_chain": right_chain.get("name"),
                "jaccard": score,
                "overlap": len(left_nodes & right_nodes),
                "left_nodes": len(left_nodes),
                "right_nodes": len(right_nodes),
            }
            if best is None or float(score or 0.0) > float(best.get("jaccard") or 0.0):
                best = row
        if best is not None:
            rows.append(best)
    rows.sort(
        key=lambda row: (-float(row.get("jaccard") or 0.0), str(row.get("left_chain")))
    )
    return rows


def compare_policies(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    left = _load_json(left_path)
    right = _load_json(right_path)
    left_directed = _policy_nodes(left, include_direction=True)
    right_directed = _policy_nodes(right, include_direction=True)
    left_layer_dim = _policy_nodes(left, include_direction=False)
    right_layer_dim = _policy_nodes(right, include_direction=False)
    left_dims = {node[1] for node in left_layer_dim}
    right_dims = {node[1] for node in right_layer_dim}
    left_layers = {node[0] for node in left_layer_dim}
    right_layers = {node[0] for node in right_layer_dim}
    return {
        "left_path": str(Path(left_path).resolve()),
        "right_path": str(Path(right_path).resolve()),
        "left_name": left.get("name"),
        "right_name": right.get("name"),
        "counts": {
            "left_chains": len(left.get("chains", [])),
            "right_chains": len(right.get("chains", [])),
            "left_pairs": len(left.get("pairs", [])),
            "right_pairs": len(right.get("pairs", [])),
            "left_directed_nodes": len(left_directed),
            "right_directed_nodes": len(right_directed),
            "left_layer_dim_nodes": len(left_layer_dim),
            "right_layer_dim_nodes": len(right_layer_dim),
        },
        "overlap": {
            "directed_node_overlap": len(left_directed & right_directed),
            "directed_node_jaccard": _jaccard(left_directed, right_directed),
            "layer_dim_overlap": len(left_layer_dim & right_layer_dim),
            "layer_dim_jaccard": _jaccard(left_layer_dim, right_layer_dim),
            "dimension_overlap": len(left_dims & right_dims),
            "dimension_jaccard": _jaccard(left_dims, right_dims),
            "layer_overlap": len(left_layers & right_layers),
            "layer_jaccard": _jaccard(left_layers, right_layers),
        },
        "nodes_only_in_left": sorted(left_layer_dim - right_layer_dim),
        "nodes_only_in_right": sorted(right_layer_dim - left_layer_dim),
        "best_left_to_right_chain_matches": _best_chain_matches(
            left, right, include_direction=False
        ),
        "best_right_to_left_chain_matches": _best_chain_matches(
            right, left, include_direction=False
        ),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    overlap = report.get("overlap", {})
    counts = report.get("counts", {})
    lines = [
        "# Neuron Risk Policy Comparison",
        "",
        f"- Left: `{report.get('left_name')}`",
        f"- Right: `{report.get('right_name')}`",
        f"- Left path: `{report.get('left_path')}`",
        f"- Right path: `{report.get('right_path')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Left chains | {counts.get('left_chains', 0)} |",
        f"| Right chains | {counts.get('right_chains', 0)} |",
        f"| Left pairs | {counts.get('left_pairs', 0)} |",
        f"| Right pairs | {counts.get('right_pairs', 0)} |",
        f"| Directed node overlap | {overlap.get('directed_node_overlap', 0)} |",
        f"| Directed node Jaccard | {_fmt(overlap.get('directed_node_jaccard'))} |",
        f"| Layer-dim node overlap | {overlap.get('layer_dim_overlap', 0)} |",
        f"| Layer-dim node Jaccard | {_fmt(overlap.get('layer_dim_jaccard'))} |",
        f"| Dimension overlap | {overlap.get('dimension_overlap', 0)} |",
        f"| Dimension Jaccard | {_fmt(overlap.get('dimension_jaccard'))} |",
        "",
        "## Best Chain Matches",
        "",
        "| Left chain | Best right chain | Jaccard | Overlap |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in report.get("best_left_to_right_chain_matches", [])[:20]:
        lines.append(
            f"| `{row.get('left_chain')}` | `{row.get('right_chain')}` | {_fmt(row.get('jaccard'))} | {row.get('overlap', 0)} |"
        )
    lines.append("")
    lines.append("## Nodes Only In Left")
    lines.append("")
    lines.extend((f"- `{node}`" for node in report.get("nodes_only_in_left", [])[:50]))
    lines.append("")
    lines.append("## Nodes Only In Right")
    lines.append("")
    lines.extend((f"- `{node}`" for node in report.get("nodes_only_in_right", [])[:50]))
    return "\n".join(lines).strip() + "\n"

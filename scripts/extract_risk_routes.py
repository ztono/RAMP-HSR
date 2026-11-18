from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts._policy_comparison import (
    compare_policies,
    render_markdown as render_compare_markdown,
)
from ramp_hsr.route_discovery import (
    analyze_error_associated_neurons,
    render_error_neuron_analysis_markdown,
    write_error_neuron_outputs,
)
from ramp_hsr.io import write_json


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _slug(value: str) -> str:
    text = re.sub("[^0-9A-Za-z]+", "_", str(value or "").strip().lower())
    return text.strip("_") or "route"


def _direction_token(direction: Any) -> str:
    text = str(direction or "absolute").lower()
    if text in {"positive", "pos", "+"}:
        return "+"
    if text in {"negative", "neg", "-"}:
        return "-"
    return "abs"


def _node_spec(node: dict[str, Any]) -> str:
    return f"{int(node['layer_index'])}:{int(node['dimension'])}:{_direction_token(node.get('direction'))}"


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _rank_chains(chains: list[dict[str, Any]]) -> list[dict[str, Any]]:

    def sort_key(row: dict[str, Any]) -> tuple[float, float, int, str]:
        auc = (
            _finite_number(row.get("route_auc", row.get("score_auc"))) or -1000000000.0
        )
        risk_ratio = _finite_number(row.get("risk_ratio")) or -1000000000.0
        active_count = int(row.get("active_count") or 0)
        return (-auc, -risk_ratio, -active_count, str(row.get("name") or ""))

    valid = [
        row
        for row in chains
        if row.get("nodes")
        and _finite_number(row.get("route_auc", row.get("score_auc"))) is not None
    ]
    return sorted(valid, key=sort_key)


def build_policy_from_report(
    report: dict[str, Any],
    *,
    name: str,
    reference_policy_path: str | Path | None,
    chain_count: int,
    pair_count: int,
) -> dict[str, Any]:
    input_info = report.get("input") or {}
    selected_chains = _rank_chains(report.get("chain_activation_analysis") or [])[
        : max(1, int(chain_count))
    ]
    policy_chains: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    for index, row in enumerate(selected_chains, start=1):
        original_name = str(row.get("name") or f"chain_{index}")
        selected_names.add(original_name)
        policy_chains.append(
            {
                "name": original_name,
                "family": str(input_info.get("label_mode") or "train_only"),
                "weight": 1.0,
                "nodes": [_node_spec(node) for node in row.get("nodes", [])],
                "discovery": {
                    "source": row.get("source"),
                    "rank": index,
                    "route_auc": row.get("route_auc", row.get("score_auc")),
                    "risk_ratio": row.get("risk_ratio"),
                    "active_rate": row.get("active_rate"),
                    "positive_rate_when_active": row.get("positive_rate_when_active"),
                    "positive_rate_when_inactive": row.get(
                        "positive_rate_when_inactive"
                    ),
                },
            }
        )
    policy_pairs: list[dict[str, Any]] = []
    for row in report.get("chain_pair_interactions") or []:
        left = str(row.get("left_chain") or "")
        right = str(row.get("right_chain") or "")
        if left not in selected_names or right not in selected_names:
            continue
        pair_index = len(policy_pairs) + 1
        policy_pairs.append(
            {
                "name": f"train_pair_{pair_index}_{_slug(left)}_{_slug(right)}",
                "left": left,
                "right": right,
                "family": str(input_info.get("label_mode") or "train_only"),
                "weight": 1.0,
                "discovery": {
                    "source": row.get("source"),
                    "route_auc": row.get("route_auc", row.get("score_auc")),
                    "risk_ratio": row.get("risk_ratio"),
                    "error_lift_vs_neither": row.get("error_lift_vs_neither"),
                    "both_active_rate": row.get("both_active_rate"),
                },
            }
        )
        if len(policy_pairs) >= max(0, int(pair_count)):
            break
    return {
        "name": name,
        "node_z_threshold": float(input_info.get("chain_node_z_threshold") or 1.0),
        "chain_active_fraction": float(
            input_info.get("chain_min_active_fraction") or 0.5
        ),
        "chains": policy_chains,
        "pairs": policy_pairs,
        "provenance": {
            "source_split": "train",
            "discovery_input": input_info.get("path"),
            "heldout_input": input_info.get("heldout_input_path"),
            "label_mode": input_info.get("label_mode"),
            "negative_mode": input_info.get("negative_mode"),
            "sample_layers": input_info.get("sample_layers"),
            "max_samples": input_info.get("max_samples"),
            "max_per_class": input_info.get("max_per_class"),
            "sample_rate": input_info.get("sample_rate"),
            "sample_hash_min": input_info.get("sample_hash_min"),
            "model_name": input_info.get("model_name"),
            "max_length": input_info.get("max_length"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "reference_policy": (
                str(Path(reference_policy_path).resolve())
                if reference_policy_path
                else None
            ),
        },
        "validation": {
            "label": input_info.get("label_mode"),
            "negative_mode": input_info.get("negative_mode"),
            "controls": [
                "entropy_bucket",
                "margin_bucket",
                "prefix_length_bucket",
                "repo_id",
                "language",
            ],
        },
    }


def _write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _parse_int_csv(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return values or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--heldout-input", default=None)
    parser.add_argument("--current-config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--policy-output-filename", default="train_only_neuron_risk.json"
    )
    parser.add_argument("--train-hidden-cache", default=None)
    parser.add_argument("--heldout-hidden-cache", default=None)
    parser.add_argument("--require-hidden-cache", action="store_true")
    parser.add_argument("--policy-name", default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-layers", default="7,14,21,27")
    parser.add_argument("--label-mode", default="base_error")
    parser.add_argument("--negative-mode", default="base_correct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--max-per-class", type=int, default=10000)
    parser.add_argument("--heldout-max-samples", type=int, default=None)
    parser.add_argument("--heldout-max-per-class", type=int, default=None)
    parser.add_argument("--top-neurons", type=int, default=200)
    parser.add_argument("--auto-chain-top-neurons", type=int, default=120)
    parser.add_argument("--auto-chain-min-length", type=int, default=2)
    parser.add_argument("--include-ranked-node-chains", action="store_true")
    parser.add_argument("--ranked-chain-global-sizes", default="4,8,12")
    parser.add_argument("--ranked-chain-per-layer-counts", default="1,2,3")
    parser.add_argument("--chain-count", type=int, default=5)
    parser.add_argument("--pair-count", type=int, default=4)
    parser.add_argument("--pairwise-chain-top-k", type=int, default=8)
    parser.add_argument("--pairwise-chain-max-pairs", type=int, default=20)
    parser.add_argument(
        "--chain-score-mode",
        choices=["reference_correct", "runtime_zscore"],
        default="runtime_zscore",
        help="Use runtime_zscore to rank discovered routes with the same per-sample layer z-score used by BP-HSR feature extraction.",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = analyze_error_associated_neurons(
        args.train_input,
        model_name=args.model_name,
        device=args.device,
        sample_layers=[int(v) for v in args.sample_layers.split(",") if v.strip()],
        label_mode=args.label_mode,
        negative_mode=args.negative_mode,
        batch_size=max(1, int(args.batch_size)),
        max_length=max(1, int(args.max_length)),
        max_samples=max(1, int(args.max_samples)),
        max_per_class=max(0, int(args.max_per_class)),
        top_neurons=max(1, int(args.top_neurons)),
        chain_score_mode=args.chain_score_mode,
        auto_chain_min_length=max(2, int(args.auto_chain_min_length)),
        auto_chain_top_neurons=max(1, int(args.auto_chain_top_neurons)),
        include_ranked_node_chains=bool(args.include_ranked_node_chains),
        ranked_chain_global_sizes=_parse_int_csv(args.ranked_chain_global_sizes),
        ranked_chain_per_layer_counts=_parse_int_csv(
            args.ranked_chain_per_layer_counts
        ),
        pairwise_chain_top_k=max(2, int(args.pairwise_chain_top_k)),
        pairwise_chain_max_pairs=max(1, int(args.pairwise_chain_max_pairs)),
        heldout_input_path=args.heldout_input,
        heldout_max_samples=args.heldout_max_samples,
        heldout_max_per_class=args.heldout_max_per_class,
        hidden_cache_path=args.train_hidden_cache,
        heldout_hidden_cache_path=args.heldout_hidden_cache,
        require_hidden_cache=bool(args.require_hidden_cache),
    )
    write_error_neuron_outputs(report, out_dir / "analysis")
    _write_text(
        out_dir / "analysis" / "error_neuron_analysis.md",
        render_error_neuron_analysis_markdown(report),
    )
    policy_name = args.policy_name
    if not policy_name:
        policy_name = (
            f"{_slug(args.model_name)}_train_only_{_slug(args.label_mode)}_route_v1"
        )
    policy = build_policy_from_report(
        report,
        name=policy_name,
        reference_policy_path=args.current_config,
        chain_count=args.chain_count,
        pair_count=args.pair_count,
    )
    policy_path = out_dir / args.policy_output_filename
    write_json(policy, policy_path)
    comparison: dict[str, Any] | None = None
    comparison_path: Path | None = None
    comparison_md_path: Path | None = None
    if args.current_config:
        current_config = Path(args.current_config)
        if current_config.is_file():
            comparison = compare_policies(policy_path, current_config)
            comparison_path = out_dir / "train_only_vs_current_comparison.json"
            comparison_md_path = out_dir / "train_only_vs_current_comparison.md"
            write_json(comparison, comparison_path)
            _write_text(comparison_md_path, render_compare_markdown(comparison))
    summary = {
        "policy": str(policy_path.resolve()),
        "analysis_json": str(
            (out_dir / "analysis" / "error_neuron_analysis.json").resolve()
        ),
        "analysis_md": str(
            (out_dir / "analysis" / "error_neuron_analysis.md").resolve()
        ),
        "comparison_json": str(comparison_path.resolve()) if comparison_path else None,
        "comparison_md": (
            str(comparison_md_path.resolve()) if comparison_md_path else None
        ),
        "train_only_policy_name": policy.get("name"),
        "current_policy_name": comparison.get("right_name") if comparison else None,
        "overlap": comparison.get("overlap") if comparison else None,
    }
    write_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

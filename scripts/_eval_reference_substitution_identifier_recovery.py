from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _branch_summary(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    total = len(rows)
    values = [row[branch] for row in rows]
    counts = Counter((str(value["outcome"]) for value in values))
    identifier_emitted = sum((bool(value["identifier_emitted"]) for value in values))
    exact = sum((bool(value["exact_match"]) for value in values))
    first_token_exact = sum(
        (
            int(value["first_token_id"]) == int(row["correct_first_token_id"])
            for (row, value) in zip(rows, values, strict=True)
        )
    )
    matches_wrong = sum((bool(value["matches_injected_wrong"]) for value in values))
    censored = sum((bool(value["generation_censored"]) for value in values))
    return {
        "samples": total,
        "complete_identifier_exact": exact,
        "complete_identifier_exact_rate": _rate(exact, total),
        "first_token_exact": first_token_exact,
        "first_token_exact_rate": _rate(first_token_exact, total),
        "identifier_emitted": identifier_emitted,
        "identifier_emission_rate": _rate(identifier_emitted, total),
        "non_identifier_output": total - identifier_emitted,
        "non_identifier_output_rate": _rate(total - identifier_emitted, total),
        "generation_censored": censored,
        "generation_censored_rate": _rate(censored, total),
        "matches_injected_wrong": matches_wrong,
        "matches_injected_wrong_rate": _rate(matches_wrong, total),
        "outcome_counts": dict(sorted(counts.items())),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    base = [bool(row["base"]["exact_match"]) for row in rows]
    mlp = [bool(row["mlp_first_token_then_base"]["exact_match"]) for row in rows]
    transitions = {
        "fail_to_pass": sum((not left and right for (left, right) in zip(base, mlp))),
        "pass_to_fail": sum((left and (not right) for (left, right) in zip(base, mlp))),
        "fail_to_fail": sum(
            (not left and (not right) for (left, right) in zip(base, mlp))
        ),
        "pass_to_pass": sum((left and right for (left, right) in zip(base, mlp))),
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_token_stratum"])].append(row)
    return {
        "samples": total,
        "denominator_policy": "all cohort occurrences; non-identifier and censored generations count as incorrect",
        "base": _branch_summary(rows, "base"),
        "mlp_first_token_then_base": _branch_summary(rows, "mlp_first_token_then_base"),
        "paired_transitions": {
            **transitions,
            "net_gain_count": transitions["fail_to_pass"] - transitions["pass_to_fail"],
            "net_gain_rate": _rate(
                transitions["fail_to_pass"] - transitions["pass_to_fail"], total
            ),
        },
        "by_pair_token_stratum": {
            name: {
                "samples": len(group_rows),
                "base": _branch_summary(group_rows, "base"),
                "mlp_first_token_then_base": _branch_summary(
                    group_rows, "mlp_first_token_then_base"
                ),
            }
            for (name, group_rows) in sorted(grouped.items())
        },
    }

from __future__ import annotations
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np

DEFAULT_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (1.0, 0.7),
    (2.0, 1.5),
    (3.0, 2.5),
    (4.0, 3.5),
)
DEFAULT_THRESHOLD = (2.0, 1.5)
CI_METRICS = (
    "base_acc_route_pool",
    "active_rate",
    "precision",
    "recall",
    "fpr",
    "npv",
    "inactive_error_rate",
    "lr_pos",
    "odds_ratio",
    "mcc",
    "precision_lift",
    "risk_difference",
)


@dataclass(frozen=True)
class InputSpec:
    model: str
    language: str
    path: Path


@dataclass
class AuditData:
    model: str
    language: str
    input_path: str
    repos: np.ndarray
    base_correct: np.ndarray
    comp: np.ndarray
    pair: np.ndarray

    @property
    def base_wrong(self) -> np.ndarray:
        return ~self.base_correct


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _extract_named_feature(
    row: dict[str, Any], name: str, default: float = 0.0
) -> float:
    if name in row:
        return _safe_float(row.get(name), default)
    uncertainty = row.get("uncertainty")
    if isinstance(uncertainty, dict) and name in uncertainty:
        return _safe_float(uncertainty.get(name), default)
    for prefix in ("route", "code"):
        names = row.get(f"{prefix}_feature_names")
        values = row.get(f"{prefix}_features")
        if not isinstance(names, list) or not isinstance(values, list):
            continue
        try:
            idx = names.index(name)
        except ValueError:
            continue
        if 0 <= idx < len(values):
            return _safe_float(values[idx], default)
    return default


def _base_correct(row: dict[str, Any]) -> bool:
    if "base_correct" in row:
        return _safe_bool(row["base_correct"])
    for key in ("base_top1_equals_target_token_id", "base_top1_equals_target_token"):
        if key in row:
            return _safe_bool(row[key])
    target = row.get("target_token_id", row.get("target_first_token_id"))
    base = row.get("base_top1_token_id", row.get("base_top1_id"))
    if target is None or base is None:
        raise ValueError("Cannot infer base_correct: missing base/target token ids")
    return int(target) == int(base)


def _repo_id(row: dict[str, Any]) -> str:
    for key in ("repo_id", "repo", "repository", "repository_id"):
        value = row.get(key)
        if value:
            return str(value)
    path = row.get("file_path") or row.get("file") or row.get("path")
    if path:
        text = str(path)
        parts = [part for part in text.replace("\\", "/").split("/") if part]
        for marker in ("python", "typescript", "train", "test", "eval"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return parts[idx + 1]
        if len(parts) >= 2:
            return parts[-2]
    sample_id = row.get("sample_id") or row.get("id")
    raise ValueError(f"Cannot recover repo_id for sample {sample_id!r}")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _iter_parquet(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Parquet input requires pandas/pyarrow") from exc
    frame = pd.read_parquet(path)
    yield from frame.to_dict(orient="records")


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _iter_jsonl(path)
    if suffix == ".csv":
        return _iter_csv(path)
    if suffix in {".parquet", ".pq"}:
        return _iter_parquet(path)
    raise ValueError(f"Unsupported input type for {path}; use JSONL, CSV, or Parquet")


def load_audit_data(spec: InputSpec) -> AuditData:
    repos: list[str] = []
    base_correct: list[bool] = []
    comp: list[float] = []
    pair: list[float] = []
    for row in _iter_rows(spec.path):
        repos.append(_repo_id(row))
        base_correct.append(_base_correct(row))
        comp.append(_extract_named_feature(row, "neuron_chain_composite_risk"))
        pair.append(_extract_named_feature(row, "neuron_chain_pair_risk"))
    if not repos:
        raise ValueError(f"No rows loaded from {spec.path}")
    return AuditData(
        model=spec.model,
        language=spec.language,
        input_path=str(spec.path),
        repos=np.asarray(repos, dtype=object),
        base_correct=np.asarray(base_correct, dtype=bool),
        comp=np.asarray(comp, dtype=np.float64),
        pair=np.asarray(pair, dtype=np.float64),
    )


def confusion_from_arrays(
    base_wrong: np.ndarray, active: np.ndarray
) -> tuple[int, int, int, int]:
    tp = int(np.count_nonzero(active & base_wrong))
    fp = int(np.count_nonzero(active & ~base_wrong))
    fn = int(np.count_nonzero(~active & base_wrong))
    tn = int(np.count_nonzero(~active & ~base_wrong))
    return (tp, fp, fn, tn)


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    n = tp + fp + fn + tn
    base_wrong = tp + fn
    base_correct = fp + tn
    active = tp + fp
    base_acc = base_correct / n if n else math.nan
    prevalence = base_wrong / n if n else math.nan
    active_rate = active / n if n else math.nan
    precision = tp / active if active else math.nan
    recall = tp / base_wrong if base_wrong else math.nan
    fpr = fp / base_correct if base_correct else math.nan
    specificity = tn / base_correct if base_correct else math.nan
    npv = tn / (tn + fn) if tn + fn else math.nan
    inactive_error = fn / (fn + tn) if fn + tn else math.nan
    lr_pos = (
        recall / fpr
        if fpr and (not math.isnan(recall))
        else math.inf if fpr == 0 and recall > 0 else math.nan
    )
    lr_neg = (
        (1.0 - recall) / specificity
        if specificity and (not math.isnan(recall))
        else math.nan
    )
    corrected = False
    if min(tp, fp, fn, tn) == 0:
        (tp2, fp2, fn2, tn2) = (tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5)
        corrected = True
    else:
        (tp2, fp2, fn2, tn2) = (float(tp), float(fp), float(fn), float(tn))
    odds_ratio = tp2 * tn2 / (fp2 * fn2) if fp2 * fn2 else math.nan
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denom) if denom > 0 else math.nan
    precision_lift = (
        precision / prevalence
        if prevalence and (not math.isnan(precision))
        else math.nan
    )
    risk_difference = (
        precision - inactive_error
        if not math.isnan(precision) and (not math.isnan(inactive_error))
        else math.nan
    )
    return {
        "N": n,
        "N_base_wrong": base_wrong,
        "N_base_correct": base_correct,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "base_acc_route_pool": base_acc,
        "prevalence_wrong": prevalence,
        "active_rate": active_rate,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "specificity": specificity,
        "npv": npv,
        "inactive_error_rate": inactive_error,
        "lr_pos": lr_pos,
        "lr_neg": lr_neg,
        "odds_ratio": odds_ratio,
        "odds_ratio_corrected": corrected,
        "mcc": mcc,
        "precision_lift": precision_lift,
        "risk_difference": risk_difference,
    }


def assert_metric_invariants(metrics: dict[str, Any]) -> float:
    n = int(metrics["N"])
    (tp, fp, fn, tn) = (int(metrics[key]) for key in ("TP", "FP", "FN", "TN"))
    assert n == tp + fp + fn + tn
    assert int(metrics["N_base_wrong"]) == tp + fn
    assert int(metrics["N_base_correct"]) == fp + tn
    assert (
        abs(metrics["base_acc_route_pool"] + metrics["prevalence_wrong"] - 1.0) < 1e-12
    )
    lhs = metrics["active_rate"] * metrics["precision"]
    rhs = metrics["prevalence_wrong"] * metrics["recall"]
    if math.isnan(lhs) or math.isnan(rhs):
        return math.nan
    err = abs(lhs - rhs)
    assert err < 1e-10
    return err


def threshold_metrics(
    data: AuditData, thresholds: Iterable[tuple[float, float]], run_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_wrong = data.base_wrong
    for comp_thr, pair_thr in thresholds:
        active = (data.comp >= comp_thr) | (data.pair >= pair_thr)
        (tp, fp, fn, tn) = confusion_from_arrays(base_wrong, active)
        metrics = metrics_from_counts(tp, fp, fn, tn)
        identity_err = assert_metric_invariants(metrics)
        rows.append(
            {
                "run_id": run_id,
                "model": data.model,
                "language": data.language,
                "threshold_comp": comp_thr,
                "threshold_pair": pair_thr,
                **metrics,
                "active_precision_recall_identity_error": identity_err,
            }
        )
    return rows


def _average_precision_fallback(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.count_nonzero(y_true))
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order].astype(np.int64)
    cumsum = np.cumsum(y)
    ranks = np.arange(1, len(y) + 1)
    precision_at_k = cumsum / ranks
    return float(precision_at_k[y == 1].sum() / positives)


def _roc_auc_fallback(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.count_nonzero(y_true))
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(ranks[y_true].sum())
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def _sweep_threshold_stats(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    positives = int(np.count_nonzero(y_true))
    negatives = int(len(y_true) - positives)
    if positives == 0 or len(y_true) == 0:
        return {
            "recall_at_precision_099": math.nan,
            "recall_at_precision_097": math.nan,
            "recall_at_precision_095": math.nan,
            "precision_at_recall_030": math.nan,
            "precision_at_recall_050": math.nan,
            "precision_at_recall_070": math.nan,
            "max_mcc": math.nan,
            "threshold_at_max_mcc": math.nan,
        }
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order].astype(np.int64)
    s_sorted = scores[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    unique_last = np.r_[np.nonzero(s_sorted[:-1] != s_sorted[1:])[0], len(s_sorted) - 1]
    tp = tp_cum[unique_last].astype(np.float64)
    fp = fp_cum[unique_last].astype(np.float64)
    fn = positives - tp
    tn = negatives - fp
    precision = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=tp + fp > 0)
    recall = tp / positives
    out: dict[str, float] = {}
    for target in (0.99, 0.97, 0.95):
        mask = precision >= target
        out[f"recall_at_precision_{int(target * 100):03d}"] = (
            float(np.nanmax(recall[mask])) if np.any(mask) else 0.0
        )
    for target in (0.3, 0.5, 0.7):
        mask = recall >= target
        out[f"precision_at_recall_{int(target * 100):03d}"] = (
            float(np.nanmax(precision[mask])) if np.any(mask) else math.nan
        )
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = np.divide(
        tp * tn - fp * fn, np.sqrt(denom), out=np.full_like(tp, np.nan), where=denom > 0
    )
    if np.all(np.isnan(mcc)):
        out["max_mcc"] = math.nan
        out["threshold_at_max_mcc"] = math.nan
    else:
        idx = int(np.nanargmax(mcc))
        out["max_mcc"] = float(mcc[idx])
        out["threshold_at_max_mcc"] = float(s_sorted[unique_last[idx]])
    return out


def curve_metrics(data: AuditData, run_id: str) -> list[dict[str, Any]]:
    y_true = data.base_wrong.astype(bool)
    prevalence = float(np.count_nonzero(y_true) / len(y_true))
    score_map = {
        "comp": data.comp,
        "pair": data.pair,
        "combined": np.maximum(
            data.comp / DEFAULT_THRESHOLD[0], data.pair / DEFAULT_THRESHOLD[1]
        ),
    }
    rows: list[dict[str, Any]] = []
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        average_precision_score = None
        roc_auc_score = None
    for score_type, scores in score_map.items():
        if roc_auc_score is not None:
            try:
                auroc = float(roc_auc_score(y_true, scores))
            except ValueError:
                auroc = math.nan
        else:
            auroc = _roc_auc_fallback(y_true, scores)
        if average_precision_score is not None:
            try:
                auprc = float(average_precision_score(y_true, scores))
            except ValueError:
                auprc = math.nan
        else:
            auprc = _average_precision_fallback(y_true, scores)
        sweep = _sweep_threshold_stats(y_true, scores)
        rows.append(
            {
                "run_id": run_id,
                "model": data.model,
                "language": data.language,
                "score_type": score_type,
                "N": len(y_true),
                "prevalence_wrong": prevalence,
                "auroc": auroc,
                "auprc": auprc,
                "auprc_baseline": prevalence,
                "auprc_lift": auprc / prevalence if prevalence else math.nan,
                **sweep,
            }
        )
    return rows


def per_repo_counts(
    data: AuditData, thresholds: Iterable[tuple[float, float]]
) -> dict[tuple[float, float], dict[str, tuple[int, int, int, int]]]:
    out: dict[tuple[float, float], dict[str, tuple[int, int, int, int]]] = {}
    base_wrong = data.base_wrong
    for threshold in thresholds:
        (comp_thr, pair_thr) = threshold
        active = (data.comp >= comp_thr) | (data.pair >= pair_thr)
        repo_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for repo, wrong, act in zip(data.repos, base_wrong, active, strict=True):
            counts = repo_counts[str(repo)]
            if act and wrong:
                counts[0] += 1
            elif act and (not wrong):
                counts[1] += 1
            elif not act and wrong:
                counts[2] += 1
            else:
                counts[3] += 1
        out[threshold] = {repo: tuple(counts) for (repo, counts) in repo_counts.items()}
    return out


def bootstrap_ci(
    data: AuditData,
    thresholds: Iterable[tuple[float, float]],
    *,
    run_id: str,
    bootstrap_b: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    repo_counts_by_threshold = per_repo_counts(data, thresholds)
    for comp_thr, pair_thr in thresholds:
        repo_counts = repo_counts_by_threshold[comp_thr, pair_thr]
        repos = sorted(repo_counts)
        count_array = np.asarray([repo_counts[repo] for repo in repos], dtype=np.int64)
        point = metrics_from_counts(*count_array.sum(axis=0).tolist())
        values = {metric: [] for metric in CI_METRICS}
        n_repos = len(repos)
        for _ in range(bootstrap_b):
            sampled = [rng.randrange(n_repos) for _ in range(n_repos)]
            counts = count_array[sampled].sum(axis=0).tolist()
            metrics = metrics_from_counts(*[int(x) for x in counts])
            for metric in CI_METRICS:
                value = metrics[metric]
                if value is not None and (not math.isnan(float(value))):
                    values[metric].append(float(value))
        for metric in CI_METRICS:
            arr = np.asarray(values[metric], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if len(arr):
                (ci_low, ci_high) = np.nanquantile(arr, [0.025, 0.975])
            else:
                ci_low = ci_high = math.nan
            rows.append(
                {
                    "run_id": run_id,
                    "model": data.model,
                    "language": data.language,
                    "threshold_comp": comp_thr,
                    "threshold_pair": pair_thr,
                    "metric": metric,
                    "point": point[metric],
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "bootstrap_B": bootstrap_b,
                    "bootstrap_unit": "repo",
                    "seed": seed,
                }
            )
    return rows


def merge_ci_wide(
    threshold_rows: list[dict[str, Any]], ci_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ci_map: dict[tuple[str, str, float, float, str], tuple[float, float]] = {}
    for row in ci_rows:
        key = (
            str(row["model"]),
            str(row["language"]),
            float(row["threshold_comp"]),
            float(row["threshold_pair"]),
            str(row["metric"]),
        )
        ci_map[key] = (float(row["ci_low"]), float(row["ci_high"]))
    merged: list[dict[str, Any]] = []
    for row in threshold_rows:
        out = dict(row)
        key_prefix = (
            str(row["model"]),
            str(row["language"]),
            float(row["threshold_comp"]),
            float(row["threshold_pair"]),
        )
        for metric in CI_METRICS:
            (low, high) = ci_map.get((*key_prefix, metric), (math.nan, math.nan))
            out[f"{metric}_ci_low"] = low
            out[f"{metric}_ci_high"] = high
        merged.append(out)
    return merged


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "--"
    if math.isinf(f):
        return "$\\infty$"
    return f"{f:.{digits}f}"


def _fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def latex_tables(threshold_rows: list[dict[str, Any]], run_id: str) -> str:
    default_rows = [
        row
        for row in threshold_rows
        if float(row["threshold_comp"]) == DEFAULT_THRESHOLD[0]
        and float(row["threshold_pair"]) == DEFAULT_THRESHOLD[1]
    ]
    lines: list[str] = []
    lines.append("% Auto-generated by scripts/eval_route_audit_extended.py")
    lines.append(f"% run_id: {run_id}")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append(
        "\\caption{Compact route audit at the default hard threshold. LR$^+$ is the positive likelihood ratio, TPR/FPR.}"
    )
    lines.append("\\label{tab:route_compact_extended}")
    lines.append("\\begin{tabular}{llrrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Model & Lang. & $N$ & BaseAcc$_{\\mathrm{route}}$ & Active & Prec. & Recall & FPR & LR$^+$ & MCC \\\\"
    )
    lines.append("\\midrule")
    for row in default_rows:
        lines.append(
            f"{row['model']} & {row['language']} & {_fmt_int(row['N'])} & {_fmt(row['base_acc_route_pool'])} & {_fmt(row['active_rate'])} & {_fmt(row['precision'])} & {_fmt(row['recall'])} & {_fmt(row['fpr'])} & {_fmt(row['lr_pos'], 2)} & {_fmt(row['mcc'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(
        "\\caption{Identifier route-audit metrics across hard thresholds. LR$^+$ is TPR/FPR and OR is the odds ratio.}"
    )
    lines.append("\\label{tab:risk_route_thresholds_id_extended}")
    lines.append("\\begin{tabular}{llrrrrrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Model & Language & $\\tau_{comp}$ & $\\tau_{pair}$ & Active & Prec. & Recall & FPR & NPV & LR$^+$ & OR & MCC \\\\"
    )
    lines.append("\\midrule")
    for row in threshold_rows:
        lines.append(
            f"{row['model']} & {row['language']} & {_fmt(row['threshold_comp'], 1)} & {_fmt(row['threshold_pair'], 1)} & {_fmt(row['active_rate'])} & {_fmt(row['precision'])} & {_fmt(row['recall'])} & {_fmt(row['fpr'])} & {_fmt(row['npv'])} & {_fmt(row['lr_pos'], 2)} & {_fmt(row['odds_ratio'], 2)} & {_fmt(row['mcc'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def parse_input_specs(
    raw_inputs: list[str], model: str | None, language: str | None
) -> list[InputSpec]:
    specs: list[InputSpec] = []
    for raw in raw_inputs:
        if "|" in raw:
            parts = raw.split("|", 2)
            if len(parts) != 3:
                raise ValueError(f"Input spec must be model|language|path: {raw!r}")
            specs.append(InputSpec(parts[0], parts[1], Path(parts[2])))
        else:
            if not model or not language:
                raise ValueError(
                    "--model and --language are required when --input is a plain path"
                )
            specs.append(InputSpec(model, language, Path(raw)))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Path or model|language|path. Repeatable.",
    )
    parser.add_argument("--model", help="Model name for single plain-path input.")
    parser.add_argument("--language", help="Language for single plain-path input.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_id", default="route_audit_extended")
    parser.add_argument("--bootstrap_B", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    specs = parse_input_specs(args.input, args.model, args.language)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_threshold_rows: list[dict[str, Any]] = []
    all_curve_rows: list[dict[str, Any]] = []
    all_ci_rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"Loading {spec.model} / {spec.language}: {spec.path}", flush=True)
        data = load_audit_data(spec)
        rows = threshold_metrics(data, DEFAULT_THRESHOLDS, args.run_id)
        curves = curve_metrics(data, args.run_id)
        cis = bootstrap_ci(
            data,
            DEFAULT_THRESHOLDS,
            run_id=args.run_id,
            bootstrap_b=args.bootstrap_B,
            seed=args.seed,
        )
        all_threshold_rows.extend(rows)
        all_curve_rows.extend(curves)
        all_ci_rows.extend(cis)
        default = next(
            (
                row
                for row in rows
                if float(row["threshold_comp"]) == DEFAULT_THRESHOLD[0]
                and float(row["threshold_pair"]) == DEFAULT_THRESHOLD[1]
            )
        )
        print(
            f"{spec.model} / {spec.language}: N={default['N']} BaseAcc_route={default['base_acc_route_pool']:.4f} Active={default['active_rate']:.4f} Prec={default['precision']:.4f} Recall={default['recall']:.4f} FPR={default['fpr']:.4f} LR+={default['lr_pos']:.2f} MCC={default['mcc']:.4f}",
            flush=True,
        )
    all_wide_rows = merge_ci_wide(all_threshold_rows, all_ci_rows)
    threshold_path = args.output_dir / "route_threshold_metrics.csv"
    curve_path = args.output_dir / "route_curve_metrics.csv"
    ci_path = args.output_dir / "route_threshold_bootstrap_ci.csv"
    wide_path = args.output_dir / "route_threshold_metrics_with_ci.csv"
    latex_path = args.output_dir / "route_latex_tables.tex"
    write_csv(threshold_path, all_threshold_rows)
    write_csv(curve_path, all_curve_rows)
    write_csv(ci_path, all_ci_rows)
    write_csv(wide_path, all_wide_rows)
    latex_path.write_text(
        latex_tables(all_threshold_rows, args.run_id), encoding="utf-8"
    )
    print("Wrote:", flush=True)
    for path in (
        threshold_path,
        curve_path,
        ci_path,
        wide_path,
        latex_path,
    ):
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()

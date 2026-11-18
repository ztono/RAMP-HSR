from __future__ import annotations
import ast
import builtins
import difflib
import keyword
import logging
import math
import re
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

TOP_ERROR_TYPES = 5
CONTEXT_SYMBOL_FIELDS = (
    "visible_symbols",
    "local_var_symbols",
    "param_symbols",
    "member_candidates",
    "attribute_candidates",
    "import_candidates",
)
TEXT_METRIC_CACHE_SIZE = 131072
COMPLETION_BOUNDARY_CHARS = set("()[]{}.,:;+-*/%&|^~!=<>@#\\\"'")
_IDENTIFIER_REGEX = re.compile("[_a-zA-Z][_a-zA-Z0-9]*")
_MEMBER_ACCESS_RE = re.compile(
    "\\b([_a-zA-Z][_a-zA-Z0-9]*)\\.([_a-zA-Z][_a-zA-Z0-9]*)\\b"
)
_STRING_PATTERN = (
    "\"([^\"\\\\]*(\\\\.[^\"\\\\]*)*)\"|\\'([^\\'\\\\]*(\\\\.[^\\'\\\\]*)*)\\'"
)
_CODE_TOKENIZER_RE = re.compile("\\w+")
_CODE_METRIC_TOKENIZER_RE = re.compile(
    "[_a-zA-Z][_a-zA-Z0-9]*|\\d+(?:\\.\\d+)?|==|!=|<=|>=|->|:=|\\*\\*|//|[^\\s]"
)
_PYTHON_KEYWORDS = frozenset(keyword.kwlist)
_PYTHON_BUILTINS = frozenset(dir(builtins))
try:
    from rapidfuzz.distance import DamerauLevenshtein as _DL
except ImportError:
    _DL = None
try:
    from codebleu import calc_codebleu as _calc_codebleu
except ImportError:
    _calc_codebleu = None


def _round_metric(value: float) -> float:
    return round(float(value), 4)


def _normalize_completion(value: Any) -> str:
    return "" if value is None else str(value).strip()


_GENERATION_SPECIAL_STOP_TOKENS = {"<|endoftext|>", "<|im_end|>", "<|endoftext|>\n"}


def _contains_generation_special_token(value: Any) -> bool:
    text = "" if value is None else str(value)
    stripped = text.strip()
    if not stripped:
        return False
    if stripped in _GENERATION_SPECIAL_STOP_TOKENS:
        return True
    return "<|" in stripped and "|>" in stripped


def _prefix_compatible_completion(pred: Any, gt: Any) -> bool:
    """Boundary-aware prefix match for span-length mismatches."""
    pred_norm = _normalize_completion(pred)
    gt_norm = _normalize_completion(gt)
    if pred_norm == gt_norm:
        return True
    if not pred_norm or not gt_norm:
        return False
    if pred_norm.startswith(gt_norm):
        suffix = pred_norm[len(gt_norm) :]
    elif gt_norm.startswith(pred_norm):
        suffix = gt_norm[len(pred_norm) :]
    else:
        return False
    return bool(suffix) and (
        suffix[0].isspace() or suffix[0] in COMPLETION_BOUNDARY_CHARS
    )


def _candidate_string_values(value: Any) -> list[str]:
    values: list[str] = []
    if value is None:
        return values
    if isinstance(value, str):
        if value:
            values.append(value)
        return values
    if isinstance(value, dict):
        for key in ("name", "symbol", "identifier", "attr", "member", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                values.append(candidate)
        return values
    if isinstance(value, (list, tuple, set)):
        for item in value:
            values.extend(_candidate_string_values(item))
    return values


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _round_metric(numerator / denominator)


@lru_cache(maxsize=TEXT_METRIC_CACHE_SIZE)
def _edit_similarity(pred: str, gt: str) -> float:
    return _round_metric(difflib.SequenceMatcher(None, pred, gt).ratio() * 100.0)


@lru_cache(maxsize=TEXT_METRIC_CACHE_SIZE)
def _normalized_damerau_levenshtein(pred: str, gt: str) -> float | None:
    if _DL is None:
        return None
    dist = _DL.distance(pred, gt)
    max_len = max(len(pred), len(gt), 1)
    return _round_metric(dist / max_len)


@lru_cache(maxsize=TEXT_METRIC_CACHE_SIZE)
def _codebleu(pred: str, gt: str, enabled: bool = True) -> float | None:
    if not enabled or _calc_codebleu is None:
        return None
    if pred == gt:
        return 1.0
    try:
        previous_logging_disable = logging.root.manager.disable
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.disable(logging.WARNING)
            result = _calc_codebleu(
                [gt], [pred], lang="python", weights=(0.25, 0.25, 0.25, 0.25)
            )
            logging.disable(previous_logging_disable)
    except Exception:
        logging.disable(previous_logging_disable)
        return None
    if not isinstance(result, dict) or "codebleu" not in result:
        return None
    return _round_metric(float(result["codebleu"]))


@lru_cache(maxsize=TEXT_METRIC_CACHE_SIZE)
def _remove_comments(code: str) -> str:
    return re.sub("#.*", "", code)


@lru_cache(maxsize=TEXT_METRIC_CACHE_SIZE)
def _extract_identifiers(source_code: str) -> list[str]:
    no_strings = re.sub(_STRING_PATTERN, "", source_code)
    return [
        token
        for token in _CODE_TOKENIZER_RE.findall(no_strings)
        if _IDENTIFIER_REGEX.fullmatch(token) and token not in _PYTHON_KEYWORDS
    ]


@lru_cache(maxsize=TEXT_METRIC_CACHE_SIZE)
def _extract_member_access_pairs(source_code: str) -> list[tuple[str, str]]:
    no_strings = re.sub(_STRING_PATTERN, "", source_code)
    return [
        (receiver, member)
        for (receiver, member) in _MEMBER_ACCESS_RE.findall(no_strings)
        if receiver not in _PYTHON_KEYWORDS and member not in _PYTHON_KEYWORDS
    ]


def _context_symbol_names(sample: dict[str, Any] | None) -> set[str]:
    if not isinstance(sample, dict):
        return set()
    names: set[str] = set()
    for key in CONTEXT_SYMBOL_FIELDS:
        for value in _candidate_string_values(sample.get(key)):
            names.update(_extract_identifiers(value))
    for key in ("receiver_expr", "receiver_base"):
        names.update(_extract_identifiers(_normalize_completion(sample.get(key))))
    for value in sample.get("receiver_chain", []) or []:
        names.update(_extract_identifiers(_normalize_completion(value)))
    names.update({"self", "cls"})
    return names


def _context_member_grounding_sets(
    sample: dict[str, Any] | None
) -> tuple[set[str], set[str]]:
    if not isinstance(sample, dict):
        return (set(), set())
    receivers: set[str] = set()
    for key in ("visible_symbols", "local_var_symbols", "param_symbols"):
        for value in _candidate_string_values(sample.get(key)):
            receivers.update(_extract_identifiers(value))
    for key in ("receiver_expr", "receiver_base"):
        receivers.update(_extract_identifiers(_normalize_completion(sample.get(key))))
    for value in sample.get("receiver_chain", []) or []:
        receivers.update(_extract_identifiers(_normalize_completion(value)))
    receivers.update({"self", "cls"})
    members: set[str] = set()
    for key in ("member_candidates", "attribute_candidates"):
        for value in _candidate_string_values(sample.get(key)):
            members.update(_extract_identifiers(value))
    return (receivers, members)


def _compute_identifier_match(
    pred_identifiers: list[str], target_identifiers: list[str]
) -> tuple[int, float, float, float]:
    identifier_exact_match = int(pred_identifiers == target_identifiers)
    pred_set = sorted(set(pred_identifiers))
    target_set = sorted(set(target_identifiers))
    true_positive = sum((1 for pred in pred_set if pred in target_set))
    false_positive = sum((1 for pred in pred_set if pred not in target_set))
    false_negative = sum((1 for target in target_set if target not in pred_set))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * true_positive / (2 * true_positive + false_positive + false_negative)
        if 2 * true_positive + false_positive + false_negative
        else 0.0
    )
    return (
        identifier_exact_match,
        _round_metric(precision),
        _round_metric(recall),
        _round_metric(f1),
    )


@lru_cache(maxsize=TEXT_METRIC_CACHE_SIZE)
def _code_metric_tokens(text: str) -> tuple[str, ...]:
    return tuple(_CODE_METRIC_TOKENIZER_RE.findall(text or ""))


def _multiset_f1(
    pred_tokens: tuple[str, ...], target_tokens: tuple[str, ...]
) -> tuple[float, float, float]:
    if not pred_tokens and (not target_tokens):
        return (1.0, 1.0, 1.0)
    if not pred_tokens:
        return (0.0, 0.0, 0.0)
    if not target_tokens:
        return (0.0, 0.0, 0.0)
    pred_counts = Counter(pred_tokens)
    target_counts = Counter(target_tokens)
    overlap = sum(
        (
            min(count, target_counts.get(token, 0))
            for (token, count) in pred_counts.items()
        )
    )
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return (_round_metric(precision), _round_metric(recall), _round_metric(f1))


def _ngram_counts(tokens: tuple[str, ...], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(
        (tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    )


def _modified_ngram_precision(
    pred_tokens: tuple[str, ...], target_tokens: tuple[str, ...], n: int
) -> tuple[int, int]:
    pred_counts = _ngram_counts(pred_tokens, n)
    if not pred_counts:
        return (0, 0)
    target_counts = _ngram_counts(target_tokens, n)
    clipped = sum(
        (
            min(count, target_counts.get(ngram, 0))
            for (ngram, count) in pred_counts.items()
        )
    )
    return (clipped, sum(pred_counts.values()))


def _smoothed_sentence_bleu(
    pred_tokens: tuple[str, ...], target_tokens: tuple[str, ...], max_order: int
) -> float:
    if max_order <= 0:
        return 0.0
    if not pred_tokens or not target_tokens:
        return 0.0
    if pred_tokens == target_tokens:
        return 1.0
    log_precision_sum = 0.0
    smooth = 1.0
    for order in range(1, max_order + 1):
        (match_count, total_count) = _modified_ngram_precision(
            pred_tokens, target_tokens, order
        )
        if total_count <= 0:
            precision = 1.0 / (smooth * 2.0)
        elif match_count > 0:
            precision = match_count / total_count
        else:
            smooth *= 2.0
            precision = 1.0 / (smooth * total_count)
        log_precision_sum += math.log(max(precision, 1e-12))
    brevity_penalty = 1.0
    if len(pred_tokens) < len(target_tokens):
        brevity_penalty = math.exp(1.0 - len(target_tokens) / max(len(pred_tokens), 1))
    return _round_metric(
        brevity_penalty * math.exp(log_precision_sum / float(max_order))
    )


def _build_completion_metric_context(
    gt: str, sample: dict[str, Any] | None
) -> dict[str, Any]:
    target_text = gt or ""
    target_no_comments = _remove_comments(target_text)
    target_identifiers = _extract_identifiers(target_no_comments)
    available_symbols = _context_symbol_names(sample)
    target_context_identifiers = sorted(
        {
            identifier
            for identifier in target_identifiers
            if identifier not in _PYTHON_BUILTINS
        }
    )
    target_grounded_identifiers = [
        identifier
        for identifier in target_context_identifiers
        if identifier in available_symbols
    ]
    (receivers, members) = _context_member_grounding_sets(sample)
    target_pairs = sorted(set(_extract_member_access_pairs(target_no_comments)))
    target_grounded_pairs = [
        pair for pair in target_pairs if pair[0] in receivers and pair[1] in members
    ]
    return {
        "target_text": target_text,
        "target_identifiers": target_identifiers,
        "available_symbols": available_symbols,
        "target_grounded_identifiers": target_grounded_identifiers,
        "member_receivers": receivers,
        "member_names": members,
        "target_grounded_member_pairs": target_grounded_pairs,
    }


def _compute_completion_metrics_with_context(
    pred: str, *, enable_codebleu: bool, context: dict[str, Any]
) -> dict[str, float]:
    pred = pred or ""
    gt = str(context.get("target_text") or "")
    pred_no_comments = _remove_comments(pred)
    pred_identifiers = _extract_identifiers(pred_no_comments)
    target_identifiers = list(context.get("target_identifiers") or [])
    pred_tokens = _code_metric_tokens(pred_no_comments)
    target_tokens = _code_metric_tokens(_remove_comments(gt))
    (id_em, id_prec, id_rec, id_f1) = _compute_identifier_match(
        pred_identifiers, target_identifiers
    )
    (tok_prec, tok_rec, tok_f1) = _multiset_f1(pred_tokens, target_tokens)
    metrics: dict[str, float] = {
        "em": float(pred == gt),
        "strip_em": float(_normalize_completion(pred) == _normalize_completion(gt)),
        "prefix_em": float(_prefix_compatible_completion(pred, gt)),
        "es": _edit_similarity(pred, gt),
        "tok_prec": tok_prec,
        "tok_rec": tok_rec,
        "tok_f1": tok_f1,
        "tok_bleu1": _smoothed_sentence_bleu(pred_tokens, target_tokens, 1),
        "tok_bleu2": _smoothed_sentence_bleu(pred_tokens, target_tokens, 2),
        "tok_bleu3": _smoothed_sentence_bleu(pred_tokens, target_tokens, 3),
        "tok_bleu4": _smoothed_sentence_bleu(pred_tokens, target_tokens, 4),
        "id_em": float(id_em),
        "id_prec": id_prec,
        "id_rec": id_rec,
        "id_f1": id_f1,
    }
    dl = _normalized_damerau_levenshtein(pred, gt)
    if dl is not None:
        metrics["dl"] = dl
    codebleu = _codebleu(pred, gt, enabled=enable_codebleu)
    if codebleu is not None:
        metrics["codebleu"] = codebleu
    available_symbols = set(context.get("available_symbols") or set())
    target_grounded_identifiers = list(context.get("target_grounded_identifiers") or [])
    pred_context_identifiers = sorted(
        {
            identifier
            for identifier in pred_identifiers
            if identifier not in _PYTHON_BUILTINS
        }
    )
    if available_symbols and (pred_context_identifiers or target_grounded_identifiers):
        grounded_count = sum(
            (
                1
                for identifier in pred_context_identifiers
                if identifier in available_symbols
            )
        )
        recovered_count = sum(
            (
                1
                for identifier in target_grounded_identifiers
                if identifier in pred_context_identifiers
            )
        )
        precision = (
            grounded_count / len(pred_context_identifiers)
            if pred_context_identifiers
            else 0.0
        )
        recall = (
            recovered_count / len(target_grounded_identifiers)
            if target_grounded_identifiers
            else None
        )
        metrics.update(
            {
                "ctx_id_count": float(len(pred_context_identifiers)),
                "ctx_id_grounded_count": float(grounded_count),
                "ctx_id_precision": _round_metric(precision),
                "ctx_id_all_grounded": float(
                    bool(pred_context_identifiers)
                    and grounded_count == len(pred_context_identifiers)
                ),
                "ctx_target_id_count": float(len(target_grounded_identifiers)),
                "ctx_target_id_recovered_count": float(recovered_count),
            }
        )
        if recall is not None:
            metrics["ctx_target_id_recall"] = _round_metric(recall)
            metrics["ctx_target_id_all_recovered"] = float(
                recovered_count == len(target_grounded_identifiers)
            )
            metrics["ctx_id_f1"] = (
                _round_metric(2 * precision * recall / (precision + recall))
                if precision + recall > 0
                else 0.0
            )
    receivers = set(context.get("member_receivers") or set())
    members = set(context.get("member_names") or set())
    target_grounded_pairs = list(context.get("target_grounded_member_pairs") or [])
    pred_pairs = sorted(set(_extract_member_access_pairs(pred_no_comments)))
    if receivers and members and (pred_pairs or target_grounded_pairs):
        grounded_count = sum(
            (
                1
                for (receiver, member) in pred_pairs
                if receiver in receivers and member in members
            )
        )
        recovered_count = sum(
            (1 for pair in target_grounded_pairs if pair in pred_pairs)
        )
        precision = grounded_count / len(pred_pairs) if pred_pairs else 0.0
        recall = (
            recovered_count / len(target_grounded_pairs)
            if target_grounded_pairs
            else None
        )
        metrics.update(
            {
                "ctx_member_access_count": float(len(pred_pairs)),
                "ctx_member_access_grounded_count": float(grounded_count),
                "ctx_member_access_precision": _round_metric(precision),
                "ctx_member_access_all_grounded": float(
                    bool(pred_pairs) and grounded_count == len(pred_pairs)
                ),
                "ctx_target_member_access_count": float(len(target_grounded_pairs)),
                "ctx_target_member_access_recovered_count": float(recovered_count),
            }
        )
        if recall is not None:
            metrics["ctx_target_member_access_recall"] = _round_metric(recall)
            metrics["ctx_target_member_access_all_recovered"] = float(
                recovered_count == len(target_grounded_pairs)
            )
            metrics["ctx_member_access_f1"] = (
                _round_metric(2 * precision * recall / (precision + recall))
                if precision + recall > 0
                else 0.0
            )
    return metrics


def _empty_metric_agg() -> dict[str, Any]:
    return {"count": 0, "sums": defaultdict(float), "counts": defaultdict(int)}


def _update_metric_agg(agg: dict[str, Any], metrics: dict[str, float] | None) -> None:
    if not metrics:
        return
    agg["count"] += 1
    for key, value in metrics.items():
        agg["sums"][key] += float(value)
        agg["counts"][key] += 1


def _summarize_metric_agg(agg: dict[str, Any]) -> dict[str, float | int]:
    count = int(agg["count"])
    if count <= 0:
        return {"count": 0}
    result: dict[str, float | int] = {"count": count}
    for key in sorted(agg["sums"].keys()):
        metric_count = int(agg["counts"].get(key, 0))
        if metric_count > 0:
            result[key] = _round_metric(agg["sums"][key] / metric_count)
    return result


def _metric_deltas(
    current: dict[str, float | int], baseline: dict[str, float | int]
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key, value in current.items():
        if key == "count" or key not in baseline:
            continue
        if isinstance(value, (int, float)) and isinstance(baseline[key], (int, float)):
            deltas[key] = _round_metric(float(value) - float(baseline[key]))
    return deltas


def _syntax_error_category(error_type: str | None, error_msg: str | None) -> str | None:
    if not error_type:
        return None
    if error_type not in {"SyntaxError", "IndentationError"}:
        return error_type
    msg = (error_msg or "").lower()
    if "expected" in msg and ":" in msg:
        return "SyntaxError: expected ':'"
    if "unexpected eof" in msg:
        return "SyntaxError: unexpected EOF"
    if "was never closed" in msg or "unmatched" in msg:
        if ")" in msg:
            return "SyntaxError: unmatched ')'"
        if "]" in msg:
            return "SyntaxError: unmatched ']'"
        if "}" in msg:
            return "SyntaxError: unmatched '}'"
        return "SyntaxError: unmatched bracket"
    if error_type == "IndentationError":
        if "expected an indented block" in msg:
            return "IndentationError: expected an indented block"
        if "unexpected indent" in msg:
            return "IndentationError: unexpected indent"
        return "IndentationError: other"
    if msg:
        return f"SyntaxError: {msg.splitlines()[0].strip()}"
    return error_type


def evaluate_python_validity(code: str | None) -> dict[str, Any]:
    result = {
        "ast_ok": False,
        "compile_ok": False,
        "error_type": None,
        "error_msg": None,
        "error_category": None,
    }
    if code is None or not code.strip():
        return result
    parsed: ast.AST | None = None
    try:
        parsed = ast.parse(code)
        result["ast_ok"] = True
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_msg"] = str(exc)
    if parsed is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SyntaxWarning)
                compile(parsed, "<srl-cc-eval>", "exec")
            result["compile_ok"] = True
        except Exception as exc:
            if result["error_type"] is None:
                result["error_type"] = type(exc).__name__
                result["error_msg"] = str(exc)
    result["error_category"] = _syntax_error_category(
        result.get("error_type"), result.get("error_msg")
    )
    return result


def _empty_validity_agg() -> dict[str, Any]:
    return {
        "count": 0,
        "ast_ok": 0,
        "compile_ok": 0,
        "indentation_error": 0,
        "error_types": Counter(),
    }


def _update_validity_agg(agg: dict[str, Any], validity: dict[str, Any] | None) -> None:
    if not validity:
        return
    agg["count"] += 1
    if validity.get("ast_ok"):
        agg["ast_ok"] += 1
    if validity.get("compile_ok"):
        agg["compile_ok"] += 1
    error_category = validity.get("error_category") or validity.get("error_type")
    if error_category:
        agg["error_types"][str(error_category)] += 1
        if str(error_category).startswith("IndentationError"):
            agg["indentation_error"] += 1


def _summarize_validity_agg(agg: dict[str, Any]) -> dict[str, Any]:
    count = int(agg["count"])
    if count <= 0:
        return {"count": 0, "top_error_types": []}
    error_counter: Counter[str] = agg["error_types"]
    top_errors = [
        {
            "error_type": error_type,
            "count": error_count,
            "rate": _safe_rate(error_count, count),
        }
        for (error_type, error_count) in error_counter.most_common(TOP_ERROR_TYPES)
    ]
    return {
        "count": count,
        "ast_parse_rate": _safe_rate(int(agg["ast_ok"]), count),
        "compile_success_rate": _safe_rate(int(agg["compile_ok"]), count),
        "indentation_error_rate": _safe_rate(int(agg["indentation_error"]), count),
        "top_error_types": top_errors,
    }


def _take_evenly(indices: list[int], limit: int) -> list[int]:
    if limit <= 0 or not indices:
        return []
    if len(indices) <= limit:
        return list(indices)
    if limit == 1:
        return [indices[0]]
    step = (len(indices) - 1) / (limit - 1)
    return [indices[round(i * step)] for i in range(limit)]


class OfflineEvaluator:
    """Frozen model factory shared by continuation and route audits."""

    def __init__(
        self,
        *,
        long_sequence_rollout_model_name=None,
        long_sequence_rollout_device=None,
        long_sequence_rollout_llm=None,
        long_sequence_rollout_batch_size=16,
    ):
        self.long_sequence_rollout_model_name = long_sequence_rollout_model_name
        self.long_sequence_rollout_device = long_sequence_rollout_device
        self._long_sequence_rollout_llm = long_sequence_rollout_llm
        self.long_sequence_rollout_batch_size = max(
            1, int(long_sequence_rollout_batch_size)
        )

    def _get_long_sequence_rollout_llm(self, samples):
        from ramp_hsr.runtime import QwenCoderLLM

        model_name = self.long_sequence_rollout_model_name
        if not model_name:
            model_name = next(
                (
                    row["base_model_name"]
                    for row in samples
                    if row.get("base_model_name")
                ),
                "Qwen/Qwen2.5-Coder-1.5B",
            )
        if self._long_sequence_rollout_llm is None:
            self._long_sequence_rollout_llm = QwenCoderLLM(
                model_name=model_name, device=self.long_sequence_rollout_device
            )
        return (self._long_sequence_rollout_llm, model_name)

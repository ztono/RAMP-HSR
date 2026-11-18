from __future__ import annotations
import hashlib
import keyword
import re
from typing import Any, Sequence

EVALUATOR_VERSION = "reviewer3-complete-identifier-evaluator-v2"
PYTHON_IDENTIFIER_PREFIX = re.compile("^(?:_|[^\\W\\d])\\w*", re.UNICODE)
TYPESCRIPT_IDENTIFIER_PREFIX = re.compile("^[A-Za-z_$][A-Za-z0-9_$]*")
TYPESCRIPT_IDENTIFIER = re.compile("^[A-Za-z_$][A-Za-z0-9_$]*$")
TYPESCRIPT_KEYWORDS = frozenset(
    "\n    abstract any as asserts async await bigint boolean break case catch class\n    const constructor continue debugger declare default delete do else enum\n    export extends false finally for from function get if implements import in\n    infer instanceof interface is keyof let module namespace never new null\n    number object of package private protected public readonly require return\n    set static string super switch symbol this throw true try type typeof\n    undefined unique unknown var void while with yield\n    ".split()
)


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value) for value in tokenizer.encode(str(text), add_special_tokens=False)
    ]


def _decode(
    tokenizer: Any, token_ids: Sequence[int], *, skip_special_tokens: bool
) -> str:
    return str(
        tokenizer.decode(
            [int(value) for value in token_ids],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    )


def _ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in token_ids:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _first_character_difference(left: str, right: str) -> int | None:
    for index, (left_character, right_character) in enumerate(zip(left, right)):
        if left_character != right_character:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _tokenizer_normalize(tokenizer: Any, text: str) -> tuple[str | None, str | None]:
    """Return the tokenizer's normalized text when its backend exposes it."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    normalizer = getattr(backend, "normalizer", None)
    normalize = getattr(normalizer, "normalize_str", None)
    if not callable(normalize):
        return (None, None)
    return (str(normalize(str(text))), type(normalizer).__name__)


def audit_reference_continuation_path(
    *, tokenizer: Any, full_character_prefix: str, remaining_identifier_suffix: str
) -> dict[str, Any]:
    """Audit the old ``Tok(prefix) || Tok(suffix)`` reference construction.

    ``full_character_prefix`` is the complete causal source prefix, not merely
    the 512-token model window.  The function does not expose any right-context
    characters to a model; it only checks the offline reference path.
    """
    prefix = str(full_character_prefix)
    suffix = str(remaining_identifier_suffix)
    expected_text = prefix + suffix
    prefix_ids = _encode(tokenizer, prefix)
    suffix_ids = _encode(tokenizer, suffix)
    concatenated_ids = [*prefix_ids, *suffix_ids]
    continuous_ids = _encode(tokenizer, expected_text)
    decoded_concatenation = _decode(
        tokenizer, concatenated_ids, skip_special_tokens=False
    )
    decoded_continuous = _decode(tokenizer, continuous_ids, skip_special_tokens=False)
    decoded_prefix = _decode(tokenizer, prefix_ids, skip_special_tokens=False)
    decoded_suffix = _decode(tokenizer, suffix_ids, skip_special_tokens=False)
    (normalized_expected, normalizer_class) = _tokenizer_normalize(
        tokenizer, expected_text
    )
    character_exact = decoded_concatenation == expected_text
    same_path = concatenated_ids == continuous_ids
    normalized_character_exact = bool(
        normalized_expected is not None and decoded_concatenation == normalized_expected
    )
    append_exact_to_decoded_prefix = decoded_concatenation == decoded_prefix + suffix
    if character_exact:
        failure_cause = None
    elif normalized_character_exact and append_exact_to_decoded_prefix:
        failure_cause = "tokenizer_normalization_of_source_prefix"
    elif decoded_suffix != suffix:
        failure_cause = "suffix_roundtrip_difference"
    elif decoded_prefix != prefix and append_exact_to_decoded_prefix:
        failure_cause = "source_prefix_roundtrip_difference"
    else:
        failure_cause = "token_boundary_decode_interaction"
    if not character_exact:
        category = "reference_path_cannot_reconstruct_characters"
    elif same_path:
        category = "reference_path_matches_continuous_tokenization"
    else:
        category = "reference_path_decodes_exactly_but_differs_from_continuous_path"
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "category": category,
        "character_reconstruction_exact": character_exact,
        "failure_cause": failure_cause,
        "tokenizer_normalizer_class": normalizer_class,
        "normalized_character_reconstruction_exact": normalized_character_exact,
        "prefix_character_roundtrip_exact": decoded_prefix == prefix,
        "suffix_character_roundtrip_exact": decoded_suffix == suffix,
        "suffix_appends_exactly_to_decoded_prefix": append_exact_to_decoded_prefix,
        "same_as_continuous_token_path": same_path,
        "continuous_decode_matches_expected": decoded_continuous == expected_text,
        "prefix_token_count": len(prefix_ids),
        "suffix_token_count": len(suffix_ids),
        "concatenated_token_count": len(concatenated_ids),
        "continuous_token_count": len(continuous_ids),
        "prefix_token_ids_sha256": _ids_sha256(prefix_ids),
        "suffix_token_ids_sha256": _ids_sha256(suffix_ids),
        "concatenated_token_ids_sha256": _ids_sha256(concatenated_ids),
        "continuous_token_ids_sha256": _ids_sha256(continuous_ids),
        "suffix_token_ids": suffix_ids,
        "continuous_tail_token_ids": continuous_ids[-max(1, len(suffix_ids) + 2) :],
        "first_character_difference": _first_character_difference(
            expected_text, decoded_concatenation
        ),
        "expected_character_count": len(expected_text),
        "decoded_character_count": len(decoded_concatenation),
        "decoded_concatenation_tail": decoded_concatenation[max(0, len(prefix) - 64) :],
        "expected_text_tail": expected_text[max(0, len(prefix) - 64) :],
    }


def decode_generated_addition(
    *,
    tokenizer: Any,
    context_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Decode generated text relative to the exact token-ID context.

    Decoding the generated IDs alone can differ at a tokenizer boundary.  The
    primary path therefore decodes ``context || generation`` and removes the
    decoded context only when it is an exact prefix.  No whitespace is stripped.
    """
    context_ids = [int(value) for value in context_token_ids]
    generated_ids = [int(value) for value in generated_token_ids]
    decoded_context = _decode(tokenizer, context_ids, skip_special_tokens=True)
    decoded_full = _decode(
        tokenizer, [*context_ids, *generated_ids], skip_special_tokens=True
    )
    standalone = _decode(tokenizer, generated_ids, skip_special_tokens=True)
    aligned = decoded_full.startswith(decoded_context)
    added_text = decoded_full[len(decoded_context) :] if aligned else standalone
    return {
        "added_text": added_text,
        "decoded_context": decoded_context,
        "decoded_full": decoded_full,
        "standalone_generated_text": standalone,
        "full_decode_prefix_aligned": aligned,
        "decode_method": (
            "full_context_difference" if aligned else "standalone_fallback"
        ),
    }


def is_identifier(value: str, language: str) -> bool:
    text = str(value)
    if language == "typescript":
        return (
            bool(TYPESCRIPT_IDENTIFIER.fullmatch(text))
            and text not in TYPESCRIPT_KEYWORDS
        )
    if language != "python":
        raise ValueError(f"unsupported language: {language}")
    return bool(text) and text.isidentifier() and (not keyword.iskeyword(text))


def _identifier_prefix(text: str, language: str) -> str:
    pattern = (
        TYPESCRIPT_IDENTIFIER_PREFIX
        if language == "typescript"
        else PYTHON_IDENTIFIER_PREFIX
    )
    match = pattern.match(text)
    return "" if match is None else match.group(0)


def evaluate_anchored_identifier(
    *,
    target_identifier: str,
    existing_identifier_prefix: str,
    generated_text: str,
    language: str,
    cursor_kind: str,
    allow_identifier_start_horizontal_whitespace: bool,
    terminal_boundary_observed: bool = False,
    generation_reached_cap: bool = False,
) -> dict[str, Any]:
    """Evaluate one generated identifier at the original target position.

    ``cursor_kind`` is either ``identifier_start`` or ``identifier_internal``.
    Internal cursors never skip whitespace.  At identifier start, the tolerant
    protocol may skip only ASCII space and tab; it never skips a newline or any
    other token.  The strict protocol skips nothing.
    """
    if cursor_kind not in {"identifier_start", "identifier_internal"}:
        raise ValueError(f"unsupported cursor kind: {cursor_kind}")
    target = str(target_identifier)
    existing = str(existing_identifier_prefix)
    generated = str(generated_text)
    if not is_identifier(target, language):
        raise ValueError(f"target is not a {language} identifier: {target!r}")
    if cursor_kind == "identifier_start" and existing:
        raise ValueError(
            "identifier-start evaluation requires an empty existing prefix"
        )
    if cursor_kind == "identifier_internal":
        if not existing:
            raise ValueError(
                "identifier-internal evaluation requires a nonempty prefix"
            )
        if not target.startswith(existing) or existing == target:
            raise ValueError("existing prefix must be a proper target prefix")
    skipped = ""
    anchored_generated = generated
    if (
        cursor_kind == "identifier_start"
        and allow_identifier_start_horizontal_whitespace
    ):
        count = len(anchored_generated) - len(anchored_generated.lstrip(" \t"))
        skipped = anchored_generated[:count]
        anchored_generated = anchored_generated[count:]
    anchored_text = existing + anchored_generated
    predicted = _identifier_prefix(anchored_text, language)
    textual_boundary = bool(predicted) and len(predicted) < len(anchored_text)
    right_boundary = bool(textual_boundary or terminal_boundary_observed)
    censored = bool(generation_reached_cap and predicted and (not right_boundary))
    valid = is_identifier(predicted, language) if predicted else False
    emitted = bool(predicted and valid)
    exact = bool(
        emitted and right_boundary and (not censored) and (predicted == target)
    )
    if not predicted:
        outcome = "no_identifier_at_target_position"
    elif not valid:
        outcome = "non_identifier_keyword_or_invalid_lexeme"
    elif censored or not right_boundary:
        outcome = "identifier_unclosed_or_generation_censored"
    elif exact:
        outcome = "complete_identifier_exact"
    elif predicted.startswith(target) and predicted != target:
        outcome = "longer_identifier_than_reference"
    elif target.startswith(predicted) and predicted != target:
        outcome = "identifier_ended_before_reference"
    else:
        outcome = "different_identifier"
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "cursor_kind": cursor_kind,
        "allow_identifier_start_horizontal_whitespace": bool(
            allow_identifier_start_horizontal_whitespace
        ),
        "skipped_horizontal_whitespace": skipped,
        "generated_text": generated,
        "anchored_generated_text": anchored_generated,
        "existing_identifier_prefix": existing,
        "predicted_identifier": predicted if emitted else "",
        "raw_identifier_prefix": predicted,
        "identifier_emitted": emitted,
        "textual_right_boundary_observed": textual_boundary,
        "terminal_boundary_observed": bool(terminal_boundary_observed),
        "right_boundary_observed": right_boundary,
        "generation_reached_cap": bool(generation_reached_cap),
        "generation_censored": censored,
        "complete_identifier_exact": exact,
        "outcome": outcome,
    }

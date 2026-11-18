from __future__ import annotations
from typing import Any

TOKENIZER_PADDING_SIDE = "left"
TOKENIZER_TRUNCATION_SIDE = "left"


def configure_tokenizer_context_policy(tokenizer: Any) -> Any:
    """Configure tokenizer padding/truncation for prefix completion.

    Decoder-only completion should keep the most recent prefix tokens when a
    context budget is applied.  HuggingFace tokenizers default to right
    truncation, which keeps stale prefix text and drops the cursor context.
    """
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = TOKENIZER_PADDING_SIDE
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = TOKENIZER_TRUNCATION_SIDE
    return tokenizer


def validate_max_context_length(max_length: int, *, context: str = "tokenizer") -> int:
    value = int(max_length)
    if value <= 0:
        raise ValueError(f"{context} max_length must be positive; got {max_length!r}")
    return value


def assert_tokenizer_context_policy(
    tokenizer: Any, *, context: str = "tokenizer"
) -> None:
    padding_side = getattr(tokenizer, "padding_side", TOKENIZER_PADDING_SIDE)
    truncation_side = getattr(tokenizer, "truncation_side", TOKENIZER_TRUNCATION_SIDE)
    if padding_side != TOKENIZER_PADDING_SIDE:
        raise RuntimeError(
            f"{context} must use padding_side={TOKENIZER_PADDING_SIDE!r}; got {padding_side!r}"
        )
    if truncation_side != TOKENIZER_TRUNCATION_SIDE:
        raise RuntimeError(
            f"{context} must use truncation_side={TOKENIZER_TRUNCATION_SIDE!r}; got {truncation_side!r}"
        )


def tokenizer_context_policy_metadata(
    tokenizer: Any, *, max_length: int | None = None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
    }
    if max_length is not None:
        metadata["max_length"] = validate_max_context_length(max_length)
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_length, int) and 0 < model_max_length < 1000000:
        metadata["model_max_length"] = int(model_max_length)
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if isinstance(name_or_path, str) and name_or_path:
        metadata["tokenizer_name_or_path"] = name_or_path
    return metadata

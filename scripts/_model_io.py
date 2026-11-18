from __future__ import annotations
from pathlib import Path
import sys
from typing import Any
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from ramp_hsr.runtime import _import_hf_stack


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype


def _load_model(model_path: Path, device: str) -> tuple[Any, Any]:
    stack = _import_hf_stack()
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("a fast tokenizer with offset mappings is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = (
        stack["AutoModelForCausalLM"]
        .from_pretrained(
            str(model_path),
            dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        .to(device)
    )
    model.eval()
    return (tokenizer, model)


def _pad_batch(
    sequences: list[list[int]], *, pad_token_id: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths)
    input_ids = torch.full(
        (len(sequences), width), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, width - len(sequence) :] = torch.tensor(
            sequence, dtype=torch.long, device=device
        )
        attention_mask[row, width - len(sequence) :] = 1
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return (input_ids, attention_mask, position_ids, lengths)

import argparse
import json
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ramp_hsr.checkpoints import load_editor
from ramp_hsr.training import (
    PackedRepairTensorDataset,
    RepairTensorDataset,
    _collate,
    _load_lm_head,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--packed-dataset", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    dataset = (
        PackedRepairTensorDataset(args.packed_dataset)
        if args.packed_dataset
        else RepairTensorDataset(args.manifest)
    )
    (editor, _) = load_editor(args.checkpoint, args.device)
    (_, _, head) = _load_lm_head(args.base_model, args.device)
    counts = Counter()
    with torch.inference_mode():
        for batch in DataLoader(
            dataset, batch_size=args.batch_size, collate_fn=_collate
        ):
            h = batch["h_last"].to(args.device)
            y = batch["target_token_id"].to(args.device)
            b = head(h).argmax(-1)
            p = head(editor(h)).argmax(-1)
            if not torch.equal(b.cpu(), batch["base_top1_token_id"]):
                raise ValueError("cached base predictions differ from frozen head")
            correct = b.eq(y)
            if not torch.equal(correct.cpu(), batch["base_correct"]):
                raise ValueError("cached base-correct labels disagree with predictions")
            counts.update(
                n=y.numel(),
                base_correct=int(correct.sum()),
                editor_correct=int(p.eq(y).sum()),
                repaired=int((~correct & p.eq(y)).sum()),
                damaged=int((correct & p.ne(y)).sum()),
            )
    (n, good) = (counts["n"], counts["base_correct"])
    result = {
        "counts": dict(counts),
        "base_accuracy": good / n,
        "editor_accuracy": counts["editor_correct"] / n,
        "error_recovery": counts["repaired"] / (n - good) if n > good else None,
        "damage": counts["damaged"] / good if good else None,
        "net": (counts["editor_correct"] - good) / n,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

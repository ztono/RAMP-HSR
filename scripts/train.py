from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from ramp_hsr.checkpoints import load_editor, editor_payload
from ramp_hsr.editor import RAMPHSR
from ramp_hsr.training import (
    PackedRepairTensorDataset,
    RepairTensorDataset,
    DistributedGlobalBatchSampler,
    _collate,
    _ddp_loss_scale,
    _load_lm_head,
    _repair_loss,
    _move_optimizer_state_to_device,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--packed-dataset", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--strength-width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument("--lambda-preserve", type=float, default=0.5)
    parser.add_argument("--lambda-preserve-margin", type=float, default=0.75)
    parser.add_argument("--lambda-l2", type=float, default=0.0001)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--preserve-margin", type=float, default=1.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.global_batch_size < 1 or args.lr <= 0:
        parser.error("epochs, global batch size, and learning rate must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist

        cuda = args.device.startswith("cuda")
        dist.init_process_group("nccl" if cuda else "gloo")
        if cuda:
            args.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
            torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    dataset = (
        PackedRepairTensorDataset(args.packed_dataset)
        if args.packed_dataset
        else RepairTensorDataset(args.manifest)
    )
    sampler = DistributedGlobalBatchSampler(
        len(dataset),
        global_batch_size=args.global_batch_size,
        num_replicas=world_size,
        rank=rank,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_collate)
    model = RAMPHSR(dataset[0]["h_last"].shape[-1], args.rank, args.strength_width).to(
        args.device
    )
    previous_epochs = 0
    optimizer_state = None
    if args.resume:
        (loaded, payload) = load_editor(args.resume, args.device)
        if payload.get("format") != "ramp_hsr_mlp_v1":
            parser.error(
                "--resume requires a checkpoint from this trainer; historical wrappers are supported for evaluation"
            )
        model.load_state_dict(loaded.state_dict(), strict=True)
        previous_epochs = int(
            payload.get("total_trained_epochs", payload.get("trained_epochs", 0))
        )
        optimizer_state = payload.get("optimizer_state_dict")
    (_, norm, head) = _load_lm_head(args.base_model, args.device)
    model.train()
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=(
                [torch.cuda.current_device()]
                if args.device.startswith("cuda")
                else None
            ),
        )
        if distributed
        else model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, args.device)
    loss_config = {
        k: getattr(args, k)
        for k in [
            "lambda_margin",
            "lambda_preserve",
            "lambda_preserve_margin",
            "lambda_l2",
            "margin",
            "preserve_margin",
        ]
    }
    loss_config.update(
        preserve_margin_risk_active_weight=1.0, hidden_state_interface="post_norm"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_path = args.manifest or args.packed_dataset / "metadata.json"
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    for epoch in range(args.epochs):
        sampler.set_epoch(previous_epochs + epoch)
        started = time.monotonic()
        total = torch.zeros(2, dtype=torch.float64, device=args.device)
        for batch in loader:
            h = batch["h_last"].to(args.device)
            base_correct = batch["base_correct"].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            (loss, _) = _repair_loss(
                h_last=h,
                repaired_h=wrapped(h),
                norm=norm,
                lm_head=head,
                target_token_id=batch["target_token_id"].to(args.device),
                base_top1_token_id=batch["base_top1_token_id"].to(args.device),
                base_correct_mask=base_correct,
                preserve_mask=base_correct,
                risk_active_mask=batch["risk_active"].to(args.device),
                repair_moe=model,
                **loss_config,
            )
            scale = _ddp_loss_scale(
                local_batch_size=h.shape[0],
                global_batch_size=args.global_batch_size,
                world_size=world_size,
            )
            (loss * scale).backward()
            optimizer.step()
            total[0] += loss.detach().double() * h.shape[0]
            total[1] += h.shape[0]
        if distributed:
            dist.all_reduce(total)
        if rank == 0:
            trained = previous_epochs + epoch + 1
            payload = editor_payload(
                model,
                optimizer_state_dict=optimizer.state_dict(),
                base_model=args.base_model,
                seed=args.seed,
                total_trained_epochs=trained,
                training_rows=len(dataset),
                global_batch_size=args.global_batch_size,
                optimizer={"name": "AdamW", "lr": args.lr, "weight_decay": 0.01},
                loss_config=loss_config,
                source_metadata_sha256=source_digest,
            )
            temporary = args.output.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(args.output)
            print(
                json.dumps(
                    {
                        "epoch": trained,
                        "loss": float(total[0] / total[1]),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

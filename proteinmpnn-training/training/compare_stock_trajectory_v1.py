#!/usr/bin/env python3
"""Evaluate source and selected adapter epochs on identical validation batches."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from train_ph_adapter_light_v1 import evaluate, load_source_model, make_loaders, seed_all


METRICS = (
    "token_nll",
    "state_nll",
    "objective",
    "collapsed_accuracy",
    "state_accuracy",
    "canonical_recall",
    "alternate_recall",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--latest", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if not (
        len(args.source)
        == len(args.checkpoint_dir)
        == len(args.name)
        == len(args.latest)
        == 3
    ):
        raise ValueError("Expected exactly three source/checkpoint/name/latest groups")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    seed_all(args.seed)
    loader_args = SimpleNamespace(
        data_dir=args.data_dir,
        rescut=3.5,
        max_protein_length=800,
        num_train_examples=6000,
        num_valid_examples=1200,
        min_train_examples=500,
        min_valid_examples=100,
        batch_size=800,
    )
    _train_loader, valid_loader, data_summary = make_loaders(loader_args)
    valid_batches = list(iter(valid_loader))
    device = torch.device("cuda:0")
    rows = []
    common_epochs = (1, 2, 5, 10, 20, 30, 40, 50, 60)

    for name, source_path, checkpoint_dir, latest in zip(
        args.name, args.source, args.checkpoint_dir, args.latest
    ):
        model, source_metadata = load_source_model(source_path, device)
        baseline = evaluate(model, valid_batches, device, 0.25, args.seed + 810000)
        epochs = sorted(set(epoch for epoch in (*common_epochs, latest) if epoch <= latest))
        trajectory = []
        for epoch in epochs:
            path = checkpoint_dir / f"epoch_{epoch}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            metrics = evaluate(model, valid_batches, device, 0.25, args.seed + 810000)
            trajectory.append(
                {
                    "epoch": epoch,
                    "metrics": metrics,
                    "minus_stock": {key: metrics[key] - baseline[key] for key in METRICS},
                }
            )
            print(json.dumps({"name": name, "epoch": epoch, "metrics": metrics}), flush=True)
        rows.append(
            {
                "name": name,
                "source": str(source_path),
                "checkpoint_dir": str(checkpoint_dir),
                "latest_snapshot_epoch": latest,
                "source_metadata": source_metadata,
                "stock": baseline,
                "trajectory": trajectory,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "status": "ok",
        "data": data_summary,
        "identical_validation_batches": True,
        "identical_decoding_order_seed": True,
        "test_split_evaluated": False,
        "models": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"completed": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare source and best pH-adapter checkpoints on identical validation batches."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from train_ph_adapter_light_v1 import evaluate, load_source_model, make_loaders, seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--adapter", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if not (len(args.source) == len(args.adapter) == len(args.name) == 3):
        raise ValueError("Expected exactly three source/adapter/name triples")
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
    # Freeze one ordering so source and adapted models see exactly the same
    # structures, batches, masks, and decoding-order seeds.
    valid_batches = list(iter(valid_loader))
    device = torch.device("cuda:0")
    rows = []
    for name, source_path, adapter_path in zip(args.name, args.source, args.adapter):
        model, source_metadata = load_source_model(source_path, device)
        baseline = evaluate(model, valid_batches, device, 0.25, args.seed + 700000)
        adapted_checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)
        model.load_state_dict(adapted_checkpoint["model_state_dict"], strict=True)
        adapted = evaluate(model, valid_batches, device, 0.25, args.seed + 700000)
        delta = {
            key: adapted[key] - baseline[key]
            for key in (
                "token_nll",
                "state_nll",
                "objective",
                "collapsed_accuracy",
                "state_accuracy",
                "canonical_recall",
                "alternate_recall",
            )
        }
        rows.append(
            {
                "name": name,
                "source": str(source_path),
                "adapter": str(adapter_path),
                "adapter_epoch": adapted_checkpoint["epoch"],
                "source_metadata": source_metadata,
                "baseline": baseline,
                "adapted": adapted,
                "adapted_minus_baseline": delta,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "status": "ok",
        "data": data_summary,
        "identical_validation_batches": True,
        "test_split_evaluated": False,
        "models": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

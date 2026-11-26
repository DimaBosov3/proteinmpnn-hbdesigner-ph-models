#!/usr/bin/env python3
"""CPU preflight on the real split, data loader, and all three checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from model_utils import featurize
from train_ph_adapter_light_v1 import load_source_model, make_loaders


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    loader_args = SimpleNamespace(
        data_dir=args.data_dir,
        rescut=3.5,
        max_protein_length=600,
        num_train_examples=64,
        num_valid_examples=32,
        min_train_examples=8,
        min_valid_examples=4,
        batch_size=800,
    )
    train_loader, valid_loader, data_summary = make_loaders(loader_args)
    batch = next(iter(valid_loader))
    device = torch.device("cpu")
    X, S, mask, _lengths, chain_M, residue_idx, _mask_self, chain_encoding = featurize(
        batch, device
    )

    models = []
    for checkpoint in args.checkpoint:
        model, metadata = load_source_model(checkpoint, device)
        model.eval()
        torch.manual_seed(101)
        low = model(
            X, S, mask, chain_M, residue_idx, chain_encoding,
            torch.full((len(batch),), 4.0),
        )
        torch.manual_seed(101)
        high = model(
            X, S, mask, chain_M, residue_idx, chain_encoding,
            torch.full((len(batch),), 10.0),
        )
        difference = float((low - high).abs().max())
        if difference != 0.0:
            raise RuntimeError(f"{checkpoint}: zero-init difference={difference}")
        models.append(
            {
                "checkpoint": str(checkpoint),
                "sha256": sha256(checkpoint),
                "metadata": metadata,
                "zero_init_ph4_vs_ph10_max_abs_difference": difference,
            }
        )

    report = {
        "status": "ok",
        "data": data_summary,
        "models": models,
        "test_split_evaluated": False,
        "real_cif_ph_only": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

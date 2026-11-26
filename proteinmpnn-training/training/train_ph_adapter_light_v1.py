#!/usr/bin/env python3
"""Lightweight pH-adapter fine-tuning for a frozen 27-state ProteinMPNN."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_utils import featurize
from model_utils_ph_adapter_v1 import (
    ALPHABET,
    STATE_PAIRS,
    PHConditionedProteinMPNN,
    freeze_source_model,
)
from utils_ph_adapter_v1 import (
    PDB_dataset,
    StructureDataset,
    StructureLoader,
    build_training_clusters,
    get_pdbs,
    loader_pdb,
    worker_init_fn,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def layer_count(state: dict[str, torch.Tensor], prefix: str) -> int:
    values = {
        int(key.split(".")[1])
        for key in state
        if key.startswith(prefix + ".") and key.split(".")[1].isdigit()
    }
    if not values:
        raise RuntimeError(f"No {prefix} weights in source checkpoint")
    return max(values) + 1


def load_source_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[PHConditionedProteinMPNN, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    source_state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in source_state.items()
    }
    hidden_dim = int(source_state["W_e.weight"].shape[0])
    vocab = int(source_state["W_s.weight"].shape[0])
    num_letters = int(source_state["W_out.weight"].shape[0])
    if vocab != len(ALPHABET) or num_letters != len(ALPHABET):
        raise RuntimeError(
            f"Expected 27-state source model, got vocab={vocab}, outputs={num_letters}"
        )
    model = PHConditionedProteinMPNN(
        num_letters=num_letters,
        vocab=vocab,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=layer_count(source_state, "encoder_layers"),
        num_decoder_layers=layer_count(source_state, "decoder_layers"),
        k_neighbors=int(checkpoint.get("num_edges", 48)),
        augment_eps=0.0,
        dropout=0.0,
        ph_num_rbf=16,
    )
    incompatible = model.load_state_dict(source_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    forbidden_missing = [
        key for key in incompatible.missing_keys if not key.startswith("ph_")
    ]
    if unexpected or forbidden_missing:
        raise RuntimeError(
            f"Source checkpoint mismatch: missing={forbidden_missing}, unexpected={unexpected}"
        )
    trainable = freeze_source_model(model)
    model.to(device)
    metadata = {
        "source_epoch": checkpoint.get("epoch"),
        "source_step": checkpoint.get("step"),
        "hidden_dim": hidden_dim,
        "num_edges": int(checkpoint.get("num_edges", 48)),
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    return model, metadata


def ph_tensor(batch: list[dict[str, object]], device: torch.device) -> torch.Tensor:
    values = torch.tensor(
        [float(item["crystal_pH"]) for item in batch],
        dtype=torch.float32,
        device=device,
    )
    if not torch.isfinite(values).all() or ((values < 0.0) | (values > 14.0)).any():
        raise ValueError("Batch contains invalid or fallback pH")
    return values


def token_nll(
    sequence: torch.Tensor, log_probs: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    loss = F.nll_loss(
        log_probs.reshape(-1, log_probs.shape[-1]),
        sequence.reshape(-1),
        reduction="none",
    ).reshape_as(sequence)
    return (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def conditional_state_statistics(
    sequence: torch.Tensor,
    log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    loss_sum = torch.zeros((), device=sequence.device)
    count = torch.zeros((), device=sequence.device)
    correct = torch.zeros((), device=sequence.device)
    canonical_count = torch.zeros((), device=sequence.device)
    canonical_correct = torch.zeros((), device=sequence.device)
    alternate_count = torch.zeros((), device=sequence.device)
    alternate_correct = torch.zeros((), device=sequence.device)
    for canonical, alternate in STATE_PAIRS:
        canonical_index = ALPHABET.index(canonical)
        alternate_index = ALPHABET.index(alternate)
        pair_mask = (
            ((sequence == canonical_index) | (sequence == alternate_index))
            & loss_mask.bool()
        )
        if not pair_mask.any():
            continue
        pair_log_probs = torch.stack(
            (log_probs[..., canonical_index], log_probs[..., alternate_index]),
            dim=-1,
        )
        pair_log_probs = pair_log_probs - torch.logsumexp(
            pair_log_probs, dim=-1, keepdim=True
        )
        target = (sequence == alternate_index).long()
        per_position = F.nll_loss(
            pair_log_probs.reshape(-1, 2), target.reshape(-1), reduction="none"
        ).reshape_as(sequence)
        prediction = pair_log_probs.argmax(dim=-1)
        loss_sum += per_position[pair_mask].sum()
        count += pair_mask.sum()
        correct += (prediction[pair_mask] == target[pair_mask]).sum()
        canonical_mask = pair_mask & (target == 0)
        alternate_mask = pair_mask & (target == 1)
        canonical_count += canonical_mask.sum()
        alternate_count += alternate_mask.sum()
        canonical_correct += (prediction[canonical_mask] == 0).sum()
        alternate_correct += (prediction[alternate_mask] == 1).sum()
    return {
        "loss": loss_sum / count.clamp_min(1.0),
        "count": count,
        "correct": correct,
        "canonical_count": canonical_count,
        "canonical_correct": canonical_correct,
        "alternate_count": alternate_count,
        "alternate_correct": alternate_correct,
    }


def collapsed_accuracy(
    sequence: torch.Tensor, log_probs: torch.Tensor, loss_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    collapse = torch.arange(len(ALPHABET), device=sequence.device)
    for canonical, alternate in STATE_PAIRS:
        collapse[ALPHABET.index(alternate)] = ALPHABET.index(canonical)
    prediction = log_probs.argmax(dim=-1)
    correct = (collapse[prediction] == collapse[sequence]) & loss_mask.bool()
    return correct.sum(), loss_mask.sum()


def make_loaders(args: argparse.Namespace) -> tuple[StructureLoader, StructureLoader, dict]:
    params = {
        "LIST": str(args.data_dir / "list.csv"),
        "VAL": str(args.data_dir / "valid_clusters.txt"),
        "TEST": str(args.data_dir / "test_clusters.txt"),
        "DIR": str(args.data_dir),
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,
        "HOMO": 0.70,
    }
    train_clusters, valid_clusters, _test_clusters = build_training_clusters(params, False)
    common = {"batch_size": 1, "shuffle": True, "pin_memory": False, "num_workers": 0}
    train_dataset = PDB_dataset(
        list(train_clusters), loader_pdb, train_clusters, params
    )
    valid_dataset = PDB_dataset(
        list(valid_clusters), loader_pdb, valid_clusters, params
    )
    train_source = torch.utils.data.DataLoader(
        train_dataset, worker_init_fn=worker_init_fn, **common
    )
    valid_source = torch.utils.data.DataLoader(
        valid_dataset, worker_init_fn=worker_init_fn, **common
    )
    print("Loading real-CIF-pH train structures...", flush=True)
    train_records = get_pdbs(
        train_source, 1, args.max_protein_length, args.num_train_examples
    )
    print("Loading real-CIF-pH validation structures...", flush=True)
    valid_records = get_pdbs(
        valid_source, 1, args.max_protein_length, args.num_valid_examples
    )
    if len(train_records) < args.min_train_examples:
        raise RuntimeError(
            f"Only {len(train_records)} real-pH train structures; need {args.min_train_examples}"
        )
    if len(valid_records) < args.min_valid_examples:
        raise RuntimeError(
            f"Only {len(valid_records)} real-pH valid structures; need {args.min_valid_examples}"
        )
    train_loader = StructureLoader(
        StructureDataset(train_records, max_length=args.max_protein_length),
        batch_size=args.batch_size,
    )
    valid_loader = StructureLoader(
        StructureDataset(valid_records, max_length=args.max_protein_length),
        batch_size=args.batch_size,
    )
    summary = {
        "train_structures": len(train_records),
        "valid_structures": len(valid_records),
        "train_batches": len(train_loader),
        "valid_batches": len(valid_loader),
        "train_ph_min": min(float(row["crystal_pH"]) for row in train_records),
        "train_ph_max": max(float(row["crystal_pH"]) for row in train_records),
        "valid_ph_min": min(float(row["crystal_pH"]) for row in valid_records),
        "valid_ph_max": max(float(row["crystal_pH"]) for row in valid_records),
    }
    return train_loader, valid_loader, summary


@torch.no_grad()
def evaluate(
    model: PHConditionedProteinMPNN,
    loader: StructureLoader,
    device: torch.device,
    state_loss_weight: float,
    seed: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "token_loss_sum": 0.0,
        "token_count": 0.0,
        "state_loss_sum": 0.0,
        "state_count": 0.0,
        "state_correct": 0.0,
        "canonical_count": 0.0,
        "canonical_correct": 0.0,
        "alternate_count": 0.0,
        "alternate_correct": 0.0,
        "collapsed_correct": 0.0,
        "collapsed_count": 0.0,
    }
    for batch_index, batch in enumerate(loader):
        if not batch:
            continue
        X, S, mask, _lengths, chain_M, residue_idx, _mask_self, chain_encoding = featurize(
            batch, device
        )
        loss_mask = mask * chain_M
        ph = ph_tensor(batch, device)
        torch.manual_seed(seed + batch_index)
        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding, ph)
        per_token = F.nll_loss(
            log_probs.reshape(-1, log_probs.shape[-1]),
            S.reshape(-1),
            reduction="none",
        ).reshape_as(S)
        state = conditional_state_statistics(S, log_probs, loss_mask)
        collapsed_correct, collapsed_count = collapsed_accuracy(S, log_probs, loss_mask)
        totals["token_loss_sum"] += float((per_token * loss_mask).sum())
        totals["token_count"] += float(loss_mask.sum())
        totals["state_loss_sum"] += float(state["loss"] * state["count"])
        for key in (
            "state_count",
            "state_correct",
            "canonical_count",
            "canonical_correct",
            "alternate_count",
            "alternate_correct",
        ):
            source_key = key.replace("state_", "") if key.startswith("state_") else key
            if key == "state_count":
                value = state["count"]
            elif key == "state_correct":
                value = state["correct"]
            else:
                value = state[source_key]
            totals[key] += float(value)
        totals["collapsed_correct"] += float(collapsed_correct)
        totals["collapsed_count"] += float(collapsed_count)
    token_nll_value = totals["token_loss_sum"] / max(totals["token_count"], 1.0)
    state_nll_value = totals["state_loss_sum"] / max(totals["state_count"], 1.0)
    return {
        "token_nll": token_nll_value,
        "state_nll": state_nll_value,
        "objective": token_nll_value + state_loss_weight * state_nll_value,
        "collapsed_accuracy": totals["collapsed_correct"] / max(totals["collapsed_count"], 1.0),
        "state_accuracy": totals["state_correct"] / max(totals["state_count"], 1.0),
        "canonical_recall": totals["canonical_correct"] / max(totals["canonical_count"], 1.0),
        "alternate_recall": totals["alternate_correct"] / max(totals["alternate_count"], 1.0),
        "tokens": int(totals["token_count"]),
        "state_tokens": int(totals["state_count"]),
        "alternate_tokens": int(totals["alternate_count"]),
    }


@torch.no_grad()
def ph_sensitivity(
    model: PHConditionedProteinMPNN,
    loader: StructureLoader,
    device: torch.device,
    low_ph: float,
    high_ph: float,
    max_batches: int,
    seed: int,
) -> dict[str, dict[str, float | bool]]:
    model.eval()
    accum = {
        canonical: {"low_sum": 0.0, "high_sum": 0.0, "count": 0}
        for canonical, _alternate in STATE_PAIRS
    }
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        X, S, mask, _lengths, chain_M, residue_idx, _mask_self, chain_encoding = featurize(
            batch, device
        )
        loss_mask = (mask * chain_M).bool()
        low = torch.full((len(batch),), low_ph, device=device)
        high = torch.full((len(batch),), high_ph, device=device)
        torch.manual_seed(seed + batch_index)
        low_log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding, low)
        torch.manual_seed(seed + batch_index)
        high_log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding, high)
        for canonical, alternate in STATE_PAIRS:
            upper_index = ALPHABET.index(canonical)
            lower_index = ALPHABET.index(alternate)
            family_mask = (
                ((S == upper_index) | (S == lower_index)) & loss_mask
            )
            if not family_mask.any():
                continue
            low_pair = torch.stack(
                (low_log_probs[..., upper_index], low_log_probs[..., lower_index]), -1
            ).softmax(-1)[..., 1]
            high_pair = torch.stack(
                (high_log_probs[..., upper_index], high_log_probs[..., lower_index]), -1
            ).softmax(-1)[..., 1]
            accum[canonical]["low_sum"] += float(low_pair[family_mask].sum())
            accum[canonical]["high_sum"] += float(high_pair[family_mask].sum())
            accum[canonical]["count"] += int(family_mask.sum())

    report = {}
    low_expected = {"D", "E", "H"}
    for canonical, values in accum.items():
        count = int(values["count"])
        low_mean = values["low_sum"] / max(count, 1)
        high_mean = values["high_sum"] / max(count, 1)
        delta = high_mean - low_mean
        report[canonical] = {
            "count": count,
            "alternate_probability_low_ph": low_mean,
            "alternate_probability_high_ph": high_mean,
            "high_minus_low": delta,
            "expected_direction": "low>high" if canonical in low_expected else "high>low",
            "direction_correct": delta < 0.0 if canonical in low_expected else delta > 0.0,
        }
    return report


def save_checkpoint(
    path: Path,
    model: PHConditionedProteinMPNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    source_checkpoint: Path,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "format": "proteinmpnn_ph_adapter_light_v1",
            "epoch": epoch,
            "source_checkpoint": str(source_checkpoint),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_edges": model.features.top_k,
            "ph_num_rbf": model.ph_num_rbf,
            "state_pairs": STATE_PAIRS,
            "validation": metrics,
            "state_loss_weight": args.state_loss_weight,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-train-examples", type=int, default=20000)
    parser.add_argument("--num-valid-examples", type=int, default=4000)
    parser.add_argument("--min-train-examples", type=int, default=1000)
    parser.add_argument("--min-valid-examples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--max-protein-length", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--rescut", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--sensitivity-batches", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This fine-tune job requires CUDA")
    device = torch.device("cuda:0")
    seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "checkpoints").mkdir()

    model, source_metadata = load_source_model(args.source_checkpoint, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)
    train_loader, valid_loader, data_summary = make_loaders(args)

    # Prove the zero-init adapter preserves the source model and is pH-invariant.
    first_valid_batch = next(iter(valid_loader))
    X, S, mask, _lengths, chain_M, residue_idx, _mask_self, chain_encoding = featurize(
        first_valid_batch, device
    )
    torch.manual_seed(args.seed)
    initial_low = model(
        X, S, mask, chain_M, residue_idx, chain_encoding,
        torch.full((len(first_valid_batch),), 4.0, device=device),
    )
    torch.manual_seed(args.seed)
    initial_high = model(
        X, S, mask, chain_M, residue_idx, chain_encoding,
        torch.full((len(first_valid_batch),), 10.0, device=device),
    )
    initial_max_difference = float((initial_low - initial_high).abs().max())
    if initial_max_difference != 0.0:
        raise RuntimeError(
            f"Zero-init adapter changed source output across pH: {initial_max_difference}"
        )

    report: dict[str, object] = {
        "status": "running",
        "model_name": args.model_name,
        "source_checkpoint": str(args.source_checkpoint),
        "source_metadata": source_metadata,
        "data": data_summary,
        "test_split_read": False,
        "initial_ph4_vs_ph10_max_abs_difference": initial_max_difference,
        "configuration": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "state_loss_weight": args.state_loss_weight,
            "mixed_precision": args.mixed_precision,
            "seed": args.seed,
            "base_frozen": True,
        },
        "history": [],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"preflight": report}, indent=2), flush=True)

    best_objective = math.inf
    best_epoch = None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_objective = 0.0
        running_batches = 0
        for batch in train_loader:
            if not batch:
                continue
            X, S, mask, _lengths, chain_M, residue_idx, _mask_self, chain_encoding = featurize(
                batch, device
            )
            loss_mask = mask * chain_M
            ph = ph_tensor(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.mixed_precision):
                log_probs = model(
                    X, S, mask, chain_M, residue_idx, chain_encoding, ph
                )
                main_loss = token_nll(S, log_probs, loss_mask)
                state = conditional_state_statistics(S, log_probs, loss_mask)
                objective = main_loss + args.state_loss_weight * state["loss"]
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            running_objective += float(objective.detach())
            running_batches += 1

        validation = evaluate(
            model,
            valid_loader,
            device,
            args.state_loss_weight,
            args.seed + epoch * 10000,
        )
        epoch_report = {
            "epoch": epoch,
            "mean_train_objective": running_objective / max(running_batches, 1),
            "train_batches": running_batches,
            "validation": validation,
            "elapsed_seconds": time.time() - start,
        }
        report["history"].append(epoch_report)
        checkpoint_path = args.output_dir / "checkpoints" / f"epoch_{epoch}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            args.source_checkpoint,
            validation,
            args,
        )
        if validation["objective"] < best_objective:
            best_objective = validation["objective"]
            best_epoch = epoch
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                args.source_checkpoint,
                validation,
                args,
            )
        report["best_epoch"] = best_epoch
        report["best_validation_objective"] = best_objective
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(epoch_report, indent=2), flush=True)

    best_checkpoint = torch.load(
        args.output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    report["ph_sensitivity"] = ph_sensitivity(
        model,
        valid_loader,
        device,
        low_ph=4.0,
        high_ph=10.0,
        max_batches=args.sensitivity_batches,
        seed=args.seed + 900000,
    )
    report["status"] = "ok"
    report["elapsed_seconds"] = time.time() - start
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"completed": report}, indent=2), flush=True)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Create inference-only public checkpoints with stable model names."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def layer_count(state: dict[str, torch.Tensor], prefix: str) -> int:
    indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith(prefix + ".") and key.split(".")[1].isdigit()
    }
    if not indices:
        raise ValueError(f"No layers found for {prefix}")
    return max(indices) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True, choices=("exp_063", "exp_082", "exp_085"))
    args = parser.parse_args()

    source = torch.load(args.input, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("module."): value.detach().cpu()
        for key, value in source["model_state_dict"].items()
    }
    hidden_dim = int(state["W_e.weight"].shape[0])
    vocab = int(state["W_s.weight"].shape[0])
    num_letters = int(state["W_out.weight"].shape[0])
    checkpoint = {
        "format": "proteinmpnn_ph_v1",
        "model_name": args.name,
        "model_state_dict": state,
        "architecture": {
            "alphabet": "ACDEFGHIKLMNPQRSTVWYcdhekrX",
            "hidden_dim": hidden_dim,
            "num_letters": num_letters,
            "vocab": vocab,
            "num_encoder_layers": layer_count(state, "encoder_layers"),
            "num_decoder_layers": layer_count(state, "decoder_layers"),
            "num_edges": 48,
            "ph_num_rbf": 16,
        },
        "interface": {
            "requires_ph": True,
            "ph_type": "continuous_scalar",
            "accepted_ph_range": [0.0, 14.0],
            "recommended_ph_range": [2.0, 11.0],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

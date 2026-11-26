#!/usr/bin/env python3
"""Validate and combine the three pH-adapter fine-tuning reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    combined = {"status": "ok", "models": [], "test_split_evaluated": False}
    for path in args.report:
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            raise RuntimeError(f"Incomplete training report: {path}")
        if payload.get("test_split_read") is not False:
            raise RuntimeError(f"Test-split invariant failed: {path}")
        combined["models"].append(
            {
                "model_name": payload["model_name"],
                "source_checkpoint": payload["source_checkpoint"],
                "source_metadata": payload["source_metadata"],
                "data": payload["data"],
                "best_epoch": payload["best_epoch"],
                "best_validation_objective": payload["best_validation_objective"],
                "best_validation": payload["history"][payload["best_epoch"] - 1]["validation"],
                "ph_sensitivity": payload["ph_sensitivity"],
                "output": str(path.parent / "best.pt"),
            }
        )
    if len(combined["models"]) != 3:
        raise RuntimeError(f"Expected three model reports, got {len(combined['models'])}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=2))
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == "__main__":
    main()

# Training and validation recipe

## Configuration

The portable launcher makes the recorded recipe explicit instead of relying on the trainer's larger default dataset sizes.

| Setting | Launcher value |
| --- | --- |
| Model families | `exp_063`, `exp_082`, `exp_085` |
| pH input | Real finite crystallization pH, range 0–14 |
| Requested train / validation records | 6,000 / 1,200 |
| Minimum train / validation records | 500 / 100 |
| Maximum combined sequence length | 800 |
| Padded residue-token batch budget | 800 |
| Epochs | 100, overridable with `EPOCHS` |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Gradient clipping | 5.0 |
| Conditional-state loss coefficient | 0.25 |
| Mixed precision | Enabled |
| Seed | 20261001 |
| pH diagnostic | pH 4 versus pH 10, at most 16 validation batches |
| Selection | Lowest validation objective |

The structural sequence model is initialized from the named 27-state checkpoint and frozen. `freeze_source_model()` makes only parameters whose names start with `ph_` trainable. pH-RBF centers, width, and the pair-projection matrix are buffers rather than optimized parameters. Dropout and coordinate augmentation are disabled by the training model loader.

The trainer supports one CUDA device per process, addressed as `cuda:0` within its allocation. The new `train_three.sh` runs three processes sequentially. It does not reproduce the site-specific controller's parallel GPU scheduling.

## Direct command

Inside an allocated CUDA environment, an equivalent single-model command is:

```bash
python training/train_ph_adapter_light_v1.py \
  --source-checkpoint "$SOURCE_ROOT/training/exp_proton_085/model_weights/epoch365_step206190.pt" \
  --model-name exp_085 \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_ROOT/exp_085_ph_v1" \
  --epochs 100 \
  --num-train-examples 6000 \
  --num-valid-examples 1200 \
  --min-train-examples 500 \
  --min-valid-examples 100 \
  --batch-size 800 \
  --max-protein-length 800 \
  --learning-rate 0.001 \
  --state-loss-weight 0.25 \
  --sensitivity-batches 16 \
  --mixed-precision
```

The output directory must not already exist. Although checkpoints include optimizer state, this entry point has no resume argument. A new invocation is not a continuation of an interrupted run.

## Outputs

```text
RUN_ROOT/exp_085_ph_v1/
  report.json
  best.pt
  checkpoints/
    epoch_1.pt
    epoch_2.pt
    ...
```

`report.json` records source metadata, actual dataset counts and pH ranges, configuration, per-epoch validation, selected epoch, and the final pH-sensitivity diagnostic. `status="ok"` is written only after training and that diagnostic finish. Each checkpoint stores the full model state, optimizer state, source checkpoint reference, epoch, neighborhood count, pH configuration, and validation metrics. The trainer does not write a separate adapter-only checkpoint.

## Metric definitions

`token_nll` is the masked mean NLL over all 27 tokens. `state_nll` is the mean NLL within the six two-state families, conditioned on the reference residue family. `objective` combines them with the configured coefficient.

`collapsed_accuracy` compares the top-1 27-token prediction to the reference after mapping lowercase states to their canonical amino acid. It does not first sum probabilities over each amino-acid family. `state_accuracy`, `canonical_recall`, and `alternate_recall` describe the within-family binary decision. These conditional state metrics should not be presented as unconditional sequence-design accuracy.

The pH diagnostic holds the input structures and reference sequences fixed while changing the scalar pH. It reports conditional alternative-state probabilities on positions belonging to each reference family. Expected direction is lower alternative-state probability at high pH for D/E/H and higher alternative-state probability at high pH for C/K/R. This is a model diagnostic, not experimental titration data.

Evaluation uses teacher-forced sequence context and stochastic decoding order. The trainer sets PyTorch seeds per batch; featurization also uses Python chain shuffling and can modify chain-mask lists. The preserved comparison scripts reuse a materialized validation batch list, but do not reset all featurization randomness between model calls. Their results should not be described as a guarantee of bitwise-identical chain ordering. Exact bitwise reproduction across platforms is not claimed.

## Checks and auxiliary scripts

`training/test_ph_adapter_v1.py` contains four small architecture tests: zero-initialized pH invariance, freezing policy, gradient flow, and invalid-pH rejection. `training/preflight_ph_adapter_v1.py` exercises the real data loader and source checkpoints and writes a JSON report. These scripts execute scientific code and should be run in your allocated compute environment.

The two comparison utilities require exactly three aligned groups of source checkpoints and trained checkpoints/directories. `compare_source_vs_adapter_v1.py` evaluates selected checkpoints; `compare_stock_trajectory_v1.py` evaluates its configured epoch snapshots plus a requested latest epoch. `summarize_three_ph_adapters_v1.py` combines three completed `report.json` files without rerunning the model.

## Checkpoint export

`tools/export_proteinmpnn_weights.py` writes format `proteinmpnn_ph_v1` with model tensors, architecture, and the explicit-pH inference interface. It drops optimizer state and local training paths. Its export metadata assumes 48 neighbors and 16 pH radial-basis features, matching the recorded recipe. Check these assumptions before exporting a different architecture.

The inference implementation is provided in the [companion release](https://github.com/DimaBosov3/proteinmpnn-hbdesigner-ph-models/releases/tag/v1.0.0). This training repository does not duplicate its weights or HBDesigner code.

## Publication checks

The Python files were copied from the existing project sources and reviewed statically for this publication. No training, tensor evaluation, data loading, scientific test execution, or checkpoint regeneration was performed as part of repository assembly. The available dependency constraints are not a complete historical environment lockfile. Validation-based model selection does not replace an independent held-out test.

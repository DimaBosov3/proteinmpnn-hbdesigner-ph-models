# pH-conditioned ProteinMPNN training

Training code for ProteinMPNN models that predict amino-acid identity and protonation-state tokens from a protein backbone and a continuous pH input. We train the pH-conditioned models using crystallization pH recorded in structure metadata. Each optimization batch supplies both backbone coordinates and pH to the network.

The repository contains the training implementation for the `exp_063`, `exp_082`, and `exp_085` model families, data loading, validation utilities, and checkpoint export. The companion [model release](https://github.com/DimaBosov3/proteinmpnn-hbdesigner-ph-models/releases/tag/v1.0.0) provides packaged inference code and weights.

## Model and training objective

The sequence alphabet is **27 tokens**, in this exact order:

```text
ACDEFGHIKLMNPQRSTVWYcdhekrX
```

It comprises 20 canonical amino acids, six alternative chemical states, and `X`. The state pairs are `C/c`, `D/d`, `E/e`, `H/h`, `K/k`, and `R/r`.

| Alternative token | State represented |
| --- | --- |
| `c` | Deprotonated cysteine |
| `d` | Protonated aspartate |
| `e` | Protonated glutamate |
| `h` | Positively charged, doubly protonated histidine |
| `k` | Neutral lysine |
| `r` | Neutral arginine |

pH is encoded as 16 radial-basis features over 0–14 and one normalized scalar. Two trainable networks use these features: one supplies an additive residual to the encoded residue representations; the other shifts the conditional odds within the six state pairs. The latter adds equal and opposite logit corrections to the two members of each pair.

Training initializes the structural sequence model from the corresponding 27-state source checkpoint. Its parameters remain frozen; optimization updates the `ph_*` networks. For the recorded 128-dimensional models, these networks contain **19,590 trainable parameters**. Their final layers are zero-initialized. The complete model receives pH during training and inference.

The loss is:

```text
27-token negative log-likelihood + 0.25 × conditional state negative log-likelihood
```

Both terms use the valid-coordinate/design-chain mask. The conditional term renormalizes each relevant canonical/alternative pair and evaluates the observed state within that pair. AdamW optimizes the trainable parameters. Selection uses the lowest validation objective; the test split is excluded from evaluation.

## Where the files are

| Path | Purpose |
| --- | --- |
| [`training/train_ph_adapter_light_v1.py`](training/train_ph_adapter_light_v1.py) | CUDA training entry point, objective, validation, checkpoint selection, pH diagnostics |
| [`training/model_utils_ph_adapter_v1.py`](training/model_utils_ph_adapter_v1.py) | pH-conditioned architecture and parameter-freezing policy |
| [`training/model_utils.py`](training/model_utils.py) | 27-state ProteinMPNN backbone model and featurization |
| [`training/utils_ph_adapter_v1.py`](training/utils_ph_adapter_v1.py) | Cluster splits, serialized structure loading, real-CIF-pH filtering, token-budget batches |
| [`training/preflight_ph_adapter_v1.py`](training/preflight_ph_adapter_v1.py) | Data/checkpoint preflight and zero-initialization checks |
| [`training/test_ph_adapter_v1.py`](training/test_ph_adapter_v1.py) | Architecture checks for gradients, freezing, pH validity, and initialization |
| [`training/compare_source_vs_adapter_v1.py`](training/compare_source_vs_adapter_v1.py) | Validation comparison of three source/trained model pairs |
| [`training/compare_stock_trajectory_v1.py`](training/compare_stock_trajectory_v1.py) | Validation snapshots at selected training epochs |
| [`training/summarize_three_ph_adapters_v1.py`](training/summarize_three_ph_adapters_v1.py) | Combine three completed training reports |
| [`tools/export_proteinmpnn_weights.py`](tools/export_proteinmpnn_weights.py) | Export a training checkpoint to the public inference format |
| [`scripts/train_one.sh`](scripts/train_one.sh) | Portable launcher for one model using the recorded training recipe |
| [`scripts/train_three.sh`](scripts/train_three.sh) | Run all three families sequentially and combine reports |
| [`examples/train_single_gpu.sbatch`](examples/train_single_gpu.sbatch) | Example Slurm submission for one GPU |
| [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) | Required files, metadata, state labels, and splits |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Configuration, commands, outputs, and metric definitions |
| [`docs/SOURCE_FILES.md`](docs/SOURCE_FILES.md) | Source-file provenance and launcher mapping |

The Python implementations retain their source filenames and contents. Portable shell launchers and English documentation are provided here separately from the site-specific execution setup.

## Requirements and inputs

Use Python 3.10+ and a CUDA-compatible PyTorch installation supporting `torch.amp.GradScaler("cuda", ...)`:

```bash
python -m pip install -r requirements.txt
```

Choose the appropriate PyTorch/CUDA build for your compute environment. The requirements list is not a locked environment.

Training requires a prepared structure dataset and the three **source training checkpoints**. Their expected paths under `SOURCE_ROOT` are:

```text
training/exp_proton_063/model_weights/epoch_last.pt
training/exp_proton_082/model_weights/epoch375_step218666.pt
training/exp_proton_085/model_weights/epoch365_step206190.pt
```

The dataset must include the 27-state sequence labels, assembly coordinates, original cluster lists, finite `crystal_pH`, and `ph_from_cif=True`. Raw PDB files alone are not the training input. See [the data contract](docs/DATA_FORMAT.md).

Source training checkpoints, datasets, and run artifacts are not bundled in this code repository. The public inference checkpoint format and the source training checkpoints have different roles; do not substitute one for the other when reproducing source initialization.

## Run training

Set these paths in your allocated compute environment:

```bash
export DATA_DIR=/path/to/prepared_dataset
export SOURCE_ROOT=/path/to/ProteinMPNN-new-proton
export RUN_ROOT=/path/to/new_run
export PYTHON=python

bash scripts/train_one.sh exp_085
```

The recorded recipe requests up to 6,000 training structures and 1,200 validation structures, 100 epochs, a batch token budget of 800, maximum combined sequence length 800, learning rate 0.001, state-loss weight 0.25, and mixed precision. Actual loaded counts are written to `report.json`.

To run all three model families sequentially on one allocated GPU:

```bash
bash scripts/train_three.sh
```

For Slurm, export the variables above, set `MODEL_NAME=exp_085` if desired, then submit from the repository root:

```bash
sbatch --export=ALL examples/train_single_gpu.sbatch
```

The Slurm example requests one GPU, two CPUs, 32 GB RAM, and two hours. Adjust partition/account and the training workload to your allocation. The trainer requires a new output directory and does not implement automatic resume.

## Checkpoints and inference

Each model run writes `report.json`, `best.pt`, and `checkpoints/epoch_<n>.pt`. Export the selected checkpoint with:

```bash
python tools/export_proteinmpnn_weights.py \
  --input "$RUN_ROOT/exp_085_ph_v1/best.pt" \
  --output exports/exp_085.pt \
  --name exp_085
```

The exporter is configured for the recorded 48-neighbor, 16-pH-RBF architecture. Use the [companion release's](https://github.com/DimaBosov3/proteinmpnn-hbdesigner-ph-models/releases/tag/v1.0.0) `proteinmpnn/run_proteinmpnn.py` with the exported checkpoint and an explicit `--ph` value. Inference produces state-preserving FASTA and canonical-amino-acid FASTA.

## Interpretation and validation scope

Crystallization pH is observational supervision. The training labels and pH-sensitivity diagnostics are not residue-specific experimental pKa measurements or evidence of physical stability or function. Validation selection and diagnostics do not constitute a held-out test evaluation.

This publication preserves existing Python source. No training, model evaluation, or scientific tests were rerun to assemble the repository. See [training details](docs/TRAINING.md) for evaluation behavior and limitations.

## Attribution

The model core and data utilities derive from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN). The upstream MIT notice is preserved in [LICENSE](LICENSE). Project-specific code implements the 27-state vocabulary, pH conditioning, conditional-state objective, and associated training utilities.

# Source files and execution entry points

This repository publishes the existing pH-conditioned ProteinMPNN training implementation with English usage documentation. The Python files retain their original filenames and contents; the directory organization separates training from public-checkpoint export.

## Python source mapping

| Published file | Project source |
| --- | --- |
| `training/model_utils.py` | `proteinmpnn-ph-adapter-light-v1/model_utils.py` |
| `training/model_utils_ph_adapter_v1.py` | `proteinmpnn-ph-adapter-light-v1/model_utils_ph_adapter_v1.py` |
| `training/utils_ph_adapter_v1.py` | `proteinmpnn-ph-adapter-light-v1/utils_ph_adapter_v1.py` |
| `training/train_ph_adapter_light_v1.py` | `proteinmpnn-ph-adapter-light-v1/train_ph_adapter_light_v1.py` |
| `training/preflight_ph_adapter_v1.py` | `proteinmpnn-ph-adapter-light-v1/preflight_ph_adapter_v1.py` |
| `training/test_ph_adapter_v1.py` | `proteinmpnn-ph-adapter-light-v1/test_ph_adapter_v1.py` |
| `training/compare_source_vs_adapter_v1.py` | `proteinmpnn-ph-adapter-light-v1/compare_source_vs_adapter_v1.py` |
| `training/compare_stock_trajectory_v1.py` | `proteinmpnn-ph-adapter-light-v1/compare_stock_trajectory_v1.py` |
| `training/summarize_three_ph_adapters_v1.py` | `proteinmpnn-ph-adapter-light-v1/summarize_three_ph_adapters_v1.py` |
| `tools/export_proteinmpnn_weights.py` | `model-release-builder/export_proteinmpnn_weights.py` |

The source locations identify the project folders used to assemble this publication; they are not additional directories required under this repository.

## Training execution

The project controller `run_three_on_ripper2_v1.sh` invoked `train_ph_adapter_light_v1.py` for each of `exp_063`, `exp_082`, and `exp_085`, using the 6,000/1,200-record recipe documented in [TRAINING.md](TRAINING.md). It then invoked `summarize_three_ph_adapters_v1.py` to combine the reports.

This repository's `scripts/train_one.sh`, `scripts/train_three.sh`, and Slurm example are new portable wrappers around that training entry point. They replace machine-specific absolute paths, CPU affinity, device probing, and concurrent three-GPU scheduling with explicit path variables and single-GPU execution. They are not copies of the site-specific controller.

The separate `train_three_ph_adapters_light_v1.sbatch` source file used different sample-count settings; it is not the configuration represented by the portable launchers here.

## Source checkpoint references

| Family | Source checkpoint relative to the source project |
| --- | --- |
| `exp_063` | `training/exp_proton_063/model_weights/epoch_last.pt` |
| `exp_082` | `training/exp_proton_082/model_weights/epoch375_step218666.pt` |
| `exp_085` | `training/exp_proton_085/model_weights/epoch365_step206190.pt` |

These identify initialization inputs, not files bundled in the repository. Published inference checkpoints are exported products of the pH-conditioned model and use the companion release's loader.

## Scope

Included: training/model/data Python source, auxiliary validation scripts, exporter, portable launch examples, and documentation. Dataset preparation, raw molecular data, checkpoints, private machine configuration, and run artifacts are outside this source publication. This repository is specifically for ProteinMPNN; the [companion release](https://github.com/DimaBosov3/proteinmpnn-hbdesigner-ph-models) hosts the combined inference package.

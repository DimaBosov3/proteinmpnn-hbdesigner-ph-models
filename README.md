# ProteinMPNN + HBDesigner: inference package

This package contains three ProteinMPNN models, `exp_063`, `exp_082`, and `exp_085`, together with pH-aware HBDesigner. All inference commands take pH as a continuous input.

## Package contents

- `proteinmpnn/weights/exp_063.pt`
- `proteinmpnn/weights/exp_082.pt`
- `proteinmpnn/weights/exp_085.pt`
- `proteinmpnn/run_proteinmpnn.py`
- `proteinmpnn/parse_multiple_chains.py`
- `hbdesigner/weights/hbd_best.pt`
- `hbdesigner/config/hbd_best.yaml`
- HBDesigner inference source code and the required packing weights

## ProteinMPNN

Requirements: Python 3.10+, PyTorch, and NumPy.

Place one or more PDB files in the `inputs/` directory and convert them to JSONL:

```bash
python proteinmpnn/parse_multiple_chains.py \
  --input_path inputs \
  --output_path parsed.jsonl
```

Run `exp_063` at pH 6.5:

```bash
python proteinmpnn/run_proteinmpnn.py \
  --jsonl_path parsed.jsonl \
  --model proteinmpnn/weights/exp_063.pt \
  --ph 6.5 \
  --out_folder outputs/exp_063_ph65 \
  --num_seq_per_target 16 \
  --sampling_temp 0.1
```

For the other two models, replace the checkpoint filename with `exp_082.pt` or `exp_085.pt`. To design only selected chains, add an option such as `--chains_to_design A,C`. If this option is omitted, all protein chains are designed.

The model uses 27 states: 20 standard amino acids, six alternative states (`c/d/e/h/k/r`), and `X`. Each run produces two FASTA outputs:

- `*_states.fasta` preserves the predicted states;
- `*_20aa.fasta` maps lowercase states to the standard amino-acid symbols `C/D/E/H/K/R`.

The `--ph` argument is required. The accepted range is 0–14; the recommended operating range is 2–11.

## HBDesigner

HBDesigner requires a separate environment with PyTorch, PyTorch Geometric, PyRosetta, and the dependencies listed in `hbdesigner/env_gpu.yaml`. Install the code with:

```bash
python -m pip install -e hbdesigner
```

Example run at pH 6.5:

```bash
run_hbdesigner \
  --pdb inputs/example.pdb \
  --out_dir outputs/hbd_ph65 \
  --design_model_ckpt hbdesigner/weights/hbd_best.pt \
  --design_model_config hbdesigner/config/hbd_best.yaml \
  --packing_model_ckpt hbdesigner/model_weights/pack.pt \
  --ph 6.5 \
  --predicted_protonation_mode free \
  --rosetta_acid_tautomer P1 \
  --n_samples 100 \
  --top_k 5 \
  --n_res 3
```

`--ph` specifies the target pH for the neural network. The `--predicted_protonation_mode free` option passes the predicted state to Rosetta as an initial assignment and allows repacking. For ASP/GLU, the model distinguishes protonated and deprotonated states but does not distinguish the P1/P2 oxygen tautomers. Set this policy explicitly with `--rosetta_acid_tautomer P1` or `P2`.

## Download the package

The complete archive, `ProteinMPNN_HBDesigner_pH_v1.tar.gz`, is available in [GitHub Releases](https://github.com/DimaBosov3/proteinmpnn-hbdesigner-ph-models/releases/tag/v1.0.0). The package paths above refer to files inside the extracted archive.

Verify the archive:

```bash
sha256sum ProteinMPNN_HBDesigner_pH_v1.tar.gz
# 0d921c36ea24db2f438e2791a9135cc3c7454f38d17e9c3e179ee7a4a7ba8f66
```

After extracting the archive, verify its contents:

```bash
sha256sum -c SHA256SUMS
```

## ProteinMPNN training source

The standalone ProteinMPNN training project is available in [`proteinmpnn-training/`](proteinmpnn-training/README.md). It includes pH-conditioned training code for `exp_063`, `exp_082`, and `exp_085`, model and data-loading modules, checkpoint export, portable launch examples, and English documentation describing the training recipe and file locations.

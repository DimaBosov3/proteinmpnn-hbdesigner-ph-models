# Training data contract

The training loader consumes a prepared, serialized ProteinMPNN structure dataset. Dataset preparation and residue-state assignment are external inputs to this repository; this code does not derive labels from raw PDB/mmCIF files or run a pKa calculator.

## Directory layout

```text
DATA_DIR/
  list.csv
  valid_clusters.txt
  test_clusters.txt
  pdb/
    ab/
      1abc.pt
      1abc_A.pt
      1abc_B.pt
```

The example identifiers illustrate the layout only. Metadata is stored at `pdb/<pdb_id[1:3]>/<pdb_id>.pt`; chain records append `_<chain_id>.pt`. Chain identifiers may contain multiple characters.

`list.csv` has a header. The loader reads these zero-based columns:

| Column | Meaning |
| --- | --- |
| 0 | Target identifier, `<pdb_id>_<chain_id>` |
| 1 | Date parseable by `dateutil.parser` |
| 2 | Resolution, parsed as a float |
| 3 | Auxiliary sequence identifier retained in the cluster record |
| 4 | Integer cluster ID |

The trainer filters at resolution 3.5 Å by default and uses its configured date cutoff. The split files contain one integer cluster ID per line. Validation IDs take precedence in the loader; test IDs are excluded from training. The cluster files, including `test_clusters.txt`, are read to construct the split. Test structures are not loaded for evaluation and no test metrics are calculated. The historical report field `test_split_read=False` refers to this evaluation policy, not to opening the split-ID file.

## Serialized records

Per-structure metadata supplies:

| Key | Purpose |
| --- | --- |
| `crystal_pH` | Scalar crystallization pH |
| `ph_from_cif` | Boolean provenance flag for the recorded CIF pH |
| `asmb_ids` | Biological assembly identifiers |
| `asmb_chains` | Comma-separated chain lists for assembly operations |
| `asmb_xform<assembly_id>` | Assembly transformations when available |

Each chain record supplies `seq` and `xyz`. `seq` uses the exact alphabet `ACDEFGHIKLMNPQRSTVWYcdhekrX`. Coordinates use the prepared dataset's atom layout, conventionally `[L, 14, 3]`, with N, CA, C, and O in slots 0–3. The loader uses observed CA positions and passes backbone coordinates to featurization. Missing assembly metadata has fallbacks in the preserved loader; complete preparation metadata should be retained for reproduction.

Only examples with `ph_from_cif=True`, finite pH in `[0, 14]`, at least four retained residues, and an allowed combined sequence length are admitted. Missing pH is not replaced with a nominal value. The data loader trusts the supplied state labels and provenance flag; it does not independently re-read the CIF files.

## Sampling and masks

One structure/chain record is sampled per cluster in a shuffled pass, with random assembly selection. Accepted records are materialized once for the run and reused across epochs. Requested example counts are upper limits; filtering may reduce the actual count. Minimum counts cause an error if insufficient real-pH records are available. There is no class weighting or state oversampling.

The loader marks the target chain for prediction and assembly chains as context; the featurizer can also mark identical-sequence chains for prediction. The loss mask is `coordinate_mask * chain_M`. The batch-size argument is a padded residue-token budget, not a number of proteins.

## Availability

This repository provides the loading contract and code. It does not distribute the prepared dataset, source training checkpoints, or the dataset-preparation pipeline. Exact reproduction requires the same input records, cluster lists, labels, source checkpoints, and a compatible compute environment.

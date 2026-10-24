# CMCA Part Encoder for CAPP-GPT

Reference implementation for *Cross-Modal Contrastive Alignment of B-Rep
Graphs with Manufacturing Feature Prototypes for CAPP-GPT* (NAMRC 55, 2027).

The encoder maps the faces of a boundary-representation (B-Rep) part into a
shared face-label embedding space. A Graph Transformer processes the attributed
face graph, while a label tower supplies one learned prototype per manufacturing
feature class. Faces are assigned the class of their nearest prototype.

The paper compares four fixed attention-bias inputs:

- attribute similarity;
- real shared-edge face adjacency;
- their unnormalized sum; and
- a shuffled-adjacency control that preserves graph statistics while attaching
  the graph to the wrong faces.

The shuffled control is read against adjacency. It separates a gain from topology from a gain from sparsity. Adding a class extends the label tower. The geometry tower stays as it is. The paper does not evaluate unseen classes.

## Repository contents

```text
configs/              one YAML file per experiment
datasets/             corpus provenance, checksums, saved splits, norm statistics
results/              reported per-seed metrics and aggregate tables
runners/              tier runner, gates, aggregation, and figure scripts
src/cmca/extract/      STEP-to-feature extractors
src/                   datasets, models, losses, training, evaluation, utilities
tests/                 regression and reproducibility tests
```

The raw CADSynth and MFInstSeg files stay outside this repository.
The three zip archives are [10.5281/zenodo.23602506](https://doi.org/10.5281/zenodo.23602506).
See [`datasets/README.md`](datasets/README.md) for checksums, the
expected local directory layout, and extraction instructions.

## Environments

Training and evaluation use Python 3.12.11 and PyTorch 2.8.0+cu128. The
reported batch ran on one NVIDIA Tesla T4 with CUDA 12.8 and cuDNN 9.10.2.

STEP extraction needs `pythonocc-core`. That package uses a separate Python 3.10 environment, so two environment files are provided.

| File | Purpose |
|---|---|
| `env-model.yml` | pinned direct dependencies for training and evaluation |
| `env-lock-studio.txt` | complete 188-package lock from the run machine |
| `env-extraction.yml` | STEP extraction with `pythonocc-core` |
| `requirements.txt` | version-floor fallback for the model environment |

Create the model and extraction environments:

```bash
conda env create -f env-model.yml
conda env create -f env-extraction.yml
```

For the exact model environment used on the run machine:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
  -r env-lock-studio.txt
```

## Data and splits

Both corpora use one saved model-level split (70/10/20, seed 42):

| Corpus | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| CADSynth | 4,244 | 606 | 1,212 | 6,062 |
| MFInstSeg | 4,116 | 588 | 1,176 | 5,880 |

The 70/10/20 split files and the train-only norm files are in `datasets/splits/`.
Test identifiers match the `*_80_20_seed42.json` files in that folder.
No config loads those files, or the `*_SUPERSEDED_80_20.json` norm files.
Normalization statistics are fitted on the 70% training partition only.

## Reported results

The repository contains the 85 runs used for tiers T1 through T6: 17 configurations at seeds 0, 1, 2, 3, and 4. Every run stops early on validation loss, with patience 10 and a cap of 60 epochs. The saved checkpoint is the one with the best validation loss. Evaluation runs once on the held-out test partition.

Each run directory contains its resolved configuration and evaluation metrics.
The aggregate files used by the paper are:

| Paper experiment | Aggregate artifact |
|---|---|
| CADSynth bias ablation | `results/summary_cadsynth.{md,json}` |
| MFInstSeg bias ablation | `results/summary_mfinstseg.{md,json}` |
| Encoder-objective comparison | `results/summary_T3_cadsynth.{md,json}` |
| Weighted ResNet-18 baseline | `results/summary_T4_cadsynth.{md,json}` |
| Unified three-head model | `results/summary_T5_mfinstseg.{md,json}` |
| Class-weighting robustness | `results/summary_T6_cadsynth.{md,json}` |

Accuracy, macro-F1, and mean intersection over union are reported where
applicable. Instance results additionally report ROC-AUC and validation-threshold
pair F1.

## Reproduce the paper experiments

Run all commands from the repository root. `runners/run_all.py` is the only
entry point for newly generated paper numbers.

List or inspect a tier before training:

```bash
python runners/run_all.py --list
python runners/run_all.py --tier T1 --seeds 0 1 2 3 4 --dry-run
```

Run and validate one tier:

```bash
python runners/run_all.py --tier T1 --seeds 0 1 2 3 4
python runners/check_tier.py --tier T1
```

Replace `T1` with `T2` through `T6` to reproduce the remaining paper tiers.
Tier T7 is follow-up scouting and is not part of this paper.

Regenerate the aggregate tables from completed per-seed metrics:

```bash
# T1: CADSynth bias ablation
python runners/aggregate_seeds.py --dataset cadsynth \
  --arms similarity adjacency both shuffled \
  --out results/summary_cadsynth.md

# T2: MFInstSeg bias ablation
python runners/aggregate_seeds.py --dataset mfinstseg \
  --arms similarity adjacency both shuffled \
  --out results/summary_mfinstseg.md

# T3: encoder by objective
python runners/aggregate_seeds.py --dataset cadsynth \
  --arms cadsynth_ce_head resnet_cadsynth_cmca \
         resnet_cadsynth_ce cadsynth_bias_both \
  --out results/summary_T3_cadsynth.md

# T4: weighted ResNet-18 cross-entropy baseline
python runners/aggregate_seeds.py --dataset cadsynth \
  --arms resnet_cadsynth_ce_weighted \
  --out results/summary_T4_cadsynth.md

# T5: unified semantic, bottom-face, and instance model
python runners/aggregate_seeds.py --dataset mfinstseg \
  --arms mfinstseg_unified \
  --out results/summary_T5_mfinstseg.md

# T6: weighted CADSynth bias mirror
python runners/aggregate_seeds.py --dataset cadsynth \
  --arms cadsynth_bias_similarity_weighted \
         cadsynth_bias_adjacency_weighted \
         cadsynth_bias_both_weighted \
         cadsynth_bias_shuffled_weighted \
  --out results/summary_T6_cadsynth.md
```

The aggregate tool writes a JSON file beside each Markdown file. It refuses a table that fails its provenance and consistency checks.

## Tests

The test suite uses synthetic fixtures. It runs without the raw corpora and without a GPU:

```bash
pytest -q
```

The passing-test count can change as tests are added. The release gate is a zero
exit status with no failures or errors. Extractor tests require the extraction
environment:

```bash
conda run -n pyoccenv python -m pytest tests/test_extractor.py -q
```

## Citation

Citation metadata is in [`CITATION.cff`](CITATION.cff). The extracted feature
archives are [10.5281/zenodo.23602506](https://doi.org/10.5281/zenodo.23602506).
The software release version, release date, preferred paper citation, and
software Zenodo DOI will be added when the submission release is tagged.

## License

This software is released under the MIT License. See [`LICENSE`](LICENSE).


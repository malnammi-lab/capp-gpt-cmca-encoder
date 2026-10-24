# Dataset manifest

The three source archives are not in this repository. They are the Zenodo record [10.5281/zenodo.23602506](https://doi.org/10.5281/zenodo.23602506). Each archive is listed below by file name and byte size. Unpack them into `../data/cadsynth/`, `../data/mfinstseg/`, and `../data/cadsynth_faces/`. Each experiment config looks for its corpus there. That folder is not part of the repository.

## Feature corpora (v2, re-extracted with real topology)

Both graph corpora were extracted from the source CAD by the extractors in `src/cmca/extract/`. Face adjacency is the shared-edge graph. CADSynth density is 0.203. MFInstSeg density is 0.215.

### CADSynth

| | |
|---|---|
| Archive name | `CadSynth feature.zip` ([10.5281/zenodo.23602506](https://doi.org/10.5281/zenodo.23602506)) |
| Original download name | `extracted_features_v2-20260904T124640Z-1-001.zip` |
| sha256 | `277076f382df4fedf05c3beb252e9173342506398333006096aacf295babc478` |
| Archive size | 105,540,100 bytes |
| Uncompressed | 1,380,541,424 bytes |
| Models | 6,062 (`*_features.json`) |
| Extra file | `raw_feature_stats.json` (per-column statistics; a diagnostic) |
| Extractor | `src/cmca/extract/extract_features_cadsynth.py` |
| Extractor sha256 | `0d046fd7854f678c…` (1,483 lines) |
| Face-adjacency density | 0.203 (range 0.079 – 0.733) |

The source set contains 6,063 STEP files. `00006062.stp` is malformed, and OpenCascade cannot parse it, so 6,062 models are written. The original extractor hit the same file and skipped it silently.

### MFInstSeg

| | |
|---|---|
| Archive name | `MFINSTASEG features.zip` ([10.5281/zenodo.23602506](https://doi.org/10.5281/zenodo.23602506)) |
| Original download name | `extracted_features_with_labels_v2-20260904T124639Z-1-001.zip` |
| sha256 | `8d16e8192e98f45ec8eb13cff7d30cbfac37041f31ff9a893db07eedbff73177` |
| Archive size | 85,397,231 bytes |
| Uncompressed | 1,299,907,599 bytes |
| Models | 5,880 (`*_features.json`) |
| Extra file | `raw_feature_stats.json` |
| Extractor | `src/cmca/extract/extract_features_mfinstaseg.py` |
| Extractor sha256 | `2011d514606c6904…` (1,670 lines) |
| Face-adjacency density | 0.215 (range 0.073 – 0.643) |
| Labels | semantic (25 classes), instance (N×N affinity), bottom-face (binary) |

### CADSynth faces 

This is the per-face view of the same parts as the CADSynth graph corpus, for the ResNet-18 baseline and both ResNet cells of the 2x2 ablation.

| | |
|---|---|
| Archive name | `extracted_faces_v2.zip` ([10.5281/zenodo.23602506](https://doi.org/10.5281/zenodo.23602506)) |
| sha256 | `6fe772cd4fcf3cdd7f77730e6287e2e50a871c7ed1e2e7ca2110ad223ca8a8b5` |
| Archive size | 66,774,500 bytes |
| Uncompressed | 1,095,929,378 bytes |
| Models | 6,062 (`*_faces.json`) |
| Faces | 169,563 |
| Extractor | `src/cmca/extract/extract_faces_cadsynth.py` |
| Extractor sha256 | `b1bbb74797e1c5fa…` (315 lines) |
| Extracted | 18 Sep 2026, `pyoccenv` (python 3.10, pythonocc-core 7.9.0) |
| UV grid | 8x8, 7 channels |
| Normalization | `unitcube`; max observed \|coordinate\| = 1.7321 = sqrt(3), the exact corner of the unit cube |
| Split file | `datasets/splits/cadsynth_70_10_20_seed42.json` |

Both views contain the same 6,062 models and 169,563 faces. Both come from the same 6,063 STEP files through the same `BRepFeatureExtractor`, and both fail on `00006062.stp`. Both rows of the 2x2 ablation use `cadsynth_70_10_20_seed42.json`: 4,244 train, 606 validation, and 1,212 test. The test ids match the earlier 80/20 test list. A gap between the ResNet row and the Graph Transformer row is a gap in method. Both rows see the same parts.

The extraction run wrote 6,042 files, skipped 20 that were already present, and failed on 1 file, `00006062.stp`.

Four models have label lists shorter than their face count, so 6 faces of 169,563 (0.0035%) receive a padded class-0 label: `00006031` (32 labels / 33 faces), `00005948` (33/35), `00004233` (26/28), `00001586` (21/22). The graph corpus uses the same six padded faces.

`extract_faces_cadsynth.py` calls `BRepFeatureExtractor.extract_face_uv_grid` from the feature extractor. Both corpora are built from one implementation, so the ResNet and Graph Transformer rows see the same geometry.

```bash
# needs env-extraction.yml (conda env pyoccenv)
# <faces-dir> is a directory outside this repository.
python src/cmca/extract/extract_faces_cadsynth.py --self-test

python src/cmca/extract/extract_faces_cadsynth.py \
    --step-dir  <CADSynth STEP files> \
    --label-dir <CADSynth labels> \
    --out-dir   <faces-dir> \
    --skip-existing

python src/cmca/extract/extract_faces_cadsynth.py \
    --check-json <faces-dir>
```

`--check-json` checks adjacency, normals, and the trim mask. A bad corpus exits with a nonzero status.

## Per-model JSON schema

Schema `2.0` (CADSynth) / `2.0-mfinstaseg` (MFInstSeg) is recorded in every file, alongside `extractor`, `extracted_utc`, `normalize`, `unit_scale`, and `unit_centre`. Any model can be traced to the run that produced it.

| Key | Shape | Notes |
|---|---|---|
| `faces_uv` | (N, 7, 8, 8) | x, y, z, nx, ny, nz, validity mask; `uv_grid_size` = [8, 8] |
| `faces_attr` | (N, 4) | divided-and-clipped view |
| `faces_attr_raw` | (N, 4) | unit-cube units, unclipped |
| `edges_attr`, `edges_attr_raw` | (E, 4) | same pair of views |
| `adjacency` | (N, N) | binary, shared-edge; symmetric, zero diagonal |
| `attr_similarity` | (N, N) | attribute cosine similarity; separate from `adjacency` |
| `semantic_labels` | (N,) | class index |
| `instance_labels` | (N, N) | MFInstSeg only; 1 where two faces share an instance |
| `bottom_labels` | (N,) | MFInstSeg only |

Column 0 of both attribute vectors is `type_index / 7`, a categorical encoded as an ordinal. `one_hot_type: true` replaces it with an 8-way indicator (4-d to 11-d). All reported runs use `false`.

Max face area is 12.5664, which is 4π. Max edge length is 6.2832, which is 2π. Those are the theoretical bounds for a solid normalized into a cube of size 2 (UV-Net protocol). Both corpora hit the bounds to five decimals.

## Splits

The current split is 70/10/20, seed 42, verified disjoint and complete. Counts were read from the files on 23 Sep 2026.

| Corpus | Split file | Train | Val | Test | Total |
|---|---|---|---|---|---|
| CADSynth | `cadsynth_70_10_20_seed42.json` | 4,244 | 606 | 1,212 | 6,062 |
| MFInstSeg | `mfinstseg_70_10_20_seed42.json` | 4,116 | 588 | 1,176 | 5,880 |

Test ids match the 80/20 files listed below. Both corpora were checked by set comparison.

`runners/make_splits.py` generated the files. Trainers and evaluators read these id lists when `split_file` is set.

The 80/20 files remain in `datasets/splits/`. No config points at them. Numbers produced under those files are outside the quotable set.

| Corpus | Split file | Train | Test |
|---|---|---|---|
| CADSynth | `cadsynth_80_20_seed42.json` | 4,850 | 1,212 |
| MFInstSeg | `mfinstseg_80_20_seed42.json` | 4,704 | 1,176 |

## Normalization statistics

| Corpus | File | Fitted on | Models | Faces / edges measured |
|---|---|---|---|---|
| CADSynth | `splits/cadsynth_norm_train.json` | training split only | 4,244 | 118,165 / 311,806 |
| MFInstSeg | `splits/mfinstseg_norm_train.json` | training split only | 4,116 | 111,909 / 289,495 |

Both current files carry `split_used: "train"`. `runners/compute_norm_stats.py` produced them on 2026-09-21, after the validation partition was carved out. The dataset class prints a warning if it loads statistics whose `split_used` is anything else.

`*_norm_train_SUPERSEDED_80_20.json` is in `datasets/splits/`. Those files record 4,850 CADSynth models (135,225 faces, 356,690 edges) and 4,704 MFInstSeg models (127,490 faces, 329,616 edges). Do not point a config at them. `n_models` distinguishes them from the current train statistics.

`raw_feature_stats.json` inside each archive is computed over all models, including the test split. It is a diagnostic. Its `note` field says so.

## Reproducing the corpora

```bash
# Commands are run from the repository root.
# Extraction needs env-extraction.yml (pythonocc). Training needs env-model.yml.
# <cadsynth-dir> and <mfinstseg-dir> are directories outside this repository.
# The 70/10/20 split files and the train-only norm files are already in
# datasets/splits/. Use those files. Regenerating the split would move the test ids.

# CADSynth
python -m src.cmca.extract.extract_features_cadsynth \
    --step-dir  <path to CADSynth STEP files> \
    --out-dir   <cadsynth-dir> \
    --normalize unitcube

# MFInstSeg
python -m src.cmca.extract.extract_features_mfinstaseg \
    --step-dir  <path to MFInstSeg STEP files> \
    --label-dir <path to MFInstSeg labels> \
    --out-dir   <mfinstseg-dir> \
    --normalize unitcube

# Re-run only to reproduce the committed statistics.
# These commands write over the current norm files.
python runners/compute_norm_stats.py \
    --corpus <cadsynth-dir> \
    --split datasets/splits/cadsynth_70_10_20_seed42.json \
    --out datasets/splits/cadsynth_norm_train.json --which train

python runners/compute_norm_stats.py \
    --corpus <mfinstseg-dir> \
    --split datasets/splits/mfinstseg_70_10_20_seed42.json \
    --out datasets/splits/mfinstseg_norm_train.json --which train
```

Both extractors carry `--self-test` (a box and a blind-hole part with known answers) and `--check-json` (validates a finished corpus). Run `--self-test` before an extraction.

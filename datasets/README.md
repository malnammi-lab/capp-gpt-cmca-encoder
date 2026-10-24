# datasets/

Split files, normalization statistics, and the location of the raw corpora.

| Path | Uploaded to GitHub? | What |
|---|---|---|
| `MANIFEST.md` | yes | provenance, checksums, counts, schema, and the current split |
| `splits/*_70_10_20_seed42.json` | yes | current train, validation, and test model ids |
| `splits/*_80_20_seed42.json` | yes | earlier train/test lists; the 70/10/20 test ids are frozen against them |
| `splits/*_norm_train.json` | yes | standardization statistics, fitted on the 70/10/20 train partition only |
| `splits/*_norm_train_SUPERSEDED_80_20.json` | yes | statistics fitted on the old, larger train partition; do not load these |
| raw corpora | no | CADSynth graphs, MFInstSeg graphs, and CADSynth face grids; Zenodo [10.5281/zenodo.23602506](https://doi.org/10.5281/zenodo.23602506) |

The split and statistics files are a few hundred KB. They belong in the repository. The reported numbers depend on these exact partitions.

## Where the corpora go

The raw corpora are not in this repository. Download the three zip files from Zenodo, [10.5281/zenodo.23602506](https://doi.org/10.5281/zenodo.23602506). Each experiment config looks for its corpus in a data folder placed next to this repository. That folder is not part of the repository. Unpack the archives there:

```
CadSynth feature.zip        105,540,100 bytes   ->  ../data/cadsynth/
MFINSTASEG features.zip      85,397,231 bytes   ->  ../data/mfinstseg/
extracted_faces_v2.zip       66,774,500 bytes   ->  ../data/cadsynth_faces/
```

Those directories then hold the CADSynth graph files (6,062 models), the MFInstSeg graph files (5,880 models), and the CADSynth face grids for the ResNet cells (6,062 models). Checksums are in `MANIFEST.md`. All three match the archives that produced every reported result.

## Splits

`runners/make_splits.py` wrote the current files on 2026-09-21, with seed 42. The split is 70/10/20.

| Corpus | Train | Val | Test | Total | Disjoint |
|---|---:|---:|---:|---:|---|
| CADSynth | 4,244 | 606 | 1,212 | 6,062 | yes |
| MFInstSeg | 4,116 | 588 | 1,176 | 5,880 | yes |

The test partition is frozen. The validation slice was carved from the old training portion, so the test ids in `*_70_10_20_seed42.json` match `*_80_20_seed42.json`. `tests/test_dataset.py` asserts that equality. A result on the new split is measured on the same held-out parts as a result on the old test set.

The 80/20 files remain in `datasets/splits/`. No config uses them.

The validation partition supports four rules:

- Early stopping uses validation loss, with patience 10 and a maximum of 60 epochs.
- Evaluation uses the checkpoint with the best validation loss.
- Any threshold is tuned on validation.
- The `Val` column in the epoch log is a held-out number.

## Normalization statistics

`splits/<corpus>_norm_train.json` comes from `runners/compute_norm_stats.py`. The files were fitted on the training partition: 4,244 CADSynth models and 4,116 MFInstSeg models.

The files `*_norm_train_SUPERSEDED_80_20.json` are in `datasets/splits/`. Do not point a config at them. Both the current and the superseded files set `split_used` to `"train"`. `n_models` distinguishes them, and the split test asserts on it.

Both current files carry `split_used: "train"`. `src/data/brep_dataset.py` prints a warning if it loads statistics whose `split_used` is anything else.

`raw_feature_stats.json` inside each archive is computed over all models, including the test split. It is a diagnostic. Its `note` field says so.

## Configs and split files

Every config in `configs/` reads a split file. All 25 point at the 70/10/20 file for their corpus.

## Corpus status

All three corpora are acquired and checksummed. See `MANIFEST.md`.

# Cap audit -- tier T2

Generated 2026-09-25T05:06:18+00:00 by `runners/check_convergence.py`.
Results root: `results/`   seeds: 0 1 2 3 4

Early stopping uses validation loss. The evaluated checkpoint is the one with the best validation loss.
Cap(s) recorded by these runs: [60]. Patience: [10].
Pre-registered rule: max_epochs 60, patience 10 (`check_convergence.REGISTERED_MAX_EPOCHS` / `REGISTERED_PATIENCE`, registered in `runners/check_convergence.py`).

`stopping_reason` is `early_stop` (the run converged) or `cap` (the run was truncated by its budget).

| arm | seed | selected epoch | cap | stopping reason | verdict |
|---|---|---|---|---|---|
| mfinstseg_bias_similarity | 0 | 32 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_similarity | 1 | 30 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_similarity | 2 | 41 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_similarity | 3 | 41 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_similarity | 4 | 47 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_adjacency | 0 | 37 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_adjacency | 1 | 34 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_adjacency | 2 | 30 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_adjacency | 3 | 30 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_adjacency | 4 | 33 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_both | 0 | 37 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_both | 1 | 31 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_both | 2 | 30 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_both | 3 | 30 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_both | 4 | 37 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_shuffled | 0 | 38 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_shuffled | 1 | 46 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_shuffled | 2 | 49 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_shuffled | 3 | 42 | 60 | `early_stop` | CONVERGED |
| mfinstseg_bias_shuffled | 4 | 36 | 60 | `early_stop` | CONVERGED |

## Val-curve sanity

Read from each run's `train.log`. The question is whether the epoch the run SELECTED is the epoch at which its own validation loss bottoms out -- not whether the curve looks converged, which patience decided during the run.

| arm | seed | validation curve |
|---|---|---|
| mfinstseg_bias_similarity | 0 | epochs 1-42 (42), min 0.2776 @32, selected 0.2776 @32 |
| mfinstseg_bias_similarity | 1 | epochs 1-40 (40), min 0.2930 @30, selected 0.2930 @30 |
| mfinstseg_bias_similarity | 2 | epochs 1-51 (51), min 0.2767 @41, selected 0.2767 @41 |
| mfinstseg_bias_similarity | 3 | epochs 1-51 (51), min 0.2945 @41, selected 0.2945 @41 |
| mfinstseg_bias_similarity | 4 | epochs 1-57 (57), min 0.3038 @47, selected 0.3038 @47 |
| mfinstseg_bias_adjacency | 0 | epochs 1-47 (47), min 0.1800 @37, selected 0.1800 @37 |
| mfinstseg_bias_adjacency | 1 | epochs 1-44 (44), min 0.1849 @34, selected 0.1849 @34 |
| mfinstseg_bias_adjacency | 2 | epochs 1-40 (40), min 0.1708 @30, selected 0.1708 @30 |
| mfinstseg_bias_adjacency | 3 | epochs 1-40 (40), min 0.1882 @30, selected 0.1882 @30 |
| mfinstseg_bias_adjacency | 4 | epochs 1-43 (43), min 0.1733 @33, selected 0.1733 @33 |
| mfinstseg_bias_both | 0 | epochs 1-47 (47), min 0.1762 @37, selected 0.1762 @37 |
| mfinstseg_bias_both | 1 | epochs 1-41 (41), min 0.1859 @31, selected 0.1859 @31 |
| mfinstseg_bias_both | 2 | epochs 1-40 (40), min 0.1739 @30, selected 0.1739 @30 |
| mfinstseg_bias_both | 3 | epochs 1-40 (40), min 0.1931 @30, selected 0.1931 @30 |
| mfinstseg_bias_both | 4 | epochs 1-47 (47), min 0.1788 @37, selected 0.1788 @37 |
| mfinstseg_bias_shuffled | 0 | epochs 1-48 (48), min 0.3121 @38, selected 0.3121 @38 |
| mfinstseg_bias_shuffled | 1 | epochs 1-56 (56), min 0.3086 @46, selected 0.3086 @46 |
| mfinstseg_bias_shuffled | 2 | epochs 1-59 (59), min 0.3374 @49, selected 0.3374 @49 |
| mfinstseg_bias_shuffled | 3 | epochs 1-52 (52), min 0.3122 @42, selected 0.3122 @42 |
| mfinstseg_bias_shuffled | 4 | epochs 1-46 (46), min 0.3331 @36, selected 0.3331 @36 |

## Per arm

- **mfinstseg_bias_similarity** -- CONVERGED -- 5/5 early-stopped
- **mfinstseg_bias_adjacency** -- CONVERGED -- 5/5 early-stopped
- **mfinstseg_bias_both** -- CONVERGED -- 5/5 early-stopped
- **mfinstseg_bias_shuffled** -- CONVERGED -- 5/5 early-stopped

## Verdict

- PASS: every audited seed stopped early, inside its cap, under one selection rule. This comparison is not budget-confounded.

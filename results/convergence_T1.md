# Cap audit -- tier T1

Generated 2026-09-24T21:43:59+00:00 by `runners/check_convergence.py`.
Results root: `results/`   seeds: 0 1 2 3 4

Early stopping uses validation loss. The evaluated checkpoint is the one with the best validation loss.
Cap(s) recorded by these runs: [60]. Patience: [10].
Pre-registered rule: max_epochs 60, patience 10 (`check_convergence.REGISTERED_MAX_EPOCHS` / `REGISTERED_PATIENCE`, registered in `runners/check_convergence.py`).

`stopping_reason` is `early_stop` (the run converged) or `cap` (the run was truncated by its budget).

| arm | seed | selected epoch | cap | stopping reason | verdict |
|---|---|---|---|---|---|
| cadsynth_bias_similarity | 0 | 50 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_similarity | 1 | 49 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_similarity | 2 | 42 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_similarity | 3 | 48 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_similarity | 4 | 37 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_adjacency | 0 | 36 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_adjacency | 1 | 33 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_adjacency | 2 | 33 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_adjacency | 3 | 35 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_adjacency | 4 | 29 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_both | 0 | 42 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_both | 1 | 32 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_both | 2 | 36 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_both | 3 | 35 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_both | 4 | 32 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_shuffled | 0 | 40 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_shuffled | 1 | 47 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_shuffled | 2 | 42 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_shuffled | 3 | 45 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_shuffled | 4 | 40 | 60 | `early_stop` | CONVERGED |

## Val-curve sanity

Read from each run's `train.log`. The question is whether the epoch the run SELECTED is the epoch at which its own validation loss bottoms out -- not whether the curve looks converged, which patience decided during the run.

| arm | seed | validation curve |
|---|---|---|
| cadsynth_bias_similarity | 0 | epochs 1-60 (60), min 0.2586 @50, selected 0.2586 @50 |
| cadsynth_bias_similarity | 1 | epochs 1-59 (59), min 0.2395 @49, selected 0.2395 @49 |
| cadsynth_bias_similarity | 2 | epochs 1-52 (52), min 0.2244 @42, selected 0.2244 @42 |
| cadsynth_bias_similarity | 3 | epochs 1-58 (58), min 0.2375 @48, selected 0.2375 @48 |
| cadsynth_bias_similarity | 4 | epochs 1-47 (47), min 0.2492 @37, selected 0.2492 @37 |
| cadsynth_bias_adjacency | 0 | epochs 1-46 (46), min 0.1305 @36, selected 0.1305 @36 |
| cadsynth_bias_adjacency | 1 | epochs 1-43 (43), min 0.1395 @33, selected 0.1395 @33 |
| cadsynth_bias_adjacency | 2 | epochs 1-43 (43), min 0.1392 @33, selected 0.1392 @33 |
| cadsynth_bias_adjacency | 3 | epochs 1-45 (45), min 0.1392 @35, selected 0.1392 @35 |
| cadsynth_bias_adjacency | 4 | epochs 1-39 (39), min 0.1439 @29, selected 0.1439 @29 |
| cadsynth_bias_both | 0 | epochs 1-52 (52), min 0.1345 @42, selected 0.1345 @42 |
| cadsynth_bias_both | 1 | epochs 1-42 (42), min 0.1364 @32, selected 0.1364 @32 |
| cadsynth_bias_both | 2 | epochs 1-46 (46), min 0.1388 @36, selected 0.1388 @36 |
| cadsynth_bias_both | 3 | epochs 1-45 (45), min 0.1438 @35, selected 0.1438 @35 |
| cadsynth_bias_both | 4 | epochs 1-42 (42), min 0.1460 @32, selected 0.1460 @32 |
| cadsynth_bias_shuffled | 0 | epochs 1-50 (50), min 0.2861 @40, selected 0.2861 @40 |
| cadsynth_bias_shuffled | 1 | epochs 1-57 (57), min 0.2915 @47, selected 0.2915 @47 |
| cadsynth_bias_shuffled | 2 | epochs 1-52 (52), min 0.2664 @42, selected 0.2664 @42 |
| cadsynth_bias_shuffled | 3 | epochs 1-55 (55), min 0.2875 @45, selected 0.2875 @45 |
| cadsynth_bias_shuffled | 4 | epochs 1-50 (50), min 0.2735 @40, selected 0.2735 @40 |

## Per arm

- **cadsynth_bias_similarity** -- CONVERGED -- 5/5 early-stopped
- **cadsynth_bias_adjacency** -- CONVERGED -- 5/5 early-stopped
- **cadsynth_bias_both** -- CONVERGED -- 5/5 early-stopped
- **cadsynth_bias_shuffled** -- CONVERGED -- 5/5 early-stopped

## Verdict

- PASS: every audited seed stopped early, inside its cap, under one selection rule. This comparison is not budget-confounded.

# Cap audit -- tier T6

Generated 2026-09-26T01:44:31+00:00 by `runners/check_convergence.py`.
Results root: `results/`   seeds: 0 1 2 3 4

Early stopping uses validation loss. The evaluated checkpoint is the one with the best validation loss.
Cap(s) recorded by these runs: [60]. Patience: [10].
Pre-registered rule: max_epochs 60, patience 10 (`check_convergence.REGISTERED_MAX_EPOCHS` / `REGISTERED_PATIENCE`, registered in `runners/check_convergence.py`).

`stopping_reason` is `early_stop` (the run converged) or `cap` (the run was truncated by its budget).

| arm | seed | selected epoch | cap | stopping reason | verdict |
|---|---|---|---|---|---|
| cadsynth_bias_similarity_weighted | 0 | 27 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_similarity_weighted | 1 | 33 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_similarity_weighted | 2 | 42 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_similarity_weighted | 3 | 30 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_similarity_weighted | 4 | 34 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_adjacency_weighted | 0 | 30 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_adjacency_weighted | 1 | 25 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_adjacency_weighted | 2 | 23 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_adjacency_weighted | 3 | 30 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_adjacency_weighted | 4 | 26 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_both_weighted | 0 | 30 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_both_weighted | 1 | 23 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_both_weighted | 2 | 23 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_both_weighted | 3 | 30 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_both_weighted | 4 | 33 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_shuffled_weighted | 0 | 32 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_shuffled_weighted | 1 | 34 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_shuffled_weighted | 2 | 34 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_shuffled_weighted | 3 | 29 | 60 | `early_stop` | CONVERGED |
| cadsynth_bias_shuffled_weighted | 4 | 36 | 60 | `early_stop` | CONVERGED |

## Val-curve sanity

Read from each run's `train.log`. The question is whether the epoch the run SELECTED is the epoch at which its own validation loss bottoms out -- not whether the curve looks converged, which patience decided during the run.

| arm | seed | validation curve |
|---|---|---|
| cadsynth_bias_similarity_weighted | 0 | epochs 1-37 (37), min 0.5135 @27, selected 0.5135 @27 |
| cadsynth_bias_similarity_weighted | 1 | epochs 1-43 (43), min 0.4974 @33, selected 0.4974 @33 |
| cadsynth_bias_similarity_weighted | 2 | epochs 1-52 (52), min 0.5037 @42, selected 0.5037 @42 |
| cadsynth_bias_similarity_weighted | 3 | epochs 1-40 (40), min 0.4593 @30, selected 0.4593 @30 |
| cadsynth_bias_similarity_weighted | 4 | epochs 1-44 (44), min 0.4672 @34, selected 0.4672 @34 |
| cadsynth_bias_adjacency_weighted | 0 | epochs 1-40 (40), min 0.2170 @30, selected 0.2170 @30 |
| cadsynth_bias_adjacency_weighted | 1 | epochs 1-35 (35), min 0.2447 @25, selected 0.2447 @25 |
| cadsynth_bias_adjacency_weighted | 2 | epochs 1-33 (33), min 0.2421 @23, selected 0.2421 @23 |
| cadsynth_bias_adjacency_weighted | 3 | epochs 1-40 (40), min 0.2400 @30, selected 0.2400 @30 |
| cadsynth_bias_adjacency_weighted | 4 | epochs 1-36 (36), min 0.2345 @26, selected 0.2345 @26 |
| cadsynth_bias_both_weighted | 0 | epochs 1-40 (40), min 0.2186 @30, selected 0.2186 @30 |
| cadsynth_bias_both_weighted | 1 | epochs 1-33 (33), min 0.2602 @23, selected 0.2602 @23 |
| cadsynth_bias_both_weighted | 2 | epochs 1-33 (33), min 0.2398 @23, selected 0.2398 @23 |
| cadsynth_bias_both_weighted | 3 | epochs 1-40 (40), min 0.2390 @30, selected 0.2390 @30 |
| cadsynth_bias_both_weighted | 4 | epochs 1-43 (43), min 0.2416 @33, selected 0.2416 @33 |
| cadsynth_bias_shuffled_weighted | 0 | epochs 1-42 (42), min 0.5581 @32, selected 0.5581 @32 |
| cadsynth_bias_shuffled_weighted | 1 | epochs 1-44 (44), min 0.5474 @34, selected 0.5474 @34 |
| cadsynth_bias_shuffled_weighted | 2 | epochs 1-44 (44), min 0.5715 @34, selected 0.5715 @34 |
| cadsynth_bias_shuffled_weighted | 3 | epochs 1-39 (39), min 0.5189 @29, selected 0.5189 @29 |
| cadsynth_bias_shuffled_weighted | 4 | epochs 1-46 (46), min 0.5098 @36, selected 0.5098 @36 |

## Per arm

- **cadsynth_bias_similarity_weighted** -- CONVERGED -- 5/5 early-stopped
- **cadsynth_bias_adjacency_weighted** -- CONVERGED -- 5/5 early-stopped
- **cadsynth_bias_both_weighted** -- CONVERGED -- 5/5 early-stopped
- **cadsynth_bias_shuffled_weighted** -- CONVERGED -- 5/5 early-stopped

## Verdict

- PASS: every audited seed stopped early, inside its cap, under one selection rule. This comparison is not budget-confounded.

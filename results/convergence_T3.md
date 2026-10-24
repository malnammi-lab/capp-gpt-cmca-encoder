# Cap audit -- tier T3

Generated 2026-09-25T09:54:00+00:00 by `runners/check_convergence.py`.
Results root: `results/`   seeds: 0 1 2 3 4

Early stopping uses validation loss. The evaluated checkpoint is the one with the best validation loss.
Cap(s) recorded by these runs: [60]. Patience: [10].
Pre-registered rule: max_epochs 60, patience 10 (`check_convergence.REGISTERED_MAX_EPOCHS` / `REGISTERED_PATIENCE`, registered in `runners/check_convergence.py`).

`stopping_reason` is `early_stop` (the run converged) or `cap` (the run was truncated by its budget).

| arm | seed | selected epoch | cap | stopping reason | verdict |
|---|---|---|---|---|---|
| cadsynth_ce_head | 0 | 46 | 60 | `early_stop` | CONVERGED |
| cadsynth_ce_head | 1 | 36 | 60 | `early_stop` | CONVERGED |
| cadsynth_ce_head | 2 | 39 | 60 | `early_stop` | CONVERGED |
| cadsynth_ce_head | 3 | 32 | 60 | `early_stop` | CONVERGED |
| cadsynth_ce_head | 4 | 43 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_cmca | 0 | 27 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_cmca | 1 | 26 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_cmca | 2 | 22 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_cmca | 3 | 22 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_cmca | 4 | 20 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_ce | 0 | 24 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_ce | 1 | 17 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_ce | 2 | 23 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_ce | 3 | 18 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_ce | 4 | 21 | 60 | `early_stop` | CONVERGED |

## Val-curve sanity

Read from each run's `train.log`. The question is whether the epoch the run SELECTED is the epoch at which its own validation loss bottoms out -- not whether the curve looks converged, which patience decided during the run.

| arm | seed | validation curve |
|---|---|---|
| cadsynth_ce_head | 0 | epochs 1-56 (56), min 0.1269 @46, selected 0.1269 @46 |
| cadsynth_ce_head | 1 | epochs 1-46 (46), min 0.1226 @36, selected 0.1226 @36 |
| cadsynth_ce_head | 2 | epochs 1-49 (49), min 0.1426 @39, selected 0.1426 @39 |
| cadsynth_ce_head | 3 | epochs 1-42 (42), min 0.1330 @32, selected 0.1330 @32 |
| cadsynth_ce_head | 4 | epochs 1-53 (53), min 0.1308 @43, selected 0.1308 @43 |
| resnet_cadsynth_cmca | 0 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_cmca | 1 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_cmca | 2 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_cmca | 3 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_cmca | 4 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_ce | 0 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_ce | 1 | only 3 lines over epochs 1-20 -- not performed |
| resnet_cadsynth_ce | 2 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_ce | 3 | only 3 lines over epochs 1-20 -- not performed |
| resnet_cadsynth_ce | 4 | only 4 lines over epochs 1-30 -- not performed |

## Per arm

- **cadsynth_ce_head** -- CONVERGED -- 5/5 early-stopped
- **resnet_cadsynth_cmca** -- CONVERGED -- 5/5 early-stopped
- **resnet_cadsynth_ce** -- CONVERGED -- 5/5 early-stopped

## Not performed

- val-curve sanity NOT PERFORMED for 10 of 15 run(s) -- train.log records only some of the run's epochs, so the lowest LOGGED validation loss need not be the lowest one reached -- src/training/train_resnet.py:401 prints an epoch line only every 10th epoch, which is the usual cause: resnet_cadsynth_cmca seed 0, resnet_cadsynth_cmca seed 1, resnet_cadsynth_cmca seed 2, resnet_cadsynth_cmca seed 3, resnet_cadsynth_cmca seed 4, resnet_cadsynth_ce seed 0, resnet_cadsynth_ce seed 1, resnet_cadsynth_ce seed 2, resnet_cadsynth_ce seed 3, resnet_cadsynth_ce seed 4.

## Verdict

- PASS: every audited seed stopped early, inside its cap, under one selection rule. This comparison is not budget-confounded.

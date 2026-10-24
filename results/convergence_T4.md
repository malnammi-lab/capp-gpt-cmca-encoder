# Cap audit -- tier T4

Generated 2026-09-25T16:36:15+00:00 by `runners/check_convergence.py`.
Results root: `results/`   seeds: 0 1 2 3 4

Early stopping uses validation loss. The evaluated checkpoint is the one with the best validation loss.
Cap(s) recorded by these runs: [60]. Patience: [10].
Pre-registered rule: max_epochs 60, patience 10 (`check_convergence.REGISTERED_MAX_EPOCHS` / `REGISTERED_PATIENCE`, registered in `runners/check_convergence.py`).

`stopping_reason` is `early_stop` (the run converged) or `cap` (the run was truncated by its budget).

| arm | seed | selected epoch | cap | stopping reason | verdict |
|---|---|---|---|---|---|
| resnet_cadsynth_ce_weighted | 0 | 23 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_ce_weighted | 1 | 22 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_ce_weighted | 2 | 23 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_ce_weighted | 3 | 27 | 60 | `early_stop` | CONVERGED |
| resnet_cadsynth_ce_weighted | 4 | 24 | 60 | `early_stop` | CONVERGED |

## Val-curve sanity

Read from each run's `train.log`. The question is whether the epoch the run SELECTED is the epoch at which its own validation loss bottoms out -- not whether the curve looks converged, which patience decided during the run.

| arm | seed | validation curve |
|---|---|---|
| resnet_cadsynth_ce_weighted | 0 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_ce_weighted | 1 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_ce_weighted | 2 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_ce_weighted | 3 | only 4 lines over epochs 1-30 -- not performed |
| resnet_cadsynth_ce_weighted | 4 | only 4 lines over epochs 1-30 -- not performed |

## Per arm

- **resnet_cadsynth_ce_weighted** -- CONVERGED -- 5/5 early-stopped

## Not performed

- val-curve sanity NOT PERFORMED for 5 of 5 run(s) -- train.log records only some of the run's epochs, so the lowest LOGGED validation loss need not be the lowest one reached -- src/training/train_resnet.py:401 prints an epoch line only every 10th epoch, which is the usual cause: resnet_cadsynth_ce_weighted seed 0, resnet_cadsynth_ce_weighted seed 1, resnet_cadsynth_ce_weighted seed 2, resnet_cadsynth_ce_weighted seed 3, resnet_cadsynth_ce_weighted seed 4.

## Verdict

- PASS: every audited seed stopped early, inside its cap, under one selection rule. This comparison is not budget-confounded.

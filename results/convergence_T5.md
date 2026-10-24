# Cap audit -- tier T5

Generated 2026-09-25T19:42:16+00:00 by `runners/check_convergence.py`.
Results root: `results/`   seeds: 0 1 2 3 4

Early stopping uses validation loss. The evaluated checkpoint is the one with the best validation loss.
Cap(s) recorded by these runs: [60]. Patience: [10].
Pre-registered rule: max_epochs 60, patience 10 (`check_convergence.REGISTERED_MAX_EPOCHS` / `REGISTERED_PATIENCE`, registered in `runners/check_convergence.py`).

`stopping_reason` is `early_stop` (the run converged) or `cap` (the run was truncated by its budget).

| arm | seed | selected epoch | cap | stopping reason | verdict |
|---|---|---|---|---|---|
| mfinstseg_unified | 0 | 38 | 60 | `early_stop` | CONVERGED |
| mfinstseg_unified | 1 | 38 | 60 | `early_stop` | CONVERGED |
| mfinstseg_unified | 2 | 44 | 60 | `early_stop` | CONVERGED |
| mfinstseg_unified | 3 | 23 | 60 | `early_stop` | CONVERGED |
| mfinstseg_unified | 4 | 33 | 60 | `early_stop` | CONVERGED |

## Val-curve sanity

Read from each run's `train.log`. The question is whether the epoch the run SELECTED is the epoch at which its own validation loss bottoms out -- not whether the curve looks converged, which patience decided during the run.

| arm | seed | validation curve |
|---|---|---|
| mfinstseg_unified | 0 | epochs 1-48 (48), min 0.4367 @38, selected 0.4367 @38 |
| mfinstseg_unified | 1 | epochs 1-48 (48), min 0.4611 @38, selected 0.4611 @38 |
| mfinstseg_unified | 2 | epochs 1-54 (54), min 0.4894 @44, selected 0.4894 @44 |
| mfinstseg_unified | 3 | epochs 1-33 (33), min 0.4702 @23, selected 0.4702 @23 |
| mfinstseg_unified | 4 | epochs 1-43 (43), min 0.4896 @33, selected 0.4896 @33 |

## Per arm

- **mfinstseg_unified** -- CONVERGED -- 5/5 early-stopped

## Verdict

- PASS: every audited seed stopped early, inside its cap, under one selection rule. This comparison is not budget-confounded.

"""The seven defects mutation testing found in the two AGGREGATOR tools.

`runners/aggregate_seeds.py` is the tool `scripts/RUNBOOK-tiers.md` names and the
one that writes the paper's table; `runners/compare_arms.py` is the per-class
diagnostic beside it. Every defect below was reproduced on a scratch copy of
`results/` on 25 Sep 2026 and written up in `weekly/GATE-DEFECTS.md`; this file
is the regression suite for the fixes.

Each test targets ONE defect and was confirmed RED against the pre-fix tree
before the fix was written. A test that passes before and after proves nothing
-- that is how this project shipped a selection-rule check with no coverage at
all -- so the regression guards (the "still" tests, which pin behaviour the fix
must not change) are labelled as such and are deliberately in the minority.

THE SEVEN
---------
1. `compare_arms.py` had a hardcoded `ARMS` tuple and no `--arms` flag, so it
   could address only T1/T2's bias arms. On T3-T6 it reported on the wrong
   cells or was skipped, and `--baseline` was validated against the frozen
   tuple rather than against the arms in play.
2. `aggregate_seeds.py` defaulted `--arms` to `DEFAULT_ARMS`, which is T1's
   arms. It once measured T1 while claiming to measure T3 and overwrote T1's
   own summary on the way out.
3. The generated table was titled "<dataset> bias ablation" whatever was
   aggregated -- over single ResNet arms and over the unified cell alike.
4. The caption printed the SELECTED epoch in a "Runs stopped at epoch(s)"
   claim. `provenance['epoch']` IS the selected epoch: `build_provenance` is
   called inside the save-on-improvement branch. T5 really stopped at
   48/48/54/33/43 while the artifact said 23-44.
5. It certified a cap-bound run as quotable whenever `stopping_reason` was
   mislabelled: it read the label and never compared `selected_epoch` to the
   cap. `check_tier` and `check_convergence` both reject that run.
6. It had no instance or bottom-face path at all (the words appeared zero
   times in the file), so T5's headline ROC-AUC had no generated table behind
   it and two of the unified model's three heads were unread.
7. Both files justified pooling per-class counts with "averaging ratios weights
   a seed that saw few examples of a rare class equally with one that saw
   many". That is false of this data: every seed scores the identical frozen
   held-out set, so there is exactly ONE per-class support vector across all 85
   runs of the batch (measured 26 Sep 2026: cadsynth 34338 faces over 60 runs,
   mfinstseg 31795 faces over 25 runs, one distinct support vector each).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

CODEBASE = Path(__file__).resolve().parents[1]
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from runners import aggregate_seeds, check_convergence, compare_arms  # noqa: E402
from src.utils.provenance import code_fingerprint                     # noqa: E402

#: Every run of a corpus scores the identical frozen split, so this is a
#: constant of the fixture too -- an arm evaluated on a different count trips
#: check_tier's different-test-sizes fault rather than whatever is under test.
N_TEST_FACES = 31795

#: A 2x2 confusion matrix so `compute_mean_iou` and the per-class section have
#: something real to work on.
CM = [[900, 40], [60, 800]]


# --------------------------------------------------------------------- #
#  fixtures: metrics files in the two layouts, plus a resolved config    #
# --------------------------------------------------------------------- #

def _prov(seed: int, fp: str, *, arm: str = "", selected: int = 38,
          epoch: int | None = None, reason: str = "early_stop", cap: int = 60,
          weights=False) -> dict:
    """A complete provenance block.

    `epoch` defaults to `selected` because that is what a sound run records:
    every trainer calls `build_provenance` inside the save-on-improvement
    branch, so the two are the same number. Defect 4 is that the tool printed
    one of them under the word "stopped".
    """
    return {
        "seed": seed,
        "epoch": selected if epoch is None else epoch,
        "selected_epoch": selected,
        "stopping_reason": reason,
        "use_class_weights": weights,
        "epoch_budget_configured": cap,
        "config_hash": "c0ffeec0ffee{:04d}".format(seed),
        # [FIX 26 Sep 2026, gate-parity wave] This was the literal
        # "82d3555d25a39d3f" for EVERY arm of the fixture. On disk the arm
        # hash is the resolved config with `seed` and `paths` dropped, so four
        # different arms cannot share one -- and `aggregate_seeds.py` now
        # refuses a table whose arms do, because the pairwise "which
        # differences the data supports" section below it would be comparing
        # an arm with itself. The fixture, not the check, was wrong: it is
        # derived from the arm name so the four arms differ and the five seeds
        # of each agree, which is the invariant
        # `tests/test_provenance_confighash.py` asserts of the real configs.
        "arm_config_hash": hashlib.sha256(
            f"arm:{arm}".encode()).hexdigest()[:16],
        "code_fingerprint": fp,
        "git_commit": None,
        "gpu": "Tesla T4",
        "wall_time_s": 931.0,
        "torch": "2.8.0+cu128",
        "python": "3.12.11",
        "platform": "Linux-6.8.0-1063-aws-x86_64-with-glibc2.39",
        "recorded_at": "2026-09-25T18:36:32+00:00",
        "evaluated": {"config_hash": "c0ffeec0ffee{:04d}".format(seed),
                      "code_fingerprint": fp, "gpu": "Tesla T4",
                      "at": "2026-09-25T18:43:48+00:00"},
    }


def flat_metrics(name: str, seed: int, acc: float, fp: str, **kw) -> dict:
    """What `src/evaluation/evaluate_semantic.py` writes: one task, flat."""
    return {
        "experiment_name": name,
        "n_classes": 2,
        "n_samples": N_TEST_FACES,
        "overall_accuracy": acc,
        "macro_precision": 0.71,
        "macro_recall": 0.69,
        "macro_f1": 0.70,
        "mean_iou": 0.55,
        "confusion_matrix": CM,
        "provenance": _prov(seed, fp, arm=name, **kw),
    }


def unified_metrics(name: str, seed: int, acc: float, fp: str, *,
                    auc: float = 0.997, bottom_f1: float = 0.946,
                    reportable: str = "checkpoint_threshold",
                    instance: bool = True, bottom: bool = True,
                    **kw) -> dict:
    """What `src/evaluation/evaluate_unified.py` writes for T5: three heads.

    Key names and nesting copied from a real run,
    `results/mfinstseg_unified/seed2/unified_eval_metrics.json`.
    """
    m = {
        "experiment_name": name,
        "epoch": 44,
        "split": "test (frozen ids)",
        "n_models": 1176,
        "semantic": {
            "n_faces": N_TEST_FACES,
            "n_classes": 2,
            "overall_accuracy": acc,
            "macro_precision": 0.92,
            "macro_recall": 0.92,
            "macro_f1": 0.9227,
            "mean_iou": 0.8614,
            "iou_per_class": {"0": 0.77, "1": 0.95},
            "per_class": {},
            "confusion_matrix": CM,
        },
        "provenance": _prov(seed, fp, arm=name, **kw),
    }
    if bottom:
        m["bottom"] = {
            "n_faces": N_TEST_FACES, "n_positive": 5101,
            "accuracy": 0.9827, "precision": 0.9436, "recall": 0.9488,
            "f1": bottom_f1, "mean_iou": 0.9387,
            "iou_not_bottom": 0.9795, "iou_bottom": 0.8979,
            "confusion_matrix": [[26000, 300], [300, 5195]],
        }
    if instance:
        m["instance"] = {
            "n_pairs": 930718, "n_positive": 79682,
            "auc": auc,
            "reportable": reportable,
            "checkpoint_threshold_source": "fitted on the validation partition",
            "note": "oracle_threshold was tuned on the test labels",
            "checkpoint_threshold": {
                "precision": 0.9395, "recall": 0.9265, "f1": 0.9330,
                "accuracy": 0.9886, "threshold": 0.8995,
                "confusion_matrix": [[840000, 5000], [5800, 73000]]},
            "oracle_threshold": {
                "precision": 0.9356, "recall": 0.9322, "f1": 0.9339,
                "accuracy": 0.9887, "threshold": 0.8947,
                "confusion_matrix": [[840100, 4900], [5700, 73100]]},
        }
    return m


def write_resolved(run: Path, seed: int, *, max_epochs: int = 60,
                   patience: int = 10) -> None:
    """The `config.resolved.yaml` `run_all.write_resolved_config` leaves.

    It is the only artifact carrying `training.patience`, which is half the
    selection rule and the half the cap arithmetic of defect 5 needs.
    """
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.resolved.yaml").write_text(yaml.safe_dump({
        "experiment_name": run.parent.name,
        "seed": seed,
        "training": {"max_epochs": max_epochs, "patience": patience,
                     "lr": 0.001, "batch_size": 8},
    }), encoding="utf-8")


def write_arm(root: Path, cell: str, *, seeds=(0, 1, 2), unified=False,
              fp=None, resolved=True, maker_kw=None, per_seed=None) -> Path:
    """One complete cell under ``root/<cell>/seed<k>/``.

    ``per_seed`` is ``{seed: {kwarg: value}}`` for the probes that need exactly
    one seed to differ from its neighbours.
    """
    fp = code_fingerprint() if fp is None else fp
    maker = unified_metrics if unified else flat_metrics
    fname = "unified_eval_metrics.json" if unified else "eval_metrics.json"
    for s in seeds:
        run = root / cell / "seed{}".format(s)
        run.mkdir(parents=True, exist_ok=True)
        kw = dict(maker_kw or {})
        kw.update((per_seed or {}).get(s, {}))
        (run / fname).write_text(
            json.dumps(maker(cell, s, 0.90 + 0.002 * s, fp, **kw)),
            encoding="utf-8")
        if resolved:
            write_resolved(run, s)
    return root / cell


def aggregate(root: Path, out: Path, dataset="mfinstseg", arms=(),
              seeds=("0", "1", "2"), extra=()):
    """Run the aggregator; return (rc, markdown, json payload)."""
    argv = ["--dataset", dataset, "--seeds", *seeds,
            "--results", str(root), "--out", str(out), *extra]
    if arms:
        argv += ["--arms", *arms]
    rc = aggregate_seeds.main(argv)
    return (rc, out.read_text(encoding="utf-8"),
            json.loads(out.with_suffix(".json").read_text(encoding="utf-8")))


# ===================================================================== #
#  DEFECT 1 -- compare_arms could address only T1/T2's four bias arms   #
# ===================================================================== #

def test_compare_arms_reports_on_the_arms_it_was_given(tmp_path):
    """RED before the fix: SystemExit(2) from argparse, `unrecognized: --arms`.

    Without the flag the tool reads the frozen `ARMS` tuple, finds no
    `cadsynth_bias_similarity/` under a T3-T6 results root, and reports on
    cells that are not the ones being audited.
    """
    root = tmp_path / "results"
    write_arm(root, "resnet_cadsynth_ce")
    write_arm(root, "resnet_cadsynth_ce_weighted")
    write_arm(root, "cadsynth_bias_similarity")   # T1's arm, present but NOT asked for

    rc = compare_arms.main(
        ["--dataset", "cadsynth", "--results", str(root),
         "--seeds", "0", "1", "2",
         "--arms", "resnet_cadsynth_ce", "resnet_cadsynth_ce_weighted",
           "--out", str(tmp_path / "cmp.json")])

    # [26 Sep, contract change] compare_arms no longer writes to the constant
    # path <results>/<dataset>_bias_comparison.json on a custom --arms run:
    # writing there unconditionally is what destroyed a committed comparison
    # for a DIFFERENT tier before any arm had been read. The default now
    # derives the name from the arms. This test passes --out explicitly, which
    # is what a test should do anyway: it asserts the --arms behaviour it is
    # named for, not a filename-derivation rule that is another test's job.
    payload = json.loads((tmp_path / "cmp.json").read_text(encoding="utf-8"))
    assert rc == 0, payload["problems"]
    assert sorted(payload["arms"]) == ["resnet_cadsynth_ce",
                                       "resnet_cadsynth_ce_weighted"]
    # The frozen tuple's arms must not leak into a comparison that did not ask
    # for them -- that is the whole defect.
    assert "cadsynth_bias_similarity" not in payload["arms"]
    assert payload["problems"] == []


def test_compare_arms_baseline_is_validated_against_the_given_arms(tmp_path):
    """RED before the fix: `choices=ARMS` rejects every T3-T6 cell name."""
    root = tmp_path / "results"
    write_arm(root, "resnet_cadsynth_ce")
    write_arm(root, "resnet_cadsynth_cmca")

    rc = compare_arms.main(
        ["--dataset", "cadsynth", "--results", str(root),
         "--seeds", "0", "1", "2",
         "--arms", "resnet_cadsynth_ce", "resnet_cadsynth_cmca",
         "--baseline", "resnet_cadsynth_cmca",
           "--out", str(tmp_path / "cmp.json")])

    # [26 Sep, contract change] compare_arms no longer writes to the constant
    # path <results>/<dataset>_bias_comparison.json on a custom --arms run:
    # writing there unconditionally is what destroyed a committed comparison
    # for a DIFFERENT tier before any arm had been read. The default now
    # derives the name from the arms. This test passes --out explicitly, which
    # is what a test should do anyway: it asserts the --arms behaviour it is
    # named for, not a filename-derivation rule that is another test's job.
    payload = json.loads((tmp_path / "cmp.json").read_text(encoding="utf-8"))
    assert rc == 0, payload["problems"]
    assert payload["baseline"] == "resnet_cadsynth_cmca"


def test_compare_arms_refuses_a_baseline_that_is_not_one_of_the_arms(tmp_path):
    """RED before the fix: argparse's own refusal names the FROZEN tuple.

    The message must name the arms actually in play, or the operator is sent
    to look for `similarity` in a tier that has no such cell.
    """
    root = tmp_path / "results"
    write_arm(root, "resnet_cadsynth_ce")

    with pytest.raises(SystemExit) as exc:
        compare_arms.main(
            ["--dataset", "cadsynth", "--results", str(root),
             "--seeds", "0", "1", "2",
             "--arms", "resnet_cadsynth_ce",
             "--baseline", "resnet_cadsynth_cmca"])

    msg = str(exc.value)
    assert "resnet_cadsynth_cmca" in msg, msg
    assert "resnet_cadsynth_ce" in msg, msg


def test_compare_arms_default_arms_are_still_T1s_four(tmp_path):
    """REGRESSION GUARD (green before and after). Existing invocations of

        python runners/compare_arms.py --dataset cadsynth

    must behave exactly as they did: the four bias arms, baseline `similarity`.
    """
    root = tmp_path / "results"
    for arm in compare_arms.ARMS:
        write_arm(root, "cadsynth_bias_{}".format(arm))

    rc = compare_arms.main(["--dataset", "cadsynth", "--results", str(root),
                            "--seeds", "0", "1", "2"])

    payload = json.loads((root / "cadsynth_bias_comparison.json")
                         .read_text(encoding="utf-8"))
    assert rc == 0, payload["problems"]
    # `load_arm` is keyed by the ARM, and `arm_dir` resolves it to the cell
    # directory `cadsynth_bias_<arm>` -- that indirection is what the new
    # --arms flag has to leave alone.
    assert sorted(payload["arms"]) == sorted(compare_arms.ARMS)
    assert payload["baseline"] == "similarity"


# ===================================================================== #
#  DEFECT 2 -- an omitted --arms silently measured T1                   #
# ===================================================================== #

def test_aggregate_seeds_refuses_to_run_without_arms(tmp_path):
    """RED before the fix: rc 0, and a T1 table written under a T3 intent.

    T1's four cells are on disk in every real results root, so "refuse when
    the named arms are not present" could not close this: the defaults ARE
    present. Only naming the arms can.
    """
    root = tmp_path / "results"
    for arm in aggregate_seeds.DEFAULT_ARMS:          # T1, present on disk
        write_arm(root, "cadsynth_bias_{}".format(arm))
    write_arm(root, "resnet_cadsynth_ce")             # the tier actually wanted

    with pytest.raises(SystemExit) as exc:
        aggregate_seeds.main(["--dataset", "cadsynth", "--results", str(root),
                              "--seeds", "0", "1", "2"])

    msg = str(exc.value)
    assert "--arms" in msg, msg
    # The refusal has to be usable: it must name what is on disk so the
    # operator can pass the right cells.
    assert "resnet_cadsynth_ce" in msg, msg


def test_aggregate_seeds_without_arms_overwrites_no_summary(tmp_path):
    """RED before the fix: T1's own summary_cadsynth.md was overwritten.

    The refusal must come BEFORE any artifact is written, or the harm the
    refusal exists to prevent has already happened.
    """
    root = tmp_path / "results"
    for arm in aggregate_seeds.DEFAULT_ARMS:
        write_arm(root, "cadsynth_bias_{}".format(arm))
    victim = root / "summary_cadsynth.md"
    victim.write_text("T1's real table -- do not clobber", encoding="utf-8")

    with pytest.raises(SystemExit):
        aggregate_seeds.main(["--dataset", "cadsynth", "--results", str(root),
                              "--seeds", "0", "1", "2"])

    assert victim.read_text(encoding="utf-8") == "T1's real table -- do not clobber"


def test_aggregate_seeds_still_runs_when_arms_are_named(tmp_path):
    """REGRESSION GUARD (green before and after): naming the arms still works."""
    root = tmp_path / "results"
    write_arm(root, "cadsynth_bias_both")

    rc, md, payload = aggregate(root, tmp_path / "s.md", dataset="cadsynth",
                                arms=("cadsynth_bias_both",))

    assert rc == 0, payload["problems"]
    assert "cadsynth_bias_both" in payload["arms"]


# ===================================================================== #
#  DEFECT 3 -- every table was titled "<dataset> bias ablation"         #
# ===================================================================== #

def test_title_does_not_say_bias_ablation_for_the_unified_cell(tmp_path):
    """RED before the fix: `# mfinstseg bias ablation -- seed-averaged`."""
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True)

    rc, md, payload = aggregate(root, tmp_path / "s.md",
                                arms=("mfinstseg_unified",))

    title = md.splitlines()[0]
    assert rc == 0, payload["problems"]
    assert "bias ablation" not in title, title
    assert "mfinstseg_unified" in title, title


def test_title_does_not_say_bias_ablation_for_a_single_resnet_arm(tmp_path):
    """RED before the fix: T4's one-cell table claimed to be a bias ablation."""
    root = tmp_path / "results"
    write_arm(root, "resnet_cadsynth_ce_weighted", maker_kw={"weights": True})

    rc, md, payload = aggregate(root, tmp_path / "s.md", dataset="cadsynth",
                                arms=("resnet_cadsynth_ce_weighted",))

    title = md.splitlines()[0]
    assert rc == 0, payload["problems"]
    assert "bias ablation" not in title, title
    assert "resnet_cadsynth_ce_weighted" in title, title


def test_title_still_says_bias_ablation_when_it_is_one(tmp_path):
    """REGRESSION GUARD (green before and after): T1/T2 keep their title."""
    root = tmp_path / "results"
    for arm in aggregate_seeds.DEFAULT_ARMS:
        write_arm(root, "cadsynth_bias_{}".format(arm))

    rc, md, payload = aggregate(root, tmp_path / "s.md", dataset="cadsynth",
                                arms=aggregate_seeds.DEFAULT_ARMS)

    assert rc == 0, payload["problems"]
    assert md.splitlines()[0] == "# cadsynth bias ablation -- seed-averaged"


# ===================================================================== #
#  DEFECT 4 -- the selected epoch printed under the word "stopped"      #
# ===================================================================== #

def test_caption_does_not_claim_a_stop_epoch_it_does_not_have(tmp_path):
    """RED before the fix: `Runs stopped at epoch(s) 38 and selected ... 38`.

    Both halves of that sentence are `provenance['epoch']` and
    `provenance['selected_epoch']`, which are the SAME number on every sound
    run -- `build_provenance` is called inside the save-on-improvement branch.
    The epoch the loop last ran is not in provenance at all.
    """
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True,
              maker_kw={"selected": 38})

    rc, md, payload = aggregate(root, tmp_path / "s.md",
                                arms=("mfinstseg_unified",))

    assert rc == 0, payload["problems"]
    assert "stopped at epoch" not in md, [l for l in md.splitlines()
                                          if "stopped at epoch" in l]
    assert "Selected epoch(s) 38" in md, md
    # And it must say WHY there is no stop-epoch column, rather than leaving a
    # reader to assume the number is missing by accident.
    assert "not recorded in provenance" in md, md
    assert "check_convergence" in md


def test_epoch_disagreeing_with_selected_epoch_is_a_problem(tmp_path):
    """RED before the fix: rc 0. `check_convergence` calls this INCONSISTENT.

    A disagreement means a checkpoint was saved outside the
    save-on-improvement branch, so the weights that were scored are not the
    selected ones -- the last-epoch bug the selection rule removed.
    """
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True,
              maker_kw={"selected": 38},
              per_seed={1: {"selected": 38, "epoch": 44}})

    rc, md, payload = aggregate(root, tmp_path / "s.md",
                                arms=("mfinstseg_unified",))

    assert rc == 1, payload["problems"]
    joined = " ".join(payload["problems"])
    assert "seed 1" in joined, joined
    assert "selected_epoch" in joined, joined


# ===================================================================== #
#  DEFECT 5 -- a cap-bound run passed whenever its label lied           #
# ===================================================================== #

def _mislabelled_cap_bound(root: Path) -> Path:
    """One cell whose seed 1 cannot have early-stopped.

    Cap 60, patience 10, selected epoch 55: patience would have had to run to
    epoch 65, five past the cap. The run is cap-bound; only its label says
    `early_stop`.
    """
    return write_arm(root, "mfinstseg_unified", unified=True,
                     maker_kw={"selected": 38},
                     per_seed={1: {"selected": 55, "reason": "early_stop"}})


def test_mislabelled_cap_bound_run_is_refused(tmp_path):
    """RED before the fix: rc 0, `All checks pass ... provenance complete`."""
    root = tmp_path / "results"
    _mislabelled_cap_bound(root)

    rc, md, payload = aggregate(root, tmp_path / "s.md",
                                arms=("mfinstseg_unified",))

    joined = " ".join(payload["problems"])
    assert rc == 1, payload["problems"]
    assert "seed 1" in joined, joined
    assert "55" in joined and "60" in joined, joined
    assert "cap-bound" in joined, joined
    assert "NOT QUOTABLE" in md


def test_aggregate_seeds_and_check_convergence_agree_on_the_same_run(tmp_path):
    """RED before the fix: check_convergence said cap-bound, this said pass.

    This is the anti-drift test. The defect was not that one tool was wrong in
    isolation -- it was that three tools read the same artifact and returned
    two verdicts, so the answer depended on which one the operator ran. If
    `check_convergence`'s arithmetic ever moves, this test goes red rather than
    the two quietly parting company again.
    """
    root = tmp_path / "results"
    _mislabelled_cap_bound(root)

    row = check_convergence.audit_run(
        "mfinstseg_unified", 1, root / "mfinstseg_unified" / "seed1")
    rc, _md, payload = aggregate(root, tmp_path / "s.md",
                                 arms=("mfinstseg_unified",))

    assert row["budget_bound"] is True, row
    assert rc == 1, payload["problems"]


def test_honest_cap_label_is_still_refused(tmp_path):
    """REGRESSION GUARD (green before and after): the 22 Sep fix stays.

    That fix covered the honest label only; it must not be lost while the
    dishonest one is being closed.
    """
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True,
              maker_kw={"selected": 38},
              per_seed={1: {"selected": 20, "reason": "cap"}})

    rc, _md, payload = aggregate(root, tmp_path / "s.md",
                                 arms=("mfinstseg_unified",))

    assert rc == 1
    assert any("did not converge" in p for p in payload["problems"]), \
        payload["problems"]


def test_a_converged_run_is_still_accepted(tmp_path):
    """REGRESSION GUARD (green before and after): selecting 38 under patience

    10 with a cap of 60 leaves 12 epochs of headroom, so the run really could
    have stopped on patience. The new arithmetic must not fail it.
    """
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True,
              maker_kw={"selected": 38})

    rc, _md, payload = aggregate(root, tmp_path / "s.md",
                                 arms=("mfinstseg_unified",))

    assert rc == 0, payload["problems"]


# ===================================================================== #
#  DEFECT 6 -- no instance or bottom-face path at all                   #
# ===================================================================== #

def test_the_instance_head_is_aggregated(tmp_path):
    """RED before the fix: `instance` appeared zero times in the tool.

    T5's headline instance ROC-AUC 0.9973 +/- 0.0002 is the paper's
    instance-segmentation row and had no generated table behind it.
    """
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True, seeds=(0, 1, 2),
              per_seed={0: {"auc": 0.9970}, 1: {"auc": 0.9972},
                        2: {"auc": 0.9974}})

    rc, md, payload = aggregate(root, tmp_path / "s.md",
                                arms=("mfinstseg_unified",))

    assert rc == 0, payload["problems"]
    inst = payload["arms"]["mfinstseg_unified"]["instance"]
    assert inst["auc_mean"] == pytest.approx(0.9972)
    assert inst["n_seeds"] == 3
    assert inst["reportable"] == ["checkpoint_threshold"]
    assert "instance" in md.lower()
    assert "0.9972" in md, md


def test_the_bottom_face_head_is_aggregated(tmp_path):
    """RED before the fix: `bottom` appeared zero times in the tool."""
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True, seeds=(0, 1, 2),
              per_seed={0: {"bottom_f1": 0.94}, 1: {"bottom_f1": 0.95},
                        2: {"bottom_f1": 0.96}})

    rc, md, payload = aggregate(root, tmp_path / "s.md",
                                arms=("mfinstseg_unified",))

    assert rc == 0, payload["problems"]
    bot = payload["arms"]["mfinstseg_unified"]["bottom"]
    assert bot["f1_mean"] == pytest.approx(0.95)
    assert bot["accuracy_mean"] == pytest.approx(0.9827)
    assert "bottom" in md.lower()


def test_a_head_present_on_only_some_seeds_is_fatal(tmp_path):
    """RED before the fix: no error, and the head was simply never read.

    Same rule the confusion matrix and macro-F1 already have in this file: a
    mean over 2 of 3 printed in a row whose seed column says 3 is worse than a
    refusal.
    """
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True, seeds=(0, 1, 2),
              per_seed={2: {"instance": False}})

    with pytest.raises(SystemExit) as exc:
        aggregate(root, tmp_path / "s.md", arms=("mfinstseg_unified",))

    msg = str(exc.value)
    assert "instance" in msg, msg
    assert "[2]" in msg or "2" in msg, msg


def test_an_oracle_row_marked_reportable_is_refused(tmp_path):
    """RED before the fix: rc 0 and the oracle number tabled as a result.

    `oracle_threshold` is fitted on the TEST labels; the metrics file's own
    note says do not report it. Reading the instance head without reading
    `reportable` is how such a number reaches a paper.
    """
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True, seeds=(0, 1, 2),
              per_seed={1: {"reportable": "oracle_threshold"}})

    rc, _md, payload = aggregate(root, tmp_path / "s.md",
                                 arms=("mfinstseg_unified",))

    joined = " ".join(payload["problems"])
    assert rc == 1, payload["problems"]
    assert "oracle_threshold" in joined, joined
    assert "seed" in joined and "1" in joined, joined


def test_a_reported_row_missing_from_SOME_seeds_is_a_problem(tmp_path):
    """RED before the fix: the head was never read, so neither was the row.

    One cell is scored by one evaluator, so it cannot produce a
    `checkpoint_threshold` block on two seeds and not on the third. That is
    the MIXED case, and this file's standing rule makes the mixed case fatal
    to the row -- it is the one that could be averaged over a subset.
    """
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True, seeds=(0, 1, 2))
    trimmed = (root / "mfinstseg_unified" / "seed2"
               / "unified_eval_metrics.json")
    m = json.loads(trimmed.read_text(encoding="utf-8"))
    del m["instance"]["checkpoint_threshold"]
    trimmed.write_text(json.dumps(m), encoding="utf-8")

    rc, md, payload = aggregate(root, tmp_path / "s.md",
                                arms=("mfinstseg_unified",))

    joined = " ".join(payload["problems"])
    assert rc == 1, payload["problems"]
    assert "[2]" in joined, joined
    assert "NOT AGGREGATED" in md


def test_a_reported_row_missing_from_EVERY_seed_is_NOT_PERFORMED(tmp_path):
    """RED before the fix: the artifact said nothing about it at all.

    Nothing can be averaged over a subset here, so it is not a failure; and
    nothing was checked, so it is not a pass. This file keeps a third heading
    for exactly that, and the AUC beside it is threshold-free and unaffected.
    """
    root = tmp_path / "results"
    write_arm(root, "mfinstseg_unified", unified=True, seeds=(0, 1, 2))
    for s in (0, 1, 2):
        f = root / "mfinstseg_unified" / "seed{}".format(s) / "unified_eval_metrics.json"
        m = json.loads(f.read_text(encoding="utf-8"))
        del m["instance"]["checkpoint_threshold"]
        f.write_text(json.dumps(m), encoding="utf-8")

    rc, md, payload = aggregate(root, tmp_path / "s.md",
                                arms=("mfinstseg_unified",))

    assert rc == 0, payload["problems"]
    assert any("NOT AGGREGATED" in n for n in payload["not_performed"]), \
        payload["not_performed"]
    # The headline number still gets out.
    assert payload["arms"]["mfinstseg_unified"]["instance"]["auc_mean"] \
        == pytest.approx(0.997)


def test_a_flat_arm_has_no_unified_heads_and_that_is_not_a_fault(tmp_path):
    """REGRESSION GUARD (green before and after): T1-T4 are semantic-only."""
    root = tmp_path / "results"
    write_arm(root, "cadsynth_bias_both")

    rc, _md, payload = aggregate(root, tmp_path / "s.md", dataset="cadsynth",
                                 arms=("cadsynth_bias_both",))

    assert rc == 0, payload["problems"]
    arm = payload["arms"]["cadsynth_bias_both"]
    assert arm["instance"] is None
    assert arm["bottom"] is None


# ===================================================================== #
#  DEFECT 7 -- a stated justification that is false of this data        #
# ===================================================================== #

#: The claim itself, in the form common to every copy of it. The two files
#: spell the verb differently ("weights" in the docstrings, "would weight" in
#: the generated caption), so the token is the clause they share.
FALSE_REASON = ("a seed that saw few examples of a rare class equally with "
                "one that saw many")


def _source(name: str) -> str:
    return (CODEBASE / "runners" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["aggregate_seeds.py", "compare_arms.py"])
def test_the_false_pooling_justification_is_gone(name):
    """RED before the fix: the sentence is in both files, twice in one.

    It cannot arise here. Every run of a corpus scores the identical frozen
    held-out set, so no seed ever saw fewer examples of any class than another
    -- measured over all 85 committed runs, one distinct per-class support
    vector per corpus.
    """
    text = " ".join(_source(name).split())
    assert FALSE_REASON not in text, (
        "{} still justifies pooling with unequal class exposure".format(name))


@pytest.mark.parametrize("name", ["aggregate_seeds.py", "compare_arms.py"])
def test_the_real_pooling_reason_is_stated(name):
    """RED before the fix: neither file mentions the frozen split at all.

    Pooling stays -- it is what training-regiment.md section 1 asks for. What
    changes is the reason given for it.
    """
    text = " ".join(_source(name).split()).lower()
    assert "same held-out set" in text or "identical held-out set" in text, name
    assert "seed-averaged confusion matrix" in text, name


def test_the_generated_caption_no_longer_carries_the_false_reason(tmp_path):
    """RED before the fix: the sentence is copied into every artifact.

    The paper's caption is copied from this artifact, so a false reason here
    becomes a false reason in the paper.
    """
    root = tmp_path / "results"
    write_arm(root, "cadsynth_bias_both")

    _rc, md, _payload = aggregate(root, tmp_path / "s.md", dataset="cadsynth",
                                  arms=("cadsynth_bias_both",))

    assert FALSE_REASON not in " ".join(md.split()), md

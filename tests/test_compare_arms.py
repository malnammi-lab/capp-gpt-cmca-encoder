"""The three defects `runners/compare_arms.py` carried while nobody owned it.

This file is the tool that NAMES A WINNING ARM, and until 21 Sep 2026 it had no
tests at all. It also carried a bug already fixed in `aggregate_seeds.py`, plus a third that is
particular to it.

1. THE LOUD HALF (T5 unreadable).  ``acc = [m["overall_accuracy"] for m in
   runs.values()]`` is a direct subscript of the FLAT layout
   `src/evaluation/evaluate_semantic.py` writes. Tier T5 is five runs of the
   unified model, scored by `evaluate_unified.py`, which reports three tasks and
   nests the semantic ones under ``["semantic"]`` with ``n_faces`` in place of
   ``n_samples``. The tool raised ``KeyError: 'overall_accuracy'`` on all five.

2. THE SILENT HALF (worse).  ``if "confusion_matrix" in m`` is a MEMBERSHIP
   TEST, not a subscript. On the nested layout it is simply False, so the
   comparison printed with no summed matrix, no per-class F1 and no per-class
   deltas -- no error, no warning, exit code 0. The per-class delta view is the
   only thing this file offers over `aggregate_seeds.py`; it could vanish
   without leaving a mark. `test_the_membership_test_is_why_it_was_silent` pins
   WHY it was silent, so the tests below cannot be satisfied by a tool that
   merely stopped crashing.

3. NO FINGERPRINT CHECK AT ALL.  ``grep -n fingerprint`` over the file returned
   nothing, while the JSON it writes advertises `aggregate_seeds.py` and
   `check_tier.py` as the quotable tooling -- both of which compare every
   recorded ``code_fingerprint`` against `code_fingerprint()` of the running
   tree . This one ranked four arms without ever asking whether the runs
   came from the code on disk.

THREE MORE, 22 Sep 2026. The first was introduced by the fix for defect 1
above, and is worse than what it replaced.

4. THE SHORT AVERAGE (verifier probe 15).  ``if f is not None: f1.append(f)``
   dropped a seed from ONE metric's list and handed the short list to
   ``mean_std``, while ``_seeds`` -- and the ``seeds`` column beside the
   number -- still counted every run. With ``macro_f1``, ``mean_iou`` and
   ``macro_precision`` deleted from one of three seeds of arm ``both``::

       both  3  1200  92.20+/-0.00% 0.9120+/-0.0000 0.9020+/-0.0000 ...

   ``problems: []``, ``seeds_found: [0, 1, 2]``, exit code 0. Three of those
   columns are means over TWO seeds. Defect 1 at least crashed; this printed a
   number, and a +/-0.0000 that reads as three runs agreeing exactly.

5. AN ENTRY COUNT REPORTED AS A RUN COUNT.  A run records up to two
   fingerprints, one for the tree that trained it and one for the tree that
   scored it, so twelve stale runs printed as "24 recorded fingerprint(s) do
   not match".

6. A WINNER THAT SURVIVED A GREP.  Every fault except the fingerprint one was
   detected AFTER the table and the ``--- X vs Y ---`` sections had been
   printed, and those sections were byte-identical on a sound tree and an
   unsound one.

No fixture here comes from ``conftest.py``: those build B-Rep JSON, which has
nothing to do with this tool, and that file is shared.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CODEBASE = Path(__file__).resolve().parents[1]
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from runners import check_tier, compare_arms            # noqa: E402
from src.utils.provenance import code_fingerprint       # noqa: E402

#: A fingerprint no source tree will ever hash to: "the code that produced this
#: run is not the code you are running".
STALE_FP = "0000deadbeef0000"

#: Every seed must score the same held-out set, or the run trips the
#: different-test-sizes fault instead of the one under test.
N_TEST_FACES = 34338

#: The summed matrix of three seeds of CM below, and the per-class F1 it gives.
CM = [[900, 40], [60, 800]]
SUMMED_3 = [[2700, 120], [180, 2400]]
F1_FROM_SUMMED_3 = {0: 0.947368421, 1: 0.941176471}


def _provenance(seed: int, fp: str | None, *, eval_fp: str | None = None) -> dict:
    prov = {
        "seed": seed,
        "epoch": 42,
        "selected_epoch": 38,
        "stopping_reason": "early_stop",
        "use_class_weights": False,
        "epoch_budget_configured": 60,
        "config_hash": "c0ffeec0ffeec0ff",
        "git_commit": None,
        "gpu": "Tesla T4",
        "wall_time_s": 931.0,
        "torch": "2.8.0+cu128",
        "python": "3.12.11",
        "platform": "Linux-6.8.0-1063-aws-x86_64-with-glibc2.39",
        "recorded_at": "2026-09-21T00:00:00+00:00",
    }
    if fp is not None:
        prov["code_fingerprint"] = fp
    if eval_fp is not None:
        prov["evaluated"] = {
            "config_hash": "c0ffeec0ffeec0ff",
            "code_fingerprint": eval_fp,
            "gpu": "Tesla T4",
            "at": "2026-09-21T00:01:00+00:00",
        }
    return prov


def flat_metrics(name: str, seed: int, acc: float, fp: str | None,
                 *, eval_fp: str | None = None, cm=CM,
                 n_samples: int = N_TEST_FACES) -> dict:
    """What `evaluate_semantic.py` writes: the semantic numbers at the top."""
    m = {
        "experiment_name": name,
        "checkpoint": f"checkpoints/{name}/seed{seed}/best.pt",
        "n_classes": 2,
        "n_samples": n_samples,
        "overall_accuracy": acc,
        "macro_precision": 0.71,
        "macro_recall": 0.69,
        "macro_f1": 0.70,
        "mean_iou": 0.55,
        "provenance": _provenance(seed, fp, eval_fp=eval_fp),
    }
    if cm is not None:
        m["confusion_matrix"] = cm
    return m


def nested_metrics(name: str, seed: int, acc: float, fp: str | None,
                   *, cm=CM, n_faces: int = N_TEST_FACES) -> dict:
    """What `evaluate_unified.py` writes for T5: three tasks, semantic nested.

    Key names copied from `src/evaluation/evaluate_unified.py`, including
    ``n_faces`` where the flat layout says ``n_samples``.
    """
    sem = {
        "n_faces": n_faces,
        "n_classes": 2,
        "overall_accuracy": acc,
        "macro_precision": 0.71,
        "macro_recall": 0.69,
        "macro_f1": 0.70,
        "mean_iou": 0.55,
        "iou_per_class": {"0": 0.6, "1": 0.5},
        "per_class": {},
    }
    if cm is not None:
        sem["confusion_matrix"] = cm
    return {
        "experiment_name": name,
        "checkpoint": f"checkpoints/{name}/seed{seed}/best.pt",
        "epoch": 42,
        "split": "test (frozen ids)",
        "n_models": 700,
        "bias_type": "both",
        "semantic": sem,
        "bottom": {"n_faces": n_faces, "accuracy": 0.9, "f1": 0.8,
                   "confusion_matrix": [[1000, 50], [50, 700]]},
        "instance": {"n_pairs": 10000, "auc": 0.93},
        "provenance": _provenance(seed, fp),
    }


def write_tree(root: Path, dataset: str, fp: str | None, *, nested: bool = False,
               seeds=(0, 1, 2), arms=compare_arms.ARMS, **kw) -> Path:
    """A complete, otherwise-sound fake bias ablation under ``root``.

    Accuracies differ per arm and per seed so the comparison has something to
    rank; every run scores the same number of faces.
    """
    maker = nested_metrics if nested else flat_metrics
    fname = "unified_eval_metrics.json" if nested else "eval_metrics.json"
    for ci, arm in enumerate(arms):
        for s in seeds:
            run = root / f"{dataset}_bias_{arm}" / f"seed{s}"
            run.mkdir(parents=True, exist_ok=True)
            (run / fname).write_text(
                json.dumps(maker(f"{dataset}_bias_{arm}", s,
                                 0.90 + 0.01 * ci + 0.002 * s, fp, **kw)),
                encoding="utf-8")
    return root


def run_cli(results: Path, dataset: str = "cadsynth", seeds=("0", "1", "2"),
            extra=()) -> int:
    return compare_arms.main([*extra, "--dataset", dataset,
                              "--seeds", *seeds, "--results", str(results)])


def payload(results: Path, dataset: str = "cadsynth") -> dict:
    return json.loads((results / f"{dataset}_bias_comparison.json")
                      .read_text(encoding="utf-8"))


# --------------------------------------------------------------------- #
#  DEFECT 1 -- the loud half: T5's nested schema                        #
# --------------------------------------------------------------------- #

def test_load_arm_reads_the_nested_unified_schema():
    """Before the fix this raised KeyError: 'overall_accuracy'."""
    runs = {s: nested_metrics("mfinstseg_unified", s, 0.90 + 0.01 * s,
                              code_fingerprint())
            for s in (0, 1, 2)}

    out = compare_arms.summarise_runs("mfinstseg_unified", runs, [])

    assert out["_seeds"] == [0, 1, 2]
    assert out["overall_accuracy"][0] == pytest.approx((0.90 + 0.91 + 0.92) / 3)
    assert out["macro_f1"][0] == pytest.approx(0.70)
    assert out["macro_precision"][0] == pytest.approx(0.71)
    assert out["macro_recall"][0] == pytest.approx(0.69)
    assert out["mean_iou"][0] == pytest.approx(0.55)
    # `n_faces` is the nested layout's name for the flat `n_samples`; reading
    # the top level here raised KeyError too, and reading nothing would have
    # disabled the different-test-sizes check.
    assert out["n_samples"] == [N_TEST_FACES] * 3


def test_load_arm_still_reads_the_flat_schema():
    """The other arms of the run matrix must be untouched by the fix."""
    runs = {s: flat_metrics("cadsynth_bias_both", s, 0.90 + 0.01 * s,
                            code_fingerprint())
            for s in (0, 1, 2)}

    out = compare_arms.summarise_runs("cadsynth_bias_both", runs, [])

    assert out["overall_accuracy"][0] == pytest.approx((0.90 + 0.91 + 0.92) / 3)
    assert out["n_samples"] == [N_TEST_FACES] * 3
    assert out["per_class_f1"][0] == pytest.approx(F1_FROM_SUMMED_3[0])


def test_load_arm_refuses_a_metrics_file_with_no_accuracy_in_either_layout():
    """Not a seed to skip: a quietly smaller sample is the failure mode."""
    runs = {0: {"experiment_name": "x",
                "provenance": _provenance(0, code_fingerprint())}}

    with pytest.raises(SystemExit) as e:
        compare_arms.summarise_runs("both", runs, [])

    msg = str(e.value)
    assert "overall_accuracy" in msg and "['semantic']" in msg, msg
    assert "Re-score" in msg, msg


def test_load_arm_refuses_a_metrics_file_with_no_test_set_size():
    """A missing denominator silently disables the moved-split check."""
    m = flat_metrics("both", 0, 0.9, code_fingerprint())
    del m["n_samples"]

    with pytest.raises(SystemExit) as e:
        compare_arms.summarise_runs("both", {0: m}, [])

    msg = str(e.value)
    assert "n_samples" in msg and "n_faces" in msg, msg
    assert "52.26" in msg, "say which real incident this guards against"


def test_cli_reads_a_whole_nested_tier_end_to_end(tmp_path, capsys):
    """What T5's four cells would have done: KeyError after 5 completed runs."""
    results = write_tree(tmp_path / "results", "mfinstseg", code_fingerprint(),
                         nested=True)

    rc = run_cli(results, dataset="mfinstseg")
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "90.20+/-0.20%" in out, out
    assert payload(results, "mfinstseg")["arms"]["similarity"][
        "overall_accuracy_mean"] == pytest.approx(0.902)


# --------------------------------------------------------------------- #
#  DEFECT 2 -- the silent half: the membership test                     #
# --------------------------------------------------------------------- #

def test_the_membership_test_is_why_it_was_silent():
    """Pins WHY the fix was needed, not merely that it works.

    The old code guarded the confusion matrix with ``if "confusion_matrix" in
    m``. If that expression were True on a nested file, the branch would have
    run and the loss of the per-class view would have had some other cause --
    and the assertions below would be proving nothing.
    """
    m = nested_metrics("mfinstseg_unified", 0, 0.9, code_fingerprint())

    assert "confusion_matrix" not in m, (
        "the membership test would have found the matrix, so the silent "
        "failure is not attributable to it")
    assert "overall_accuracy" not in m, (
        "the subscript would not have raised either")
    # and both are reachable through the accessor, one level down
    assert check_tier.semantic_block(m)["confusion_matrix"] == CM


def test_nested_layout_yields_the_summed_matrix_and_per_class_f1():
    """The silent half, fixed: per-class F1 from the SUMMED matrix."""
    runs = {s: nested_metrics("mfinstseg_unified", s, 0.90 + 0.01 * s,
                              code_fingerprint())
            for s in (0, 1, 2)}

    out = compare_arms.summarise_runs("mfinstseg_unified", runs, [])

    assert out["_has_confusion_matrix"] is True, (
        "per-class F1 vanished for the nested layout without an error")
    assert out["confusion_matrix_summed"].tolist() == SUMMED_3
    assert out["per_class_f1"][0] == pytest.approx(F1_FROM_SUMMED_3[0])
    assert out["per_class_f1"][1] == pytest.approx(F1_FROM_SUMMED_3[1])
    assert out["per_class_support"] == {0: 2820, 1: 2580}


def test_cli_on_the_nested_schema_prints_the_per_class_deltas(tmp_path, capsys):
    """End to end: the section that silently disappeared is back."""
    results = write_tree(tmp_path / "results", "mfinstseg", code_fingerprint(),
                         nested=True)

    rc = run_cli(results, dataset="mfinstseg")
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "--- adjacency vs similarity ---" in out
    assert "largest gains" in out, out
    assert "no confusion matrix recorded" not in out, out
    assert "weakest classes" in out


def test_a_seed_without_a_confusion_matrix_is_refused_not_dropped():
    """Summing a subset would print per-class F1 over fewer runs than the mean."""
    runs = {s: nested_metrics("mfinstseg_unified", s, 0.90 + 0.01 * s,
                              code_fingerprint())
            for s in (0, 1, 2)}
    del runs[2]["semantic"]["confusion_matrix"]

    with pytest.raises(SystemExit) as e:
        compare_arms.summarise_runs("mfinstseg_unified", runs, [])

    msg = str(e.value)
    assert "[0, 1]" in msg and "[2]" in msg, msg
    assert "Re-score" in msg, msg


def test_an_arm_with_no_confusion_matrix_at_all_is_a_fault(tmp_path, capsys):
    """An arm with no confusion matrix exits 2 and is recorded as a fault."""
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint(),
                         cm=None)

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    assert "no confusion matrix recorded for arm(s)" in out, out
    assert any("no confusion matrix recorded" in p
               for p in payload(results)["problems"])


def test_confusion_matrices_of_different_shapes_are_refused():
    runs = {0: flat_metrics("both", 0, 0.90, code_fingerprint()),
            1: flat_metrics("both", 1, 0.91, code_fingerprint(),
                            cm=[[9, 4, 1], [6, 8, 1], [1, 1, 9]])}

    with pytest.raises(SystemExit) as e:
        compare_arms.summarise_runs("both", runs, [])

    assert "differ in shape across seeds" in str(e.value)


# --------------------------------------------------------------------- #
#  DEFECT 3 -- it never asked which code produced the winner            #
# --------------------------------------------------------------------- #

def _fingerprint_problems(problems):
    return [p for p in problems
            if "do not match the code in this tree" in p
            or "recorded no code_fingerprint" in p]


def test_cli_rejects_a_comparison_of_runs_this_tree_did_not_produce(tmp_path, capsys):
    """The defect: four arms ranked, exit 0, fingerprint never mentioned."""
    results = write_tree(tmp_path / "results", "cadsynth", STALE_FP)

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, "a comparison of entirely superseded runs exited zero"
    assert "do not match the code in this tree" in out, out
    assert STALE_FP in out
    assert code_fingerprint() in out, (
        "the message must name the LIVE fingerprint, or the reader cannot tell "
        "which side moved")
    assert "Re-run the affected arms" in out, "say what to do about it"


def test_the_stale_tree_is_otherwise_sound(tmp_path, capsys):
    """Attributes the rejection above to the fingerprint and nothing else.

    The same tree with this tree's fingerprint passes, so the non-zero exit
    cannot be blamed on a missing seed, a moved split or an absent matrix.
    """
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint())

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "recorded fingerprint(s) match this tree" in out, out
    assert code_fingerprint() in out
    assert payload(results)["fingerprints_match_this_tree"] is True
    assert payload(results)["problems"] == []


def test_a_stale_SCORING_fingerprint_is_caught_too(tmp_path, capsys):
    """Trained by this tree, scored by an older one -- equally untraceable.

    `provenance_from_checkpoint` records the evaluating tree's fingerprint
    separately under ``provenance["evaluated"]``, so a number produced by
    today's weights through yesterday's scorer would otherwise pass.
    """
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint(),
                         eval_fp=STALE_FP)

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    assert "(scored by)" in out, out
    assert STALE_FP in out


def test_a_run_with_no_recorded_fingerprint_is_a_fault_not_a_blank(tmp_path, capsys):
    """"Nothing recorded" is the case a .get() default turns into "nothing to
    check". An unfingerprinted run is untraceable, not presumed current."""
    results = write_tree(tmp_path / "results", "cadsynth", None)

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    assert "recorded no code_fingerprint at all" in out, out
    assert "runners/run_all.py" in out, "say what to do about it"
    assert _fingerprint_problems(payload(results)["problems"])


def test_the_fingerprint_verdict_is_written_into_the_artifact(tmp_path, capsys):
    """stdout scrolls; the JSON is what a later reader has.

    This script's product is a file on disk that outlives the terminal it was
    printed in, so the verdict has to travel with the numbers.
    """
    results = write_tree(tmp_path / "results", "cadsynth", STALE_FP)

    run_cli(results)
    capsys.readouterr()
    got = payload(results)

    assert got["code_fingerprint_live"] == code_fingerprint()
    assert set(got["code_fingerprint_stale"].values()) == {STALE_FP}
    assert got["fingerprints_match_this_tree"] is False
    assert got["quotable"] is False
    assert _fingerprint_problems(got["problems"]), got["problems"]
    # every run is listed, training side, four arms x three seeds
    assert len(got["code_fingerprint_recorded"]) == 12


def test_the_fingerprint_banner_prints_above_the_table(tmp_path, capsys):
    """A reader who copies the table never scrolls to the bottom."""
    results = write_tree(tmp_path / "results", "cadsynth", STALE_FP)

    run_cli(results)
    out = capsys.readouterr().out

    banner = out.index("do not match the code in this tree")
    table = out.index("overall acc")
    winner = out.index("--- adjacency vs similarity ---")
    assert banner < table < winner, (
        "the stale-code warning sits below the numbers it invalidates")


# --------------------------------------------------------------------- #
#  other faults are reported and exit non-zero                          #
# --------------------------------------------------------------------- #

def test_arms_scored_on_different_test_sets_are_a_fault(tmp_path, capsys):
    """The 52.26-vs-64.24 defect: it printed "!!" and exited 0."""
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint(),
                         arms=("similarity", "adjacency", "both"))
    write_tree(results, "cadsynth", code_fingerprint(), arms=("shuffled",),
               n_samples=N_TEST_FACES - 11)

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    assert "different test-set sizes" in out, out
    assert "70_10_20_seed42" in out


def test_a_missing_seed_is_still_a_fault(tmp_path, capsys):
    """Pre-existing behaviour, pinned so the exit-code rework cannot drop it."""
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint())
    (results / "cadsynth_bias_shuffled" / "seed2" / "eval_metrics.json").unlink()

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    assert "incomplete arms" in out
    assert "missing seeds [2]" in out


def test_an_entirely_absent_arm_is_reported_and_non_zero(tmp_path, capsys):
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint(),
                         arms=("similarity", "adjacency", "both"))

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    assert "no runs found for: shuffled" in out
    assert any("shuffled" in p for p in payload(results)["problems"])


def test_no_runs_at_all_exits_one(tmp_path, capsys):
    rc = run_cli(tmp_path / "empty")

    assert rc == 1
    assert "Run the tier with runners/run_all.py first" in capsys.readouterr().err


# --------------------------------------------------------------------- #
#  the three tools must not drift apart again                           #
# --------------------------------------------------------------------- #

def test_compare_arms_uses_check_tiers_accessor_rather_than_a_third_copy():
    """One accessor for the two layouts, imported by all three readers."""
    assert compare_arms.semantic_block is check_tier.semantic_block
    assert compare_arms.headline_metrics is check_tier.headline_metrics


def test_compare_arms_resolves_the_layout_through_aggregate_seeds():
    """One implementation of "where does an arm's results live"."""
    from runners import aggregate_seeds

    assert compare_arms.arm_dir is aggregate_seeds.arm_dir
    assert compare_arms.load_runs is aggregate_seeds.load_runs


# --------------------------------------------------------------------- #
#  22 Sep 2026 -- verifier probe 15: the fix above introduced this one  #
# --------------------------------------------------------------------- #
#
#  `summarise_runs` collected each metric with `if v is not None:
#  xs.append(v)` and handed the SHORT list to `mean_std`, while `_seeds` --
#  and the `seeds` column printed beside the number -- still counted every
#  run. With macro_f1, mean_iou and macro_precision deleted from ONE of three
#  seeds of arm `both`, the row read
#
#      both  3  1200  92.20+/-0.00% 0.9120+/-0.0000 0.9020+/-0.0000 ...
#
#  with problems: [], seeds_found: [0, 1, 2] and exit code 0. Three of those
#  columns are means over TWO seeds, and two identical values print as
#  +/-0.0000, which reads as three runs agreeing exactly.


def test_a_metric_missing_from_one_seed_is_not_averaged_over_the_rest():
    """The probe, at the unit the defect lives in."""
    runs = {s: flat_metrics("cadsynth_bias_both", s, 0.922,
                            code_fingerprint())
            for s in (0, 1, 2)}
    for k in ("macro_f1", "mean_iou", "macro_precision"):
        del runs[1][k]

    with pytest.raises(SystemExit) as e:
        compare_arms.summarise_runs("both", runs, [])

    msg = str(e.value)
    # it must name the arm, the metrics, and BOTH sides of the split -- an
    # operator cannot go and re-score "some seeds"
    assert "arm both" in msg, msg
    for k in ("macro_f1", "mean_iou", "macro_precision"):
        assert k in msg, msg
    assert "[0, 2]" in msg and "[1]" in msg, msg
    assert "a mean over 2 of 3" in msg, msg
    assert "evaluate_semantic.py" in msg, "say what to do about it"
    assert "--seeds" in msg, "and name the other way out"
    # macro_recall was recorded by every seed and is not part of the complaint
    assert "macro_recall" not in msg, msg


def test_the_short_average_is_never_reached_not_merely_reported():
    """The refusal comes before `mean_std` sees the short list.

    A version that computed the 2-seed mean and then complained about it would
    still have put the wrong number in the artifact, which is where a number
    outlives the complaint printed beside it.
    """
    runs = {s: flat_metrics("cadsynth_bias_both", s, 0.922,
                            code_fingerprint())
            for s in (0, 1, 2)}
    del runs[1]["macro_f1"]

    with pytest.raises(SystemExit):
        compare_arms.summarise_runs("both", runs, [])


def test_probe_15_end_to_end_refuses_instead_of_printing_a_three_seed_row(
        tmp_path, capsys):
    """The verifier's probe: exit 0 and a fabricated +/-0.0000 before this."""
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint())
    run = results / "cadsynth_bias_both" / "seed1" / "eval_metrics.json"
    m = json.loads(run.read_text(encoding="utf-8"))
    for k in ("macro_f1", "mean_iou", "macro_precision"):
        del m[k]
    run.write_text(json.dumps(m), encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        run_cli(results)

    out = capsys.readouterr().out
    assert "0.0000" not in out, out
    assert "both" not in out, "the row was printed before the refusal"
    assert "a mean over 2 of 3" in str(e.value)
    # and no artifact claiming three seeds was left behind
    assert not (results / "cadsynth_bias_comparison.json").exists()


def test_an_aborted_run_invalidates_the_artifact_it_could_not_rewrite(
        tmp_path, capsys):
    """Aborting is right, but a stale "sound": true file must not survive it.

    The fingerprint check reports rather than raises precisely because an
    abort leaves the previous <dataset>_bias_comparison.json on disk, unmarked,
    long after the stderr it printed to has scrolled away. So the refusal path
    stamps that file before it reads anything.
    """
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint())
    assert run_cli(results) == 0
    capsys.readouterr()
    assert payload(results)["sound"] is True

    # the same tree, one seed of `both` now missing three metrics
    run = results / "cadsynth_bias_both" / "seed1" / "eval_metrics.json"
    m = json.loads(run.read_text(encoding="utf-8"))
    for k in ("macro_f1", "mean_iou", "macro_precision"):
        del m[k]
    run.write_text(json.dumps(m), encoding="utf-8")

    with pytest.raises(SystemExit):
        run_cli(results)
    capsys.readouterr()

    got = payload(results)
    assert got["sound"] is False, "the superseded verdict outlived the refusal"
    assert got["complete"] is False
    assert "arms" not in got, "numbers from the previous run were left readable"
    assert any("INCOMPLETE RUN" in p for p in got["problems"]), got["problems"]
    assert "invalidated_at" in got


def test_a_metric_absent_from_every_seed_still_prints_a_dash(tmp_path, capsys):
    """The distinction the refusal turns on, pinned.

    "macro P and R stay optional on purpose" is about a column NO seed
    recorded: `mean_std` returns (None, None), the cell prints "-", and
    nothing is fabricated. Making that fatal too would have been the easy
    over-correction, and it would stop the tool reading any run scored before
    mIoU was added.
    """
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint())
    for arm in compare_arms.ARMS:
        for s in (0, 1, 2):
            f = (results / ("cadsynth_bias_" + arm) / ("seed%d" % s)
                 / "eval_metrics.json")
            m = json.loads(f.read_text(encoding="utf-8"))
            del m["mean_iou"]
            f.write_text(json.dumps(m), encoding="utf-8")

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "recorded by NO seed of that arm" in out, out
    assert "0 of 3 seed(s) recorded: mean_iou" in out, out
    assert compare_arms.UNSOUND not in out, "a '-' column is not a fault"
    got = payload(results)
    assert got["arms"]["both"]["metric_n"]["mean_iou"] == 0
    assert got["arms"]["both"]["mean_iou_mean"] is None
    assert got["problems"] == []


def test_the_artifact_records_the_true_n_behind_every_mean(tmp_path, capsys):
    """`seeds_found: [0, 1, 2]` beside a 2-seed mean was unfalsifiable in JSON."""
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint())

    run_cli(results)
    capsys.readouterr()
    arm = payload(results)["arms"]["both"]

    assert set(arm["metric_n"]) == set(compare_arms.METRIC_COLUMNS)
    for k, n in arm["metric_n"].items():
        assert n == len(arm["seeds_found"]), (k, n, arm["seeds_found"])


def test_fmt_refuses_a_mean_whose_n_is_not_the_rows_seed_count():
    """The second net, at the point the number becomes visible.

    `summarise_runs` already refuses to build such a summary. This is what
    stands between any FUTURE caller -- or a regression in that check -- and a
    printed table, so it is tested directly rather than assumed unreachable.
    """
    with pytest.raises(SystemExit) as e:
        compare_arms._fmt((0.892, 0.0), 2, 3, "macro_f1", "both")

    msg = str(e.value)
    assert "mean over 2 seed(s) but the row says 3" in msg, msg
    assert "probe 15" in msg, msg
    # the honest cases still format
    assert "-" in compare_arms._fmt((None, None), 0, 3, "mean_iou", "both")
    assert "0.8920" in compare_arms._fmt((0.892, 0.0), 3, 3, "macro_f1", "both")


def test_fmt_refuses_a_column_with_values_but_no_mean():
    with pytest.raises(SystemExit) as e:
        compare_arms._fmt((None, None), 2, 3, "macro_f1", "both")

    assert "no mean but 2 recorded value(s)" in str(e.value)


# --------------------------------------------------------------------- #
#  22 Sep 2026 -- a fingerprint ENTRY count is not a run count          #
# --------------------------------------------------------------------- #

def test_stale_fingerprints_report_runs_and_entries_as_different_numbers(
        tmp_path, capsys):
    """12 runs x (trained, scored) printed as "24 ... do not match"."""
    results = write_tree(tmp_path / "results", "cadsynth", STALE_FP,
                         eval_fp=STALE_FP)

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    assert "12 run(s) -- 24 recorded fingerprint(s) -- do not match" in out, out
    assert "the entry count is not a run count" in out, out
    # and the runs are NAMED, so "re-run the affected arms" is actionable
    assert "both seed 0" in out, out

    got = payload(results)
    assert got["code_fingerprint_stale_run_count"] == 12
    assert got["code_fingerprint_stale_entry_count"] == 24
    assert len(got["code_fingerprint_stale_runs"]) == 12
    assert len(got["code_fingerprint_stale"]) == 24
    assert "both seed 1" in got["code_fingerprint_stale_runs"]


def test_a_run_stale_on_one_side_only_is_one_run_and_one_entry(tmp_path, capsys):
    """Trained by this tree, scored by an older one: 12 runs, 12 stale entries.

    The two counts coincide here, which is exactly why counting entries went
    unnoticed. Both are still reported.
    """
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint(),
                         eval_fp=STALE_FP)

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    assert "12 run(s) -- 12 recorded fingerprint(s) -- do not match" in out, out
    got = payload(results)
    assert got["code_fingerprint_stale_run_count"] == 12
    assert got["code_fingerprint_stale_entry_count"] == 12


def test_the_passing_fingerprint_line_also_separates_runs_from_entries(
        tmp_path, capsys):
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint(),
                         eval_fp=code_fingerprint())

    assert run_cli(results) == 0
    out = capsys.readouterr().out

    assert "all 24 recorded fingerprint(s) match this tree" in out, out
    assert "across 12 run(s)" in out, out


# --------------------------------------------------------------------- #
#  22 Sep 2026 -- a winner must not survive a grep of an unsound log    #
# --------------------------------------------------------------------- #

def _unmarked_content(out):
    return [l for l in out.splitlines()
            if l.strip() and compare_arms.UNSOUND not in l]


def test_no_line_of_an_unsound_report_can_be_grepped_without_the_verdict(
        tmp_path, capsys):
    """Probes 4, 5, 6 and 8: the ranking and the "--- X vs Y ---" sections
    printed on every fault path, so an operator of an 85-run batch who greps a
    captured log for the winning arm found one on a stale tree."""
    results = write_tree(tmp_path / "results", "cadsynth", STALE_FP)

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    assert _unmarked_content(out) == [], _unmarked_content(out)
    # the greps an operator actually runs
    assert not [l for l in out.splitlines() if l.startswith("---")]
    assert not [l for l in out.splitlines() if l.startswith("both")]
    for line in out.splitlines():
        if "vs similarity" in line or "macro F1" in line:
            assert line.startswith(compare_arms.UNSOUND), line


def test_a_sound_report_carries_no_mark_anywhere(tmp_path, capsys):
    """Or the mark means nothing."""
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint())

    assert run_cli(results) == 0
    out = capsys.readouterr().out

    assert compare_arms.UNSOUND not in out, out
    assert "--- adjacency vs similarity ---" in out, out
    assert payload(results)["sound"] is True


@pytest.mark.parametrize("fault", ["stale_fp", "no_fp", "missing_arm",
                                   "missing_seed", "no_cm", "mixed_n"])
def test_every_fault_path_marks_the_whole_report(tmp_path, capsys, fault):
    """Not just the fingerprint one.

    Detection now happens before any output at all.
    """
    root = tmp_path / "results"
    fp = code_fingerprint()
    if fault == "stale_fp":
        results = write_tree(root, "cadsynth", STALE_FP)
    elif fault == "no_fp":
        results = write_tree(root, "cadsynth", None)
    elif fault == "missing_arm":
        results = write_tree(root, "cadsynth", fp,
                             arms=("similarity", "adjacency", "both"))
    elif fault == "missing_seed":
        results = write_tree(root, "cadsynth", fp)
        (results / "cadsynth_bias_shuffled" / "seed2"
         / "eval_metrics.json").unlink()
    elif fault == "no_cm":
        results = write_tree(root, "cadsynth", fp, cm=None)
    else:
        results = write_tree(root, "cadsynth", fp,
                             arms=("similarity", "adjacency", "both"))
        write_tree(results, "cadsynth", fp, arms=("shuffled",),
                   n_samples=N_TEST_FACES - 11)

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    assert _unmarked_content(out) == [], _unmarked_content(out)
    assert payload(results)["sound"] is False


def test_the_fail_banner_lists_every_fault_above_the_table(tmp_path, capsys):
    """Not only the fingerprint ones. "A reader who copies the table never
    scrolls to the bottom" was only ever acted on for one fault."""
    results = write_tree(tmp_path / "results", "cadsynth", code_fingerprint(),
                         cm=None)
    (results / "cadsynth_bias_shuffled" / "seed2" / "eval_metrics.json").unlink()

    rc = run_cli(results)
    out = capsys.readouterr().out

    assert rc == 2, out
    banner = out.index("THIS COMPARISON IS NOT SOUND")
    assert banner < out.index("no confusion matrix recorded for arm(s)"), out
    assert banner < out.index("incomplete arms"), out
    assert banner < out.index("overall acc"), out


def test_the_not_quotable_banner_names_the_dataset_it_was_run_on(tmp_path,
                                                                 capsys):
    """`NOT_QUOTABLE` is a template; an unformatted "{ds}" is an instruction
    nobody can follow. Caught by running the tool, not by reading it."""
    results = write_tree(tmp_path / "results", "mfinstseg", code_fingerprint())

    run_cli(results, dataset="mfinstseg")
    out = capsys.readouterr().out

    assert "{ds}" not in out, out
    assert "aggregate_seeds.py --dataset mfinstseg" in out, out

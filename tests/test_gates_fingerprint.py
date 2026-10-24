"""The two gate defects in the fingerprint and metrics-schema checks.

the gates never asked whether the code still existed
------------------------------------------------------------
`check_tier.py` and `aggregate_seeds.py` both collected every run's recorded
``code_fingerprint`` and complained if they DISAGREED WITH EACH OTHER. That is
a real check -- it catches an arm re-run halfway through a tier -- but it is not
the check the field was added for. A tier produced entirely by superseded code
agrees with itself perfectly and sails through: the committed T1 metrics all
record ``17a31863a847eb83``, this tree hashes something else, and nothing in the
repository noticed. The fix puts ``code_fingerprint()`` of the RUNNING tree on
one side of the comparison.

The tests below are built so that the old, agreement-only logic cannot pass
them: `test_the_old_agreement_check_alone_would_have_let_this_through` asserts
that the very tree the gate rejects has exactly ONE distinct recorded
fingerprint, so the pre-fix check had nothing to fire on.

the table step could not read T5's metrics
---------------------------------------------------
`evaluate_semantic.py` writes a FLAT metrics file; `evaluate_unified.py` (T5)
scores three tasks and nests the semantic ones under ``["semantic"]``.
`aggregate_seeds.summarise_arm` subscripted the top level, so T5's table step
would have raised ``KeyError: 'overall_accuracy'`` after five completed runs --
and its confusion-matrix branch, a membership test rather than a subscript,
would have failed *silently*, dropping the per-class IoU without a word.

No fixture in this file comes from ``conftest.py``: the corpus fixtures there
build B-Rep JSON, which has nothing to do with gates, and the file is shared.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

CODEBASE = Path(__file__).resolve().parents[1]
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from runners import aggregate_seeds, check_tier          # noqa: E402
from src.utils.provenance import code_fingerprint        # noqa: E402

#: A fingerprint no source tree will ever hash to. Used as "the code that made
#: this run is not the code you are running".
STALE_FP = "0000deadbeef0000"

#: The four cells of tier T1, as `runners/run_all.py` names them.
T1_CELLS = ("cadsynth_bias_similarity", "cadsynth_bias_adjacency",
            "cadsynth_bias_both", "cadsynth_bias_shuffled")

#: Every seed must score the same held-out set or check_tier fails it for a
#: different reason than the one under test.
N_TEST_FACES = 34338


#: [FIXTURE FIX 26 Sep 2026] `src/utils/provenance.py:157-172` hashes
#: the resolved config INCLUDING `seed` and `paths`, so the five seeds of an
#: arm carry five DIFFERENT digests, and `:355` stamps one arm_config_hash on
#: every run -- constant across an arm, distinct between arms. All 17 arms in
#: `results/` match that: 5 distinct config_hash values and exactly 1
#: arm_config_hash each.
def _arm_hash(arm: str) -> str:
    """One digest per experiment: constant across seeds, distinct across arms."""
    return hashlib.sha256(f"arm:{arm}".encode()).hexdigest()[:16]


def _config_hash(arm: str, seed: int) -> str:
    """One digest per RUN: the seed is inside the config that is hashed."""
    return hashlib.sha256(f"cfg:{arm}:seed{seed}".encode()).hexdigest()[:16]


#: Sentinel for "leave this key out entirely", distinct from None, which is a
#: legal recorded value for git_commit.
_ABSENT = object()


def _provenance(seed: int, fp, *, eval_fp=_ABSENT, epoch: int = 42,
                git_commit=None, arm: str = "fixture_arm") -> dict:
    """A provenance block that passes `check_regiment_fields` completely.

    Everything the regiment's section-4 checklist requires is present, so any
    error the gate reports on a tree built from this is about the fingerprint
    and not about a field that happened to be missing.

    [21 Sep 2026] ``eval_fp`` now defaults to ``fp`` rather than to "omit the
    block". Every evaluator in the repository writes
    ``provenance["evaluated"]["code_fingerprint"]`` -- both branches of
    `src/utils/provenance.provenance_from_checkpoint` stamp it, and the four
    committed T1 metrics files on disk all carry it -- so a fixture without one
    was not a realistic run, and a test suite built on one is how a missing
    scoring fingerprint came to look acceptable.

    Args:
        fp: The training fingerprint. ``None`` omits the ``code_fingerprint``
            key entirely, which is the shape defect 1 was about.
        eval_fp: The scoring fingerprint. Defaults to ``fp``; ``None`` omits the
            whole ``evaluated`` block.
        git_commit: Recorded as-is. A run carrying only a commit is the exact
            case `check_regiment_fields` accepts.
    """
    if eval_fp is _ABSENT:
        eval_fp = fp
    prov = {
        "seed": seed,
        # [FIXED 22 Sep 2026] Running past the best epoch
        # cannot happen under the selection rule, and the cap audit -- now wired into
        # check_tier -- correctly rejected it, which is how the stale fixture
        # was found.
        #
        # All four trainers build provenance INSIDE the
        # `if ep_vloss < best_val_loss:` branch (train_semantic.py:544/563,
        # train_unified.py:525/543, train_link.py:439/450,
        # train_resnet.py:412/430) and set `best_epoch = epoch` in that same
        # branch. So the checkpoint on disk is the one written at the last
        # IMPROVING epoch, and its provenance `epoch` IS `selected_epoch`.
        # A disagreement means a checkpoint was saved outside that branch --
        # the last-epoch bug the selection rule removed.
        "epoch": epoch,
        "selected_epoch": epoch,
        "stopping_reason": "early_stop",
        "use_class_weights": False,
        "epoch_budget_configured": 60,
        "config_hash": _config_hash(arm, seed),
        "arm_config_hash": _arm_hash(arm),
        "git_commit": git_commit,
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
            "config_hash": _config_hash(arm, seed),
            "code_fingerprint": eval_fp,
            "gpu": "Tesla T4",
            "at": "2026-09-21T00:01:00+00:00",
        }
    return prov


def _flat_metrics(name: str, seed: int, acc: float, fp, *,
                  eval_fp=_ABSENT, git_commit=None) -> dict:
    """What `evaluate_semantic.py` writes: metrics at the top level."""
    return {
        "experiment_name": name,
        "checkpoint": f"checkpoints/{name}/seed{seed}/best.pt",
        "n_classes": 2,
        "n_samples": N_TEST_FACES,
        "overall_accuracy": acc,
        "macro_precision": 0.71,
        "macro_recall": 0.69,
        "macro_f1": 0.70,
        "mean_iou": 0.55,
        "confusion_matrix": [[900, 40], [60, 800]],
        "provenance": _provenance(seed, fp, eval_fp=eval_fp,
                                  git_commit=git_commit, arm=name),
    }


def _nested_metrics(name: str, seed: int, acc: float, fp: str) -> dict:
    """What `evaluate_unified.py` writes for T5: three tasks, semantic nested.

    Key names copied from ``evaluate_unified.py``'s metrics dict, including
    ``n_faces`` where the flat layout says ``n_samples``.
    """
    return {
        "experiment_name": name,
        "checkpoint": f"checkpoints/{name}/seed{seed}/best.pt",
        "epoch": 42,
        "split": "test (frozen ids)",
        "n_models": 700,
        "attributes": "standardized",
        "bias_type": "both",
        "semantic": {
            "n_faces": N_TEST_FACES,
            "n_classes": 2,
            "overall_accuracy": acc,
            "macro_precision": 0.71,
            "macro_recall": 0.69,
            "macro_f1": 0.70,
            "mean_iou": 0.55,
            "iou_per_class": {"0": 0.6, "1": 0.5},
            "per_class": {},
            "confusion_matrix": [[900, 40], [60, 800]],
        },
        "bottom": {
            "n_faces": N_TEST_FACES, "n_positive": 400, "accuracy": 0.9,
            "precision": 0.8, "recall": 0.8, "f1": 0.8, "mean_iou": 0.7,
            "confusion_matrix": [[1000, 50], [50, 700]],
        },
        # [FIXTURE FIX 26 Sep 2026] Every real T5 file carries `reportable`:
        # `evaluate_unified.py:624` writes "checkpoint_threshold" only
        # when it also wrote the row, and
        # results/mfinstseg_unified/seed0 has both rows. Keys copied
        # from that file.
        "instance": {
            "n_pairs": 10000, "n_positive": 2000, "auc": 0.93,
            "reportable": "checkpoint_threshold", "note": "",
            "checkpoint_threshold": {
                "precision": 0.80, "recall": 0.79, "f1": 0.795,
                "accuracy": 0.91, "threshold": 0.87,
                "confusion_matrix": [[7600, 400], [800, 1200]],
            },
            "oracle_threshold": {
                "precision": 0.81, "recall": 0.80, "f1": 0.805,
                "accuracy": 0.92, "threshold": 0.86,
                "confusion_matrix": [[7650, 350], [750, 1250]],
            },
        },
        "provenance": _provenance(seed, fp, arm=name),
    }


def _write_tier(root: Path, fp, *, seeds=(0, 1), eval_fp=_ABSENT,
                cells=T1_CELLS, git_commit=None) -> Path:
    """Build a complete, otherwise-sound fake tier under ``root``.

    Accuracies differ per seed (the seed-spread check rejects identical ones)
    and per cell, and every run scores the same number of faces.
    """
    for ci, cell in enumerate(cells):
        for s in seeds:
            run = root / cell / f"seed{s}"
            run.mkdir(parents=True, exist_ok=True)
            acc = 0.90 + 0.01 * ci + 0.002 * s
            (run / "eval_metrics.json").write_text(
                json.dumps(_flat_metrics(cell, s, acc, fp, eval_fp=eval_fp,
                                         git_commit=git_commit)),
                encoding="utf-8")
    return root


# --------------------------------------------------------------------- #
#  check_tier.py                                               #
# --------------------------------------------------------------------- #

def _fingerprint_errors(errors: list[str]) -> list[str]:
    return [e for e in errors if "do not match the code in this tree" in e]


def test_check_tier_rejects_a_tier_recorded_by_code_that_is_not_this_tree(tmp_path):
    """The defect, reproduced: every run agrees, and every run is stale."""
    results = _write_tier(tmp_path / "results", STALE_FP)

    errors, _warns = check_tier.check("T1", results, [0, 1])

    fp_errs = _fingerprint_errors(errors)
    assert fp_errs, (
        "check_tier accepted a tier whose every run records a fingerprint that "
        f"is not this tree's. Errors reported: {errors}")
    assert len(fp_errs) == 1
    assert STALE_FP in fp_errs[0], fp_errs[0]
    assert code_fingerprint() in fp_errs[0], (
        "the error must name the LIVE fingerprint, or the reader cannot tell "
        "which side moved")
    # the message has to say what to do about it
    assert "Re-run" in fp_errs[0]


def test_the_old_agreement_check_alone_would_have_let_this_through(tmp_path):
    """Pins WHY the fix was needed, not just that it works.

    If this ever fails it means the fixture accidentally writes more than one
    fingerprint, and the rejection above would prove nothing: the pre-fix
    "cells were produced by different code versions" error would have caught it
    on its own.
    """
    results = _write_tier(tmp_path / "results", STALE_FP)

    recorded = set()
    for cell in T1_CELLS:
        for s in (0, 1):
            m = json.loads((results / cell / f"seed{s}" / "eval_metrics.json")
                           .read_text(encoding="utf-8"))
            recorded.add(m["provenance"]["code_fingerprint"])
    assert recorded == {STALE_FP}, (
        "this tree must be unanimous for the test above to mean anything")

    errors, _ = check_tier.check("T1", results, [0, 1])
    assert not [e for e in errors if "different code versions" in e], (
        "the agreement check fired, so the rejection is not attributable to the "
        "live comparison")


def test_check_tier_accepts_a_tier_produced_by_this_very_tree(tmp_path):
    """The gate must not simply reject everything: the converse case passes."""
    results = _write_tier(tmp_path / "results", code_fingerprint())

    errors, _warns = check_tier.check("T1", results, [0, 1])

    assert errors == [], f"a sound tier was rejected: {errors}"


def test_check_tier_also_catches_a_run_SCORED_by_superseded_code(tmp_path):
    """Trained by this tree, scored by an older one -- equally untraceable.

    `provenance_from_checkpoint` records the evaluating tree's fingerprint under
    ``provenance["evaluated"]``, because re-scoring a checkpoint elsewhere must
    not overwrite the training run's. That second fingerprint is provenance too,
    and a number produced by today's weights through yesterday's scorer is not a
    result of today's tree.
    """
    results = _write_tier(tmp_path / "results", code_fingerprint(),
                          eval_fp=STALE_FP)

    errors, _warns = check_tier.check("T1", results, [0, 1])

    fp_errs = _fingerprint_errors(errors)
    assert fp_errs, f"a stale EVALUATION fingerprint was not caught: {errors}"
    assert "(scored by)" in fp_errs[0], fp_errs[0]


def test_check_tier_reports_stale_fingerprints_even_when_provenance_is_incomplete(tmp_path):
    """The reason a re-run is needed must not hide behind a different error.

    A run missing ``wall_time_s`` is already rejected. If the fingerprint were
    harvested only in the else-branch of that check, the operator would fix the
    missing field, re-run the gate, and only THEN learn the run has to be
    redone anyway.
    """
    results = _write_tier(tmp_path / "results", STALE_FP)
    run = results / T1_CELLS[0] / "seed0"
    m = json.loads((run / "eval_metrics.json").read_text(encoding="utf-8"))
    del m["provenance"]["wall_time_s"]
    (run / "eval_metrics.json").write_text(json.dumps(m), encoding="utf-8")

    errors, _ = check_tier.check("T1", results, [0, 1])

    assert any("provenance missing" in e for e in errors)
    assert _fingerprint_errors(errors), (
        "the stale fingerprint was suppressed by an unrelated provenance gap: "
        f"{errors}")


def test_check_tier_cli_exits_nonzero_on_a_stale_tier(tmp_path, capsys):
    """End to end through main(), which is how chain_v3.sh invokes it."""
    results = _write_tier(tmp_path / "results", STALE_FP)

    rc = check_tier.main(["--tier", "T1", "--seeds", "0", "1",
                          "--results", str(results)])
    out = capsys.readouterr().out

    assert rc == 1, "a stale tier exited zero -- the chain would have continued"
    assert "do not match the code in this tree" in out
    assert "NOT sound" in out


def test_check_tier_cli_exits_zero_on_a_tier_from_this_tree(tmp_path, capsys):
    results = _write_tier(tmp_path / "results", code_fingerprint())

    rc = check_tier.main(["--tier", "T1", "--seeds", "0", "1",
                          "--results", str(results)])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "IS SOUND" in out
    assert code_fingerprint() in out, (
        "the pass message should state which tree it checked against")


# --------------------------------------------------------------------- #
#  aggregate_seeds.py and the nested T5 schema                 #
# --------------------------------------------------------------------- #

def test_summarise_arm_reads_the_nested_unified_schema(tmp_path):
    """Before the fix this raised KeyError: 'overall_accuracy'."""
    runs = {s: _nested_metrics("mfinstseg_unified", s, 0.90 + 0.01 * s,
                               code_fingerprint())
            for s in (0, 1, 2)}

    out = aggregate_seeds.summarise_arm(runs)

    assert out["n_seeds"] == 3
    assert out["accuracy_mean"] == pytest.approx((0.90 + 0.91 + 0.92) / 3)
    assert out["macro_f1_mean"] == pytest.approx(0.70)
    assert out["mean_iou_mean"] == pytest.approx(0.55)
    # the silent half of the bug: the confusion matrix lives under ["semantic"]
    # too, and a membership test on the top level found nothing without raising
    assert "confusion_matrix_summed" in out, (
        "per-class IoU vanished for the nested layout without an error")
    assert out["confusion_matrix_summed"] == [[2700, 120], [180, 2400]]
    assert out["code_fingerprints"] == [code_fingerprint()]


def test_summarise_arm_still_reads_the_flat_schema(tmp_path):
    """The other three quarters of the run matrix must be unaffected."""
    runs = {s: _flat_metrics("cadsynth_bias_both", s, 0.90 + 0.01 * s,
                             code_fingerprint())
            for s in (0, 1, 2)}

    out = aggregate_seeds.summarise_arm(runs)

    assert out["accuracy_mean"] == pytest.approx((0.90 + 0.91 + 0.92) / 3)
    assert out["confusion_matrix_summed"] == [[2700, 120], [180, 2400]]


def test_summarise_arm_refuses_a_metrics_file_with_no_accuracy_in_either_layout():
    """Not a seed to skip: a quietly smaller sample is the failure mode here."""
    runs = {0: {"experiment_name": "x", "provenance": _provenance(0, code_fingerprint())}}

    with pytest.raises(SystemExit) as e:
        aggregate_seeds.summarise_arm(runs)

    msg = str(e.value)
    assert "overall_accuracy" in msg and "['semantic']" in msg, msg


def test_aggregate_seeds_end_to_end_on_the_nested_schema(tmp_path):
    """The T5 table step, which is what would have raised after 5 runs."""
    results = tmp_path / "results"
    for s in (0, 1, 2):
        run = results / "mfinstseg_unified" / f"seed{s}"
        run.mkdir(parents=True)
        (run / "unified_eval_metrics.json").write_text(
            json.dumps(_nested_metrics("mfinstseg_unified", s, 0.90 + 0.01 * s,
                                       code_fingerprint())),
            encoding="utf-8")
    out_md = tmp_path / "summary_mfinstseg.md"

    rc = aggregate_seeds.main(["--dataset", "mfinstseg",
                               "--arms", "mfinstseg_unified",
                               "--seeds", "0", "1", "2",
                               "--results", str(results),
                               "--out", str(out_md)])

    assert rc == 0, out_md.read_text(encoding="utf-8")
    text = out_md.read_text(encoding="utf-8")
    assert "| mfinstseg_unified | 3 | 91.00%" in text, text
    assert "All checks pass" in text


# --------------------------------------------------------------------- #
#  aggregate_seeds.py                                          #
# --------------------------------------------------------------------- #

def test_aggregate_seeds_marks_a_stale_table_not_quotable(tmp_path):
    """Loud, non-zero, and stamped into the artifact -- not a silent table."""
    results = _write_tier(tmp_path / "results", STALE_FP, seeds=(0, 1, 2),
                          cells=("cadsynth_bias_both",))
    out_md = tmp_path / "summary_cadsynth.md"

    rc = aggregate_seeds.main(["--dataset", "cadsynth",
                               "--arms", "cadsynth_bias_both",
                               "--seeds", "0", "1", "2",
                               "--results", str(results),
                               "--out", str(out_md)])

    assert rc == 1, "a stale table exited zero"
    text = out_md.read_text(encoding="utf-8")
    assert "do not match the code in this tree" in text, text
    assert STALE_FP in text and code_fingerprint() in text
    assert "NOT quotable" in text
    # the verdict must be visible before the numbers, not only under them
    banner = text.index("NOT QUOTABLE")
    table = text.index("| arm | seeds |")
    assert banner < table, "the NOT QUOTABLE banner sits below the table"
    # and it must survive into the machine-readable twin
    data = json.loads(out_md.with_suffix(".json").read_text(encoding="utf-8"))
    assert any("do not match the code in this tree" in p
               for p in data["problems"]), data["problems"]


def test_aggregate_seeds_passes_a_table_produced_by_this_tree(tmp_path):
    results = _write_tier(tmp_path / "results", code_fingerprint(),
                          seeds=(0, 1, 2), cells=("cadsynth_bias_both",))
    out_md = tmp_path / "summary_cadsynth.md"

    rc = aggregate_seeds.main(["--dataset", "cadsynth",
                               "--arms", "cadsynth_bias_both",
                               "--seeds", "0", "1", "2",
                               "--results", str(results),
                               "--out", str(out_md)])

    text = out_md.read_text(encoding="utf-8")
    assert rc == 0, text
    assert "All checks pass" in text
    assert "NOT QUOTABLE" not in text


# --------------------------------------------------------------------- #
#  the two tools must not drift apart again                             #
# --------------------------------------------------------------------- #

def test_aggregate_seeds_uses_check_tiers_accessor_rather_than_its_own_copy():
    """One accessor, imported, so a third layout cannot be half-supported."""
    assert aggregate_seeds.semantic_block is check_tier.semantic_block
    assert aggregate_seeds.headline_metrics is check_tier.headline_metrics


def test_the_dead_private_alias_is_gone():
    """[21 Sep 2026, defect 5] `_headline = headline_metrics` had no consumers.

    It was introduced as a compatibility shim for "anything that already
    imported it". Nothing did: a grep over the whole codebase found the
    assignment itself and one assertion in this file, which is a shim kept alive
    by its own test. A second public name for the same function is a second
    thing a future reader has to decide between, and the next accessor bug will
    be fixed in one of them.
    """
    assert not hasattr(check_tier, "_headline"), (
        "the dead alias is back; `headline_metrics` is the only name")


# --------------------------------------------------------------------- #
#  DEFECT 1 -- a missing fingerprint was skipped, and the gate then lied #
# --------------------------------------------------------------------- #

def test_a_run_with_only_a_git_commit_is_NOW_caught_upstream_TOO():
    """[TIGHTENED 22 Sep 2026, H3] The precondition for defect 1 is gone.

    The earlier test pinned `check_regiment_fields` treating code provenance as satisfied by
    EITHER a commit or a fingerprint -- a forward-compatibility choice for the
    day the repo exists, and the reason `check_tier` had to grow its own
    missing-fingerprint error rather than leave it to the upstream check.

    That OR is now gone: `src/utils/provenance.check_regiment_fields` requires
    `code_fingerprint`, because there is no ``.git`` under ``codebase/``, so the
    commit branch had no reachable use, and on the day there IS a repo
    `build_provenance` still records a fingerprint unconditionally. The
    reasoning is at the branch itself.

    **The tests below do not stop testing the hole they were written for.**
    `check_tier` is the gate that reads a tier off disk; it must report the
    absence itself, with the live fingerprint named and an instruction to
    re-run, rather than rely on any upstream call -- which is why they assert on
    the message, not merely on an exit code. Defence in depth is the point: two
    independent checks, and this one is now the second, not the only.
    """
    from src.utils.provenance import check_regiment_fields

    prov = _provenance(0, None, git_commit="abc1234")

    assert "code_fingerprint" not in prov
    assert check_regiment_fields(prov) == ["code_fingerprint"], (
        "provenance.py accepts a commit-only run again, so a run with no code "
        "provenance at all is being called quotable by the shared check")


def test_check_tier_fails_a_tier_whose_runs_record_no_fingerprint(tmp_path):
    """The verifier's probe: 4 cells x 3 seeds, git_commit only, EXIT 0 before.

    Every run is otherwise perfect -- complete provenance, distinct accuracies,
    one test-set size, one epoch budget. The old gate added nothing to
    `fp_by_run`, compared nothing, and printed "every run's recorded
    fingerprint matches this tree".
    """
    results = _write_tier(tmp_path / "results", None, seeds=(0, 1, 2),
                          git_commit="abc1234")

    errors, _warns = check_tier.check("T1", results, [0, 1, 2])

    missing_errs = [e for e in errors if "records NO code_fingerprint" in e]
    assert len(missing_errs) == 12, (
        f"expected one error for each of the 12 runs, got {len(missing_errs)}: "
        f"{errors}")
    # it must say what to do, and it must not pretend a commit will serve
    assert "run_all.py" in missing_errs[0], missing_errs[0]
    assert "git_commit does not" in missing_errs[0], missing_errs[0]
    # and the reason must not be mistaken for the other, already-working check
    assert not _fingerprint_errors(errors), (
        "absence was reported as a MISMATCH; the two are different problems "
        "with different fixes")


def test_check_tier_cli_exits_nonzero_on_a_tier_with_no_fingerprints(tmp_path, capsys):
    """The probe end to end: chain_v3.sh reads this exit code and nothing else."""
    results = _write_tier(tmp_path / "results", None, seeds=(0, 1, 2),
                          git_commit="abc1234")

    rc = check_tier.main(["--tier", "T1", "--seeds", "0", "1", "2",
                          "--results", str(results)])
    out = capsys.readouterr().out

    assert rc == 1, (
        "a tier that compared ZERO fingerprints exited 0 -- the 85-run batch "
        "would have chained straight past it")
    assert "IS SOUND" not in out
    assert "records NO code_fingerprint" in out


def test_check_tier_fails_a_run_with_no_scoring_fingerprint(tmp_path):
    """Trained by this tree, scored by something that left no trace."""
    results = _write_tier(tmp_path / "results", code_fingerprint(),
                          eval_fp=None)

    errors, _warns = check_tier.check("T1", results, [0, 1])

    scoring = [e for e in errors if "['evaluated']['code_fingerprint']" in e]
    assert len(scoring) == 8, f"{len(scoring)} of 8 runs flagged: {errors}"
    assert "Re-score" in scoring[0], scoring[0]


# --------------------------------------------------------------------- #
#  DEFECTS 2 and 3 -- aggregate_seeds could not see either hole          #
# --------------------------------------------------------------------- #

def _aggregate(results, out_md, arm="cadsynth_bias_both",
               seeds=("0", "1", "2")):
    rc = aggregate_seeds.main(["--dataset", "cadsynth", "--arms", arm,
                               "--seeds", *seeds, "--results", str(results),
                               "--out", str(out_md)])
    return rc, out_md.read_text(encoding="utf-8")


def test_aggregate_seeds_catches_a_table_SCORED_by_superseded_code(tmp_path):
    """Defect 2. check_tier failed this tier; the table said All checks pass.

    `summarise_arm` collected only ``provenance["code_fingerprint"]``, so the
    staleness check in `main` never saw
    ``provenance["evaluated"]["code_fingerprint"]`` -- the fingerprint of the
    code that actually produced the numbers in the table.
    """
    results = _write_tier(tmp_path / "results", code_fingerprint(),
                          seeds=(0, 1, 2), cells=("cadsynth_bias_both",),
                          eval_fp=STALE_FP)
    # the two gates must not disagree about the same tree
    errors, _ = check_tier.check("T1", results, [0, 1, 2])
    assert _fingerprint_errors(errors), "check_tier no longer catches this"

    rc, text = _aggregate(results, tmp_path / "summary_cadsynth.md")

    assert rc == 1, "a table scored by superseded code exited 0"
    assert "All checks pass" not in text
    assert "do not match the code in this tree" in text, text
    assert STALE_FP in text, "the stale SCORING fingerprint is not named"
    data = json.loads((tmp_path / "summary_cadsynth.json")
                      .read_text(encoding="utf-8"))
    assert any(STALE_FP in p for p in data["problems"]), data["problems"]


def test_summarise_arm_exposes_the_scoring_fingerprint(tmp_path):
    """The unit under defect 2: the field main() reads has to exist."""
    runs = {s: _flat_metrics("cadsynth_bias_both", s, 0.90 + 0.01 * s,
                             code_fingerprint(), eval_fp=STALE_FP)
            for s in (0, 1, 2)}

    out = aggregate_seeds.summarise_arm(runs)

    assert out["code_fingerprints"] == [code_fingerprint()]
    assert out["eval_code_fingerprints"] == [STALE_FP]


def test_aggregate_seeds_catches_a_table_with_no_fingerprints_at_all(tmp_path):
    """Defect 3. ``any(c != live for c in [])`` is False, so absence passed.

    Every run carries a git_commit and nothing else, so `check_regiment_fields`
    is satisfied and the old table printed "one code version across the table"
    having compared none.
    """
    results = _write_tier(tmp_path / "results", None, seeds=(0, 1, 2),
                          cells=("cadsynth_bias_both",), git_commit="abc1234")

    rc, text = _aggregate(results, tmp_path / "summary_cadsynth.md")

    assert rc == 1, "a table with zero recorded fingerprints exited 0"
    assert "All checks pass" not in text
    assert "record NO code_fingerprint" in text, text
    assert "[0, 1, 2]" in text, "the failing seeds are not named"
    assert "NOT QUOTABLE" in text


def test_aggregate_seeds_catches_a_missing_scoring_fingerprint(tmp_path):
    """The other half of defect 3: an absent ``evaluated`` block."""
    results = _write_tier(tmp_path / "results", code_fingerprint(),
                          seeds=(0, 1, 2), cells=("cadsynth_bias_both",),
                          eval_fp=None)

    rc, text = _aggregate(results, tmp_path / "summary_cadsynth.md")

    assert rc == 1
    assert "no provenance['evaluated']['code_fingerprint']" in text, text


def test_a_sound_table_still_passes_and_says_what_it_verified(tmp_path):
    """The converse: the new checks must not reject a healthy table."""
    results = _write_tier(tmp_path / "results", code_fingerprint(),
                          seeds=(0, 1, 2), cells=("cadsynth_bias_both",))

    rc, text = _aggregate(results, tmp_path / "summary_cadsynth.md")

    assert rc == 0, text
    assert "All checks pass" in text
    assert "training AND scoring fingerprint" in text
    assert code_fingerprint() in text


# --------------------------------------------------------------------- #
#  DEFECT 6 -- link metrics are a THIRD layout, not a semantic one       #
# --------------------------------------------------------------------- #

def _link_metrics(seed, acc, fp, *, reportable="argmax", auc=0.93):
    """The shape `evaluate_link()` actually writes.

    Copied from the ``metrics = {...}`` literal in
    `src/evaluation/evaluate_link.py` (lines 439-466 on 21 Sep 2026), including
    ``reportable`` and the oracle row's "do not report" note.
    """
    return {
        "experiment_name": "mfinstseg_two_stage",
        "link_checkpoint": "checkpoints/link.pth",
        "semantic_checkpoint": "checkpoints/sem.pth",
        "n_pairs": 10000, "n_positive": 2000, "n_negative": 8000,
        "argmax": {"accuracy": acc, "precision": 0.80, "recall": 0.79,
                   "f1": 0.795, "auc": auc,
                   "confusion_matrix": [[7600, 400], [800, 1200]]},
        "oracle_threshold": {"value": 0.42, "accuracy": 0.95, "precision": 0.9,
                             "recall": 0.9, "f1": 0.9,
                             "confusion_matrix": [[7800, 200], [300, 1700]],
                             "note": "tuned on the test labels -- upper bound "
                                     "only, do not report"},
        "train_threshold": {"value": 0.51, "accuracy": acc - 0.01,
                            "precision": 0.78, "recall": 0.77, "f1": 0.775},
        "reportable": reportable,
        "provenance": _provenance(seed, fp,
                                  arm="mfinstseg_two_stage"),
    }


def test_the_gate_and_the_runner_agree_on_what_a_metrics_file_is_called(tmp_path):
    """`run_all.metrics_path` listed four names; `_metrics_file` listed three.

    A link run therefore wrote its result, satisfied run_all's own quotability
    gate, and was reported by check_tier as "no metrics file". Driving both
    functions over the same directories keeps one list in two places.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_all_for_gatetest", CODEBASE / "runners" / "run_all.py")
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)

    for n in ("eval_metrics.json", "unified_eval_metrics.json",
              "link_eval_metrics.json", "metrics.json"):
        d = tmp_path / n.replace(".json", "")
        d.mkdir()
        (d / n).write_text("{}", encoding="utf-8")
        assert check_tier._metrics_file(d) == d / n, (
            f"check_tier does not recognise {n}")
        assert run_all.metrics_path(d) == d / n, (
            f"run_all does not recognise {n}")


def test_a_link_metrics_file_is_recognised_rather_than_read_as_semantic():
    """The obvious fix was not sufficient; this is why.

    Before: `_metrics_file` returned None, so the file was invisible. After
    only adding the filename: `semantic_block` fell through to its FLAT branch
    and `headline_metrics` returned ``(None, None, None, None)`` -- accuracy,
    macro-F1, mIoU and n all absent for a file whose numbers are all present.
    """
    m = _link_metrics(0, 0.88, code_fingerprint())

    assert check_tier.is_link_metrics(m)
    # the semantic accessor must refuse it loudly, not return four Nones
    with pytest.raises(ValueError) as e:
        check_tier.headline_metrics(m)
    assert "link_eval_metrics.json" in str(e.value), str(e.value)
    with pytest.raises(ValueError):
        check_tier.semantic_block(m)
    # a genuinely semantic file must be unaffected
    assert not check_tier.is_link_metrics(
        _flat_metrics("cadsynth_bias_both", 0, 0.9, code_fingerprint()))
    assert not check_tier.is_link_metrics(
        _nested_metrics("mfinstseg_unified", 0, 0.9, code_fingerprint()))


def test_link_headline_reads_the_reportable_row_and_never_the_oracle():
    """`reportable` is the evaluator's own statement of which row may be quoted.

    The oracle row is fitted on the test labels. It is the highest number in
    the file, so a gate that picked a row by guessing would pick it.
    """
    m = _link_metrics(0, 0.88, code_fingerprint(), reportable="train_threshold")
    acc, f1, auc, n = check_tier.link_headline(m)
    assert acc == pytest.approx(0.87), "read the argmax row, not train_threshold"
    assert (f1, auc, n) == (pytest.approx(0.775), pytest.approx(0.93), 10000)

    m2 = _link_metrics(0, 0.88, code_fingerprint(), reportable="argmax")
    assert check_tier.link_headline(m2)[0] == pytest.approx(0.88)

    m3 = _link_metrics(0, 0.88, code_fingerprint(),
                       reportable="oracle_threshold")
    with pytest.raises(ValueError) as e:
        check_tier.link_headline(m3)
    assert "do not report" in str(e.value)

    m4 = _link_metrics(0, 0.88, code_fingerprint())
    del m4["reportable"]
    with pytest.raises(ValueError) as e:
        check_tier.link_headline(m4)
    assert "no 'reportable' key" in str(e.value)

    m5 = _link_metrics(0, 0.88, code_fingerprint(), reportable="train_threshold")
    del m5["train_threshold"]
    with pytest.raises(ValueError) as e:
        check_tier.link_headline(m5)
    assert "carries no such block" in str(e.value)


class _FakeCell:
    """[22 Sep 2026] `trainer` added: `runners/run_all.Cell` has always had one,
    and `check_tier` now asks it whether the cell's trainer reads
    `loss.use_class_weights` at all before deciding that a config which states
    no policy is a fault. `train_link.py` does not read it, which is why a link
    tier states no policy and is right not to."""

    def __init__(self, name, trainer="src.training.train_link"):
        self.name = name
        self.trainer = trainer


class _FakeRunAll:
    """A one-cell tier table whose cell is a link run.

    No entry in `runners/run_all.TIERS` uses `train_link` today, so defect 6 is
    LATENT: nothing in the 85-run batch reaches it. This substitutes a tier
    table that does, so the link path is exercised end to end rather than only
    at the accessor level.
    """
    TIERS = {"TLINK": [_FakeCell("mfinstseg_two_stage")]}
    WEIGHTED_TIERS = frozenset()


def _write_link_tier(root, fp, *, seeds=(0, 1, 2), auc=0.93, accs=None):
    for i, s in enumerate(seeds):
        run = root / "mfinstseg_two_stage" / f"seed{s}"
        run.mkdir(parents=True, exist_ok=True)
        acc = accs[i] if accs else 0.88 + 0.002 * s
        (run / "link_eval_metrics.json").write_text(
            json.dumps(_link_metrics(s, acc, fp, auc=auc)), encoding="utf-8")
    return root


def test_check_tier_passes_a_healthy_link_tier(tmp_path, monkeypatch):
    """Latent today, but it must be right on the day a link cell is added."""
    monkeypatch.setattr(check_tier, "_load_run_all", lambda: _FakeRunAll)
    results = _write_link_tier(tmp_path / "results", code_fingerprint())

    errors, warns = check_tier.check("TLINK", results, [0, 1, 2])

    assert errors == [], errors
    # mIoU is UNDEFINED for pair classification, not merely missing: the gate
    # must not nag about it on every link run for the rest of the project
    assert not any("mean_iou" in w for w in warns), warns


def test_check_tier_fails_a_link_tier_with_a_broken_auc(tmp_path, monkeypatch):
    """The link-specific range check earns its place."""
    monkeypatch.setattr(check_tier, "_load_run_all", lambda: _FakeRunAll)
    results = _write_link_tier(tmp_path / "results", code_fingerprint(),
                               auc=1.7)

    errors, _ = check_tier.check("TLINK", results, [0, 1, 2])

    assert any("AUC 1.7 outside (0, 1]" in e for e in errors), errors


def test_check_tier_still_applies_every_other_rule_to_a_link_tier(tmp_path, monkeypatch):
    """Recognising the layout must not turn the other checks off."""
    monkeypatch.setattr(check_tier, "_load_run_all", lambda: _FakeRunAll)
    # stale fingerprint, and three seeds with identical accuracy
    results = _write_link_tier(tmp_path / "results", STALE_FP,
                               accs=[0.88, 0.88, 0.88])

    errors, _ = check_tier.check("TLINK", results, [0, 1, 2])

    assert _fingerprint_errors(errors), errors
    assert any("same accuracy" in e for e in errors), errors


def test_aggregate_seeds_reports_a_link_run_instead_of_calling_it_missing(tmp_path):
    """It has no accuracy/macro-F1/mIoU column, so it is not a row in this table.

    `load_runs` looked only for the two semantic filenames, so a directory of
    finished link runs was reported as "missing seeds [0, 1, 2]" -- sending the
    operator to look for a crash that never happened.

    [CHANGED 22 Sep 2026] Raising contradicted
    the script's own reasoning under the fingerprint check in `main`, and cost
    the artifact: see
    `test_a_link_arm_does_not_destroy_the_table_of_the_arms_beside_it`. The
    verdict is now written to disk -- over the previous summary, which is the
    file people read -- and the exit status is non-zero. The sentences the old
    SystemExit carried are asserted here in their new home.
    """
    results = _write_link_tier(tmp_path / "results", code_fingerprint())
    out_md = tmp_path / "s.md"

    rc = aggregate_seeds.main(["--dataset", "mfinstseg",
                               "--arms", "mfinstseg_two_stage",
                               "--seeds", "0", "1", "2",
                               "--results", str(results),
                               "--out", str(out_md)])

    # rc 2, not 1: no arm produced a semantic run, so there is no table at all.
    assert rc == 2
    assert out_md.is_file(), "the verdict was not written"
    text = out_md.read_text(encoding="utf-8")
    assert "link_eval_metrics.json" in text and "SEMANTIC table" in text, text
    assert "check_tier.py" in text, "the message must point at the tool that can"
    assert "NOT QUOTABLE" in text
    assert "All checks pass" not in text
    data = json.loads(out_md.with_suffix(".json").read_text(encoding="utf-8"))
    assert any("link_eval_metrics.json" in p for p in data["problems"]), \
        data["problems"]


# --------------------------------------------------------------------- #
#  22 SEP DEFECT 1 -- one link arm destroyed the table of three good ones #
# --------------------------------------------------------------------- #
#
# `load_runs` raised SystemExit the moment it met a link metrics file, which
# aborted `main` before anything was written. A re-verifier probed a mixed
# results root -- three healthy semantic arms plus one link arm -- and got
# exit 1 with NO summary file for the three good arms, while the previous
# summary_<dataset>.md sat on disk still saying "All checks pass".
#
# That also contradicted this script's own reasoning under the fingerprint
# check in `main` ("WHY THIS REPORTS RATHER THAN RAISES"), which argues that
# raising is WEAKER here precisely because the stale artifact survives. The two
# could not both stand; the reasoning won.

_LINK_ARM = "cadsynth_link"


def _write_link_arm(results, fp, seeds=(0, 1, 2), arm=_LINK_ARM):
    for s in seeds:
        run = results / arm / f"seed{s}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "link_eval_metrics.json").write_text(
            json.dumps(_link_metrics(s, 0.88 + 0.002 * s, fp)), encoding="utf-8")
    return results


def test_a_link_arm_does_not_destroy_the_table_of_the_arms_beside_it(tmp_path):
    """The re-verifier's probe, as a test: 3 healthy semantic arms + 1 link arm."""
    good = ("cadsynth_bias_similarity", "cadsynth_bias_adjacency",
            "cadsynth_bias_both")
    results = _write_tier(tmp_path / "results", code_fingerprint(),
                          seeds=(0, 1, 2), cells=good)
    _write_link_arm(results, code_fingerprint())
    out_md = tmp_path / "summary_cadsynth.md"

    rc = aggregate_seeds.main(["--dataset", "cadsynth",
                               "--arms", *good, _LINK_ARM,
                               "--seeds", "0", "1", "2",
                               "--results", str(results),
                               "--out", str(out_md)])

    assert rc == 1, "a table with an unusable arm in it must exit non-zero"
    text = out_md.read_text(encoding="utf-8")
    # the three good arms still have their rows -- this is what raising cost
    for arm in good:
        assert f"| {arm} | 3 |" in text, f"{arm} lost its row\n{text}"
    # and the link arm is named as a problem rather than silently dropped
    assert "NOT QUOTABLE" in text
    assert "All checks pass" not in text
    assert _LINK_ARM in text and "link_eval_metrics.json" in text
    data = json.loads(out_md.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(data["arms"]) == set(good), (
        "the link arm must not appear as a row in the semantic table")
    assert any(_LINK_ARM in p and "link_eval_metrics.json" in p
               for p in data["problems"]), data["problems"]


def test_load_runs_still_aborts_for_a_caller_that_has_nothing_to_write(tmp_path):
    """`compare_arms.py` and `make_figures.py` call this with four arguments.

    They have no partial artifact to protect, and for them the alternative --
    counting a finished link run as a MISSING seed -- is the misleading message
    the branch was written to prevent. So the abort is still the default, and
    only a caller that passes a collector (and therefore undertakes to report
    them itself) gets the seeds back.
    """
    results = _write_link_arm(tmp_path / "results", code_fingerprint())

    with pytest.raises(SystemExit) as e:
        aggregate_seeds.load_runs(results, "cadsynth", _LINK_ARM, [0, 1, 2])
    assert "link_eval_metrics.json" in str(e.value)

    collected = []
    found, missing = aggregate_seeds.load_runs(
        results, "cadsynth", _LINK_ARM, [0, 1, 2], collected)
    assert (found, missing, collected) == ({}, [], [0, 1, 2]), (
        "a link seed is neither a loaded run nor a missing one")


def test_an_empty_results_root_still_leaves_a_verdict_on_disk(tmp_path):
    """The previous summary_<dataset>.md stays on disk with its "All checks pass"
    footer otherwise, and it is the file people read.
    """
    out_md = tmp_path / "summary_cadsynth.md"

    rc = aggregate_seeds.main(["--dataset", "cadsynth",
                               "--arms", "cadsynth_bias_both",
                               "--seeds", "0", "1", "2",
                               "--results", str(tmp_path / "nothing_here"),
                               "--out", str(out_md)])

    assert rc == 2, "rc 2 still distinguishes 'no table at all' from 'bad table'"
    assert out_md.is_file() and out_md.with_suffix(".json").is_file()
    text = out_md.read_text(encoding="utf-8")
    assert "NOT QUOTABLE" in text and "All checks pass" not in text
    assert "no rows at all" in text, text


# --------------------------------------------------------------------- #
#  22 SEP DEFECT 2 -- a config that states no weighting policy           #
# --------------------------------------------------------------------- #
#
# check_tier.py read `.get("loss", {}).get("use_class_weights")` and then threw
# the None away (`[w for w in per_tier_weight if w is not None]`). A config that
# forgot to state its policy was therefore neither compared nor mentioned: three
# cells saying nothing beside one saying `false` read as unanimity, and the tier
# was declared SOUND. Decision 2 allows one STATED policy per table plus the
# labelled weighted tiers; a cell that states nothing agrees with nothing.

_RESOLVED_HEAD = ("# Written by runners/run_all.py -- the exact config of one run.\n"
                  "model:\n  name: brepsegnet\n"
                  "training:\n  max_epochs: 60\n  patience: 10\n"
                  "loss:\n  type: cross_entropy\n")


def _write_policy(results, cell, policy=_ABSENT, seed=0):
    """Write a cell's config.resolved.yaml; `policy=_ABSENT` omits the key."""
    text = _RESOLVED_HEAD
    if policy is not _ABSENT:
        text += "  use_class_weights: {}\n".format("true" if policy else "false")
    (results / cell / f"seed{seed}" / "config.resolved.yaml").write_text(
        text, encoding="utf-8")


def _policy_errors(errors):
    return [e for e in errors if "use_class_weights" in e or "weighting" in e]


def test_check_tier_fails_a_cell_whose_config_states_no_weighting_policy(tmp_path):
    """The silent pass. One cell says `false`, three say nothing at all."""
    results = _write_tier(tmp_path / "results", code_fingerprint(), seeds=(0, 1))
    _write_policy(results, T1_CELLS[0], policy=False)
    for cell in T1_CELLS[1:]:
        _write_policy(results, cell)              # no use_class_weights line

    errors, _warns = check_tier.check("T1", results, [0, 1])

    named = _policy_errors(errors)
    assert len(named) == 3, f"expected one failure per silent cell:\n{errors}"
    for cell in T1_CELLS[1:]:
        assert any(e.startswith(f"{cell}: ") for e in named), (cell, named)
    # the message has to say what to do, not only that something is wrong
    assert any("use_class_weights: false" in e for e in named), named
    assert any("decision 2" in e for e in named), named


def test_check_tier_accepts_a_tier_whose_cells_all_state_the_same_policy(tmp_path):
    """The converse: stating the policy must not itself become a failure."""
    results = _write_tier(tmp_path / "results", code_fingerprint(), seeds=(0, 1))
    for cell in T1_CELLS:
        _write_policy(results, cell, policy=False)

    errors, _warns = check_tier.check("T1", results, [0, 1])

    assert errors == [], errors


def test_check_tier_still_catches_two_DIFFERENT_stated_policies(tmp_path):
    """The check that already worked must keep working after the fix."""
    results = _write_tier(tmp_path / "results", code_fingerprint(), seeds=(0, 1))
    for cell in T1_CELLS:
        _write_policy(results, cell, policy=False)
    _write_policy(results, T1_CELLS[1], policy=True)

    errors, _warns = check_tier.check("T1", results, [0, 1])

    assert any("different class-weighting policies" in e for e in errors), errors
    assert any("[False, True]" in e for e in errors), errors


def test_a_config_with_no_loss_block_at_all_is_a_failure_too(tmp_path):
    """`.get("loss", {})` made this case indistinguishable from the one above."""
    results = _write_tier(tmp_path / "results", code_fingerprint(), seeds=(0, 1))
    for cell in T1_CELLS:
        _write_policy(results, cell, policy=False)
    (results / T1_CELLS[2] / "seed0" / "config.resolved.yaml").write_text(
        "model:\n  name: brepsegnet\n", encoding="utf-8")

    errors, _warns = check_tier.check("T1", results, [0, 1])

    assert any(e.startswith(f"{T1_CELLS[2]}: ") and "no `loss:` block" in e
               for e in errors), errors


def test_a_missing_resolved_config_is_reported_as_NOT_PERFORMED(tmp_path):
    """Absent is not the same as "agrees", and it is not the same as corrupt.

    `runners/check_convergence.py` reports the same condition the same way
    (CFG_ABSENT -> NOT PERFORMED). What matters is that the gate says which
    check it could not run instead of counting the cell as agreeing.
    """
    results = _write_tier(tmp_path / "results", code_fingerprint(), seeds=(0, 1))

    errors, warns = check_tier.check("T1", results, [0, 1])

    assert errors == [], errors
    for cell in T1_CELLS:
        assert any(w.startswith(f"{cell}: no config.resolved.yaml")
                   and "NOT PERFORMED" in w for w in warns), (cell, warns)


def test_the_trainers_that_require_a_policy_are_read_from_their_own_source():
    """The predicate is derived, not a second list that can go stale.

    A trainer requires the key exactly when it calls
    `src.utils.class_weights.class_weighting_enabled` -- which raises rather
    than defaulting, so such a trainer cannot have started from a config that
    omits it. train_unified and train_link never mention class weighting (the
    unified `loss:` block weights the three TASKS, not the classes), and
    configs/mfinstseg_unified.yaml accordingly has no such key. T5 is in the
    imminent batch, so this is the difference between a gate that fails T5 for
    the wrong reason and one that does not.
    """
    req = check_tier._trainer_requires_weighting_policy
    assert req("src.training.train_semantic") is True
    assert req("src.training.train_resnet") is True
    assert req("src.training.train_unified") is False
    assert req("src.training.train_link") is False
    assert req("src.training.train_does_not_exist") is None
    assert req(None) is None

    unified = (CODEBASE / "configs" / "mfinstseg_unified.yaml").read_text(
        encoding="utf-8")
    assert "use_class_weights" not in unified, (
        "configs/mfinstseg_unified.yaml now states a weighting policy; "
        "check_tier's reason for excusing T5 needs revisiting")


class _FakeUnifiedRunAll:
    """A one-cell tier whose trainer has no class weighting at all."""

    TIERS = {"TUNI": [_FakeCell("mfinstseg_unified",
                                trainer="src.training.train_unified")]}
    WEIGHTED_TIERS = frozenset()


def test_a_cell_whose_trainer_has_no_weighting_is_reported_not_counted(
        tmp_path, monkeypatch):
    """T5's shape: no policy to state, so it is excluded OUT LOUD."""
    monkeypatch.setattr(check_tier, "_load_run_all", lambda: _FakeUnifiedRunAll)
    results = tmp_path / "results"
    for s in (0, 1, 2):
        run = results / "mfinstseg_unified" / f"seed{s}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "unified_eval_metrics.json").write_text(
            json.dumps(_nested_metrics("mfinstseg_unified", s, 0.90 + 0.002 * s,
                                       code_fingerprint())), encoding="utf-8")
        (run / "config.resolved.yaml").write_text(_RESOLVED_HEAD, encoding="utf-8")

    errors, warns = check_tier.check("TUNI", results, [0, 1, 2])

    assert errors == [], errors
    assert any("does not read loss.use_class_weights" in w for w in warns), warns


class _FakeUnknownTrainerRunAll:
    TIERS = {"TX": [_FakeCell("mystery_cell", trainer="src.training.train_ghost")]}
    WEIGHTED_TIERS = frozenset()


def test_a_cell_whose_trainer_cannot_be_found_is_a_failure(tmp_path, monkeypatch):
    """A check that cannot be applied is not a check that passed."""
    monkeypatch.setattr(check_tier, "_load_run_all",
                        lambda: _FakeUnknownTrainerRunAll)
    results = tmp_path / "results"
    run = results / "mystery_cell" / "seed0"
    run.mkdir(parents=True)
    (run / "eval_metrics.json").write_text(
        json.dumps(_flat_metrics("mystery_cell", 0, 0.9, code_fingerprint())),
        encoding="utf-8")

    errors, _warns = check_tier.check("TX", results, [0])

    assert any("train_ghost" in e and "cannot be applied" in e
               for e in errors), errors


# --------------------------------------------------------------------- #
#  22 SEP, minor -- one fingerprint per gate run, taken once             #
# --------------------------------------------------------------------- #

def test_the_live_fingerprint_is_hashed_once_per_gate_run(tmp_path, monkeypatch):
    """`code_fingerprint` is `@lru_cache`d, so the repeats were cheap rather than a
    re-hash of src/ + runners/ per bad run -- but a verdict should be built from
    one reading of the tree, not from eight.
    """
    calls = []
    real = check_tier.code_fingerprint

    def counting(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(check_tier, "code_fingerprint", counting)
    # four cells x two seeds, every one of them recording no fingerprint
    results = _write_tier(tmp_path / "results", None, seeds=(0, 1),
                          git_commit="abc1234")

    errors, _warns = check_tier.check("T1", results, [0, 1])

    assert len(errors) >= 8, "the fixture must exercise the per-run branch"
    assert len(calls) == 1, (
        f"code_fingerprint() was called {len(calls)} times in one check()")


# ---------------------------------------------------------------------------
# The partial-average guard in the tool that makes the paper's table.
#
# [22 Sep 2026] An integration verifier found this LIVE in
# runners/aggregate_seeds.py after the equivalent fix had landed in
# runners/compare_arms.py and was never carried across. With mean_iou missing
# from two of five seeds, the whole documented gate chain -- check_tier then
# aggregate_seeds, the two tools scripts/RUNBOOK-tiers.md names -- returned
# exit 0, printed "TIER T1 IS SOUND" and "All checks pass", and emitted a row
# whose seed column said 5 carrying a mean over 3.
#
# The verifier also proved no test covered it: adding the guard left the suite
# byte-identical at 527 passed. These two tests are that gap closed.
# ---------------------------------------------------------------------------


def _arm_runs(n_seeds=5, drop_from=(), metric="mean_iou"):
    """`{seed: metrics}` for one arm, optionally missing `metric` on some seeds."""
    runs = {}
    for seed in range(n_seeds):
        m = {
            "overall_accuracy": 0.92 + seed * 0.002,
            "macro_f1": 0.90 + seed * 0.002,
            "mean_iou": 0.85 + seed * 0.002,
            "n_samples": 34338,
            "provenance": {"seed": seed, "epoch": 30, "selected_epoch": 30},
        }
        if seed in drop_from:
            del m[metric]
        runs[seed] = m
    return runs


@pytest.mark.parametrize("metric", ["mean_iou", "macro_f1"])
def test_a_metric_missing_from_some_seeds_refuses_instead_of_averaging(metric):
    """The defect: a mean over 3 printed in a row whose seed count says 5.

    [REWRITTEN 26 Sep 2026, gate-parity wave] The refusal is unchanged in strength
    -- nothing is averaged, the mean stays None, `main` appends the problem,
    the banner says NOT QUOTABLE and the exit status is 1 -- but it is now
    REPORTED rather than raised, and the raise was the defect.

    aggregate_seeds.py's own "WHY THIS REPORTS RATHER THAN RAISES" paragraph
    calls itself "the rule for the whole script": raising here aborted before
    a single line of the artifact was written (measured 26 Sep: a 391-byte
    stderr message and no table), so the previous `summary_<dataset>.md`
    survived on disk still carrying its "All checks pass" footer -- and that
    is the file people read. The end-to-end version of this test, which checks
    that the stale artifact is written OVER, is
    `tests/test_gatefix_aggregate_coverage.py`
    `::test_a_metric_missing_from_some_seeds_is_reported_not_raised`.
    """
    runs = _arm_runs(drop_from=(3, 4), metric=metric)

    out = aggregate_seeds.summarise_arm(runs)

    # no partial average is offered to the caller under any key
    assert out[f"{metric}_mean"] is None
    assert out[f"{metric}_std"] is None
    assert metric in out["metrics_not_aggregated"]

    gap = out["metric_gaps"][metric]
    assert gap["have"] == [0, 1, 2], "must name which seeds DO carry it"
    assert gap["missing"] == [3, 4], "and which seeds do not"
    # the true n against the claimed n -- that arithmetic is the whole point,
    # and it is what `main` prints from these two lists.
    assert (len(gap["have"]), out["n_seeds"]) == (3, 5)


@pytest.mark.parametrize("metric", ["mean_iou", "macro_f1"])
def test_a_metric_absent_from_every_seed_is_withheld_not_averaged(metric):
    """Absent everywhere claims nothing; absent from SOME fabricates a mean.

    [CORRECTED 26 Sep 2026, gate-parity wave] This test was called
    `..._is_not_an_error` and its reason was "check_tier merely warns when
    mean_iou is absent, so making the all-absent case fatal here would reject
    tiers that tool accepts". That reason is no longer true of check_tier and
    was the wrong way round: measured 26 Sep on a scratch copy of the
    committed T1 tree with `macro_f1` deleted from all five seeds of one cell,
    `check_tier --tier T1` gave rc 1 with 5 FAILs ("... seed 0: macro_f1
    absent", one per seed) while `aggregate_seeds` gave rc 0 and "All checks
    pass" with the column printed as a dash.

    `summarise_arm`'s behaviour is unchanged and is what this test still
    pins: nothing is fabricated, both means stay None. The VERDICT is `main`'s
    and is now a problem -- see
    `tests/test_gatefix_aggregate_coverage.py::test_a_metric_missing_from_every_seed_is_a_problem`.
    """
    runs = _arm_runs(drop_from=(0, 1, 2, 3, 4), metric=metric)

    out = aggregate_seeds.summarise_arm(runs)

    assert out["n_seeds"] == 5
    assert out[f"{metric}_mean"] is None
    assert out[f"{metric}_std"] is None
    assert out["metric_gaps"][metric] == {"have": [], "missing": [0, 1, 2, 3, 4]}


# ---------------------------------------------------------------------------
# The selection-rule check -- the single most expensive fix in this project,
# and it shipped with no test at all.
#
# [22 Sep 2026] Two gates once rejected any tier whose runs stopped at
# different epochs, reading `provenance["epoch"]`:
#
#     if len(per_tier_epoch) > 1:
#         errors.append("cells were trained for different epoch budgets: "
#                       f"{sorted(per_tier_epoch)} -- not comparable")
#
# Under the fixed-35-epoch scheme that was right. The selection rule replaced the budget
# with early stopping on validation loss, so arms now stop at different epochs
# BY DESIGN, and the line failed EVERY sound tier -- 30-35 GPU-hours of batch
# run coming back un-aggregatable. It was replaced by a check on the SELECTION
# RULE: one `max_epochs` and one `patience` across a tier's cells.
#
# A mutation test proved the replacement was uncovered: neutralising both halves
# of the comparison left the suite at 531 passed, byte-identical. These tests
# are that gap closed. They must fail if either half is disabled.
# ---------------------------------------------------------------------------


def _write_rule(run: Path, *, max_epochs=60, patience=10) -> None:
    """The `config.resolved.yaml` the gate reads the patience half from."""
    (run / "config.resolved.yaml").write_text(
        f"training:\n  max_epochs: {max_epochs}\n  patience: {patience}\n"
        f"loss:\n  use_class_weights: false\n", encoding="utf-8")


def _tier_with_rules(root: Path, fp: str, per_cell_rule: dict) -> Path:
    """A sound tier whose seeds stop at DIFFERENT epochs, per the selection rule.

    `per_cell_rule` maps a cell name to its ``(max_epochs, patience)``. Cells
    absent from it get the tier's normal rule.
    """
    stop_at = {0: 19, 1: 31, 2: 44}
    for ci, cell in enumerate(T1_CELLS):
        cap, pat = per_cell_rule.get(cell, (60, 10))
        for s, ep in stop_at.items():
            run = root / cell / f"seed{s}"
            run.mkdir(parents=True, exist_ok=True)
            m = _flat_metrics(cell, s, 0.90 + 0.01 * ci + 0.002 * s, fp)
            # Each seed stops where its own validation curve ran out of
            # patience. `epoch == selected_epoch` because provenance is built
            # inside the save-on-improvement branch -- see `_provenance`.
            m["provenance"]["epoch"] = ep
            m["provenance"]["selected_epoch"] = ep
            m["provenance"]["epoch_budget_configured"] = cap
            (run / "eval_metrics.json").write_text(json.dumps(m),
                                                   encoding="utf-8")
            _write_rule(run, max_epochs=cap, patience=pat)
    return root


def test_seeds_stopping_at_different_epochs_is_sound(tmp_path):
    """The normal case.

    Every seed early-stops on its own validation curve, so the tier spans a
    range of epochs. The old gate called that "different epoch budgets -- not
    comparable" and failed it.
    """
    root = _tier_with_rules(tmp_path / "results", code_fingerprint(), {})

    errors, warns = check_tier.check("T1", root, [0, 1, 2])

    assert errors == [], (
        "a sound tier was rejected. Arms stop at different epochs "
        "by design; comparability is one SELECTION RULE, not one epoch count")
    assert any("span" in w for w in warns), (
        "the epoch spread should still be REPORTED as a note -- it is "
        "information the reader wants, just not a verdict")


def test_a_cell_held_to_a_different_cap_is_rejected(tmp_path):
    """The other direction: a gate that accepts everything is no gate."""
    root = _tier_with_rules(tmp_path / "results", code_fingerprint(),
                            {"cadsynth_bias_both": (40, 10)})

    errors, _ = check_tier.check("T1", root, [0, 1, 2])

    assert any("DIFFERENT epoch caps" in e for e in errors), (
        f"one cell capped at 40 while the rest ran to 60 is the unequal-budget "
        f"confound the selection rule removed, and it must fail. Got: {errors}")


def test_a_cell_held_to_a_different_patience_is_rejected(tmp_path):
    """Patience is the other half of the rule, and the easier half to forget.

    Two arms capped at 60 but stopped on patience 5 and patience 20 did not
    receive the same training, even though both records say "60".
    """
    root = _tier_with_rules(tmp_path / "results", code_fingerprint(),
                            {"cadsynth_bias_both": (60, 20)})

    errors, _ = check_tier.check("T1", root, [0, 1, 2])

    assert any("DIFFERENT patience" in e for e in errors), (
        f"same cap, different patience is still two selection rules. "
        f"Got: {errors}")


# ---------------------------------------------------------------------------
# P1 item 10 -- the separation standard error used one arm's seed count for
# BOTH arms.
#
#     se = ((variance(xa) + variance(xb)) / len(xa)) ** 0.5
#
# The standard error of a difference of means is sqrt(var_a/n_a + var_b/n_b);
# the two terms have different denominators whenever the arms have different
# seed counts. A crashed or unscorable seed in one arm produces exactly that.
#
# Measured over 200,000 random unequal-arm comparisons (n_a=5, n_b in 2..4):
# the printed verdict changed in 5.5% of them, and the error is one-directional
# -- the old formula UNDERSTATES the standard error, inflating t, so it claims
# more than the data supports. In one worked case it understated the SE by 31%
# and inflated t by 1.45x.
#
# The verdict beside the number is computed FROM the number, so this decided
# what the table told a reader the data supported.
# ---------------------------------------------------------------------------


def _welch_se(xa, xb):
    """sqrt(var_a/n_a + var_b/n_b) -- what the code must now compute."""
    import statistics
    return (statistics.variance(xa) / len(xa)
            + statistics.variance(xb) / len(xb)) ** 0.5


def _buggy_se(xa, xb):
    """The superseded formula: both variances over arm a's count."""
    import statistics
    return ((statistics.variance(xa) + statistics.variance(xb)) / len(xa)) ** 0.5


def test_the_separation_se_uses_each_arms_own_seed_count(tmp_path):
    """Unequal arms must get the Welch SE, not arm a's count used twice.

    Asserted on the rendered VERDICT, which is the thing a reader acts on. The
    accuracies below were searched for specifically because the two formulas
    disagree on them:

        buggy   t = 1.009  ->  "suggestive only"
        correct t = 0.883  ->  "indistinguishable"

    An earlier version of this test asserted on the seed-count column instead
    and passed against the restored bug -- i.e. it proved nothing. A mutation
    check caught that. This one fails when the old formula is put back.
    """
    fp = code_fingerprint()
    root = tmp_path / "results"
    accs = {"cadsynth_bias_adjacency": [0.9066, 0.9024, 0.9119],   # n=3
            "cadsynth_bias_similarity": [0.9072, 0.9156]}          # n=2
    for cell, values in accs.items():
        for s, acc in enumerate(values):
            run = root / cell / f"seed{s}"
            run.mkdir(parents=True)
            _write_rule(run)
            (run / "eval_metrics.json").write_text(
                json.dumps(_flat_metrics(cell, s, acc, fp)), encoding="utf-8")

    rc = aggregate_seeds.main(
        ["--dataset", "cadsynth", "--results", str(root),
         "--arms", "adjacency", "similarity", "--seeds", "0", "1", "2"])
    assert rc in (0, 1), rc
    md = (root / "summary_cadsynth.md").read_text(encoding="utf-8")

    row = [l for l in md.splitlines() if "adjacency vs similarity" in l]
    assert row, f"no pairwise row was rendered:\n{md}"

    assert "indistinguishable" in row[0], (
        f"the buggy SE -- both variances over arm a's seed count -- gives "
        f"t=1.009 and prints 'suggestive only' here; the correct Welch SE "
        f"gives t=0.883 and prints 'indistinguishable'. Row: {row[0]}")

    assert "**3 vs 2**" in row[0], (
        f"a comparison between arms of different size must SAY so, or the "
        f"reader cannot tell the verdict rests on 2 seeds. Row: {row[0]}")


def test_the_seed_count_note_states_what_actually_ran(tmp_path):
    """The seed-count note states the counts that actually ran."""
    note = aggregate_seeds._seed_count_note(
        {"adjacency": [0.96] * 5, "similarity": [0.93] * 2})

    assert "DIFFER" in note and "5" in note and "2" in note, note
    assert "five seeds per arm" not in note.lower(), (
        "the hardcoded claim is back")

    equal = aggregate_seeds._seed_count_note({"a": [0.9] * 5, "b": [0.9] * 5})
    assert "5 seed(s) per arm" in equal, equal
    assert "provisional" not in equal, "5 seeds is the regiment's requirement"


# ---------------------------------------------------------------------------
# T4's evaluator crash: a WEIGHTED ResNet checkpoint would not load.
#
# [25 Sep 2026] All five runs of tier T4 trained successfully and then all five
# died at evaluation:
#
#     RuntimeError: size mismatch for _class_weights: copying a param with
#     shape torch.Size([25]) ... the shape in current model is torch.Size([0]).
#
# `ResNetClassificationModel` registers `_class_weights` as the weight vector
# when one is passed and `torch.empty(0)` when it is not. The trainer passes it
# whenever weighting is on; `evaluate_semantic._build_resnet` never did.
#
# The bug needs ResNet AND weighting TOGETHER. Every ResNet cell before T4 is
# unweighted and every weighted cell before T4 is semantic, so neither the runs
# nor the smoke pass had exercised the combination. This test is that hole
# closed: it is the one cell of the trainer x weighting grid nothing covered.
#
# Test-policy category 1 (selection/evaluation semantics): a checkpoint that
# trained must be loadable by the evaluator that scores it.
# ---------------------------------------------------------------------------


def _resnet_ckpt(tmp_path, *, weighted: bool, n_classes: int = 25):
    """A checkpoint shaped exactly as `train_resnet` saves one."""
    import torch
    from src.models.resnet_encoder import ResNetClassificationModel

    weights = torch.ones(n_classes) * 1.5 if weighted else None
    model = ResNetClassificationModel(n_classes=n_classes, embedding_dim=512,
                                      dropout=0.3, class_weights=weights)
    p = tmp_path / ("weighted.pth" if weighted else "plain.pth")
    torch.save({"model": model.state_dict(), "n_classes": n_classes}, p)
    return p


@pytest.mark.parametrize("weighted", [True, False])
def test_the_resnet_evaluator_loads_a_checkpoint_either_weighting_produced(
        tmp_path, weighted):
    """Both cells of the ResNet x weighting grid must load.

    `weighted=True` is tier T4 and the four T6 arms; it is the case that failed.
    `weighted=False` is T3 and must keep working — a fix that only moved the
    breakage would pass the first half of this test and fail the second.
    """
    import torch
    from src.evaluation.evaluate_semantic import _build_resnet

    ckpt_path = _resnet_ckpt(tmp_path, weighted=weighted)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = {"model": {"embedding_dim": 512, "dropout": 0.3},
           "dataset": {"n_classes": 25},
           "loss": {"type": "cross_entropy",
                    "use_class_weights": weighted}}

    model, label_encoder, n_classes = _build_resnet(cfg, ckpt, "cpu")

    assert n_classes == 25
    assert label_encoder is None, "the ResNet path has no label tower"

    saved = ckpt["model"]["_class_weights"]
    assert model._class_weights.shape == saved.shape, (
        f"the evaluator built a model whose _class_weights is "
        f"{tuple(model._class_weights.shape)} while the checkpoint carries "
        f"{tuple(saved.shape)} -- this is the T4 crash")
    if weighted:
        assert model._class_weights.numel() == 25
        assert torch.allclose(model._class_weights, saved), (
            "the buffer must be restored FROM the checkpoint, not recomputed")
    else:
        assert model._class_weights.numel() == 0


def test_the_evaluator_does_not_load_loosely(tmp_path):
    """A genuinely incompatible checkpoint must still raise.

    `strict=False` would have fixed T4 by silencing every mismatch, including
    the ones that mean a checkpoint and a model disagree about architecture.
    This pins that the fix did not take that route.
    """
    import torch
    from src.evaluation.evaluate_semantic import _build_resnet

    ckpt_path = _resnet_ckpt(tmp_path, weighted=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["model"].pop("class_head.weight")          # a real architecture gap

    cfg = {"model": {"embedding_dim": 512, "dropout": 0.3},
           "dataset": {"n_classes": 25},
           "loss": {"type": "cross_entropy", "use_class_weights": True}}

    with pytest.raises(RuntimeError, match="[Mm]issing key"):
        _build_resnet(cfg, ckpt, "cpu")

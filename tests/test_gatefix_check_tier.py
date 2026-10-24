"""The six `check_tier.py` gate defects of `weekly/GATE-DEFECTS.md` (26 Sep 2026).

Every test here was written BEFORE the fix and run against the unfixed gate, so
each one is known to fail on the code that shipped the batch. What each covers,
in the numbering of the fixer brief (and the G-numbers of GATE-DEFECTS.md):

1. G1/G2  the nested unified layout was half-unchecked. `headline_metrics`
          reaches T5's numbers through `semantic_block`, which returns
          ``m["semantic"]`` -- so ``instance`` and ``bottom`` were read by
          NOTHING. ``instance.auc = 2.5`` and ``bottom.f1 = -3.0`` in one seed
          gave ``TIER T5 IS SOUND``, and so did deleting both blocks outright.
          The oracle guard (`link_headline`'s refusal of
          ``reportable == "oracle_threshold"``) sat behind `is_link_metrics`,
          which is False for T5 -- the one tier in the batch that writes an
          oracle row.
2. G5     ``optional = ("mean_iou",)`` demoted an absent mIoU to a note on
          every segmentation tier, while `aggregate_seeds` refuses the same
          tree. Two gates disagreeing about the same fault.
3. G6     the pass banner claimed "seeds present and distinct"; nothing checked
          distinctness. A byte-copy of seed0 relabelled ``seed: 4`` passed, and
          the arm then reported a 5-seed error bar over 4 real runs.
4. G8     ``arm_config_hash`` was read by no gate at all.
5.        the class-weighting policy was read from ``sorted(found)[0]`` -- the
          LOWEST seed -- so flipping seeds 1-4 passed with output byte-identical
          to clean, and no gate ever compared a config against the
          ``use_class_weights`` the RUN recorded.
6. G4     the decision-2 check was guarded by ``if tier not in
          WEIGHTED_TIERS``, so for T4 and T6 -- whose entire claim is "the
          weighted mirror" -- nothing verified that anything was weighted.

Each test builds a COMPLETE, SOUND tier first and asserts `check` returns no
errors on it, then mutates one thing. That shape is deliberate: a test that only
asserts an error appears cannot tell a fix from a gate that fails everything,
and this project has already found one check that was uncovered precisely
because its test passed before and after.

Nothing here reads `codebase/results`. Every tree is built under ``tmp_path``.
No fixture comes from ``conftest.py`` -- those build B-Rep JSON, which has
nothing to do with gates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

CODEBASE = Path(__file__).resolve().parents[1]
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from runners import check_tier                            # noqa: E402
from src.utils.provenance import code_fingerprint         # noqa: E402

run_all = check_tier._load_run_all()

#: The fingerprint of the tree the tests are running from. Every fixture run
#: records it, so the live-vs-recorded check passes and any error a test sees
#: is the one it is about.
LIVE_FP = code_fingerprint()

#: Every seed must score the same held-out set, or the gate fails the tier for
#: a different reason than the one under test.
N_SAMPLES = 34338

#: Cell names read from the tier table rather than hardcoded, so a rename there
#: breaks these tests loudly instead of leaving them testing empty trees.
T1_CELLS = [c.name for c in run_all.TIERS["T1"]]
T4_CELL = run_all.TIERS["T4"][0].name
T5_CELL = run_all.TIERS["T5"][0].name
T6_CELLS = [c.name for c in run_all.TIERS["T6"]]


# --------------------------------------------------------------------- #
#  fixtures: a sound tier on disk                                       #
# --------------------------------------------------------------------- #

def _digest(*parts) -> str:
    """A short, DETERMINISTIC digest -- `hash()` is salted per process."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:13]


def _provenance(cell: str, seed: int, *, ucw, cap: int = 60,
                epoch: int = 42) -> dict:
    """A provenance block that passes `check_regiment_fields` completely.

    ``config_hash`` differs per seed and ``arm_config_hash`` does not, which is
    the invariant `src/utils/provenance.py` documents: run_all pins a different
    ``seed`` and ``paths`` block into each resolved config, and
    `arm_config_hash` drops exactly those two keys.
    """
    return {
        "seed": seed,
        "epoch": epoch,
        "selected_epoch": epoch,
        "stopping_reason": "early_stop",
        "use_class_weights": ucw,
        "epoch_budget_configured": cap,
        "config_hash": f"{_digest(cell, seed)}cfg",
        "arm_config_hash": f"{_digest(cell)}arm",
        "code_fingerprint": LIVE_FP,
        "git_commit": None,
        "gpu": "Tesla T4",
        "wall_time_s": 931.0,
        "torch": "2.8.0+cu128",
        "python": "3.12.11",
        "platform": "Linux-6.8.0-1063-aws-x86_64-with-glibc2.39",
        "recorded_at": "2026-09-26T00:00:00+00:00",
        "evaluated": {
            "config_hash": f"cfg{seed}",
            "code_fingerprint": LIVE_FP,
            "gpu": "Tesla T4",
            "at": "2026-09-26T00:01:00+00:00",
        },
    }


def _flat_metrics(cell: str, seed: int, acc: float, *, ucw) -> dict:
    """What `evaluate_semantic.py` writes: the semantic numbers at the top."""
    return {
        "experiment_name": cell,
        "checkpoint": f"results/{cell}/seed{seed}/checkpoint.pth",
        "n_classes": 2,
        "n_samples": N_SAMPLES,
        "overall_accuracy": acc,
        "macro_precision": 0.71,
        "macro_recall": 0.69,
        "macro_f1": 0.70,
        "mean_iou": 0.55,
        "iou_per_class": {"0": 0.6, "1": 0.5},
        "confusion_matrix": [[900, 40], [60, 800 + seed]],
        "provenance": _provenance(cell, seed, ucw=ucw),
    }


def _nested_metrics(cell: str, seed: int, acc: float) -> dict:
    """What `evaluate_unified.py` writes for T5: three heads, semantic nested.

    Key names and nesting copied from the ``metrics = {...}`` literal at
    `src/evaluation/evaluate_unified.py:463-512`, including ``n_faces`` where
    the flat layout says ``n_samples`` and the ``checkpoint_threshold`` /
    ``oracle_threshold`` rows under ``instance``.
    """
    return {
        "experiment_name": cell,
        "checkpoint": f"results/{cell}/seed{seed}/checkpoint.pth",
        "epoch": 42,
        "split": "test (frozen ids)",
        "n_models": 1176,
        "attributes": "standardized",
        "bias_type": "both",
        "semantic": {
            "n_faces": N_SAMPLES,
            "n_classes": 2,
            "overall_accuracy": acc,
            "macro_precision": 0.71,
            "macro_recall": 0.69,
            "macro_f1": 0.70,
            "mean_iou": 0.55,
            "iou_per_class": {"0": 0.6, "1": 0.5},
            "per_class": {
                "0": {"accuracy": 0.8, "precision": 0.9, "recall": 0.8,
                      "f1": 0.85, "support": 456},
            },
            "confusion_matrix": [[900, 40], [60, 800 + seed]],
        },
        "bottom": {
            "n_faces": N_SAMPLES,
            "n_positive": 5101,
            "accuracy": 0.98,
            "precision": 0.95,
            "recall": 0.92,
            "f1": 0.93,
            "mean_iou": 0.93,
            "iou_not_bottom": 0.97,
            "iou_bottom": 0.88,
            "confusion_matrix": [[26453, 241], [373, 4728]],
        },
        "instance": {
            "n_pairs": 930718,
            "n_positive": 79682,
            "auc": 0.997,
            "reportable": "checkpoint_threshold",
            "checkpoint_threshold_source": "fitted on the validation partition",
            "note": "oracle_threshold was tuned on the test labels",
            "checkpoint_threshold": {
                "precision": 0.936, "recall": 0.942, "f1": 0.939,
                "accuracy": 0.989, "threshold": 0.877,
                "confusion_matrix": [[845904, 5132], [4558, 75124]],
            },
            "oracle_threshold": {
                "precision": 0.936, "recall": 0.943, "f1": 0.939,
                "accuracy": 0.989, "threshold": 0.877,
                "confusion_matrix": [[845900, 5136], [4544, 75138]],
            },
        },
        "provenance": _provenance(cell, seed, ucw=None),
    }


def _link_metrics(cell: str, seed: int, acc: float) -> dict:
    """What `evaluate_link.py` writes: a THIRD layout with no mIoU at all."""
    return {
        "experiment_name": cell,
        "n_pairs": 930718,
        "n_positive": 79682,
        "argmax": {"accuracy": acc, "precision": 0.8, "recall": 0.8,
                   "f1": 0.8, "auc": 0.93},
        "oracle_threshold": {"value": 0.5, "accuracy": 0.99, "precision": 0.9,
                             "recall": 0.9, "f1": 0.9, "note": "do not report"},
        "reportable": "argmax",
        "provenance": _provenance(cell, seed, ucw=None),
    }


def _resolved_config(cell: str, seed: int, *, ucw, cap: int = 60,
                     patience: int = 10, loss_block: bool = True) -> str:
    """The ``config.resolved.yaml`` `run_all.write_resolved_config` writes."""
    cfg: dict = {
        "experiment_name": cell,
        "seed": seed,
        "training": {"max_epochs": cap, "patience": patience},
        "paths": {"results": f"results/{cell}/seed{seed}"},
    }
    if loss_block:
        loss: dict = {"type": "ce"}
        if ucw is not None:
            loss["use_class_weights"] = ucw
        cfg["loss"] = loss
    return yaml.safe_dump(cfg, sort_keys=True)


def _write_run(results: Path, cell: str, seed: int, metrics: dict,
               cfg_text: str | None, filename: str = "eval_metrics.json"):
    run = results / cell / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / filename).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if cfg_text is not None:
        (run / "config.resolved.yaml").write_text(cfg_text, encoding="utf-8")
    return run


def write_flat_tier(results: Path, cells, seeds=(0, 1, 2), *, ucw=False) -> Path:
    """A complete, sound semantic tier: distinct accuracies, one test set."""
    for ci, cell in enumerate(cells):
        for s in seeds:
            _write_run(results, cell, s,
                       _flat_metrics(cell, s, 0.90 + 0.01 * ci + 0.002 * s,
                                     ucw=ucw),
                       _resolved_config(cell, s, ucw=ucw))
    return results


def write_unified_tier(results: Path, seeds=(0, 1, 2, 3, 4)) -> Path:
    """A complete, sound T5: `train_unified` states no weighting policy."""
    for s in seeds:
        _write_run(results, T5_CELL, s, _nested_metrics(T5_CELL, s,
                                                        0.94 + 0.001 * s),
                   _resolved_config(T5_CELL, s, ucw=None),
                   filename="unified_eval_metrics.json")
    return results


def load(results: Path, cell: str, seed: int,
         filename: str = "eval_metrics.json") -> dict:
    return json.loads((results / cell / f"seed{seed}" / filename)
                      .read_text(encoding="utf-8"))


def save(results: Path, cell: str, seed: int, m: dict,
         filename: str = "eval_metrics.json") -> None:
    (results / cell / f"seed{seed}" / filename).write_text(
        json.dumps(m, indent=2), encoding="utf-8")


def matching(errors, needle: str) -> list[str]:
    return [e for e in errors if needle in e]


# --------------------------------------------------------------------- #
#  the fixtures themselves must be sound, or nothing below means anything
# --------------------------------------------------------------------- #

def test_the_flat_fixture_tier_is_sound(tmp_path):
    results = write_flat_tier(tmp_path / "results", T1_CELLS)
    errors, _warns = check_tier.check("T1", results, [0, 1, 2])
    assert errors == [], (
        f"the clean fixture must pass, or no mutation below proves "
        f"anything: {errors}")


def test_the_unified_fixture_tier_is_sound(tmp_path):
    results = write_unified_tier(tmp_path / "results")
    errors, _warns = check_tier.check("T5", results, [0, 1, 2, 3, 4])
    assert errors == [], f"the clean T5 fixture must pass: {errors}"


def test_the_weighted_fixture_tiers_are_sound(tmp_path):
    """T4 and T6 are the labelled weighted tiers; clean, they must pass."""
    r4 = write_flat_tier(tmp_path / "t4", [T4_CELL], seeds=(0, 1, 2), ucw=True)
    errors, _ = check_tier.check("T4", r4, [0, 1, 2])
    assert errors == [], f"a clean weighted T4 was rejected: {errors}"

    r6 = write_flat_tier(tmp_path / "t6", T6_CELLS, seeds=(0, 1, 2), ucw=True)
    errors, _ = check_tier.check("T6", r6, [0, 1, 2])
    assert errors == [], f"a clean weighted T6 was rejected: {errors}"


# --------------------------------------------------------------------- #
#  DEFECT 1 (G1/G2) -- the nested unified layout                        #
# --------------------------------------------------------------------- #

def test_out_of_range_instance_auc_and_bottom_f1_are_caught(tmp_path):
    """The reproduced probe: ``instance.auc = 2.5``, ``bottom.f1 = -3.0``.

    Before the fix this tier printed ``TIER T5 IS SOUND``, exit 0, stdout
    byte-identical to the clean baseline -- including "metrics in range".
    """
    results = write_unified_tier(tmp_path / "results")
    m = load(results, T5_CELL, 2, "unified_eval_metrics.json")
    m["instance"]["auc"] = 2.5
    m["bottom"]["f1"] = -3.0
    save(results, T5_CELL, 2, m, "unified_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])

    assert matching(errors, "instance.auc"), (
        f"instance.auc = 2.5 was not reported. Errors: {errors}")
    assert matching(errors, "bottom.f1"), (
        f"bottom.f1 = -3.0 was not reported. Errors: {errors}")


def test_a_rate_buried_in_per_class_is_range_checked(tmp_path):
    """The walk must reach the leaves, not just the three block headlines."""
    results = write_unified_tier(tmp_path / "results")
    m = load(results, T5_CELL, 1, "unified_eval_metrics.json")
    m["semantic"]["per_class"]["0"]["f1"] = 1.7
    m["semantic"]["iou_per_class"]["1"] = -0.2
    m["instance"]["checkpoint_threshold"]["precision"] = 44.0
    save(results, T5_CELL, 1, m, "unified_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])

    assert matching(errors, "semantic.per_class.0.f1"), errors
    assert matching(errors, "semantic.iou_per_class.1"), errors
    assert matching(errors, "instance.checkpoint_threshold.precision"), errors


def test_deleting_the_instance_and_bottom_blocks_is_caught(tmp_path):
    """The second half of the probe: both blocks removed entirely passed."""
    results = write_unified_tier(tmp_path / "results")
    m = load(results, T5_CELL, 3, "unified_eval_metrics.json")
    del m["instance"]
    del m["bottom"]
    save(results, T5_CELL, 3, m, "unified_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])

    hits = matching(errors, "instance")
    assert hits, f"a T5 run with no instance block passed: {errors}"
    assert any("bottom" in e for e in hits), hits


def test_a_flat_metrics_file_in_a_unified_cell_is_caught(tmp_path):
    """The layout the tier table says the cell was scored with is a fact."""
    results = write_unified_tier(tmp_path / "results")
    _write_run(results, T5_CELL, 4, _flat_metrics(T5_CELL, 4, 0.94, ucw=None),
               _resolved_config(T5_CELL, 4, ucw=None),
               filename="unified_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])

    assert matching(errors, "evaluate_unified"), (
        f"a flat file in a cell the tier table scores with evaluate_unified "
        f"was accepted: {errors}")


def test_marking_the_oracle_row_reportable_is_refused_on_a_unified_file(tmp_path):
    """G2: the refusal existed, behind a predicate that is False for T5.

    ``instance.reportable`` = ``"oracle_threshold"`` -- a row tuned on the TEST
    labels -- gave exit 0 and stdout byte-identical to the clean baseline. T5 is
    the only tier in the batch that writes an oracle row at all.
    """
    results = write_unified_tier(tmp_path / "results")
    m = load(results, T5_CELL, 0, "unified_eval_metrics.json")
    m["instance"]["reportable"] = "oracle_threshold"
    save(results, T5_CELL, 0, m, "unified_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])

    hits = matching(errors, "oracle_threshold")
    assert hits, f"an oracle row marked reportable was accepted: {errors}"
    assert any("TEST" in e or "test labels" in e for e in hits), hits


def test_a_unified_file_whose_reportable_names_a_missing_block_is_caught(tmp_path):
    results = write_unified_tier(tmp_path / "results")
    m = load(results, T5_CELL, 0, "unified_eval_metrics.json")
    m["instance"]["reportable"] = "train_threshold"
    save(results, T5_CELL, 0, m, "unified_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])
    assert matching(errors, "train_threshold"), errors


# --------------------------------------------------------------------- #
#  DEFECT 2 (G5) -- mean_iou was optional here and fatal next door      #
# --------------------------------------------------------------------- #

def test_one_seed_missing_mean_iou_is_an_error_not_a_note(tmp_path):
    """`aggregate_seeds` exits 1 on this tree; `check_tier` printed a note."""
    results = write_flat_tier(tmp_path / "results", T1_CELLS)
    m = load(results, T1_CELLS[0], 2)
    del m["mean_iou"]
    save(results, T1_CELLS[0], 2, m)

    errors, warns = check_tier.check("T1", results, [0, 1, 2])

    assert matching(errors, "mean_iou"), (
        f"a semantic run with no mean_iou was demoted to a warning. "
        f"errors={errors} warns={warns}")
    assert not matching(warns, "no mean_iou"), warns


def test_every_seed_missing_mean_iou_is_an_error(tmp_path):
    """The worse half: both tools certified a table with an empty mIoU column."""
    results = write_flat_tier(tmp_path / "results", T1_CELLS)
    for cell in T1_CELLS:
        for s in (0, 1, 2):
            m = load(results, cell, s)
            del m["mean_iou"]
            save(results, cell, s, m)

    errors, _ = check_tier.check("T1", results, [0, 1, 2])
    assert len(matching(errors, "mean_iou")) == len(T1_CELLS) * 3, errors


def test_a_missing_mean_iou_in_the_nested_layout_is_an_error(tmp_path):
    results = write_unified_tier(tmp_path / "results")
    m = load(results, T5_CELL, 1, "unified_eval_metrics.json")
    del m["semantic"]["mean_iou"]
    save(results, T5_CELL, 1, m, "unified_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])
    assert matching(errors, "mean_iou"), errors


def test_a_link_layout_is_never_asked_for_a_mean_iou(tmp_path):
    """The reason `optional` existed, and the reason it may now go.

    mIoU is undefined for pair classification: `evaluate_link.py` writes none,
    and before the link branch existed (FIX 21 Sep 2026, defect 6) a link file
    fell through the semantic accessor, which is what ``optional`` was
    tolerating. Requiring mIoU in the semantic branch must not resurrect that.
    """
    results = tmp_path / "results"
    for s in (0, 1, 2, 3, 4):
        _write_run(results, T5_CELL, s, _link_metrics(T5_CELL, s, 0.9 + 0.001 * s),
                   _resolved_config(T5_CELL, s, ucw=None),
                   filename="link_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])

    assert not matching(errors, "mean_iou"), (
        f"a link run was asked for a mean_iou it cannot have: {errors}")
    # it is still wrong for a cell the tier table scores with evaluate_unified,
    # and the gate must say so
    assert matching(errors, "evaluate_unified"), errors


# --------------------------------------------------------------------- #
#  DEFECT 3 (G6) -- "seeds present and distinct" tested nothing         #
# --------------------------------------------------------------------- #

def test_a_duplicated_seed_is_detected(tmp_path):
    """The probe: seed4 is a byte-copy of seed0, relabelled ``seed: 4``.

    Before the fix: exit 0, ``TIER T5 IS SOUND``, and the arm then reported a
    5-seed error bar over 4 real runs.
    """
    results = write_unified_tier(tmp_path / "results")
    m0 = load(results, T5_CELL, 0, "unified_eval_metrics.json")
    dup = copy.deepcopy(m0)
    dup["provenance"]["seed"] = 4
    save(results, T5_CELL, 4, dup, "unified_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])

    hits = [e for e in errors if "identical" in e.lower()]
    assert hits, f"a duplicated run passed as a fifth seed: {errors}"
    assert "0" in hits[0] and "4" in hits[0], hits


def test_the_seed_spread_check_alone_would_not_have_caught_the_duplicate(tmp_path):
    """Pins WHY the fix was needed: the existing collapse check cannot fire.

    `SEED_SPREAD_EPS` catches five identical accuracies. One duplicate among
    five leaves a perfectly healthy spread, so if this ever fails the rejection
    above proves nothing.
    """
    results = write_unified_tier(tmp_path / "results")
    m0 = load(results, T5_CELL, 0, "unified_eval_metrics.json")
    dup = copy.deepcopy(m0)
    dup["provenance"]["seed"] = 4
    save(results, T5_CELL, 4, dup, "unified_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])
    assert not matching(errors, "the same accuracy"), (
        "the seed-spread check fired, so the rejection is not attributable to "
        "the duplicate-run check")


def test_two_seeds_sharing_a_config_hash_is_an_error(tmp_path):
    """`config_hash` answers "is this the same RUN?" and must differ per seed.

    `src/utils/provenance.arm_config_hash`'s docstring states the invariant:
    run_all pins a different ``seed`` and ``paths`` block into every resolved
    config, so the five seeds of one arm carry five different ``config_hash``
    values. Two that agree mean the seed never reached the config.
    """
    results = write_flat_tier(tmp_path / "results", T1_CELLS)
    m = load(results, T1_CELLS[1], 2)
    m["provenance"]["config_hash"] = load(
        results, T1_CELLS[1], 0)["provenance"]["config_hash"]
    save(results, T1_CELLS[1], 2, m)

    errors, _ = check_tier.check("T1", results, [0, 1, 2])
    assert matching(errors, "config_hash"), errors


# --------------------------------------------------------------------- #
#  DEFECT 4 (G8) -- arm_config_hash was read by no gate at all          #
# --------------------------------------------------------------------- #

def test_one_seed_with_a_foreign_arm_config_hash_is_an_error(tmp_path):
    """The probe: one seed set to ``deadbeefdeadbeef`` passed."""
    results = write_flat_tier(tmp_path / "results", T1_CELLS)
    m = load(results, T1_CELLS[0], 1)
    m["provenance"]["arm_config_hash"] = "deadbeefdeadbeef"
    save(results, T1_CELLS[0], 1, m)

    errors, _ = check_tier.check("T1", results, [0, 1, 2])
    hits = matching(errors, "arm_config_hash")
    assert hits, f"five seeds of one arm disagreed on the arm hash: {errors}"
    assert "deadbeefdeadbeef" in hits[0], hits


def test_two_cells_of_one_tier_sharing_an_arm_config_hash_is_an_error(tmp_path):
    """Two arms that hash the same are one experiment counted twice."""
    results = write_flat_tier(tmp_path / "results", T1_CELLS)
    shared = load(results, T1_CELLS[0], 0)["provenance"]["arm_config_hash"]
    for s in (0, 1, 2):
        m = load(results, T1_CELLS[1], s)
        m["provenance"]["arm_config_hash"] = shared
        save(results, T1_CELLS[1], s, m)

    errors, _ = check_tier.check("T1", results, [0, 1, 2])
    hits = matching(errors, "arm_config_hash")
    assert hits, f"two arms sharing one arm hash passed: {errors}"
    assert T1_CELLS[0] in hits[0] and T1_CELLS[1] in hits[0], hits


def test_a_run_with_no_arm_config_hash_is_an_error(tmp_path):
    """`build_provenance` stamps it on every run it builds."""
    results = write_flat_tier(tmp_path / "results", T1_CELLS)
    m = load(results, T1_CELLS[2], 0)
    del m["provenance"]["arm_config_hash"]
    save(results, T1_CELLS[2], 0, m)

    errors, _ = check_tier.check("T1", results, [0, 1, 2])
    assert matching(errors, "arm_config_hash"), errors


# --------------------------------------------------------------------- #
#  DEFECT 5 -- the policy was read from the LOWEST seed only            #
# --------------------------------------------------------------------- #

def test_flipping_the_policy_in_seeds_1_and_2_is_caught(tmp_path):
    """The probe: seed0 honest, the rest flipped -- output byte-identical.

    ``cfg_seed = sorted(found)[0]`` read seed0's config and no other.
    """
    results = write_flat_tier(tmp_path / "results", T1_CELLS)
    for s in (1, 2):
        (results / T1_CELLS[0] / f"seed{s}" / "config.resolved.yaml").write_text(
            _resolved_config(T1_CELLS[0], s, ucw=True), encoding="utf-8")
        m = load(results, T1_CELLS[0], s)
        m["provenance"]["use_class_weights"] = True
        save(results, T1_CELLS[0], s, m)

    errors, _ = check_tier.check("T1", results, [0, 1, 2])
    assert matching(errors, "use_class_weights") or matching(errors, "policies"), (
        f"a cell whose seeds were trained under two different weighting "
        f"policies passed: {errors}")


def test_a_config_that_disagrees_with_what_the_run_recorded_is_caught(tmp_path):
    """`build_provenance` copies ``loss.use_class_weights`` into provenance.

    So the resolved config beside a run and the run's own record cannot
    legitimately disagree: one of the two artifacts does not belong to the run.
    No gate compared them on any tier.
    """
    results = write_flat_tier(tmp_path / "results", T1_CELLS)
    m = load(results, T1_CELLS[3], 1)
    m["provenance"]["use_class_weights"] = True      # config still says false
    save(results, T1_CELLS[3], 1, m)

    errors, _ = check_tier.check("T1", results, [0, 1, 2])
    hits = matching(errors, "use_class_weights")
    assert hits, f"config and provenance disagreed and nothing noticed: {errors}"
    assert any("provenance" in e for e in hits), hits


def test_a_policy_missing_from_a_higher_seeds_config_is_caught(tmp_path):
    """`class_weighting_enabled` refuses to guess -- on every seed, not seed0."""
    results = write_flat_tier(tmp_path / "results", T1_CELLS)
    (results / T1_CELLS[0] / "seed2" / "config.resolved.yaml").write_text(
        _resolved_config(T1_CELLS[0], 2, ucw=None), encoding="utf-8")

    errors, _ = check_tier.check("T1", results, [0, 1, 2])
    assert matching(errors, "use_class_weights"), errors


# --------------------------------------------------------------------- #
#  DEFECT 6 (G4) -- nothing verified a WEIGHTED tier was weighted       #
# --------------------------------------------------------------------- #

def test_a_weighted_tier_whose_every_artifact_says_unweighted_is_caught(tmp_path):
    """The G4 probe, on T4: config AND provenance flipped on every seed.

    ``if tier not in run_all.WEIGHTED_TIERS`` skipped the decision-2 check
    outright for exactly the two tiers whose claim is that they are weighted.
    """
    results = write_flat_tier(tmp_path / "results", [T4_CELL], seeds=(0, 1, 2),
                              ucw=False)

    errors, _ = check_tier.check("T4", results, [0, 1, 2])
    assert matching(errors, "weight"), (
        f"a WEIGHTED tier every artifact of which says unweighted was declared "
        f"sound: {errors}")


def test_one_unweighted_seed_in_a_weighted_tier_is_caught(tmp_path):
    """T6 is four weighted arms; one seed that was not weighted breaks it."""
    results = write_flat_tier(tmp_path / "results", T6_CELLS, seeds=(0, 1, 2),
                              ucw=True)
    (results / T6_CELLS[2] / "seed1" / "config.resolved.yaml").write_text(
        _resolved_config(T6_CELLS[2], 1, ucw=False), encoding="utf-8")
    m = load(results, T6_CELLS[2], 1)
    m["provenance"]["use_class_weights"] = False
    save(results, T6_CELLS[2], 1, m)

    errors, _ = check_tier.check("T6", results, [0, 1, 2])
    hits = matching(errors, "weight")
    assert hits, f"an unweighted run inside the weighted mirror passed: {errors}"
    assert any(T6_CELLS[2] in e for e in hits), hits


def test_the_unweighted_tiers_one_policy_rule_still_fires(tmp_path):
    """Decision 2's original check must not have been traded for the new one."""
    results = write_flat_tier(tmp_path / "results", T1_CELLS, seeds=(0, 1, 2),
                              ucw=False)
    for s in (0, 1, 2):
        (results / T1_CELLS[1] / f"seed{s}" / "config.resolved.yaml").write_text(
            _resolved_config(T1_CELLS[1], s, ucw=True), encoding="utf-8")
        m = load(results, T1_CELLS[1], s)
        m["provenance"]["use_class_weights"] = True
        save(results, T1_CELLS[1], s, m)

    errors, _ = check_tier.check("T1", results, [0, 1, 2])
    assert (matching(errors, "decision 2")
            or matching(errors, "different class-weighting")), errors


# --------------------------------------------------------------------- #
#  the banner may not claim what it does not test                       #
# --------------------------------------------------------------------- #

def test_the_pass_banner_names_only_checks_that_exist():
    """G6 was found by grepping the banner for a word nothing implemented."""
    src = (CODEBASE / "runners" / "check_tier.py").read_text(encoding="utf-8")
    body = src.split('print("  every check passed')[1]
    assert "distinct" in body, "the banner no longer claims distinctness"
    # and the claim is now backed by a check somewhere in the file
    assert src.count("distinct") > 1, (
        "'distinct' appears only in the pass banner -- the exact shape of G6")


def test_auc_only_is_a_legal_reportable_and_names_no_block(tmp_path):
    """An OVER-FIRING guard, not a mutation test: it passes before and after.

    `evaluate_unified.py:513-527` writes `oracle_threshold` unconditionally and
    `checkpoint_threshold` only when the checkpoint carried a validation-fitted
    threshold; `:499-500` then sets ``reportable`` to ``"checkpoint_threshold"``
    or to ``"auc_only"``. So a healthy file can say ``auc_only`` and carry no
    block of that name, and the new "reportable names a block this file does
    not have" refusal must not fire on it -- a gate failing for the wrong
    reason is only marginally better than one passing for the wrong reason.
    """
    results = write_unified_tier(tmp_path / "results")
    for s in (0, 1, 2, 3, 4):
        m = load(results, T5_CELL, s, "unified_eval_metrics.json")
        del m["instance"]["checkpoint_threshold"]
        m["instance"]["reportable"] = "auc_only"
        m["instance"]["checkpoint_threshold_source"] = None
        save(results, T5_CELL, s, m, "unified_eval_metrics.json")

    errors, _ = check_tier.check("T5", results, [0, 1, 2, 3, 4])
    assert errors == [], (
        f"a threshold-free unified run, which evaluate_unified.py writes when "
        f"the checkpoint carried no validation-fitted threshold, was rejected: "
        f"{errors}")

#!/usr/bin/env python3
"""
check_tier.py -- is a finished tier's output actually sound?

A tier that exits zero has only proved that nothing crashed. This asks the
different question: are the numbers it produced trustworthy enough to spend the
next tier's GPU hours on top of them, and eventually to put in a paper?

It is meant to run BETWEEN tiers. `chain_v3.sh` stops the sequence when this
fails, so a broken tier is found in minutes rather than after another twenty
runs have been stacked on it.

WHAT IT CHECKS
=================================================

Per run
  provenance complete   the regiment's own section-4 checklist; without it the
                        number is not traceable and may not be quoted
  metrics finite        a NaN loss silently produces a NaN accuracy that prints
                        as a number in some formatters
  accuracy in (0, 1]    a 0.0 means the head never learned; a >1 means the
                        denominator is wrong
  fingerprints PRESENT  both the training code_fingerprint and the
                        evaluated code_fingerprint are required on every run
  the right layout      three metrics layouts exist, not two: link runs write
                        argmax/oracle_threshold and have no overall_accuracy,
                        macro_f1 or mean_iou. They are recognised explicitly
                        (`is_link_metrics`) and scored through `link_headline`
                        on the row the evaluator marked reportable, never the
                        oracle row
  mIoU <= macro-F1      a mathematical invariant on the same predictions: both
                        divide by the same errors, but F1 counts true positives
                        twice. A violation means the two were computed from
                        different predictions or different confusion matrices

Per cell, across its seeds
  all seeds present     a missing seed is a silently smaller sample
  identical n_samples   every seed must score the SAME held-out set; a differing
                        count means the split moved between runs
  results actually VARY  if five seeds give identical accuracy to 6 decimal
                        places, the seed is not reaching the thing it is meant
                        to randomise. The run looks perfect and the error bars
                        are a fiction. This is the check most likely to catch a
                        real regression in the seeding layer

Per tier, across its cells
  one SELECTION RULE    arms stop at DIFFERENT epochs BY DESIGN. What
                        must match is the rule the arms were SELECTED under:
                        one training.max_epochs and one training.patience, the
                        same pair run_all.check_selection_parity demands of the
                        configs before a tier launches. See
                        `recorded_selection_rule`
  one code fingerprint  arms produced by different code are not comparable
  fingerprint is LIVE   the check above is an AGREEMENT check: a
                        whole tier produced by superseded code passes it,
                        because every run agrees with every other run. Every
                        recorded fingerprint -- training AND scoring -- is
                        compared against code_fingerprint() of the tree the gate
                        is running from
  one weighting policy  the class-weighting rule, except the labelled weighted
                        tiers. A cell whose config does not STATE
                        a policy is not counted as agreeing with the ones
                        that do: if its trainer requires the key it is a
                        failure, and if its trainer has no weighting at all
                        (train_unified, train_link) it is reported as a cell
                        the rule could not be applied to

Usage
-----
    python runners/check_tier.py --tier T1
    python runners/check_tier.py --tier T3 --results results
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.class_weights import (class_weighting_enabled,  # noqa: E402
                                     required_training_budget)
from src.utils.provenance import check_regiment_fields, code_fingerprint  # noqa: E402
from runners import check_convergence  # noqa: E402

#: below this spread across seeds, the seeds are not doing anything
SEED_SPREAD_EPS = 1e-6


def _load_run_all():
    spec = importlib.util.spec_from_file_location(
        "run_all", REPO / "runners" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trainer_requires_weighting_policy(trainer: str) -> bool | None:
    """Does *trainer* refuse to start without ``loss.use_class_weights``?

    Read off the trainer's own source rather than from a list kept here, for
    the same reason the weighted-tier set lives in `run_all.py`: a second list
    is a list that goes stale. `src/training/train_semantic.py` and
    `train_resnet.py` call `class_weighting_enabled` and will not run without
    the key, so a resolved config of theirs that lacks it is corrupt.
    `train_unified.py` and `train_link.py` never mention class weighting -- the
    unified model's `loss:` block weights the three TASKS, not the classes --
    and `configs/mfinstseg_unified.yaml` accordingly carries no such key. Tier
    T5 is in the imminent batch, so failing it for not stating a policy its
    trainer cannot read would be a gate failing for the wrong reason.

    Returns:
        True or False, or None if the trainer module cannot be found -- which
        the caller reports rather than resolving either way.
    """
    if not trainer:
        return None
    path = REPO / (trainer.replace(".", "/") + ".py")
    if not path.is_file():
        return None
    return "class_weighting_enabled" in path.read_text(encoding="utf-8")


def _metrics_file(run_dir: Path) -> Path | None:
    """The metrics file an evaluator left in *run_dir*, whatever it named it.

    The four names are the same list, in the same order, as
    `runners/run_all.py` `metrics_path`. `tests/test_gates_fingerprint.py`
    pins them together so they cannot drift apart.
    """
    for name in ("eval_metrics.json", "unified_eval_metrics.json",
                 "link_eval_metrics.json", "metrics.json"):
        if (run_dir / name).exists():
            return run_dir / name
    return None


def is_link_metrics(m: dict) -> bool:
    """True for a ``link_eval_metrics.json`` -- a THIRD layout, not a semantic one.

    the ``metrics = {...}`` literal at the end of `evaluate_link()`
    (`src/evaluation/evaluate_link.py:439-466`) writes an
    ``argmax`` block and an
    ``oracle_threshold`` block, unconditionally and side by side, and nothing
    else in this repository writes that pair. There is no ``overall_accuracy``,
    no ``macro_f1`` and no ``mean_iou`` anywhere in the file: the numbers are
    binary accuracy/precision/recall/F1/AUC over face PAIRS.

    This predicate exists because simply adding the filename to `_metrics_file`
    is NOT enough. `semantic_block` would fall through to its FLAT branch, every
    semantic key would come back absent, and the gate would report "accuracy
    absent, macro_f1 absent" for a link run that is in perfect health -- a gate
    failing for the wrong reason, which is only marginally better than one
    passing for the wrong reason.
    """
    return (isinstance(m.get("argmax"), dict)
            and isinstance(m.get("oracle_threshold"), dict))


def link_headline(m: dict) -> tuple[float | None, float | None,
                                    float | None, int | None]:
    """(accuracy, F1, AUC, n_pairs) from the row evaluate_link marked REPORTABLE.

    Which row may be quoted is a fact the evaluator records, not one this gate
    may guess: ``oracle_threshold`` is fitted on the test labels and its own
    note says "do not report". So the row is selected by ``m["reportable"]``.

    Raises:
        ValueError: If ``reportable`` is absent, names the oracle row, or names
            a block the file does not contain. Each of those means the gate
            cannot tell which number is the result, and picking one anyway is
            how an oracle number reaches a paper.
    """
    which = m.get("reportable")
    if which is None:
        raise ValueError(
            "link metrics file has no 'reportable' key, so there is no way to "
            "tell which threshold row may be quoted. evaluate_link.py writes it "
            "on every run; a file without it was written by something else or "
            "hand-edited. Re-score the run with "
            "src/evaluation/evaluate_link.py.")
    if which == "oracle_threshold":
        raise ValueError(
            "link metrics file marks 'oracle_threshold' reportable. That row is "
            "tuned on the TEST labels and is an upper bound only -- the file's "
            "own note says do not report it. Re-score with "
            "--tune-threshold-on-train, or quote the argmax row.")
    row = m.get(which)
    if not isinstance(row, dict):
        raise ValueError(
            f"link metrics file marks {which!r} reportable but carries no such "
            f"block (top-level keys: {sorted(m)}). The file is inconsistent with "
            f"itself; re-score the run.")
    return (row.get("accuracy"), row.get("f1"),
            m["argmax"].get("auc"), m.get("n_pairs"))


def semantic_block(m: dict) -> dict:
    """The sub-dict that actually holds the semantic-task numbers.

    Two layouts are in circulation, and both are right for what wrote them:

    * FLAT -- ``evaluate_semantic.py`` scores one task, so ``overall_accuracy``
      sits at the top level next to ``provenance``.
    * NESTED -- ``evaluate_unified.py`` (tier T5) scores three tasks, so the
      semantic ones live under ``["semantic"]`` beside ``["instance"]`` and
      ``["bottom"]``, and the denominator is called ``n_faces`` rather than
      ``n_samples``.

    Returning the block instead of a fixed tuple is what lets
    `aggregate_seeds.py` reach the confusion matrix through this same accessor.
    A second copy of the branch is how one of the two tools ends up silently
    reading nothing on T5.

    Raises:
        ValueError: If handed a ``link_eval_metrics.json``. Link runs go
            through `link_headline`; see `is_link_metrics`.
    """
    if is_link_metrics(m):
        raise ValueError(
            "this is a link_eval_metrics.json (it has 'argmax' and "
            "'oracle_threshold'), which is a THIRD metrics layout and carries "
            "no overall_accuracy, macro_f1 or mean_iou at all. Reading it as a "
            "semantic file yields four Nones and no error. Use "
            "check_tier.link_headline(m) for link runs. If you reached here "
            "from aggregate_seeds.py: that script builds the SEMANTIC table and "
            "has no column a link run belongs in.")
    nested = m.get("semantic")
    if isinstance(nested, dict):
        return nested
    return m


def headline_metrics(
    m: dict,
) -> tuple[float | None, float | None, float | None, int | None]:
    """(accuracy, macro_f1, mIoU, n_samples) from either metrics layout."""
    s = semantic_block(m)
    n = s.get("n_samples")
    if n is None:
        n = s.get("n_faces")
    return (s.get("overall_accuracy"), s.get("macro_f1"), s.get("mean_iou"), n)


# the NESTED unified layout -- three heads

#: Keys whose value is a RATE -- a proportion that cannot be negative and
#: cannot exceed 1. Collected BY NAME because the layouts nest them at four
#: different depths (`semantic.macro_f1`, `bottom.f1`,
#: `instance.checkpoint_threshold.precision`, `semantic.per_class.7.recall`).
#:
#: Counts (`n_faces`, `n_pairs`, `n_positive`, `support`) and the similarity
#: cutoff (`threshold`) are deliberately NOT here: they are not proportions,
#: and range-checking them to [0, 1] would be a gate failing for the wrong
#: reason, which is only marginally better than one passing for the wrong one.
RATE_KEYS = frozenset({
    "overall_accuracy", "accuracy", "precision", "recall", "f1",
    "macro_precision", "macro_recall", "macro_f1", "auc",
    "mean_iou", "iou_not_bottom", "iou_bottom",
})

#: Sub-dicts whose KEYS are class ids and whose every VALUE is a rate, so the
#: name test above cannot be applied to the leaf and is applied to the parent.
RATE_VALUED_BLOCKS = frozenset({"iou_per_class"})

#: The three heads `evaluate_unified.py` scores, named as its `metrics = {...}`
#: literal names them (src/evaluation/evaluate_unified.py:463-512). All three
#: are written unconditionally, so a file carrying some and not others was
#: hand-edited or written by something else.
UNIFIED_BLOCKS = ("semantic", "instance", "bottom")

#: The evaluator the tier table names for a cell scored in that layout. Read
#: from `run_all.TIERS` rather than from a second list kept here, for the same
#: reason `_trainer_requires_weighting_policy` reads the trainer's source: a
#: second list is a list that goes stale.
UNIFIED_EVALUATOR = "src.evaluation.evaluate_unified"

#: The one value of ``reportable`` that names no block. `evaluate_unified.py`
#: writes it when the checkpoint carried no validation-fitted threshold
#: (:499-500): there is then no quotable threshold row at all and the AUC,
#: which is threshold-free, is the result. It is a legal answer, so it is
#: listed here rather than read as "names a block the file does not have".
REPORTABLE_NO_BLOCK = frozenset({"auc_only"})


def is_unified_metrics(m: dict) -> bool:
    """True for the NESTED three-head layout `evaluate_unified.py` writes.

    Selected on the blocks PRESENT rather than on the filename: `_metrics_file`
    already accepts four names, and a renamed file is still the file it is.
    Any ONE of the three blocks is enough to recognise the layout, so a file
    that has lost two of them is recognised and then failed for having lost
    them -- rather than silently re-read as some other layout.
    """
    return any(isinstance(m.get(b), dict) for b in UNIFIED_BLOCKS)


def cell_expects_unified(cell) -> bool:
    """Does the TIER TABLE say this cell is scored by `evaluate_unified`?

    The layout a run should have is a fact the tier table states, not one the
    gate may infer from the file in front of it -- inferring it is how a file
    with no `instance` block reads as "a flat run, nothing to check here".
    """
    return getattr(cell, "evaluator", None) == UNIFIED_EVALUATOR


def rate_errors(label: str, m: dict) -> list[str]:
    """Every numeric rate anywhere in *m*, checked finite and inside [0, 1].

    The bound is INCLUSIVE at both ends, unlike the headline accuracy's
    ``(0, 1]``: a per-class IoU or recall of exactly 0 is a real answer for a
    class the model never predicts, and failing a tier for one would be a gate
    failing for the wrong reason. A zero OVERALL accuracy is a different claim
    -- "the head never learned" -- and keeps its own stricter check in `check`.

    ``provenance`` is excluded: it records the run, not the measurements, and
    nothing in it is a rate.
    """
    errors: list[str] = []

    def walk(d: dict, path: str, parent: str | None) -> None:
        for key, value in d.items():
            where = f"{path}.{key}" if path else str(key)
            if isinstance(value, dict):
                walk(value, where, str(key))
            elif isinstance(value, bool):
                continue                      # a flag is not a rate
            elif (isinstance(value, (int, float))
                    and (key in RATE_KEYS or parent in RATE_VALUED_BLOCKS)):
                if not math.isfinite(value):
                    errors.append(
                        f"{label}: {where} is {value} -- not a finite number. "
                        f"Whatever produced it diverged or divided by zero, "
                        f"and it prints as a number in some formatters.")
                elif not (0.0 <= value <= 1.0):
                    errors.append(
                        f"{label}: {where} is {value}, outside [0, 1]. It is a "
                        f"rate, so a value outside that range means a wrong "
                        f"denominator or a hand-edited file. Re-score the run "
                        f"with its evaluator.")

    walk({k: v for k, v in m.items() if k != "provenance"}, "", None)
    return errors


def oracle_row_errors(label: str, m: dict) -> list[str]:
    """Refuse any block that quotes a threshold fitted on the TEST labels.

    The refusal existed -- in
    `link_headline` -- but `link_headline` is reachable only behind
    `is_link_metrics`, which needs a TOP-LEVEL ``argmax`` and a top-level
    ``oracle_threshold``. T5 nests its oracle row under ``instance`` and has no
    ``argmax`` at all, so the predicate is False and the guard never ran on the
    ONE tier in the batch that writes an oracle row. Setting
    ``instance.reportable`` to ``"oracle_threshold"`` gave exit 0 and stdout
    byte-identical to the clean baseline.

    The question is asked of any block, at any depth, that carries an
    ``oracle_threshold`` sub-dict OR a ``reportable`` key -- so a fourth layout
    that nests either somewhere else is covered on the day it is written
    rather than after it reaches a paper. `runners/aggregate_seeds.py` refuses
    the same shapes; two gates giving two verdicts on one file is the failure
    mode this suite is least able to afford.
    """
    errors: list[str] = []

    def walk(d: dict, path: str) -> None:
        has_oracle = isinstance(d.get("oracle_threshold"), dict)
        which = d.get("reportable")
        if has_oracle or which is not None:
            block = path or "the top level"
            if which is None:
                errors.append(
                    f"{label}: {block} carries an 'oracle_threshold' row and "
                    f"no 'reportable' key, so there is no way to tell which "
                    f"row may be quoted -- and the oracle row is the one tuned "
                    f"on the TEST labels. Re-score the run with its evaluator, "
                    f"which writes 'reportable' on every run.")
            elif which == "oracle_threshold":
                errors.append(
                    f"{label}: {block} marks 'oracle_threshold' reportable. "
                    f"That row is tuned on the TEST labels and is an upper "
                    f"bound only -- the file's own note says do not report it. "
                    f"Quote the checkpoint_threshold row (fitted on the "
                    f"validation partition), or re-score the run.")
            elif which in REPORTABLE_NO_BLOCK:
                pass            # legal, and names no block on purpose
            elif not isinstance(d.get(which), dict):
                errors.append(
                    f"{label}: {block} marks {which!r} reportable but carries "
                    f"no such block (its keys: {sorted(d)}). The file cannot "
                    f"say which of its numbers may be quoted, which is how an "
                    f"oracle number reaches a paper. It is inconsistent with "
                    f"itself; re-score the run.")
        for key, value in d.items():
            if isinstance(value, dict):
                walk(value, f"{path}.{key}" if path else str(key))

    walk({k: v for k, v in m.items() if k != "provenance"}, "")
    return errors


def unified_layout_errors(label: str, m: dict, *, expected: bool) -> list[str]:
    """The checks a NESTED unified metrics file gets and a flat one does not.

    Args:
        label: ``"<cell> seed <k>"``, quoted in every message.
        m: The parsed metrics file.
        expected: True when the tier table scores this cell with
            `evaluate_unified`. A cell that SHOULD be nested and is not is a
            failure in its own right: it means the file beside the run was not
            written by the evaluator the table names.
    """
    errors: list[str] = []
    present = [b for b in UNIFIED_BLOCKS if isinstance(m.get(b), dict)]
    if not present:
        if expected:
            errors.append(
                f"{label}: the tier table scores this cell with "
                f"{UNIFIED_EVALUATOR}, which writes a NESTED file carrying "
                f"{', '.join(UNIFIED_BLOCKS)} -- and this metrics file carries "
                f"none of them (top-level keys: "
                f"{sorted(k for k in m if k != 'provenance')}). It was not "
                f"written by that evaluator, so the three heads this cell "
                f"exists to report are not in evidence. Re-score the run with "
                f"src/evaluation/evaluate_unified.py.")
        return errors
    missing = [b for b in UNIFIED_BLOCKS if b not in present]
    if missing:
        errors.append(
            f"{label}: this is a nested unified metrics file (it has "
            f"{', '.join(present)}) but it has no "
            f"{' block and no '.join(missing)} block. evaluate_unified.py "
            f"writes all three unconditionally "
            f"(src/evaluation/evaluate_unified.py:463-512), so a file missing "
            f"one was hand-edited or written by something else -- and each "
            f"missing head is a row this tier exists to produce.")
    errors.extend(rate_errors(label, m))
    errors.extend(oracle_row_errors(label, m))
    return errors


def run_digest(m: dict) -> str:
    """A digest of everything a run MEASURED, with its provenance removed.

    Stripping `provenance` is what makes a relabelled copy irrelevant -- the
    confusion matrix, the per-class table and the headline rates are what two
    runs cannot share by accident.

    `SEED_SPREAD_EPS` does not cover this: it catches identical accuracies,
    and one duplicate among several leaves a healthy spread.
    """
    body = {k: v for k, v in m.items() if k != "provenance"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# the selection rule -- what "comparable" means under the selection rule
#: Resolved-config states, named exactly as `runners/check_convergence.py`
#: names them (check_convergence.py:168-170), so the two gates say the same
#: word about the same file.
CFG_OK = "ok"
CFG_ABSENT = "absent"
CFG_MALFORMED = "malformed"


def recorded_selection_rule(label: str, run_dir: Path, prov: dict):
    """The ``(cap, patience)`` one finished run was actually held to.

    Comparable runs share ``training.max_epochs`` and ``training.patience``.
    Stopping epochs differ. This function reads the cap and patience the run recorded.

    Two artifacts carry the rule and both are read, because either one alone
    can be the one that moved:

    * ``provenance["epoch_budget_configured"]`` -- the cap.
      `build_provenance` copies it from ``training.max_epochs``
      (src/utils/provenance.py:227) on every run.
    * ``config.resolved.yaml`` beside the run -- the exact config the run was
      launched from, written by `run_all.write_resolved_config`. It is the only
      artifact that carries ``patience``, and it is read through
      `class_weights.required_training_budget`, the same helper the trainers
      call, so this gate refuses exactly the configs a run could not have
      started from.

    Args:
        label: ``"<cell> seed <k>"``, quoted in every message.
        run_dir: The run's directory, which holds its ``config.resolved.yaml``.
        prov: The run's provenance block.

    Returns:
        ``(cap, patience, errors, not_performed)``. ``cap`` and ``patience`` are
        None when that half of the rule could not be established; a None is
        never counted as agreeing with anything. ``not_performed`` is
        a sentence naming a check that did NOT run, for the caller to print as
        a note -- never as a pass.
    """
    errors: list[str] = []
    cap = prov.get("epoch_budget_configured")
    if cap is None:
        errors.append(
            f"{label}: provenance records no epoch_budget_configured, so the "
            f"cap this run was held to is unknown and it cannot be shown to "
            f"share a selection rule with the rest of the tier. "
            f"src/utils/provenance.py:227 copies that key from "
            f"training.max_epochs on every run build_provenance stamps, so a "
            f"run without one was not stamped by it. An unknown rule is not a "
            f"matching rule. Re-run the cell through runners/run_all.py.")

    cfg_path = run_dir / "config.resolved.yaml"
    if not cfg_path.is_file():
        # Absent, not silent -- the same answer `check_convergence` gives the
        # same condition (CFG_ABSENT -> NOT PERFORMED, check_convergence.py:229).
        return cap, None, errors, (
            f"{label}: no config.resolved.yaml in {run_dir}, so "
            f"training.patience is unknown and the patience half of the "
            f"selection-rule check was NOT PERFORMED for this run. "
            f"run_all.write_resolved_config writes that file for every cell it "
            f"launches, so a run without one did not come from run_all. The "
            f"cap half still ran.")

    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        errors.append(
            f"{label}: {cfg_path} is not valid YAML "
            f"({exc.__class__.__name__}: {' '.join(str(exc).split())}). It is "
            f"the only artifact that records the patience this run was "
            f"selected under, so the selection rule cannot be established. A "
            f"resolved config is written once, by run_all, and never edited; "
            f"delete it and re-run the cell.")
        return cap, None, errors, None

    t_cfg = cfg.get("training") if isinstance(cfg, dict) else None
    if not isinstance(t_cfg, dict):
        errors.append(
            f"{label}: {cfg_path} has no `training:` block, so it states no "
            f"selection rule and {run_dir.name}'s trainer cannot have started "
            f"from it -- every trainer calls "
            f"src.utils.class_weights.required_training_budget, which raises "
            f"on a config without training.max_epochs and training.patience. "
            f"This resolved config is not the one the run used. Re-run the "
            f"cell through runners/run_all.py.")
        return cap, None, errors, None

    try:
        cfg_cap, patience = required_training_budget(
            t_cfg, where=f"{label} ({cfg_path})")
    except (KeyError, ValueError) as exc:
        # Not swallowed, translated -- the same shape as the
        # `class_weighting_enabled` call below in `check`: this gate's contract
        # is to report every cell's problems in one pass and exit 1, so the
        # helper's refusal becomes a FAIL line rather than a traceback that
        # hides the other cells. KeyError renders with quotes and escaped
        # newlines, hence args[0].
        msg = exc.args[0] if exc.args else str(exc)
        errors.append(f"{label}: " + " ".join(str(msg).split()))
        return cap, None, errors, None

    if cap is not None and cfg_cap != cap:
        errors.append(
            f"{label}: provenance says the cap was {cap} but the resolved "
            f"config beside the run sets training.max_epochs={cfg_cap}. "
            f"epoch_budget_configured is copied from that very key "
            f"(src/utils/provenance.py:227), so one of the two artifacts does "
            f"not belong to this run and the rule it was selected under is not "
            f"established.")
    return cap, patience, errors, None


def check(tier: str, results: Path, seeds: list[int]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings)."""
    run_all = _load_run_all()
    if tier not in run_all.TIERS:
        return ([f"unknown tier {tier!r}; known: {', '.join(sorted(run_all.TIERS))}"], [])

    errors, warns = [], []
    per_tier_fp, per_tier_weight = set(), set()
    # What must match across runs is the cap and the patience. Stopping
    # epochs differ. The maps below name which run recorded which value.
    cap_by_run: dict[str, int] = {}
    patience_by_run: dict[str, int] = {}
    selected_by_run: dict[str, int] = {}
    #: Runs whose patience could not be read at all. Printed as NOT PERFORMED,
    #: never counted as agreement.
    rule_not_performed: list[str] = []
    # Computed once. Every comparison below uses this value.
    live_fp = code_fingerprint()
    #: {"<cell> seed <k>": fingerprint} -- every fingerprint any run recorded,
    #: for the recorded-vs-live comparison at the end of this function.
    fp_by_run: dict[str, str] = {}
    # Three more per-run collections,
    # all of them keyed by run for the same reason the selection rule is: an
    # error that names two values and not which run recorded them costs the
    # operator the search this gate exists to save.
    #: {arm_config_hash: {cell names that recorded it}} -- one arm hash per
    #: cell, and no two cells sharing one.
    arm_hash_owners: dict[str, set] = {}
    #: {cell: {stated class-weighting policies}} -- a SET per cell. The policy
    #: is read from every seed.
    per_cell_policy: dict[str, set] = {}
    #: {"<cell> seed <k>": provenance["use_class_weights"]} -- what each RUN
    #: recorded, as opposed to what its config states.
    weight_by_run: dict[str, object] = {}

    for cell in run_all.TIERS[tier]:
        accs, ns_seen, found = {}, {}, []
        # Every seed is kept so the cell checks can ask whether the runs are
        # distinct, share one arm, and share the tier's class-weight policy.
        prov_by_seed: dict[int, dict] = {}
        digest_by_seed: dict[int, str] = {}

        for s in seeds:
            run_dir = results / cell.name / f"seed{s}"
            mp = _metrics_file(run_dir)
            if mp is None:
                errors.append(f"{cell.name} seed {s}: no metrics file in {run_dir}")
                continue
            found.append(s)
            m = json.loads(mp.read_text(encoding="utf-8"))

            prov = m.get("provenance") or {}
            label = f"{cell.name} seed {s}"
            prov_by_seed[s] = prov
            digest_by_seed[s] = run_digest(m)
            weight_by_run[label] = prov.get("use_class_weights")

            # Collected before the completeness check. Training and scoring
            # fingerprints are both required.
            rec_fp = prov.get("code_fingerprint")
            if rec_fp:
                fp_by_run[label] = rec_fp
            else:
                errors.append(
                    f"{cell.name} seed {s}: provenance records NO "
                    f"code_fingerprint, so there is nothing to compare against "
                    f"this tree ({live_fp}) and the run cannot be "
                    f"shown to be a result of this code. A git_commit does not "
                    f"substitute here -- this folder is not a repository, which "
                    f"is why the fingerprint exists. Re-run the cell through "
                    f"runners/run_all.py, which stamps it via "
                    f"src/utils/provenance.build_provenance().")
            evaluated = prov.get("evaluated") or {}
            eval_fp = evaluated.get("code_fingerprint")
            if eval_fp:
                fp_by_run[f"{label} (scored by)"] = eval_fp
            else:
                errors.append(
                    f"{cell.name} seed {s}: provenance has no "
                    f"['evaluated']['code_fingerprint'], so the code that "
                    f"SCORED this run is unknown. Every evaluator in this "
                    f"repository stamps it -- src/utils/provenance.py:282-287 "
                    f"writes it on both branches of provenance_from_checkpoint "
                    f"-- so a metrics file without it was not written by one of "
                    f"them. Re-score the run with its evaluator.")

            # Collected before the completeness check, same as the fingerprints.
            cap, patience, rule_errors, not_performed = recorded_selection_rule(
                label, run_dir, prov)
            errors.extend(rule_errors)
            if cap is not None:
                cap_by_run[label] = cap
            if patience is not None:
                patience_by_run[label] = patience
            if not_performed:
                rule_not_performed.append(not_performed)
            sel = prov.get("selected_epoch")
            if sel is not None:
                selected_by_run[label] = sel

            gaps = check_regiment_fields(prov) if prov else ["provenance (absent)"]
            if gaps:
                errors.append(f"{cell.name} seed {s}: provenance missing "
                              f"{', '.join(gaps)} -- not quotable")
            else:
                per_tier_fp.add(prov.get("code_fingerprint"))
                if prov.get("seed") != s:
                    errors.append(f"{cell.name} seed {s}: provenance records seed "
                                  f"{prov.get('seed')!r} -- the run is mislabelled")

            # Three layouts reach this point. A link run has no
            # overall_accuracy, macro_f1 or mean_iou at all. mean_iou is
            # undefined for pair classification, so it is not warned about --
            # and AUC, which only a link run has, is range-checked instead.
            # Link runs are recognised by `is_link_metrics` and scored by
            # `link_headline`, whose `required` never mentions mIoU.
            # `evaluate_semantic.py` and `evaluate_unified.py` both write
            # `mean_iou` unconditionally, so a semantic run without one was
            # hand-edited or is not a segmentation run.
            #
            # The nested unified layout is checked as a whole -- see
            # `unified_layout_errors`.
            expects_unified = cell_expects_unified(cell)
            auc = None
            if is_link_metrics(m):
                acc, f1, auc, n = link_headline(m)
                miou = None
                required = (("accuracy", acc), ("f1", f1), ("auc", auc))
                if expects_unified:
                    errors.append(
                        f"{label}: this is a link_eval_metrics.json (it has "
                        f"'argmax' and 'oracle_threshold'), but the tier table "
                        f"scores this cell with {UNIFIED_EVALUATOR}, which "
                        f"writes the nested three-head layout. The file beside "
                        f"this run was not written by the evaluator the table "
                        f"names, so the numbers it holds are not the numbers "
                        f"this cell claims. Re-score with "
                        f"src/evaluation/evaluate_unified.py.")
            else:
                acc, f1, miou, n = headline_metrics(m)
                required = (("accuracy", acc), ("macro_f1", f1),
                            ("mean_iou", miou))
                errors.extend(unified_layout_errors(
                    label, m, expected=expects_unified))

            for name, v in required:
                if v is None:
                    errors.append(f"{cell.name} seed {s}: {name} absent")
                elif not math.isfinite(v):
                    errors.append(f"{cell.name} seed {s}: {name} is {v} -- training diverged")

            if acc is not None and math.isfinite(acc):
                if not (0.0 < acc <= 1.0):
                    errors.append(f"{cell.name} seed {s}: accuracy {acc} outside (0, 1]")
                accs[s] = acc
            if auc is not None and math.isfinite(auc) and not (0.0 < auc <= 1.0):
                errors.append(f"{cell.name} seed {s}: AUC {auc} outside (0, 1]")
            if (f1 is not None and miou is not None
                    and math.isfinite(f1) and math.isfinite(miou)
                    and miou > f1 + 1e-9):
                errors.append(
                    f"{cell.name} seed {s}: mIoU {miou:.4f} > macro-F1 {f1:.4f}, "
                    f"which is impossible on the same predictions")
            if n is not None:
                ns_seen[s] = n

        missing = sorted(set(seeds) - set(found))
        if missing:
            errors.append(f"{cell.name}: seeds {missing} never produced a result")

        if len(set(ns_seen.values())) > 1:
            errors.append(
                f"{cell.name}: seeds scored DIFFERENT test-set sizes {ns_seen} -- "
                f"the split moved between runs, so the seeds are not comparable")

        if len(accs) >= 2:
            spread = max(accs.values()) - min(accs.values())
            if spread < SEED_SPREAD_EPS:
                errors.append(
                    f"{cell.name}: all {len(accs)} seeds produced the same accuracy "
                    f"to within {SEED_SPREAD_EPS} ({sorted(set(accs.values()))}). "
                    f"The seed is not reaching what it is meant to randomise, so "
                    f"the error bars would be fiction.")
            elif spread > 0.15:
                warns.append(
                    f"{cell.name}: accuracy varies by {spread:.3f} across seeds "
                    f"(min {min(accs.values()):.4f}, max {max(accs.values()):.4f}) -- "
                    f"unusually unstable; check for a diverged run")
            if len(accs) >= 3:
                warns.append(
                    f"{cell.name}: accuracy {statistics.mean(accs.values()):.4f} "
                    f"+/- {statistics.stdev(accs.values()):.4f} over {len(accs)} seeds")

        # A relabelled copy satisfies the seed-mislabel check, which asks only
        # whether `provenance.seed` matches the directory name. Identical
        # metrics with provenance excluded are a copy.
        by_digest: dict[str, list] = {}
        for s, dg in sorted(digest_by_seed.items()):
            by_digest.setdefault(dg, []).append(s)
        for dg, dup in sorted(by_digest.items()):
            if len(dup) > 1:
                errors.append(
                    f"{cell.name}: seeds {dup} recorded IDENTICAL metrics -- "
                    f"same confusion matrix, same per-class table, same "
                    f"headline rates (payload digest {dg}, provenance "
                    f"excluded). Two runs of the same model on the same test "
                    f"set with different seeds cannot agree to the last "
                    f"decimal of every number, so one of these directories is "
                    f"a copy of another and this cell has fewer real runs than "
                    f"seeds. An error bar over them would be fiction. Re-run "
                    f"the duplicated seed(s) through runners/run_all.py.")

        # config_hash identifies a run. arm_config_hash identifies an experiment.
        # The two must agree across the seeds of one cell.
        by_config: dict[str, list] = {}
        for s, prov_s in sorted(prov_by_seed.items()):
            ch = prov_s.get("config_hash")
            if ch is None:
                continue        # already reported by check_regiment_fields
            by_config.setdefault(ch, []).append(s)
        for ch, share in sorted(by_config.items()):
            if len(share) > 1:
                errors.append(
                    f"{cell.name}: seeds {share} recorded the SAME config_hash "
                    f"{ch}. That digest covers the resolved config including "
                    f"`seed` and `paths` (src/utils/provenance.py:157-172), so "
                    f"the five seeds of one arm carry five different values by "
                    f"construction. Two that agree mean the seed never reached "
                    f"the config -- the runs were launched from one and the "
                    f"same file -- so they are not independent samples and "
                    f"their spread is not an error bar.")

        arms_seen: dict[str, list] = {}
        no_arm_hash = []
        for s, prov_s in sorted(prov_by_seed.items()):
            ah = prov_s.get("arm_config_hash")
            if not ah:
                no_arm_hash.append(s)
                continue
            arms_seen.setdefault(ah, []).append(s)
            arm_hash_owners.setdefault(ah, set()).add(cell.name)
        if no_arm_hash:
            errors.append(
                f"{cell.name}: seeds {no_arm_hash} record NO arm_config_hash. "
                f"src/utils/provenance.py:355 stamps it on every run "
                f"build_provenance builds, so a run without one was not built "
                f"by it -- and it is the only field that can show these seeds "
                f"are the same experiment rather than five different ones. "
                f"Re-run the cell through runners/run_all.py.")
        if len(arms_seen) > 1:
            errors.append(
                f"{cell.name}: the seeds of this ONE arm recorded DIFFERENT "
                f"arm_config_hash values "
                f"{ {h: v for h, v in sorted(arms_seen.items())} }. That digest "
                f"is the config with `seed` and `paths` dropped, so it is "
                f"identical across the seeds of an arm and moves the moment any "
                f"setting that can change a number changes. A disagreement "
                f"means these seeds were not all trained on the same "
                f"experiment, so they may not be averaged.")

        # A cell that produced no metrics file at all already has a hard
        # error above ("seeds ... never produced a result"); asking what
        # weighting policy it stated would add a second line about a cell that
        # did not run.
        if found:
            # Every seed's resolved config is read and checked against that
            # seed's recorded use_class_weights. A missing policy is not agreement.
            requires_policy = _trainer_requires_weighting_policy(
                getattr(cell, "trainer", None))
            if requires_policy is None:
                errors.append(
                    f"{cell.name}: the tier table gives this cell the trainer "
                    f"{getattr(cell, 'trainer', None)!r}, which this gate cannot "
                    f"find under {REPO}. Without it there is no way to tell whether "
                    f"the cell needs a class-weighting policy, so the decision-2 "
                    f"check cannot be applied -- and a check that cannot be applied "
                    f"is not a check that passed. Fix runners/run_all.py:TIERS.")
            elif not requires_policy:
                # train_unified / train_link have no class weighting at all, so
                # there is no policy to state and nothing to compare. Said out loud
                # rather than dropped: the operator should be able to see WHY a cell
                # is absent from the comparison.
                warns.append(
                    f"{cell.name}: trainer {cell.trainer} does not read "
                    f"loss.use_class_weights at all, so this cell states no "
                    f"class-weighting policy and is not part of the decision-2 "
                    f"comparison for this tier.")
                # ...which also means the tier table may not label it weighted.
                if tier in run_all.WEIGHTED_TIERS:
                    errors.append(
                        f"{cell.name}: {tier} is in run_all.WEIGHTED_TIERS -- "
                        f"its whole claim is that it is the WEIGHTED mirror -- "
                        f"but this cell's trainer {cell.trainer} has no class "
                        f"weighting at all, so no run of it can have been "
                        f"weighted. The tier table and the trainer contradict "
                        f"each other; one of them is wrong.")
                # NOT checked here: what such a run's provenance RECORDS.
                # `configs/mfinstseg_two_stage.yaml:55` -- a link config, whose
                # trainer has no class weighting -- states
                # `use_class_weights: false`, and `build_provenance` copies the
                # key from whatever config it is handed
                # (src/utils/provenance.py:337). So a healthy link run records
                # False for a setting its trainer never reads, and failing it
                # for that would be a gate failing for the wrong reason.
            else:
                # The findings are collected per seed and REPORTED per cell:
                # reading five seeds must not turn one cell's one problem into
                # five identical lines, and the cell is the unit the operator
                # acts on (re-run the cell). Which seeds were affected is named
                # inside each message.
                stated_by_seed: dict = {}
                cfg_absent: list = []
                cfg_no_loss: list = []
                cfg_unreadable: list = []
                unreadable_msg = ""
                for s_i in sorted(found):
                    seed_cfg = (results / cell.name / f"seed{s_i}"
                                / "config.resolved.yaml")
                    label_i = f"{cell.name} seed {s_i}"
                    if not seed_cfg.is_file():
                        cfg_absent.append(s_i)
                        continue
                    cfg = yaml.safe_load(seed_cfg.read_text(encoding="utf-8"))
                    loss_cfg = cfg.get("loss") if isinstance(cfg, dict) else None
                    if not isinstance(loss_cfg, dict):
                        cfg_no_loss.append(s_i)
                        continue
                    try:
                        policy = class_weighting_enabled(
                            loss_cfg, where=f"{label_i} ({seed_cfg})")
                    except (KeyError, TypeError) as exc:
                        # Not swallowed, translated: this gate's contract is to
                        # report every cell's problems in one pass and exit 1, so
                        # the helper's refusal becomes a FAIL line rather than a
                        # traceback that hides the other cells. KeyError renders
                        # with quotes and escaped newlines, hence args[0].
                        msg = exc.args[0] if exc.args else str(exc)
                        cfg_unreadable.append(s_i)
                        unreadable_msg = unreadable_msg or " ".join(
                            str(msg).split())
                        continue
                    stated_by_seed[s_i] = policy
                    recorded = prov_by_seed.get(s_i, {}).get("use_class_weights")
                    if recorded is not policy:
                        errors.append(
                            f"{cell.name}: the resolved config beside seed "
                            f"{s_i} sets loss.use_class_weights={policy!r}, but "
                            f"that run's own provenance records "
                            f"use_class_weights={recorded!r}. "
                            f"src/utils/provenance.build_provenance copies that "
                            f"key straight out of the config it was handed "
                            f"(provenance.py:337), so one of the two artifacts "
                            f"does not belong to this run and the policy it was "
                            f"trained under is not established. On this corpus "
                            f"the setting is worth about 18 points. Re-run the "
                            f"cell through runners/run_all.py.")
                if cfg_absent:
                    # Absent, not silent. `runners/check_convergence.py`
                    # reports the same condition the same way (CFG_ABSENT ->
                    # NOT PERFORMED): run_all.write_resolved_config writes this
                    # file for every cell it launches, so a run without one did
                    # not come from run_all, and the honest thing for a gate to
                    # say is which check it could not run.
                    warns.append(
                        f"{cell.name}: no config.resolved.yaml beside seed(s) "
                        f"{cfg_absent}, so the class-weighting policy THOSE "
                        f"RUNS were trained under is unknown and the decision-2 "
                        f"one-policy check was NOT PERFORMED for them. "
                        f"run_all.write_resolved_config writes that file for "
                        f"every cell it launches, so a run without one did not "
                        f"come from run_all.")
                if cfg_no_loss:
                    errors.append(
                        f"{cell.name}: the resolved config beside seed(s) "
                        f"{cfg_no_loss} has no `loss:` block, so it states no "
                        f"class-weighting policy, and {cell.trainer} cannot "
                        f"have started from it -- that trainer calls "
                        f"src.utils.class_weights.class_weighting_enabled, "
                        f"which raises on a config without "
                        f"loss.use_class_weights. That resolved config is not "
                        f"the one the run used. Re-run the cell through "
                        f"runners/run_all.py.")
                if cfg_unreadable:
                    errors.append(
                        f"{cell.name}: " + unreadable_msg
                        + f" Read from the resolved config of seed(s) "
                          f"{cfg_unreadable}; every seed is read separately, "
                          f"because until 26 Sep 2026 only the lowest one was.")
                stated = set(stated_by_seed.values())
                if len(stated) > 1:
                    errors.append(
                        f"{cell.name}: the seeds of this ONE cell were trained "
                        f"under DIFFERENT class-weighting policies -- "
                        f"{dict(sorted(stated_by_seed.items()))}, read from "
                        f"each seed's own config.resolved.yaml. They are not "
                        f"five samples of one experiment, so they may not be "
                        f"averaged into one row; on this corpus the setting is "
                        f"worth about 18 points. Re-run the odd seeds under "
                        f"the cell's policy.")
                per_cell_policy[cell.name] = stated
                per_tier_weight |= stated

    # What must match is one (max_epochs, patience), the question
    # `run_all.check_selection_parity` asks before launch, read from what the
    # runs recorded. Stopping epochs differ. See `recorded_selection_rule`.
    def _who(by_run: dict, limit: int = 8) -> str:
        items = sorted(by_run.items(), key=lambda kv: (kv[1], kv[0]))
        shown = "; ".join(f"{k} = {v}" for k, v in items[:limit])
        return shown + ("" if len(items) <= limit
                        else f"; (+{len(items) - limit} more)")

    caps = set(cap_by_run.values())
    if len(caps) > 1:
        errors.append(
            f"cells of this tier were held to DIFFERENT epoch caps "
            f"{sorted(caps)}, so they were not selected under one rule and "
            f"their results are not comparable. gives "
            f"every cell of a comparison ONE selection rule -- same "
            f"training.max_epochs, same training.patience -- and "
            f"run_all.check_selection_parity refuses to launch a tier whose "
            f"configs disagree, so a disagreement HERE means a config was "
            f"edited between arms. Recorded: {_who(cap_by_run)}. Re-run the "
            f"odd cells under the tier's rule; do not extend one arm alone, "
            f"which is the unequal-budget confound the selection rule removed.")
    patiences = set(patience_by_run.values())
    if len(patiences) > 1:
        errors.append(
            f"cells of this tier recorded DIFFERENT patience values "
            f"{sorted(patiences)} in their resolved configs. Same rule, same "
            f"reason as the cap above: one selection rule per comparison, and "
            f"patience is half of it -- two arms capped at 60 but stopped on "
            f"patience 5 and patience 20 did not receive the same training. "
            f"Recorded: {_who(patience_by_run)}.")
    if rule_not_performed:
        warns.append(
            f"the patience half of the selection-rule check was NOT PERFORMED "
            f"for {len(rule_not_performed)} run(s): "
            + " | ".join(rule_not_performed[:4])
            + ("" if len(rule_not_performed) <= 4
               else f" | (+{len(rule_not_performed) - 4} more)"))
    if selected_by_run:
        lo, hi = min(selected_by_run.values()), max(selected_by_run.values())
        warns.append(
            f"selected epochs across this tier span {lo}-{hi}. Under decision "
            f"B that is expected, not a defect: every run early-stops on its "
            f"own validation curve. Whether any run was TRUNCATED by the cap "
            f"instead is a different question, owned by the cap audit "
            f"(runners/check_convergence.py), which this gate runs.")

    # Five seeds of one arm share arm_config_hash, and two arms of one tier
    # must not. The committed runs are what can catch a cell launched from the
    # wrong config file.
    for ah, owners in sorted(arm_hash_owners.items()):
        if len(owners) > 1:
            errors.append(
                f"cells {sorted(owners)} of this tier recorded the SAME "
                f"arm_config_hash {ah}. That digest is the resolved config with "
                f"`seed` and `paths` dropped and nothing else "
                f"(src/utils/provenance.ARM_HASH_EXCLUDED_KEYS), so two cells "
                f"that agree on it were launched from the same experiment -- "
                f"including the same `experiment_name`, the same corpus and the "
                f"same split. They are one arm appearing twice in the table, "
                f"and a comparison between them compares nothing. Check which "
                f"config each cell was actually launched from.")

    per_tier_fp.discard(None)
    if len(per_tier_fp) > 1:
        errors.append(f"cells were produced by different code versions: "
                      f"{sorted(per_tier_fp)} -- not comparable")

    # Agreement among the recorded fingerprints is not enough. Each one is
    # compared with this tree.
    if fp_by_run:
        live = live_fp          # hashed once, at the top of this function
        stale = {k: v for k, v in sorted(fp_by_run.items()) if v != live}
        if stale:
            shown = ", ".join(f"{k}={v}" for k, v in list(stale.items())[:6])
            more = "" if len(stale) <= 6 else f" (+{len(stale) - 6} more)"
            errors.append(
                f"{len(stale)} recorded fingerprint(s) do not match the code in "
                f"this tree. Live code_fingerprint() is {live}; recorded "
                f"{shown}{more}. These numbers were produced by code that has "
                f"since changed, so they are not results of the current tree. "
                f"Re-run the affected cells, or -- if you are deliberately "
                f"auditing an archived tier -- run this gate from the tree that "
                f"produced it. Either way, do not quote them as current results.")
    # A labelled weighted tier is still checked. Being on that list is
    # permission to be weighted, not an exemption from the check.
    if tier in run_all.WEIGHTED_TIERS:
        unweighted = sorted(name for name, pols in per_cell_policy.items()
                            if pols and pols != {True})
        if unweighted:
            errors.append(
                f"{tier} is one of the LABELLED WEIGHTED tiers "
                f"{sorted(run_all.WEIGHTED_TIERS)} -- the weighted mirror is "
                f"the entire claim it exists to make -- but the resolved "
                f"configs of {unweighted} state "
                f"{sorted({p for n in unweighted for p in per_cell_policy[n]})}. "
                f"A cell of this tier that was not weighted is not a footnote, "
                f"it is the headline configuration under another name, and "
                f"quoting it as the weighted result would misstate the one "
                f"question this tier answers (the class-weighting rule, Q5).")
        not_weighted_runs = sorted(
            f"{k} = {v!r}" for k, v in weight_by_run.items() if v is not True)
        if not_weighted_runs:
            errors.append(
                f"{tier} is a labelled WEIGHTED tier, but "
                f"{len(not_weighted_runs)} run(s) recorded a provenance "
                f"use_class_weights that is not True: "
                + "; ".join(not_weighted_runs[:8])
                + ("" if len(not_weighted_runs) <= 8
                   else f"; (+{len(not_weighted_runs) - 8} more)")
                + ". The config states the intent; provenance records what the "
                  "run was actually handed, and it is the half no gate read "
                  "before 26 Sep 2026.")
    elif len(per_tier_weight) > 1:
        errors.append(f"cells used different class-weighting policies: "
                      f"{sorted(per_tier_weight)} -- the class-weighting rule allows "
                      f"one policy per table, plus the labelled weighted tiers "
                      f"{sorted(run_all.WEIGHTED_TIERS)}")

    return errors, warns


def cap_audit(tier: str, results: Path, seeds: list[int]):
    """Run `runners/check_convergence.py` over the same tier. (errors, notes).

    `advisor-guidance/spec-check-convergence.md` section 5 requires a call
    (subprocess or import) so a tier gate fails when convergence fails, with
    `check_convergence.py` still runnable standalone.

    WHY IT IS CALLED FROM `main` AND NOT FROM `check`
    -------------------------------------------------
    They are two audits with two reports and one verdict. `check` answers "are
    these numbers sound and comparable?" from the metrics files; the cap audit
    answers "did any run simply run out of budget?" from the stopping records,
    the resolved configs and the validation curves in `train.log`. Folding the
    second into the first would interleave their findings into one undated list
    and make "which tool said this?" unanswerable at the exact moment an
    operator needs to know which remedy applies -- re-run at a raised cap is
    not the same instruction as re-run the odd cell under the tier's rule.
    `main` prints them as two sections and fails on either.

    The audit's own `errors` are surfaced verbatim: they already name the arm,
    the seed and the remedy. Its `notes` are NOT PERFORMED lines and become
    notes here, never passes.
    """
    run_all = _load_run_all()
    arms = [c.name for c in run_all.TIERS[tier]]
    _rows, _verdicts, errors, notes = check_convergence.audit(
        arms, results, seeds)
    return errors, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)

    results = REPO / args.results
    errors, warns = check(args.tier, results, args.seeds)

    print("=" * 74)
    print(f"TIER {args.tier} -- results check")
    print("=" * 74)
    for w in warns:
        print(f"  note   {w}")
    if warns:
        print()
    for e in errors:
        print(f"  FAIL   {e}")
    if errors:
        print(f"\n  {len(errors)} problem(s) in the results check.")
    else:
        # Every claim below is made by a check in `check` above -- if one is
        # removed, its line here goes with it. A banner is a claim, not
        # decoration.
        print("  every check passed: provenance complete, seeds present and "
              "distinct")
        print("  runs (distinct payloads, distinct config_hash, one "
              "arm_config_hash),")
        print("  one test set, ONE SELECTION RULE (same max_epochs, same "
              "patience),")
        print("  one class-weighting policy, config and provenance agreeing on "
              "it,")
        print("  every metric rate in range, and every run's recorded "
              "fingerprint matches")
        print(f"  this tree ({code_fingerprint()}).")

    # The cap audit is skipped only when there is no tier to audit -- `check`
    # has already reported the unknown name, and inventing an arm list here
    # would turn a typo into a second, different error.
    cap_errors, cap_notes = [], []
    if args.tier in _load_run_all().TIERS:
        cap_errors, cap_notes = cap_audit(args.tier, results, args.seeds)
        print()
        print("=" * 74)
        print(f"TIER {args.tier} -- cap audit (runners/check_convergence.py)")
        print("=" * 74)
        for n in cap_notes:
            print(f"  note   {n}")
        if cap_notes:
            print()
        for e in cap_errors:
            print(f"  FAIL   {e}")
        if cap_errors:
            print(f"\n  {len(cap_errors)} problem(s) in the cap audit.")
        else:
            print("  every audited seed stopped early, inside its cap, under "
                  "one selection")
            print("  rule. No arm was truncated by its budget.")

    total = len(errors) + len(cap_errors)
    print()
    if total:
        print(f"{total} problem(s) across the two audits. This tier's numbers "
              f"are NOT sound.")
        print("Do not start the next tier and do not quote anything from this one.")
        return 1
    if warns or cap_notes:
        print("Note the NOT PERFORMED / not-compared lines above: they are "
              "checks that\ndid not run, not checks that passed.")
    print(f"\nTIER {args.tier} IS SOUND.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

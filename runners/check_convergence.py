#!/usr/bin/env python3
"""
check_convergence.py -- the cap audit: did any arm simply run out of budget?

Under a run ends in exactly one of two ways, and the
trainer records which one in ``provenance["stopping_reason"]``:

    early_stop   validation loss stopped improving and patience ran out. The
                 run finished on its own terms, and the checkpoint it selected
                 is the best one it was ever going to find.
    cap          the loop reached ``training.max_epochs`` with patience still
                 intact. The run was TRUNCATED: it had not stopped improving,
                 it ran out of budget.

Those two are not interchangeable. An arm whose seeds early-stopped at epoch 22
and an arm whose seeds were still improving when the cap cut them off at 60 did
not receive the same training, so the difference between their accuracies is
partly a difference in budget rather than a difference in the thing under test.

The advisors' batch-run protocol therefore ends with a cap audit: "any cap-bound
arm re-runs its comparison at a raised cap". This script is that audit.
`runners/compare_arms.py` prints the stopping reasons it happens to see and then
says the verdict is owned elsewhere -- this is the script it means.

WHAT IT CHECKS, AND WHY EACH ONE EARNED ITS PLACE
=================================================

Per run
  metrics file readable      a metrics file that is not JSON is a corrupt
                             artifact. The audit RAISES on it rather than
                             scoring around it, and the message names the arm
                             and seed, because a bare JSONDecodeError in a
                             20-run tier says nothing about which run to look at
  provenance block present   `metrics["provenance"]` is where every fact below
                             lives. An evaluator that wrote none produced an
                             untraceable number
  regiment fields present    `epoch`, `selected_epoch` and `stopping_reason` are
                             all REQUIRED by
                             provenance.check_regiment_fields. `stamp_selection`
                             writes the last two after the loop, so their
                             absence means the loop did not finish; `epoch` is
                             written with the checkpoint itself, so a run that
                             cannot say what epoch it is is not quotable either.
                             None of the three may be treated as optional: an
                             absent field must not let a check be skipped and
                             the run still pass
  stopping_reason legal      exactly "early_stop" or "cap"
                             (provenance.STOPPING_REASONS). Free text would pass
                             the regiment's field check and then silently defeat
                             this audit, which matches on the literal "cap"
  not cap-bound              THE POINT OF THE SCRIPT. One cap-bound seed makes
                             its arm cap-bound, because the arm's mean is taken
                             over that truncated seed
  epoch == selected_epoch    every trainer writes provenance inside the
                             save-on-improvement branch, so the checkpoint's own
                             epoch IS the selected epoch. A disagreement means
                             something saved outside that branch and the file
                             being scored is not the one selected
  the recorded cap is the    `epoch_budget_configured` is copied from
  configured one             `training.max_epochs` (provenance.py:227), and
                             `config.resolved.yaml` is that same config saved
                             beside the run. If they disagree, one of the two
                             artifacts is lying about the budget the run was
                             held to
  reason agrees with the     a run stops early only after `patience` consecutive
  arithmetic                 non-improving epochs, so early_stop implies
                             selected_epoch + patience <= cap, and cap implies
                             selected_epoch + patience > cap. A run that
                             violates its own arithmetic is mislabelled, and a
                             mislabelled "early_stop" is a cap-bound run hiding
                             from this audit. This is the sharpest check in the
                             file, so the patience it needs is NOT optional:
                             a `config.resolved.yaml` that exists but cannot
                             supply `training.patience` is a corrupt artifact
                             and fails the run (see `_resolved_config`)
  val-curve sanity           the epoch the run selected must be the epoch at
                             which its own validation loss bottoms out. Read
                             from the per-epoch lines of `train.log`, which is
                             an artifact of the run rather than a re-derivation
                             from the weights. See `val_curve_report`

Per arm, across its seeds
  all seeds present          a missing seed is an unaudited seed; an arm cannot
                             be certified on the seeds that happened to finish

Per tier, across its arms
  one cap                    `run_all.check_selection_parity` enforces this from
                             the CONFIGS before launch. This checks the same
                             thing from what the runs actually recorded, which
                             is the only version that can catch a config edited
                             between arms

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
=========================================
Convergence is decided during the run, by validation loss and patience, and
written down. Re-deriving a PLATEAUED/STILL_LEARNING opinion from a train-loss
window would give this audit a second verdict that can disagree with the
trainer's, so that machinery is not here.

The curve work that remains is each run's selected epoch and val-curve sanity.
That part is implemented in `val_curve_report`: the curve is not asked whether
it looks converged, it is asked whether the epoch the run SELECTED is the epoch
its validation loss actually bottoms out at. Where the curve cannot answer (no
`train.log`, a trainer that logs only every 10th epoch, a curve that stops
short of the selected epoch) the run is listed by name under "not performed";
it is never counted as having passed.

IT AUDITS THE CAP. IT DOES NOT AUDIT THE TIER
=============================================
A pass here says one thing: no audited run was truncated by its epoch budget.
It says nothing about the code fingerprint the runs recorded, nothing about
whether the metrics exist or are in range, nothing about seed distinctness or
class-weighting policy. Those belong to `runners/check_tier.py` and
`runners/aggregate_seeds.py`, and widening this script to answer them would
give the project a SECOND verdict on each that can disagree with the first --
the same objection that keeps a train-loss plateau verdict out of this file.

So the scope stays narrow. Every run, passing or failing, prints the
`NOT_CHECKED` block naming what was not looked at and which tool owns it, and
the markdown report carries the same list. `check_convergence` is the SECOND
gate a tier record runs, and an operator who runs it alone sees a green
"IS NOT BUDGET-CONFOUNDED" line with nothing beside it. A gate whose scope has
to be inferred from its source will be over-read.

IT MUST NOT DAMAGE WHAT IT AUDITS
=================================
This script reads a results tree to decide something about it. It does not
rewrite an existing report.

The writer REFUSES to replace an existing report. The decision is made
before a single run is read, so a refusal costs nothing and changes nothing:

    (no flag)     write the report at <results>/convergence_<tier>.md only if
                  that would CREATE it. If it exists: exit 3, nothing read,
                  nothing written. Exit 3 and not 2 -- see Exit codes below:
                  declining to write a report is not a usage error and is not
                  a verdict about the tier, and a caller has to be able to
                  tell all three apart
    --report P    the same rule at P -- `--report` is unchanged except that it,
                  too, will not clobber
    --no-report   READ-ONLY. Audit, print, write nothing anywhere. This is the
                  mode every verification agent on this project needed
    --force       replace the report on purpose, having said so

`--report` and `--no-report` are mutually exclusive.

Usage
-----
    python runners/check_convergence.py --tier T1 --no-report
    python runners/check_convergence.py --tier T1 --seeds 0 1 2 --results results
    python runners/check_convergence.py --tier T1 --report /tmp/T1.regen.md
    python runners/check_convergence.py --experiment cadsynth_bias_both

Exit codes
----------
THE AUDIT RAN, and the code is its verdict on the tier:

  0   every audited arm converged inside its budget, under one selection rule
  1   the tier FAILED the audit -- an arm is cap-bound, incomplete or
      mislabelled, or the arms did not all run under one selection rule

THE AUDIT DID NOT RUN, and the code says nothing about the tier:

  2   the INVOCATION is wrong: unknown tier, no results directory. Nothing
      about the tier is known and a human has to fix the command
  3   the audit DID NOT run because the report target already EXISTS and
      --force was not given. Nothing was read and nothing was written. This is
      NOT a cap-audit failure: the tier is sound or not and this invocation
      never looked. Re-ask with --no-report for the verdict, or --force to
      replace the report on purpose

2 and 3 are deliberately different numbers, because a caller that gates on
this tool has to react to them differently: 2 stops and asks a human, 3
re-asks read-only.

A corrupt artifact (unparseable metrics JSON) RAISES instead of exiting: it is
a bug in whatever wrote the file, not a verdict about the run. The traceback
is the report, and its exit status is not one of the four above.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.provenance import STOPPING_REASONS  # noqa: E402

#: Patience is 10 and the cap is 60. Every config sets these.
#: Changing them means re-running the affected comparison.
REGISTERED_MAX_EPOCHS = 60
REGISTERED_PATIENCE = 10

#: 0 and 1 are verdicts about a tier that was read.
#: 2 and 3 mean the audit did not run.
EXIT_OK = 0
EXIT_TIER_FAILED = 1
EXIT_USAGE = 2
#: Report already exists and --force was not given.
EXIT_REPORT_EXISTS = 3

#: Per-run verdicts. Only CONVERGED is a pass.
CONVERGED = "CONVERGED"
CAP_BOUND = "CAP-BOUND"
MISSING = "MISSING"
UNSTAMPED = "UNSTAMPED"
INVALID = "INVALID"
#: A run whose label says early_stop but whose own numbers contradict it. It is
#: NOT reported as CONVERGED: the whole point of the audit is that a truncated
#: run must not pass as a finished one just because of what it calls itself.
INCONSISTENT = "INCONSISTENT"
#: The run may be intact, but one of the audit's own inputs is corrupt (a
#: resolved config that cannot supply patience, a provenance with no cap), so
#: the checks that matter could not be run. Distinct from INCONSISTENT, which
#: means the run's numbers contradict each other, and never a pass: an audit
#: that could not audit must not certify.
UNAUDITABLE = "UNAUDITABLE"

#: Resolved-config states, kept apart so the report can say which one it saw.
CFG_OK = "ok"
CFG_ABSENT = "absent"
CFG_MALFORMED = "malformed"

#: Which remedies the failures actually call for. Computed ONCE by
#: `remedy_kinds` and used by both the stdout summary and the markdown report,
#: so the two prescribe the same fixes for the same invocation. It is a LIST,
#: not a choice: a tier can be cap-bound AND have a
#: seed that never ran, and a reader told only about one of them will fix only
#: one of them.
REMEDY_CAP = "cap"              # a run recorded, or provably hid, a cap hit
REMEDY_UNKNOWN = "unknown"      # a run could not be ASKED about the cap
REMEDY_SELECTION = "selection"  # a run contradicts its own selection record
REMEDY_MISSING = "missing"      # a run produced nothing

#: train.log prints validation loss to 4 decimals, so two epochs whose true
#: losses differ by less than half a unit in the last place can print the same
#: number. Only a gap wider than that is evidence of a real disagreement.
LOG_LOSS_TOL = 1.5e-4

#: (question, owner). Printed on a pass as well as a failure.
NOT_CHECKED = (
    ("the code_fingerprint each run recorded, against the tree running now",
     "runners/check_tier.py, runners/aggregate_seeds.py, runners/compare_arms.py"),
    ("whether any metric is present or in range -- overall_accuracy, macro_f1, "
     "mean_iou, the instance/bottom blocks",
     "runners/check_tier.py, runners/aggregate_seeds.py"),
    ("whether the seeds are distinct runs rather than copies of one another",
     "runners/check_tier.py (see weekly/GATE-DEFECTS.md G6)"),
    ("whether a weighted tier was actually weighted -- provenance."
     "use_class_weights against the resolved config",
     "runners/check_tier.py (see weekly/GATE-DEFECTS.md G4)"),
    ("whether the seeds of one arm resolved to distinct config_hash values",
     "runners/aggregate_seeds.py (see weekly/GATE-DEFECTS.md G7)"),
    ("whether the tier's summary table was built from these runs",
     "runners/aggregate_seeds.py"),
)

#: The sentence that frames the list. Kept beside it so stdout and the report
#: cannot drift into framing the same list differently.
NOT_CHECKED_HEADING = (
    "NOT CHECKED BY THIS AUDIT -- the verdict above is about the EPOCH BUDGET "
    "and nothing else.")
NOT_CHECKED_WHY = (
    "This is the second gate a tier record runs, so reading its green line as "
    "\"the tier is sound\" is a mistake two paper-facing documents have "
    "already made in print (weekly/WEEK6-SWEEP-FINDINGS.md:245). "
    "`runners/check_tier.py --tier <T>` is the full gate, and it calls this "
    "audit (check_tier.cap_audit) as one of its checks -- so running check_tier "
    "gets you this verdict too, and running only this one does not get you "
    "check_tier's.")


def not_checked_lines(indent="  "):
    """The disclosure as plain lines, for stdout."""
    out = [indent + NOT_CHECKED_HEADING, indent + NOT_CHECKED_WHY, ""]
    for question, owner in NOT_CHECKED:
        out.append("{}  - {}".format(indent, question))
        out.append("{}      owned by: {}".format(indent, owner))
    return out


def not_checked_markdown():
    """The same disclosure as a markdown section, for the kept report."""
    lines = ["", "## Not checked by this audit", "",
             NOT_CHECKED_HEADING, "", NOT_CHECKED_WHY, ""]
    for question, owner in NOT_CHECKED:
        lines.append("- {} — owned by `{}`".format(question, owner))
    return lines


class ConvergenceVerdict(NamedTuple):
    """What one audit concluded, in the shape `check_tier.py` can consume.

    `runners/check_tier.py::check(tier, results, seeds)` returns
    ``(errors, warnings)`` and its `main()` renders them. This audit is the
    sibling gate spec section 5 requires check_tier to CALL, so `check()` here
    returns the same two lists first, plus the rows behind them:

        errors       gating failures. A caller that extends its own error list
                     with this one gets the right exit code and nothing else to
                     decide. EMPTY on a scouting tier (see `gates`), because a
                     scouting verdict must not fail a P2 gate.
        warnings     things the audit did NOT check and the reason, plus -- on
                     a scouting tier -- every demoted failure, so a caller that
                     reads only these two lists still SEES them.
        rows         one dict per (arm, seed): see `audit_run`.
        arm_verdicts {arm: one-line verdict}.
        notes        the NOT-PERFORMED lines alone, for the report section.
        failures     every failure the audit found, gating or not. On a
                     gating tier this is `errors`; on a scouting tier `errors`
                     is empty and this is not.
        gates        False for a tier in run_all.SCOUTING_TIERS.
        usage_error  the errors are about the INVOCATION (unknown tier,
                     missing results directory), not about any run. The CLI
                     turns this into exit 2 rather than exit 1.
        root         the directory actually audited -- `results/scouting` for a
                     scouting tier, which is where run_all writes it.
    """

    errors: list
    warnings: list
    rows: list
    arm_verdicts: dict
    notes: list
    failures: list
    gates: bool
    usage_error: bool
    root: object


_RUN_ALL = None


def _load_run_all():
    """Load `runners/run_all.py` for its tier table and its metrics filenames.

    Imported by path rather than duplicated: the tier table is the authoritative
    list of arms whose numbers may enter the paper, and a second copy here would
    quietly stop auditing an arm added to the real one. The same argument
    applies to the metrics filenames, which is why `_metrics_file` calls
    run_all's own `metrics_path` (runners/run_all.py:382-388) instead
    of keeping a list beside it.

    Cached, because `audit_run` needs it too and re-exec'ing the module for
    every one of 20 runs is wasteful.
    """
    global _RUN_ALL
    if _RUN_ALL is None:
        spec = importlib.util.spec_from_file_location(
            "run_all", REPO / "runners" / "run_all.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "metrics_path"):
            raise AttributeError(
                "runners/run_all.py no longer defines metrics_path(), which is "
                "where this audit gets the list of filenames an evaluator may "
                "write. Point _metrics_file at whatever replaced it -- do NOT "
                "re-add a private copy of the list here, because the copy that "
                "used to live here is how link_eval_metrics.json went "
                "unaudited.")
        _RUN_ALL = mod
    return _RUN_ALL


def _metrics_file(run_dir: Path):
    """The metrics file in *run_dir*, per run_all's own list of filenames."""
    return _load_run_all().metrics_path(run_dir)


def _resolved_config(run_dir: Path):
    """Read `training.patience` / `training.max_epochs` from the run's config.

    `run_all.write_resolved_config` writes the exact config of the run next to
    its results, so this is the selection rule that run actually used rather
    than whatever the config file happens to say today.

    Returns ``(patience, max_epochs, status, problem)``:

    * ``status == CFG_ABSENT``  -- no config.resolved.yaml. patience is None and
      the arithmetic check is reported, by run, as NOT PERFORMED.
    * ``status == CFG_MALFORMED`` -- the file IS there but cannot supply the
      selection rule. `problem` is a sentence saying so. This is FATAL to the
      run's verdict, deliberately: `class_weights.required_training_budget`
      (src/utils/class_weights.py:151-157) refuses to start a run whose config
      omits `patience` or `max_epochs`, so a resolved config that omits them
      cannot be the config any run was launched from -- it is corrupt. Letting
      it return None here is exactly how a tier of 20 cap-bound seeds passed
      this audit with the patience arithmetic silently skipped.
    * ``status == CFG_OK`` -- both values present and positive ints.
    """
    cfg_path = run_dir / "config.resolved.yaml"
    if not cfg_path.is_file():
        return None, None, CFG_ABSENT, None

    text = cfg_path.read_text(encoding="utf-8")
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return None, None, CFG_MALFORMED, (
            "{} is not parseable YAML ({}). run_all.write_resolved_config "
            "wrote it, so a file it cannot parse is a damaged artifact, not a "
            "configuration. Re-run this seed."
            .format(cfg_path, str(exc).replace("\n", " ")[:160]))
    if not isinstance(cfg, dict):
        return None, None, CFG_MALFORMED, (
            "{} holds {} instead of a config mapping (the file is {} bytes). "
            "An empty or truncated resolved config cannot say what selection "
            "rule the run used. Re-run this seed."
            .format(cfg_path, type(cfg).__name__, len(text)))

    training = cfg.get("training")
    if not isinstance(training, dict):
        return None, None, CFG_MALFORMED, (
            "{} has no `training:` block, so it cannot say what selection rule "
            "the run used. Re-run this seed.".format(cfg_path))

    missing = [k for k in ("patience", "max_epochs") if k not in training]
    if missing:
        return None, None, CFG_MALFORMED, (
            "{} sets no training.{}. The selection rule fixes ONE selection rule for "
            "every cell, and src/utils/class_weights.py:151-157 REFUSES to "
            "start a run whose config omits it -- so no run can have been "
            "launched from this file. It is a corrupt artifact, and without "
            "patience the audit cannot check that this run's stopping_reason "
            "agrees with its own arithmetic, which is the only check that "
            "catches a cap-bound run mislabelled early_stop. Restore the "
            "resolved config or re-run this seed."
            .format(cfg_path, " and training.".join(missing)))

    out = {}
    for key in ("patience", "max_epochs"):
        value = training[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return None, None, CFG_MALFORMED, (
                "{}: training.{} is {!r} ({}), not a positive integer. The "
                "audit's patience arithmetic cannot run on it. Re-run this "
                "seed.".format(cfg_path, key, value, type(value).__name__))
        out[key] = value
    return out["patience"], out["max_epochs"], CFG_OK, None


# val-curve sanity
#: Reasons the validation curve could not answer. Each one names a condition
#: of the ARTIFACTS, never a judgement about the run: a run whose curve could
#: not be read is listed under "not performed" with every other run that shares
#: its reason, and is never counted as having passed the check.
NP_NO_LOG = "no train.log in the run directory"
NP_NO_VAL_LINE = ("train.log carries no per-epoch validation-loss line "
                  "(\"Epoch n/N | ... | Val Loss x\")")
NP_SPARSE = ("train.log records only some of the run's epochs, so the lowest "
             "LOGGED validation loss need not be the lowest one reached -- "
             "src/training/train_resnet.py:401 prints an epoch line only every "
             "10th epoch, which is the usual cause")
NP_SHORT = "the logged curve stops before the selected epoch"
NP_NO_SELECTED = "the run recorded no selected_epoch to check the curve against"

#: "Epoch  17/60" -- the first field of every trainer's per-epoch line.
_EPOCH_FIELD = re.compile(r"^Epoch\s+(\d+)\s*/\s*\d+$")
#: "Val Loss 0.3184" (train_semantic.py:529, train_resnet.py:399) or
#: "Val 0.3184" (train_unified.py:519). "Val Acc 88.00%" does not match: the
#: value must be a bare number, inf or nan.
_VAL_FIELD = re.compile(
    r"^Val(?:\s+Loss)?\s+(-?\d*\.?\d+(?:[eE][+-]?\d+)?|-?inf|nan)$")


def _assumed_patience(run_dir: Path) -> str:
    """The sentence appended to any verdict reached on a patience this run did
    not itself record. An assumption that decides a verdict has to be visible
    in the line that states the verdict, not only in a note further down."""
    return (" Patience {} is the pre-registered value, "
            "not this run's own: there is no "
            "config.resolved.yaml in {} to state it. If this run was really "
            "held to some other patience then it was not run under the "
            "registered selection rule, which fails this check too. Restore "
            "the resolved config or re-run the seed through "
            "runners/run_all.py.".format(REGISTERED_PATIENCE, run_dir))


def parse_val_curve(log_text: str) -> dict:
    """``{epoch: val_loss}`` from the per-epoch lines of a train.log.

    Deliberately reads the log rather than the checkpoint: `train.log` sits in
    the run directory next to the metrics file the rest of this audit reads,
    while the `.pth` is usually not synced down from the studio, and loading it
    would pull torch into a gate that otherwise needs only json and yaml.
    """
    curve = {}
    for line in log_text.splitlines():
        if not line.startswith("Epoch"):
            continue
        fields = [f.strip() for f in line.split("|")]
        head = _EPOCH_FIELD.match(fields[0])
        if head is None:
            continue
        for field in fields[1:]:
            val = _VAL_FIELD.match(field)
            if val is not None:
                curve[int(head.group(1))] = float(val.group(1))
                break
    return curve


def val_curve_report(arm: str, seed: int, run_dir: Path, selected):
    """Val-curve sanity for one run: ``(summary, errors, not_performed)``.

    The question asked of the curve is the only one the selection rule leaves open:
    the run SELECTED an epoch and the checkpoint scored is that epoch's, so the
    validation loss recorded at that epoch must be the lowest the run ever saw.
    It is not asked whether the curve "looks converged" -- patience already
    decided that during the run, and a second opinion here could only disagree
    with the trainer's.

    `not_performed` is one of the NP_* reasons whenever the curve cannot
    answer. `audit` groups the runs by reason and names every one of them, so a
    run the curve could not check is listed rather than quietly counted as a
    pass.
    """
    log = run_dir / "train.log"
    if not log.is_file():
        return "not performed (no train.log)", [], NP_NO_LOG

    curve = parse_val_curve(log.read_text(encoding="utf-8", errors="replace"))
    if not curve:
        return "not performed (no val line in train.log)", [], NP_NO_VAL_LINE

    epochs = sorted(curve)
    span = "{}-{}".format(epochs[0], epochs[-1])
    nan_at = [e for e in epochs if math.isnan(curve[e])]
    if nan_at:
        return ("nan @{}".format(nan_at[0]), [
            "{} seed {}: the validation loss in {} is nan at epoch(s) {}. A run "
            "whose validation loss went nan did not select a checkpoint on any "
            "meaningful criterion, whatever its stopping_reason says."
            .format(arm, seed, log, nan_at[:5])], None)

    dense = epochs[-1] - epochs[0] + 1 == len(epochs)
    if not dense:
        return ("only {} lines over epochs {} -- not performed"
                .format(len(epochs), span), [], NP_SPARSE)

    if selected is None:
        return ("epochs {} ({}) -- not performed, no selected_epoch"
                .format(span, len(epochs)), [], NP_NO_SELECTED)

    if selected not in curve:
        return ("epochs {} ({}) -- not performed, stops short of epoch {}"
                .format(span, len(epochs), selected), [], NP_SHORT)

    best_loss = min(curve.values())
    best_epoch = min(e for e in epochs if curve[e] == best_loss)
    at_selected = curve[selected]
    summary = "epochs {} ({}), min {:.4f} @{}, selected {:.4f} @{}".format(
        span, len(epochs), best_loss, best_epoch, at_selected, selected)

    if at_selected - best_loss > LOG_LOSS_TOL:
        return summary, [
            "{} seed {}: val-curve sanity FAILED. {} records validation loss "
            "{:.4f} at epoch {}, lower than the {:.4f} at the selected epoch "
            "{}. The selection rule scores the BEST-validation-loss checkpoint, so the "
            "weights being quoted are not the ones the curve says were best. "
            "Either the save-on-improvement branch did not fire at epoch {}, "
            "or selected_epoch was stamped from the wrong variable. This is a "
            "checkpoint-selection fault, not a budget one: raising the cap "
            "would not change it."
            .format(arm, seed, log, best_loss, best_epoch, at_selected,
                    selected, best_epoch)], None

    if best_epoch != selected:
        summary += " (tied within the log's 4-decimal precision)"
    return summary, [], None


# per-run audit

def audit_run(arm: str, seed: int, run_dir: Path) -> dict:
    """Audit one `results/<arm>/seed<k>/`.

    Returns a row dict. `row["errors"]` is empty exactly when the run converged
    and is internally consistent. `row["budget_bound"]` is True only when the
    cap is what went wrong, which is what decides whether "re-run at a raised
    cap" is the right advice; an INCONSISTENT run is not automatically one.
    """
    row = {"arm": arm, "seed": seed, "selected_epoch": None, "cap": None,
           "patience": None, "reason": None, "verdict": MISSING, "errors": [],
           "epoch": None, "budget_bound": False, "contradicted": False,
           "unauditable": False, "config_status": None,
           "patience_source": None,
           "val_curve": "-", "val_curve_skipped": None}

    mp = _metrics_file(run_dir)
    if mp is None:
        row["errors"].append(
            "{} seed {}: no metrics file in {}. An arm cannot be certified "
            "converged on the seeds that happened to finish -- run this seed, "
            "or drop it from --seeds and say so in the paper."
            .format(arm, seed, run_dir))
        return row

    raw = mp.read_text(encoding="utf-8")
    try:
        metrics = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Raising is right -- a metrics file that is not JSON is a corrupt
        # artifact, not a run with a verdict. But a bare JSONDecodeError in a
        # 20-run tier does not say WHICH run to open, so it is re-raised with
        # the run named. (Re-raise, not swallow: the traceback still ends here.)
        raise ValueError(
            "{} seed {}: {} is not valid JSON ({}). The evaluator that wrote it "
            "was interrupted or the file was truncated in transfer. Delete it "
            "and re-run this seed; do not hand-edit it."
            .format(arm, seed, mp, exc)) from exc

    if not isinstance(metrics, dict) or "provenance" not in metrics:
        row["verdict"] = INVALID
        row["errors"].append(
            "{} seed {}: {} carries no `provenance` block. Every fact this "
            "audit reads -- selected_epoch, stopping_reason, the configured "
            "cap -- lives there, so this file cannot be audited and the number "
            "in it is not traceable to a run. Re-run this seed."
            .format(arm, seed, mp))
        return row

    prov = metrics["provenance"]
    if not isinstance(prov, dict):
        row["verdict"] = INVALID
        row["errors"].append(
            "{} seed {}: {} has a `provenance` that is {}, not a mapping. "
            "Re-run this seed.".format(arm, seed, mp, type(prov).__name__))
        return row

    reason = prov.get("stopping_reason")
    selected = prov.get("selected_epoch")
    cap = prov.get("epoch_budget_configured")
    epoch = prov.get("epoch")
    patience, cfg_max_epochs, cfg_status, cfg_problem = _resolved_config(run_dir)

    # A run launched by run_all always has `config.resolved.yaml` beside it, so
    # an ABSENT one means the artifact set is incomplete. It must NOT mean the
    # patience arithmetic below is skipped.
    #
    # When the run does not state its own patience, the PRE-REGISTERED
    # patience is used and every message says which one it used. If a run was
    # really held to some other patience then it was not run under the
    # registered rule, which this audit fails anyway (see `audit`) -- so there
    # is no reading of the artifacts under which silence was the right answer.
    #
    # A MALFORMED config gets no such fallback. It is fatal above, and scoring
    # around a corrupt artifact is the thing this audit exists to refuse.
    patience_used, patience_source = patience, "run"
    if patience is None and cfg_status == CFG_ABSENT:
        patience_used, patience_source = REGISTERED_PATIENCE, "pre-registered"

    row.update(selected_epoch=selected, cap=cap, patience=patience, epoch=epoch,
               config_status=cfg_status, patience_source=patience_source,
               reason=reason if reason is not None else "-")

    # Read the curve before any early return: on an interrupted run it is the
    # one artifact that can still say how far the loop actually got.
    row["val_curve"], curve_errors, curve_skipped = val_curve_report(
        arm, seed, run_dir, selected)
    row["val_curve_skipped"] = curve_skipped

    # `epoch`, `selected_epoch` and `stopping_reason` are ALL required by
    # provenance.check_regiment_fields (src/utils/provenance.py:307-308). None
    # of them is optional here either: a check that quietly does not run is how
    # a truncated run passes an audit built to catch it.
    absent = [name for name, value in (("epoch", epoch),
                                       ("selected_epoch", selected),
                                       ("stopping_reason", reason))
              if value is None]
    if absent:
        row["verdict"] = UNSTAMPED
        row["errors"].append(
            "{} seed {}: provenance is missing the required regiment field(s) "
            "{} (epoch={!r}, selected_epoch={!r}, stopping_reason={!r}). "
            "All three are required by check_regiment_fields "
            "(src/utils/provenance.py:307-308): `epoch` is written with the "
            "checkpoint itself and `stamp_selection` adds the other two AFTER "
            "the training loop, so a missing one means this artifact does not "
            "describe a finished run -- the loop never finished: either it was "
            "killed, or it returned early: src/training/train_semantic.py "
            "returns at its `if start_epoch > max_epochs:` branch (line 429 "
            "when this was written) as soon as a resumed checkpoint is "
            "already at the cap, BEFORE the `stamp_selection` call (line "
            "588), so a run pinned at the cap comes back looking exactly "
            "like this. This run is therefore NOT "
            "evidence that the tier is free of cap-bound arms, and it is not "
            "quotable. Re-run it."
            .format(arm, seed, absent, epoch, selected, reason))
        row["errors"].extend(curve_errors)
        return row

    if reason not in STOPPING_REASONS:
        row["verdict"] = INVALID
        row["errors"].append(
            "{} seed {}: stopping_reason={!r} is not one of {}. This audit "
            "matches on the literal string 'cap', so a third value is not a "
            "label -- it is a hole in the audit. Fix the trainer that wrote it."
            .format(arm, seed, reason, STOPPING_REASONS))
        row["errors"].extend(curve_errors)
        return row

    row["verdict"] = CONVERGED if reason == "early_stop" else CAP_BOUND

    if reason == "cap":
        row["budget_bound"] = True
        row["errors"].append(
            "{} seed {}: TRUNCATED -- the loop reached the cap ({} epochs) "
            "with patience intact, selecting epoch {}. It had not stopped "
            "improving. Every arm of this comparison must re-run at a raised "
            "cap; extending this arm alone would reproduce the unequal-budget "
            "confound the selection rule removed."
            .format(arm, seed, cap, selected))

    if cfg_status == CFG_MALFORMED:
        row["unauditable"] = True
        row["errors"].append("{} seed {}: {}".format(arm, seed, cfg_problem))

    if epoch != selected:
        row["contradicted"] = True
        row["errors"].append(
            "{} seed {}: provenance epoch={} but selected_epoch={}. Every "
            "trainer writes provenance inside the save-on-improvement branch, "
            "so these are the same number on a sound run. A disagreement means "
            "a checkpoint was saved outside that branch and the weights being "
            "scored are not the selected ones -- the last-epoch bug the selection rule "
            "removed.".format(arm, seed, epoch, selected))

    if curve_errors:
        row["contradicted"] = True
        row["errors"].extend(curve_errors)

    if cap is None:
        row["unauditable"] = True
        row["errors"].append(
            "{} seed {}: provenance records no epoch_budget_configured, so the "
            "cap this run was held to is unknown and 'did it hit the cap?' "
            "cannot be answered from the artifact.".format(arm, seed))
    else:
        if cfg_max_epochs is not None and cfg_max_epochs != cap:
            row["contradicted"] = True
            row["errors"].append(
                "{} seed {}: provenance says the cap was {} but the resolved "
                "config beside the run sets training.max_epochs={}. "
                "epoch_budget_configured is copied from that very key "
                "(src/utils/provenance.py:227), so one of the two artifacts "
                "does not belong to this run and the budget it was held to is "
                "not established.".format(arm, seed, cap, cfg_max_epochs))
        if selected > cap:
            row["contradicted"] = True
            row["errors"].append(
                "{} seed {}: selected_epoch {} exceeds the cap {}. The loop "
                "cannot select an epoch it never ran."
                .format(arm, seed, selected, cap))
        elif patience_used is not None:
            # Stopping early requires `patience` non-improving epochs AFTER the
            # selected one, all of them inside the cap. Hitting the cap means
            # that never happened. The same inequality decides both, in
            # opposite senses.
            could_have_stopped_early = selected + patience_used <= cap
            source = ("" if patience_source == "run" else _assumed_patience(run_dir))
            if reason == "early_stop" and not could_have_stopped_early:
                # The run really is cap-bound; only its label says otherwise.
                # So the cap remedy DOES apply to it.
                row["contradicted"] = True
                row["budget_bound"] = True
                row["errors"].append(
                    "{} seed {}: labelled early_stop, but selecting epoch {} "
                    "under patience {} would need the loop to reach epoch {}, "
                    "and the cap is {}. Patience never ran out, so this run is "
                    "cap-bound and mislabelled -- a mislabelled early_stop is "
                    "a truncated run hiding from this audit.{}"
                    .format(arm, seed, selected, patience_used,
                            selected + patience_used, cap, source))
            if reason == "cap" and could_have_stopped_early:
                row["contradicted"] = True
                row["errors"].append(
                    "{} seed {}: labelled cap, but epoch {} plus patience {} "
                    "is {}, within the cap {}. Patience should have fired there "
                    "and ended the run, so the label and the loop disagree.{}"
                    .format(arm, seed, selected, patience_used,
                            selected + patience_used, cap, source))

    # A run that failed a check is not a converged run, whatever its
    # stopping_reason says. Leaving it as CONVERGED in the table would put a
    # reassuring word next to a FAIL line about the same seed. The two failure
    # words are kept apart: INCONSISTENT means the run's own numbers contradict
    # each other, UNAUDITABLE means the audit's inputs were too damaged to ask.
    if row["verdict"] == CONVERGED:
        if row["contradicted"]:
            row["verdict"] = INCONSISTENT
        elif row["unauditable"]:
            row["verdict"] = UNAUDITABLE

    return row


def audit(arms, root: Path, seeds):
    """Audit every (arm, seed). Returns (rows, arm_verdicts, errors, notes)."""
    rows, errors, notes = [], [], []
    arm_verdicts = {}
    caps, patiences = set(), set()

    for arm in arms:
        arm_rows = [audit_run(arm, s, root / arm / "seed{}".format(s))
                    for s in seeds]
        rows.extend(arm_rows)
        for r in arm_rows:
            errors.extend(r["errors"])
            if r["cap"] is not None:
                caps.add(r["cap"])
            if r["patience"] is not None:
                patiences.add(r["patience"])

        verdicts = [r["verdict"] for r in arm_rows]
        n_cap = verdicts.count(CAP_BOUND)
        n_ok = verdicts.count(CONVERGED)
        unscorable = [r["seed"] for r in arm_rows
                      if r["verdict"] in (MISSING, UNSTAMPED, INVALID)]
        contradictory = [r["seed"] for r in arm_rows
                         if r["verdict"] == INCONSISTENT]
        unauditable = [r["seed"] for r in arm_rows
                       if r["verdict"] == UNAUDITABLE]

        detail = []
        if n_cap:
            detail.append("{}/{} seed(s) hit the cap".format(n_cap, len(seeds)))
        if contradictory:
            detail.append("seeds {} contradict their own stopping record"
                          .format(contradictory))
        if unauditable:
            detail.append("seeds {} could not be audited (a corrupt resolved "
                          "config or an unrecorded cap)".format(unauditable))
        if unscorable:
            detail.append("seeds {} produced nothing this audit can score"
                          .format(unscorable))

        if n_cap:
            arm_verdicts[arm] = (
                "CAP-BOUND -- {}; this arm's comparison must re-run at a "
                "raised cap".format("; ".join(detail)))
        elif contradictory:
            arm_verdicts[arm] = "NOT SOUND -- " + "; ".join(detail)
        elif unauditable:
            arm_verdicts[arm] = "NOT AUDITABLE -- " + "; ".join(detail)
        elif unscorable:
            arm_verdicts[arm] = "INCOMPLETE -- " + "; ".join(detail)
        else:
            arm_verdicts[arm] = ("CONVERGED -- {}/{} early-stopped"
                                 .format(n_ok, len(seeds)))

    if len(caps) > 1:
        errors.append(
            "the audited arms recorded DIFFERENT caps {}. The selection rule gives "
            "every cell of a comparison one selection rule "
            "(run_all.check_selection_parity enforces it from the configs at "
            "launch), so differing caps mean a config changed mid-tier and the "
            "arms are not comparable whatever their stopping reasons say."
            .format(sorted(caps)))
    if len(patiences) > 1:
        errors.append(
            "the audited arms recorded DIFFERENT patience values {} in their "
            "resolved configs. Same rule, same reason: one selection rule per "
            "comparison.".format(sorted(patiences)))

    # Agreement across arms is not enough. The shared values must be the
    # registered cap and patience.
    for label, seen, registered, key in (
            ("max_epochs", caps, REGISTERED_MAX_EPOCHS, "cap"),
            ("patience", patiences, REGISTERED_PATIENCE, "patience")):
        off = sorted(v for v in seen if v != registered)
        if not off:
            continue
        who = _names(rows, lambda r, k=key, reg=registered:
                     r[k] is not None and r[k] != reg)
        errors.append(
            "the audited runs were held to {}={}, not the PRE-REGISTERED "
            "the pre-registered "
            "{}={} (set in this file, and in every configs/*.yaml). "
            "{} run(s): {}{}. These arms may be comparable with each "
            "other and they are not the protocol the paper describes, so the "
            "cap audit cannot certify them: 'no arm was truncated by its "
            "budget' is a statement about the registered rule, and this is not "
            "that rule. Either they were launched "
            "from an edited config -- re-run them through runners/run_all.py -- "
            "or the registered constants were changed, which means "
            "check_convergence.REGISTERED_{} updated to match, and every arm "
            "of the comparison re-run under the new rule."
            .format(label, off, label, registered, len(who),
                    ", ".join(who[:6]),
                    "" if len(who) <= 6 else " and {} more".format(len(who) - 6),
                    label.upper()))

    # NOT-PERFORMED notes, each keyed on the condition it actually describes.
    # A malformed config is NOT reported here: it is a FAIL above, by run.
    absent_cfg = ["{} seed {}".format(r["arm"], r["seed"]) for r in rows
                  if r["config_status"] == CFG_ABSENT]
    if absent_cfg:
        notes.append(
            "no config.resolved.yaml in {}, so {} do not state the selection "
            "rule they were held to: the cap-against-config cross-check was "
            "NOT PERFORMED for them, and their patience arithmetic ran against "
            "the PRE-REGISTERED patience {} rather than a recorded one (every "
            "verdict that used it says so). run_all.write_resolved_config "
            "writes that file for every cell it launches, so a run without one "
            "did not come from run_all."
            .format(", ".join(absent_cfg[:6])
                    + ("" if len(absent_cfg) <= 6
                       else " and {} more".format(len(absent_cfg) - 6)),
                    "{} run(s)".format(len(absent_cfg)), REGISTERED_PATIENCE))

    skipped = {}
    for r in rows:
        if r["val_curve_skipped"]:
            skipped.setdefault(r["val_curve_skipped"], []).append(
                "{} seed {}".format(r["arm"], r["seed"]))
    for reason, who in skipped.items():
        notes.append(
            "val-curve sanity NOT PERFORMED for {} of {} run(s) -- {}: {}."
            .format(len(who), len(rows), reason, ", ".join(who)))

    return rows, arm_verdicts, errors, notes


# remedy: one set of conclusions, two renderings

def _names(rows, pred):
    return ["{} seed {}".format(r["arm"], r["seed"]) for r in rows if pred(r)]


def _budget_bound_names(rows):
    return _names(rows, lambda r: r["budget_bound"])


def _cannot_say(rows):
    """Runs of which the cap question could not be ASKED.

    An UNSTAMPED run never recorded how it ended; an INVALID one answered in a
    word this audit does not match on; an unauditable one lost the patience its
    arithmetic check needs. None of the three is a run that merely did not
    happen, so "no arm was cap-bound" is not a conclusion any of them supports.
    """
    return _names(rows, lambda r: r["verdict"] in (UNSTAMPED, INVALID)
                  or r["unauditable"])


def _selection_fault_names(rows):
    """Runs whose own numbers contradict each other WITHOUT the cap being the
    cause -- a last-epoch save, or a checkpoint that is not the minimum of the
    run's validation curve. Raising the cap cannot fix either, so they must not
    be swept into the cap remedy.
    """
    return _names(rows, lambda r: r["contradicted"] and not r["budget_bound"])


def _missing_names(rows):
    return _names(rows, lambda r: r["verdict"] == MISSING)


def remedy_kinds(rows):
    """Every remedy these rows call for, in reporting order.

    Called by BOTH the stdout summary and the markdown report, so the two
    prescribe the same fixes. The markdown is the artifact that is kept and
    read later.
    """
    kinds = []
    if _budget_bound_names(rows):
        kinds.append(REMEDY_CAP)
    if _cannot_say(rows):
        kinds.append(REMEDY_UNKNOWN)
    if _selection_fault_names(rows):
        kinds.append(REMEDY_SELECTION)
    if _missing_names(rows):
        kinds.append(REMEDY_MISSING)
    return kinds


def remedy_stdout(rows):
    kinds = remedy_kinds(rows)
    out = []
    if REMEDY_CAP in kinds:
        out.append("CAP: {} hit their budget. Re-run the WHOLE comparison at a "
                   "raised cap -- never one arm alone."
                   .format(", ".join(_budget_bound_names(rows))))
    if REMEDY_UNKNOWN in kinds:
        cannot = _cannot_say(rows)
        out.append("CAP UNSETTLED: no run RECORDED a cap hit, but {} run(s) "
                   "could not be ASKED ({}). This audit does NOT establish "
                   "that the tier is free of cap-bound arms."
                   .format(len(cannot), ", ".join(cannot)))
    if REMEDY_CAP not in kinds and REMEDY_UNKNOWN not in kinds:
        out.append("No run in this audit was cap-bound.")
    if REMEDY_SELECTION in kinds:
        faults = _selection_fault_names(rows)
        out.append("SELECTION: {} {} their own selection record "
                   "(epoch against selected_epoch, or the validation curve "
                   "against the epoch that was scored). Raising the cap cannot "
                   "fix a checkpoint-selection fault -- fix the trainer and "
                   "re-run those seeds."
                   .format(", ".join(faults),
                           "contradicts" if len(faults) == 1 else "contradict"))
    if REMEDY_MISSING in kinds:
        out.append("MISSING: {}. The failures above are missing or unfinished "
                   "runs. Produce them with runners/run_all.py, then audit "
                   "again.".format(", ".join(_missing_names(rows))))
    if not kinds:
        out.append("The failure(s) above are tier-level -- the arms did not all "
                   "run under one selection rule. Bring the configs back into "
                   "line and re-run the arms that differ.")
    return out


def remedy_markdown(rows):
    kinds = remedy_kinds(rows)
    lines = ["", "## Remedy", ""]
    if REMEDY_CAP in kinds:
        lines += [
            "**Cap-bound.** A cap-bound arm re-runs its WHOLE comparison at a "
            "raised cap -- every arm of the table, to the same new cap. "
            "Extending one arm alone reproduces the unequal-budget confound "
            "the selection rule removed.",
            "",
            "Budget-bound run(s): {}.".format(
                ", ".join(_budget_bound_names(rows))),
            "",
        ]
    if REMEDY_UNKNOWN in kinds:
        cannot = _cannot_say(rows)
        lines += [
            "**The cap question is unsettled.** No run RECORDED hitting its "
            "budget, but {} run(s) could not be ASKED ({}), so this report "
            "does NOT establish that the tier is free of budget-bound arms."
            .format(len(cannot), ", ".join(cannot)),
            "",
            "Fix or re-run those seeds. A resumed run whose checkpoint already "
            "sits at the cap returns from `src/training/train_semantic.py` at "
            "its `if start_epoch > max_epochs:` branch (line 429 when this was "
            "written), before the `stamp_selection` call (line 588), and comes "
            "back looking exactly like a merely interrupted run -- rule that "
            "case out before concluding these were interruptions.",
            "",
        ]
    if REMEDY_CAP not in kinds and REMEDY_UNKNOWN not in kinds:
        lines += ["**No run in this audit was cap-bound.**", ""]
    if REMEDY_SELECTION in kinds:
        lines += [
            "**Checkpoint selection.** {} {} their own selection "
            "record -- `epoch` against `selected_epoch`, or the validation "
            "curve against the epoch that was scored. Raising the cap cannot "
            "fix a last-epoch save or a checkpoint that is not the val-curve "
            "minimum; fix the trainer and re-run those seeds."
            .format(", ".join(_selection_fault_names(rows)),
                    "contradicts" if len(_selection_fault_names(rows)) == 1
                    else "contradict"),
            "",
        ]
    if REMEDY_MISSING in kinds:
        lines += [
            "**Missing runs.** {} produced nothing this audit can score. "
            "Produce them with `runners/run_all.py` and audit again."
            .format(", ".join(_missing_names(rows))),
            "",
        ]
    if not kinds:
        lines += [
            "The failure(s) above are tier-level -- the arms did not all run "
            "under one selection rule. Bring the configs back into line and "
            "re-run the arms that differ.",
            "",
        ]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


# rendering

_ROW = "  {:<34} {:>4} {:>9} {:>5}  {:<11} {}"
_CURVE_ROW = "  {:<34} {:>4}  {}"


def format_rows(rows):
    out = [_ROW.format("arm", "seed", "selected", "cap", "reason", "verdict"),
           "  " + "-" * 80]
    for r in rows:
        out.append(_ROW.format(
            r["arm"], r["seed"],
            "-" if r["selected_epoch"] is None else r["selected_epoch"],
            "-" if r["cap"] is None else r["cap"],
            r["reason"] or "-", r["verdict"]))
    return out


def format_val_curves(rows):
    """One line per run: what its own validation curve says about its choice."""
    out = ["  val-curve sanity -- does the selected epoch sit at the minimum "
           "of the run's own validation loss? (read from train.log)",
           _CURVE_ROW.format("arm", "seed", "curve"),
           "  " + "-" * 80]
    for r in rows:
        out.append(_CURVE_ROW.format(r["arm"], r["seed"], r["val_curve"]))
    return out


def write_report(path: Path, *, label: str, root: Path, seeds, rows,
                 arm_verdicts, errors, notes, overwrite: bool = False) -> Path:
    """Write the markdown cap audit. Returns the path written.

    REFUSES to replace an existing file unless *overwrite*. The guard is here,
    in the writer, and not only in `main` -- `main` decides before the audit
    runs so that a refusal is free, but a caller reaching this function
    directly must not be able to skip that decision. A cap audit is the only
    record of what some earlier run concluded.

    It RAISES. There is no try/except here turning a refused write into a
    silent skip: a caller that asked for a file and got none has to hear so.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            "refusing to overwrite {}. A cap audit is a READ of the results "
            "tree; it must not be the thing that destroys an artifact in it, "
            "and the report already there is the only copy of whatever run "
            "produced it. Pass overwrite=True -- the CLI's --force -- if "
            "replacing it is what you mean, or write somewhere else."
            .format(path))
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    caps = sorted({r["cap"] for r in rows if r["cap"] is not None})
    pats = sorted({r["patience"] for r in rows if r["patience"] is not None})
    lines = [
        "# Cap audit -- {}".format(label),
        "",
        "Generated {} by `runners/check_convergence.py`.".format(stamp),
        "Results root: `{}`   seeds: {}".format(
            root, " ".join(str(s) for s in seeds)),
        "",
        "Early stopping uses validation loss. "
        "The evaluated checkpoint is the one with the best validation loss.",
        "Cap(s) recorded by these runs: {}. Patience: {}.".format(
            caps if caps else "none recorded", pats if pats else "not recorded"),
        "Pre-registered rule: max_epochs {}, patience {} "
        "(`runners/check_convergence.py`, `REGISTERED_MAX_EPOCHS` / "
        "`REGISTERED_PATIENCE`).".format(REGISTERED_MAX_EPOCHS, REGISTERED_PATIENCE),
        "",
        "`stopping_reason` is `early_stop` (the run converged) or `cap` (the "
        "run was truncated by its budget).",
        "",
        "| arm | seed | selected epoch | cap | stopping reason | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| {} | {} | {} | {} | `{}` | {} |".format(
            r["arm"], r["seed"],
            "-" if r["selected_epoch"] is None else r["selected_epoch"],
            "-" if r["cap"] is None else r["cap"],
            r["reason"] or "-", r["verdict"]))
    lines += [
        "",
        "## Val-curve sanity",
        "",
        "Read from each run's `train.log`. The question is whether the epoch "
        "the run SELECTED is the epoch at which its own validation loss "
        "bottoms out -- not whether the curve looks converged, which patience "
        "decided during the run.",
        "",
        "| arm | seed | validation curve |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append("| {} | {} | {} |".format(
            r["arm"], r["seed"], r["val_curve"]))
    lines += ["", "## Per arm", ""]
    for arm, v in arm_verdicts.items():
        lines.append("- **{}** -- {}".format(arm, v))
    if notes:
        lines += ["", "## Not performed", ""] + ["- " + n for n in notes]
    lines += ["", "## Verdict", ""]
    if errors:
        lines += ["- FAIL: " + e for e in errors]
        lines += remedy_markdown(rows)
    else:
        lines.append(
            "- PASS: every audited seed stopped early, inside its cap, under "
            "one selection rule. This comparison is not budget-confounded.")
    # The scope disclosure goes AFTER the verdict, on both paths. The markdown
    # is the artifact kept and read weeks later (see `remedy_kinds`), so a
    # report that states PASS without stating what it did not look at is the
    # same over-reading trap as the stdout line, in durable form.
    lines += not_checked_markdown()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# the importable gate -- `runners/check_tier.py` calls this

def resolve_root(tier, experiment, results):
    """Where this audit reads from. Returns ``(root, problem)``.

    `problem` is a sentence about the INVOCATION, not about any run: an
    unknown tier, a results directory that is not there. Those are exit-2
    conditions for the CLI and `usage_error` verdicts for a caller, and they
    are kept apart from run failures because "this tier did not pass" and "you
    pointed me at nothing" are different things to tell an operator.

    A scouting tier is routed to `results/scouting`, which is where
    `run_all.main` writes it (runners/run_all.py:429-430). Auditing plain
    `results/` for those arms would report every seed missing for a tier that
    ran perfectly well.
    """
    run_all = _load_run_all()
    root = Path(results)
    if tier is not None:
        if tier not in run_all.TIERS:
            return None, "unknown tier {!r}; known: {}".format(
                tier, ", ".join(sorted(run_all.TIERS)))
        if tier in run_all.SCOUTING_TIERS:
            root = root / "scouting"
    if not root.is_dir():
        return None, ("no results directory {}. Pass --results, or run the "
                      "tier first with runners/run_all.py.".format(root))
    if experiment is not None and not (root / experiment).is_dir():
        scout = root / "scouting" / experiment
        msg = "no results directory {}.".format(root / experiment)
        if scout.is_dir():
            msg += (" It exists under {} -- that is a scouting run; pass "
                    "--results {}/scouting.".format(scout, root))
        return None, msg
    return root, None


def resolve_report_target(report, no_report, force, name, root):
    """Where the markdown goes. Returns ``(path_or_None, problem_or_None)``.

    Decided BEFORE any run is read, for the same reason `resolve_root` is: an
    invocation that cannot write where it was told to is a fact about the
    INVOCATION, not a verdict about the tier. Deciding it first means a refusal
    reads nothing, writes nothing, creates no parent directory, and cannot be
    confused with a tier failure -- it exits EXIT_REPORT_EXISTS (3), its own
    code, and the audit simply did not run.

    That code is NOT EXIT_USAGE. The invocation was legal; only the write was
    refused. While the two shared 2, `scripts/batch/tier-cycle.sh` could not
    tell "the report is already there" from "unknown tier" and, reducing its
    gate lines with `grep -E "=[1-9]"`, could not tell either from "this tier
    is budget-confounded": a healthy tier stopped the unattended chain.

    * ``no_report``      -- read-only. ``(None, None)``: nothing to write, and
                            nothing that can go wrong with writing it.
    * ``report``         -- that path, subject to the same no-clobber rule as
                            the default. `--report` is otherwise untouched.
    * neither            -- ``<root>/convergence_<name>.md``, the path the
                            batch's own reports live at.

    An existing target without *force* is the refusal. `exists()` rather than
    `is_file()`: a directory sitting on the target path is also something this
    script must not be trusted to resolve on its own.
    """
    if no_report:
        return None, None
    path = Path(report) if report else Path(root) / "convergence_{}.md".format(name)
    if path.exists() and not force:
        return None, (
            "refusing to overwrite the existing report {}\n"
            "\n"
            "A cap audit READS the results tree. It must not be the thing "
            "that destroys an artifact in it, and that report is the only "
            "copy of whatever run produced it -- the batch's own audit, or an "
            "earlier regeneration. Nothing has been read and nothing has been "
            "written.\n"
            "\n"
            "THIS IS NOT A VERDICT ABOUT THE TIER (exit {}, not {}). No run "
            "was audited, so this says nothing about whether any arm was "
            "cap-bound. Pick one:\n"
            "\n"
            "  --no-report        audit and print; write nothing at all "
            "(read-only)\n"
            "  --report <path>    write the regenerated report somewhere "
            "else, then diff it\n"
            "  --force            replace {} on purpose\n"
            .format(path, EXIT_REPORT_EXISTS, EXIT_TIER_FAILED, path.name))
    return path, None


def _usage_verdict(problem, root=None) -> ConvergenceVerdict:
    return ConvergenceVerdict(errors=[problem], warnings=[], rows=[],
                              arm_verdicts={}, notes=[], failures=[problem],
                              gates=True, usage_error=True, root=root)


def _verdict(tier, arms, root, seeds) -> ConvergenceVerdict:
    """Run the audit over *arms* and package it for a caller.

    The scouting rule is applied HERE rather than at exit-code time, so that a
    caller which only reads `errors` and `warnings` -- which is all
    `check_tier.py` does with its own two lists -- cannot fail a P2 gate on a
    P3 tier, and cannot silently drop the findings either: every demoted
    failure is repeated in `warnings`, prefixed with why it does not gate.
    """
    rows, arm_verdicts, failures, notes = audit(arms, root, seeds)
    gates = tier is None or tier not in _load_run_all().SCOUTING_TIERS
    if gates:
        return ConvergenceVerdict(
            errors=list(failures), warnings=list(notes), rows=rows,
            arm_verdicts=arm_verdicts, notes=notes, failures=failures,
            gates=True, usage_error=False, root=root)
    demoted = [
        "{} is a scouting tier (run_all.SCOUTING_TIERS), so this does NOT "
        "gate -- nothing from this tier enters P2: {}".format(tier, f)
        for f in failures]
    return ConvergenceVerdict(
        errors=[], warnings=list(notes) + demoted, rows=rows,
        arm_verdicts=arm_verdicts, notes=notes, failures=failures,
        gates=False, usage_error=False, root=root)


def check(tier: str, results, seeds) -> ConvergenceVerdict:
    """Audit every arm of *tier*. THE ENTRY POINT FOR `check_tier.py`.

    Deliberately the same argument order as
    `runners/check_tier.py::check(tier, results, seeds)`: *results* is the
    results ROOT (already joined to the repo, as check_tier receives it), and
    *seeds* is the list of seeds the tier was run with.

        verdict = check_convergence.check(tier, results, seeds)
        errors.extend(verdict.errors)      # [] on a scouting tier
        warns.extend(verdict.warnings)

    Returns a `ConvergenceVerdict`; see that class for every field. It is a
    NamedTuple, so ``errors, warnings, *rest = check(...)`` works, but a
    two-value unpack RAISES rather than dropping the rest.
    """
    root, problem = resolve_root(tier, None, results)
    if problem is not None:
        return _usage_verdict(problem)
    arms = [c.name for c in _load_run_all().TIERS[tier]]
    return _verdict(tier, arms, root, list(seeds))


def check_experiment(experiment: str, results, seeds) -> ConvergenceVerdict:
    """Audit ONE arm by directory name (the `--experiment` mode).

    No tier means no scouting demotion: an arm audited by name is audited on
    its own terms and its failures gate.
    """
    root, problem = resolve_root(None, experiment, results)
    if problem is not None:
        return _usage_verdict(problem)
    return _verdict(None, [experiment], root, list(seeds))


def _print_footer(report_path, args) -> None:
    """The last thing every invocation prints, pass or fail.

    Two jobs, both about not being over-read.

    1. SAY WHETHER A FILE EXISTS NOW. A run that wrote nothing and a run that
       refreshed the report must not look the same.
    2. STATE THE SCOPE. `not_checked_lines` on the pass path as well as the
       fail path -- the pass is the one that gets quoted.
    """
    if report_path:
        print("Report: {}".format(report_path))
    else:
        print("No report was written (read-only audit). Pass --report <path> "
              "for a markdown copy; --force to replace an existing one.")
    print()
    for line in not_checked_lines():
        print(line)


def main(argv=None) -> int:
    """CLI over `check`/`check_experiment` -- parse, render, choose an exit code.

    Every verdict in here comes from the functions above, so the gate
    `check_tier.py` calls and the report a human reads can never reach
    different conclusions about the same results directory.
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tier", help="audit every arm of this tier")
    g.add_argument("--experiment", help="audit one arm by directory name")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--results", default="results")
    w = ap.add_mutually_exclusive_group()
    w.add_argument("--report", default=None,
                   help="where to write the markdown report "
                        "(default: <results>/convergence_<tier>.md). Will not "
                        "overwrite an existing file without --force")
    w.add_argument("--no-report", action="store_true",
                   help="READ-ONLY: print the audit and write nothing at all")
    ap.add_argument("--force", action="store_true",
                    help="replace the report if it already exists. Without "
                         "this, an audit never destroys an existing report")
    args = ap.parse_args(argv)

    results = REPO / args.results
    label = ("tier {}".format(args.tier) if args.tier
             else "experiment {}".format(args.experiment))

    # Both invocation questions -- "is there anything here to audit?" and "can
    # I write where I was told to?" -- are settled before a single run is read.
    # A refusal at this point has touched nothing.
    root, problem = resolve_root(args.tier, args.experiment, results)
    if problem is not None:
        print(problem)
        return EXIT_USAGE
    report_path, write_problem = resolve_report_target(
        args.report, args.no_report, args.force,
        args.tier or args.experiment, root)
    if write_problem is not None:
        # EXIT_REPORT_EXISTS, not EXIT_USAGE. The invocation was legal and
        # the tier may be perfectly sound -- this run refused to replace an
        # existing report and therefore read nothing.
        print(write_problem)
        return EXIT_REPORT_EXISTS

    if args.tier:
        verdict = check(args.tier, results, args.seeds)
    else:
        verdict = check_experiment(args.experiment, results, args.seeds)

    if verdict.usage_error:
        for problem in verdict.errors:
            print(problem)
        return EXIT_USAGE

    rows = verdict.rows
    print("=" * 84)
    print("{} -- convergence check (early stop on validation loss, "
          "best-val checkpoint)".format(label.upper()))
    print("pre-registered rule: max_epochs {}, patience {} "
          "(`runners/check_convergence.py`)"
          .format(REGISTERED_MAX_EPOCHS, REGISTERED_PATIENCE))
    print("=" * 84)
    for line in format_rows(rows):
        print(line)
    print()
    for line in format_val_curves(rows):
        print(line)
    print()
    for arm, v in verdict.arm_verdicts.items():
        print("  {:<34} {}".format(arm, v))
    print()
    for n in verdict.notes:
        print("  note   {}".format(n))
    if verdict.notes:
        print()

    # `report_path` was decided before the audit ran and is already known to
    # be safe to write (absent, or --force). `overwrite` is passed anyway so
    # the writer's own guard is the one that has the final say -- if the file
    # appeared between that decision and now, the FileExistsError raises here
    # rather than the audit silently replacing it.
    if report_path is not None:
        report_path = write_report(
            report_path,
            label=label, root=verdict.root, seeds=args.seeds, rows=rows,
            arm_verdicts=verdict.arm_verdicts, errors=verdict.failures,
            notes=verdict.notes, overwrite=args.force)

    if verdict.failures:
        for e in verdict.failures:
            print("  FAIL   {}".format(e))
        print("\n{} problem(s). This comparison is NOT a fair one: its arms "
              "did not all converge under one selection rule."
              .format(len(verdict.failures)))
        # Only prescribe the cap remedy when the cap is actually the problem,
        # and reach the same conclusions the markdown report reaches:
        # `remedy_kinds` decides once, for both.
        for line in remedy_stdout(rows):
            print(line)
        _print_footer(report_path, args)
        # A scouting tier is P3 work and never enters P2, so its verdicts are
        # reported but do not stop the chain. Announced out loud rather than
        # applied by quietly skipping the arms above. `verdict.gates` is the
        # same flag a caller sees, so the CLI and check_tier.py demote exactly
        # the same tiers.
        if not verdict.gates:
            print("\n{} is a scouting tier (run_all.SCOUTING_TIERS): the "
                  "failures above are REPORTED but do not gate, because "
                  "nothing from this tier enters P2. Exit code forced to 0."
                  .format(args.tier))
            return EXIT_OK
        return EXIT_TIER_FAILED

    print("  every audited seed stopped early, inside its cap, under one")
    print("  selection rule. No arm was truncated by its budget.")
    print("\n{} IS NOT BUDGET-CONFOUNDED.".format(label.upper()))
    _print_footer(report_path, args)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

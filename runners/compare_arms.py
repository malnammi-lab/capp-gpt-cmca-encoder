#!/usr/bin/env python3
"""
compare_arms.py -- side-by-side look at one tier's arms.

Diagnostic only. Nothing this script prints is for the paper. The quotable
table is `runners/aggregate_seeds.py`. The checklist is `runners/check_tier.py`.
This file prints a NOT QUOTABLE banner on every run.

It reads `results/<cell>/seed<k>/eval_metrics.json`. Differing stopping epochs
are expected. Semantic metrics are read from the flat layout and from the
nested `semantic` block. A metric present for some seeds of an arm and absent
for others is a fault. A metric no seed recorded prints as "-". Training and
evaluated fingerprints are compared with this tree before the table. While any
fault is open, content lines are prefixed `!UNSOUND!`.

EXIT CODES
----------
0 -- every arm complete, every fingerprint this tree's, one test-set size, a
     confusion matrix everywhere, and every column a mean over every seed in
     its row. (Still NOT QUOTABLE: see the banner.)
2 -- at least one fault. They are printed as a FAIL list ABOVE the table and
     again at the end, every content line is prefixed `!UNSOUND!`, and they are
     stored under "problems" in the JSON. Faults are: a missing or incomplete
     arm, a recorded fingerprint that is not this tree's or absent, arms scored
     on different test-set sizes, and an arm with no confusion matrix.
1 -- the table could not be built at all, and no artifact is written: no runs
     found, or a metrics file that cannot support the row it would appear in
     (no accuracy, no test-set size, a confusion matrix on only some seeds,
     matrices of different shapes, or a metric recorded by some seeds of an arm
     and not others). These raise SystemExit with the remedy in the message.

Usage:
    python runners/compare_arms.py --dataset cadsynth
    python runners/compare_arms.py --dataset mfinstseg
    python runners/compare_arms.py --dataset cadsynth --baseline similarity
    python runners/compare_arms.py --dataset cadsynth --seeds 0 1 2
    python runners/compare_arms.py --dataset cadsynth         --arms resnet_cadsynth_ce resnet_cadsynth_ce_weighted         --baseline resnet_cadsynth_ce
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from runners.aggregate_seeds import arm_dir, load_runs, mean_std  # noqa: E402
from runners.check_tier import headline_metrics, semantic_block  # noqa: E402
from src.utils.provenance import code_fingerprint  # noqa: E402

# Shuffled is the control: adjacency with node labels permuted.
# Compare it with adjacency.
# T1/T2's default --arms. Tests compare against this tuple.
ARMS = ("similarity", "adjacency", "both", "shuffled")

#: The five mean +/- std columns of the table, in the order they are printed.
#: Presence is tracked for all of them by one loop in `summarise_runs`.
METRIC_COLUMNS = ("overall_accuracy", "macro_precision", "macro_recall",
                  "macro_f1", "mean_iou")

#: Prefixed onto every content line of the report
#: while any fault is open. Kept as a constant so the tests grep for the same
#: token an operator is told to grep for, and so there is one spelling of it.
UNSOUND = "!UNSOUND!"

NOT_QUOTABLE = """\
!! DIAGNOSTIC ONLY -- NOT QUOTABLE. This tool is not on the quotability
   checklist. For any number that enters
   the paper run:
       python runners/aggregate_seeds.py --dataset {ds}
       python runners/check_tier.py      --tier <T1..T7>
"""

#: The name this tool stamps into every artifact it writes. It is also the
#: token `owner_of` looks for, so a file that does not carry it is a file this
#: tool did not write and will not write over.
WRITTEN_BY = "runners/compare_arms.py"

#: Longest readable slug allowed in a derived filename before a digest stands
#: in for it. Chosen so `<dataset>_comparison_<slug>.json` stays well inside
#: the 255 bytes NTFS and ext4 both allow for one path component.
_MAX_SLUG = 60


def _short(dataset: str, arm: str) -> str:
    """`cadsynth_bias_both` -> `both`; `cadsynth_ce_head` -> `ce_head`.

    A cell name already carries the dataset, and so does the filename it goes
    into, so repeating it makes a derived name unreadable without making it
    any more unique.
    """
    for prefix in ("{}_bias_".format(dataset), "{}_".format(dataset)):
        if arm.startswith(prefix) and len(arm) > len(prefix):
            return arm[len(prefix):]
    return arm


def comparison_filename(dataset: str, arms: List[str]) -> str:
    """The file `arms` belong in -- DERIVED from the arms, not from --dataset.

    The default T1/T2 tuple keeps the documented name, because for THOSE arms
    the name is not a guess: the comparison of `similarity`, `adjacency`,
    `both` and `shuffled` IS the bias ablation.

    Any other arm list gets a name built from the arms themselves, so two
    comparisons collide only when they are the same comparison. The readable
    form is used while it stays short and its pieces stay distinct; otherwise a
    digest of the exact arm list stands in, which is unreadable but cannot
    collide by accident.
    """
    if list(arms) == list(ARMS):
        return "{}_bias_comparison.json".format(dataset)
    pieces = [_short(dataset, a) for a in arms]
    slug = "-".join(pieces)
    if len(slug) > _MAX_SLUG or len(set(pieces)) != len(pieces):
        digest = hashlib.sha1("\0".join(arms).encode("utf-8")).hexdigest()[:12]
        slug = "{}-and-{}-more-{}".format(pieces[0][:24], len(arms) - 1, digest)
    return "{}_comparison_{}.json".format(dataset, slug)


def owner_of(path: Path, dataset: str, arms: List[str]):
    """Is `path` this tool's artifact for exactly `arms`?

    Returns `(True, "")` when writing `arms` there would replace a comparison
    of the same arms.

    Returns `(False, reason)` otherwise, and the caller REFUSES rather than
    writing. Nothing is swallowed here: a foreign or unreadable file is not an
    error being ignored, it is the answer to the question asked, and `reason`
    is the text the refusal prints. The caller's message names `--out` and
    `--force`, so an operator who does mean to replace the file is never stuck.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, ("it is not valid JSON ({}), so what it holds cannot be "
                       "identified".format(exc))
    if not isinstance(doc, dict):
        return False, "its top level is a {}, not an object".format(
            type(doc).__name__)
    if doc.get("written_by") != WRITTEN_BY:
        return False, ("it was not written by {} (written_by={!r})"
                       .format(WRITTEN_BY, doc.get("written_by")))
    # Our own invalidation stub: an earlier run of THIS tool that did not
    # finish. It names no arms because it is written before any are read, and
    # refusing on it would leave one aborted run blocking the path for good.
    if doc.get("complete") is False and "arms" not in doc:
        return True, ""
    if "arms_requested" in doc:
        # The exact list the operator gave.
        there = list(doc["arms_requested"])
    elif isinstance(doc.get("arms"), dict):
        # A file with no arms_requested, such as the committed
        # results/<dataset>_bias_comparison.json: the arm keys are all there
        # is to go on.
        there = list(doc["arms"])
    else:
        return False, "it names no arms, so it cannot be matched against these"
    if sorted(there) != sorted(arms):
        reason = ("it is the comparison of {} -- a different set of arms"
                  .format(", ".join(sorted(there)) or "no arms"))
        # `comparison_filename` derives a name through `_short`, so `similarity`
        # and `cadsynth_bias_similarity` -- which `aggregate_seeds.arm_dir`
        # resolves to ONE directory -- derive the same filename while reading as
        # two different lists here. That is deliberate: `arm_dir` prefers a
        # literal directory, so if `results/similarity` ever existed alongside
        # `results/cadsynth_bias_similarity` the two spellings WOULD be
        # different comparisons, and a guard that assumed otherwise would
        # silently overwrite one with the other. The stricter answer is kept
        # and the likely cause is named instead, so the refusal is actionable
        # rather than baffling.
        if sorted(_short(dataset, a) for a in there) == sorted(
                _short(dataset, a) for a in arms):
            reason += (" (the same arms under a different spelling -- it "
                       "records them as {}. Re-run naming them that way, or "
                       "--force if the two spellings really are different "
                       "directories here)".format(" ".join(there)))
        return False, reason
    return True, ""


def title_lines(dataset: str, arms: List[str], seeds: List[int]) -> List[str]:
    """The banner's heading, named after what was actually compared.

    For the default four arms the title is the bias ablation, and that title
    is true.
    """
    tail = "   (seed-averaged, seeds requested {})".format(seeds)
    if list(arms) == list(ARMS):
        return ["BIAS ABLATION -- {}{}".format(dataset, tail)]
    head = "ARM COMPARISON -- {}: {}".format(dataset, " vs ".join(arms))
    if len(head) + len(tail) <= 78:
        return [head + tail]
    # Too wide for the rule above it. The arm names stay on the title line --
    # they ARE the title -- and the seed note moves down rather than pushing
    # them off the end. `say` marks both lines, so neither can be read without
    # the verdict.
    return [head, "  {}".format(tail.strip())]



def per_class_f1_from_cm(cm) -> Dict[int, float]:
    """Per-class F1 from a (summed) confusion matrix, rows = true class.

    Seed-averaged per-class numbers come from SUMMING the seeds' confusion
    matrices and scoring once, never from averaging per-seed F1 ratios. That is
    the same rule `aggregate_seeds.py` states for IoU.

    * `training-regiment.md` section 1 asks for "per-class tables from the
      seed-averaged confusion matrix"; the summed matrix IS that artifact, so
      pooling is what the requirement says rather than an inference from it.
    * A pooled ratio is one estimate taken over every seed's predictions at
      once. A mean of per-seed ratios is a different estimator, and for a class
      a seed never predicts it is 0/0 -- undefined, and silently scored 0 by
      whoever averages it.
    * It collapses the seed dimension, so a pooled per-class number has no
      error bar and must never be printed with one.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp),
                          where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp),
                   where=denom > 0)
    return {int(i): float(v) for i, v in enumerate(f1)}


def per_class_support_from_cm(cm) -> Dict[int, int]:
    return {int(i): int(v) for i, v in enumerate(np.asarray(cm).sum(axis=1))}


def load_arm(dataset: str, arm: str, results: Path,
             seeds: List[int]) -> Optional[Dict]:
    """Seed-average one arm from `results/<cell>/seed<k>/eval_metrics.json`.

    The evaluated epoch is `selected_epoch` in the metrics provenance,
    beside `stopping_reason`.
    """
    runs, missing = load_runs(results, dataset, arm, seeds)
    if not runs:
        return None
    return summarise_runs(arm, runs, missing)


def summarise_runs(arm: str, runs: Dict[int, Dict],
                   missing: List[int]) -> Dict:
    """Seed-average one arm's already-loaded metrics files.

    Args:
        arm: Arm name, used in error messages only.
        runs: {seed: parsed metrics file}, flat or nested layout.
        missing: Seeds that were requested and not found.

    Raises:
        SystemExit: If a run carries no accuracy or no test-set size in either
            layout, or if only SOME of the seeds carry a confusion matrix, or
            if those matrices disagree in shape, or if only SOME of the seeds
            recorded one of the mean +/- std metrics. Every one of these would
            otherwise average a smaller sample than the header claims.
    """
    METRICS = METRIC_COLUMNS
    def prov(m, key):
        return (m.get("provenance") or {}).get(key)

    # Flat layout (`evaluate_semantic.py`): overall_accuracy and n_samples at
    # the top level. Nested layout (tier T5, `evaluate_unified.py`): semantic
    # metrics under ["semantic"], denominator `n_faces`. Both layouts go
    # through the accessor in check_tier.py.
    values = {k: [] for k in METRICS}      # type: Dict[str, List[float]]
    recorded_by = {k: [] for k in METRICS}  # type: Dict[str, List[int]]
    absent_from = {k: [] for k in METRICS}  # type: Dict[str, List[int]]
    n_samples = []
    for s in sorted(runs):
        m = runs[s]
        sem = semantic_block(m)
        a, f, i, n = headline_metrics(m)
        if a is None:
            raise SystemExit(
                "arm {} seed {}: this metrics file has no overall_accuracy in "
                "either layout -- not at the top level (flat, "
                "evaluate_semantic) and not under ['semantic'] (nested, "
                "evaluate_unified). Top-level keys are {}. Dropping the seed "
                "and averaging the rest would report a smaller sample as if it "
                "were the full one, so this stops instead. Re-score the run "
                "with src/evaluation/evaluate_semantic.py.".format(
                    arm, s, sorted(m)))
        if n is None:
            raise SystemExit(
                "arm {} seed {}: no n_samples (flat) and no n_faces (nested) in "
                "this metrics file. The 'arms evaluated on different test sizes' "
                "check further down would then compare an empty set and pass, so "
                "a split that moved between runs -- the defect behind this "
                "project's 52.26-vs-64.24 discrepancy -- would go unreported. "
                "Re-score the run.".format(arm, s))
        n_samples.append(n)
        # macro P and R are optional: an arm that recorded neither prints "-"
        # rather than a fabricated value (see main()). Optional means "absent
        # from every seed of the arm". Absent from some of them is not a
        # missing column; it is a column that lies about its n.
        for key, v in (("overall_accuracy", a),
                       ("macro_precision", sem.get("macro_precision")),
                       ("macro_recall", sem.get("macro_recall")),
                       ("macro_f1", f),
                       ("mean_iou", i)):
            if v is None:
                absent_from[key].append(s)
            else:
                values[key].append(v)
                recorded_by[key].append(s)

    # A metric recorded by some
    # seeds and not others cannot be averaged under this arm's seed count, and
    # there is no honest way to print it in a row whose `seeds` column is a
    # single number. Reporting it into the artifact and carrying on is what the
    # fingerprint check does, and that is right for a verdict ABOUT runs that
    # were read; here the row itself would be false, and it would be the row a
    # reader copies. This is the same refusal, for the same reason and in the
    # same function, as the confusion-matrix check immediately below.
    partial = [k for k in METRICS if recorded_by[k] and absent_from[k]]
    if partial:
        detail = "; ".join(
            "{} recorded by seed(s) {} and missing from seed(s) {} -> a mean "
            "over {} of {}".format(k, recorded_by[k], absent_from[k],
                                   len(recorded_by[k]), len(runs))
            for k in partial)
        raise SystemExit(
            "arm {}: {}. The table prints ONE seed count per row -- it would "
            "say {} -- so those columns would claim a sample size they do not "
            "have, and two identical values presented as though there were {} "
            "show up as an impressively tight +/-0.0000. Averaging what is "
            "left is how a "
            "wrong number comes to look like a right one, which is why the "
            "confusion-matrix check below refuses the same shape. Re-score the "
            "seeds that are missing a metric with "
            "src/evaluation/evaluate_semantic.py (it reads the best-val "
            "checkpoint; no retraining), or pass --seeds without them so the "
            "header states the smaller n. A metric absent from EVERY seed of "
            "an arm is a different case and still prints '-'.".format(
                arm, detail, len(runs), len(runs)))

    # A seed that carries no matrix is not skipped: summing over a subset
    # would print per-class F1 from a different set of runs than the means above
    # it, which is the "quietly smaller sample" failure aggregate_seeds refuses
    # for the same reason.
    cms, without = [], []
    for s in sorted(runs):
        cm = semantic_block(runs[s]).get("confusion_matrix")
        if cm is None:
            without.append(s)
        else:
            cms.append(np.asarray(cm, dtype=np.int64))
    if cms and without:
        raise SystemExit(
            "arm {}: seeds {} carry a confusion matrix and seeds {} do not. The "
            "summed matrix would cover a different set of runs from the mean "
            "+/- std printed beside it, so per-class F1 under a {}-seed mean "
            "would silently be over {}. Re-score the seeds that are missing "
            "one.".format(arm, [s for s in sorted(runs) if s not in without],
                          without, len(runs), len(cms)))
    summed = None
    if cms:
        shapes = {c.shape for c in cms}
        if len(shapes) != 1:
            raise SystemExit(
                "arm {}: confusion matrices differ in shape across seeds: {}. "
                "The seeds are not scoring the same class set; do not "
                "aggregate.".format(arm, shapes))
        summed = np.sum(cms, axis=0)

    # `selected_epoch` and `stopping_reason` are written by
    # src/utils/provenance.build_provenance() under the selection rule. Runs recorded
    # before that fall back to the older `epoch` field, and to None; the report
    # below prints "?" rather than inventing a value.
    selected = sorted({prov(m, "selected_epoch") or prov(m, "epoch")
                       for m in runs.values()} - {None})
    reasons = sorted({prov(m, "stopping_reason") for m in runs.values()} - {None})

    # Every fingerprint any run of this arm recorded, for
    # the live-vs-recorded comparison in main(). Both are collected: a run
    # trained by this tree but SCORED by an older one is just as untraceable,
    # and `provenance_from_checkpoint` records the scoring one separately under
    # ["evaluated"]. A run that recorded NO fingerprint is listed by seed rather
    # than passed over -- "no value" is precisely the case a `.get()` default
    # would turn into "nothing to check".
    #
    # `_fingerprint_entries` is the same facts with the SEED still attached as
    # a number rather than baked into a label. Two entries can belong to one
    # run.
    fps, no_fp, fp_entries = {}, [], []
    for s in sorted(runs):
        pr = (runs[s].get("provenance") or {})
        train_fp = pr.get("code_fingerprint")
        eval_fp = (pr.get("evaluated") or {}).get("code_fingerprint")
        if train_fp:
            label = "seed {}".format(s)
            fps[label] = train_fp
            fp_entries.append({"seed": s, "role": "trained", "label": label,
                               "fingerprint": train_fp})
        if eval_fp:
            label = "seed {} (scored by)".format(s)
            fps[label] = eval_fp
            fp_entries.append({"seed": s, "role": "scored", "label": label,
                               "fingerprint": eval_fp})
        if not train_fp and not eval_fp:
            no_fp.append(s)

    return {
        "_arm": arm,
        "_seeds": sorted(runs),
        "_missing_seeds": missing,
        "_fingerprints": fps,
        "_fingerprint_entries": fp_entries,
        "_seeds_without_fingerprint": no_fp,
        "_has_confusion_matrix": summed is not None,
        "_selected_epochs": selected,
        "_stopping_reasons": reasons,
        # How many seeds each column's mean is actually over. After the
        # refusal above it is either 0 (prints "-") or every seed of the arm.
        # Carried explicitly and checked at the point of printing.
        "_metric_n": {k: len(values[k]) for k in METRICS},
        "_metric_absent_from": {k: absent_from[k] for k in METRICS},
        "n_samples": n_samples,
        "overall_accuracy": mean_std(values["overall_accuracy"]),
        "macro_precision": mean_std(values["macro_precision"]),
        "macro_recall": mean_std(values["macro_recall"]),
        "macro_f1": mean_std(values["macro_f1"]),
        "mean_iou": mean_std(values["mean_iou"]),
        "confusion_matrix_summed": summed,
        "per_class_f1": per_class_f1_from_cm(summed) if summed is not None else {},
        "per_class_support": (per_class_support_from_cm(summed)
                              if summed is not None else {}),
    }


def _fmt(pair, n, n_seeds, name="", arm="", pct=False, width=16):
    """mean +/- std in a fixed-width column, or a dash when never recorded.

    `n` is how many seeds the mean is
    actually over and `n_seeds` is what the row's `seeds` column says. They
    must agree, or the cell is a claim about a sample size that does not exist.
    `summarise_runs` already refuses to build such a summary; this is the
    second net, at the point where the number stops being a float in a dict and
    becomes a number in a table someone copies. It costs one comparison and it
    is the only check that sits between any future caller and the printed page.

    Raises:
        SystemExit: If a mean is about to be printed over a different number of
            seeds from the one printed beside it, or if a column with no mean
            nevertheless collected values.
    """
    mean, std = pair
    if mean is None:
        if n:
            raise SystemExit(
                "INTERNAL: column {!r} of arm {!r} has no mean but {} recorded "
                "value(s). mean_std() returns (None, None) only for an empty "
                "list, so the two disagree and one of them is wrong. Do not "
                "print this table.".format(name, arm, n))
        return "{:>{w}}".format("-", w=width)
    if n != n_seeds:
        raise SystemExit(
            "INTERNAL: column {!r} of arm {!r} is a mean over {} seed(s) but "
            "the row says {}. This is verifier probe 15: the seeds column "
            "reads {} while the number beside it is an average of {}, and two "
            "identical values print as +/-0.0000 as if {} runs had agreed. "
            "summarise_runs() should have refused this arm already -- that it "
            "did not is a defect in it, not a reason to print the row. Nothing "
            "here is usable until it is fixed.".format(
                name, arm, n, n_seeds, n_seeds, n, n_seeds))
    if pct:
        cell = "{:.2f}+/-{:.2f}%".format(100 * mean, 100 * std)
    else:
        cell = "{:.4f}+/-{:.4f}".format(mean, std)
    return "{:>{w}}".format(cell, w=width)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cadsynth",
                   choices=["cadsynth", "mfinstseg"])
    p.add_argument("--results", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    # The arms to compare. A value may be an ARM ("adjacency", resolved by
    # `arm_dir` to `<dataset>_bias_adjacency`) or a whole CELL name
    # ("resnet_cadsynth_ce"); `arm_dir` prefers a literal directory, which is
    # what lets T3-T6's cells be named at all.
    p.add_argument("--arms", nargs="+", default=list(ARMS),
                   help="arms or cell names to compare "
                        "(default: %(default)s -- T1/T2's bias arms)")
    # Validated below against the arms actually in play. The default is None
    # rather than "similarity" so that "the operator said nothing" stays
    # distinguishable from "the operator asked for similarity" -- not the same
    # question on a tier that has no `similarity` cell.
    p.add_argument("--baseline", default=None,
                   help="arm the others are compared against (default: "
                        "`similarity` when it is among --arms, else the first "
                        "of --arms)")
    p.add_argument("--top", type=int, default=6,
                   help="how many per-class gains/losses to list")
    p.add_argument("--out", default=None,
                   help="where to write the comparison JSON (default: "
                        "<results>/<dataset>_bias_comparison.json for the "
                        "default --arms, else a name derived from the arms)")
    # The guard is a guard, not a wall: an
    # operator who means to replace another comparison says so here, and it is
    # then their decision and not an accident of the filename.
    p.add_argument("--force", action="store_true",
                   help="write over an existing comparison of DIFFERENT arms")
    args = p.parse_args(argv)

    # Both refusals RAISE rather than report, and that does not contradict the
    # reasoning under the fingerprint check further down: that paragraph is
    # about a VERDICT on runs which were read successfully, where the stale
    # artifact has to be written over. Here nothing has been read, the
    # invalidation stamp below has not run, and no artifact exists to correct
    # -- the invocation itself is unanswerable.
    arms = list(args.arms)
    dupes = sorted({a for a in arms if arms.count(a) > 1})
    if dupes:
        raise SystemExit(
            "--arms names {} more than once. Each arm is one column of the "
            "comparison and one key of the comparison JSON, so a repeat "
            "would either overwrite its own row or be silently dropped. "
            "Given: {}.".format(dupes, " ".join(arms)))
    if args.baseline is None:
        baseline = ARMS[0] if ARMS[0] in arms else arms[0]
    elif args.baseline not in arms:
        raise SystemExit(
            "--baseline {!r} is not one of the arms being compared. Every "
            "other arm is reported as a per-class DELTA against the baseline, "
            "so a baseline outside --arms has no column to subtract from and "
            "the deltas would silently be taken against whichever arm sorted "
            "first. Arms given: {}. Pass --baseline as one of those, or add "
            "{!r} to --arms.".format(args.baseline, " ".join(arms),
                                     args.baseline))
    else:
        baseline = args.baseline

    results = REPO / args.results
    # `comparison_filename` derives the name from the arms, so the only
    # comparison that can land on another's file is the same comparison. The
    # default tuple keeps the documented name (see that function).
    out = (Path(args.out) if args.out
           else results / comparison_filename(args.dataset, arms))

    # A derived name cannot collide by accident,
    # but --out points anywhere the operator likes, and the two committed bias
    # comparisons are exactly the files most worth protecting. So the
    # destination is IDENTIFIED before anything is written to it: this tool's
    # artifact for these same arms may be replaced, its own unfinished stub may
    # be replaced, and everything else is refused with --out and --force named.
    # This raises rather than reporting, for the same reason the --arms and
    # --baseline refusals above do: nothing has been read, there is no verdict
    # to record, and writing anything at all IS the harm.
    if out.is_file() and not args.force:
        ours, why = owner_of(out, args.dataset, arms)
        if not ours:
            raise SystemExit(
                "refusing to write over {}: {}.\n"
                "This run compares {}. A verification tool that destroys the "
                "evidence it verifies is worse than no tool: the file above "
                "would be gone before a single arm had been read, and an "
                "aborted run would leave a stub in its place.\n"
                "Pass --out <path> to write this comparison somewhere else, "
                "or --force if replacing that file is what you meant."
                .format(out, why, " ".join(arms)))

    # `summarise_runs` aborts on a metrics file with no accuracy, none with no
    # test-set size, a confusion matrix on only some seeds, matrices of
    # different shapes, or a metric recorded by only some seeds. Aborting is
    # right: the row would be false. An abort leaves the previous comparison
    # sitting there, still saying "sound": true about a tree that no longer
    # produces it, and that file outlives the stderr the abort goes to. So the
    # stale artifact is invalidated BEFORE the arms are read, and rewritten
    # with the real payload only if this run gets all the way to the end.
    # Nothing is created that did not already exist, nothing is deleted, and
    # no try/except sits between a failure and its exit -- the failure still
    # raises and still says what to do.
    #
    # The stamp lands only on a file this run was entitled to write: the guard
    # above has already refused a foreign one, and `--force` is the operator
    # saying the replacement is deliberate. The stale record of THIS comparison
    # has to be written over so it cannot be believed.
    if out.is_file():
        out.write_text(json.dumps({
            "written_by": WRITTEN_BY,
            "quotable": False,
            "sound": False,
            "complete": False,
            "dataset": args.dataset,
            "seeds_requested": args.seeds,
            "arms_requested": arms,
            "invalidated_at": datetime.now(timezone.utc).isoformat(),
            "problems": [
                "INCOMPLETE RUN. compare_arms.py started at the timestamp "
                "above and did not finish writing this file. If you are "
                "reading this text, that run refused to build the table (see "
                "its stderr for which metrics file and what to do). The "
                "numbers that used to be here came from an EARLIER state of "
                "results/ and of the code and were not reproduced, so they are "
                "not evidence about the tree you have. Re-run: python "
                "runners/compare_arms.py --dataset {} --arms {}."
                .format(args.dataset, " ".join(arms))],
        }, indent=2), encoding="utf-8")

    got = {a: load_arm(args.dataset, a, results, args.seeds) for a in arms}
    have = [a for a in arms if got[a]]
    if not have:
        print("No eval_metrics.json found under {}/seed*/ (nor for the other "
              "arms {}).\nRun the tier with runners/run_all.py first."
              .format(arm_dir(results, args.dataset, arms[0]),
                      ", ".join(arms[1:]) or "-- none were given"),
              file=sys.stderr)
        return 1
    missing = [a for a in arms if not got[a]]

    #  EVERY FAULT IS DECIDED HERE, BEFORE ANYTHING IS PRINTED             #
    # Faults are known before anything is printed, and every line below
    # carries the verdict.
    #
    #: Everything that makes this comparison unsound. It drives the exit code
    #: and is written into the JSON.
    problems = []          # type: List[str]
    if missing:
        problems.append("no runs found for arm(s): {}".format(", ".join(missing)))

    # It REPORTS and exits non-zero rather than raising, for the reason
    # `aggregate_seeds.py` sets out at its own fingerprint check: the product of
    # this script is an artifact on disk, and the previous comparison at
    # `out` -- whatever `comparison_filename` or `--out` resolved it to --
    # stays there, unmarked, if this aborts before writing. The stale verdict has to be written INTO the artifact.
    # The banner prints above the table because a reader who copies the table
    # never scrolls to the bottom. (That reasoning is about a VERDICT on runs
    # that were read successfully. It is not a general licence to carry on: a
    # metrics file that cannot support the row it would appear in still aborts
    # in summarise_runs, because there the row itself would be false.)
    live_fp = code_fingerprint()
    # One entry per recorded fingerprint, carrying the run it came from. A run
    # records up to TWO -- the tree that trained it and the tree that scored
    # it -- so an entry count is not a run count. Both numbers are stated, and
    # named.
    entries = [dict(e, arm=a,
                    run="{} seed {}".format(a, e["seed"]),
                    key="{} {}".format(a, e["label"]))
               for a in have for e in got[a]["_fingerprint_entries"]]
    recorded_fps = {e["key"]: e["fingerprint"] for e in entries}
    recorded_runs = sorted({e["run"] for e in entries})
    stale_entries = sorted((e for e in entries if e["fingerprint"] != live_fp),
                           key=lambda e: e["key"])
    stale_fps = {e["key"]: e["fingerprint"] for e in stale_entries}
    stale_runs = sorted({e["run"] for e in stale_entries})
    unrecorded = ["{} seed {}".format(a, s) for a in have
                  for s in got[a]["_seeds_without_fingerprint"]]
    fp_problems = []
    if stale_fps:
        shown = ", ".join("{}={}".format(e["key"], e["fingerprint"])
                          for e in stale_entries[:6])
        more = ("" if len(stale_fps) <= 6
                else " (+{} more entries)".format(len(stale_fps) - 6))
        runs_shown = ", ".join(stale_runs[:6])
        if len(stale_runs) > 6:
            runs_shown += " (+{} more runs)".format(len(stale_runs) - 6)
        fp_problems.append(
            "{} run(s) -- {} recorded fingerprint(s) -- do not match the code "
            "in this tree. A run records up to TWO fingerprints, the tree that "
            "TRAINED it and the tree that SCORED it, so the entry count is not "
            "a run count and must not be read as one. The stale runs are {}. "
            "Live code_fingerprint() is {}; recorded {}{}. These numbers were "
            "produced by code that has since changed, so the arm this script "
            "names below is not a result of the current tree. Re-run the "
            "affected arms, or -- if you are deliberately inspecting an "
            "archived tier -- run this script from the tree that produced it. "
            "Either way, do not carry the ranking forward as current."
            .format(len(stale_runs), len(stale_fps), runs_shown, live_fp,
                    shown, more))
    if unrecorded:
        fp_problems.append(
            "{} run(s) recorded no code_fingerprint at all ({}), so there is "
            "nothing to compare against this tree's {}. An unfingerprinted run "
            "is untraceable under training-regiment.md section 4 -- it is not a "
            "run that happens to match. Re-run it through runners/run_all.py."
            .format(len(unrecorded), ", ".join(unrecorded[:6]), live_fp))
    problems.extend(fp_problems)

    incomplete = {a: got[a]["_missing_seeds"] for a in have
                  if got[a]["_missing_seeds"]}
    if incomplete:
        problems.append(
            "incomplete arms -- mean +/- std over fewer than the requested "
            "seeds: {}".format(dict(sorted(incomplete.items()))))

    n_test_all = {n for a in have for n in got[a]["n_samples"]}
    if len(n_test_all) > 1:
        problems.append(
            "arms were evaluated on different test-set sizes {} -- the split "
            "moved between runs, so the ranking below compares numbers from "
            "different held-out sets. Every config must point at "
            "datasets/splits/{}_70_10_20_seed42.json."
            .format(sorted(n_test_all), args.dataset))

    # Every evaluator in src/evaluation writes a matrix; its absence means
    # the metrics file is not what it should be.
    no_cm = [a for a in have if not got[a]["_has_confusion_matrix"]]
    if no_cm:
        problems.append(
            "no confusion matrix recorded for arm(s) {} -- per-class F1, the "
            "per-class deltas and the weakest-class list could not be produced "
            "for them. Re-score those runs with "
            "src/evaluation/evaluate_semantic.py.".format(", ".join(no_cm)))

    # A column NO seed of an arm recorded is not a fault -- it prints "-" and
    # fabricates nothing, and summarise_runs has already refused the mixed
    # case outright. It is still said out loud, because a "-" in a fixed-width
    # table is easy to read as a zero or to skim past, and because the reader
    # deserves to know the column is absent rather than bad.
    never_recorded = {
        a: [k for k in METRIC_COLUMNS if got[a]["_metric_n"][k] == 0]
        for a in have}
    never_recorded = {a: ks for a, ks in never_recorded.items() if ks}

    #  PRINTING -- every content line carries the verdict decided above    #
    sound = not problems
    #: Prefixed onto EVERY content line while any
    #: fault is open. A banner can be scrolled past, and an operator grepping a
    #: captured log for an arm name or for "vs" never sees a banner at all --
    #: but a prefix lands inside the line their grep matched. `grep '^---'`
    #: finds nothing here; `grep 'both vs similarity'` returns the mark with
    #: it; `grep -c '!UNSOUND!'` is the whole check.
    mark = "" if sound else UNSOUND + " "

    def say(line=""):
        """One line of the report, marked when the comparison is not sound."""
        print(mark + line if line.strip() else line)

    say("=" * 78)
    for line in title_lines(args.dataset, arms, args.seeds):
        say(line)
    say("=" * 78)
    # `.format(ds=...)` fills the two command lines the banner tells the reader
    # to run instead; a banner that says "--dataset {ds}" is a banner nobody
    # can follow. It is split into lines so `say` can mark each one.
    for line in NOT_QUOTABLE.format(ds=args.dataset).rstrip("\n").split("\n"):
        say(line)
    say()

    if problems:
        say("!" * 78)
        say("{} PROBLEM(S) -- THIS COMPARISON IS NOT SOUND. Every line below is"
            .format(len(problems)))
        say("prefixed {} so that no number here can be read, copied"
            .format(UNSOUND))
        say("or grepped out of this log without the verdict attached:")
        for pr in problems:
            say("FAIL  {}".format(pr))
        say("!" * 78)
        say()
    if not fp_problems:
        # Printed even when OTHER faults are open: "the runs did come from this
        # tree" is a separate fact from "this comparison is sound", and losing
        # it on every unsound run would cost an operator the one thing that
        # tells them a re-run is not needed. It is marked like every other line.
        #
        # The pass side has to distinguish runs
        # from entries for the same reason the fail side does: "all 24 match"
        # over twelve runs invites the same misreading in the other direction.
        # The phrase "recorded fingerprint(s) match this tree" is kept
        # contiguous -- it is the success token an operator greps for.
        say("code fingerprint: all {} recorded fingerprint(s) match this tree "
            "({}), across {} run(s) -- a run records one for the tree that "
            "trained it and one for the tree that scored it, so the two counts "
            "are not the same number."
            .format(len(recorded_fps), live_fp, len(recorded_runs)))
        say()

    if missing:
        say("NOTE: no runs found for: {}".format(", ".join(missing)))
        say()

    # mIoU is what AAGNet and SFRGNN report for semantic segmentation; macro F1
    # is what we track across our own arms. Runs evaluated before mIoU was added
    # show "-" rather than a fabricated value -- re-run evaluate_semantic to fill
    # it in; it needs no retraining.
    say("{:<12}{:>6}{:>9}{:>17}{:>16}{:>16}{:>16}{:>16}".format(
        "arm", "seeds", "n_test", "overall acc", "macro P", "macro R",
        "macro F1", "mIoU"))
    for a in have:
        m = got[a]
        n_seeds = len(m["_seeds"])
        n_test = sorted(set(m["n_samples"]))
        n_test_s = str(n_test[0]) if len(n_test) == 1 else "MIXED"
        # `_metric_n[...]` is how many seeds each cell is really an average
        # of, and `_fmt` refuses to print one that disagrees with the `seeds`
        # column two cells to its left.
        say("{:<12}{:>6}{:>9}{}{}{}{}{}".format(
            a, n_seeds, n_test_s,
            _fmt(m["overall_accuracy"], m["_metric_n"]["overall_accuracy"],
                 n_seeds, "overall_accuracy", a, pct=True, width=17),
            _fmt(m["macro_precision"], m["_metric_n"]["macro_precision"],
                 n_seeds, "macro_precision", a),
            _fmt(m["macro_recall"], m["_metric_n"]["macro_recall"],
                 n_seeds, "macro_recall", a),
            _fmt(m["macro_f1"], m["_metric_n"]["macro_f1"],
                 n_seeds, "macro_f1", a),
            _fmt(m["mean_iou"], m["_metric_n"]["mean_iou"],
                 n_seeds, "mean_iou", a)))
    say()
    if never_recorded:
        say("Columns printed '-' were recorded by NO seed of that arm. (By "
            "SOME but not")
        say("all of them is refused outright -- see summarise_runs.) Re-score "
            "with")
        say("src/evaluation/evaluate_semantic.py to fill them in: it reads the "
            "best-val")
        say("checkpoint and needs no retraining.")
        for a, ks in sorted(never_recorded.items()):
            say("     {:<12} 0 of {} seed(s) recorded: {}".format(
                a, len(got[a]["_seeds"]), ", ".join(ks)))
        say()

    # Differing stopping epochs are expected. The exit code does not
    # depend on them.
    say("Selection: early stop on val loss, patience 10, cap 60;")
    say("best-validation checkpoint evaluated). Differing epochs are expected.")
    for a in have:
        eps = got[a]["_selected_epochs"]
        reasons = got[a]["_stopping_reasons"]
        eps_s = ", ".join(str(e) for e in eps) if eps else "? (not recorded)"
        reason_s = ", ".join(reasons) if reasons else "? (not recorded)"
        say("     {:<12} selected epoch(s) {:<24} reason: {}"
            .format(a, eps_s, reason_s))
    capped = [a for a in have if "cap" in got[a]["_stopping_reasons"]]
    if capped:
        say()
        say("!! CAP AUDIT: {} contain a run that stopped at the cap rather "
            "than early.".format(", ".join(capped)))
        say("   training-regiment.md section 4 says the cap must never be the "
            "stopping")
        say("   reason; a cap-bound arm re-runs its whole comparison at a "
            "raised cap.")
        say("   runners/check_convergence.py owns that verdict, not this "
            "script.")
    say()

    if incomplete:
        say("!! incomplete arms (mean +/- std over fewer than the requested "
            "seeds):")
        for a, ms in incomplete.items():
            say("     {:<12} missing seeds {}".format(a, ms))
        say()

    if len(n_test_all) > 1:
        say("!! arms evaluated on different test sizes {} -- check the "
            "split_file in each".format(sorted(n_test_all)))
        say("   config. Under the selection rule every config must point at")
        say("   datasets/splits/{}_70_10_20_seed42.json.".format(args.dataset))
        say()

    # These sections are NOT suppressed when the run is unsound. The per-class
    # delta view is the only thing this file offers over aggregate_seeds.py.
    # Suppressing it would take away the evidence that tells an operator WHICH
    # fault they are looking at. The mark is what stops a winner being lifted
    # out of this section; see `mark` above.
    # `baseline` was validated against --arms at the top of main(). This line
    # handles the one case left: an arm that WAS asked for and produced no
    # runs. That is already a `problems` entry ("no runs found for arm(s)"),
    # so the run is unsound and every line below carries !UNSOUND! -- the
    # fallback keeps the delta view readable, it does not decide anything.
    base = baseline if baseline in have else have[0]
    others = [a for a in have if a != base]
    bpc = got[base]["per_class_f1"]
    if others and not bpc:
        say("no confusion matrix recorded for {}; skipping per-class deltas."
            .format(base))
        others = []

    for a in others:
        apc = got[a]["per_class_f1"]
        if not apc:
            say("--- {} vs {}: no confusion matrix recorded, skipped ---"
                .format(a, base))
            continue
        support = got[a]["per_class_support"]
        common = sorted(set(bpc) & set(apc))
        deltas = [(apc[c] - bpc[c], c) for c in common]
        wins = sum(1 for d, _ in deltas if d > 0)
        bm, bs = got[base]["macro_f1"]
        am, asd = got[a]["macro_f1"]
        say("--- {} vs {} ---".format(a, base))
        if bm is not None and am is not None:
            say("    macro F1 {:.4f}+/-{:.4f} -> {:.4f}+/-{:.4f}  ({:+.4f})"
                .format(bm, bs, am, asd, am - bm))
        say("    classes improved: {}/{}  (per-class F1 from the SUMMED "
            "confusion matrix)".format(wins, len(common)))
        up = sorted(deltas, reverse=True)[:args.top]
        dn = sorted(deltas)[:args.top]
        say("    largest gains : "
            + (", ".join("c{}:{:+.3f}".format(c, d) for d, c in up if d > 0)
               or "none"))
        say("    largest losses: "
            + (", ".join("c{}:{:+.3f}".format(c, d) for d, c in dn if d < 0)
               or "none"))
        weak = sorted((apc[c], c) for c in common)[:3]
        say("    weakest classes: "
            + ", ".join("c{} F1={:.3f} (n={})".format(c, f, support.get(c, 0))
                        for f, c in weak))
        say()

    # Lands beside the seed sweep it describes, and names its writer and its
    # status. (`out` is resolved at the top of main(): a stale copy of this
    # file is invalidated before the arms are read, so an abort cannot leave
    # one behind claiming to be sound.)
    payload = {
        "written_by": "runners/compare_arms.py",
        "quotable": False,
        # The counterpart of the invalidation stamp at the
        # top of main(): this run reached the end, so the file describes it.
        "complete": True,
        # "quotable" is False on every run of this tool by
        # design, so it says nothing about whether THIS run is even readable as
        # a diagnostic. "sound" is that, and it is the field a later reader of
        # the JSON has to look at before anything else in it.
        "sound": not problems,
        "quotable_tooling": ["runners/aggregate_seeds.py",
                             "runners/check_tier.py"],
        "dataset": args.dataset,
        "results_root": str(args.results),
        "seeds_requested": args.seeds,
        # Which arms this comparison was asked for, and which one the deltas
        # are taken against.
        "arms_requested": arms,
        "baseline": base,
        # The fingerprint verdict travels with the numbers.
        # A reader of this file can tell which tree produced the ranking without
        # having had to watch stdout.
        "code_fingerprint_live": live_fp,
        "code_fingerprint_recorded": recorded_fps,
        "code_fingerprint_stale": stale_fps,
        # Entries above, RUNS here. They are
        # different numbers and the difference is a factor of two.
        "code_fingerprint_stale_runs": stale_runs,
        "code_fingerprint_stale_entry_count": len(stale_fps),
        "code_fingerprint_stale_run_count": len(stale_runs),
        "runs_without_code_fingerprint": unrecorded,
        "fingerprints_match_this_tree": not (stale_fps or unrecorded),
        "problems": problems,
        "selection_scheme": "early stop on val loss, patience 10, "
                            "cap 60, best-val checkpoint",
        "arms": {
            a: {
                "seeds_found": got[a]["_seeds"],
                "seeds_missing": got[a]["_missing_seeds"],
                "selected_epochs": got[a]["_selected_epochs"],
                "stopping_reasons": got[a]["_stopping_reasons"],
                "overall_accuracy_mean": got[a]["overall_accuracy"][0],
                "overall_accuracy_std": got[a]["overall_accuracy"][1],
                "macro_precision_mean": got[a]["macro_precision"][0],
                "macro_recall_mean": got[a]["macro_recall"][0],
                "macro_f1_mean": got[a]["macro_f1"][0],
                "macro_f1_std": got[a]["macro_f1"][1],
                "mean_iou_mean": got[a]["mean_iou"][0],
                "n_samples": sorted(set(got[a]["n_samples"])),
                # How many seeds each mean above is over. Every entry is
                # len(seeds_found) or 0 -- summarise_runs refuses anything
                # between.
                "metric_n": dict(got[a]["_metric_n"]),
            } for a in have
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    say("diagnostic summary written to {}".format(out))

    # Every fault this script can see reaches the exit code, and every one
    # of them is printed here as well as stamped into the JSON.
    if problems:
        say()
        say("=" * 78)
        say("{} PROBLEM(S) -- this comparison is not sound and the arm named "
            "above is".format(len(problems)))
        say("not a finding. Nothing here was quotable in any case (see the "
            "banner); with")
        say("these open it is not usable as a diagnostic either:")
        for pr in problems:
            say("  FAIL  {}".format(pr))
        say("=" * 78)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

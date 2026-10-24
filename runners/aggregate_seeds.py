#!/usr/bin/env python3
"""
aggregate_seeds.py -- turn a tier's per-seed runs into the table the paper quotes.

Reads ``results/<arm>/seed<k>/eval_metrics.json``. Reports mean +/- std over
the seeds. Per-class numbers come from the summed confusion matrix. Every seed
of a corpus scores the same held-out test ids, so each class has one support
count. It exits non-zero when the table fails the checklist.

Per-class IoU in the lower table is the ratio from the summed confusion matrix.
The mIoU column of the main table is the mean of each seed's own mIoU, with a
standard deviation. Those two numbers differ in the fourth decimal. The
markdown caption names both.

Pooling stays, for the reasons that do hold:

* `training-regiment.md` section 1 asks for "per-class tables from the
  seed-averaged confusion matrix"; the summed matrix IS that artifact, so
  pooling is what the requirement says rather than an inference from it.
* A pooled ratio is one estimate taken over every seed's predictions at once.
  A mean of per-seed ratios is a different estimator. For a class a seed never
  predicts, the per-seed ratio is undefined.
* A pooled per-class number has no error bar and must never be printed with one.

That makes TWO mIoUs, computed from the same runs by two
methods, differing in the 4th decimal: `mean_iou_mean` +/- `mean_iou_std` (the
mean over seeds of each seed's own mIoU -- has an error bar, and is the mIoU
column of the main table) and `mean_iou_from_summed_cm` (recomputed from the
summed matrix -- no seed dimension, so it structurally cannot have one, and is
the mIoU of the per-class section). Both are stored. The markdown caption names each one.

Usage
-----
``--arms`` is required.

    python runners/aggregate_seeds.py --dataset cadsynth \
        --arms similarity adjacency both shuffled
    python runners/aggregate_seeds.py --dataset cadsynth --seeds 0 1 2 3 4 \
        --arms resnet_cadsynth_ce resnet_cadsynth_cmca --out results/T3.md
    python runners/aggregate_seeds.py --dataset mfinstseg \
        --arms mfinstseg_unified --out results/summary_T5_mfinstseg.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.metrics import compute_mean_iou          # noqa: E402
from src.utils.provenance import (check_regiment_fields,  # noqa: E402
                                  code_fingerprint)
from runners.check_tier import (_load_run_all,  # noqa: E402
                                headline_metrics, recorded_selection_rule,
                                run_digest, semantic_block)
# When a run does not record patience, use this value. It is the same
# number check_convergence.py uses.
from runners.check_convergence import REGISTERED_PATIENCE  # noqa: E402

#: T1 and T2's four bias arms. Named in the refusal when --arms is omitted.
DEFAULT_ARMS = ("similarity", "adjacency", "both", "shuffled")

#: Heads evaluate_unified.py writes beside semantic.
UNIFIED_HEADS = ("bottom", "instance")

#: The bottom-face head's own metrics, in the order they are aggregated.
BOTTOM_METRICS = ("accuracy", "precision", "recall", "f1", "mean_iou",
                  "iou_bottom", "iou_not_bottom")

#: The per-threshold metrics inside whichever instance row is `reportable`.
#: ``auc`` is NOT here: it is threshold-free and lives on the head itself,
#: which is why it is the number T5 quotes.
INSTANCE_ROW_METRICS = ("precision", "recall", "f1", "accuracy", "threshold")


def _seed_count_note(per_arm_acc: dict) -> str:
    """One sentence stating the seed counts the verdicts below were computed on."""
    counts = {len(v) for v in per_arm_acc.values()}
    if len(counts) == 1:
        n = counts.pop()
        return (f"Computed on **{n} seed(s) per arm**."
                + (" Fewer than the 5 the regiment requires, so these verdicts "
                   "are provisional." if n < 5 else ""))
    lo, hi = min(counts), max(counts)
    return (f"**Seed counts DIFFER across arms ({lo}-{hi}).** Per arm: "
            + ", ".join(f"{a} {len(v)}" for a, v in sorted(per_arm_acc.items()))
            + ". A difference of means between arms of unequal size is weaker "
              "than one between equal arms, and the regiment asks for 5 seeds "
              "everywhere. Treat every verdict below as provisional until the "
              "short arms are completed.")


def arm_dir(results: Path, dataset: str, arm: str) -> Path:
    """Locate an arm's results directory.

    A directory named exactly ``arm`` wins. Otherwise the path is
    ``<dataset>_bias_<arm>``.
    """
    literal = results / arm
    if literal.is_dir():
        return literal
    return results / f"{dataset}_bias_{arm}"


def load_runs(results: Path, dataset: str, arm: str, seeds: list[int],
              link_seeds: list[int] | None = None):
    """Return {seed: metrics} for one arm, and the seeds that are missing.

    Args:
        link_seeds: a list supplied by a caller that undertakes to REPORT link
            runs itself. Seeds whose only metrics file is a link file are
            appended to it and kept OUT of ``missing``. A caller that supplies
            nothing gets `SystemExit` instead -- see the branch below.
            `main` in this module supplies one; `runners/compare_arms.py` and
            `runners/make_figures.py` call this function with four arguments
            and so still abort, which for them is still the right answer: they
            have no half-written artifact of their own to protect.
    """
    base = arm_dir(results, dataset, arm)
    found, missing = {}, []
    for s in seeds:
        run = base / f"seed{s}"
        f = None
        # evaluators name their output differently; the unified one writes
        # unified_eval_metrics.json, so an arm-level aggregate must accept both
        # or T5 silently looks like a tier with no results.
        for name in ("eval_metrics.json", "unified_eval_metrics.json"):
            if (run / name).is_file():
                f = run / name
                break
        if f is not None:
            found[s] = json.loads(f.read_text(encoding="utf-8"))
        elif (run / "link_eval_metrics.json").is_file():
            # A link metrics file is not a missing semantic seed. Without
            # link_seeds the arm is refused. With link_seeds the seed is
            # collected for the link artifact and left out of this table.
            if link_seeds is None:
                raise SystemExit(
                    f"arm {arm!r} seed {s}: {run} holds a link_eval_metrics.json "
                    f"and no semantic metrics file. aggregate_seeds.py builds the "
                    f"SEMANTIC table (accuracy / macro-F1 / mIoU over faces); a "
                    f"link run measures face PAIRS and has no column here. Check "
                    f"the link run with `python runners/check_tier.py`, which reads "
                    f"that layout, and drop this arm from --arms.")
            link_seeds.append(s)
        else:
            missing.append(s)
    return found, missing


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def unified_head(m: dict, head: str):
    """The ``bottom`` / ``instance`` sub-dict of a unified metrics file, or None.

    Selected on the SHAPE of the file, the
    way `check_tier.semantic_block` is: a nested ``semantic`` block is what
    `evaluate_unified.py` writes and what `evaluate_semantic.py` does not, so
    a flat file has no other heads by construction and returns None without
    the caller having to know which evaluator ran. Selecting on the filename
    instead would make `unified_eval_metrics.json` load-bearing, and the one
    thing this repository has learned twice is that a layout test on a name is
    a layout test that is wrong the first time a name changes.
    """
    if not isinstance(m.get("semantic"), dict):
        return None
    block = m.get(head)
    return block if isinstance(block, dict) else None


def _head_seeds(runs: dict, head: str) -> list:
    """Seeds carrying ``head``; refuse the MIXED case, allow all-or-nothing.

    The same rule `summarise_arm` already applies to the confusion matrix and
    to macro_f1/mean_iou, in the same words and for the same reason: a mean
    over 2 of 3 printed in a row whose seed column says 3 carries a false
    claim about its own sample size. All-absent is a different thing -- a
    semantic-only tier has no such head and nothing is fabricated by saying so.
    """
    have = sorted(s for s, m in runs.items() if unified_head(m, head) is not None)
    missing = sorted(set(runs) - set(have))
    if have and missing:
        raise SystemExit(
            f"the {head!r} head is recorded by seed(s) {have} and missing from "
            f"seed(s) {missing}. Aggregating what is present would print a mean "
            f"over {len(have)} of {len(runs)} in a row whose seed count says "
            f"{len(runs)}. Every seed of one cell is scored by the same "
            f"evaluator, so a head on some seeds and not others means the "
            f"missing ones were scored by a different tool or a different "
            f"version of it. Re-score them with "
            f"src/evaluation/evaluate_unified.py -- no retraining needed.")
    return have


def _head_metric(runs: dict, seeds: list, head: str, name: str) -> tuple:
    """(mean, std) of one head metric, refusing a partial average."""
    values = [unified_head(runs[s], head).get(name) for s in seeds]
    present = [v for v in values if v is not None]
    if present and len(present) != len(seeds):
        absent = [s for s, v in zip(seeds, values) if v is None]
        raise SystemExit(
            f"{head}.{name} is recorded by {len(present)} of {len(seeds)} "
            f"seeds and missing from seed(s) {absent}. Same refusal as the "
            f"semantic columns above: a mean over a subset printed under the "
            f"full seed count is a number that misstates its own sample size. "
            f"Re-score those seeds.")
    return mean_std(present)


def summarise_bottom(runs: dict):
    """Seed-averaged bottom-face head, or None when no seed has one."""
    seeds = _head_seeds(runs, "bottom")
    if not seeds:
        return None
    out = {"n_seeds": len(seeds), "seeds": seeds,
           "n_faces": sorted({unified_head(runs[s], "bottom").get("n_faces")
                              for s in seeds} - {None}),
           "n_positive": sorted({unified_head(runs[s], "bottom").get("n_positive")
                                 for s in seeds} - {None})}
    for name in BOTTOM_METRICS:
        m, sd = _head_metric(runs, seeds, "bottom", name)
        out[f"{name}_mean"], out[f"{name}_std"] = m, sd
    return out


def summarise_instance(runs: dict):
    """Seed-averaged instance head, or None when no seed has one.

    ``auc`` is threshold-free and is aggregated unconditionally -- it is T5's
    headline number. The precision/recall/F1 row is aggregated ONLY when every
    seed names the same `reportable` row, that row is not the oracle one, and
    every seed carries it. Which row may be quoted is a fact the evaluator
    records and not one this tool may guess: `oracle_threshold` is fitted on
    the TEST labels and the metrics file's own note says do not report it.
    That is the same refusal `check_tier.link_headline` makes for link runs;
    it is made here rather than imported because the unified layout nests the
    blocks under ``instance`` and that function reads a top-level pair.

    The refusals are RECORDED, not raised: the AUC row is still worth printing
    beside a NOT QUOTABLE banner, and `main` turns each of these into a
    `problems` entry with the seeds named.
    """
    seeds = _head_seeds(runs, "instance")
    if not seeds:
        return None
    blocks = {s: unified_head(runs[s], "instance") for s in seeds}
    auc_m, auc_sd = _head_metric(runs, seeds, "instance", "auc")
    out = {
        "n_seeds": len(seeds), "seeds": seeds,
        "n_pairs": sorted({b.get("n_pairs") for b in blocks.values()} - {None}),
        "n_positive": sorted({b.get("n_positive")
                              for b in blocks.values()} - {None}),
        "auc_mean": auc_m, "auc_std": auc_sd,
        "reportable": sorted({b.get("reportable")
                              for b in blocks.values()} - {None}),
        "seeds_without_reportable": sorted(
            s for s, b in blocks.items() if b.get("reportable") is None),
        "oracle_seeds": sorted(
            s for s, b in blocks.items()
            if b.get("reportable") == "oracle_threshold"),
        "seeds_missing_the_reported_row": [],
        "reported_row": None,
    }
    if (len(out["reportable"]) == 1 and not out["oracle_seeds"]
            and not out["seeds_without_reportable"]):
        row = out["reportable"][0]
        absent = sorted(s for s, b in blocks.items()
                        if not isinstance(b.get(row), dict))
        out["seeds_missing_the_reported_row"] = absent
        if not absent:
            out["reported_row"] = row
            for name in INSTANCE_ROW_METRICS:
                values = [blocks[s][row].get(name) for s in seeds]
                present = [v for v in values if v is not None]
                if present and len(present) != len(seeds):
                    raise SystemExit(
                        f"instance.{row}.{name} is recorded by "
                        f"{len(present)} of {len(seeds)} seeds. A mean over a "
                        f"subset under the full seed count misstates its own "
                        f"sample size; re-score the seeds that are missing it.")
                m, sd = mean_std(present)
                out[f"{name}_mean"], out[f"{name}_std"] = m, sd
    return out


def cap_bound_problems(label: str, prov: dict, patience) -> list:
    """Whether this run is cap-bound, from the selected epoch, patience, and cap.

    Stopping early needs `patience` non-improving epochs after the selected
    epoch, all inside the cap. The label `stopping_reason` is not the decision.

    Args:
        label: ``"arm <a> seed <k>"``, quoted in every message.
        prov: the run's provenance block.
        patience: from the run's own `config.resolved.yaml`, or None when that
            file is absent. None uses the pre-registered patience and the
            message says so.

    Returns:
        A list of problems, empty when the record is self-consistent.
    """
    cap = prov.get("epoch_budget_configured")
    selected = prov.get("selected_epoch")
    reason = prov.get("stopping_reason")
    # Both absences are already `problems` from `check_regiment_fields` and
    # from `seeds_without_cap`; re-reporting them here would double-count.
    if cap is None or selected is None:
        return []
    if selected > cap:
        return [f"{label}: selected_epoch {selected} exceeds the cap {cap}. "
                f"The loop cannot select an epoch it never ran, so one of the "
                f"two numbers does not belong to this run. Re-run the seed."]
    patience_used, source = patience, ""
    if patience_used is None:
        patience_used = REGISTERED_PATIENCE
        source = (f" (this run records no patience of its own, so the "
                  f"pre-registered {REGISTERED_PATIENCE} was used -- the same "
                  f"fallback runners/check_convergence.py makes)")
    could_have_stopped_early = selected + patience_used <= cap
    problems = []
    if reason == "early_stop" and not could_have_stopped_early:
        problems.append(
            f"{label}: labelled early_stop, but selecting epoch {selected} "
            f"under patience {patience_used} would need the loop to reach "
            f"epoch {selected + patience_used}, and the cap is {cap}. Patience "
            f"never ran out, so this run is cap-bound and mislabelled -- a "
            f"truncated run hiding from the cap audit. Its mean is taken over "
            f"a run that had not stopped improving. Re-run the WHOLE "
            f"comparison at a raised cap, never this cell alone."
            f"{source}")
    if reason == "cap" and could_have_stopped_early:
        problems.append(
            f"{label}: labelled cap, but epoch {selected} plus patience "
            f"{patience_used} is {selected + patience_used}, within the cap "
            f"{cap}. Patience should have fired there and ended the run, so "
            f"the label and the loop disagree and neither can be trusted about "
            f"this run.{source}")
    return problems


def summarise_arm(runs: dict) -> dict:
    """Mean and standard deviation across seeds, plus the summed confusion matrix.

    Semantic numbers are read through `check_tier.semantic_block`, which
    accepts the flat layout and the nested ``semantic`` layout.
    """
    acc, f1, miou, cms = [], [], [], []
    f1_by_seed: dict = {}
    miou_by_seed: dict = {}
    for s, m in sorted(runs.items()):
        a, f, i, _n = headline_metrics(m)
        if a is None:
            raise SystemExit(
                f"seed {s}: this metrics file has no overall_accuracy in either "
                f"layout -- not at the top level (flat, evaluate_semantic) and "
                f"not under ['semantic'] (nested, evaluate_unified). Top-level "
                f"keys are {sorted(m)}. Dropping the seed and averaging the rest "
                f"would report a smaller sample as if it were the full one, so "
                f"this stops instead. Re-score the run.")
        acc.append(a)
        f1_by_seed[s] = f
        miou_by_seed[s] = i
        if f is not None:
            f1.append(f)
        if i is not None:
            miou.append(i)
        cm = semantic_block(m).get("confusion_matrix")
        if cm is not None:
            cms.append(np.asarray(cm, dtype=np.int64))

    # A missing macro-F1 or mIoU is not aggregated. The mean stays unset,
    # main records the problem, and the artifact is still written.
    metric_gaps: dict = {}
    for name, by_seed in (("macro_f1", f1_by_seed), ("mean_iou", miou_by_seed)):
        have = sorted(s for s, v in by_seed.items() if v is not None)
        missing = sorted(s for s, v in by_seed.items() if v is None)
        if missing:
            metric_gaps[name] = {"have": have, "missing": missing}
    if "macro_f1" in metric_gaps:
        f1 = []
    if "mean_iou" in metric_gaps:
        miou = []

    summed = None
    if cms:
        if len(cms) != len(runs):
            raise SystemExit(
                f"only {len(cms)} of {len(runs)} seeds carry a confusion matrix. "
                f"The summed matrix would cover a different set of runs from the "
                f"means above it, and the per-class IoU printed beside a 5-seed "
                f"mean would silently be over fewer seeds. Re-score the runs "
                f"that are missing one.")
        shapes = {c.shape for c in cms}
        if len(shapes) != 1:
            raise SystemExit(
                f"confusion matrices differ in shape across seeds: {shapes}. "
                f"The arms are not scoring the same class set; do not aggregate.")
        summed = np.sum(cms, axis=0)

    epochs = {m.get("provenance", {}).get("epoch") for m in runs.values()}
    selected = {(m.get("provenance") or {}).get("selected_epoch")
                for m in runs.values()}
    caps = {(m.get("provenance") or {}).get("epoch_budget_configured")
            for m in runs.values()}

    out = {
        "n_seeds": len(runs),
        "seeds": sorted(runs),
        "accuracy_mean": mean_std(acc)[0], "accuracy_std": mean_std(acc)[1],
        "macro_f1_mean": mean_std(f1)[0], "macro_f1_std": mean_std(f1)[1],
        # This mIoU is the mean over seeds of each seed's own
        # `mean_iou`, so it has a standard deviation and appears in the table
        # with an error bar. `mean_iou_from_summed_cm` below is the OTHER
        # method -- recomputed from the summed confusion matrix, no seed
        # dimension, no error bar -- and the two differ in the 4th decimal
        # because averaging ratios is not the ratio of sums. Both are kept, and
        # `main` now prints a caption saying which is which, so the paper's
        # caption can be copied from the artifact rather than reconstructed.
        "mean_iou_mean": mean_std(miou)[0], "mean_iou_std": mean_std(miou)[1],
        # The unified model's other two
        # heads. None on every flat tier, which is not a fault: T1-T4 and T6
        # score one task and have no such block. See `summarise_bottom` /
        # `summarise_instance`.
        "bottom": summarise_bottom(runs),
        "instance": summarise_instance(runs),
        "epochs": sorted(e for e in epochs if e is not None),
        # provenance["epoch"] is the selected epoch. On a sound run it matches
        # selected_epoch. The last epoch the loop ran is in train.log.
        "caps": sorted(c for c in caps if c is not None),
        "seeds_without_cap": sorted(
            s for s, m in runs.items()
            if (m.get("provenance") or {}).get("epoch_budget_configured") is None),
        "selected_epochs": sorted(e for e in selected if e is not None),
        # Seeds whose stopping_reason label is "cap". Cap-boundness itself
        # is decided by cap_bound_problems from the epoch arithmetic.
        "cap_bound_seeds": sorted(
            s for s, m in runs.items()
            if (m.get("provenance") or {}).get("stopping_reason") == "cap"),
        # provenance epoch and selected_epoch disagree when the scored
        # checkpoint is not the selected one.
        "seeds_with_epoch_ne_selected": sorted(
            s for s, m in runs.items()
            if ((m.get("provenance") or {}).get("epoch") is not None
                and (m.get("provenance") or {}).get("selected_epoch") is not None
                and (m["provenance"]["epoch"]
                     != m["provenance"]["selected_epoch"]))),
        "seeds_without_stopping_reason": sorted(
            s for s, m in runs.items()
            if (m.get("provenance") or {}).get("stopping_reason") is None),
        # Stated class-weight policies and missing policies are kept apart.
        "class_weighting_policies": sorted(
            {p for p in ((m.get("provenance") or {}).get("use_class_weights")
                         for m in runs.values()) if p is not None}),
        "seeds_without_weighting_policy": sorted(
            s for s, m in runs.items()
            if (m.get("provenance") or {}).get("use_class_weights") is None),
        # The same field per SEED. The
        # set above answers "does this table agree with itself?"; naming the
        # runs is what a labelled weighted tier needs, because there the
        # question is not agreement but whether the agreed value is True.
        "use_class_weights_by_seed": {
            s: (m.get("provenance") or {}).get("use_class_weights")
            for s, m in sorted(runs.items())},
        "config_hashes": sorted({m.get("provenance", {}).get("config_hash")
                                 for m in runs.values()} - {None}),
        # Per seed, so a collision can name the runs. A set cannot.
        "config_hash_by_seed": {
            s: (m.get("provenance") or {}).get("config_hash")
            for s, m in sorted(runs.items())},
        "arm_config_hashes": sorted(
            {(m.get("provenance") or {}).get("arm_config_hash")
             for m in runs.values()} - {None, ""}),
        "arm_config_hash_by_seed": {
            s: (m.get("provenance") or {}).get("arm_config_hash")
            for s, m in sorted(runs.items())},
        "seeds_without_arm_config_hash": sorted(
            s for s, m in runs.items()
            if not (m.get("provenance") or {}).get("arm_config_hash")),
        # `run_digest` is check_tier's own -- see the import. Provenance is
        # stripped inside it, which is what makes a relabelled `provenance.seed`
        # irrelevant to the comparison.
        "payload_digest_by_seed": {s: run_digest(m)
                                   for s, m in sorted(runs.items())},
        # Which of macro_f1 / mean_iou could not be aggregated, and why.
        "metric_gaps": metric_gaps,
        "metrics_not_aggregated": sorted(metric_gaps),
        "code_fingerprints": sorted({m.get("provenance", {}).get("code_fingerprint")
                                     for m in runs.values()} - {None}),
        # Scoring fingerprints are separate from the training fingerprints.
        # Seeds that recorded neither are listed, so an empty set is not
        # treated as agreement.
        "eval_code_fingerprints": sorted(
            {((m.get("provenance") or {}).get("evaluated") or {}).get("code_fingerprint")
             for m in runs.values()} - {None}),
        "seeds_without_code_fingerprint": sorted(
            s for s, m in runs.items()
            if not (m.get("provenance") or {}).get("code_fingerprint")),
        "seeds_without_eval_code_fingerprint": sorted(
            s for s, m in runs.items()
            if not (((m.get("provenance") or {}).get("evaluated") or {})
                    .get("code_fingerprint"))),
    }
    if summed is not None:
        out["confusion_matrix_summed"] = summed.tolist()
        miou_s, per_class = compute_mean_iou(summed)
        out["mean_iou_from_summed_cm"] = miou_s
        out["iou_per_class_from_summed_cm"] = {str(k): v for k, v in per_class.items()}
    return out


def _table_title(dataset: str, arms, summary: dict) -> str:
    """The `# ...` heading, written from what was actually aggregated.

    This was the constant f-string
    ``f"# {dataset} bias ablation -- seed-averaged"``, printed over every
    table this tool has ever produced. T4's single weighted ResNet cell is not
    an ablation of anything; T5's unified cell is one model with three heads;
    T3 mixes a CMCA arm, two ResNet baselines and one bias arm. All four were
    titled "bias ablation", and the title is the line a reader carries away
    and the line a caption gets copied from.

    A tier is a bias ablation when every cell it aggregates is a bias cell --
    either one of T1/T2's bare arm names, which `arm_dir` expands to
    ``<dataset>_bias_<arm>``, or a literal cell whose name carries the
    ``_bias_`` infix (T6's four weighted arms do). Anything else is named for
    what it is.
    """
    arms = list(arms)
    is_bias = all(a in DEFAULT_ARMS or "_bias_" in a for a in arms)
    heads = ""
    if any((summary.get(a) or {}).get("instance")
           or (summary.get(a) or {}).get("bottom") for a in arms):
        heads = " (unified model: semantic, bottom-face and instance heads)"
    if is_bias:
        return f"# {dataset} bias ablation -- seed-averaged{heads}"
    if len(arms) == 1:
        return f"# {dataset} / {arms[0]} -- seed-averaged{heads}"
    return (f"# {dataset} -- seed-averaged over {len(arms)} cells: "
            f"{', '.join(arms)}{heads}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="cadsynth | mfinstseg")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="REQUIRED. The arms or cell names to aggregate, e.g. "
                         "`similarity adjacency both shuffled` (T1/T2) or "
                         "`mfinstseg_unified` (T5).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None,
                    help="markdown output (default results/summary_<dataset>.md)")
    args = ap.parse_args(argv)

    results = REPO / args.results

    # --arms is required. Raising here happens before any file is written.
    if args.arms is None:
        found = []
        if results.is_dir():
            found = sorted(
                d.name for d in results.iterdir()
                if d.is_dir() and any(
                    (s / "eval_metrics.json").is_file()
                    or (s / "unified_eval_metrics.json").is_file()
                    or (s / "link_eval_metrics.json").is_file()
                    for s in sorted(d.glob("seed*"))))
        raise SystemExit(
            "--arms is required: this tool will not guess which cells you "
            "meant.\n"
            "It used to default to T1/T2's four bias arms "
            f"({', '.join(DEFAULT_ARMS)}). Those cells are present in every "
            "results root, so the default never failed -- it silently "
            "measured T1 while the operator was aggregating another tier, and "
            "overwrote T1's own summary on the way out, because --out "
            "defaults from --dataset and not from the tier.\n"
            f"Cells with runs under {results}: "
            + (", ".join(found) if found else
               "NONE FOUND -- check --results; it may not be a results root")
            + ".\n"
            "Name the cells of the tier you are aggregating, and give --out a "
            "tier-specific path so one tier's table cannot land on another's, "
            "e.g.\n"
            f"    python runners/aggregate_seeds.py --dataset {args.dataset} "
            f"--arms {found[0] if found else '<cell> [<cell> ...]'} "
            f"--out results/summary_<tier>_{args.dataset}.md\n"
            "`runners/check_tier.py --tier <T>` knows each tier's cell list "
            "and is the tool to ask if you do not.")

    summary, problems = {}, []
    #: {"<arm> seed <k>": patience} from each run's own config.resolved.yaml.
    patience_by_run: dict = {}
    #: Checks that did NOT run, printed under Quotability as NOT PERFORMED.
    #: Kept apart from `problems`: a check that could not be performed is not a
    #: check that failed, and neither one is a check that passed.
    rule_not_performed: list = []

    for arm in args.arms:
        # A link arm is a PROBLEM, not an abort. See the note
        # in `load_runs`: the healthy arms beside it still get their table, the
        # banner at the top of the file says NOT QUOTABLE, and the exit status
        # is 1. That is this script's established contract for every other
        # unusable input, and the link arm now shares it.
        link_seeds: list[int] = []
        runs, missing = load_runs(results, args.dataset, arm, args.seeds,
                                  link_seeds)
        if link_seeds:
            problems.append(
                f"arm {arm}: seeds {sorted(link_seeds)} hold a "
                f"link_eval_metrics.json and no semantic metrics file, under "
                f"{arm_dir(results, args.dataset, arm)}/seed*/. This is the "
                f"SEMANTIC table (accuracy / macro-F1 / mIoU over faces); a "
                f"link run measures face PAIRS and has no column here, so those "
                f"seeds are not missing -- they do not belong in this table at "
                f"all. Check them with `python runners/check_tier.py`, which "
                f"reads that layout, and drop this arm from --arms. The other "
                f"arms are unaffected and their rows below are complete, but "
                f"this file keeps the NOT QUOTABLE banner until --arms is "
                f"corrected: it is not the table that was asked for.")
        if not runs:
            if not link_seeds:
                problems.append(f"arm {arm}: no runs found under "
                                f"{arm_dir(results, args.dataset, arm)}/seed*/")
            continue
        if missing:
            problems.append(f"arm {arm}: missing seeds {missing} "
                            f"(have {sorted(runs)})")
        base = arm_dir(results, args.dataset, arm)
        for s, m in runs.items():
            prov = m.get("provenance") or {}
            gaps = check_regiment_fields(prov) if prov else ["provenance (absent)"]
            if gaps:
                problems.append(f"arm {arm} seed {s}: provenance missing "
                                f"{', '.join(gaps)} -- NOT quotable")
            # The patience half of the selection
            # rule lives only in the run's config.resolved.yaml, which
            # `load_runs` does not read, so it is collected here beside the
            # provenance check that shares the loop.
            # The cap half of the rule is already collected per arm by
            # `summarise_arm` straight from provenance; what only the resolved
            # config can give is patience.
            _cap, patience, rule_errors, not_performed = recorded_selection_rule(
                f"arm {arm} seed {s}", base / f"seed{s}", prov)
            problems.extend(rule_errors)
            if patience is not None:
                patience_by_run[f"{arm} seed {s}"] = patience
            if not_performed:
                rule_not_performed.append(not_performed)
            # Cap-boundness from the
            # arithmetic, not from the label. `patience` is the run's own, or
            # None -- `cap_bound_problems` falls back to the pre-registered
            # value exactly as check_convergence does, and says so.
            problems.extend(
                cap_bound_problems(f"arm {arm} seed {s}", prov, patience))
            # `epoch` and `selected_epoch`
            # are the same number on a sound run -- `build_provenance` is
            # called inside the save-on-improvement branch. A disagreement
            # means a checkpoint was saved outside it, so the weights that
            # were scored are not the ones that were selected. This file used
            # to print the two side by side as if they were different facts,
            # under the words "stopped at" and "selected", and never compared
            # them; `check_convergence` has called that run INCONSISTENT since
            # 22 Sep.
            ep, sel = prov.get("epoch"), prov.get("selected_epoch")
            if ep is not None and sel is not None and ep != sel:
                problems.append(
                    f"arm {arm} seed {s}: provenance epoch={ep} but "
                    f"selected_epoch={sel}. Every trainer writes provenance "
                    f"inside the save-on-improvement branch, so these are the "
                    f"same number on a sound run (0 mismatches across all 85 "
                    f"committed runs). A disagreement means a checkpoint was "
                    f"saved outside that branch and the weights being scored "
                    f"are not the selected ones -- the last-epoch bug decision "
                    f"B removed. Re-run this seed.")
        summary[arm] = summarise_arm(runs)

    # An empty table is still written, then the process exits 2.
    if not summary:
        problems.append(
            f"no arm produced a semantic metrics file, so the table below has "
            f"no rows at all. Requested arms: {', '.join(args.arms)}; seeds "
            f"{args.seeds}; under {results}/. Either the tier has not been run, "
            f"or --results / --arms name the wrong place.")

    # Comparable runs share max_epochs and patience. Stopping epochs differ
    # and are reported, not treated as a fault.
    all_selected = sorted({e for a in summary.values()
                           for e in a["selected_epochs"]})
    all_caps = sorted({c for a in summary.values() for c in a["caps"]})

    # The instance head decides which of its
    # threshold rows may be quoted, and `summarise_instance` records rather
    # than raises so the AUC row still prints. These are the refusals.
    for arm, a in sorted(summary.items()):
        inst = a.get("instance")
        if not inst:
            continue
        if inst["oracle_seeds"]:
            problems.append(
                f"arm {arm}: seed(s) {inst['oracle_seeds']} mark "
                f"'oracle_threshold' as the reportable instance row. That row "
                f"is fitted on the TEST labels and is an upper bound only -- "
                f"the metrics file's own `note` says do not report it. "
                f"src/evaluation/evaluate_unified.py can write only "
                f"'checkpoint_threshold' or 'auc_only', so a file saying "
                f"otherwise was hand-edited or written by something else. "
                f"Re-score those seeds; the AUC above is threshold-free and "
                f"is unaffected, but no precision/recall/F1 row is aggregated "
                f"while this stands.")
        if inst["seeds_without_reportable"]:
            problems.append(
                f"arm {arm}: seed(s) {inst['seeds_without_reportable']} record "
                f"no instance `reportable` key, so there is no way to tell "
                f"which threshold row may be quoted. Choosing one anyway is "
                f"how an oracle number reaches a paper. Re-score those seeds.")
        if len(inst["reportable"]) > 1:
            problems.append(
                f"arm {arm}: the seeds disagree about which instance row is "
                f"reportable {inst['reportable']}. They are one cell scored by "
                f"one evaluator; a disagreement means they were not. No row is "
                f"aggregated until they agree.")
        absent_row = inst["seeds_missing_the_reported_row"]
        if absent_row and len(absent_row) < inst["n_seeds"]:
            # MIXED: some seeds of one cell carry the block their own
            # `reportable` names and some do not. Same rule as mean_iou and
            # the confusion matrix above -- the mixed case is the fatal one,
            # because it is the one that could be averaged over a subset.
            problems.append(
                f"arm {arm}: seed(s) {absent_row} name "
                f"{inst['reportable'][0]!r} reportable and carry no such "
                f"block, while their neighbours in the same cell do. One cell "
                f"is scored by one evaluator; that cannot produce both. The "
                f"file is inconsistent with itself, so no threshold row is "
                f"aggregated. Re-score those seeds.")
        elif absent_row:
            # ALL ABSENT: nothing to average over a subset of, and nothing is
            # fabricated -- the `instance row` column prints NOT AGGREGATED.
            # It is still not a pass, so it goes under NOT PERFORMED, which
            # this file keeps apart from both problems and passes on purpose.
            rule_not_performed.append(
                f"arm {arm}: every seed names {inst['reportable'][0]!r} as the "
                f"reportable instance row and no seed carries that block, so "
                f"the instance precision/recall/F1 row was NOT AGGREGATED and "
                f"the threshold-row check was NOT PERFORMED. The ROC-AUC "
                f"beside it is threshold-free and is unaffected. "
                f"src/evaluation/evaluate_unified.py writes the named block on "
                f"every run, so a file without it was written by something "
                f"else or trimmed after the fact.")

    # Cap-bound arms fail the table. The decision is the epoch arithmetic.
    cap_bound = {arm: a["cap_bound_seeds"]
                 for arm, a in sorted(summary.items()) if a["cap_bound_seeds"]}
    if cap_bound:
        problems.append(
            "arm(s) did not converge -- they stopped at the max_epochs cap with "
            "patience intact, so their means are taken over truncated runs: "
            + "; ".join(f"{arm} seed(s) {seeds}"
                        for arm, seeds in cap_bound.items())
            + ". Under the regiment's cap audit a cap-bound arm re-runs its "
              "WHOLE comparison at a raised cap -- never one arm alone, which "
              "would give that arm a longer budget than the others. "
              "`runners/check_convergence.py --tier <T>` is the detailed audit.")

    no_reason = {arm: a["seeds_without_stopping_reason"]
                 for arm, a in sorted(summary.items())
                 if a["seeds_without_stopping_reason"]}
    if no_reason:
        problems.append(
            "seed(s) record no `stopping_reason`, so this table cannot tell a "
            "converged run from a truncated one: "
            + "; ".join(f"{arm} seed(s) {seeds}"
                        for arm, seeds in no_reason.items())
            + ". The selection rule requires it on every run; a checkpoint left by an "
              "interrupted run has it missing, which is the correct verdict for "
              "an interrupted run. Re-run those seeds through runners/run_all.py.")

    if len(all_caps) > 1:
        problems.append(
            f"arms of this table were held to DIFFERENT epoch caps {all_caps}, "
            f"so they were not selected under one rule and the rows below are "
            f"not comparable. gives every arm of a "
            f"comparison one selection rule -- same training.max_epochs, same "
            f"training.patience. Recorded per arm: "
            + "; ".join(f"{arm} {a['caps']}"
                        for arm, a in sorted(summary.items())) +
            ". Re-run the odd arm under the table's rule; do not extend one arm "
            "alone, which is the unequal-budget confound the selection rule removed.")
    no_cap = {arm: a["seeds_without_cap"] for arm, a in summary.items()
              if a["seeds_without_cap"]}
    if no_cap:
        problems.append(
            f"these seeds record no provenance['epoch_budget_configured'], so "
            f"the cap they were held to is unknown and they cannot be shown to "
            f"share a selection rule with the rest of the table: "
            + "; ".join(f"{arm} seeds {s}" for arm, s in sorted(no_cap.items())) +
            ". src/utils/provenance.py:227 copies that key from "
            "training.max_epochs on every run build_provenance stamps. An "
            "unknown rule is not a matching rule. Re-run those seeds through "
            "runners/run_all.py.")
    all_patiences = sorted(set(patience_by_run.values()))
    if len(all_patiences) > 1:
        problems.append(
            f"arms of this table recorded DIFFERENT patience values "
            f"{all_patiences} in their resolved configs. Patience is half the "
            f"selection rule: two arms capped at the same 60 but stopped on "
            f"patience 5 and patience 20 did not receive the same training. "
            + "; ".join(f"{k} = {v}" for k, v in sorted(patience_by_run.items()))
            + ".")

    # Seeds of one arm must be distinct runs of one experiment.
    for arm, a in sorted(summary.items()):
        by_digest: dict = {}
        for s, dg in sorted(a["payload_digest_by_seed"].items()):
            by_digest.setdefault(dg, []).append(s)
        for dg, dup in sorted(by_digest.items()):
            if len(dup) > 1:
                problems.append(
                    f"arm {arm}: seeds {dup} recorded IDENTICAL metrics -- same "
                    f"confusion matrix, same per-class table, same headline "
                    f"rates (payload digest {dg}, provenance excluded). Two runs "
                    f"of one model on one test set with different seeds cannot "
                    f"agree to the last decimal of every number, so one of these "
                    f"directories is a copy of another and this arm has fewer "
                    f"real runs than seeds. Its row below says {a['n_seeds']} "
                    f"seeds and its standard deviation would be fiction. Re-run "
                    f"the duplicated seed(s) through runners/run_all.py.")

        by_config: dict = {}
        for s, ch in sorted(a["config_hash_by_seed"].items()):
            if ch is None:
                continue        # already reported by check_regiment_fields
            by_config.setdefault(ch, []).append(s)
        for ch, share in sorted(by_config.items()):
            if len(share) > 1:
                problems.append(
                    f"arm {arm}: seeds {share} recorded the SAME config_hash "
                    f"{ch}. That digest covers the resolved config INCLUDING "
                    f"`seed` and `paths` (src/utils/provenance.py:157-172), so "
                    f"the seeds of one arm carry a different value each by "
                    f"construction. Two that agree mean the seed never reached "
                    f"the config -- the runs were launched from one and the same "
                    f"file -- so they are not independent samples and their "
                    f"spread is not an error bar.")

        if a["seeds_without_arm_config_hash"]:
            problems.append(
                f"arm {arm}: seeds {a['seeds_without_arm_config_hash']} record "
                f"NO arm_config_hash. `check_regiment_fields` does not require "
                f"that key, so nothing else in this tool would notice it was "
                f"gone -- and it is the only field that can show these seeds are "
                f"one experiment rather than several. src/utils/provenance.py "
                f"stamps it on every run build_provenance builds, so a run "
                f"without one was not built by it. Re-run the arm through "
                f"runners/run_all.py.")
        if len(a["arm_config_hashes"]) > 1:
            groups = {h: sorted(s for s, v
                                in a["arm_config_hash_by_seed"].items() if v == h)
                      for h in a["arm_config_hashes"]}
            problems.append(
                f"arm {arm}: the seeds of this ONE arm recorded DIFFERENT "
                f"arm_config_hash values {groups}. That digest is the resolved "
                f"config with `seed` and `paths` dropped, so it is identical "
                f"across the seeds of an arm and moves the moment any setting "
                f"that can change a number changes. A disagreement means these "
                f"seeds were not trained on the same experiment, so they may not "
                f"be averaged into one row.")

    # The other half of the same invariant, and it bites harder here than in
    # check_tier: the "Which differences the data supports" section below
    # compares every PAIR of arms in the table, so two arms that are secretly
    # one experiment produce a pairwise verdict about nothing.
    arm_hash_owners: dict = {}
    for arm, a in sorted(summary.items()):
        for h in a["arm_config_hashes"]:
            arm_hash_owners.setdefault(h, set()).add(arm)
    for h, owners in sorted(arm_hash_owners.items()):
        if len(owners) > 1:
            problems.append(
                f"arms {sorted(owners)} of this table recorded the SAME "
                f"arm_config_hash {h}. That digest is the resolved config with "
                f"`seed` and `paths` dropped and nothing else "
                f"(src/utils/provenance.ARM_HASH_EXCLUDED_KEYS), so two arms "
                f"that agree on it were launched from the same experiment -- "
                f"same experiment_name, same corpus, same split. They are one "
                f"arm appearing twice, and the pairwise comparison between them "
                f"further down compares nothing. Check which config each of "
                f"them was actually launched from.")

    # A headline metric missing from some or all seeds is reported, and the
    # column is not averaged.
    for arm, a in sorted(summary.items()):
        for name, gap in sorted(a["metric_gaps"].items()):
            if gap["have"]:
                problems.append(
                    f"arm {arm}: {name} is recorded by seed(s) {gap['have']} and "
                    f"missing from seed(s) {gap['missing']}. Averaging what is "
                    f"present would print a mean over {len(gap['have'])} of "
                    f"{a['n_seeds']} in a row whose seed count says "
                    f"{a['n_seeds']}, with a standard deviation that looks like "
                    f"agreement, so the column reads NOT AGGREGATED instead. "
                    f"Re-score the seeds that are missing it with "
                    f"src/evaluation/evaluate_semantic.py -- no retraining "
                    f"needed -- or pass --seeds without them so the table states "
                    f"the smaller n.")
            else:
                problems.append(
                    f"arm {arm}: {name} is missing from EVERY seed "
                    f"{gap['missing']}, so the column reads NOT AGGREGATED. Both "
                    f"evaluators write that key unconditionally "
                    f"(src/evaluation/evaluate_semantic.py, evaluate_unified.py), "
                    f"so a semantic metrics file without it was hand-edited or "
                    f"written by something else. check_tier.py refuses it too --"
                    f" its mean_iou escape was removed on 26 Sep 2026 -- while "
                    f"this tool used to print the column as a dash and exit 0. "
                    f"Re-score the arm.")

    # One stated class-weight policy across the table. A missing policy is
    # reported separately from a disagreement.
    stated = sorted({p for a in summary.values()
                     for p in a["class_weighting_policies"]})
    unknown = {arm: a["seeds_without_weighting_policy"]
               for arm, a in summary.items()
               if a["seeds_without_weighting_policy"]}
    if len(stated) > 1:
        problems.append(
            f"arms of this table used DIFFERENT class-weighting policies "
            f"{stated}, which the class-weighting rule does not allow: one policy per "
            f"table, plus the labelled weighted tiers, which get a table of "
            f"their own. Recorded per arm: "
            + "; ".join(f"{arm} {a['class_weighting_policies']}"
                        for arm, a in sorted(summary.items())) +
            ". Split the weighted arms into their own --arms run.")
    if stated and unknown:
        problems.append(
            f"some arms of this table state a class-weighting policy {stated} "
            f"and these seeds state none: "
            + "; ".join(f"{arm} seeds {s}" for arm, s in sorted(unknown.items())) +
            ". Unknown is not agreement -- a seed that says nothing is not "
            "agreeing with the ones that do, so the decision-2 one-policy rule "
            "cannot be shown to hold across this table. build_provenance "
            "records the key from loss.use_class_weights "
            "(src/utils/provenance.py:219), so a run without it either had no "
            "`loss:` block or was not stamped by build_provenance.")
    elif unknown and not stated:
        # Every arm is silent. That is normal for a table built from
        # train_unified or train_link, which have no class weighting at all,
        # and a defect for anything else -- and this script cannot tell the two
        # apart, because it knows arms and not trainers. Said out loud and
        # handed to the gate that CAN tell.
        rule_not_performed.append(
            "no run in this table records provenance['use_class_weights'], so "
            "the decision-2 one-policy check was NOT PERFORMED. That is "
            "expected for a table built from train_unified or train_link, "
            "which have no class weighting to state, and a defect for any "
            "other trainer -- this script sees arms, not trainers, and cannot "
            "tell which. `python runners/check_tier.py --tier <T>` reads the "
            "tier table and does tell.")

    # A labelled weighted tier must record use_class_weights true. Agreement
    # among unweighted runs is not enough for that tier.
    _run_all = _load_run_all()
    _cell_tier = {c.name: tier for tier, cells in _run_all.TIERS.items()
                  for c in cells}
    for arm in args.arms:
        tier, a = _cell_tier.get(arm), summary.get(arm)
        if a is None or tier is None or tier not in _run_all.WEIGHTED_TIERS:
            continue
        wrong = {s: v for s, v in a["use_class_weights_by_seed"].items()
                 if v is not True}
        if wrong:
            problems.append(
                f"arm {arm} is a cell of {tier}, which is on "
                f"run_all.WEIGHTED_TIERS "
                f"{sorted(_run_all.WEIGHTED_TIERS)} -- the weighted mirror is "
                f"the entire claim that tier exists to make -- but seed(s) "
                f"{sorted(wrong)} recorded "
                f"provenance['use_class_weights'] = "
                f"{[wrong[s] for s in sorted(wrong)]}, not True. Those runs "
                f"were not weighted, so this table is a table of the unweighted "
                f"experiment under the weighted tier's name. Re-run the cell "
                f"from its weighted config, or aggregate it as what it is.")

    # Printed on every run, pass or fail. A passing footer is exactly where an
    # operator concludes that everything was checked, and three of the four
    # things check_tier asks about class weighting are outside this tool's
    # reach: it reads metrics files, not resolved configs, not trainer sources,
    # and it is handed an --arms list rather than a tier.
    _unknown_cells = [arm for arm in args.arms if arm not in _cell_tier]
    rule_not_performed.append(
        "class weighting, what was NOT checked here: this tool reads each "
        "run's provenance only. It does not read the runs' "
        "config.resolved.yaml, so a config and a provenance that DISAGREE "
        "about `use_class_weights` are not compared; it does not read the "
        "trainer's source, so it cannot tell a cell that has no class "
        "weighting to state (train_unified, train_link) from one that owed a "
        "policy and gave none; and it checks the labelled-weighted-tier rule "
        "only for cells it can find in run_all.TIERS. "
        "`python runners/check_tier.py --tier <T>` does all three."
        + (f" Not found in run_all.TIERS, so unchecked here: "
           f"{_unknown_cells}." if _unknown_cells else ""))

    all_code = sorted({c for a in summary.values() for c in a["code_fingerprints"]})
    if len(all_code) > 1:
        problems.append(f"arms were produced by different code versions: "
                        f"{all_code}. Re-run the odd one out.")

    # Fingerprints are compared with this tree. A mismatch is written into
    # the table and the process exits 1. Raising before the table is written
    # would leave the previous summary on disk.
    live_fp = code_fingerprint()
    for arm, a in sorted(summary.items()):
        if a["seeds_without_code_fingerprint"]:
            problems.append(
                f"arm {arm}: seeds {a['seeds_without_code_fingerprint']} record "
                f"NO code_fingerprint, so there is nothing to compare against "
                f"this tree (live {live_fp}) and they cannot be shown to be "
                f"results of this code. A git commit is not a substitute for "
                f"the fingerprint. Re-run those seeds through runners/run_all.py.")
        if a["seeds_without_eval_code_fingerprint"]:
            problems.append(
                f"arm {arm}: seeds {a['seeds_without_eval_code_fingerprint']} "
                f"have no provenance['evaluated']['code_fingerprint'], so the "
                f"code that SCORED them is unknown. Every evaluator stamps it "
                f"(src/utils/provenance.py:282-287); a metrics file without it "
                f"was not written by one of them. Re-score those seeds.")
    stale_arms = {}
    for arm, a in summary.items():
        recorded = sorted(set(a["code_fingerprints"])
                          | set(a["eval_code_fingerprints"]))
        odd = [c for c in recorded if c != live_fp]
        if odd:
            stale_arms[arm] = odd
    if stale_arms:
        problems.append(
            f"recorded code fingerprints do not match the code in this tree "
            f"(live {live_fp}): " +
            "; ".join(f"{arm} recorded {fps}" for arm, fps in sorted(stale_arms.items())) +
            ". These are the training and scoring fingerprints together: a "
            "number produced by today's weights through yesterday's scorer is "
            "no more a result of this tree than the reverse. Re-run the "
            "affected arms, or run this table from the tree that produced them; "
            "do not quote them as current results.")

    out_path = Path(args.out) if args.out else results / f"summary_{args.dataset}.md"

    def _one_or_span(values, none_text: str) -> str:
        """"60", "19-50", or an explicit "not recorded" -- never a bare [] ."""
        if not values:
            return none_text
        if len(values) == 1:
            return str(values[0])
        return f"{min(values)}-{max(values)}"

    def _one_or_list(values, none_text: str) -> str:
        """A single value, or every distinct one -- for things that must AGREE."""
        if not values:
            return none_text
        if len(values) == 1:
            return str(values[0])
        return str(values)

    lines = [
        _table_title(args.dataset, args.arms, summary),
        "",
    ]
    # The banner sits above the table.
    if problems:
        lines += [
            f"> **NOT QUOTABLE -- {len(problems)} unresolved problem(s).** "
            f"See *Quotability* at the end of this file. The numbers below are "
            f"printed so you can see what changed, not so you can cite them.",
            "",
        ]
    lines += [
        f"Selected epoch(s) {_one_or_span(all_selected, 'NOT RECORDED')} -- "
        f"the epoch whose checkpoint was scored. There is deliberately NO "
        f"stop-epoch column: `provenance['epoch']` holds the SAME number as "
        f"`selected_epoch` (build_provenance is called inside the "
        f"save-on-improvement branch), and the last epoch the loop actually "
        f"ran is not recorded in provenance at all -- it is in each run's "
        f"train.log, which `runners/check_convergence.py` reads and this tool "
        f"does not. Differing selected epochs are expected under early "
        f"stopping and are not a defect; whether any run was CAP-BOUND is "
        f"decided above from the arithmetic (selected + patience vs the cap), "
        f"not from its stopping_reason label.",
        "",
        # The main-table mIoU is the mean of per-seed mIoU. The lower table
        # recomputes mIoU from the summed confusion matrix.
        "**Two mIoU numbers, two methods.** The **mIoU column below** is the "
        "mean over seeds of each seed's own `mean_iou` (`mean_iou_mean` +/- "
        "`mean_iou_std` in the JSON beside this file): one value per seed, so "
        "it carries a standard deviation like every other column. The "
        "**per-class IoU table further down** is recomputed from the "
        "**summed** confusion matrix (`iou_per_class_from_summed_cm`, and its "
        "own mIoU `mean_iou_from_summed_cm`): the seeds' matrices are added "
        "together first and the ratios taken once, which is what "
        "training-regiment.md section 1 means by \"per-class tables from the "
        "seed-averaged confusion matrix\". Summing first is the right choice "
        "per class because that requirement asks for it and because a pooled "
        "ratio is one estimate over every seed's predictions at once, defined "
        "even for a class some seed never predicts. It is **not** justified "
        "by seeds having seen different numbers of examples of a class: they "
        "have not. Every run of a corpus scores the identical held-out set, "
        "so there is exactly one per-class support vector across all 85 runs "
        "of this batch. Pooling collapses the seed dimension, so the "
        "per-class number has **no error bar and cannot be given one**. The "
        "two mIoUs therefore disagree slightly (typically in the 4th "
        "decimal); neither is wrong, and a paper must not present one as a "
        "correction of the other.",
        "",
        "Mean +/- standard deviation across seeds.",
        "",
        "| arm | seeds | accuracy | macro-F1 | mIoU (per-seed mean) |",
        "|---|---|---|---|---|",
    ]
    for arm in args.arms:
        a = summary.get(arm)
        if not a:
            lines.append(f"| {arm} | 0 | - | - | - |")
            continue

        def cell(m, s, pct=False, name=None):
            # NOT AGGREGATED is a refusal. "-" means the metric was not recorded.
            if name is not None and name in a["metrics_not_aggregated"]:
                return "NOT AGGREGATED"
            if m is None:
                return "-"
            return (f"{100*m:.2f}% ± {100*s:.2f}" if pct
                    else f"{m:.4f} ± {s:.4f}")
        lines.append(
            f"| {arm} | {a['n_seeds']} | "
            f"{cell(a['accuracy_mean'], a['accuracy_std'], pct=True)} | "
            f"{cell(a['macro_f1_mean'], a['macro_f1_std'], name='macro_f1')} | "
            f"{cell(a['mean_iou_mean'], a['mean_iou_std'], name='mean_iou')} |")

    # Per-class IoU from the summed confusion matrix.
    summed_arms = [a for a in args.arms
                   if summary.get(a) and "mean_iou_from_summed_cm" in summary[a]]
    if summed_arms:
        class_ids = sorted(
            {k for a in summed_arms
             for k in summary[a]["iou_per_class_from_summed_cm"]},
            key=lambda k: (len(k), k))
        lines += [
            "", "## Per-class IoU (summed confusion matrix)", "",
            "Every number in this section comes from the seeds' confusion "
            "matrices ADDED TOGETHER, with the ratios taken once at the end. "
            "It has no seed dimension, so no column here has an error bar, by "
            "construction rather than by omission. The `mIoU (summed)` column "
            "is `mean_iou_from_summed_cm`; the `mIoU (per-seed mean)` column "
            "repeats the main table's `mean_iou_mean` +/- `mean_iou_std` "
            "beside it so the difference between the two methods is visible "
            "rather than asserted.",
            "",
            "| arm | mIoU (summed) | mIoU (per-seed mean) | "
            + " | ".join(f"IoU class {c}" for c in class_ids) + " |",
            "|---|---|---|" + "---|" * len(class_ids),
        ]
        for arm in summed_arms:
            a = summary[arm]
            per_class = a["iou_per_class_from_summed_cm"]
            seeded = ("-" if a["mean_iou_mean"] is None
                      else f"{a['mean_iou_mean']:.4f} ± {a['mean_iou_std']:.4f}")
            cells = []
            for c in class_ids:
                v = per_class.get(c)
                cells.append("-" if v is None else f"{v:.4f}")
            lines.append(
                f"| {arm} | {a['mean_iou_from_summed_cm']:.4f} | {seeded} | "
                + " | ".join(cells) + " |")

    head_arms = [a for a in args.arms
                 if summary.get(a) and (summary[a]["bottom"]
                                        or summary[a]["instance"])]
    if head_arms:
        lines += [
            "", "## Bottom-face and instance heads (unified model)", "",
            "The table above is the SEMANTIC head only. "
            "`src/evaluation/evaluate_unified.py` scores three tasks from one "
            "model and one forward pass; these are the other two, over the "
            "same seeds and the same held-out set.",
            "",
            "**Instance ROC-AUC is threshold-free** and is the number to "
            "quote for the instance task. The precision/recall/F1 row beside "
            "it depends on a threshold, and WHICH row may be quoted is a fact "
            "the evaluator records in `instance.reportable` -- it is read "
            "here, never guessed. `oracle_threshold` is fitted on the test "
            "labels, is an upper bound only, and is refused: a run marking it "
            "reportable puts this table under the NOT QUOTABLE banner and no "
            "threshold row is aggregated at all.",
            "",
            "| arm | seeds | bottom acc | bottom F1 | bottom mIoU | "
            "instance ROC-AUC | instance row | row F1 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for arm in head_arms:
            a = summary[arm]
            bot, inst = a["bottom"], a["instance"]

            def pair(block, key, pct=False):
                if not block or block.get(f"{key}_mean") is None:
                    return "-"
                m, s = block[f"{key}_mean"], block[f"{key}_std"]
                return (f"{100*m:.2f}% ± {100*s:.2f}" if pct
                        else f"{m:.4f} ± {s:.4f}")
            row_name = (inst or {}).get("reported_row") or "NOT AGGREGATED"
            lines.append(
                f"| {arm} | {a['n_seeds']} | "
                f"{pair(bot, 'accuracy', pct=True)} | {pair(bot, 'f1')} | "
                f"{pair(bot, 'mean_iou')} | {pair(inst, 'auc')} | "
                f"{row_name} | {pair(inst, 'f1')} |")
        lines += [
            "",
            "Bottom-face accuracy is a percentage; every other column is a "
            "ratio. `bottom mIoU` is the mean over seeds of each seed's own "
            "`bottom.mean_iou` and carries an error bar, like the semantic "
            "mIoU column above and unlike the per-class section.",
        ]

    per_arm_acc = {}
    for arm in args.arms:
        # Link seeds are reported by the loop above; collect them again here
        # rather than abort, and let such an arm contribute no accuracies.
        runs, _ = load_runs(results, args.dataset, arm, args.seeds, [])
        # same accessor as summarise_arm; a direct subscript here
        # would raise on T5 after the table above had already been built.
        accs = [a for a in (headline_metrics(m)[0] for m in runs.values())
                if a is not None]
        if len(accs) >= 2:
            per_arm_acc[arm] = accs

    if len(per_arm_acc) >= 2:
        lines += [
            "", "## Which differences the data supports", "",
            "Accuracy, pairwise. **Overlap** means the two arms' seed ranges "
            "intersect, so at least one seed of the weaker arm scored at least "
            "as well as one seed of the stronger. A difference with overlap is "
            "suggestive and should not be stated as established.",
            "",
            "| comparison | n | difference | ranges | verdict |",
            "|---|---|---|---|---|",
        ]
        import itertools as _it
        for a, b in _it.combinations(per_arm_acc, 2):
            xa, xb = per_arm_acc[a], per_arm_acc[b]
            ma, mb = statistics.mean(xa), statistics.mean(xb)
            lo_hi = f"{min(xa)*100:.2f}-{max(xa)*100:.2f} vs {min(xb)*100:.2f}-{max(xb)*100:.2f}"
            overlap = not (min(xa) > max(xb) or min(xb) > max(xa))
            # Each arm uses its own seed count.
            se = (statistics.variance(xa) / len(xa)
                  + statistics.variance(xb) / len(xb)) ** 0.5
            t = abs(ma - mb) / se if se > 0 else float("inf")
            if not overlap and t >= 2.3:
                verdict = "**separated** -- no seed overlap"
            elif t < 1.0:
                verdict = "indistinguishable"
            else:
                verdict = "suggestive only -- ranges overlap"
            n_col = (f"{len(xa)}" if len(xa) == len(xb)
                     else f"**{len(xa)} vs {len(xb)}**")
            lines.append(f"| {a} vs {b} | {n_col} | {100*(ma-mb):+.2f} pts | {lo_hi} | {verdict} |")
        lines += [
            "",
            _seed_count_note(per_arm_acc),
            "Treat `separated` as reportable, `suggestive only` as a remark, and "
            "`indistinguishable` as evidence that the two settings do the same "
            "thing.",
        ]

    lines += ["", "## Quotability (training-regiment.md section 4)", ""]
    if problems:
        lines.append("**This table is NOT quotable.** Unresolved:")
        lines += [f"- {p}" for p in problems]
    else:
        # The pass sentence names the checks above. Stopping epochs are not required to match.
        lines.append("All checks pass: every arm selected under the same rule "
                     "(one training.max_epochs, one training.patience), every "
                     "requested seed present, every seed a DISTINCT run (a "
                     "distinct payload digest and a distinct config_hash) of "
                     "ONE experiment (a single arm_config_hash per arm, and no "
                     "two arms sharing one), every run's provenance complete, "
                     "both headline metrics present on every seed, one "
                     "class-weighting policy across the table, one code "
                     "version, and every run's training AND scoring "
                     f"fingerprint recorded and equal to this tree's "
                     f"({live_fp}). Read the NOT PERFORMED list below before "
                     f"treating this as a full gate.")
    if rule_not_performed:
        # NOT PERFORMED is its own heading. Folding these into "Unresolved"
        # would call them failures, and leaving them out of a passing footer
        # would call them passes; they are neither.
        lines += ["", "**Checks that were NOT PERFORMED** (not failures, and "
                      "not passes -- they did not run):"]
        lines += [f"- {n}" for n in rule_not_performed]
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    (out_path.with_suffix(".json")).write_text(
        json.dumps({"dataset": args.dataset, "seeds": args.seeds,
                    "arms": summary, "problems": problems,
                    # Checks that did not run are recorded in the
                    # machine-readable artifact too, not only in the prose. A
                    # consumer that reads `problems` alone would otherwise see
                    # an empty list and conclude everything was checked.
                    "not_performed": rule_not_performed}, indent=2),
        encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWritten: {out_path}")
    print(f"         {out_path.with_suffix('.json')}")

    if not summary:
        print(f"\nNo runs found at all -- {len(problems)} problem(s) and no "
              f"table. Has the tier been run? The verdict was written to "
              f"{out_path} over whatever it said before.", file=sys.stderr)
        return 2
    if problems:
        print(f"\n{len(problems)} problem(s) -- this table may not be quoted.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

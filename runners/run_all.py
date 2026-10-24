#!/usr/bin/env python3
"""
run_all.py -- the only entry point for numbers that may enter the paper.

`training-regiment.md` section 3 rule 2:

    "runners/run_all.py is the entry point (build it first, task 1.2): given a
     tier name and seed list, it trains + evaluates every cell and writes
     codebase/results/<exp>/seed<k>/metrics.json plus the aggregate table.
     No hand-launched one-off runs for paper numbers."

How a run is made reproducible
------------------------------
For every (cell, seed) this script writes a **fully resolved config** to

    results/<exp>/seed<k>/config.resolved.yaml

with the seed pinned and the checkpoint and figure paths already scoped to that
seed, and then points both the trainer and the evaluator at that file. Two
consequences matter:

1. The five seeds of one arm cannot overwrite each other.
2. Nothing writes into the shared legacy result directories, so the single-seed
   results already committed under results/figures/<arm>/ are left intact.

Note that `--seed` is deliberately NOT passed to the trainer here. The resolved
config already carries both the seed and the scoped paths; passing the flag as
well would scope the paths a second time.

Failure policy
--------------
A failing cell does not stop the sweep -- with 20 runs and a finite GPU budget,
stopping at run 3 wastes the machine. But nothing is swallowed: every failure is
recorded with its exit code and stderr tail, printed in the final summary, and
the process exits non-zero. A cell whose metrics file lacks any regiment
provenance field is reported as UNQUOTABLE even though training succeeded.

FAILED and KILLED are reported as different things
--------------------------------------------------
CPython block-buffers stdout when it is a file rather than a tty.

1. The child is launched with ``-u`` (`child_command`), so a run killed
   mid-epoch leaves a log readable up to the instant it died.
2. A run stopped from OUTSIDE is reported as KILLED, not FAILED
   (`classify_exit`). The two want opposite responses: a FAILED run has a bug
   and its log is where you fix it; a KILLED run has no bug and its log explains
   nothing, because the cause was never inside the process.

Usage
-----
    python runners/run_all.py --list
    python runners/run_all.py --tier T1 --seeds 0 1 2 3 4 --dry-run
    python runners/run_all.py --tier T1 --seeds 0 1 2 3 4
    python runners/run_all.py --tier T1 --seeds 0 1 2 3 4 --force   # re-run finished cells
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.provenance import check_regiment_fields  # noqa: E402


class Cell:
    """One (config, trainer, evaluator) triple -- a single column of a tier."""

    def __init__(self, name, config, trainer, evaluator):
        self.name = name
        self.config = config
        self.trainer = trainer
        self.evaluator = evaluator


def _semantic(name, config):
    return Cell(name, config,
                "src.training.train_semantic", "src.evaluation.evaluate_semantic")


#: The run matrix of training-regiment.md section 2.
#: Guessing an evaluator would produce a number nobody can defend, which is
#: exactly what `resolve_tier` exists to prevent.
TIERS: dict[str, list[Cell]] = {
    "T1": [_semantic(f"cadsynth_bias_{a}", f"configs/cadsynth_bias_{a}.yaml")
           for a in ("similarity", "adjacency", "both", "shuffled")],
    "T2": [_semantic(f"mfinstseg_bias_{a}", f"configs/mfinstseg_bias_{a}.yaml")
           for a in ("similarity", "adjacency", "both", "shuffled")],
    # The evaluator is evaluate_semantic: it dispatches on
    # `model.type == "resnet_encoder"` (evaluate_semantic.py:258), builds a
    # CrossModalAlignmentModel and scores it through UVGridLabelDataset. Same
    # scorer as the graph cells, so the 2x2 table's four cells are measured by
    # one piece of code.
    "T3": [
        _semantic("cadsynth_ce_head", "configs/cadsynth_ce_head.yaml"),
        Cell("resnet_cadsynth_cmca", "configs/resnet_cadsynth.yaml",
             "src.training.train_resnet", "src.evaluation.evaluate_semantic"),
        Cell("resnet_cadsynth_ce", "configs/resnet_cadsynth_ce.yaml",
             "src.training.train_resnet", "src.evaluation.evaluate_semantic"),
    ],
    "T4": [Cell("resnet_cadsynth_ce_weighted",
                "configs/resnet_cadsynth_ce_weighted.yaml",
                "src.training.train_resnet", "src.evaluation.evaluate_semantic")],
    "T5": [Cell("mfinstseg_unified", "configs/mfinstseg_unified.yaml",
                "src.training.train_unified", "src.evaluation.evaluate_unified")],
    # T6 is IN the paper: the weighted mirror of the CADSynth headline table,
    # answering Q5 -- does the topology gain survive the other class-weighting
    # policy? One robustness sentence plus a compact table; the headline stays
    # unweighted.
    "T6": [_semantic(f"cadsynth_bias_{a}_weighted",
                     f"configs/cadsynth_bias_{a}_weighted.yaml")
           for a in ("similarity", "adjacency", "both", "shuffled")],
    # T7 is P3 SCOUTING and never appears in P2. Its configs write under
    # results/scouting/ so the separation is visible on disk, not just in a
    # document. Q6-Q8: does Q5 replicate on MFInstSeg, and does the
    # CMCA-vs-CE advantage depend on the weighting policy?
    "T7": [_semantic(f"mfinstseg_bias_{a}_weighted",
                     f"configs/mfinstseg_bias_{a}_weighted.yaml")
           for a in ("similarity", "adjacency", "both", "shuffled")] + [
        _semantic("cadsynth_ce_head_weighted",
                  "configs/cadsynth_ce_head_weighted.yaml"),
        Cell("resnet_cadsynth_cmca_weighted",
             "configs/resnet_cadsynth_cmca_weighted.yaml",
             "src.training.train_resnet", "src.evaluation.evaluate_semantic"),
    ],
}

#: Tiers whose cells are weighted ON PURPOSE. These are the labelled
#: exceptions; everything else must be unweighted. `check_tier.py` and
#: `tests/test_seeding.py` both read this set.
WEIGHTED_TIERS = frozenset({"T4", "T6", "T7"})

#: Tiers excluded from P2's quotability rule. Nothing here enters the paper.
SCOUTING_TIERS = frozenset({"T7"})

TIER_NOTES = {
    "T1": "CADSynth bias ablation + shuffled control -- the paper's main table",
    "T2": "the same claim on MFInstSeg",
    "T3": "2x2 encoder x objective; the GT+CMCA cell reuses T1's `both` arm",
    "T4": "ResNet footnote: the same CE cell under the other class-weighting policy",
    "T5": "MFInstSeg Unified, the single instance-segmentation row",
    "T6": "weighted mirror of the CADSynth table (in the paper; answers Q5)",
    "T7": "weighted mirror, the rest -- P3 SCOUTING, never in P2, results/scouting/",
}


def resolve_tier(tier: str) -> list[Cell]:
    if tier not in TIERS:
        raise SystemExit(f"unknown tier {tier!r}; known: {', '.join(sorted(TIERS))}")
    cells = TIERS[tier]
    blind = [c.name for c in cells if c.evaluator is None]
    if blind:
        raise SystemExit(
            f"tier {tier} has cells with no evaluator wired up: {', '.join(blind)}.\n"
            f"The ResNet arms train fine but nothing here knows how to score them\n"
            f"into an eval_metrics.json. Wire the evaluator in TIERS before running\n"
            f"this tier, or run {tier}'s graph-encoder cells only.")
    return cells


def check_selection_parity(cells: list[Cell]) -> tuple:
    """Refuse to start a tier whose cells do not share one SELECTION RULE.

    Under early stopping, cells are expected to stop at DIFFERENT epochs; that
    is the design, not a defect. What must match is the rule they were selected
    under: the same cap and the same patience. The selected epoch is reported
    per run in provenance and audited afterwards by the cap audit, which is a
    different question ("did anything hit the ceiling?") asked at a different
    time.
    """
    rules = {}
    for c in cells:
        t = yaml.safe_load((REPO / c.config).read_text(encoding="utf-8")).get("training", {})
        rules[c.name] = (t.get("max_epochs"), t.get("patience"))
    distinct = set(rules.values())
    if len(distinct) != 1:
        raise SystemExit(
            "cells of this tier were configured with DIFFERENT selection rules, "
            "so their results would not be comparable:\n"
            + "\n".join(f"    {k:<32} max_epochs={v[0]} patience={v[1]}"
                        for k, v in rules.items()))
    rule = distinct.pop()
    if rule[0] is None or rule[1] is None:
        raise SystemExit(
            "this tier's configs do not set training.max_epochs and "
            "training.patience. replaced the fixed "
            "epoch budget with one selection rule for every cell; a config "
            "without these was written under the superseded scheme.\n"
            + "\n".join(f"    {k:<32} max_epochs={v[0]} patience={v[1]}"
                        for k, v in rules.items()))
    return rule


def write_resolved_config(cell: Cell, seed: int, run_dir: Path) -> Path:
    """Write the exact config this run uses, with seed and scoped paths baked in."""
    cfg = yaml.safe_load((REPO / cell.config).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = int(seed)
    paths = cfg.setdefault("paths", {})
    paths["checkpoint"] = str((run_dir / "checkpoint.pth").relative_to(REPO)).replace("\\", "/")
    paths["figures"] = str(run_dir.relative_to(REPO)).replace("\\", "/") + "/"
    # Stage-2 keys, harmless when the trainer does not read them.
    if "link_checkpoint" in paths:
        paths["link_checkpoint"] = str((run_dir / "link.pth").relative_to(REPO)).replace("\\", "/")
    if "link_figures" in paths:
        paths["link_figures"] = paths["figures"]

    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.resolved.yaml"
    out.write_text(
        f"# Written by runners/run_all.py -- the exact config of one run.\n"
        f"# base: {cell.config}   cell: {cell.name}   seed: {seed}\n"
        + yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8")
    return out


#: How many header lines `_run` writes into a log before the child starts: the
#: ``$ <command>`` line and one blank line. `child_output_lines` drops them, so
#: that "the log is empty" means "the CHILD wrote nothing", not "the file is".
LOG_HEADER_LINES = 2

#: A non-zero exit whose child produced this many non-blank lines or fewer is
#: KILLED, not FAILED. A Python process that RAISES writes a traceback: the
#: "Traceback (most recent call last):" banner, at least one frame line, and the
#: exception line -- three at an absolute minimum, and in practice far more,
#: since the trainers print a config banner before any model is built. Writing
#: nothing at all means the process never got to unwind.
EMPTY_LOG_MAX_LINES = 2

#: Signals worth naming in a verdict. On POSIX, `subprocess` reports a
#: signal-killed child as a NEGATIVE returncode.
KILL_SIGNALS = {
    2: "SIGINT -- Ctrl-C, or the operator stopping the sweep",
    9: "SIGKILL -- the Linux OOM killer, `kill -9`, or the host reclaiming the box",
    15: "SIGTERM -- `kill`, a job scheduler, or a container shutting down",
}

#: The shell convention for the same event: a child killed by signal N exits
#: 128+N. That is what reaches us when the sweep was launched through a shell
#: -- nohup, sbatch, a notebook cell -- which is how the 85-run batch will be
#: launched, so both forms have to be recognised. 3221225786 == 0xC000013A is
#: the Windows equivalent, STATUS_CONTROL_C_EXIT, which arrives either unsigned
#: or as its signed twin depending on the Python build.
KILL_EXIT_CODES = {128 + n: name for n, name in KILL_SIGNALS.items()}
KILL_EXIT_CODES[3221225786] = "STATUS_CONTROL_C_EXIT -- Ctrl-C on Windows"
KILL_EXIT_CODES[-1073741510] = "STATUS_CONTROL_C_EXIT -- Ctrl-C on Windows"


def child_command(module: str, cfg_path: Path) -> list[str]:
    """The exact argv `_run` launches for *module*.

    ``-u`` is load-bearing and must not be dropped. Without it CPython
    block-buffers stdout whenever it is a file or a pipe rather than a tty, so
    a run killed mid-training loses whatever is still sitting in that buffer --
    which, for a tier that dies before its first flush, is everything. With
    ``-u`` the log is readable up to the instant the process died.
    """
    return [sys.executable, "-u", "-m", module, "--config",
            str(cfg_path.relative_to(REPO)).replace("\\", "/")]


def child_output_lines(log_text: str) -> list[str]:
    """The child's own non-blank lines -- *log_text* minus `_run`'s header."""
    return [ln for ln in log_text.splitlines()[LOG_HEADER_LINES:] if ln.strip()]


def classify_exit(returncode: int, log_text: str) -> tuple[str, str]:
    """Classify one finished child as ``"ok"``, ``"killed"`` or ``"failed"``.

    Returns ``(verdict, reason)``. *reason* is ``""`` for ``"ok"`` and for
    ``"failed"``, whose reason is the traceback already sitting in the log.

    KILLED and FAILED are different events and want opposite responses. A FAILED
    run has a bug in it: its log ends in a traceback, that log is where you fix
    it, and re-running it unchanged just reproduces the bug. A KILLED run has no
    bug: it was stopped from outside -- OOM, a signal, a scheduler, a
    disconnected session -- its log explains nothing because the cause was never
    inside the process, and re-running it after freeing VRAM is often all it
    takes.

    Three things identify a kill:

    * a negative returncode -- POSIX `subprocess` reporting the signal directly;
    * an exit code in `KILL_EXIT_CODES` -- the same signal seen through a shell;
    * a non-zero exit with an empty log -- the process died without unwinding,
      so it never wrote the traceback a genuine failure would have written.
      With ``-u`` in `child_command` an empty log is a real signal: the child
      wrote no traceback because it never unwound.

    What this deliberately CANNOT see
    ---------------------------------
    A Windows ``TerminateProcess`` kill -- Task Manager's "End task", or
    `Popen.kill` -- gives the child exit code 1, which is also what an ordinary
    Python error exit gives, so it is reported FAILED. Windows simply does not
    record that the process was terminated, and guessing from the shape of the
    log would misfile real bugs as kills, which
    is the expensive direction of this error -- it sends the operator to check
    free VRAM instead of reading their own exception.

    The batch runs on Linux, where a kill arrives as -9 or as 137 and IS caught,
    so the gap is on the machine where the operator already knows they pressed
    the button. Say so rather than let the verdict look exhaustive.
    """
    if returncode == 0:
        return "ok", ""
    if returncode < 0:
        sig = -returncode
        named = KILL_SIGNALS.get(sig)
        return "killed", ("killed by signal %d" % sig
                          + (": %s" % named if named else ""))
    if returncode in KILL_EXIT_CODES:
        return "killed", f"exit {returncode} == 128+N: {KILL_EXIT_CODES[returncode]}"
    n = len(child_output_lines(log_text))
    if n <= EMPTY_LOG_MAX_LINES:
        return "killed", (
            f"exit {returncode} after writing {n} line(s) of output. The child "
            f"runs with -u, so this is not buffering: it wrote no traceback "
            f"because it never unwound. Something outside the process stopped "
            f"it -- check the OOM killer (dmesg -T | tail), free VRAM, free "
            f"disk, and the job scheduler.")
    return "failed", ""


def _run(module: str, cfg_path: Path, log: Path) -> tuple[int, str, str, str]:
    """Run one module as a subprocess, streaming its output to *log*.

    Returns ``(returncode, log tail, verdict, reason)``; see `classify_exit`.
    """
    cmd = child_command(module, cfg_path)
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh,
                              stderr=subprocess.STDOUT, text=True)
    text = log.read_text(encoding="utf-8", errors="replace")
    tail = "\n".join(text.splitlines()[-15:])
    verdict, reason = classify_exit(proc.returncode, text)
    return proc.returncode, tail, verdict, reason


def metrics_path(run_dir: Path) -> Path | None:
    """The metrics file an evaluator left in *run_dir*, whatever it named it."""
    for name in ("eval_metrics.json", "unified_eval_metrics.json",
                 "link_eval_metrics.json", "metrics.json"):
        if (run_dir / name).exists():
            return run_dir / name
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", help="T1, T2, T3 or T5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--results", default="results", help="root for run outputs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the resolved paths; run nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells that already have a metrics file")
    ap.add_argument("--list", action="store_true", help="show the tiers and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("Tiers (training-regiment.md section 2):\n")
        for t in sorted(TIERS):
            cells = TIERS.get(t, [])
            print(f"  {t}  {TIER_NOTES[t]}")
            for c in cells:
                mark = "" if c.evaluator else "   [no evaluator wired]"
                print(f"        {c.name:<28} {c.config}{mark}")
            print()
        return 0

    if not args.tier:
        ap.error("--tier is required (or use --list)")

    cells = resolve_tier(args.tier)
    max_epochs, patience = check_selection_parity(cells)
    root = REPO / args.results
    # A scouting tier writes under results/scouting/ so that its
    # exclusion from P2 is visible on disk, not only in a document. The
    # per-run resolved config overwrites `paths`, so routing HERE is what
    # actually decides where the files land -- setting scouting paths in the
    # config alone would be silently ignored.
    if args.tier in SCOUTING_TIERS:
        root = root / "scouting"

    plan = [(c, s) for c in cells for s in args.seeds]
    print("=" * 74)
    print(f"TIER {args.tier} -- {TIER_NOTES[args.tier]}")
    print(f"{len(cells)} cells x {len(args.seeds)} seeds = {len(plan)} runs")
    print(f"selection: early stop on val loss, patience {patience}, "
          f"cap {max_epochs} epochs")
    print("=" * 74)

    if args.dry_run:
        for c, s in plan:
            run_dir = root / c.name / f"seed{s}"
            print(f"  {c.name:<28} seed {s}  ->  "
                  f"{run_dir.relative_to(REPO)}".replace("\\", "/"))
        collisions = len({(c.name, s) for c, s in plan})
        print(f"\n{collisions} distinct output directories for {len(plan)} runs "
              f"-- {'OK' if collisions == len(plan) else 'COLLISION, DO NOT RUN'}")
        return 0

    started = time.time()
    done, failed, killed, unquotable, skipped = [], [], [], [], []

    for i, (cell, seed) in enumerate(plan, 1):
        run_dir = root / cell.name / f"seed{seed}"
        tag = f"[{i}/{len(plan)}] {cell.name} seed {seed}"

        if not args.force and metrics_path(run_dir) is not None:
            print(f"{tag}: already has metrics, skipping (use --force to redo)")
            skipped.append(tag)
            continue

        print(f"\n{tag}: training ...", flush=True)
        cfg_path = write_resolved_config(cell, seed, run_dir)
        t0 = time.time()

        rc, tail, verdict, reason = _run(cell.trainer, cfg_path,
                                         run_dir / "train.log")
        if verdict == "killed":
            print(f"{tag}: TRAINING KILLED (exit {rc})\n    {reason}\n{tail}")
            killed.append((tag, "train", rc, reason, tail))
            continue
        if rc != 0:
            print(f"{tag}: TRAINING FAILED (exit {rc})\n{tail}")
            failed.append((tag, "train", rc, tail))
            continue

        print(f"{tag}: evaluating ...", flush=True)
        rc, tail, verdict, reason = _run(cell.evaluator, cfg_path,
                                         run_dir / "eval.log")
        if verdict == "killed":
            print(f"{tag}: EVALUATION KILLED (exit {rc})\n    {reason}\n{tail}")
            killed.append((tag, "eval", rc, reason, tail))
            continue
        if rc != 0:
            print(f"{tag}: EVALUATION FAILED (exit {rc})\n{tail}")
            failed.append((tag, "eval", rc, tail))
            continue

        mp = metrics_path(run_dir)
        if mp is None:
            print(f"{tag}: evaluator exited 0 but wrote no metrics file into "
                  f"{run_dir}")
            failed.append((tag, "no-metrics", 0, ""))
            continue

        metrics = json.loads(mp.read_text(encoding="utf-8"))
        prov = metrics.get("provenance", {})
        missing = check_regiment_fields(prov) if prov else ["provenance (absent)"]
        mins = (time.time() - t0) / 60
        if missing:
            print(f"{tag}: done in {mins:.1f} min but UNQUOTABLE -- "
                  f"missing provenance: {', '.join(missing)}")
            unquotable.append((tag, missing))
        else:
            print(f"{tag}: done in {mins:.1f} min, provenance complete")
        done.append(tag)

    # ---- summary --------------------------------------------------------- #
    print("\n" + "=" * 74)
    print(f"TIER {args.tier} SUMMARY  ({(time.time() - started) / 60:.1f} min)")
    print("=" * 74)
    print(f"  completed    {len(done)}")
    print(f"  skipped      {len(skipped)}  (already had metrics)")
    print(f"  failed       {len(failed)}  (the run raised; its log has the traceback)")
    print(f"  killed       {len(killed)}  (stopped from outside; its log explains nothing)")
    print(f"  unquotable   {len(unquotable)}  (trained, but provenance incomplete)")
    for tag, stage, rc, tail in failed:
        print(f"\n  FAILED {tag} during {stage} (exit {rc})")
        for line in tail.splitlines()[-6:]:
            print(f"      {line}")
    for tag, stage, rc, reason, tail in killed:
        print(f"\n  KILLED {tag} during {stage} (exit {rc})")
        print(f"      {reason}")
        for line in tail.splitlines()[-6:]:
            print(f"      {line}")
    for tag, missing in unquotable:
        print(f"  UNQUOTABLE {tag}: missing {', '.join(missing)}")

    if killed:
        print(f"\n{len(killed)} of {len(plan)} runs were KILLED rather than failing on their\n"
              f"own. Do not go hunting for the bug in those logs -- there is none in\n"
              f"them. Find what stopped the process (OOM killer, VRAM, disk, the job\n"
              f"scheduler, a closed session), then re-run those cells with --force.")
    if failed or killed or unquotable:
        print("\nThis tier is NOT complete. Nothing here may enter the paper until\n"
              "every run above is completed with full provenance.")
        return 1
    if len(done) + len(skipped) != len(plan):
        print("\nFewer runs finished than planned.")
        return 1
    dataset = {"T1": "cadsynth", "T2": "mfinstseg", "T3": "cadsynth",
               "T5": "mfinstseg", "T6": "cadsynth",
               "T7": "mfinstseg"}.get(args.tier, "<dataset>")
    print(f"\nTier {args.tier} complete: {len(plan)} runs, all with provenance.\n"
          f"Next: python runners/aggregate_seeds.py --dataset {dataset}\n"
          f"      (writes results/summary_{dataset}.md; exits non-zero if the\n"
          f"       table fails the regiment's section-4 checklist).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

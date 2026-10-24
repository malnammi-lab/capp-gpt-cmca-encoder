"""`aggregate_seeds.py` must refuse every tree `check_tier.py` refuses.

WHY THIS FILE EXISTS
--------------------
The 26 Sep 2026 gate wave hardened `runners/check_tier.py` and left
`runners/aggregate_seeds.py` alone. That is the wrong way round for a defect
of this shape: `check_tier` prints a verdict to a terminal, while
`aggregate_seeds` WRITES THE PAPER'S TABLE -- `results/summary_*.md` is the
artifact `scripts/RUNBOOK-tiers.md` names and the file whose numbers are
copied into the paper. A fault the table does not see is a fault that reaches
print.

THE MEASUREMENT (26 Sep 2026)
-----------------------------
Scratch copies of the COMMITTED tree (T1's four cadsynth bias cells, and T6's
four weighted ones), one fault injected per copy, both tools run over the same
copy, nothing under `codebase/results` opened for writing. Every run's
`code_fingerprint` and `provenance.evaluated.code_fingerprint` were first
re-stamped to the live tree's, so the known fingerprint drift of the live tree
is not present and the injected fault is the ONLY fault. `check_tier` was run
as `--tier T1` / `--tier T6`, i.e. both of its audits.

    fault injected                      aggregate_seeds       check_tier
    (T1) nothing -- baseline            rc 0 all checks pass  rc 0 SOUND
    foreign code_fingerprint, 1 run     rc 1, 2 problems      rc 1, 2 FAILs
    a missing seed                      rc 1, 1 problem       rc 1, 3 FAILs
    seed4 := byte-copy of seed0         rc 0 ALL CHECKS PASS  rc 1, 2 FAILs  MISS
    two seeds sharing a config_hash     rc 0 ALL CHECKS PASS  rc 1, 1 FAIL   MISS
    foreign arm_config_hash, 1 seed     rc 0 ALL CHECKS PASS  rc 1, 1 FAIL   MISS
    early_stop label, cap-bound epoch   rc 1, 1 problem       rc 1, 2 FAILs
    macro_f1 deleted from 1 of 5 seeds  rc 1, SystemExit (*)  rc 1, 1 FAIL
    mean_iou deleted from 1 of 5 seeds  rc 1, SystemExit (*)  rc 1, 1 FAIL
    macro_f1 deleted from ALL 5 seeds   rc 0 ALL CHECKS PASS  rc 1, 5 FAILs  MISS
    mean_iou deleted from ALL 5 seeds   rc 0 ALL CHECKS PASS  rc 1, 5 FAILs  MISS
    (T6) nothing -- baseline            rc 0 all checks pass  rc 0 SOUND
    T6, every run says unweighted       rc 0 ALL CHECKS PASS  rc 1, 2 FAILs  MISS

SIX false passes. Which verdict an operator got depended on which command they
ran, and `scripts/RUNBOOK-tiers.md` names the one that passed.

(*) Same verdict, but by `SystemExit` raised from `summarise_arm` BEFORE the
artifact was written -- a 391-byte stderr message and no table. That is what
this script's own "WHY THIS REPORTS RATHER THAN RAISES" paragraph forbids and
calls "the rule for the whole script": the previous `summary_<dataset>.md`
then survives on disk still saying "All checks pass", and it is the file
people read. See `test_a_metric_missing_from_some_seeds_is_reported_not_raised`.

After the fix, the same thirteen trees: every fault rc 1 from BOTH tools, both
baselines rc 0 from both.

WHAT IS STILL NARROWER HERE, ON PURPOSE
---------------------------------------
The weighted-tier row is closed by reading `run_all.TIERS` and
`run_all.WEIGHTED_TIERS` -- the project's single statement of which cell
belongs to which tier -- and NOT by re-deriving check_tier's weighting logic.
Three parts of that logic stay check_tier's, because they need files this tool
does not open: the runs' `config.resolved.yaml` (so a config and a provenance
that disagree are not compared here), the trainer's source (so a cell that has
no class weighting to state cannot be told from one that owed a policy), and
the tier's full cell list (this tool aggregates the `--arms` it is given). The
tool now PRINTS all three as NOT PERFORMED on every run, pass or fail, so an
operator reading a passing footer is not told that a check ran when it did
not. See `test_a_table_that_states_a_policy_still_says_what_it_did_not_check`.
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

from runners import aggregate_seeds, check_tier          # noqa: E402
from src.utils.provenance import code_fingerprint        # noqa: E402

#: T1's four cells, named exactly as `runners/run_all.TIERS["T1"]` names them,
#: so one synthetic tree can be handed to BOTH tools -- `check_tier` reads the
#: tier table and `aggregate_seeds` reads `--arms`.
T1_CELLS = ("cadsynth_bias_similarity", "cadsynth_bias_adjacency",
            "cadsynth_bias_both", "cadsynth_bias_shuffled")

#: T4's one cell. T4 is on `run_all.WEIGHTED_TIERS`.
T4_CELL = "resnet_cadsynth_ce_weighted"

SEEDS = (0, 1, 2, 3, 4)

#: Every run of a corpus scores the identical frozen test partition, so this is
#: constant across the tree; a differing count is check_tier's
#: different-test-sizes fault and not the one under test.
N_TEST_FACES = 34338

CM = [[900, 40], [60, 800]]


def _arm_hash(cell: str) -> str:
    """Constant across an arm's seeds, distinct between arms -- as on disk."""
    return hashlib.sha256(f"arm:{cell}".encode()).hexdigest()[:16]


def _config_hash(cell: str, seed: int) -> str:
    """One digest per RUN: `seed` is inside the config that is hashed."""
    return hashlib.sha256(f"cfg:{cell}:seed{seed}".encode()).hexdigest()[:16]


def metrics(cell: str, seed: int, *, weights: bool = False) -> dict:
    """What `src/evaluation/evaluate_semantic.py` writes: one task, flat.

    Accuracy varies with the seed so the five payloads are DISTINCT (the
    duplicate-seed check) and the spread stays inside check_tier's window
    (`SEED_SPREAD_EPS` < spread <= 0.15). mIoU stays below macro-F1, which
    check_tier requires of the same predictions.
    """
    fp = code_fingerprint()
    return {
        "experiment_name": cell,
        "n_classes": 2,
        "n_samples": N_TEST_FACES,
        "overall_accuracy": 0.900 + 0.002 * seed,
        "macro_precision": 0.71,
        "macro_recall": 0.69,
        "macro_f1": 0.700 + 0.001 * seed,
        "mean_iou": 0.550 + 0.001 * seed,
        "iou_per_class": {"0": 0.49, "1": 0.61},
        "per_class": {},
        "confusion_matrix": CM,
        "provenance": {
            "seed": seed,
            "epoch": 38,
            "selected_epoch": 38,
            "stopping_reason": "early_stop",
            "use_class_weights": weights,
            "epoch_budget_configured": 60,
            "config_hash": _config_hash(cell, seed),
            "arm_config_hash": _arm_hash(cell),
            "code_fingerprint": fp,
            "git_commit": None,
            "gpu": "Tesla T4",
            "wall_time_s": 931.0,
            "torch": "2.8.0+cu128",
            "python": "3.12.11",
            "platform": "Linux-6.8.0-1063-aws-x86_64-with-glibc2.39",
            "recorded_at": "2026-09-25T18:36:32+00:00",
            "evaluated": {"config_hash": _config_hash(cell, seed),
                          "code_fingerprint": fp, "gpu": "Tesla T4",
                          "at": "2026-09-25T18:43:48+00:00"},
        },
    }


def write_tree(root: Path, cells=T1_CELLS, *, weights: bool = False) -> Path:
    """A complete, sound tier: every cell, every seed, resolved configs."""
    for cell in cells:
        for s in SEEDS:
            run = root / cell / f"seed{s}"
            run.mkdir(parents=True, exist_ok=True)
            (run / "eval_metrics.json").write_text(
                json.dumps(metrics(cell, s, weights=weights), indent=2),
                encoding="utf-8")
            (run / "config.resolved.yaml").write_text(yaml.safe_dump({
                "experiment_name": cell,
                "seed": s,
                "loss": {"use_class_weights": weights},
                "training": {"max_epochs": 60, "patience": 10},
            }), encoding="utf-8")
    return root


def load(root: Path, cell: str, seed: int) -> dict:
    return json.loads((root / cell / f"seed{seed}" / "eval_metrics.json")
                      .read_text(encoding="utf-8"))


def save(root: Path, cell: str, seed: int, m: dict) -> None:
    (root / cell / f"seed{seed}" / "eval_metrics.json").write_text(
        json.dumps(m, indent=2), encoding="utf-8")


def aggregate(root: Path, out: Path, *, cells=T1_CELLS, dataset="cadsynth"):
    """Run the aggregator over the tree; return (rc, markdown, json payload)."""
    rc = aggregate_seeds.main(["--dataset", dataset,
                               "--seeds", *[str(s) for s in SEEDS],
                               "--results", str(root), "--out", str(out),
                               "--arms", *cells])
    return (rc, out.read_text(encoding="utf-8"),
            json.loads(out.with_suffix(".json").read_text(encoding="utf-8")))


def matching(items, *needles) -> list:
    return [i for i in items if all(n in i for n in needles)]


# ===================================================================== #
#  the tree both tools must agree is SOUND                              #
# ===================================================================== #

def test_the_unmutated_tree_passes_both_tools(tmp_path):
    """REGRESSION GUARD (green before and after).

    Without this, every assertion below could be satisfied by a tool that
    refuses everything.
    """
    root = write_tree(tmp_path / "results")

    rc, md, payload = aggregate(root, tmp_path / "s.md")
    errors, _warns = check_tier.check("T1", root, list(SEEDS))

    assert rc == 0, payload["problems"]
    assert "All checks pass" in md
    assert errors == [], errors


def test_the_distinctness_digest_is_check_tiers_own():
    """One implementation of "are these two runs the same run?", not two.

    A second copy here is a copy that goes stale, which is the defect this
    whole file is about: two gates drifting apart on one question.
    """
    assert aggregate_seeds.run_digest is check_tier.run_digest


# ===================================================================== #
#  fault 3 -- seed4 is a byte-copy of seed0, `provenance.seed` relabelled
# ===================================================================== #

def test_a_duplicated_seed_is_a_problem(tmp_path):
    """RED before the fix: rc 0 on the synthetic tree, and on the real tree 1
    problem (the pre-existing fingerprint drift) with no mention of the
    duplicate -- while check_tier reported `seeds [0, 4] recorded IDENTICAL
    metrics`.

    The arm then prints a five-seed error bar over four real runs, which is
    the one number in the table a reader cannot check.
    """
    root = write_tree(tmp_path / "results")
    dup = load(root, "cadsynth_bias_both", 0)
    dup["provenance"]["seed"] = 4
    save(root, "cadsynth_bias_both", 4, dup)

    rc, md, payload = aggregate(root, tmp_path / "s.md")
    errors, _ = check_tier.check("T1", root, list(SEEDS))

    assert matching(errors, "IDENTICAL metrics"), errors      # the reference
    assert rc == 1, payload["problems"]
    assert matching(payload["problems"], "IDENTICAL", "[0, 4]"), \
        payload["problems"]
    assert "All checks pass" not in md


# ===================================================================== #
#  fault 4 -- two seeds of one arm sharing a `config_hash`              #
# ===================================================================== #

def test_two_seeds_sharing_a_config_hash_is_a_problem(tmp_path):
    """RED before the fix: `summarise_arm` collected `config_hashes` into a
    SET and `main` never read it, so two seeds launched from one and the same
    resolved config were averaged as independent samples.
    """
    root = write_tree(tmp_path / "results")
    m = load(root, "cadsynth_bias_both", 1)
    m["provenance"]["config_hash"] = _config_hash("cadsynth_bias_both", 0)
    save(root, "cadsynth_bias_both", 1, m)

    rc, md, payload = aggregate(root, tmp_path / "s.md")
    errors, _ = check_tier.check("T1", root, list(SEEDS))

    assert matching(errors, "SAME config_hash"), errors        # the reference
    assert rc == 1, payload["problems"]
    assert matching(payload["problems"], "config_hash", "[0, 1]"), \
        payload["problems"]
    assert "All checks pass" not in md


# ===================================================================== #
#  fault 5 -- a foreign / absent `arm_config_hash`                      #
# ===================================================================== #

def test_a_foreign_arm_config_hash_is_a_problem(tmp_path):
    """RED before the fix: the string `arm_config_hash` did not occur in
    aggregate_seeds.py at all, so seeds from two different experiments were
    averaged into one row.
    """
    root = write_tree(tmp_path / "results")
    m = load(root, "cadsynth_bias_both", 2)
    m["provenance"]["arm_config_hash"] = "deadbeefdeadbeef"
    save(root, "cadsynth_bias_both", 2, m)

    rc, _md, payload = aggregate(root, tmp_path / "s.md")
    errors, _ = check_tier.check("T1", root, list(SEEDS))

    assert matching(errors, "arm_config_hash"), errors         # the reference
    assert rc == 1, payload["problems"]
    assert matching(payload["problems"], "arm_config_hash",
                    "deadbeefdeadbeef"), payload["problems"]


def test_a_seed_with_no_arm_config_hash_is_a_problem(tmp_path):
    """Absence is not agreement. `check_regiment_fields` does not require this
    key (`src/utils/provenance.py` lists `config_hash`, not the arm one), so
    nothing else in this tool would have noticed.
    """
    root = write_tree(tmp_path / "results")
    m = load(root, "cadsynth_bias_both", 2)
    del m["provenance"]["arm_config_hash"]
    save(root, "cadsynth_bias_both", 2, m)

    rc, _md, payload = aggregate(root, tmp_path / "s.md")
    errors, _ = check_tier.check("T1", root, list(SEEDS))

    assert matching(errors, "NO arm_config_hash"), errors      # the reference
    assert rc == 1, payload["problems"]
    assert matching(payload["problems"], "arm_config_hash", "[2]"), \
        payload["problems"]


def test_two_arms_sharing_an_arm_config_hash_is_a_problem(tmp_path):
    """The other half of the invariant: five seeds share it, two arms must not.

    Two arms that agree on it were launched from the same experiment, so the
    "which differences the data supports" section below the table would be
    comparing an arm with itself.
    """
    root = write_tree(tmp_path / "results")
    for s in SEEDS:
        m = load(root, "cadsynth_bias_shuffled", s)
        m["provenance"]["arm_config_hash"] = _arm_hash("cadsynth_bias_both")
        save(root, "cadsynth_bias_shuffled", s, m)

    rc, _md, payload = aggregate(root, tmp_path / "s.md")
    errors, _ = check_tier.check("T1", root, list(SEEDS))

    assert matching(errors, "SAME arm_config_hash"), errors    # the reference
    assert rc == 1, payload["problems"]
    assert matching(payload["problems"], "arm_config_hash",
                    "cadsynth_bias_both"), payload["problems"]


# ===================================================================== #
#  fault 7 -- a headline metric deleted                                 #
# ===================================================================== #

@pytest.mark.parametrize("metric", ["macro_f1", "mean_iou"])
def test_a_metric_missing_from_every_seed_is_a_problem(tmp_path, metric):
    """RED before the fix: the ALL-ABSENT case printed the column as `-` and
    exited 0.

    aggregate_seeds.py's own docstring records the measurement: "deleting it
    from ALL FIVE seeds passed BOTH tools and printed the table's mIoU column
    as '-'". check_tier's half was fixed on 26 Sep (`optional = ("mean_iou",)`
    removed); this half was not. Both evaluators write both keys
    unconditionally, so a semantic run without one was hand-edited.
    """
    root = write_tree(tmp_path / "results")
    for s in SEEDS:
        m = load(root, "cadsynth_bias_both", s)
        del m[metric]
        save(root, "cadsynth_bias_both", s, m)

    rc, md, payload = aggregate(root, tmp_path / "s.md")
    errors, _ = check_tier.check("T1", root, list(SEEDS))

    assert matching(errors, f"{metric} absent"), errors        # the reference
    assert rc == 1, payload["problems"]
    assert matching(payload["problems"], metric, "EVERY seed"), \
        payload["problems"]
    assert "All checks pass" not in md
    # and nothing is fabricated in its place
    assert payload["arms"]["cadsynth_bias_both"][f"{metric}_mean"] is None


@pytest.mark.parametrize("metric", ["macro_f1", "mean_iou"])
def test_a_metric_missing_from_some_seeds_is_reported_not_raised(tmp_path,
                                                                 metric):
    """RED before the fix: `SystemExit` from `summarise_arm`, raised before a
    single line of the artifact was written.

    That is the failure mode aggregate_seeds.py's own "WHY THIS REPORTS RATHER
    THAN RAISES" paragraph exists to forbid, and the paragraph states it is
    "the rule for the whole script": the previous `summary_<dataset>.md`
    survives on disk with its "All checks pass" footer, and it is the file
    people read. The verdict has to be written OVER it.

    The refusal itself is not softened -- rc is still 1, the banner still says
    NOT QUOTABLE, and no partial average is printed.
    """
    root = write_tree(tmp_path / "results")
    stale = tmp_path / "s.md"
    stale.write_text("# stale table\n\nAll checks pass: nothing was wrong.\n",
                     encoding="utf-8")
    m = load(root, "cadsynth_bias_both", 2)
    del m[metric]
    save(root, "cadsynth_bias_both", 2, m)

    rc, md, payload = aggregate(root, stale)

    assert rc == 1, payload["problems"]
    assert "All checks pass" not in md, "the stale verdict survived"
    assert matching(payload["problems"], metric, "[0, 1, 3, 4]", "[2]"), \
        payload["problems"]
    # the mean over the four seeds that DO have it is never printed
    assert payload["arms"]["cadsynth_bias_both"][f"{metric}_mean"] is None
    assert "NOT AGGREGATED" in md


# ===================================================================== #
#  fault 8 -- the check this tool legitimately cannot make              #
# ===================================================================== #

def test_a_labelled_weighted_tier_that_is_not_weighted_is_a_problem(tmp_path):
    """RED before the fix: rc 0, "All checks pass", and the passing footer said
    "one class-weighting policy across the table" -- which was TRUE and beside
    the point. A tier whose configs and provenance unanimously say `false`
    agrees with itself perfectly.

    Measured on a scratch copy of the committed T6 with every run's
    `use_class_weights` and `config.resolved.yaml` set to false:
    `check_tier --tier T6` rc 1, 2 FAILs; `aggregate_seeds` rc 0, ALL CHECKS
    PASS.

    This tool is handed `--arms` and not a tier, which was taken to mean it
    could not know. It can: the cells are NAMED, and `run_all.TIERS` says
    which tier each belongs to. Reading that table is not a second copy of
    check_tier's weighting logic -- the halves that need the resolved config
    and the trainer's source stay check_tier's, and are printed below as NOT
    PERFORMED.
    """
    root = write_tree(tmp_path / "results", cells=(T4_CELL,), weights=False)

    rc, md, payload = aggregate(root, tmp_path / "s.md", cells=(T4_CELL,))
    errors, _ = check_tier.check("T4", root, list(SEEDS))

    assert matching(errors, "LABELLED WEIGHTED"), errors       # the reference
    assert rc == 1, payload["problems"]
    assert matching(payload["problems"], "WEIGHTED_TIERS", T4_CELL), \
        payload["problems"]
    assert "All checks pass" not in md
    # ...and the halves it still cannot make are named rather than implied
    hits = matching(payload["not_performed"], "weighting")
    assert hits, payload["not_performed"]
    assert "check_tier.py --tier" in " ".join(hits), hits
    assert "NOT PERFORMED" in md


def test_a_table_that_states_a_policy_still_says_what_it_did_not_check(tmp_path):
    """The same line on the healthy T1 tree: a passing footer is where an
    operator is most likely to conclude that everything was checked.
    """
    root = write_tree(tmp_path / "results")

    rc, md, payload = aggregate(root, tmp_path / "s.md")

    assert rc == 0, payload["problems"]
    assert "All checks pass" in md
    hits = matching(payload["not_performed"], "weighting")
    assert hits, payload["not_performed"]
    # the three halves that are check_tier's, each named
    joined = " ".join(hits)
    for owed in ("config.resolved.yaml", "trainer", "run_all.TIERS",
                 "check_tier.py --tier"):
        assert owed in joined, (owed, joined)
    assert "NOT PERFORMED" in md


# ===================================================================== #
#  the agreement matrix itself                                          #
# ===================================================================== #

def _dup_seed(root):
    m = load(root, "cadsynth_bias_both", 0)
    m["provenance"]["seed"] = 4
    save(root, "cadsynth_bias_both", 4, m)


def _shared_config_hash(root):
    m = load(root, "cadsynth_bias_both", 1)
    m["provenance"]["config_hash"] = _config_hash("cadsynth_bias_both", 0)
    save(root, "cadsynth_bias_both", 1, m)


def _foreign_arm_hash(root):
    m = load(root, "cadsynth_bias_both", 2)
    m["provenance"]["arm_config_hash"] = "deadbeefdeadbeef"
    save(root, "cadsynth_bias_both", 2, m)


def _foreign_fingerprint(root):
    m = load(root, "cadsynth_bias_both", 2)
    m["provenance"]["code_fingerprint"] = "deadbeefdeadbeef"
    save(root, "cadsynth_bias_both", 2, m)


def _missing_seed(root):
    (root / "cadsynth_bias_both" / "seed3" / "eval_metrics.json").unlink()


def _cap_bound_mislabelled(root):
    m = load(root, "cadsynth_bias_both", 1)
    m["provenance"]["epoch"] = 59
    m["provenance"]["selected_epoch"] = 59          # cap 60, patience 10
    m["provenance"]["stopping_reason"] = "early_stop"
    save(root, "cadsynth_bias_both", 1, m)


def _drop_macro_f1_everywhere(root):
    for s in SEEDS:
        m = load(root, "cadsynth_bias_both", s)
        del m["macro_f1"]
        save(root, "cadsynth_bias_both", s, m)


def _drop_mean_iou_everywhere(root):
    for s in SEEDS:
        m = load(root, "cadsynth_bias_both", s)
        del m["mean_iou"]
        save(root, "cadsynth_bias_both", s, m)


@pytest.mark.parametrize("inject", [
    _foreign_fingerprint, _missing_seed, _dup_seed, _shared_config_hash,
    _foreign_arm_hash, _cap_bound_mislabelled, _drop_macro_f1_everywhere,
    _drop_mean_iou_everywhere,
], ids=["foreign_fingerprint", "missing_seed", "duplicated_seed",
        "shared_config_hash", "foreign_arm_config_hash",
        "cap_bound_mislabelled", "macro_f1_deleted", "mean_iou_deleted"])
def test_both_gates_refuse_every_fault(tmp_path, inject):
    """One tree, one fault, two tools, one verdict.

    Four of these eight were refused by `check_tier` and passed by the tool
    that writes the table. Which verdict an operator got depended on which
    command they ran, and `scripts/RUNBOOK-tiers.md` names this one.
    """
    root = write_tree(tmp_path / "results")
    inject(root)

    rc, md, payload = aggregate(root, tmp_path / "s.md")
    # BOTH of check_tier's audits, which is what `check_tier.py --tier T1`
    # runs and what the measurement in this file's docstring used. `check`
    # alone is not the reference: the cap-bound-mislabelled fault is caught by
    # `cap_audit` (it reads the resolved configs and the stopping records) and
    # `check` returns [] for it, so a probe that asked only `check` would have
    # called this tool's correct refusal a false positive.
    errors, _ = check_tier.check("T1", root, list(SEEDS))
    cap_errors, _ = check_tier.cap_audit("T1", root, list(SEEDS))
    errors = errors + cap_errors

    assert errors, "check_tier did not refuse it either -- the probe is wrong"
    assert rc == 1, payload["problems"]
    assert payload["problems"], payload
    assert "NOT QUOTABLE" in md

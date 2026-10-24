"""`runners/compare_arms.py` writes over evidence it did not produce.

GATE DEFECT, 26 Sep 2026. The tool resolved its output to ONE constant path,
``results/<dataset>_bias_comparison.json``, from ``--dataset`` alone -- and
stamped an invalidation stub there BEFORE reading a single arm. Two harms, both
reproduced against the committed T1 artifact before this file was written:

* **Cross-tier.** ``--arms resnet_cadsynth_ce resnet_cadsynth_cmca`` (T6) wrote
  its comparison into ``cadsynth_bias_comparison.json`` -- T1's committed,
  ``"sound": true`` bias ablation. The file went from
  ``arms = [adjacency, both, shuffled, similarity]`` to
  ``arms = [resnet_cadsynth_ce, resnet_cadsynth_cmca]``.
* **Aborted runs.** ``--arms nosuch_arm_a nosuch_arm_b`` found nothing, printed
  "No eval_metrics.json found", returned 1 -- and left that same committed file
  holding an eight-key stub. NOTHING was read, so nothing about the arms could
  have justified touching it.

The 26 Sep ``--arms`` fix made the tool reachable on T3-T6 for the first time,
which WIDENS this: every tier it can now address is a tier it can now overwrite
T1's and T2's artifacts from.

A third defect rides along: the banner said ``BIAS ABLATION -- <dataset>`` for
every run, so a T6 ResNet comparison announced itself as T1's bias ablation.

MEASURED against a pre-fix tree (this file's own reverted copy of the tool,
with the constant output path, no ``--out``/``--force``, the constant banner
and neither `comparison_filename` nor `owner_of`): **13 of these 17 tests fail
there and 4 pass**. The 4 that pass are the regression guards --
``..._still_...`` plus `test_baseline_is_validated_against_the_arms_given`,
which checks item 4 of the brief and was already correct before this fix. They
are not evidence of the fix; they are what stops it breaking
`scripts/batch/tier-cycle.sh:97`, `scripts/batch/verify-tier.js:105` and
`README.md:153`, all of which call

    python runners/compare_arms.py --dataset <ds> --results results

with no ``--arms`` and expect ``results/<ds>_bias_comparison.json``. Each is
marked as a guard in its own docstring. (`scripts/RUNBOOK-tiers.md` calls this
tool NOWHERE -- `grep -n compare scripts/RUNBOOK-tiers.md` matches only
`launch-tier.sh`'s fingerprint pre-flight.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CODEBASE = Path(__file__).resolve().parents[1]
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from runners import compare_arms                        # noqa: E402
from src.utils.provenance import code_fingerprint       # noqa: E402

#: Every arm must score the same held-out set or the different-test-sizes
#: fault fires instead of the defect under test.
N_TEST_FACES = 34338
CM = [[900, 40], [60, 800]]

#: T6's two ResNet cells: real directory names from `results/`, and the pair
#: the reproduction used. They contain no `_bias_` infix, so `arm_dir`
#: resolves them as literal directories.
OTHER_CELLS = ("resnet_cadsynth_ce", "resnet_cadsynth_cmca")

BIAS_JSON = "cadsynth_bias_comparison.json"


def _metrics(name: str, seed: int, acc: float, fp: str) -> dict:
    return {
        "experiment_name": name,
        "checkpoint": "checkpoints/{}/seed{}/best.pt".format(name, seed),
        "n_classes": 2,
        "n_samples": N_TEST_FACES,
        "overall_accuracy": acc,
        "macro_precision": 0.71,
        "macro_recall": 0.69,
        "macro_f1": 0.70,
        "mean_iou": 0.55,
        "confusion_matrix": CM,
        "provenance": {
            "seed": seed,
            "epoch": 42,
            "selected_epoch": 38,
            "stopping_reason": "early_stop",
            "use_class_weights": False,
            "epoch_budget_configured": 60,
            "config_hash": "c0ffeec0ffeec0ff",
            "gpu": "Tesla T4",
            "code_fingerprint": fp,
            "evaluated": {"code_fingerprint": fp,
                          "at": "2026-09-26T00:00:00+00:00"},
        },
    }


def _write(root: Path, dirname: str, seeds, acc0: float, fp: str) -> None:
    for i, s in enumerate(seeds):
        run = root / dirname / "seed{}".format(s)
        run.mkdir(parents=True, exist_ok=True)
        (run / "eval_metrics.json").write_text(
            json.dumps(_metrics(dirname, s, acc0 + 0.002 * i, fp)),
            encoding="utf-8")


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A results root holding BOTH T1's four bias arms and T6's two cells."""
    root = tmp_path / "results"
    fp = code_fingerprint()
    for i, arm in enumerate(compare_arms.ARMS):
        _write(root, "cadsynth_bias_{}".format(arm), (0, 1, 2),
               0.90 + 0.01 * i, fp)
    for i, cell in enumerate(OTHER_CELLS):
        _write(root, cell, (0, 1, 2), 0.76 + 0.01 * i, fp)
    return root


def _run(root: Path, *extra: str) -> int:
    return compare_arms.main(["--dataset", "cadsynth", "--seeds", "0", "1", "2",
                              "--results", str(root), *extra])


def _make_committed_bias_comparison(root: Path) -> bytes:
    """Run the documented invocation, and return the bytes it committed."""
    assert _run(root) == 0, "the bias ablation itself must be sound here"
    committed = (root / BIAS_JSON).read_bytes()
    assert json.loads(committed)["sound"] is True
    assert sorted(json.loads(committed)["arms"]) == sorted(compare_arms.ARMS)
    return committed


# --------------------------------------------------------------------- #
#  THE DEFECT: another tier's run lands on T1's committed artifact       #
# --------------------------------------------------------------------- #

def test_a_comparison_of_other_arms_does_not_overwrite_the_bias_comparison(tree):
    """A comparison of other arms leaves the committed bias-comparison file unchanged."""
    committed = _make_committed_bias_comparison(tree)

    rc = _run(tree, "--arms", *OTHER_CELLS, "--baseline", OTHER_CELLS[0])
    assert rc == 0, "the ResNet pair is complete and this tree's -- it is sound"

    after = (tree / BIAS_JSON).read_bytes()
    assert after == committed, (
        "a comparison of {} overwrote T1's committed bias ablation. Its arms "
        "are now {}.".format(list(OTHER_CELLS),
                             sorted(json.loads(after).get("arms", {}))))


def test_the_other_comparison_is_written_somewhere_and_names_its_arms(tree):
    """It must still produce an artifact -- just not on top of another one."""
    _make_committed_bias_comparison(tree)
    assert _run(tree, "--arms", *OTHER_CELLS, "--baseline", OTHER_CELLS[0]) == 0

    written = [p for p in tree.glob("*.json") if p.name != BIAS_JSON]
    assert len(written) == 1, (
        "expected exactly one new comparison artifact, found {}"
        .format([p.name for p in written]))
    doc = json.loads(written[0].read_text(encoding="utf-8"))
    assert sorted(doc["arms"]) == sorted(OTHER_CELLS)
    assert doc["sound"] is True


def test_an_aborted_run_does_not_stub_another_comparisons_artifact(tree):
    """Nothing was read, so nothing justified touching a different tier's file.

    Reproduced before the fix: `--arms nosuch_arm_a nosuch_arm_b` returned 1
    and left `cadsynth_bias_comparison.json` holding the INCOMPLETE RUN stub.
    """
    committed = _make_committed_bias_comparison(tree)

    rc = _run(tree, "--arms", "nosuch_arm_a", "nosuch_arm_b",
              "--baseline", "nosuch_arm_a")
    assert rc == 1, "no runs found is exit 1"

    after = (tree / BIAS_JSON).read_bytes()
    assert after == committed, (
        "an aborted run that read NO arm replaced T1's committed comparison. "
        "It now says: {}".format(
            json.loads(after).get("problems", ["<no problems key>"])[0][:80]))


def test_out_names_the_destination(tree):
    """There was no way at all to say where the artifact should go."""
    dest = tree.parent / "scratch" / "t6_comparison.json"
    assert _run(tree, "--arms", *OTHER_CELLS, "--baseline", OTHER_CELLS[0],
                "--out", str(dest)) == 0
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert sorted(doc["arms"]) == sorted(OTHER_CELLS)


def test_out_refuses_to_land_on_a_comparison_of_different_arms(tree):
    """`--out` must not become a new way to do the same damage."""
    committed = _make_committed_bias_comparison(tree)

    with pytest.raises(SystemExit) as exc:
        _run(tree, "--arms", *OTHER_CELLS, "--baseline", OTHER_CELLS[0],
             "--out", str(tree / BIAS_JSON))
    assert "--force" in str(exc.value)
    assert (tree / BIAS_JSON).read_bytes() == committed


def test_force_is_the_way_to_overwrite_on_purpose(tree):
    """The refusal is a guard, not a wall: an operator who means it may say so."""
    _make_committed_bias_comparison(tree)
    assert _run(tree, "--arms", *OTHER_CELLS, "--baseline", OTHER_CELLS[0],
                "--out", str(tree / BIAS_JSON), "--force") == 0
    doc = json.loads((tree / BIAS_JSON).read_text(encoding="utf-8"))
    assert sorted(doc["arms"]) == sorted(OTHER_CELLS)


# --------------------------------------------------------------------- #
#  THE BANNER: titled from what was compared                            #
# --------------------------------------------------------------------- #

def test_the_banner_does_not_call_a_resnet_comparison_a_bias_ablation(tree, capsys):
    _run(tree, "--arms", *OTHER_CELLS, "--baseline", OTHER_CELLS[0])
    out = capsys.readouterr().out
    assert "BIAS ABLATION" not in out, (
        "a comparison of {} announced itself as a bias ablation"
        .format(list(OTHER_CELLS)))
    heading = [ln for ln in out.splitlines() if "ARM COMPARISON" in ln]
    assert len(heading) == 1, "expected one heading, got {}".format(heading)
    for cell in OTHER_CELLS:
        assert cell in heading[0], "the heading does not name {}".format(cell)
    # The seed note survives the split onto its own line; it is what tells a
    # reader the table is a mean and not one run.
    assert len([ln for ln in out.splitlines() if "seed-averaged" in ln]) == 1


# --------------------------------------------------------------------- #
#  REGRESSION GUARDS -- these pass BEFORE the fix too. See module head.  #
# --------------------------------------------------------------------- #

def test_the_documented_invocation_still_writes_the_documented_path(tree):
    """GUARD for tier-cycle.sh:97 / verify-tier.js:105 / README.md:153.

    Line numbers checked against the files on 26 Sep 2026. `tier-cycle.sh` is
    owned elsewhere, so confirm with `grep -n compare_arms` rather than
    trusting `:97` if it has moved.
    """
    assert _run(tree) == 0
    doc = json.loads((tree / BIAS_JSON).read_text(encoding="utf-8"))
    assert doc["sound"] is True
    assert sorted(doc["arms"]) == sorted(compare_arms.ARMS)
    assert [p.name for p in tree.glob("*.json")] == [BIAS_JSON]


def test_the_bias_arms_still_get_the_bias_ablation_banner(tree, capsys):
    """GUARD: the heading of the default invocation.

    `paper-content/results-T1.md:306-310` and `results-T2.md:371-375` quote
    this tool's stdout tables and name this invocation as the way to reproduce
    them. Neither quotes the banner line itself -- what this guards is the
    phrase an operator greps a captured tier log for.
    """
    _run(tree)
    out = capsys.readouterr().out
    assert ("BIAS ABLATION -- cadsynth   (seed-averaged, seeds requested "
            "[0, 1, 2])") in out


def test_rerunning_the_same_comparison_still_refreshes_its_own_artifact(tree):
    """GUARD: the 22 Sep invalidation stamp applies to OUR OWN file, and a
    second run of the same comparison must not be refused as an overwrite."""
    _make_committed_bias_comparison(tree)
    assert _run(tree) == 0
    assert json.loads((tree / BIAS_JSON).read_text(encoding="utf-8"))["sound"]


def test_baseline_is_validated_against_the_arms_given(tree):
    """GUARD for the 26 Sep defect-9 fix: `choices=ARMS` is gone.

    This passes before this gate fix as well -- it is the check for item 4 of
    the brief, and the answer is that item 4 was ALREADY correct.
    """
    with pytest.raises(SystemExit) as exc:
        _run(tree, "--arms", *OTHER_CELLS, "--baseline", "similarity")
    assert "not one of the arms being compared" in str(exc.value)
    assert _run(tree, "--arms", *OTHER_CELLS,
                "--baseline", OTHER_CELLS[1]) == 0


# --------------------------------------------------------------------- #
#  THE GUARD'S OWN BRANCHES                                              #
#                                                                        #
#  `owner_of` is the whole of what stands between this tool and another  #
#  tier's evidence, and `comparison_filename` is what keeps two          #
#  comparisons off one path. Neither had a test of its own.              #
# --------------------------------------------------------------------- #

def test_a_foreign_json_at_out_is_refused_and_left_alone(tree):
    """Anything this tool did not write is evidence to somebody."""
    _make_committed_bias_comparison(tree)
    foreign = tree / "someone_elses.json"
    foreign.write_text(json.dumps({"written_by": "runners/aggregate_seeds.py",
                                   "arms": {"similarity": {}}}),
                       encoding="utf-8")
    before = foreign.read_bytes()

    with pytest.raises(SystemExit) as exc:
        _run(tree, "--arms", *OTHER_CELLS, "--baseline", OTHER_CELLS[0],
             "--out", str(foreign))
    msg = str(exc.value)
    assert "refusing to write over" in msg
    assert "aggregate_seeds" in msg, "the refusal must say WHY: {}".format(msg)
    assert foreign.read_bytes() == before


def test_an_unparseable_file_at_out_is_refused_not_replaced(tree):
    """A file we cannot identify is the one we are least entitled to destroy."""
    junk = tree / "not_json.json"
    junk.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        _run(tree, "--arms", *OTHER_CELLS, "--baseline", OTHER_CELLS[0],
             "--out", str(junk))
    assert "not valid JSON" in str(exc.value)
    assert junk.read_text(encoding="utf-8") == "{ this is not json"


def test_our_own_aborted_stub_may_be_replaced(tree):
    """The new refusal must not be so strict it bricks its own path.

    The stub is written before any arm is read, so it names no arms and cannot
    be matched by arm list. `owner_of` has an explicit branch for it; without
    that branch one aborted run would make the tool's own crash permanent.
    This is NOT one of the four that pass pre-fix -- it drives `--out`, which
    did not exist there -- it guards the fix against overshooting.
    """
    dest = tree / "t6.json"
    dest.write_text(json.dumps({
        "written_by": compare_arms.WRITTEN_BY,
        "quotable": False, "sound": False, "complete": False,
        "dataset": "cadsynth", "seeds_requested": [0, 1, 2],
        "invalidated_at": "2026-09-26T00:00:00+00:00",
        "problems": ["INCOMPLETE RUN."],
    }), encoding="utf-8")

    assert _run(tree, "--arms", *OTHER_CELLS, "--baseline", OTHER_CELLS[0],
                "--out", str(dest)) == 0
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["complete"] is True
    assert sorted(doc["arms"]) == sorted(OTHER_CELLS)


def test_derived_names_separate_every_distinct_comparison():
    """Two arm lists share a filename only when they are the same list."""
    f = compare_arms.comparison_filename
    lists = [
        list(compare_arms.ARMS),
        list(OTHER_CELLS),
        list(reversed(OTHER_CELLS)),
        ["similarity", "both"],
        ["resnet_cadsynth_ce"],
        # long enough to force the digest branch
        ["cell_with_a_very_long_name_{}".format(i) for i in range(6)],
    ]
    names = [f("cadsynth", a) for a in lists]
    assert len(set(names)) == len(names), dict(zip(names, map(str, lists)))
    assert names[0] == "cadsynth_bias_comparison.json"
    # the digest branch stays a legal, bounded path component
    assert len(names[-1]) < 120, names[-1]
    assert names[-1].endswith(".json")


def test_two_spellings_of_one_arm_share_a_name_and_the_guard_says_so():
    """MEASURED, not assumed: `similarity` and `cadsynth_bias_similarity` are
    one directory (`aggregate_seeds.arm_dir` prefers a literal dir, and
    `results/similarity` does not exist), so `comparison_filename` deriving ONE
    name for both is correct, not a collision.

    `owner_of` still compares the spellings as recorded, because if
    `results/similarity` ever did exist the two WOULD be different directories.
    So the second spelling is refused rather than silently overwriting -- and
    the refusal has to name the cause, or an operator reads it as two different
    comparisons of arms that are plainly the same.
    """
    f = compare_arms.comparison_filename
    assert (f("cadsynth", ["similarity", "both"])
            == f("cadsynth", ["cadsynth_bias_similarity", "cadsynth_bias_both"]))

    doc = {"written_by": compare_arms.WRITTEN_BY,
           "arms_requested": ["cadsynth_bias_similarity", "cadsynth_bias_both"]}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        ours, why = compare_arms.owner_of(p, "cadsynth", ["similarity", "both"])
    assert ours is False, "the stricter answer must stand"
    assert "different spelling" in why, why
    assert "--force" in why, why


def test_only_the_default_tuple_claims_the_documented_name():
    """A SUBSET or SUPERSET of the bias arms is not the bias ablation."""
    f = compare_arms.comparison_filename
    documented = "cadsynth_bias_comparison.json"
    assert f("cadsynth", list(compare_arms.ARMS)) == documented
    assert f("cadsynth", list(compare_arms.ARMS)[:3]) != documented
    assert f("cadsynth", list(compare_arms.ARMS) + ["ce_head"]) != documented
    # ...and the dataset is still part of the name, so the two datasets'
    # comparisons cannot land on each other.
    assert f("mfinstseg", list(OTHER_CELLS)) != f("cadsynth", list(OTHER_CELLS))

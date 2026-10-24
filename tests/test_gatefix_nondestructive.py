"""A verification tool must not damage the evidence it verifies.

`runners/check_convergence.py` is a READ of the results tree: it opens every
run's metrics file, its resolved config and its train.log, and concludes
whether the tier was truncated by its epoch budget. Nothing about that question
requires writing anything.

It wrote anyway. Until this file existed, every invocation rewrote
`<results>/convergence_<tier>.md` in place, with no backup and no prompt, so
merely AUDITING a committed tier destroyed the committed audit of that tier --
the one artifact that recorded what the batch itself concluded. The damage is
silent and total: the replaced file is the only copy, and the replacement looks
exactly like it except for a timestamp. `paper-content/results-T3.md:700-709`
records a verification agent discovering this and working around it by hand
(`--report <scratch>/convergence_T3.regen.md`); `results-T4.md:1083` and
`results-T6.md:961` record the same workaround being applied again for T4 and
T6. Three records, three hand-applied workarounds, because the default was
wrong.

This file pins the default shut, and pins the two escapes open:

    (no flag)     write the report only if it would CREATE the file; refuse,
                  without reading anything, if it would replace one
    --no-report   audit and print, write nothing at all -- the read-only mode
                  every verification agent on this project actually needed
    --force       replace an existing report, on purpose, having said so

Second theme, same tool, different risk. `check_convergence` audits the CAP and
nothing else: it exits 0 on a tier whose runs carry a stale `code_fingerprint`,
and on a tier whose metrics are missing outright, because neither is a
statement about the epoch budget. That scope is defensible and it is kept
(widening it would give this tool a second verdict that can disagree with
`check_tier`'s, which its own module docstring refuses for exactly this
reason). What is NOT defensible is a bare green line: it is the second gate the
tier records run, so an operator who runs it alone reads "IS NOT
BUDGET-CONFOUNDED" as "the tier is sound". `weekly/WEEK6-SWEEP-FINDINGS.md:245`
records two paper-facing documents already making that mistake in print. So the
tool now states what it does not check, on stdout and in the report, on pass as
well as on failure -- and the tests below fail if that disclosure goes away or
stops naming the tool that does own each question.

Test-policy category 3 (gate-tool verdicts), `advisor-guidance/
test-policy-2026-09-23.md:15`. Every test here was confirmed RED against the
unfixed `check_convergence.py` before the fix was written.

Fixtures are local to this file, the same convention
`tests/test_check_convergence.py` follows.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

CODEBASE = Path(__file__).resolve().parents[1]
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, CODEBASE / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cc():
    return _load("check_convergence_nd", "runners/check_convergence.py")


@pytest.fixture(scope="module")
def run_all():
    return _load("run_all_for_nondestructive_test", "runners/run_all.py")


def t1_arms(run_all):
    return [c.name for c in run_all.TIERS["T1"]]


def write_run(root, arm, seed, *, selected_epoch=22, reason="early_stop",
              cap=60, patience=10, fingerprint="9d88e9213dbb9164",
              metrics=None):
    """One `results/<arm>/seed<k>/` in the shape `run_all.py` lays out.

    `metrics=<dict>` replaces the metric block wholesale, which is how the
    deleted-metric case below is built. `fingerprint` is recorded in
    provenance exactly as the trainers record it, so a tier can be given a
    fingerprint that matches no tree.
    """
    run_dir = Path(root) / arm / "seed{}".format(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    prov = {
        "seed": seed,
        "epoch": selected_epoch,
        "selected_epoch": selected_epoch,
        "stopping_reason": reason,
        "use_class_weights": False,
        "epoch_budget_configured": cap,
        "config_hash": "0123456789abcdef",
        "code_fingerprint": fingerprint,
        "git_commit": None,
        "gpu": "Tesla T4",
        "wall_time_s": 1234.5,
    }
    payload = {"experiment_name": arm, "overall_accuracy": 0.9607,
               "macro_f1": 0.88, "mean_iou": 0.81, "n_samples": 34338}
    if metrics is not None:
        payload = dict(metrics)
    payload["provenance"] = prov
    (run_dir / "eval_metrics.json").write_text(
        json.dumps(payload), encoding="utf-8")
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump({"experiment_name": arm, "seed": seed,
                        "training": {"max_epochs": cap, "patience": patience,
                                     "batch_size": 4}}, sort_keys=False),
        encoding="utf-8")
    return run_dir


def build_tier(root, arms, seeds=(0, 1, 2, 3, 4), **kw):
    for arm in arms:
        for s in seeds:
            write_run(root, arm, s, **kw)


#: What a refusal exits. NOT 2: `check_convergence.EXIT_REPORT_EXISTS`, the
#: code that means the audit DECLINED to run and read nothing, as against 2
#: (the invocation was wrong) and 1 (the tier failed the cap audit). While
#: this was 2, scripts/batch/tier-cycle.sh scored a healthy tier as a gate
#: failure and stopped the whole batch -- see the module docstring of
#: tests/test_gatefix_convergence_exit.py, which pins all four codes.
REFUSED = 3


COMMITTED = (
    "# Cap audit -- tier T1\n\nGenerated 2026-09-25T19:36:00+00:00 by "
    "`runners/check_convergence.py`.\nThis is the BATCH's own report. If a "
    "later audit replaced it, that replacement is the defect.\n")


def snapshot(root) -> dict:
    """Every file under *root*, by relative path, with its bytes.

    Compared before and after an invocation this catches a write, a truncation,
    a deletion and a newly created file alike -- which is what "wrote nothing"
    has to mean for a tool whose failure mode was a silent in-place rewrite.
    """
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def plant_committed_report(root, name="convergence_T1.md") -> Path:
    path = Path(root) / name
    path.write_text(COMMITTED, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# 1. the default must not destroy a committed report
# --------------------------------------------------------------------------

def test_the_default_invocation_refuses_to_overwrite_a_committed_report(
        cc, run_all, tmp_path, capsys):
    """THE DEFECT. `python runners/check_convergence.py --tier T1` rewrote
    `<results>/convergence_T1.md` in place. Auditing the committed tree was
    therefore indistinguishable from vandalising it, and the three tier records
    that ran this tool each had to invent the same `--report <scratch>`
    workaround by hand."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms)
    report = plant_committed_report(tmp_path)
    before = snapshot(tmp_path)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == REFUSED, out
    assert report.read_text(encoding="utf-8") == COMMITTED, (
        "the committed cap audit was overwritten by an audit that was only "
        "asked to READ the tier")
    assert snapshot(tmp_path) == before, (
        "a refused audit still changed the results tree")
    assert "--no-report" in out and "--force" in out and "--report" in out, (
        "the refusal must name all three ways forward, or the operator's only "
        "move is to delete the file: " + out)


def test_a_first_audit_still_writes_its_report(cc, run_all, tmp_path):
    """The guard is about REPLACING, not about writing. A tier that has never
    been audited still gets its report at the documented default path -- this
    is the contract `tests/test_check_convergence.py::
    test_report_defaults_into_the_results_dir` pins, and it must survive."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path)])

    assert rc == 0
    assert (tmp_path / "convergence_T1.md").is_file()


def test_the_report_flag_refuses_to_overwrite_without_force(
        cc, run_all, tmp_path, capsys):
    """`--report` aimed at an existing file is refused, and that file and the results tree stay unchanged."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms)
    report = plant_committed_report(tmp_path, "elsewhere.md")
    before = snapshot(tmp_path)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path),
                  "--report", str(report)])
    out = capsys.readouterr().out

    assert rc == REFUSED, out
    assert report.read_text(encoding="utf-8") == COMMITTED
    assert snapshot(tmp_path) == before


def test_force_is_the_one_way_to_replace_a_report(cc, run_all, tmp_path):
    """Deliberate regeneration in place stays possible -- it just has to be
    said out loud. Without this the guard would be a wall, not a gate."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms)
    report = plant_committed_report(tmp_path)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--force"])

    assert rc == 0
    text = report.read_text(encoding="utf-8")
    assert text != COMMITTED
    assert "# Cap audit -- tier T1" in text


def test_write_report_itself_refuses_to_clobber(cc, tmp_path):
    """The guard lives in `write_report`, not only in the CLI, so a future
    caller that reaches the writer directly cannot reintroduce the defect.
    It RAISES -- no try/except turns this into a silent skip."""
    path = tmp_path / "convergence_T1.md"
    path.write_text(COMMITTED, encoding="utf-8")
    kwargs = dict(label="tier T1", root=tmp_path, seeds=[0], rows=[],
                  arm_verdicts={}, errors=[], notes=[])

    with pytest.raises(FileExistsError) as excinfo:
        cc.write_report(path, **kwargs)

    assert str(path) in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == COMMITTED
    assert cc.write_report(path, overwrite=True, **kwargs) == path
    assert path.read_text(encoding="utf-8") != COMMITTED


# --------------------------------------------------------------------------
# 2. a read-only audit mode, and one that says so
# --------------------------------------------------------------------------

def test_the_audit_can_run_with_no_write_at_all_and_says_so(
        cc, run_all, tmp_path, capsys):
    """A read-only audit is what every verification agent on this project
    needed and none of them could find. `--no-report` did it, and stdout never
    mentioned it, so three records worked around the default instead.

    Two claims: the results tree is byte-identical afterwards, and the run
    ANNOUNCES that it wrote nothing and how to get a file if you wanted one."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms)
    plant_committed_report(tmp_path)
    before = snapshot(tmp_path)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert snapshot(tmp_path) == before, "a read-only audit wrote to the tree"
    assert "No report was written" in out, (
        "a read-only run must say it was read-only; silence is how an "
        "operator concludes the file was refreshed: " + out)
    assert "--report" in out


def test_report_and_no_report_together_are_refused(cc, run_all, tmp_path):
    """`--report` together with `--no-report` exits 2 and writes no report file."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms)

    with pytest.raises(SystemExit) as excinfo:
        cc.main(["--tier", "T1", "--results", str(tmp_path),
                 "--report", str(tmp_path / "r.md"), "--no-report"])

    assert excinfo.value.code == 2
    assert not (tmp_path / "r.md").exists()


# --------------------------------------------------------------------------
# 3. a green line from this tool must state its own scope
# --------------------------------------------------------------------------

def _disclosure_subjects(cc):
    return [subject for subject, _owner in cc.NOT_CHECKED]


def test_a_passing_tier_still_states_what_was_not_checked(
        cc, run_all, tmp_path, capsys):
    """PASS is where the disclosure matters most: the failure path already
    prints reasons, the pass path printed one reassuring sentence."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "IS NOT BUDGET-CONFOUNDED" in out
    assert "NOT CHECKED BY THIS AUDIT" in out, out
    for subject in _disclosure_subjects(cc):
        assert subject in out, (
            "{!r} is in NOT_CHECKED but never reaches stdout".format(subject))
    for owner in ("runners/check_tier.py", "runners/aggregate_seeds.py"):
        assert owner in out, (
            "the disclosure must name the tool that DOES own each question, "
            "or it just tells the operator to worry: " + out)


def test_a_stale_fingerprint_and_a_deleted_metric_pass_and_are_disclosed(
        cc, run_all, tmp_path, capsys):
    """The judgment, pinned. This tier's runs record a `code_fingerprint` that
    matches no tree, and their metrics files carry no accuracy at all. The cap
    audit still exits 0, because neither fact is a statement about the epoch
    budget and a second fingerprint verdict here could contradict
    `check_tier`'s. So the exit code is NOT widened -- the output is. If a
    later change makes this tier fail, that is a scope decision and this test
    is where it gets argued, not a silent drift."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, fingerprint="ffffffffffffffff",
               metrics={"experiment_name": "x", "n_samples": 34338})

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "NOT CHECKED BY THIS AUDIT" in out
    assert "code_fingerprint" in out, (
        "the tier's runs carry a fingerprint from no tree and the audit that "
        "passed them never mentions that it did not look: " + out)
    assert "overall_accuracy" in out or "metric" in out


def test_the_report_carries_the_same_disclosure(cc, run_all, tmp_path):
    """The markdown is the artifact kept and read weeks later -- the file's own
    comment at `remedy_kinds` says so. A report that says PASS without saying
    what it did not check is the same trap in durable form."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms)
    report = tmp_path / "regen.md"

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path),
                  "--report", str(report)])

    assert rc == 0
    text = report.read_text(encoding="utf-8")
    assert "## Not checked by this audit" in text, text[-800:]
    for subject in _disclosure_subjects(cc):
        assert subject in text
    assert "runners/check_tier.py" in text

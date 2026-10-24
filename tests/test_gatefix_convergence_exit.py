"""The cap audit must not report a sound tier as a broken one.

THE DEFECT. Making `runners/check_convergence.py` non-destructive (26 Sep) was
right (`tests/test_gatefix_nondestructive.py` pins that fix, and every test here is
written so that fix survives). But the refusal was given the SAME exit code, 2,
the tool already used for "you pointed me at nothing" -- and
`scripts/batch/tier-cycle.sh` calls the tool at its DEFAULT report path, where
the file it refuses to replace is the one the batch itself wrote minutes
earlier.

Measured against a scratch copy of the real, healthy T1 before this file
existed:

    $ python runners/check_convergence.py --tier T1 --results <copy of T1>
    check_convergence=2
    refusing to overwrite the existing report .../convergence_T1.md

    $ python runners/check_convergence.py --tier T1 --results <copy> --no-report
    check_convergence=0
    TIER T1 IS NOT BUDGET-CONFOUNDED.

The same tier, in the same state, is a 2 and a 0. `tier-cycle.sh` reduces
the studio gates with `grep -E "=[1-9]"`, so that 2 is a gate failure: the
chain prints NOT SOUND, refuses to launch the next tier and exits 1. The chain
is what runs unattended overnight, so one healthy tier stops a whole batch at
its first tier. Nothing is even lost loudly: the audit never READ the tier, so
the summary grep that ends the payload finds no verdict line in /tmp/g2 either
and the operator gets no cap verdict at all, only a refusal.

THE CONTRACT this file pins. An exit code that means "I declined to write a
report" must not be indistinguishable from "this tier failed the cap audit",
nor from "your command was wrong":

    0   the audit RAN; every audited arm converged inside its budget
    1   the audit RAN and the tier FAILED it (cap-bound, incomplete,
        mislabelled, or the arms did not share one selection rule)
    2   the audit COULD NOT run because the INVOCATION is wrong (unknown tier,
        no results directory). Nothing about the tier is known
    3   the audit DID NOT run because the report target already exists and
        --force was not given. Nothing read, nothing written. NOT a verdict
        about the tier -- ask again with --no-report to get one

and the chain has to act on that distinction rather than on "nonzero".

Test-policy category 3 (gate-tool verdicts), `advisor-guidance/
test-policy-2026-09-23.md:15`. Fixtures are local to the file, the convention
`tests/test_check_convergence.py` and `tests/test_gatefix_nondestructive.py`
both follow.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

CODEBASE = Path(__file__).resolve().parents[1]
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

#: `scripts/batch/` sits BESIDE the codebase, not inside it: the chain drives
#: the repo from outside so it still runs while a pull is mid-flight.
TIER_CYCLE = CODEBASE.parent / "scripts" / "batch" / "tier-cycle.sh"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, CODEBASE / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cc():
    return _load("check_convergence_exit", "runners/check_convergence.py")


@pytest.fixture(scope="module")
def run_all():
    return _load("run_all_for_exit_test", "runners/run_all.py")


def write_run(root, arm, seed, *, selected_epoch=22, reason="early_stop",
              cap=60, patience=10):
    """One `results/<arm>/seed<k>/`, in the shape `run_all.py` lays out."""
    run_dir = Path(root) / arm / "seed{}".format(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_metrics.json").write_text(json.dumps({
        "experiment_name": arm, "overall_accuracy": 0.9607, "macro_f1": 0.88,
        "mean_iou": 0.81, "n_samples": 34338,
        "provenance": {
            "seed": seed, "epoch": selected_epoch,
            "selected_epoch": selected_epoch, "stopping_reason": reason,
            "use_class_weights": False, "epoch_budget_configured": cap,
            "config_hash": "0123456789abcdef",
            "code_fingerprint": "9d88e9213dbb9164",
        }}), encoding="utf-8")
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump({"experiment_name": arm, "seed": seed,
                        "training": {"max_epochs": cap, "patience": patience,
                                     "batch_size": 4}}, sort_keys=False),
        encoding="utf-8")
    return run_dir


def build_tier(root, run_all, tier="T1", seeds=(0, 1, 2, 3, 4), **kw):
    for cell in run_all.TIERS[tier]:
        for s in seeds:
            write_run(root, cell.name, s, **kw)


#: What the batch itself wrote. Byte-compared after every refusal below: the
#: whole point of the 26 Sep fix is that this survives being audited.
COMMITTED = (
    "# Cap audit -- tier T1\n\nGenerated 2026-09-25T19:36:00+00:00 by "
    "`runners/check_convergence.py`.\nThis is the BATCH own report.\n")


def snapshot(root) -> dict:
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ==========================================================================
# 1. the four exit codes, and that they are four
# ==========================================================================

def test_a_healthy_tier_whose_report_exists_is_not_reported_as_a_failure(
        cc, run_all, tmp_path, capsys):
    """THE DEFECT, in one assertion.

    The tier is sound -- the same rows that exit 0 under --no-report in the
    next test. The only thing in the way is a file the tool is right not to
    replace. That must not come back as 1 (the tier failed the cap audit) and
    must not come back as 2 (the invocation was wrong): both are claims about
    something this invocation never looked at.
    """
    build_tier(tmp_path, run_all)
    (tmp_path / "convergence_T1.md").write_text(COMMITTED, encoding="utf-8")

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == 3, (
        "a healthy tier whose report already exists exited {} -- a code the "
        "chain reads as a gate failure. Output:\n{}".format(rc, out))


def test_the_same_tier_read_only_exits_zero(cc, run_all, tmp_path, capsys):
    """The other half of the proof: the tier really is sound. Without this,
    exit 3 above could be covering a genuine cap failure."""
    build_tier(tmp_path, run_all)
    (tmp_path / "convergence_T1.md").write_text(COMMITTED, encoding="utf-8")

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "IS NOT BUDGET-CONFOUNDED" in out, out


def test_a_cap_bound_tier_still_exits_one(cc, run_all, tmp_path, capsys):
    """Exit 1 keeps its meaning: the audit ran and the tier failed it."""
    build_tier(tmp_path, run_all, selected_epoch=60, reason="cap")

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "TRUNCATED" in out, out


def test_an_unknown_tier_still_exits_two(cc, tmp_path, capsys):
    rc = cc.main(["--tier", "T99", "--results", str(tmp_path), "--no-report"])
    assert rc == 2, capsys.readouterr().out


def test_a_missing_results_directory_still_exits_two(cc, tmp_path, capsys):
    rc = cc.main(["--tier", "T1", "--results", str(tmp_path / "nope"),
                  "--no-report"])
    assert rc == 2, capsys.readouterr().out


def test_the_four_outcomes_have_four_distinct_codes(cc, run_all, tmp_path,
                                                    capsys):
    """Every code measured from a real invocation, in one test, so collapsing
    any two of them fails here even if the tests above are read alone."""
    good = tmp_path / "good"
    build_tier(good, run_all)
    rc_pass = cc.main(["--tier", "T1", "--results", str(good)])

    planted = tmp_path / "planted"
    build_tier(planted, run_all)
    (planted / "convergence_T1.md").write_text(COMMITTED, encoding="utf-8")
    rc_report = cc.main(["--tier", "T1", "--results", str(planted)])

    bad = tmp_path / "bad"
    build_tier(bad, run_all, selected_epoch=60, reason="cap")
    rc_fail = cc.main(["--tier", "T1", "--results", str(bad), "--no-report"])

    rc_usage = cc.main(["--tier", "T99", "--results", str(good), "--no-report"])
    capsys.readouterr()

    codes = {"pass": rc_pass, "tier failed": rc_fail,
             "usage error": rc_usage, "report exists": rc_report}
    assert codes == {"pass": 0, "tier failed": 1, "usage error": 2,
                     "report exists": 3}, codes
    assert len(set(codes.values())) == 4, (
        "two different outcomes share an exit code, so a caller cannot tell "
        "them apart: {}".format(codes))


def test_the_refusal_still_writes_nothing_at_all(cc, run_all, tmp_path,
                                                 capsys):
    """The 26 Sep fix has to survive the new code. A refusal reads nothing and
    writes nothing -- it does not create the report it declined to replace and
    does not touch the committed one."""
    build_tier(tmp_path, run_all)
    report = tmp_path / "convergence_T1.md"
    report.write_text(COMMITTED, encoding="utf-8")
    before = snapshot(tmp_path)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path)])
    out = capsys.readouterr().out

    assert rc == 3, out
    assert report.read_text(encoding="utf-8") == COMMITTED
    assert snapshot(tmp_path) == before, "a refused audit changed the tree"
    assert "--no-report" in out and "--force" in out, out


def test_a_first_audit_still_writes_its_report_and_exits_zero(
        cc, run_all, tmp_path, capsys):
    """Exit 3 is about REPLACING. A tier audited for the first time still gets
    its report at the documented default path and still exits 0."""
    build_tier(tmp_path, run_all)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path)])
    capsys.readouterr()

    assert rc == 0
    assert (tmp_path / "convergence_T1.md").is_file()


def test_force_replaces_the_report_and_exits_on_the_verdict(
        cc, run_all, tmp_path, capsys):
    """With --force the tool is answering about the TIER again, so its exit
    code is the verdict, not 3."""
    build_tier(tmp_path, run_all)
    (tmp_path / "convergence_T1.md").write_text(COMMITTED, encoding="utf-8")

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--force"])
    capsys.readouterr()

    assert rc == 0
    assert (tmp_path / "convergence_T1.md").read_text(
        encoding="utf-8") != COMMITTED


# ==========================================================================
# 2. the contract has to be readable from the tool itself
# ==========================================================================

def test_help_documents_what_each_exit_code_means(cc, capsys):
    """An operator debugging an overnight batch reads --help, not the source.
    The distinction between "I declined to write" and "this tier failed" is
    the one that cost a batch, so --help has to state it."""
    with pytest.raises(SystemExit):
        cc.main(["--help"])
    out = capsys.readouterr().out

    assert "Exit codes" in out, out
    for code in ("0", "1", "2", "3"):
        assert re.search(r"^\s*{}\s".format(code), out, re.M), (
            "--help does not document exit code {}:\n{}".format(code, out))
    assert re.search(r"3\s+the audit DID NOT run", out), (
        "--help must say 3 means the audit did not run -- not that the tier "
        "failed:\n{}".format(out))


# ==========================================================================
# 3. the chain -- scripts/batch/tier-cycle.sh
# ==========================================================================

BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash on PATH")


def payload_body() -> str:
    """The multi-line ssh payload of step 2, as the REMOTE shell receives it.

    `tier-cycle.sh` wraps it in `"bash -lc ' ... '"`, so the LOCAL shell strips
    one level of double-quote escaping before sending. Undoing exactly that
    (\\" -> ", \\$ -> $) reproduces the remote text, which is the string these
    tests have to exercise -- asserting against the escaped source would test
    the wrong thing.
    """
    lines = TIER_CYCLE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("GATES=$(ssh"))
    end = next(i for i, l in enumerate(lines)
               if i > start and l.startswith("'\""))
    body = "\n".join(lines[start + 1:end])
    return body.replace('\\"', '"').replace("\\$", "$")


def payload_commands() -> str:
    """The payload with its whole-line comments removed: what the remote shell
    actually RUNS.

    The flag assertions below are about INVOCATIONS, not prose. The payload
    documents at the point of use that the re-ask is deliberately not the
    replace flag, and asserting over the raw text made that explanation itself
    the failure -- the next person to document the decision would break the
    test again. Only whole-line comments are dropped, so an inline one still
    fails: that is the safe direction.
    """
    return "\n".join(l for l in payload_body().splitlines()
                      if not l.lstrip().startswith("#"))


def ssh_payloads() -> list:
    """Every `bash -lc '...'` payload in the script, as (line_number, text).
    The multi-line one is appended under line 0."""
    lines = TIER_CYCLE.read_text(encoding="utf-8").splitlines()
    out = []
    for i, line in enumerate(lines, 1):
        if "bash -lc '" not in line:
            continue
        rest = line.split("bash -lc '", 1)[1]
        if rest.strip() == "":
            continue          # the multi-line payload, handled separately
        out.append((i, rest.rsplit("'", 1)[0]))
    out.append((0, payload_body()))
    return out


def test_no_apostrophe_has_crept_back_into_any_ssh_payload():
    """THE 25 SEP DEFECT, pinned. Every payload is wrapped in single quotes,
    so one apostrophe anywhere inside closes the quote early. The word T1/T2s
    in a COMMENT carried one; the remote login shell (zsh) reported an
    unmatched quote on stderr while stderr was being discarded, so the whole
    gate step emitted nothing and T4 was scored as passing a gate that had
    never run. A comment is not a safe place for an apostrophe here."""
    for lineno, text in ssh_payloads():
        assert "'" not in text, (
            "apostrophe inside the ssh payload at tier-cycle.sh line {} -- it "
            "closes the single quote early and the gate silently produces no "
            "output. Offending text: {!r}".format(lineno or "the multi-line payload", text))


def test_every_single_line_payload_is_quote_balanced():
    """The counterpart: exactly two apostrophes on each single-line ssh call
    (the opening and closing delimiter), so the quote closes where it looks
    like it closes."""
    lines = TIER_CYCLE.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        if "bash -lc '" not in line or line.rstrip().endswith("-lc '"):
            continue
        assert line.count("'") == 2, (
            "tier-cycle.sh:{} has {} apostrophes on a single-line ssh call; "
            "two (open and close) is the only safe number: {}"
            .format(i, line.count("'"), line))


def _guard_block() -> str:
    """The empty-output guard, lifted verbatim from the script."""
    src = TIER_CYCLE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(src)
                 if l.startswith('if ! echo "$GATES"'))
    end = next(i for i, l in enumerate(src) if i > start and l == "fi")
    return "\n".join(src[start:end + 1])


def chain_header() -> str:
    """tier-cycle.sh's leading comment block -- what an operator reads when
    the unattended chain has exited and left only a number behind."""
    out = []
    for line in TIER_CYCLE.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.startswith("#"):
            break
        out.append(line)
    return "\n".join(out)


def chain_exit_codes() -> set:
    """Every code the chain can actually RETURN, read from its executable
    lines only: a comment discussing check_convergence's exit 3 is not one of
    the chain's own codes."""
    codes = set()
    for line in TIER_CYCLE.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        codes.update(int(m) for m in re.findall(r"\bexit (\d+)", line))
    return codes


def test_every_exit_code_the_chain_can_return_is_documented_in_its_header():
    """The same defect as this file's subject, one level up.

    The chain exits 2 when the tier DIED and 3 when it TIMED OUT; the tool it
    calls exits 2 for a bad invocation and 3 for "I declined to write a
    report". The two sets are unrelated and both are in play in the same
    overnight run, so the chain's own codes cannot be left to be inferred from
    the source at 3 a.m. Every code it can return must be named where the
    operator looks first.
    """
    codes = chain_exit_codes()
    assert codes, "no exit codes found in tier-cycle.sh -- the parser is wrong"
    header = chain_header()
    undocumented = [c for c in sorted(codes)
                    if not re.search(r"^#\s+{}\s+\S".format(c), header, re.M)]
    assert not undocumented, (
        "tier-cycle.sh can exit {} but its header documents none of {}. An "
        "unattended chain leaves nothing behind but its exit code."
        "\nHeader:\n{}"
        .format(sorted(codes), undocumented, header))


@needs_bash
def test_the_empty_output_guard_is_still_present_and_still_fires(tmp_path):
    """A gate that produces no output is a gate that did not run, and must
    never be indistinguishable from a gate that passed. This RUNS the guard in
    real bash with the empty $GATES the 25 Sep quoting bug produced, and
    requires the documented exit 6."""
    guard = _guard_block()
    assert "exit 6" in guard, guard

    script = tmp_path / "guard.sh"
    script.write_text('GATES=""\n' + guard + "\necho REACHED_THE_END\n",
                      encoding="utf-8")
    p = subprocess.run([BASH, str(script)], capture_output=True, text=True)

    assert p.returncode == 6, (
        "the empty-output guard did not fire on empty gate output (rc={}): "
        "{}{}".format(p.returncode, p.stdout, p.stderr))
    assert "REACHED_THE_END" not in p.stdout
    assert "NOT a pass" in p.stdout, p.stdout


@needs_bash
def test_the_guard_fires_on_silence_not_on_failure(tmp_path):
    """The guard must fire on SILENCE only: a gate step that really ran and
    really failed has to reach the FAILED reduction below it rather than be
    swallowed by the did-not-run branch."""
    script = tmp_path / "guard_ok.sh"
    script.write_text('GATES="check_tier=1"\n' + _guard_block()
                      + "\necho REACHED_THE_END\n", encoding="utf-8")
    p = subprocess.run([BASH, str(script)], capture_output=True, text=True)

    assert p.returncode == 0 and "REACHED_THE_END" in p.stdout, (
        p.returncode, p.stdout, p.stderr)


def _gate_step_script(tmp_path, name, convergence_rc, *, readonly_rc=0):
    """The REAL step-2 payload, wired to a stub `python`.

    The stub is a shell FUNCTION rather than a file on PATH: a file would need
    a POSIX exec bit, which Windows does not carry, and the point here is to
    exercise the payload, not the loader.

    `convergence_rc` is what check_convergence returns at its DEFAULT report
    path; `readonly_rc` what it returns under --no-report.

    The payload opens with `cd ~/capp-gpt/codebase`, the studio path. It is
    redirected at the temp dir so the test does not depend on what this
    machine happens to have in $HOME -- the cd is not what is under test.

    The stub ALSO appends each argv to `$TESTDIR/calls.log`. The payload sends
    every gate command stdout to /tmp/g<n>, so what the stub prints is not
    visible on the chain stdout -- the log is how these tests see which
    invocations actually happened, and with which flags.
    """
    body = payload_body().replace("cd ~/capp-gpt/codebase",
                                  'cd "$TESTDIR"', 1)
    script = tmp_path / name
    script.write_text(
        'TESTDIR="{}"\nT=T1\nDS=cadsynth\nARMS=x\nAGG_OUT=/dev/null\n'
        "python() {{\n"
        '  echo "$*" >> "$TESTDIR/calls.log"\n'
        '  case "$*" in\n'
        "    *check_convergence*--no-report*) echo AUDIT_RAN; return {};;\n"
        "    *check_convergence*) echo DEFAULT_PATH; return {};;\n"
        "    *) return 0;;\n"
        "  esac\n"
        "}}\n".format(tmp_path.as_posix(), readonly_rc, convergence_rc)
        + body + "\n", encoding="utf-8")
    return script


def _convergence_line(p):
    return [l for l in p.stdout.splitlines()
            if l.startswith("check_convergence=")]


def _convergence_calls(tmp_path):
    """Every check_convergence invocation the payload actually made."""
    log = tmp_path / "calls.log"
    if not log.is_file():
        return []
    return [l for l in log.read_text(encoding="utf-8").splitlines()
            if "check_convergence" in l]


@needs_bash
def test_the_chain_does_not_report_a_sound_tier_as_a_gate_failure(tmp_path):
    """THE CHAIN HALF OF THE DEFECT, executed.

    The real step-2 payload runs in real bash against a stub that reproduces
    exactly what the healthy T1 does: refuse at the default report path, exit
    0 read-only. The line the chain reduces with `grep -E "=[1-9]"` must come
    out `check_convergence=0`. Before the fix the payload emitted
    `check_convergence=2`, which that grep turns into NOT SOUND and a halted
    batch.
    """
    script = _gate_step_script(tmp_path, "gates_sound.sh", 3)
    p = subprocess.run([BASH, str(script)], capture_output=True, text=True)

    line = _convergence_line(p)
    assert line, "the gate step emitted no check_convergence line: {}{}".format(
        p.stdout, p.stderr)
    assert line[0] == "check_convergence=0", (
        "the chain reported a SOUND tier as {} -- `grep -E \"=[1-9]\"` at "
        "reduction turns that into NOT SOUND and stops the batch.\n"
        "stdout:\n{}\nstderr:\n{}".format(line[0], p.stdout, p.stderr))


@needs_bash
def test_the_chain_reaudits_read_only_rather_than_forcing(tmp_path):
    """HOW it recovers matters. The chain must ask again READ-ONLY, never with
    --force: forcing would make the overnight batch the thing that destroys
    the committed cap audit, which is the defect the 26 Sep fix removed."""
    script = _gate_step_script(tmp_path, "gates_readonly.sh", 3)
    p = subprocess.run([BASH, str(script)], capture_output=True, text=True)
    calls = _convergence_calls(tmp_path)

    assert any("--no-report" in c for c in calls), (
        "the chain never re-ran the audit read-only, so it still has no cap "
        "verdict for this tier. Calls made: {}\n{}{}"
        .format(calls, p.stdout, p.stderr))
    assert not any("--force" in c for c in calls), (
        "the chain passed --force: that restores the in-place rewrite of the "
        "committed report that tests/test_gatefix_nondestructive.py exists to "
        "prevent. Calls made: {}".format(calls))
    assert "--force" not in payload_commands(), (
        "--force appears in a COMMAND in the gate payload, not merely in a "
        "comment about it: {}".format(
            [l for l in payload_commands().splitlines() if "--force" in l]))


@needs_bash
def test_a_sound_tier_now_reaches_the_summary_grep_with_a_verdict(tmp_path):
    """The refusal also cost the operator the verdict LINE: with nothing
    audited, the summary grep that ends the payload found no cap verdict in
    /tmp/g2 at all. After the retry it must find one."""
    body = payload_commands()
    assert "/tmp/g2" in body, "the audit output no longer lands in /tmp/g2"
    assert body.count("/tmp/g2") >= 2, (
        "the read-only re-ask must write to /tmp/g2 too, or the summary grep "
        "still reads the refusal: {}".format(body))
    script = _gate_step_script(tmp_path, "gates_grep.sh", 3)
    subprocess.run([BASH, str(script)], capture_output=True, text=True)

    calls = _convergence_calls(tmp_path)
    assert len(calls) == 2, (
        "expected exactly two check_convergence calls (default path, then the "
        "read-only re-ask), got: {}".format(calls))
    assert "--no-report" not in calls[0] and "--no-report" in calls[1], calls

    # The re-ask has to land in the SAME file the summary grep reads, or the
    # operator still sees the refusal where the cap verdict should be. Checked
    # against the payload text: /tmp is the studio tmp, not this machine.
    reask = [l for l in body.splitlines()
             if "check_convergence" in l and "--no-report" in l]
    assert len(reask) == 1, reask
    assert "> /tmp/g2" in reask[0], (
        "the read-only re-ask does not write to /tmp/g2, so the summary grep "
        "at the end of the payload still reads the refusal: {}".format(reask))


@needs_bash
def test_the_chain_still_reports_a_genuinely_cap_bound_tier(tmp_path):
    """The fix must not become a blanket suppressor. When the tool answers
    about the TIER rather than about the report, that answer passes through
    unchanged and still stops the chain."""
    script = _gate_step_script(tmp_path, "gates_capbound.sh", 1)
    p = subprocess.run([BASH, str(script)], capture_output=True, text=True)

    line = _convergence_line(p)
    assert line and line[0] == "check_convergence=1", (
        "a cap-bound tier must still reach the chain as a failure: {}\n{}"
        .format(p.stdout, p.stderr))
    assert _convergence_calls(tmp_path) == ["runners/check_convergence.py "
                                            "--tier T1 --results results"], (
        "a real verdict must not be retried read-only -- only the "
        "report-exists code is: {}".format(_convergence_calls(tmp_path)))


@needs_bash
def test_a_usage_error_still_reaches_the_chain_as_a_failure(tmp_path):
    """Exit 2 means the audit could not run at all: an unknown tier, a results
    directory that is not there. That is not something to retry read-only and
    not something to pass over -- it must still stop the chain."""
    script = _gate_step_script(tmp_path, "gates_usage.sh", 2)
    p = subprocess.run([BASH, str(script)], capture_output=True, text=True)

    line = _convergence_line(p)
    assert line and line[0] == "check_convergence=2", (p.stdout, p.stderr)
    assert re.search(r"=[1-9]", line[0]), (
        "the chain reduces its gate lines with `grep -E \"=[1-9]\"`, so a usage "
        "error has to match it: {}".format(line[0]))
    assert not any("--no-report" in c for c in _convergence_calls(tmp_path)), (
        "exit 2 means the audit could not run at all; re-asking read-only "
        "would only produce the same error: {}"
        .format(_convergence_calls(tmp_path)))


@needs_bash
def test_the_gate_step_still_emits_check_tier_so_the_guard_can_see_it(
        tmp_path):
    """The empty-output guard keys on the literal `check_tier=`. Editing the
    convergence line must not disturb it, or the guard stops being able to
    tell a silent gate step from a passing one."""
    script = _gate_step_script(tmp_path, "gates_guardkey.sh", 3)
    p = subprocess.run([BASH, str(script)], capture_output=True, text=True)

    assert any(l.startswith("check_tier=") for l in p.stdout.splitlines()), (
        "the gate step no longer emits a check_tier= line, so the "
        "empty-output guard can never fire:\n{}{}"
        .format(p.stdout, p.stderr))

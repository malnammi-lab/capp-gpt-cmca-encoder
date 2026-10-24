"""KILLED is not FAILED, and a killed run must leave a readable log.

Why this file exists
--------------------

    "run_all._run: add -u to the child command and flag 'exit != 0 with empty
     log' as a distinct KILLED verdict. Why: the T2 failure left zero
     diagnosable output."

A 20-run tier died partway through. The runs had been training for many minutes,
but every train.log was EMPTY: CPython block-buffers stdout when it is a file
rather than a tty, so nothing the trainer printed had ever reached disk. The
post-mortem could not distinguish a crash from an OOM kill from someone closing
the session, because the only surviving evidence was a non-zero exit code -- and
run_all printed the same "FAILED (exit N)" line for all three.

That matters again immediately: the next batch is 85 runs on one machine, and
something will die.

So two behaviours are pinned here:

* `child_command` passes ``-u``, BEFORE ``-m``, so the log survives the kill;
* `classify_exit` calls a run KILLED, not FAILED, when it was stopped from
  outside, and `main` reports and counts the two separately.

Nothing in this file launches a training run. The verdict logic is a pure
function of (returncode, log text), and the two places it is wired in -- `_run`
and `main` -- are exercised with `subprocess.run` and `_run` stubbed out, which
is the whole point: these conditions are reachable in a test precisely because
the classification does not need a GPU to happen.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

CODEBASE = Path(__file__).resolve().parents[1]


def _load_run_all():
    """Import runners/run_all.py by path.

    `runners/` is not a package -- `run_all.py` is a script with a shebang and
    it inserts the repo root on sys.path at import time -- so a plain
    `import runners.run_all` does not work. `tests/test_seeding.py` and
    `runners/check_tier.py` both load it exactly this way.
    """
    spec = importlib.util.spec_from_file_location(
        "run_all", CODEBASE / "runners" / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_all():
    return _load_run_all()


#: A config path that only has to be *under the repo* -- `child_command` calls
#: `relative_to(REPO)` on it, which is string arithmetic and never touches disk.
CFG = "results/_unit/seed0/config.resolved.yaml"

#: A complete regiment provenance block, so that a healthy cell in the sweep
#: fixture below counts as `done` rather than as `unquotable`. The field list is
#: `src/utils/provenance.check_regiment_fields`.
PROVENANCE = {
    "seed": 0, "epoch": 22, "selected_epoch": 22, "stopping_reason": "early_stop",
    "config_hash": "0" * 16, "code_fingerprint": "9d88e9213dbb9164",
    "gpu": "test", "wall_time_s": 1.0,
}


# --------------------------------------------------------------------- #
#  1. -u: the child must be unbuffered                                   #
# --------------------------------------------------------------------- #

def test_child_command_passes_dash_u(run_all):
    """The fix itself. Without -u a killed run leaves an empty log."""
    cmd = run_all.child_command("src.training.train_semantic", run_all.REPO / CFG)
    assert "-u" in cmd, (
        f"the child command has no -u: {cmd}\n"
        f"CPython block-buffers stdout when it is redirected to a file, so a "
        f"run killed mid-training would again leave an empty train.log and "
        f"nothing to diagnose it with.")


def test_dash_u_comes_before_dash_m(run_all):
    """-u after -m is not an interpreter flag, it is an argument to the module.

    `python -m pkg -u` hands "-u" to pkg's argparse, which rejects it. The
    ordering is the difference between an unbuffered run and a run that does not
    start at all.
    """
    cmd = run_all.child_command("src.training.train_semantic", run_all.REPO / CFG)
    assert cmd[0] == sys.executable
    assert cmd.index("-u") < cmd.index("-m"), (
        f"-u must be an interpreter flag, so it must precede -m: {cmd}")


def test_the_command_still_points_the_module_at_the_resolved_config(run_all):
    """-u must not have displaced the rest of the invocation."""
    cmd = run_all.child_command("src.evaluation.evaluate_semantic",
                                run_all.REPO / CFG)
    assert cmd[cmd.index("-m") + 1] == "src.evaluation.evaluate_semantic"
    assert cmd[cmd.index("--config") + 1] == CFG


def test_run_launches_the_unbuffered_command(run_all, tmp_path, monkeypatch):
    """`_run` must launch what `child_command` builds -- not its own list.

    Asserting on the helper alone would pass even if `_run` went back to
    assembling `[sys.executable, "-m", ...]` inline, which is exactly the state
    this audit item found. So capture the argv that reaches `subprocess.run`.
    """
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    log = tmp_path / "train.log"
    run_all._run("src.training.train_semantic", run_all.REPO / CFG, log)

    assert "-u" in seen["cmd"], (
        f"subprocess.run was called without -u: {seen['cmd']}")
    assert seen["cmd"].index("-u") < seen["cmd"].index("-m")
    assert seen["cmd"] == run_all.child_command(
        "src.training.train_semantic", run_all.REPO / CFG)
    # and the log still captures both streams into one file
    assert seen["kwargs"]["stderr"] is subprocess.STDOUT
    assert log.read_text(encoding="utf-8").startswith("$ ")


# --------------------------------------------------------------------- #
#  2. classify_exit: what counts as KILLED                               #
# --------------------------------------------------------------------- #

def _log(*child_lines: str) -> str:
    """A log exactly as `_run` writes it: the command header, then the child."""
    return "$ python -u -m src.training.train_semantic --config c.yaml\n\n" + \
        "".join(line + "\n" for line in child_lines)


def test_exit_zero_is_ok(run_all):
    verdict, reason = run_all.classify_exit(0, _log("epoch 1", "done"))
    assert verdict == "ok"
    assert reason == ""


def test_exit_zero_with_an_empty_log_is_still_ok(run_all):
    """A quiet success is not a kill. The empty-log rule is only for rc != 0."""
    assert run_all.classify_exit(0, _log())[0] == "ok"


@pytest.mark.parametrize("sig,word", [(9, "SIGKILL"), (15, "SIGTERM"),
                                      (2, "SIGINT")])
def test_negative_returncode_is_killed_and_names_the_signal(run_all, sig, word):
    """POSIX subprocess reports a signal-killed child as -N."""
    verdict, reason = run_all.classify_exit(-sig, _log("epoch 1", "epoch 2",
                                                       "epoch 3", "epoch 4"))
    assert verdict == "killed", (
        f"a child killed by signal {sig} was reported as {verdict!r}")
    assert word in reason, f"the verdict does not name the signal: {reason!r}"
    assert str(sig) in reason


def test_an_unnamed_signal_is_still_killed(run_all):
    """Not being in the lookup table must not demote a kill to a failure."""
    verdict, reason = run_all.classify_exit(-11, _log("a", "b", "c", "d"))
    assert verdict == "killed"
    assert "11" in reason


@pytest.mark.parametrize("code,word", [(137, "SIGKILL"), (143, "SIGTERM"),
                                       (130, "SIGINT")])
def test_shell_reported_128_plus_n_is_killed(run_all, code, word):
    """Through nohup/sbatch/a notebook cell the same kill arrives as 128+N.

    The 85-run batch will be launched through a shell, so this is the form that
    will actually be seen when the OOM killer takes a run.
    """
    verdict, reason = run_all.classify_exit(code, _log("epoch 1", "epoch 2",
                                                       "epoch 3", "epoch 4"))
    assert verdict == "killed", (
        f"exit {code} is 128+{code - 128} -- a signal seen through a shell -- "
        f"but it was classified {verdict!r}")
    assert word in reason


@pytest.mark.parametrize("code", [3221225786, -1073741510])
def test_windows_control_c_exit_is_killed(run_all, code):
    """0xC000013A, unsigned or signed depending on the Python build."""
    assert run_all.classify_exit(code, _log("a", "b", "c", "d"))[0] == "killed"


def test_nonzero_exit_with_an_empty_log_is_killed(run_all):
    """The rule, verbatim: 'exit != 0 with empty log' -> KILLED.

    This is the T2 case. The run had been training for minutes and left nothing
    behind. A process that RAISES writes a traceback on its way out; a process
    that writes nothing at all never unwound, so it was stopped from outside.
    """
    verdict, reason = run_all.classify_exit(1, _log())
    assert verdict == "killed", (
        "a non-zero exit with no child output was reported as "
        f"{verdict!r}. That is the T2 failure mode: FAILED sends the operator "
        "to read a log that does not exist.")
    assert "0 line" in reason
    # the verdict must say what to do next, not just what happened
    assert "OOM" in reason


def test_the_header_line_is_not_mistaken_for_child_output(run_all):
    """`_run` writes `$ <cmd>` and a blank line before the child says anything.

    If those were counted, a genuinely empty log would look like one line of
    output and the empty-log rule would never fire.
    """
    assert run_all.child_output_lines(_log()) == []
    assert run_all.child_output_lines(_log("x")) == ["x"]
    assert run_all.LOG_HEADER_LINES == 2


def test_blank_lines_from_the_child_do_not_count_as_output(run_all):
    assert run_all.child_output_lines(_log("", "  ", "")) == []
    assert run_all.classify_exit(1, _log("", "  ", ""))[0] == "killed"


# --------------------------------------------------------------------- #
#  3. classify_exit: what must NOT be called KILLED                      #
# --------------------------------------------------------------------- #

def test_a_run_that_raised_is_failed_not_killed(run_all):
    """A traceback is proof the process unwound, so nothing killed it.

    Misfiling a real bug as KILLED is the expensive direction of this error: it
    tells the operator to go look at free VRAM instead of at the exception.
    """
    tb = _log(
        "loading configs/cadsynth_bias_both.yaml",
        "Traceback (most recent call last):",
        '  File "src/training/train_semantic.py", line 210, in main',
        "    raise KeyError(...)",
        "KeyError: 'epochs'",
    )
    verdict, reason = run_all.classify_exit(1, tb)
    assert verdict == "failed", (
        "a run that left a traceback was reported as KILLED; the operator "
        "would be sent to check the OOM killer instead of reading the "
        "exception.")
    assert reason == ""


def test_the_empty_log_threshold_sits_just_below_a_minimal_traceback(run_all):
    """Three lines is the shortest real traceback; two or fewer is a kill.

    Pinned explicitly because the boundary is the whole content of the rule.
    """
    assert run_all.EMPTY_LOG_MAX_LINES == 2
    assert run_all.classify_exit(1, _log("a", "b"))[0] == "killed"
    assert run_all.classify_exit(1, _log("a", "b", "c"))[0] == "failed"


def test_run_classifies_a_real_empty_log_end_to_end(run_all, tmp_path,
                                                    monkeypatch):
    """`_run` must read the file it wrote and reach the KILLED verdict.

    The header offset is the fragile part: `_run` writes the header and
    `classify_exit` skips it, and the two agreeing is only testable by letting
    `_run` write a real file. The child is stubbed, so nothing trains.
    """
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=1)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    log = tmp_path / "train.log"
    rc, tail, verdict, reason = run_all._run(
        "src.training.train_semantic", run_all.REPO / CFG, log)

    assert rc == 1
    assert verdict == "killed", (
        f"a child that exited 1 having written nothing was classified "
        f"{verdict!r}; the log on disk was:\n{log.read_text(encoding='utf-8')!r}")
    assert reason
    assert tail.startswith("$ ")


def test_run_classifies_a_real_traceback_log_end_to_end(run_all, tmp_path,
                                                        monkeypatch):
    """The other direction, so the header offset cannot be off by one either way."""
    def fake_run(cmd, **kwargs):
        fh = kwargs["stdout"]
        fh.write("loading config\nTraceback (most recent call last):\n"
                 "  File \"x.py\", line 1, in <module>\nRuntimeError: boom\n")
        fh.flush()
        return subprocess.CompletedProcess(args=cmd, returncode=1)

    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    rc, tail, verdict, reason = run_all._run(
        "src.training.train_semantic", run_all.REPO / CFG, tmp_path / "train.log")

    assert (rc, verdict, reason) == (1, "failed", "")
    assert "RuntimeError: boom" in tail


# --------------------------------------------------------------------- #
#  4. the sweep reports the two verdicts separately                      #
# --------------------------------------------------------------------- #

@pytest.fixture
def swept(run_all, tmp_path, monkeypatch, capsys):
    """Run `main` over tier T1 with the child stubbed: cell 1 killed, cell 2 raised.

    `write_resolved_config` is stubbed as well, so this test writes nothing
    anywhere under the repo -- five other agents are editing this tree.
    """
    def fake_write(cell, seed, run_dir):
        cfg = tmp_path / f"{cell.name}.yaml"
        cfg.write_text("{}", encoding="utf-8")
        return cfg

    outcomes = {
        "cadsynth_bias_similarity": (137, "$ python -u -m ...", "killed",
                                     "exit 137 == 128+9: SIGKILL"),
        "cadsynth_bias_adjacency": (1, "KeyError: 'epochs'", "failed", ""),
    }

    def fake_run(module, cfg_path, log):
        outcome = outcomes.get(cfg_path.stem)
        if outcome is not None:
            return outcome
        if module.startswith("src.evaluation"):
            # a healthy cell leaves a quotable metrics file behind, so that the
            # other two cells land in `done` and cannot inflate either count
            log.parent.mkdir(parents=True, exist_ok=True)
            (log.parent / "eval_metrics.json").write_text(
                json.dumps({"accuracy": 0.96, "provenance": PROVENANCE}),
                encoding="utf-8")
        return 0, "", "ok", ""

    monkeypatch.setattr(run_all, "write_resolved_config", fake_write)
    monkeypatch.setattr(run_all, "_run", fake_run)

    rc = run_all.main(["--tier", "T1", "--seeds", "0",
                       "--results", str(tmp_path / "results")])
    return rc, capsys.readouterr().out


def test_a_killed_run_is_labelled_killed_not_failed(swept):
    _, out = swept
    assert "TRAINING KILLED (exit 137)" in out, (
        "a run stopped by the OOM killer was not labelled KILLED:\n" + out)
    assert "TRAINING FAILED (exit 137)" not in out


def test_the_summary_counts_killed_apart_from_failed(swept):
    _, out = swept
    summary = out.split("SUMMARY")[-1]
    assert "killed       1" in summary, (
        "the summary has no separate killed count:\n" + summary)
    assert "failed       1" in summary, (
        "the killed run was folded into the failed count, or the genuinely "
        "failed run was lost:\n" + summary)
    assert "KILLED [1/4] cadsynth_bias_similarity seed 0" in summary
    assert "FAILED [2/4] cadsynth_bias_adjacency seed 0" in summary


def test_the_killed_verdict_carries_its_reason_into_the_summary(swept):
    _, out = swept
    assert "SIGKILL" in out.split("SUMMARY")[-1], (
        "the summary names no cause for the kill, so it is no more diagnosable "
        "than the bare exit code the T2 post-mortem had.")


def test_a_killed_run_is_not_sent_back_to_read_its_own_log(swept):
    """The operational point: KILLED and FAILED need different next steps."""
    _, out = swept
    advice = out.split("SUMMARY")[-1]
    assert "OOM" in advice or "VRAM" in advice
    assert "--force" in advice, (
        "nothing tells the operator how to re-run the killed cells")


def test_a_tier_with_a_killed_run_exits_non_zero(swept):
    rc, out = swept
    assert rc == 1
    assert "This tier is NOT complete" in out


def test_a_killed_cell_does_not_stop_the_sweep(swept):
    """20 runs and a finite GPU budget: cell 1 dying must not cancel cells 3-4."""
    _, out = swept
    assert "[4/4] cadsynth_bias_shuffled seed 0" in out, (
        "the sweep stopped after the killed cell:\n" + out)

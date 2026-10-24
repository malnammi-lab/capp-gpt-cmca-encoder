"""Tests for `runners/check_convergence.py`.

No run exists under the new protocol yet (results/ was archived on 21 Sep), so
every test builds a synthetic `results/<arm>/seed<k>/` tree in the exact shape
`runners/run_all.py` writes: an `eval_metrics.json` carrying the provenance
block `src/utils/provenance.py` produces, plus the `config.resolved.yaml` that
`write_resolved_config` drops beside it.

The point of the audit is that a truncated run must never pass as a converged
one, so most of these tests are about the ways a truncated run could slip
through: labelled `cap` and ignored, labelled something else, never labelled at
all, or labelled `early_stop` while the arithmetic says patience never fired.

Fixtures are local to this file on purpose -- `tests/conftest.py` is shared.
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
    return _load("check_convergence", "runners/check_convergence.py")


@pytest.fixture(scope="module")
def run_all():
    return _load("run_all_for_convergence_test", "runners/run_all.py")


def write_run(root, arm, seed, *, selected_epoch, reason, cap=60, patience=10,
              epoch="same", metrics_name="eval_metrics.json",
              with_config=True, stamped=True, drop_epoch=False,
              config_text=None, cfg_max_epochs=None, log=None):
    """Fabricate one `results/<arm>/seed<k>/` the way run_all lays it out.

    `epoch="same"` mirrors the real trainers: provenance is built inside the
    save-on-improvement branch, so the checkpoint's `epoch` IS the epoch that
    later gets stamped as `selected_epoch`.

    The damage knobs exist because every one of them was a way a truncated run
    got past this audit once:

    drop_epoch      remove the REQUIRED provenance field `epoch` entirely, as
                    opposed to `epoch=<n>` which sets it to a wrong value
    config_text     write config.resolved.yaml with this exact text instead of
                    a well-formed config -- an empty file, or one with the
                    training block damaged. The FILE IS STILL THERE, which is
                    the case `with_config=False` cannot express
    cfg_max_epochs  set training.max_epochs to something other than the cap the
                    provenance records
    log             write this text as the run's train.log, so the validation
                    curve can be read (see `dense_log`)
    """
    run_dir = Path(root) / arm / "seed{}".format(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    prov = {
        "seed": seed,
        "epoch": selected_epoch if epoch == "same" else epoch,
        "selected_epoch": selected_epoch if stamped else None,
        "stopping_reason": reason if stamped else None,
        "use_class_weights": False,
        "epoch_budget_configured": cap,
        "config_hash": "0123456789abcdef",
        "code_fingerprint": "9d88e9213dbb9164",
        "git_commit": None,
        "gpu": "Tesla T4",
        "wall_time_s": 1234.5,
    }
    metrics = {
        "experiment_name": arm,
        "overall_accuracy": 0.9607,
        "macro_f1": 0.88,
        "mean_iou": 0.81,
        "n_samples": 34338,
        "provenance": prov,
    }
    if drop_epoch:
        del prov["epoch"]
    (run_dir / metrics_name).write_text(json.dumps(metrics), encoding="utf-8")

    if with_config:
        if config_text is not None:
            (run_dir / "config.resolved.yaml").write_text(
                config_text, encoding="utf-8")
        else:
            (run_dir / "config.resolved.yaml").write_text(
                yaml.safe_dump({"experiment_name": arm, "seed": seed,
                                "training": {"max_epochs": (cap
                                                            if cfg_max_epochs is None
                                                            else cfg_max_epochs),
                                             "patience": patience,
                                             "batch_size": 4}},
                               sort_keys=False),
                encoding="utf-8")
    if log is not None:
        (run_dir / "train.log").write_text(log, encoding="utf-8")
    return run_dir


def build_tier(root, arms, seeds=(0, 1, 2, 3, 4), **kw):
    for arm in arms:
        for s in seeds:
            write_run(root, arm, s, **kw)


def dense_log(best_epoch, last_epoch, cap=60, val_at=None):
    """A train_semantic-shaped train.log whose val loss bottoms out at
    *best_epoch*. The line format is the one src/training/train_semantic.py
    prints (an "Epoch n/N | ... | Val Loss x | Val Acc y%" line per epoch).

    `val_at` overrides individual epochs, e.g. {7: float("nan")}.
    """
    lines = ["$ python -m src.training.train_semantic configs/x.yaml", ""]
    for e in range(1, last_epoch + 1):
        v = 1.0 - 0.02 * min(e, best_epoch)
        if e > best_epoch:
            v += 0.004 * (e - best_epoch)
        if val_at and e in val_at:
            v = val_at[e]
        lines.append("Epoch {:3d}/{} | Loss 0.5000 | Acc 90.00% | "
                     "Val Loss {:.4f} | Val Acc 88.00%".format(e, cap, v))
    return "\n".join(lines) + "\n"


def sparse_log(epochs, best_epoch, cap=60):
    """A train_resnet-shaped log: src/training/train_resnet.py:401 prints an
    epoch line only every 10th epoch, so the curve has holes in it."""
    lines = ["$ python -m src.training.train_resnet configs/x.yaml", ""]
    for e in epochs:
        v = 1.0 - 0.02 * min(e, best_epoch)
        if e > best_epoch:
            v += 0.004 * (e - best_epoch)
        lines.append("Epoch {:3d}/{} | Loss 0.5000 | Acc 90.00% | LR 1.00e-03 "
                     "| Val Loss {:.4f} | Val Acc 88.00%".format(e, cap, v))
    return "\n".join(lines) + "\n"


def t1_arms(run_all):
    return [c.name for c in run_all.TIERS["T1"]]


# --------------------------------------------------------------------------
# 0. the inputs this audit is built on must actually exist
# --------------------------------------------------------------------------

def test_provenance_emits_the_two_fields_this_audit_reads():
    """The audit is worthless if the trainers stopped recording its inputs."""
    from src.utils.provenance import STOPPING_REASONS, check_regiment_fields

    assert STOPPING_REASONS == ("early_stop", "cap"), (
        "the cap audit matches on the literal string 'cap'; "
        "STOPPING_REASONS changed to {}".format(STOPPING_REASONS,))

    complete = {"seed": 0, "epoch": 22, "config_hash": "x", "gpu": "T4",
                "wall_time_s": 1.0, "selected_epoch": 22,
                "stopping_reason": "early_stop", "code_fingerprint": "y"}
    assert check_regiment_fields(complete) == []

    for dropped in ("selected_epoch", "stopping_reason"):
        partial = dict(complete)
        partial[dropped] = None
        assert dropped in check_regiment_fields(partial), (
            "{} is no longer a required regiment field, so a run could reach "
            "the cap audit without it".format(dropped))


def test_every_trainer_records_a_stopping_reason():
    """A trainer that never stamps one produces runs this audit must reject."""
    trainers = ["train_semantic.py", "train_resnet.py", "train_unified.py"]
    for t in trainers:
        src = (CODEBASE / "src" / "training" / t).read_text(encoding="utf-8")
        assert "stamp_selection(" in src, "{} never stamps selection".format(t)
        assert 'stopping_reason = "early_stop" if stopped_early else "cap"' in src, (
            "{} no longer derives stopping_reason the way the audit "
            "assumes".format(t))


# --------------------------------------------------------------------------
# 1. the happy path
# --------------------------------------------------------------------------

def test_all_seeds_early_stopped_passes(cc, run_all, tmp_path, capsys):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop")

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "IS NOT BUDGET-CONFOUNDED" in out
    for arm in arms:
        assert "{}".format(arm) in out
    assert "CAP-BOUND" not in out


def test_selected_epochs_may_differ_between_arms(cc, run_all, tmp_path, capsys):
    """The selection rule EXPECTS unequal epochs; only the cap makes a run unfair."""
    arms = t1_arms(run_all)
    for i, arm in enumerate(arms):
        build_tier(tmp_path, [arm], selected_epoch=12 + 7 * i,
                   reason="early_stop")

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    assert rc == 0, capsys.readouterr().out


# --------------------------------------------------------------------------
# 2. the failure this script exists for
# --------------------------------------------------------------------------

def test_one_cap_bound_seed_fails_the_whole_arm(cc, run_all, tmp_path, capsys):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop")
    # one seed of one arm ran out of budget
    write_run(tmp_path, arms[2], 3, selected_epoch=58, reason="cap")

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "TRUNCATED" in out
    assert "{} seed 3".format(arms[2]) in out
    assert "CAP-BOUND -- 1/5" in out
    assert "re-run" in out.lower()
    # the arms that did converge are not slandered
    assert "CONVERGED -- 5/5" in out


def test_a_cap_bound_run_is_never_reported_as_converged(cc, tmp_path):
    write_run(tmp_path, "arm", 0, selected_epoch=58, reason="cap")
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    assert row["verdict"] == cc.CAP_BOUND
    assert row["errors"]


# --------------------------------------------------------------------------
# 3. ways a truncated run could hide from the audit
# --------------------------------------------------------------------------

def test_unstamped_run_is_not_scored_as_converged(cc, run_all, tmp_path, capsys):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop")
    write_run(tmp_path, arms[0], 1, selected_epoch=30, reason="early_stop",
              stamped=False)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "UNSTAMPED" in out
    assert "INCOMPLETE -- seeds [1]" in out
    assert "loop never finished" in out


def test_illegal_stopping_reason_is_rejected_not_ignored(cc, tmp_path):
    write_run(tmp_path, "arm", 0, selected_epoch=30, reason="finished")
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    assert row["verdict"] == cc.INVALID
    assert "not one of" in row["errors"][0]
    assert "hole in the audit" in row["errors"][0]


def test_early_stop_too_close_to_the_cap_is_impossible(cc, tmp_path):
    """patience 10, cap 60: selecting epoch 55 cannot have early-stopped."""
    write_run(tmp_path, "arm", 0, selected_epoch=55, reason="early_stop",
              cap=60, patience=10)
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    assert row["errors"], "a mislabelled cap-bound run passed the audit"
    assert "cap-bound and mislabelled" in row["errors"][0]
    assert row["verdict"] == cc.INCONSISTENT, (
        "a run that failed a consistency check was still labelled {} in the "
        "table".format(row["verdict"]))


def test_a_failing_run_is_never_labelled_converged_in_the_table(
        cc, run_all, tmp_path, capsys):
    """The verdict column must not reassure about a seed that has a FAIL."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop")
    write_run(tmp_path, arms[3], 0, selected_epoch=55, reason="early_stop")

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "INCONSISTENT" in out
    assert "NOT SOUND -- seeds [0] contradict their own stopping record" in out
    assert "{}".format(arms[3]) in out


def test_cap_label_with_an_early_selection_is_impossible(cc, tmp_path):
    """patience 10, cap 60: selecting epoch 20 means patience fired at 30."""
    write_run(tmp_path, "arm", 0, selected_epoch=20, reason="cap",
              cap=60, patience=10)
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    msgs = " ".join(row["errors"])
    assert "label and the loop disagree" in msgs


def test_epoch_and_selected_epoch_must_agree(cc, tmp_path):
    """A last-epoch save would leave epoch != selected_epoch."""
    write_run(tmp_path, "arm", 0, selected_epoch=22, reason="early_stop",
              epoch=32)
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    msgs = " ".join(row["errors"])
    assert "epoch=32 but selected_epoch=22" in msgs
    assert "not the selected ones" in msgs


def test_selected_epoch_beyond_the_cap_is_impossible(cc, tmp_path):
    write_run(tmp_path, "arm", 0, selected_epoch=75, reason="cap", cap=60)
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    assert any("exceeds the cap" in e for e in row["errors"])


def test_a_missing_cap_is_an_error_not_a_pass(cc, tmp_path):
    run_dir = write_run(tmp_path, "arm", 0, selected_epoch=22,
                        reason="early_stop")
    m = json.loads((run_dir / "eval_metrics.json").read_text())
    del m["provenance"]["epoch_budget_configured"]
    (run_dir / "eval_metrics.json").write_text(json.dumps(m), encoding="utf-8")

    row = cc.audit_run("arm", 0, run_dir)
    assert any("epoch_budget_configured" in e for e in row["errors"])


# --------------------------------------------------------------------------
# 4. completeness and comparability
# --------------------------------------------------------------------------

def test_a_missing_seed_fails_the_arm(cc, run_all, tmp_path, capsys):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, seeds=(0, 1, 2, 3), selected_epoch=22,
               reason="early_stop")

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "MISSING" in out
    assert "no metrics file" in out
    # nothing was cap-bound, so it must not prescribe the cap remedy
    assert "raised cap" not in out
    assert "missing or unfinished runs" in out


def test_arms_held_to_different_caps_are_not_comparable(cc, run_all, tmp_path,
                                                        capsys):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms[:3], selected_epoch=22, reason="early_stop",
               cap=60)
    build_tier(tmp_path, arms[3:], selected_epoch=22, reason="early_stop",
               cap=100)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "DIFFERENT caps [60, 100]" in out


def test_arms_held_to_different_patience_are_not_comparable(cc, run_all,
                                                            tmp_path, capsys):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms[:3], selected_epoch=22, reason="early_stop",
               patience=10)
    build_tier(tmp_path, arms[3:], selected_epoch=22, reason="early_stop",
               patience=20)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "DIFFERENT patience values [10, 20]" in out


def test_absent_resolved_config_is_announced_not_silently_skipped(
        cc, run_all, tmp_path, capsys):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop",
               with_config=False)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "NOT PERFORMED" in out, (
        "the patience check was skipped without saying so")


# --------------------------------------------------------------------------
# 5. shapes of the real tree
# --------------------------------------------------------------------------

def test_unified_metrics_filename_is_audited(cc, run_all, tmp_path, capsys):
    """T5's evaluator writes unified_eval_metrics.json, not eval_metrics.json."""
    arm = run_all.TIERS["T5"][0].name
    for s in range(5):
        write_run(tmp_path, arm, s, selected_epoch=41, reason="cap",
                  metrics_name="unified_eval_metrics.json")

    rc = cc.main(["--tier", "T5", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "MISSING" not in out, "the unified metrics layout was not read"
    assert "CAP-BOUND -- 5/5" in out


def test_scouting_tier_reports_but_does_not_gate(cc, run_all, tmp_path, capsys):
    scout_root = tmp_path / "scouting"
    arms = [c.name for c in run_all.TIERS["T7"]]
    build_tier(scout_root, arms, selected_epoch=58, reason="cap")

    rc = cc.main(["--tier", "T7", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "CAP-BOUND" in out, "a scouting failure was hidden, not reported"
    assert "Exit code forced to 0" in out


def test_single_experiment_mode(cc, tmp_path, capsys):
    for s in (0, 1, 2):
        write_run(tmp_path, "cadsynth_bias_both", s, selected_epoch=31,
                  reason="early_stop")

    rc = cc.main(["--experiment", "cadsynth_bias_both", "--seeds", "0", "1", "2",
                  "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "EXPERIMENT CADSYNTH_BIAS_BOTH" in out


# --------------------------------------------------------------------------
# 6. CLI contract
# --------------------------------------------------------------------------

def test_unknown_tier_exits_2(cc, tmp_path, capsys):
    rc = cc.main(["--tier", "T99", "--results", str(tmp_path), "--no-report"])
    assert rc == 2
    assert "unknown tier" in capsys.readouterr().out


def test_missing_results_dir_exits_2(cc, tmp_path, capsys):
    rc = cc.main(["--tier", "T1", "--results", str(tmp_path / "nope"),
                  "--no-report"])
    assert rc == 2
    assert "no results directory" in capsys.readouterr().out


def test_unknown_experiment_exits_2_and_points_at_the_scouting_root(
        cc, tmp_path, capsys):
    """A scouting arm lives under results/scouting/, not results/."""
    write_run(tmp_path / "scouting", "mfinstseg_bias_both_weighted", 0,
              selected_epoch=18, reason="early_stop")

    rc = cc.main(["--experiment", "mfinstseg_bias_both_weighted",
                  "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 2
    assert "no results directory" in out
    assert "--results {}/scouting".format(tmp_path) in out


def test_report_states_the_rule_and_the_verdict(cc, run_all, tmp_path):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop")
    write_run(tmp_path, arms[1], 0, selected_epoch=57, reason="cap")

    report = tmp_path / "convergence_T1.md"
    rc = cc.main(["--tier", "T1", "--results", str(tmp_path),
                  "--report", str(report)])
    assert rc == 1

    text = report.read_text(encoding="utf-8")
    assert "# Cap audit -- tier T1" in text
    assert "Cap(s) recorded by these runs: [60]" in text
    assert "Patience: [10]" in text
    assert "| {} | 0 | 57 | 60 | `cap` | CAP-BOUND |".format(arms[1]) in text
    assert "## Remedy" in text
    assert "never one arm alone" not in text  # the printed line, not the report
    assert "Extending one arm alone" in text


def test_report_defaults_into_the_results_dir(cc, run_all, tmp_path):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop")

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "convergence_T1.md").is_file()


# --------------------------------------------------------------------------
# 7. the false passes a verifier found on 21 Sep -- one test per defect
#
# Every tree below records selected_epoch=55 under cap 60 with patience 10.
# That is physically impossible for an early_stop: patience would have had to
# run to epoch 65, eleven epochs past the cap. So every one of these arms is
# REALLY cap-bound, and the patience arithmetic is the only check that can say
# so. Anything that lets that check be skipped while the tier still passes is
# the defect these tests exist to hold shut.
# --------------------------------------------------------------------------

IMPOSSIBLE = dict(selected_epoch=55, reason="early_stop", cap=60, patience=10)


def test_a_config_that_cannot_supply_patience_is_fatal_not_skipped(
        cc, run_all, tmp_path, capsys):
    """DEFECT 1. config.resolved.yaml IS there; only training.patience is gone.

    The old code returned None from `_patience_of` and the arithmetic check --
    the sharpest in the file -- was skipped, so all 20 cap-bound seeds printed
    CONVERGED and the tier exited 0.
    """
    arms = t1_arms(run_all)
    cfg = yaml.safe_dump({"experiment_name": "x", "seed": 0,
                          "training": {"max_epochs": 60, "batch_size": 4}})
    build_tier(tmp_path, arms, config_text=cfg, **IMPOSSIBLE)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, "a tier of 20 cap-bound seeds passed with patience missing"
    assert "CONVERGED -- 5/5" not in out
    assert "IS NOT BUDGET-CONFOUNDED" not in out
    assert "sets no training.patience" in out
    assert "corrupt artifact" in out
    # and it must not be blamed on the cap being fine: the audit could not ask
    assert "CAP UNSETTLED" in out


def test_an_empty_resolved_config_is_fatal_not_skipped(cc, run_all, tmp_path,
                                                       capsys):
    """DEFECT 2. Same false pass through `yaml.safe_load(...) or {}`."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, config_text="", **IMPOSSIBLE)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, "an empty config.resolved.yaml still skipped the check"
    assert "the file is 0 bytes" in out
    assert "IS NOT BUDGET-CONFOUNDED" not in out


def test_unparseable_resolved_config_is_fatal(cc, tmp_path):
    write_run(tmp_path, "arm", 0, config_text="training: [: :\n", **IMPOSSIBLE)
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    assert row["verdict"] != cc.CONVERGED
    assert any("not parseable YAML" in e for e in row["errors"])


def test_a_non_integer_patience_is_fatal(cc, tmp_path):
    cfg = yaml.safe_dump({"training": {"max_epochs": 60, "patience": "ten"}})
    write_run(tmp_path, "arm", 0, config_text=cfg, **IMPOSSIBLE)
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    assert row["verdict"] != cc.CONVERGED
    assert any("not a positive integer" in e for e in row["errors"])


def test_a_malformed_config_is_not_reported_as_a_missing_one(
        cc, run_all, tmp_path, capsys):
    """DEFECT 3. The note was keyed on `patience is None`, not on absence, so a
    damaged config sent the reader hunting for a file already in the run dir."""
    arms = t1_arms(run_all)
    cfg = yaml.safe_dump({"training": {"max_epochs": 60, "batch_size": 4}})
    build_tier(tmp_path, arms, config_text=cfg, selected_epoch=22,
               reason="early_stop")

    cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    for arm in arms:
        for s in range(5):
            assert (tmp_path / arm / "seed{}".format(s)
                    / "config.resolved.yaml").is_file()
    assert "no config.resolved.yaml" not in out, (
        "the audit said the file was missing while it was sitting in all 20 "
        "run directories")
    assert "sets no training.patience" in out


def test_an_absent_config_is_still_reported_as_absent(cc, run_all, tmp_path,
                                                      capsys):
    """The other half of defect 3: a genuinely absent file must still say so."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop",
               with_config=False)

    cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert "no config.resolved.yaml" in out
    assert "NOT PERFORMED" in out
    assert "did not come from run_all" in out


def test_the_markdown_remedy_never_contradicts_the_printed_one(
        cc, run_all, tmp_path):
    """DEFECT 4. The markdown appended the cap remedy whenever `errors` was
    non-empty, with no cap-bound test, while stdout branched correctly. On a
    tier whose only fault was an unrun seed the two disagreed in the SAME
    invocation -- and the markdown is the artifact kept and read weeks later."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, seeds=(0, 1, 2, 3), selected_epoch=22,
               reason="early_stop")

    report = tmp_path / "convergence_T1.md"
    rc = cc.main(["--tier", "T1", "--results", str(tmp_path),
                  "--report", str(report)])
    assert rc == 1
    text = report.read_text(encoding="utf-8")

    assert "## Remedy" in text
    assert "raised cap" not in text, (
        "the markdown prescribed a raised cap for a tier whose only fault was "
        "a seed that never ran")
    assert "No run in this audit was cap-bound" in text
    assert "Missing runs" in text


def test_stdout_and_markdown_agree_about_the_cap_on_every_shape(
        cc, run_all, tmp_path):
    """The two renderings are driven by one decision, so pick four shapes and
    require them to agree about whether the cap is the problem."""
    arms = t1_arms(run_all)
    shapes = {
        "cap": dict(selected_epoch=58, reason="cap"),
        "mislabelled": dict(selected_epoch=55, reason="early_stop"),
        "selection": dict(selected_epoch=22, reason="early_stop", epoch=33),
        "unstamped": dict(selected_epoch=22, reason="early_stop",
                          stamped=False),
    }
    for name, kw in shapes.items():
        root = tmp_path / name
        build_tier(root, arms, selected_epoch=22, reason="early_stop")
        write_run(root, arms[1], 2, **kw)
        report = root / "r.md"
        cc.main(["--tier", "T1", "--results", str(root), "--report",
                 str(report)])
        rows, _, _, _ = cc.audit(arms, root, [0, 1, 2, 3, 4])
        kinds = cc.remedy_kinds(rows)
        text = report.read_text(encoding="utf-8")
        cap_in_md = "raised cap" in text
        assert cap_in_md == (cc.REMEDY_CAP in kinds), (
            "{}: markdown says raised-cap={} but the remedy kinds are {}"
            .format(name, cap_in_md, kinds))


def test_an_inconsistent_run_does_not_trigger_the_cap_remedy(
        cc, run_all, tmp_path, capsys):
    """DEFECT 5. INCONSISTENT is also raised by the epoch != selected_epoch
    check, which is a checkpoint-selection bug. Raising the cap cannot fix a
    last-epoch save, and sending someone to re-run 20 GPU-days is not free."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop")
    write_run(tmp_path, arms[2], 1, selected_epoch=22, reason="early_stop",
              epoch=32)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "INCONSISTENT" in out
    assert "raised cap" not in out, (
        "a last-epoch save was prescribed the budget remedy")
    assert "SELECTION:" in out
    assert "Raising the cap cannot fix a checkpoint-selection fault" in out


def test_a_missing_epoch_field_is_not_a_pass(cc, run_all, tmp_path, capsys):
    """DEFECT 6. `epoch` is a REQUIRED regiment field
    (src/utils/provenance.py:307-308). The old code read it with .get() and
    guarded the comparison with `if epoch is not None`, so deleting it made the
    row print CONVERGED with exit 0 and no error at all."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop",
               drop_epoch=True)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, "a run that cannot say what epoch it is passed the audit"
    assert "CONVERGED -- 5/5" not in out
    assert "required regiment field(s) ['epoch']" in out


def test_a_missing_epoch_field_alone_is_reported_as_unstamped(cc, tmp_path):
    write_run(tmp_path, "arm", 0, selected_epoch=22, reason="early_stop",
              drop_epoch=True)
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    assert row["verdict"] == cc.UNSTAMPED
    assert row["errors"]


def test_an_unstamped_run_leaves_the_cap_question_open(cc, run_all, tmp_path,
                                                       capsys):
    """DEFECT 7. src/training/train_semantic.py returns at its
    `if start_epoch > max_epochs:` branch before `stamp_selection`, so a
    checkpoint already sitting AT THE CAP comes back unstamped. The tool used
    to answer "No arm was cap-bound" for exactly the case it exists to catch."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop")
    write_run(tmp_path, arms[0], 4, selected_epoch=22, reason="early_stop",
              stamped=False)

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "No arm was cap-bound" not in out
    assert "No run in this audit was cap-bound" not in out
    assert "CAP UNSETTLED" in out
    assert "does NOT establish" in out


def test_the_metrics_filenames_come_from_run_all(cc, run_all, tmp_path):
    """DEFECT 8. The filenames were a second copy of run_all's list and had
    fallen a name behind it, so a link-evaluated arm read as 20 missing runs.

    Read run_all's list out of its own source so that a fifth name added there
    is checked here without this test being edited.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(run_all.metrics_path).lstrip())
    names = sorted({n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and n.value.endswith(".json")})
    assert "link_eval_metrics.json" in names, (
        "run_all.metrics_path no longer lists the link evaluator's filename; "
        "this test is reading the wrong function")

    for i, name in enumerate(names):
        run_dir = write_run(tmp_path, "arm{}".format(i), 0, selected_epoch=22,
                            reason="early_stop", metrics_name=name)
        assert cc._metrics_file(run_dir) == run_dir / name, (
            "check_convergence does not read {}, which run_all writes"
            .format(name))

    # structural, not behavioural: a second copy of the list is the defect,
    # and a copy that is correct TODAY is still the defect, because it is the
    # copy that goes stale the next time run_all gains a filename.
    body = inspect.getsource(cc._metrics_file)
    assert "metrics_path" in body, (
        "_metrics_file no longer delegates to run_all.metrics_path")
    assert ".json" not in body, (
        "the filename list has been copied back into check_convergence.py; "
        "get it from run_all.metrics_path so the two cannot drift again")
    assert not hasattr(cc, "METRICS_NAMES"), (
        "the private METRICS_NAMES copy is back")


def test_a_corrupt_metrics_file_raises_and_names_the_run(cc, run_all, tmp_path):
    """DEFECT 9. It did raise -- correctly -- but the traceback stopped inside
    json and never said which of 20 runs to open."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop")
    bad = tmp_path / arms[3] / "seed2" / "eval_metrics.json"
    bad.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])

    msg = str(exc.value)
    assert arms[3] in msg and "seed 2" in msg, msg
    assert str(bad) in msg, msg
    assert isinstance(exc.value.__cause__, json.JSONDecodeError)


def test_metrics_without_a_provenance_block_is_not_a_pass(cc, tmp_path):
    run_dir = write_run(tmp_path, "arm", 0, selected_epoch=22,
                        reason="early_stop")
    (run_dir / "eval_metrics.json").write_text(
        json.dumps({"overall_accuracy": 0.96}), encoding="utf-8")
    row = cc.audit_run("arm", 0, run_dir)
    assert row["verdict"] == cc.INVALID
    assert any("no `provenance` block" in e for e in row["errors"])


def test_a_cap_the_config_does_not_agree_with_is_an_error(cc, tmp_path):
    """epoch_budget_configured is copied from training.max_epochs
    (src/utils/provenance.py:227), so a disagreement means one of the two
    artifacts does not belong to this run."""
    write_run(tmp_path, "arm", 0, selected_epoch=22, reason="early_stop",
              cap=60, cfg_max_epochs=100)
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    assert row["verdict"] != cc.CONVERGED
    assert any("does not belong to this run" in e for e in row["errors"])


# --------------------------------------------------------------------------
# 8. val-curve sanity (DEFECT 11) -- the piece of the spec's demotion header
#    that had never been implemented and whose absence was never disclosed
# --------------------------------------------------------------------------

def test_the_val_curve_parser_reads_every_trainer_line_format():
    """One regex serves three trainers; if any of them changes its epoch line
    the curve silently empties, so the formats are pinned here."""
    cc_mod = _load("cc_parser", "runners/check_convergence.py")
    text = "\n".join([
        # src/training/train_semantic.py
        "Epoch   1/60 | Loss 1.2289 | Acc 65.10% | Val Loss 0.9010 | Val Acc 70.00%",
        # src/training/train_resnet.py (an LR field in the middle)
        "Epoch   2/60 | Loss 0.7655 | Acc 76.65% | LR 1.00e-03 | Val Loss 0.8000 | Val Acc 74.00%",
        # src/training/train_unified.py ("Val", not "Val Loss")
        "Epoch   3/60 | Loss 0.5774 | Val 0.7100 | Sem 82.03% | Bot 90.00% | InstAUC 0.910 | Thresh 0.500",
        "Test accuracy 96.07%",
    ])
    assert cc_mod.parse_val_curve(text) == {1: 0.9010, 2: 0.8000, 3: 0.7100}
    assert cc_mod.parse_val_curve("no epoch lines here") == {}


def test_a_checkpoint_that_is_not_the_val_minimum_fails(cc, run_all, tmp_path,
                                                        capsys):
    """The run says it selected epoch 22; its own validation curve bottoms out
    at 31. Whatever weights are being scored, they are not the best-val ones
    the selection rule says to score."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop",
               log=dense_log(best_epoch=31, last_epoch=41))

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 1, "a checkpoint that is not the val-curve minimum passed"
    assert "val-curve sanity FAILED" in out
    assert "raised cap" not in out, (
        "a selection fault was prescribed the budget remedy")


def test_a_curve_that_agrees_with_the_selection_passes(cc, run_all, tmp_path,
                                                       capsys):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop",
               log=dense_log(best_epoch=22, last_epoch=32))

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "val-curve sanity" in out
    assert "min 0.5600 @22" in out


def test_a_tie_inside_the_logs_precision_is_not_called_a_failure(cc, tmp_path):
    """train.log prints 4 decimals. Two epochs whose true losses differ by less
    than that print the same number, and calling that a selection bug would
    fail a sound run -- which on an 85-run batch costs GPU-days."""
    log = dense_log(best_epoch=22, last_epoch=32, val_at={17: 0.5600})
    write_run(tmp_path, "arm", 0, selected_epoch=22, reason="early_stop",
              log=log)
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    assert row["errors"] == []
    assert "tied within" in row["val_curve"]


def test_a_nan_validation_loss_fails(cc, tmp_path):
    log = dense_log(best_epoch=22, last_epoch=32,
                    val_at={7: float("nan")})
    write_run(tmp_path, "arm", 0, selected_epoch=22, reason="early_stop",
              log=log)
    row = cc.audit_run("arm", 0, tmp_path / "arm" / "seed0")
    assert any("nan at epoch" in e for e in row["errors"])
    assert row["verdict"] != cc.CONVERGED


def test_a_sparse_log_is_declared_not_performed_never_assumed_good(
        cc, run_all, tmp_path, capsys):
    """src/training/train_resnet.py:401 logs only every 10th epoch, so the
    lowest LOGGED loss need not be the lowest reached. That must be announced,
    not silently treated as a pass -- and it must not be treated as a failure
    either."""
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop",
               log=sparse_log([1, 10, 20, 30], best_epoch=22))

    rc = cc.main(["--tier", "T1", "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "NOT PERFORMED" in out
    assert "every 10th epoch" in out
    assert "not performed" in out.lower()


def test_a_run_without_a_train_log_says_so_by_name(cc, tmp_path, capsys):
    write_run(tmp_path, "cadsynth_bias_both", 0, selected_epoch=22,
              reason="early_stop")
    rc = cc.main(["--experiment", "cadsynth_bias_both", "--seeds", "0",
                  "--results", str(tmp_path), "--no-report"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "NOT PERFORMED" in out
    assert "no train.log in the run directory" in out
    assert "cadsynth_bias_both seed 0" in out


def test_the_report_carries_the_val_curve_section(cc, run_all, tmp_path):
    arms = t1_arms(run_all)
    build_tier(tmp_path, arms, selected_epoch=22, reason="early_stop",
               log=dense_log(best_epoch=22, last_epoch=32))
    report = tmp_path / "r.md"
    rc = cc.main(["--tier", "T1", "--results", str(tmp_path),
                  "--report", str(report)])
    assert rc == 0
    text = report.read_text(encoding="utf-8")
    assert "## Val-curve sanity" in text
    assert "min 0.5600 @22" in text


def test_the_spec_demotion_header_still_asks_for_val_curve_sanity():
    """If the spec is ever rewritten, this test is the reminder that the shape
    of the audit was pinned to that sentence.

    [FIX 22 Sep 2026] `advisor-guidance/` is the working folder; it is not part of the
    repository that goes to GitHub.
    So the test passed here and raised FileNotFoundError there, which is how it
    was found: `scripts/sync-to-colab-copy.py` runs the suite inside the synced
    copy and refused the sync.

    It would have failed for anyone who cloned the released repo, too.

    The assertion is still worth keeping where the spec IS present, because it
    is the only thing tying this tool's scope to the sentence that set it. So it
    skips rather than fails when the spec is absent -- an absent private
    document is not a defect in the released code.
    """
    spec_path = (CODEBASE.parent / "advisor-guidance"
                 / "spec-check-convergence.md")
    if not spec_path.exists():
        pytest.skip(
            f"{spec_path.name} is not present at {spec_path.parent} -- this is "
            f"the working folder, which does not ship with the "
            f"codebase. Run this from the full project tree to exercise it.")
    spec = spec_path.read_text(encoding="utf-8")
    header = spec.split("\n", 1)[0]
    assert "val-curve sanity" in header.lower(), (
        "the demotion header no longer asks for val-curve sanity; "
        "re-read it before trusting runners/check_convergence.py's scope")

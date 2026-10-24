"""
Tests for src/utils/seeding.py and src/utils/provenance.py.

These guard the three regiment requirements that had no enforcement before
2026-09-17:

  1. a run without a seed must fail loudly rather than train unseeded
  2. five seeds of one arm must not overwrite each other
  3. every checkpoint must carry the six provenance fields

Requirement 2 is the one that would have cost real GPU time: T1 is 4 arms x 5
seeds and every config wrote to a single fixed path, so 16 of the 20 runs would
have been silently destroyed.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.provenance import (build_provenance, check_regiment_fields,
                                  code_fingerprint, config_hash,
                                  provenance_from_checkpoint)
from src.utils.seeding import make_generator, seed_scoped, set_global_seed

CODEBASE = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- #
#  1. an unseeded run must be impossible                                 #
# --------------------------------------------------------------------- #

def test_seed_none_raises():
    """The old code silently skipped seeding when `seed:` was absent."""
    with pytest.raises(TypeError, match="no seed"):
        set_global_seed(None)


@pytest.mark.parametrize("bad", ["42", 4.2, True, [0]])
def test_non_integer_seed_raises(bad):
    with pytest.raises(TypeError):
        set_global_seed(bad)


def test_negative_seed_raises():
    with pytest.raises(ValueError):
        set_global_seed(-1)


def test_generator_requires_a_seed():
    with pytest.raises(TypeError):
        make_generator(None)


# --------------------------------------------------------------------- #
#  2. seeding actually controls all four generators                      #
# --------------------------------------------------------------------- #

def _draw():
    return (torch.randn(4).tolist(), np.random.rand(4).tolist(),
            [random.random() for _ in range(4)])


def test_same_seed_reproduces_all_three_generators():
    set_global_seed(7)
    first = _draw()
    set_global_seed(7)
    assert _draw() == first


def test_different_seeds_diverge():
    set_global_seed(0)
    a = _draw()
    set_global_seed(1)
    assert _draw() != a


def test_cudnn_is_pinned():
    """The regiment names cudnn explicitly."""
    set_global_seed(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_returns_the_seed():
    assert set_global_seed(3) == 3


# --------------------------------------------------------------------- #
#  3. seed-scoped paths: the anti-overwrite guarantee                    #
# --------------------------------------------------------------------- #

def test_five_seeds_produce_five_distinct_checkpoints():
    """THE regression test for the 16-of-20-runs-destroyed defect."""
    base = "results/checkpoints/cadsynth_bias_both.pth"
    paths = {seed_scoped(base, s) for s in range(5)}
    assert len(paths) == 5, f"seeds collide: {paths}"


def test_five_seeds_produce_five_distinct_figure_dirs():
    base = "results/figures/cadsynth_bias_both/"
    dirs = {seed_scoped(base, s, is_dir=True) for s in range(5)}
    assert len(dirs) == 5


def test_scoped_file_layout():
    got = seed_scoped("results/checkpoints/arm.pth", 3)
    assert got == Path("results/checkpoints/arm/seed3.pth")


def test_scoped_dir_layout():
    got = seed_scoped("results/figures/arm/", 3, is_dir=True)
    assert got == Path("results/figures/arm/seed3")


def test_none_seed_leaves_every_existing_path_untouched():
    """Backward compatibility: results produced before today must stay readable."""
    for p, is_dir in (("results/checkpoints/arm.pth", False),
                      ("results/figures/arm/", True)):
        assert seed_scoped(p, None, is_dir=is_dir) == Path(p)


# --------------------------------------------------------------------- #
#  4. provenance                                                         #
# --------------------------------------------------------------------- #

def test_provenance_carries_every_regiment_field():
    prov = build_provenance({"a": 1}, seed=2, epoch=35, wall_time_s=12.34,
                            selected_epoch=17, stopping_reason="early_stop")
    assert check_regiment_fields(prov) == []
    assert prov["seed"] == 2 and prov["epoch"] == 35
    assert prov["selected_epoch"] == 17
    assert prov["stopping_reason"] == "early_stop"
    assert prov["wall_time_s"] == 12.3
    assert len(prov["code_fingerprint"]) == 16
    # `assert prov["gpu"]` was here and could not fail: gpu_name() returns a
    # non-empty string on both branches. Assert it says which branch instead.
    gpu = prov["gpu"]
    if torch.cuda.is_available():
        assert gpu == torch.cuda.get_device_name(0)
    else:
        assert "cpu" in gpu.lower(), f"no CUDA here, but gpu reads {gpu!r}"


def test_check_regiment_fields_names_what_is_missing():
    prov = build_provenance({"a": 1}, seed=0, epoch=None, wall_time_s=None)
    missing = check_regiment_fields(prov)
    assert "epoch" in missing and "wall_time_s" in missing


def test_config_hash_tracks_values_not_formatting():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_code_fingerprint_is_stable_and_short():
    a, b = code_fingerprint(), code_fingerprint()
    assert a == b and len(a) == 16


def test_code_fingerprint_refuses_an_empty_tree(tmp_path):
    """Silently hashing nothing would give every run the same fake provenance."""
    with pytest.raises(FileNotFoundError):
        code_fingerprint(tmp_path)


def test_eval_provenance_is_carried_from_the_checkpoint_not_reinvented():
    """Re-scoring elsewhere must not overwrite the training run's GPU or time."""
    trained = build_provenance({"a": 1}, seed=4, epoch=35, wall_time_s=900.0,
                               selected_epoch=22, stopping_reason="early_stop")
    trained["gpu"] = "Tesla V100"                       # pretend it trained there
    out = provenance_from_checkpoint({"provenance": trained}, {"a": 1})
    assert out["seed"] == 4 and out["epoch"] == 35
    assert out["gpu"] == "Tesla V100"                   # not this machine's
    assert out["wall_time_s"] == 900.0
    assert out["selected_epoch"] == 22                  # carried, not recomputed
    assert out["stopping_reason"] == "early_stop"
    assert check_regiment_fields(out) == []
    assert "evaluated" in out                           # this pass is recorded too


def test_a_checkpoint_without_provenance_is_marked_not_fabricated():
    """An old checkpoint must produce a file that reads as unquotable."""
    out = provenance_from_checkpoint({"epoch": 12}, {"a": 1})
    assert "provenance_missing" in out
    assert "seed" not in out                            # nothing invented
    assert check_regiment_fields(out) != []             # and it fails the gate


# --------------------------------------------------------------------- #
#  5. one shared class-weighting policy (the class-weighting rule)             #
# --------------------------------------------------------------------- #

def test_absent_class_weighting_raises_instead_of_defaulting():
    """A config that omits use_class_weights raises KeyError."""
    from src.utils.class_weights import class_weighting_enabled
    with pytest.raises(KeyError, match="use_class_weights"):
        class_weighting_enabled({"type": "cross_entropy"}, where="test")


@pytest.mark.parametrize("value", [True, False])
def test_class_weighting_returns_what_the_config_says(value):
    from src.utils.class_weights import class_weighting_enabled
    assert class_weighting_enabled({"use_class_weights": value}, where="t") is value


@pytest.mark.parametrize("bad", ["true", 1, None, "yes"])
def test_non_boolean_class_weighting_raises(bad):
    """`use_class_weights: "false"` is a truthy string and would weight silently."""
    from src.utils.class_weights import class_weighting_enabled
    with pytest.raises(TypeError):
        class_weighting_enabled({"use_class_weights": bad}, where="t")


def test_every_paper_run_shares_one_class_weighting_policy():
    """Decision 2: one policy across every arm AND the ResNet baseline.

    Scope is taken from `runners/run_all.py`'s tier table rather than from a
    glob, because that table is the authoritative list of runs whose numbers may
    enter the paper, and it knows which trainer each config uses. Configs driven
    by train_unified are excluded: that trainer has no class-weighting path at
    all, so its arms are unweighted by construction (asserted separately below).
    """
    import importlib.util
    import yaml

    spec = importlib.util.spec_from_file_location(
        "run_all", CODEBASE / "runners" / "run_all.py")
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)

    reads_the_key = {"src.training.train_semantic", "src.training.train_resnet"}

    def policy(cfg_path):
        loss = yaml.safe_load(
            (CODEBASE / cfg_path).read_text(encoding="utf-8")).get("loss", {})
        return loss.get("use_class_weights", "ABSENT")

    # T4 is the ONE deliberate exception: decision 2 asks for the headline
    # policy everywhere plus "the other setting in one footnote". Excluding it
    # here and asserting it separately below is what makes it a footnote rather
    # than a second, undeclared policy.
    # [21 Sep 2026] was `t != "T4"`. T6 and T7 are weighted on purpose too.
    headline = [c for t, cells in run_all.TIERS.items()
                if t not in run_all.WEIGHTED_TIERS
                for c in cells if c.trainer in reads_the_key]
    assert headline, "no paper runs found -- the test is not testing anything"

    values = {c.config: policy(c.config) for c in headline}
    absent = sorted(n for n, v in values.items() if v == "ABSENT")
    assert not absent, (
        f"these paper configs set no class-weighting policy, so the trainer "
        f"would raise at launch: {absent}")
    assert set(values.values()) == {False}, (
        f"decision 2 requires ONE shared policy (unweighted) across every "
        f"headline arm and the baseline; found {values}")

    # the footnote must actually be the OTHER policy, or T4 measures nothing
    t4 = [c for t in sorted(run_all.WEIGHTED_TIERS)
          for c in run_all.TIERS[t] if c.trainer in reads_the_key]
    assert t4, "T4 has no cell that reads the class-weighting key"
    t4_values = {c.config: policy(c.config) for c in t4}
    assert set(t4_values.values()) == {True}, (
        f"T4 is the footnote arm and must run the OPPOSITE policy to the "
        f"headline, otherwise it duplicates T3; found {t4_values}")


def test_the_t4_footnote_differs_from_its_headline_cell_in_exactly_one_key():
    """A footnote that changed two things would not attribute the difference."""
    import yaml
    base = yaml.safe_load(
        (CODEBASE / "configs/resnet_cadsynth_ce.yaml").read_text(encoding="utf-8"))
    foot = yaml.safe_load(
        (CODEBASE / "configs/resnet_cadsynth_ce_weighted.yaml").read_text(encoding="utf-8"))

    # identity keys are expected to differ; nothing else may
    for k in ("experiment_name", "paths"):
        base.pop(k, None); foot.pop(k, None)

    diffs = []
    for k in set(base) | set(foot):
        if base.get(k) != foot.get(k):
            if k == "loss":
                b, f = dict(base.get(k, {})), dict(foot.get(k, {}))
                inner = [ik for ik in set(b) | set(f) if b.get(ik) != f.get(ik)]
                assert inner == ["use_class_weights"], (
                    f"loss blocks differ in more than the weighting: {inner}")
            else:
                diffs.append(k)
    assert not diffs, (
        f"the T4 footnote differs from resnet_cadsynth_ce.yaml in {diffs}; it "
        f"may differ ONLY in loss.use_class_weights, or the footnote conflates "
        f"the weighting with whatever else changed")


def test_the_unified_trainer_has_no_hidden_class_weighting():
    """Its arms are unweighted by construction, so they match decision 2."""
    src = (CODEBASE / "src" / "training" / "train_unified.py").read_text(
        encoding="utf-8")
    assert "use_class_weights" not in src
    assert "compute_class_weights" not in src


# --------------------------------------------------------------------- #
#  6. the CLI contract the regiment depends on                           #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("module", [
    "src.training.train_semantic",
    "src.training.train_resnet",
    "src.training.train_link",
    "src.training.train_unified",
])
def test_every_trainer_accepts_a_seed_flag(module):
    """run_all.py drives the 5-seed sweep through --seed; all four must take it."""
    proc = subprocess.run([sys.executable, "-m", module, "--help"],
                          cwd=CODEBASE, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "--seed" in proc.stdout, f"{module} has no --seed flag"


# --------------------------------------------------------------------- #
#  7. every runner must at least import                                  #
# --------------------------------------------------------------------- #

def test_every_runner_imports():
    """[ADDED 21 Sep 2026] `check_tier.py` was left with a literal backslash-n in
    a line-continuation and pytest reported 176 passed, because nothing in the
    suite imports it -- the gates are invoked as subprocesses by the chain, not
    by tests. A file that cannot be parsed is a broken gate, and a broken gate
    is how an unsound tier gets through.
    """
    import ast
    runners = sorted((CODEBASE / "runners").glob("*.py"))
    assert runners, "no runners found -- this test is not testing anything"
    broken = []
    for f in runners:
        try:
            ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as e:
            broken.append(f"{f.name}:{e.lineno}: {e.msg}")
    assert not broken, "runners that do not parse:\n  " + "\n  ".join(broken)


def test_the_weighted_tier_set_is_the_single_source_of_truth():
    """T6 would have failed the policy gate for doing what it was designed to do.

    [ADDED 21 Sep 2026] T6 is four deliberately weighted CADSynth arms and T7
    six more, so the first gate run on either
    would have reported a policy violation. The set now lives beside the tier
    table; this pins that nobody reintroduces a literal.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_all", CODEBASE / "runners" / "run_all.py")
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)

    assert run_all.WEIGHTED_TIERS == frozenset({"T4", "T6", "T7"})
    assert run_all.SCOUTING_TIERS == frozenset({"T7"})
    assert set(run_all.TIERS) == {f"T{i}" for i in range(1, 8)}, (
        "the tier table must hold T1 through T7")
    # every weighted tier must actually exist in the table
    for t in run_all.WEIGHTED_TIERS:
        assert t in run_all.TIERS, f"{t} is marked weighted but is not a tier"

    # Check the CODE, not the comments.
    gate = (CODEBASE / "runners" / "check_tier.py").read_text(encoding="utf-8")
    code_lines = [ln for ln in gate.splitlines()
                  if not ln.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    assert 'tier != "T4"' not in code, (
        "check_tier.py has gone back to hardcoding T4 as the only weighted tier")
    assert "WEIGHTED_TIERS" in code, (
        "check_tier.py no longer consults the authoritative weighted-tier set")

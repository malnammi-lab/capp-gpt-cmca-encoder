"""The arm config hash, and the tightened code-provenance rule.

Two things are pinned here.

1. **`arm_config_hash`** -- the P1 work order's "second, seed-and-paths-free
   config hash so arms are comparable across seeds". `config_hash` is the RUN
   identity: `runners/run_all.write_resolved_config` pins a different ``seed``
   and a different ``paths`` block into each of an arm's five runs, so the five
   carry five different values. Nothing could therefore assert "these five runs
   are the same experiment with a different seed" -- which is the claim the
   paper's error bars rest on -- and nothing could catch a seed that
   accidentally ran a different config.

   The danger in a hash like this is over-exclusion: drop one key too many and
   it stops distinguishing arms while still looking authoritative. So the tests
   below go in both directions -- stable across the exclusions, and moved by
   every setting that can change a number.

2. **`check_regiment_fields` requires `code_fingerprint`** (tightened 22 Sep
   2026). There is no ``.git``
   under ``codebase/``, so accepting a commit in its place had no reachable use,
   and that let a run with no code provenance at all be called quotable.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

CODEBASE = Path(__file__).resolve().parents[1]
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))

from runners import run_all                                # noqa: E402
from src.utils.provenance import (ARM_HASH_EXCLUDED_KEYS,  # noqa: E402
                                  arm_config_hash, build_provenance,
                                  check_regiment_fields, config_hash)

#: The arm the proof is built on: a real T2 cell with a real config on disk.
BASE_CONFIG = CODEBASE / "configs" / "mfinstseg_bias_both.yaml"
ARM = "mfinstseg_bias_both"
SEEDS = (0, 1, 2, 3, 4)


@pytest.fixture(scope="module")
def base_cfg() -> dict:
    return yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def resolved(tmp_path_factory) -> dict:
    """The five resolved configs of one arm, built by the REAL runner.

    `runners.run_all.write_resolved_config` is called rather than reimplemented:
    a copy of its two mutations here would keep passing on the day the runner
    grows a third, which is precisely the day this hash would start differing
    across seeds without anyone noticing.

    It scopes its paths with ``relative_to(run_all.REPO)``, so the run dirs have
    to sit under that root. ``run_all.REPO`` is repointed at a tmp tree holding
    a copy of the real config rather than writing five run dirs into
    ``codebase/results/``, and put back afterwards.
    """
    cell = next(c for c in run_all.resolve_tier("T2") if c.name == ARM)
    root = tmp_path_factory.mktemp("resolved")
    (root / "configs").mkdir()
    (root / cell.config).write_bytes((CODEBASE / cell.config).read_bytes())

    real_repo = run_all.REPO
    run_all.REPO = root
    try:
        out = {}
        for s in SEEDS:
            path = run_all.write_resolved_config(
                cell, s, root / "results" / ARM / f"seed{s}")
            out[s] = yaml.safe_load(path.read_text(encoding="utf-8"))
    finally:
        run_all.REPO = real_repo
    return out


# --------------------------------------------------------------------- #
#  1. the defect: five seeds, five run hashes, and (now) one arm hash    #
# --------------------------------------------------------------------- #

def test_the_five_seeds_of_one_arm_have_five_different_config_hashes(resolved):
    """The precondition, pinned. This is why the second hash had to exist."""
    hashes = {s: config_hash(c) for s, c in resolved.items()}
    assert len(set(hashes.values())) == 5, (
        f"config_hash stopped tracking the seed and the paths, so it is no "
        f"longer the RUN identity: {hashes}")


def test_the_five_seeds_of_one_arm_share_one_arm_config_hash(resolved):
    """The claim the error bars rest on, made checkable."""
    hashes = {s: arm_config_hash(c) for s, c in resolved.items()}
    assert len(set(hashes.values())) == 1, (
        f"the five seeds of {ARM} do not agree on arm_config_hash, so nothing "
        f"can assert they are one experiment: {hashes}")


def test_the_two_hashes_are_not_the_same_digest(resolved):
    """If they agreed, the new one would be excluding nothing."""
    cfg = resolved[0]
    assert arm_config_hash(cfg) != config_hash(cfg)


def test_build_provenance_records_both_hashes(resolved):
    """Emitted on every run, or the gates have nothing to compare."""
    provs = {s: build_provenance(c, seed=s, epoch=22, wall_time_s=9.0,
                                 selected_epoch=12,
                                 stopping_reason="early_stop")
             for s, c in resolved.items()}
    assert len({p["config_hash"] for p in provs.values()}) == 5
    assert len({p["arm_config_hash"] for p in provs.values()}) == 1
    assert len(provs[0]["arm_config_hash"]) == 16
    # and it must be the same value the function returns for that config
    assert provs[3]["arm_config_hash"] == arm_config_hash(resolved[3])


def test_arm_config_hash_survives_the_evaluator_into_the_metrics_file(resolved):
    """`provenance_from_checkpoint` copies the training block, so it travels."""
    from src.utils.provenance import provenance_from_checkpoint
    trained = build_provenance(resolved[2], seed=2, epoch=22, wall_time_s=9.0,
                               selected_epoch=12, stopping_reason="early_stop")
    out = provenance_from_checkpoint({"provenance": trained}, resolved[2])
    assert out["arm_config_hash"] == trained["arm_config_hash"], (
        "the arm hash was dropped or re-derived on the scoring machine")


# --------------------------------------------------------------------- #
#  2. the exclusion list is exactly what the runner rewrites             #
# --------------------------------------------------------------------- #

def test_the_exclusion_list_is_exactly_seed_and_paths():
    """Under-exclude and the hash is seed-dependent; over-exclude and it lies."""
    assert set(ARM_HASH_EXCLUDED_KEYS) == {"seed", "paths"}


def test_nothing_outside_the_exclusion_list_differs_between_two_seeds(resolved):
    """The same boundary, measured against the runner instead of asserted.

    The only defensible content for `ARM_HASH_EXCLUDED_KEYS` is "what
    `write_resolved_config` changes between two seeds of one arm". Two resolved
    configs are diffed at the top level and the differing keys must be exactly
    that set.
    """
    a, b = resolved[0], resolved[4]
    assert set(a) == set(b), "the runner added a key to one seed and not another"
    differing = {k for k in a if a[k] != b[k]}
    assert differing == set(ARM_HASH_EXCLUDED_KEYS), (
        f"seeds 0 and 4 of {ARM} differ in {sorted(differing)}, but "
        f"arm_config_hash only drops {sorted(ARM_HASH_EXCLUDED_KEYS)}. Either "
        f"the runner grew a per-seed key that this hash still digests (so the "
        f"five seeds will no longer agree), or one of the excluded keys stopped "
        f"being rewritten (so it is being dropped for no reason).")


@pytest.mark.parametrize("path,value", [
    (("bias", "type"), "adjacency"),
    (("loss", "use_class_weights"), True),
    (("loss", "type"), "cross_entropy"),
    (("training", "max_epochs"), 80),
    (("training", "patience"), 15),
    (("training", "lr"), 3.0e-4),
    (("training", "batch_size"), 8),
    (("dataset", "split_file"), "datasets/splits/mfinstseg_80_20_seed42.json"),
    (("dataset", "norm_stats"), "datasets/splits/somewhere_else.json"),
    (("dataset", "json_dir"), "../data/old_corpus/"),
    (("dataset", "attributes"), "raw"),
    (("model", "d_model"), 256),
    (("model", "version"), "v2"),
    (("experiment_name",), "a_different_arm"),
])
def test_arm_config_hash_moves_when_a_setting_that_can_move_a_number_moves(
        base_cfg, path, value):
    """The over-exclusion guard: every one of these must still be visible.

    Three of them are PATHS -- ``split_file``, ``norm_stats``, ``json_dir`` --
    and a hash described as "paths-free" is exactly the thing that would quietly
    stop seeing them. They are inputs: they name the split, the training-split
    normalisation statistics and the corpus. Two arms reading different ones are
    different experiments.
    """
    before = arm_config_hash(base_cfg)
    cfg = copy.deepcopy(base_cfg)
    node = cfg
    for k in path[:-1]:
        node = node[k]
    assert node[path[-1]] != value, "the fixture already holds the new value"
    node[path[-1]] = value
    assert arm_config_hash(cfg) != before, (
        f"changing {'.'.join(path)} left arm_config_hash at {before}: the hash "
        f"is blind to it, so an arm that ran with it changed would pass the "
        f"cross-seed check as the same experiment")


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda c: c.__setitem__("seed", 3), id="seed"),
    pytest.param(lambda c: c["paths"].__setitem__(
        "checkpoint", "results/x/seed3/checkpoint.pth"), id="paths-checkpoint"),
    pytest.param(lambda c: c["paths"].__setitem__(
        "figures", "results/x/seed3/"), id="paths-figures"),
    pytest.param(lambda c: c.__setitem__("paths", {}), id="paths-emptied"),
])
def test_arm_config_hash_is_blind_to_the_seed_and_the_output_paths(
        base_cfg, mutate):
    before = arm_config_hash(base_cfg)
    cfg = copy.deepcopy(base_cfg)
    mutate(cfg)
    assert arm_config_hash(cfg) == before


def test_semantic_checkpoint_is_kept_because_it_is_an_input_not_an_output():
    """`configs/mfinstseg_two_stage.yaml` -- the frozen Stage-1 weights.

    It is a top-level path, so "seed-and-paths-free" reads like an argument for
    dropping it. It is an INPUT: the link stage loads those weights, and two
    arms that load different ones are not the same experiment. `run_all` does
    not rewrite it per seed, so keeping it costs nothing.
    """
    cfg = yaml.safe_load((CODEBASE / "configs" / "mfinstseg_two_stage.yaml")
                         .read_text(encoding="utf-8"))
    assert "semantic_checkpoint" in cfg, (
        "the two-stage config no longer names its Stage-1 checkpoint; this "
        "test is pinning a key that has moved")
    before = arm_config_hash(cfg)
    other = copy.deepcopy(cfg)
    other["semantic_checkpoint"] = "results/checkpoints/some_other_stage1.pth"
    assert arm_config_hash(other) != before


def test_every_config_in_configs_has_a_distinct_arm_hash():
    """A collision would make two arms indistinguishable to the cross-seed check."""
    by_hash: dict = {}
    for f in sorted((CODEBASE / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text(encoding="utf-8"))
        by_hash.setdefault(arm_config_hash(cfg), []).append(f.name)
    collisions = {h: n for h, n in by_hash.items() if len(n) > 1}
    assert not collisions, f"two arms share one arm_config_hash: {collisions}"


# --------------------------------------------------------------------- #
#  3. degenerate input raises instead of returning a constant            #
# --------------------------------------------------------------------- #

def test_a_config_that_is_nothing_but_seed_and_paths_raises():
    """The digest of an empty dict is a constant: every arm would agree."""
    with pytest.raises(ValueError, match="nothing is left of this config"):
        arm_config_hash({"seed": 0, "paths": {"checkpoint": "a.pth"}})


def test_a_non_dict_config_raises():
    with pytest.raises(TypeError, match="expects the resolved config dict"):
        arm_config_hash(["not", "a", "config"])


def test_the_shared_canonicaliser_did_not_move_config_hash():
    """`config_hash` was refactored onto `_canonical`; its output must not move.

    Every metrics file on disk records a `config_hash`, and
    `src/evaluation/inference.py` compares a recorded one against a live one to
    decide whether a checkpoint and a config match. A refactor that shifted the
    digest would turn every one of those comparisons into a false mismatch.
    """
    assert config_hash({"a": 1, "b": 2}) == "43258cff783fe703"
    assert config_hash({"b": 2, "a": 1}) == config_hash({"a": 1, "b": 2})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


# --------------------------------------------------------------------- #
#  4. code provenance now requires the fingerprint                       #
# --------------------------------------------------------------------- #

def _complete(**over) -> dict:
    prov = {
        "seed": 0, "epoch": 42, "selected_epoch": 38,
        "stopping_reason": "early_stop", "config_hash": "c0ffeec0ffeec0ff",
        "gpu": "Tesla T4", "wall_time_s": 931.0,
        "code_fingerprint": "9d88e9213dbb9164", "git_commit": None,
    }
    prov.update(over)
    return prov


def test_a_run_with_only_a_git_commit_is_not_quotable():
    """[TIGHTENED 22 Sep 2026] The old rule accepted this and returned ``[]``.

    A run carrying a commit and no fingerprint has no code provenance in this
    project: ``codebase/`` is a synced folder with no ``.git``, so the commit
    can only have come from hand-editing or from tooling outside this
    repository. `check_tier` and `aggregate_seeds` each had to grow their own
    absence check because this one accepted the shape; a tier built entirely
    from it once exited 0 declaring itself SOUND having compared zero
    fingerprints.
    """
    prov = _complete(git_commit="abc1234")
    del prov["code_fingerprint"]

    assert check_regiment_fields(prov) == ["code_fingerprint"], (
        "a run with a commit and no fingerprint is being called quotable again")


def test_a_null_fingerprint_is_the_same_as_an_absent_one():
    assert check_regiment_fields(
        _complete(code_fingerprint=None)) == ["code_fingerprint"]


def test_a_fingerprint_alone_is_enough_and_a_commit_is_not_required():
    """The real shape of every run this project produces: no repo, no commit."""
    assert check_regiment_fields(_complete()) == []
    prov = _complete()
    del prov["git_commit"]
    assert check_regiment_fields(prov) == []


def test_git_commit_is_still_recorded_even_though_it_is_not_evidence(resolved):
    """Tightening the CHECK must not stop the field being written.

    The day task 0b.1 creates the repo, `git_commit()` starts returning a real
    hash and it lands in the block as before -- alongside the fingerprint, which
    `build_provenance` records unconditionally, which is why requiring it costs
    nothing then either.
    """
    prov = build_provenance(resolved[0], seed=0, epoch=22, wall_time_s=9.0,
                            selected_epoch=12, stopping_reason="early_stop")
    assert "git_commit" in prov
    assert prov["code_fingerprint"] and len(prov["code_fingerprint"]) == 16
    assert check_regiment_fields(prov) == []

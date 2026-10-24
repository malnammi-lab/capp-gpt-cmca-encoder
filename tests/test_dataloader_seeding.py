"""
Tests for DataLoader reproducibility -- P1.

The claim the batch run makes
----------------------------
T1B is 85 runs across 5 seeds, and every error bar in the paper is the spread
of those seeds.  `check_tier.py` already fails a cell whose five seeds produce
*identical* accuracy, on the grounds that the error bars would be fiction.
The mirror-image failure has had no guard at all: seeds that differ for the
wrong reason, or a run that cannot reproduce its own published number.

Before this file, DataLoader shuffling in all four trainers was seeded only by
the GLOBAL torch RNG.  Two consequences, both silent:

  1. The epoch order depended on how much other code had drawn from the global
     RNG before the loader was built.  Adding a layer to the model, or building
     a val loader first, reshuffled training.  `test_a_loader_without_a_generator
     _is_at_the_mercy_of_the_global_rng` demonstrates that on the real class, so
     the rest of this file is known to have teeth.
  2. With `num_workers > 0` each worker re-seeds from an unpredictable base and
     numpy inside the dataset is duplicated across workers.  Every loader in
     this repo is `num_workers=0` today, so this one is dormant -- which is
     exactly why it would be introduced by a one-word speed change and noticed
     by nobody.

What is asserted here is the strong form: same seed -> byte-identical batch
order, different seed -> different batch order, and the real trainer passes a
generator whose seed is a function of that run's seed.  "generator is not None"
would pass against a generator seeded with a constant, which reproduces
beautifully and makes all five seeds identical.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.utils import seeding
from src.utils.seeding import (derive_seed, loader_kwargs, seed_worker,
                               set_global_seed)

CODEBASE = Path(__file__).resolve().parents[1]
TRAINERS = sorted((CODEBASE / "src" / "training").glob("train_*.py"))

# Names of every Dataset class the trainers construct.  Used by the
# seed-order test below.
DATASET_CLASSES = {"PreExtractedBRepDataset", "UVGridLabelDataset"}


# --------------------------------------------------------------------- #
#  helpers                                                              #
# --------------------------------------------------------------------- #

def _batch_order(seed: int, role: str = "train_loader", *,
                 burn_global: int = 0) -> list[int]:
    """Full epoch order of a shuffling loader built through `loader_kwargs`.

    *burn_global* draws that many values from the global torch RNG first.  A
    correctly generator-seeded loader must be indifferent to it.
    """
    for _ in range(burn_global):
        torch.rand(1)
    loader = DataLoader(list(range(64)), batch_size=4, shuffle=True,
                        **loader_kwargs(seed, role))
    return [int(x) for batch in loader for x in batch]


def _unseeded_batch_order(global_seed: int, burn_global: int) -> list[int]:
    """Epoch order of a loader built the OLD way -- no ``generator=``."""
    torch.manual_seed(global_seed)
    for _ in range(burn_global):
        torch.rand(1)
    loader = DataLoader(list(range(64)), batch_size=4, shuffle=True)
    return [int(x) for batch in loader for x in batch]


# --------------------------------------------------------------------- #
#  1. the generator actually drives the batch order                     #
# --------------------------------------------------------------------- #

def test_the_same_seed_reproduces_the_batch_order_exactly():
    assert _batch_order(0) == _batch_order(0)
    assert _batch_order(4) == _batch_order(4)


def test_the_five_regiment_seeds_produce_five_different_batch_orders():
    """A constant-seeded generator would reproduce perfectly and still be wrong.

    If all five orders collapsed to one, the five runs of an arm would differ
    only by weight init and the error bars would understate the real spread.
    """
    orders = [tuple(_batch_order(s)) for s in range(5)]
    assert len(set(orders)) == 5, (
        "seeds 0-4 did not give 5 distinct batch orders; got "
        f"{len(set(orders))} distinct")


def test_the_shuffle_is_not_the_identity():
    """Guards against a 'reproducible' order that is simply unshuffled."""
    assert _batch_order(0) != list(range(64))


def test_two_loaders_in_one_run_do_not_share_a_permutation():
    """Per-role sub-seeds: the val loader must not walk the train order."""
    assert _batch_order(0, "train_loader") != _batch_order(0, "val_loader")


# --------------------------------------------------------------------- #
#  2. the bug this replaces -- proof the tests above have teeth          #
# --------------------------------------------------------------------- #

def test_a_loader_without_a_generator_is_at_the_mercy_of_the_global_rng():
    """This is what the four trainers did until the P1 audit fix.

    Same global seed, same dataset, same batch size -- and a different epoch
    order, purely because something else drew from the global RNG in between.
    In a trainer that "something else" is the model's own weight init, so
    changing `n_layers` silently changed the training order too.
    """
    assert _unseeded_batch_order(0, 0) != _unseeded_batch_order(0, 137)


def test_a_generator_seeded_loader_ignores_the_global_rng_state():
    """The fix, stated as the negation of the test above."""
    assert _batch_order(0, burn_global=0) == _batch_order(0, burn_global=137)


def test_building_another_loader_first_does_not_change_the_train_order():
    """Why each loader gets its own generator instead of one shared object.

    A single shared ``torch.Generator`` advances as each loader draws from it,
    so the train order would depend on how many loaders were constructed
    before it. Adding the validation loader would then have changed the
    training order of every arm without anyone touching the training code.
    """
    baseline = _batch_order(0, "train_loader")

    val = DataLoader(list(range(64)), batch_size=4, shuffle=True,
                     **loader_kwargs(0, "val_loader"))
    _ = [int(x) for b in val for x in b]        # consume it
    assert _batch_order(0, "train_loader") == baseline


# --------------------------------------------------------------------- #
#  3. derive_seed: stable, distinct, and loud when unseeded             #
# --------------------------------------------------------------------- #

def test_derive_seed_is_stable_across_processes():
    """`hash()` is salted per process; SHA256 is not.

    If this regressed to `hash()`, a run could not reproduce its own batch
    order on a second invocation, let alone on the studio box.
    """
    prog = ("import sys; sys.path.insert(0, %r);"
            "from src.utils.seeding import derive_seed;"
            "print(derive_seed(3, 'train_loader'))" % str(CODEBASE))
    outs = set()
    for salt in ("0", "1", "12345"):
        env_cmd = [sys.executable, "-c", prog]
        res = subprocess.run(env_cmd, capture_output=True, text=True,
                             cwd=str(CODEBASE),
                             env={**_base_env(), "PYTHONHASHSEED": salt})
        assert res.returncode == 0, res.stderr
        outs.add(res.stdout.strip())
    assert len(outs) == 1, f"derive_seed differs between processes: {outs}"


def _base_env() -> dict:
    import os
    return {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}


def test_derive_seed_separates_seeds_and_roles():
    seeds = {derive_seed(s, "train_loader") for s in range(5)}
    assert len(seeds) == 5
    assert derive_seed(0, "train_loader") != derive_seed(0, "val_loader")


def test_derive_seed_stays_in_manual_seed_range():
    for s in range(5):
        for role in ("train_loader", "val_loader", "test_loader"):
            v = derive_seed(s, role)
            assert isinstance(v, int) and 0 <= v < 2 ** 31


@pytest.mark.parametrize("bad_seed", [None, "3", 4.2, True])
def test_an_unseeded_loader_raises_instead_of_shuffling_anyway(bad_seed):
    with pytest.raises(TypeError):
        loader_kwargs(bad_seed, "train_loader")


@pytest.mark.parametrize("bad_role", ["", 7, None])
def test_a_loader_with_no_role_raises(bad_role):
    with pytest.raises((TypeError, ValueError)):
        loader_kwargs(0, bad_role)


# --------------------------------------------------------------------- #
#  4. worker_init_fn                                                     #
# --------------------------------------------------------------------- #

def test_loader_kwargs_carries_the_repo_worker_init_fn():
    kw = loader_kwargs(0, "train_loader")
    assert kw["worker_init_fn"] is seed_worker
    assert isinstance(kw["generator"], torch.Generator)
    assert kw["generator"].initial_seed() == derive_seed(0, "train_loader")


def test_seed_worker_reseeds_numpy_from_the_torch_base_seed():
    """What makes num_workers>0 safe rather than silently unreproducible."""
    def draw(torch_seed: int) -> float:
        torch.manual_seed(torch_seed)
        seed_worker(0)
        return float(np.random.rand())

    assert draw(1234) == draw(1234)
    assert draw(1234) != draw(5678)


# --------------------------------------------------------------------- #
#  5. structural: no trainer may add an unseeded loader later           #
# --------------------------------------------------------------------- #

def _dataloader_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "DataLoader"]


def _loader_kwargs_role(call: ast.Call) -> str | None:
    """The role string of a ``**loader_kwargs(seed, "role")`` splat, if any."""
    for kw in call.keywords:
        if kw.arg is not None:
            continue
        v = kw.value
        if (isinstance(v, ast.Call)
                and getattr(v.func, "id", None) == "loader_kwargs"
                and len(v.args) == 2
                and isinstance(v.args[1], ast.Constant)):
            return v.args[1].value
    return None


def test_the_trainers_between_them_build_ten_dataloaders():
    """Pins the count, so a new loader cannot slip in unnoticed."""
    counts = {p.name: len(_dataloader_calls(p)) for p in TRAINERS}
    assert counts == {
        "train_link.py": 2,
        "train_resnet.py": 3,
        "train_semantic.py": 3,
        "train_unified.py": 2,
    }, counts


@pytest.mark.parametrize("trainer", TRAINERS, ids=lambda p: p.name)
def test_every_dataloader_in_every_trainer_is_seeded(trainer):
    calls = _dataloader_calls(trainer)
    assert calls, f"no DataLoader found in {trainer.name} -- has it moved?"
    unseeded = [c.lineno for c in calls if _loader_kwargs_role(c) is None]
    assert not unseeded, (
        f"{trainer.name}: DataLoader at line(s) {unseeded} has no "
        f"`**loader_kwargs(seed, \"<role>\")`. Its shuffle order would fall "
        f"back to the global torch RNG and its workers would get no base seed.")


@pytest.mark.parametrize("trainer", TRAINERS, ids=lambda p: p.name)
def test_loader_roles_are_unique_within_a_trainer(trainer):
    """Two loaders sharing a role share a permutation -- see test 1."""
    roles = [_loader_kwargs_role(c) for c in _dataloader_calls(trainer)]
    assert len(roles) == len(set(roles)), f"{trainer.name}: duplicate roles {roles}"


# --------------------------------------------------------------------- #
#  6. the related P1 item: seed BEFORE dataset construction             #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("trainer", TRAINERS, ids=lambda p: p.name)
def test_the_seed_is_set_before_any_dataset_is_constructed(trainer):
    """Advisor P1: "train_unified.py: seed BEFORE dataset construction".

    `bias_type: shuffled` permutes each model's adjacency at dataset load
    time.  If the dataset is built before `set_global_seed` runs, the control
    arm's permutation is not a function of the seed the run then records, and
    the arm cannot be reproduced from its own provenance block.
    """
    tree = ast.parse(trainer.read_text(encoding="utf-8"))
    seed_lines = [n.lineno for n in ast.walk(tree)
                  if isinstance(n, ast.Call)
                  and getattr(n.func, "id", None) == "set_global_seed"]
    ds_lines = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) in DATASET_CLASSES]
    assert seed_lines, f"{trainer.name} never calls set_global_seed"
    assert ds_lines, f"{trainer.name} constructs no dataset -- has it moved?"
    assert max(seed_lines) < min(ds_lines), (
        f"{trainer.name}: set_global_seed at line {max(seed_lines)} runs AFTER "
        f"the first dataset construction at line {min(ds_lines)}.")


# --------------------------------------------------------------------- #
#  7. end to end: the real trainer, on the real synthetic corpus         #
# --------------------------------------------------------------------- #

class _RecordingDataLoader(DataLoader):
    """A DataLoader that records what the trainer asked for."""

    calls: list = []

    def __init__(self, *args, **kwargs):
        gen = kwargs.get("generator")
        _RecordingDataLoader.calls.append({
            "shuffle": kwargs.get("shuffle"),
            "generator_seed": None if gen is None else gen.initial_seed(),
            "worker_init_fn": kwargs.get("worker_init_fn"),
        })
        super().__init__(*args, **kwargs)


def _semantic_config(corpus, out_dir, *, seed_in_config: int) -> dict:
    """Minimal real train_semantic config over the session corpus fixture."""
    return {
        "experiment_name": "dataloader-seeding",
        "seed": seed_in_config,
        "dataset": {
            "json_dir": str(corpus["json_dir"]),
            "split_file": str(corpus["split_file"]),
            "norm_stats": str(corpus["norm_stats"]),
            "n_classes": corpus["n_classes"],
            "attributes": "standardized",
        },
        "bias": {"type": "both", "lambda_adj": 1.0, "lambda_sim": 1.0},
        "model": {"type": "semantic_segmentation", "version": "v1",
                  "d_model": 32, "n_heads": 2, "n_layers": 1,
                  "dim_feedforward": 32, "dropout": 0.1},
        "label_encoder": {"embedding_dim": 32},
        "loss": {"type": "prototype_infonce", "temperature": 0.07,
                 "use_class_weights": False},
        "training": {"max_epochs": 1, "patience": 1, "batch_size": 2,
                     "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0},
        "paths": {"checkpoint": str(out_dir / "ckpt.pth"),
                  "figures": str(out_dir / "figs") + "/"},
        "device": "cpu",
    }


def _run_semantic(corpus, out_dir, monkeypatch, seed: int) -> list[dict]:
    """Run the real trainer once and return what its DataLoaders were given."""
    import src.training.train_semantic as ts

    monkeypatch.setattr(ts, "DataLoader", _RecordingDataLoader)
    _RecordingDataLoader.calls = []
    ts.train_semantic_model(_semantic_config(corpus, out_dir,
                                             seed_in_config=999),
                            seed_override=seed)
    return list(_RecordingDataLoader.calls)


def test_the_real_trainer_seeds_all_three_loaders_from_the_run_seed(
        corpus, tmp_path, monkeypatch):
    """End to end: no stubs below the trainer's own DataLoader calls.

    `seed: 999` in the config must lose to `seed_override=2`, so the generator
    seeds prove the value that actually reached the loaders came from the run
    seed the checkpoint records.
    """
    seen = _run_semantic(corpus, tmp_path, monkeypatch, seed=2)

    assert len(seen) == 3, f"expected train/val/test loaders, got {len(seen)}"
    assert [c["shuffle"] for c in seen] == [True, False, False]

    for call in seen:
        assert call["worker_init_fn"] is seed_worker
        assert call["generator_seed"] is not None

    expected = {derive_seed(2, r)
                for r in ("train_loader", "val_loader", "test_loader")}
    assert {c["generator_seed"] for c in seen} == expected
    # not the config's 999, and not the raw run seed either
    assert 999 not in {c["generator_seed"] for c in seen}
    assert 2 not in {c["generator_seed"] for c in seen}


def test_two_run_seeds_give_the_real_trainer_two_sets_of_loader_seeds(
        corpus, tmp_path, monkeypatch):
    """The thing the error bars depend on: seed 0 and seed 3 really differ."""
    a = _run_semantic(corpus, tmp_path / "a", monkeypatch, seed=0)
    b = _run_semantic(corpus, tmp_path / "b", monkeypatch, seed=3)
    sa = [c["generator_seed"] for c in a]
    sb = [c["generator_seed"] for c in b]
    assert len(sa) == len(sb) == 3
    assert not set(sa) & set(sb), f"seed 0 and seed 3 shared loader seeds: {sa} {sb}"


def test_the_trainers_still_pass_num_workers_zero_everywhere():
    """Documents the dormant half of the fix.

    `worker_init_fn` is never called while num_workers is 0.  If this test
    starts failing, someone has raised it -- which is fine now, and would not
    have been before.
    """
    hits = []
    for p in TRAINERS:
        for call in _dataloader_calls(p):
            for kw in call.keywords:
                if kw.arg == "num_workers":
                    hits.append((p.name, call.lineno))
    assert not hits, (
        f"num_workers is now set at {hits}; that is safe because every loader "
        f"passes worker_init_fn, but update this test to say so.")

"""
DataLoader determinism in the four EVALUATION scripts -- P1.

What this file decides, and why it is not the same answer as the trainers
------------------------------------------------------------------------
The P1 work order says "pass ``generator=``/``worker_init_fn=`` from seeding.py
into EVERY DataLoader".  `tests/test_dataloader_seeding.py` did that for the ten
loaders in the four TRAINERS, where it was load-bearing: those loaders shuffle,
and a shuffling loader with no ``generator=`` draws its permutation from the
global torch RNG, so the epoch order moved whenever anything else consumed from
that RNG first.

The nine loaders in the four evaluation scripts are a different case, and this
file is the argument that they are -- made per loader, and then GATED so it
cannot quietly stop being true.

    src/evaluation/evaluate_semantic.py
      1. test_loader                      shuffle=False
    src/evaluation/evaluate_unified.py
      2. loader                           shuffle=False
    src/evaluation/evaluate_link.py
      3. train_loader (threshold tuning)  shuffle=False
      4. test_loader  (split-file branch) shuffle=False
      5. test_loader  (random_split path) shuffle=False
    src/evaluation/inference.py
      6. semantic mode                    shuffle=False
      7. resnet mode  (UVGridLabelDataset) shuffle=False
      8. link mode                        shuffle=False
      9. unified mode                     shuffle=False

All nine are ``shuffle=False`` and all nine are ``num_workers=0``.  Three facts
follow, each MEASURED below rather than asserted:

1. ``shuffle=False`` builds a ``SequentialSampler``, which never consults
   ``generator=``.  `test_a_sequential_loader_ignores_the_generator_entirely`
   shows generator(1), generator(99999) and no generator producing one identical
   order.  Passing a generator here would change nothing at all.

2. ``num_workers=0`` means PyTorch never calls ``worker_init_fn``.  And even if
   it did, both evaluation dataset classes preload everything in ``__init__``
   and their ``__getitem__`` is a pure dict/tuple lookup -- no numpy, no
   ``random``, nothing to reseed.  `test_the_eval_datasets_getitem_is_pure`
   measures that.  ``bias_type="shuffled"``, the only randomness anywhere near
   this path, draws its permutation from a generator keyed on the MODEL ID
   (`src/data/brep_dataset.py:_shuffle_adjacency`), not from any global state.

3. The metrics these loaders feed are sums over all faces -- accuracy, a
   confusion matrix, and a threshold swept along an ROC curve.  None depends on
   the order the faces arrived in.
   `test_the_metrics_do_not_depend_on_the_order_faces_arrive_in` permutes the
   inputs and shows the numbers are bit-identical.

So `**loader_kwargs(seed, role)` on these nine would be inert ceremony -- and
worse than inert.  ``loader_kwargs`` RAISES on ``seed=None``, which is the
default for `evaluate_semantic`, `evaluate_unified` and `evaluate_link`, and
`inference.py` has no ``--seed`` option at all.  Adding it would make four
working evaluators demand a seed they otherwise have no use for, and inventing
one would put a fake seed-dependence into a result that must NOT depend on the
seed: evaluating one frozen checkpoint on one frozen id list must give the same
number every time, whatever seed the caller happened to type.

What is actually at risk, and what this file gates
--------------------------------------------------
The hazard is not today's code, it is tomorrow's one-word change.  Flip any of
these nine to ``shuffle=True``, or raise ``num_workers`` for speed over the
5,880-model MFInstSeg corpus, and the reasoning above collapses -- silently,
because the evaluator still runs and still writes a metrics file.

`test_every_evaluation_dataloader_is_order_free_or_seeded` is therefore a
structural gate: every DataLoader in these four files must be ``shuffle=False``
AND ``num_workers``-free, or else carry ``**loader_kwargs(seed, "<role>")``.
`test_the_gate_rejects_a_shuffling_unseeded_loader` proves the gate has teeth by
running it against a module that breaks the rule, so this file is known not to
be asserting a tautology.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.brep_dataset import PreExtractedBRepDataset
from src.data.collate import collate_fn_brep
from src.data.uv_dataset import UVGridLabelDataset
from src.evaluation import inference as INF
from src.utils.metrics import (AdaptiveThresholdFinder, compute_accuracy,
                               compute_confusion_matrix)

CODEBASE = Path(__file__).resolve().parents[1]
EVAL_DIR = CODEBASE / "src" / "evaluation"

EVALUATORS = [
    EVAL_DIR / "evaluate_semantic.py",
    EVAL_DIR / "evaluate_unified.py",
    EVAL_DIR / "evaluate_link.py",
    EVAL_DIR / "inference.py",
]

# The count each file is expected to build.  Pinned so a tenth loader cannot
# slip in without someone reading the argument in this docstring first.
EXPECTED_LOADER_COUNTS = {
    "evaluate_semantic.py": 1,
    "evaluate_unified.py": 1,
    "evaluate_link.py": 3,
    "inference.py": 4,
}


# --------------------------------------------------------------------- #
#  AST helpers -- shared by the gate and by the gate's own teeth test    #
# --------------------------------------------------------------------- #

def _dataloader_calls(source: str) -> list[ast.Call]:
    tree = ast.parse(source)
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "DataLoader"]


def _kwarg(call: ast.Call, name: str):
    """The literal value of keyword *name*, or None if absent/non-literal."""
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


def _has_loader_kwargs(call: ast.Call) -> bool:
    """True if the call splats ``**loader_kwargs(seed, "role")``."""
    for kw in call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Call) \
                and getattr(kw.value.func, "id", None) == "loader_kwargs":
            return True
    return False


def _offending_loaders(source: str, name: str) -> list[str]:
    """Every DataLoader in *source* that is neither order-free nor seeded.

    This is the gate itself, factored out so the teeth test below can run the
    exact same function against a deliberately-broken module.
    """
    bad = []
    for call in _dataloader_calls(source):
        if _has_loader_kwargs(call):
            continue                      # seeded: fine whatever it does
        shuffle = _kwarg(call, "shuffle")
        workers = _kwarg(call, "num_workers")
        if shuffle is not False:
            bad.append(f"{name}:{call.lineno} shuffle={shuffle!r} "
                       f"with no **loader_kwargs(...)")
        elif workers not in (None, 0):
            bad.append(f"{name}:{call.lineno} num_workers={workers!r} "
                       f"with no **loader_kwargs(...)")
    return bad


# --------------------------------------------------------------------- #
#  1. the measured facts the "no ceremony" decision rests on             #
# --------------------------------------------------------------------- #

def test_a_sequential_loader_ignores_the_generator_entirely():
    """shuffle=False -> SequentialSampler, which never reads `generator=`.

    This is the whole reason `**loader_kwargs(...)` is absent from the nine
    evaluation loaders: it could not change their order if it were there.
    """
    ds = list(range(20))

    def order(gen):
        kw = {} if gen is None else {"generator": torch.Generator().manual_seed(gen)}
        return [int(x) for b in DataLoader(ds, batch_size=3, shuffle=False, **kw)
                for x in b]

    assert order(1) == order(99999) == order(None) == ds


def test_a_sequential_loader_is_indifferent_to_the_global_rng():
    """The failure mode that bit the trainers cannot reach these loaders."""
    ds = list(range(20))

    def order(burn):
        torch.manual_seed(0)
        for _ in range(burn):
            torch.rand(1)
        return [int(x) for b in DataLoader(ds, batch_size=3, shuffle=False)
                for x in b]

    assert order(0) == order(137) == ds


def test_a_shuffling_loader_without_a_generator_is_not_indifferent():
    """Proof the gate below guards a REAL hazard, not a hypothetical one.

    Flip one of the nine to shuffle=True and this is what it inherits.
    """
    ds = list(range(64))

    def order(burn):
        torch.manual_seed(0)
        for _ in range(burn):
            torch.rand(1)
        return [int(x) for b in DataLoader(ds, batch_size=4, shuffle=True)
                for x in b]

    assert order(0) != order(137)


def test_the_metrics_do_not_depend_on_the_order_faces_arrive_in():
    """Accuracy, the confusion matrix and the swept threshold are all sums."""
    rng = np.random.default_rng(0)
    preds = rng.integers(0, 5, 500)
    gts = rng.integers(0, 5, 500)
    sims = rng.random(500).astype(np.float32)
    tgts = (rng.random(500) < 0.3).astype(np.int64)

    perm = rng.permutation(500)

    assert compute_accuracy(preds, gts) == compute_accuracy(preds[perm], gts[perm])
    assert np.array_equal(compute_confusion_matrix(preds, gts, 5),
                          compute_confusion_matrix(preds[perm], gts[perm], 5))
    assert (AdaptiveThresholdFinder().find_optimal_threshold(sims, tgts)
            == AdaptiveThresholdFinder().find_optimal_threshold(sims[perm],
                                                                tgts[perm]))


# --------------------------------------------------------------------- #
#  2. the real dataset classes: nothing for worker_init_fn to reseed     #
# --------------------------------------------------------------------- #

def _eval_cfg(corpus) -> dict:
    """A config in the shape every shipped graph config has."""
    return {
        "experiment_name": "eval_loader_seeding",
        "dataset": {
            "json_dir": str(corpus["json_dir"]),
            "split_file": str(corpus["split_file"]),
            "n_classes": corpus["n_classes"],
            "attributes": "standardized",
            "norm_stats": str(corpus["norm_stats"]),
            "one_hot_type": False,
        },
        "bias": {"type": "similarity", "lambda_adj": 1.0, "lambda_sim": 1.0},
        "device": "cpu",
    }


def test_the_eval_datasets_getitem_is_pure(corpus):
    """`__getitem__` is a lookup, so a worker has no RNG state to inherit.

    If this ever fails, `worker_init_fn` stops being inert and the gate's
    `num_workers` clause becomes load-bearing rather than precautionary.
    """
    cfg = _eval_cfg(corpus)
    ds = PreExtractedBRepDataset(
        verbose=False, **INF._resolve_data_view(cfg, where="t"))

    torch.manual_seed(0)
    np.random.seed(0)
    first = ds[0]
    torch.manual_seed(12345)
    np.random.seed(999)
    for _ in range(50):
        torch.rand(1)
        np.random.rand()
    second = ds[0]

    for key, val in first.items():
        if torch.is_tensor(val):
            assert torch.equal(val, second[key]), f"{key} moved between reads"
        else:
            assert val == second[key], f"{key} moved between reads"


def test_the_resnet_eval_dataset_getitem_is_pure(tmp_path):
    """Same claim for the loader in `inference.py`'s resnet mode.

    `UVGridLabelDataset` reads ``*_faces.json``, which the shared corpus
    fixture (``*_features.json``) does not contain, so this builds its own.
    """
    rng = np.random.default_rng(0)
    for mid in ("00000000", "00000001"):
        (tmp_path / f"{mid}_faces.json").write_text(json.dumps({
            "faces": [{"face_id": i,
                       "uv_grid": rng.random((7, 4, 4)).tolist(),
                       "label": int(rng.integers(0, 5))}
                      for i in range(3)]}))

    uv = UVGridLabelDataset(json_dir=str(tmp_path), n_classes=5, verbose=False)
    assert len(uv) == 6

    torch.manual_seed(7)
    np.random.seed(7)
    g0, l0 = uv[0]
    torch.manual_seed(8888)
    np.random.seed(8888)
    for _ in range(50):
        torch.rand(1)
        np.random.rand()
    g1, l1 = uv[0]
    assert torch.equal(g0, g1) and torch.equal(l0, l1)


def test_a_real_evaluation_loader_walks_the_split_file_order(corpus):
    """End to end on the real classes: eval order is fixed and seed-free.

    Built exactly as `evaluate_unified` builds it, drained twice from wildly
    different global RNG states, and compared against the split file's own test
    id list.  This is the positive form of the claim the gate protects.
    """
    cfg = _eval_cfg(corpus)
    test_ids = json.loads(Path(corpus["split_file"]).read_text())["test"]
    ds_kwargs = INF._resolve_data_view(cfg, where="t")
    ds_kwargs.pop("split_file", None)
    test_ds = PreExtractedBRepDataset(
        model_ids=test_ids, verbose=False, split_file=cfg["dataset"]["split_file"],
        **ds_kwargs)

    def ids(burn_global):
        torch.manual_seed(burn_global)
        for _ in range(burn_global):
            torch.rand(1)
        loader = DataLoader(test_ds, batch_size=2, shuffle=False,
                            collate_fn=collate_fn_brep)
        return [m for batch in loader for m in batch["model_ids"]]

    a, b = ids(0), ids(137)
    assert a == b, f"evaluation order moved with the global RNG: {a} vs {b}"
    assert a == list(test_ids), (
        f"loader order is not the split file's test order: {a} vs {test_ids}")


# --------------------------------------------------------------------- #
#  3. the structural gate                                                #
# --------------------------------------------------------------------- #

def test_the_evaluators_between_them_build_nine_dataloaders():
    """Pins the count, so a tenth loader cannot slip in unnoticed."""
    counts = {p.name: len(_dataloader_calls(p.read_text(encoding="utf-8")))
              for p in EVALUATORS}
    assert counts == EXPECTED_LOADER_COUNTS, counts
    assert sum(counts.values()) == 9


@pytest.mark.parametrize("path", EVALUATORS, ids=lambda p: p.name)
def test_every_evaluation_dataloader_is_order_free_or_seeded(path):
    """THE GATE.

    Order-free (``shuffle=False``, no workers) is fine and is what all nine are
    today -- see this module's docstring for why that is a decision and not an
    oversight.  Anything else must pass ``**loader_kwargs(seed, "<role>")``.
    """
    calls = _dataloader_calls(path.read_text(encoding="utf-8"))
    assert calls, f"no DataLoader found in {path.name} -- has it moved?"
    bad = _offending_loaders(path.read_text(encoding="utf-8"), path.name)
    assert not bad, (
        "these evaluation DataLoaders are neither order-free nor seeded:\n  "
        + "\n  ".join(bad)
        + "\n\nA shuffling evaluation loader with no `generator=` takes its "
          "permutation from the global torch RNG, so the order moves whenever "
          "anything else draws from it first; with num_workers>0 and no "
          "`worker_init_fn` each worker inherits the parent's numpy state. "
          "Either revert to shuffle=False/num_workers=0, or add "
          "`**loader_kwargs(seed, \"<role>\")` from src/utils/seeding.py AND "
          "thread a real seed into this script -- note that `loader_kwargs` "
          "raises on seed=None and inference.py currently has no --seed.")


@pytest.mark.parametrize("path", EVALUATORS, ids=lambda p: p.name)
def test_every_evaluation_dataloader_is_explicit_about_shuffling(path):
    """`shuffle` defaults to False, but relying on the default hides intent."""
    missing = [c.lineno for c in _dataloader_calls(path.read_text(encoding="utf-8"))
               if _kwarg(c, "shuffle") is None and not _has_loader_kwargs(c)]
    assert not missing, (
        f"{path.name}: DataLoader at line(s) {missing} does not say "
        f"shuffle=... explicitly, so the gate above cannot read its intent.")


# --------------------------------------------------------------------- #
#  4. the gate's teeth                                                   #
# --------------------------------------------------------------------- #

_BROKEN_SHUFFLE = """
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=c)
"""

_BROKEN_WORKERS = """
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=8)
"""

_FIXED_SHUFFLE = """
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, shuffle=True,
                    **loader_kwargs(seed, "test_loader"))
"""

_FIXED_WORKERS = """
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=8,
                    **loader_kwargs(seed, "test_loader"))
"""


def test_the_gate_rejects_a_shuffling_unseeded_loader():
    """Teeth: the WRONG behaviour, caught."""
    bad = _offending_loaders(_BROKEN_SHUFFLE, "fake_eval.py")
    assert len(bad) == 1 and "shuffle=True" in bad[0], bad


def test_the_gate_rejects_an_unseeded_loader_with_workers():
    bad = _offending_loaders(_BROKEN_WORKERS, "fake_eval.py")
    assert len(bad) == 1 and "num_workers=8" in bad[0], bad


@pytest.mark.parametrize("source", [_FIXED_SHUFFLE, _FIXED_WORKERS])
def test_the_gate_accepts_the_same_loader_once_it_is_seeded(source):
    """Teeth, other half: the gate demands a fix, not merely shuffle=False."""
    assert _offending_loaders(source, "fake_eval.py") == []


def test_the_gate_does_not_pass_everything_it_is_shown():
    """Guards against a gate that returns [] no matter what."""
    assert _offending_loaders(_BROKEN_SHUFFLE, "x.py") != []
    assert _offending_loaders(_BROKEN_WORKERS, "x.py") != []


# --------------------------------------------------------------------- #
#  5. ITEM 2 regression: the T5 evaluator's threshold provenance         #
# --------------------------------------------------------------------- #
#
# `evaluate_unified.py` documented, in the present tense, the leak that
# the selection rule fixed: "with a train/test only split that loader is the TRAINING
# set" and "the price of having no validation split". Under the selection rule the
# threshold is fitted on the val partition (train_unified.py:499-505, fed from
# val_loader at train_unified.py:260-263). These pin the correction.

_UNIFIED_SRC = (EVAL_DIR / "evaluate_unified.py").read_text(encoding="utf-8")


def test_the_unified_evaluator_does_not_describe_the_leak_as_current():
    """The exact sentences that survived the selection rule, now gone or labelled."""
    for stale in ("that loader is the TRAINING set",
                  "the price of having\nno validation split",
                  "(the price of having no validation split)"):
        assert stale not in _UNIFIED_SRC, (
            f"evaluate_unified.py still states {stale!r} as current behaviour; "
            f"the selection rule fitted the threshold on the validation partition.")


def test_the_unified_evaluator_says_where_the_threshold_came_from():
    assert "VALIDATION partition" in _UNIFIED_SRC
    assert "train_unified.py:499-505" in _UNIFIED_SRC
    assert "checkpoint_threshold_source" in _UNIFIED_SRC


def test_the_threshold_comes_from_the_validation_partition():
    """The reportable instance threshold is the one fitted on validation."""
    assert "fitted on the VALIDATION partition" in _UNIFIED_SRC
    assert "Do not report the oracle row" in _UNIFIED_SRC


def test_train_unified_really_does_tune_the_threshold_on_the_val_loader():
    """The docstring's claim, checked against the trainer rather than trusted."""
    src = (CODEBASE / "src" / "training" / "train_unified.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    tuned = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "find_optimal_threshold"]
    assert len(tuned) == 1, f"expected one threshold fit, found {tuned}"
    # The arrays it is handed are accumulated in the `for batch in val_loader`
    # loop, and nowhere else in the file.
    assert "find_optimal_threshold(\n                np.array(val_inst_sims)" in src
    assert "for batch in val_loader:" in src
    assert src.count("val_inst_sims.") == src.count("val_inst_sims.extend"), (
        "val_inst_sims is mutated somewhere other than the val loop")

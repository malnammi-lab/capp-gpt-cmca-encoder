#!/usr/bin/env python3
"""Norm-stats provenance guard.

Two separate checks live in `src/data/brep_dataset.check_norm_stats_provenance`,
and they are tested separately here because they catch different failures:

CHECK 1 -- `split_used` must be "train".

CHECK 2 -- the split file the statistics were measured on must be the split
    file the dataset is built from. A file can say `split_used: "train"` and
    still describe a different split.

CHECK 3 -- `n_models` plus `n_models_missing` must equal the size of the
    partition named. The current train partitions are 4244 CADSynth models
    and 4116 MFInstSeg models.

CHECKS 2 and 3 run only when the caller passes the split file in use.
`test_wiring_*` pins that forwarding.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from src.data.brep_dataset import (
    CMCADataset,
    _split_basename,
    check_norm_stats_provenance,
    load_split,
)

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "datasets" / "splits"

LIVE_SPLIT = SPLITS / "cadsynth_70_10_20_seed42.json"
OLD_SPLIT = SPLITS / "cadsynth_80_20_seed42.json"
LIVE_STATS = SPLITS / "cadsynth_norm_train.json"
SUPERSEDED_STATS = SPLITS / "cadsynth_norm_train_SUPERSEDED_80_20.json"

# CHECK 3 counts the partition, so a stats file paired with the REAL live split
# in a test has to claim the real live train size. Read from the file rather
# than hardcoded, so the day the split is regenerated these tests follow it.
LIVE_TRAIN_N = len(load_split(LIVE_SPLIT, "train"))

# The six call sites that build a CMCADataset from a config. Each must forward
# `dataset.split_file` into its ds_kwargs or CHECKS 2 and 3 are unreachable.
DS_KWARGS_CALL_SITES = [
    "src/training/train_semantic.py",
    "src/training/train_unified.py",
    "src/training/train_link.py",
    "src/evaluation/evaluate_semantic.py",
    "src/evaluation/evaluate_unified.py",
    "src/evaluation/evaluate_link.py",
]


# --------------------------------------------------------------------------- #
# local fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def make_stats(corpus, tmp_path):
    """Write a norm-stats file that is a copy of the fixture's with overrides.

    `None` as a value DELETES the key, so a file with no `split_used` or no
    `split_file` at all can be built. The numeric statistics are the fixture's
    real ones, so a file that passes the guard also standardises correctly.
    """
    base = json.loads(Path(corpus["norm_stats"]).read_text())
    counter = {"n": 0}

    def _make(**overrides):
        payload = dict(base)
        payload.setdefault("corpus", str(corpus["json_dir"]))
        for k, v in overrides.items():
            if v is None:
                payload.pop(k, None)
            else:
                payload[k] = v
        counter["n"] += 1
        out = tmp_path / f"norm_{counter['n']}.json"
        out.write_text(json.dumps(payload, indent=2))
        return out

    return _make


def _dataset(corpus, norm_stats, **kw):
    """Construct the real dataset over the synthetic corpus, standardized view."""
    return CMCADataset(
        json_dir=corpus["json_dir"],
        model_ids=corpus["train_ids"],
        n_classes=corpus["n_classes"],
        attributes="standardized",
        norm_stats=norm_stats,
        verbose=False,
        **kw,
    )


# --------------------------------------------------------------------------- #
# CHECK 1 -- split_used must be "train"   (the item)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("split_used", ["val", "test", "all"])
def test_check1_non_train_split_used_raises(make_stats, split_used):
    stats_path = make_stats(split_used=split_used)
    stats = json.loads(stats_path.read_text())

    with pytest.raises(ValueError) as excinfo:
        check_norm_stats_provenance(stats, stats_path)

    msg = str(excinfo.value)
    assert str(stats_path) in msg, "the message must name the offending file"
    assert repr(split_used) in msg, "the message must name the split it used"
    assert "'train'" in msg, "the message must name the split it should have used"
    assert "compute_norm_stats.py" in msg, "the message must say how to fix it"


@pytest.mark.parametrize("split_used", ["val", "test", "all"])
def test_check1_fires_at_dataset_construction(corpus, make_stats, split_used):
    """Not just the helper -- the real load site in CMCADataset.__init__."""
    stats_path = make_stats(split_used=split_used)

    with pytest.raises(ValueError) as excinfo:
        _dataset(corpus, stats_path)

    assert repr(split_used) in str(excinfo.value)


def test_check1_missing_split_used_raises(make_stats):
    """No provenance field at all is not a pass; it is unverifiable."""
    stats_path = make_stats(split_used=None)
    stats = json.loads(stats_path.read_text())
    assert "split_used" not in stats

    with pytest.raises(ValueError) as excinfo:
        check_norm_stats_provenance(stats, stats_path)

    msg = str(excinfo.value)
    assert "no 'split_used' field" in msg
    assert "compute_norm_stats.py" in msg


def test_check1_train_still_loads_and_standardises(corpus, make_stats):
    """The guard must not break the good path."""
    stats_path = make_stats(split_used="train")
    ds = _dataset(corpus, stats_path)
    assert len(ds) == len(corpus["train_ids"])
    assert ds.face_std is not None and ds.edge_std is not None


def test_check1_holds_for_the_real_live_stats_files():
    """Both shipped norm-stat files are fitted on train. Regression guard."""
    for name in ("cadsynth_norm_train.json", "mfinstseg_norm_train.json"):
        stats = json.loads((SPLITS / name).read_text())
        assert stats["split_used"] == "train", name
        check_norm_stats_provenance(stats, SPLITS / name)   # must not raise


# --------------------------------------------------------------------------- #
# CHECK 2 -- the stats must have been measured on the split now in use
# --------------------------------------------------------------------------- #

def test_check2_reproduces_the_real_2026_09_21_near_miss(tmp_path):
    """The actual incident, with the actual superseded artifact.

    A byte-identical copy of `cadsynth_norm_train_SUPERSEDED_80_20.json` placed
    under the live filename, loaded for a run configured with the 70/10/20 split.
    CHECK 1 passes it (`split_used` is "train"); CHECK 2 must stop it.
    """
    assert SUPERSEDED_STATS.exists(), SUPERSEDED_STATS
    impostor = tmp_path / "cadsynth_norm_train.json"
    impostor.write_bytes(SUPERSEDED_STATS.read_bytes())

    stats = json.loads(impostor.read_text())
    assert stats["split_used"] == "train", "CHECK 1 cannot see this one"
    check_norm_stats_provenance(stats, impostor)            # CHECK 1 alone: passes

    with pytest.raises(ValueError) as excinfo:
        check_norm_stats_provenance(stats, impostor, split_file=LIVE_SPLIT)

    msg = str(excinfo.value)
    assert "STALE NORM STATS" in msg
    assert "cadsynth_80_20_seed42.json" in msg, "must name the split it was measured on"
    assert "cadsynth_70_10_20_seed42.json" in msg, "must name the split in use"
    assert str(stats["n_models"]) in msg, "must show how many models it was fitted on"
    assert "compute_norm_stats.py" in msg, "must say how to fix it"


def test_check2_fires_at_dataset_construction(corpus, make_stats):
    stats_path = make_stats(
        split_used="train",
        split_file="datasets/splits/cadsynth_80_20_seed42.json")

    with pytest.raises(ValueError) as excinfo:
        _dataset(corpus, stats_path, split_file=LIVE_SPLIT)

    assert "STALE NORM STATS" in str(excinfo.value)


def test_check2_matching_split_loads(corpus, make_stats):
    stats_path = make_stats(
        split_used="train",
        split_file="datasets/splits/cadsynth_70_10_20_seed42.json",
        # CHECK 3 counts the partition too, so the good path has to claim the
        # size of the split it is paired with.
        n_models=LIVE_TRAIN_N, n_models_missing=0)
    ds = _dataset(corpus, stats_path, split_file=LIVE_SPLIT)
    assert len(ds) == len(corpus["train_ids"])
    assert Path(ds.split_file) == LIVE_SPLIT


@pytest.mark.parametrize("recorded", [
    "datasets/splits/cadsynth_70_10_20_seed42.json",                    # path inside the repository
    "/tmp/splits/cadsynth_70_10_20_seed42.json",                        # absolute path
    "C:\\Users\\x\\P2\\datasets\\splits\\cadsynth_70_10_20_seed42.json",  # absolute Windows
    "datasets\\splits\\cadsynth_70_10_20_seed42.json",                  # relative Windows
    "cadsynth_70_10_20_seed42.json",                                    # bare basename
    "  datasets/splits/cadsynth_70_10_20_seed42.json  ",                # stray whitespace
    "datasets/splits/CADSynth_70_10_20_seed42.json",                    # case differs
])
def test_check2_does_not_raise_on_equivalent_spellings(make_stats, recorded):
    """The same logical file, spelled the way each machine spells it."""
    stats_path = make_stats(split_used="train", split_file=recorded,
                            n_models=LIVE_TRAIN_N, n_models_missing=0)
    stats = json.loads(stats_path.read_text())
    check_norm_stats_provenance(stats, stats_path, split_file=LIVE_SPLIT)


@pytest.mark.parametrize("recorded", [
    "datasets/splits/cadsynth_80_20_seed42.json",       # the superseded scheme
    "datasets/splits/mfinstseg_70_10_20_seed42.json",   # the other corpus
    "datasets/splits/cadsynth_70_10_20_seed7.json",     # a different seed
])
def test_check2_raises_on_a_genuinely_different_split(make_stats, recorded):
    stats_path = make_stats(split_used="train", split_file=recorded)
    stats = json.loads(stats_path.read_text())
    with pytest.raises(ValueError) as excinfo:
        check_norm_stats_provenance(stats, stats_path, split_file=LIVE_SPLIT)
    assert "STALE NORM STATS" in str(excinfo.value)


def test_check2_missing_split_file_field_raises_when_a_split_is_given(make_stats):
    """Unverifiable provenance is a failure, not a pass."""
    stats_path = make_stats(split_used="train", split_file=None)
    stats = json.loads(stats_path.read_text())
    assert "split_file" not in stats

    with pytest.raises(ValueError) as excinfo:
        check_norm_stats_provenance(stats, stats_path, split_file=LIVE_SPLIT)

    msg = str(excinfo.value)
    assert "no 'split_file' field" in msg
    assert "cadsynth_70_10_20_seed42.json" in msg


def test_check2_is_skipped_when_the_caller_names_no_split(make_stats):
    """The one legitimate way past CHECKS 2 and 3: the random_split path.

    [AUDIT-FIXES P1b] All six now forward `split_file`
    (`test_wiring_*` below), so the only caller that reaches here is one with no
    split file configured at all, which builds its partitions with random_split
    and has nothing to compare the statistics against.
    """
    stats_path = make_stats(split_used="train", split_file=None)
    stats = json.loads(stats_path.read_text())
    check_norm_stats_provenance(stats, stats_path)          # no raise


def test_check2_holds_for_the_real_live_pairs():
    """Each shipped norm-stats file matches its shipped 70/10/20 split."""
    for corpus_name in ("cadsynth", "mfinstseg"):
        stats_path = SPLITS / f"{corpus_name}_norm_train.json"
        split_path = SPLITS / f"{corpus_name}_70_10_20_seed42.json"
        stats = json.loads(stats_path.read_text())
        check_norm_stats_provenance(stats, stats_path, split_file=split_path)


def test_check2_catches_the_superseded_file_for_both_corpora():
    """Neither superseded file can be passed off as the live one."""
    for corpus_name in ("cadsynth", "mfinstseg"):
        stats_path = SPLITS / f"{corpus_name}_norm_train_SUPERSEDED_80_20.json"
        split_path = SPLITS / f"{corpus_name}_70_10_20_seed42.json"
        stats = json.loads(stats_path.read_text())
        with pytest.raises(ValueError):
            check_norm_stats_provenance(stats, stats_path, split_file=split_path)


@pytest.mark.parametrize("empty", [None, ""])
def test_check2_treats_an_empty_split_file_as_no_split_file(corpus, empty):
    """[DEFECT 3] A falsy `split_file` must mean what it means to every caller.

    The guard tested `if split_file is None`, while train_semantic.py,
    train_unified.py, train_link.py, train_resnet.py and all three evaluators
    choose their split path with `if split_file:`. A config carrying
    `split_file: ""` therefore took the random_split branch in the trainer and
    still forwarded "" into ds_kwargs, and the guard fired on a CORRECT live
    pair -- the shipped cadsynth statistics against the shipped cadsynth split.
    A guard that stops a good run is a guard someone switches off.
    """
    stats = json.loads(LIVE_STATS.read_text())
    check_norm_stats_provenance(stats, LIVE_STATS, split_file=empty)  # no raise

    # and through the real constructor, the way ds_kwargs delivers it
    ds = CMCADataset(json_dir=corpus["json_dir"], model_ids=corpus["train_ids"],
                     n_classes=corpus["n_classes"], attributes="standardized",
                     norm_stats=corpus["norm_stats"], split_file=empty,
                     verbose=False)
    assert len(ds) == len(corpus["train_ids"])


# --------------------------------------------------------------------------- #
# CHECK 3 -- the statistics must COUNT the partition they name
#
# [DEFECT 4] Strictly stronger than CHECK 2: a file that fails CHECK 2 for the
# 80/20 scheme fails this too, and this also catches the file CHECK 2 cannot
# see -- one whose name and `split_file` field are both correct.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("corpus_name,superseded_n", [
    ("cadsynth", 4850),         # live train is 4244
    ("mfinstseg", 4704),        # live train is 4116
])
def test_check3_catches_stats_whose_name_is_right_and_contents_are_not(
        tmp_path, corpus_name, superseded_n):
    """The real incident, one level deeper than CHECK 2 can see.

    Take the REAL live stats file, leave its name and its `split_file` field
    exactly as shipped, and set `n_models` to the superseded partition's size --
    i.e. statistics actually fitted on the old 80/20 train partition, filed
    under the live name with the live provenance string. CHECKS 1 and 2 both
    pass it. Nothing else in the tree would notice.
    """
    live = SPLITS / f"{corpus_name}_norm_train.json"
    split = SPLITS / f"{corpus_name}_70_10_20_seed42.json"
    stats = json.loads(live.read_text())
    assert stats["n_models"] == len(load_split(split, "train")), (
        "the shipped pair must agree before this test forges a disagreement")

    stats["n_models"] = superseded_n
    forged = tmp_path / f"{corpus_name}_norm_train.json"
    forged.write_text(json.dumps(stats, indent=2))

    # CHECKS 1 and 2 are blind to it: the name and the recorded split are right.
    assert stats["split_used"] == "train"
    assert _split_basename(stats["split_file"]) == _split_basename(split)

    with pytest.raises(ValueError) as excinfo:
        check_norm_stats_provenance(stats, forged, split_file=split)

    msg = str(excinfo.value)
    assert "STALE NORM STATS (CONTENTS)" in msg
    assert str(superseded_n) in msg, "must name the count the file claims"
    assert str(len(load_split(split, "train"))) in msg, "must name the real size"
    assert "compute_norm_stats.py" in msg, "must say how to fix it"


def test_check3_counts_coverage_not_reads(make_stats):
    """`n_models_missing` is part of the count, not noise.

    compute_norm_stats.py skips a split id whose feature file is absent and
    records how many it skipped. Statistics measured over a partition of 4244
    where 3 ids had no file report n_models=4241, n_models_missing=3 -- and are
    legitimate. It is the COVERAGE that must match the partition.
    """
    ok = make_stats(split_used="train", split_file=str(LIVE_SPLIT),
                    n_models=LIVE_TRAIN_N - 3, n_models_missing=3)
    check_norm_stats_provenance(json.loads(ok.read_text()), ok,
                                split_file=LIVE_SPLIT)          # no raise

    bad = make_stats(split_used="train", split_file=str(LIVE_SPLIT),
                     n_models=LIVE_TRAIN_N - 3, n_models_missing=0)
    with pytest.raises(ValueError) as excinfo:
        check_norm_stats_provenance(json.loads(bad.read_text()), bad,
                                    split_file=LIVE_SPLIT)
    assert "STALE NORM STATS (CONTENTS)" in str(excinfo.value)


@pytest.mark.parametrize("overrides", [
    {"n_models": None},                       # field absent
    {"n_models_missing": None},               # field absent
    {"n_models": "4244"},                     # string, not int
    {"n_models": -1},                         # negative
    {"n_models_missing": True},               # bool masquerading as int
])
def test_check3_refuses_stats_it_cannot_count(make_stats, overrides):
    """Uncountable provenance is a failure, not a pass.

    Defaulting a missing `n_models` to "assume it matches" would reopen the hole
    on demand: strip two fields from the superseded file and it sails through.
    """
    kw = dict(split_used="train", split_file=str(LIVE_SPLIT),
              n_models=LIVE_TRAIN_N, n_models_missing=0)
    kw.update(overrides)
    stats_path = make_stats(**kw)
    stats = json.loads(stats_path.read_text())

    with pytest.raises(ValueError) as excinfo:
        check_norm_stats_provenance(stats, stats_path, split_file=LIVE_SPLIT)
    msg = str(excinfo.value)
    assert "n_models" in msg
    assert "compute_norm_stats.py" in msg


def test_check3_holds_for_the_real_live_pairs():
    """Both shipped pairs satisfy the content check as shipped. Regression guard.

    cadsynth  4244 + 0 == 4244 train ids
    mfinstseg 4116 + 0 == 4116 train ids
    """
    for corpus_name in ("cadsynth", "mfinstseg"):
        stats_path = SPLITS / f"{corpus_name}_norm_train.json"
        split_path = SPLITS / f"{corpus_name}_70_10_20_seed42.json"
        stats = json.loads(stats_path.read_text())
        assert stats["n_models_missing"] == 0, corpus_name
        assert stats["n_models"] == len(load_split(split_path, "train")), corpus_name
        check_norm_stats_provenance(stats, stats_path, split_file=split_path)


# --------------------------------------------------------------------------- #
# THE WIRING -- a check nothing calls protects nothing
#
# [DEFECT 1] `grep -rn "split_file=" src/ runners/` returned nothing outside
# brep_dataset.py itself: every production dataset was built without naming its
# split, so CHECKS 2 and 3 took the early return on all 85 planned runs.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rel", DS_KWARGS_CALL_SITES)
def test_wiring_every_ds_kwargs_site_forwards_split_file(rel):
    """Parse the call site and require `split_file=` in its ds_kwargs.

    Structural rather than behavioural because it pins all six sites, including
    the three evaluators, which need a trained checkpoint to exercise end to
    end. The behavioural half is the train_semantic test below.
    """
    src = (REPO / rel).read_text()
    tree = ast.parse(src, filename=rel)

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "ds_kwargs"
                   for t in node.targets):
            continue
        assert isinstance(node.value, ast.Call), f"{rel}: ds_kwargs is not a call"
        found.append({kw.arg for kw in node.value.keywords})

    assert found, f"{rel}: no `ds_kwargs = dict(...)` assignment found"
    for keys in found:
        assert "split_file" in keys, (
            f"{rel}: ds_kwargs does not forward split_file, so the norm-stats "
            f"staleness check (CHECKS 2 and 3 in src/data/brep_dataset.py) "
            f"cannot run for any run this file launches.")
        assert "norm_stats" in keys, f"{rel}: ds_kwargs lost norm_stats"


def _semantic_cfg(corpus, out_dir, norm_stats):
    """The smallest real train_semantic config over the fixture corpus."""
    return {
        "experiment_name": "wiring",
        "seed": 3,
        "dataset": {"json_dir": str(corpus["json_dir"]),
                    "split_file": str(corpus["split_file"]),
                    "norm_stats": str(norm_stats),
                    "n_classes": corpus["n_classes"],
                    "attributes": "standardized"},
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


def test_wiring_a_real_training_run_is_stopped_by_each_check(
        corpus, make_stats, tmp_path):
    """End to end through the real trainer: the control, then both failures.

    This is the test the incident would have needed. A trainer run is configured
    with the live split file and handed, in turn:
      * the correct statistics                      -> trains
      * statistics measured on the 80/20 split      -> CHECK 2 stops it
      * statistics whose name and split_file field are right and whose contents
        cover the wrong number of models            -> CHECK 3 stops it
    """
    from src.training.train_semantic import train_semantic_model

    n_train = len(corpus["train_ids"])

    # -- control: the good pair must still train, or the two raises below
    #    prove nothing about staleness.
    good = make_stats(split_used="train", split_file=str(corpus["split_file"]),
                      n_models=n_train, n_models_missing=0)
    train_semantic_model(_semantic_cfg(corpus, tmp_path / "good", good),
                         seed_override=3)

    # -- CHECK 2: the 2026-09-21 artifact, under the live filename.
    wrong_split = make_stats(
        split_used="train",
        split_file="datasets/splits/cadsynth_80_20_seed42.json",
        n_models=4850, n_models_missing=0)
    with pytest.raises(ValueError) as excinfo:
        train_semantic_model(_semantic_cfg(corpus, tmp_path / "c2", wrong_split),
                             seed_override=3)
    assert "STALE NORM STATS" in str(excinfo.value)

    # -- CHECK 3: same name, same recorded split, wrong contents.
    wrong_contents = make_stats(
        split_used="train", split_file=str(corpus["split_file"]),
        n_models=n_train + 1, n_models_missing=0)
    with pytest.raises(ValueError) as excinfo:
        train_semantic_model(_semantic_cfg(corpus, tmp_path / "c3", wrong_contents),
                             seed_override=3)
    assert "STALE NORM STATS (CONTENTS)" in str(excinfo.value)


def test_wiring_the_shared_fixture_carries_real_provenance(corpus):
    """[DEFECT 2] The fixture is what makes the wiring survivable.

    The prescribed one-line fix -- add `split_file=ds_cfg.get("split_file")` to
    the six sites -- took the suite from 320 passed to 2 failed / 11 errors,
    because tests/conftest.py wrote `{"split_used": "train", "stats": {}}` and a
    stats file with no `split_file` field is fatal once a caller names a split.
    The fix is the fixture, not a weaker guard: the fixture now writes the same
    provenance block runners/compute_norm_stats.py writes, and it must keep
    satisfying both checks against its own split.
    """
    stats = json.loads(Path(corpus["norm_stats"]).read_text())
    assert stats["split_used"] == "train"
    assert Path(stats["split_file"]) == Path(corpus["split_file"])
    assert stats["n_models"] == len(load_split(corpus["split_file"], "train"))
    assert stats["n_models_missing"] == 0
    check_norm_stats_provenance(stats, corpus["norm_stats"],
                                split_file=corpus["split_file"])


# --------------------------------------------------------------------------- #
# [DEFECT 5] The same failure class, twenty lines below the guard.
#
# A missing feature file was counted and skipped; a corrupt one was printed and
# skipped. Either way the model left the dataset while every count downstream --
# the checkpoint, eval_metrics.json, the provenance block -- still reported the
# partition size. Wrong data, normal-looking numbers: what the guard above
# exists to stop.
# --------------------------------------------------------------------------- #

@pytest.fixture
def mutable_corpus(corpus, tmp_path):
    """A writable copy of the session corpus -- never mutate the shared one."""
    dst = tmp_path / "features"
    shutil.copytree(corpus["json_dir"], dst)
    return dst


def test_a_corrupt_feature_file_raises_instead_of_dropping_the_model(
        corpus, mutable_corpus):
    ids = list(corpus["train_ids"])
    (mutable_corpus / f"{ids[2]}_features.json").write_text('{"n_faces": 3}')

    with pytest.raises(RuntimeError) as excinfo:
        CMCADataset(json_dir=mutable_corpus, model_ids=ids, n_classes=4,
                    verbose=False)

    msg = str(excinfo.value)
    assert ids[2] in msg, "the message must name the file that failed"
    assert "re-extract" in msg.lower(), "the message must say what to do"


def test_a_missing_feature_file_raises_instead_of_shrinking_the_run(
        corpus, mutable_corpus):
    ids = list(corpus["train_ids"])
    (mutable_corpus / f"{ids[1]}_features.json").unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        CMCADataset(json_dir=mutable_corpus, model_ids=ids, n_classes=4,
                    verbose=False)

    msg = str(excinfo.value)
    assert f"{ids[1]}_features.json" in msg
    assert f"1 of {len(ids)}" in msg, "the message must quantify the shortfall"


def test_an_intact_corpus_still_loads_every_requested_model(corpus, mutable_corpus):
    """The control for the two above."""
    ids = list(corpus["train_ids"])
    ds = CMCADataset(json_dir=mutable_corpus, model_ids=ids, n_classes=4,
                     verbose=False)
    assert len(ds) == len(ids)


def _write_faces(d: Path, mid: str, n_faces: int = 4) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{mid}_faces.json").write_text(json.dumps({"faces": [
        {"face_id": i, "label": i % 4,
         "uv_grid": [[[0.0] * 8 for _ in range(8)] for _ in range(7)]}
        for i in range(n_faces)]}))


def test_uv_dataset_raises_on_a_corrupt_face_file(tmp_path):
    d = tmp_path / "faces"
    for mid in ("00000000", "00000001"):
        _write_faces(d, mid)
    (d / "00000001_faces.json").write_text('{"faces": [{"face_id": 0}]}')

    from src.data.uv_dataset import UVGridLabelDataset
    with pytest.raises(RuntimeError) as excinfo:
        UVGridLabelDataset(json_dir=d, n_classes=4, verbose=False)
    assert "00000001_faces.json" in str(excinfo.value)


def test_uv_dataset_raises_on_a_missing_face_file(tmp_path):
    d = tmp_path / "faces"
    _write_faces(d, "00000000")

    from src.data.uv_dataset import UVGridLabelDataset
    with pytest.raises(FileNotFoundError) as excinfo:
        UVGridLabelDataset(json_dir=d, n_classes=4,
                           model_ids=["00000000", "00000009"], verbose=False)
    assert "00000009_faces.json" in str(excinfo.value)


def test_uv_dataset_still_loads_an_intact_corpus(tmp_path):
    """The control for the two above."""
    d = tmp_path / "faces"
    for mid in ("00000000", "00000001"):
        _write_faces(d, mid)

    from src.data.uv_dataset import UVGridLabelDataset
    ds = UVGridLabelDataset(json_dir=d, n_classes=4, verbose=False)
    assert len(ds) == 8 and ds.model_ids == ["00000000", "00000001"]


# --------------------------------------------------------------------------- #
# the comparison primitive
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path,expected", [
    ("datasets/splits/a.json", "a.json"),
    ("datasets\\splits\\a.json", "a.json"),
    ("/tmp/splits/a.json", "a.json"),
    ("C:\\Users\\x\\splits\\A.JSON", "a.json"),
    ("a.json", "a.json"),
    ("  a.json  ", "a.json"),
    (Path("datasets/splits/a.json"), "a.json"),
])
def test_split_basename_normalises(path, expected):
    assert _split_basename(path) == expected


def test_split_basename_keeps_the_distinctions_that_matter():
    """70_10_20 vs 80_20, corpus, and seed must all still separate."""
    a = _split_basename("datasets/splits/cadsynth_70_10_20_seed42.json")
    for other in ("datasets/splits/cadsynth_80_20_seed42.json",
                  "datasets/splits/mfinstseg_70_10_20_seed42.json",
                  "datasets/splits/cadsynth_70_10_20_seed7.json"):
        assert _split_basename(other) != a

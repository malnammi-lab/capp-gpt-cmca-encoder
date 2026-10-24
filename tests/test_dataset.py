"""Dataset contract: attribute views, bias resolution, split files, one-hot.

Several of these guard bugs that were live in this codebase and produced wrong
numbers without ever raising.
"""

import json

import numpy as np
import pytest
import torch

from src.data.brep_dataset import CMCADataset, load_split


# --------------------------------------------------------------------------- #
# bias resolution -- the defect the whole rebuild was about
# --------------------------------------------------------------------------- #

def test_adjacency_and_similarity_are_distinct(corpus):
    """The original bug: `bias_matrix` was attribute cosine similarity, and the
    encoder consumed it believing it was topology. They must never be the same
    array."""
    ds = CMCADataset(json_dir=corpus["json_dir"], n_classes=corpus["n_classes"],
                     bias_type="adjacency", verbose=False)
    item = ds[0]
    assert not torch.allclose(item["adjacency"], item["attr_similarity"])


def test_adjacency_is_binary_symmetric_zero_diagonal(dataset):
    for i in range(len(dataset)):
        a = dataset[i]["adjacency"]
        assert torch.allclose(a, a.T), "adjacency must be symmetric"
        assert torch.diagonal(a).abs().max() == 0, "no face is adjacent to itself"
        assert set(a.unique().tolist()) <= {0.0, 1.0}, "adjacency must be binary"


def test_adjacency_density_varies_across_models(dataset):
    """The defect this suite exists to prevent: the old matrix was exactly
    1.000 for EVERY part in both corpora -- zero variance, therefore zero
    information about topology.

    Note the property is corpus-level, not per-model. A single model may
    legitimately be fully connected: a tetrahedron has four faces all mutually
    adjacent, density exactly 1.0. What cannot happen with real topology is
    that every model reports the same density.
    """
    dens = []
    for i in range(len(dataset)):
        a = dataset[i]["adjacency"]
        n = a.shape[0]
        if n > 1:
            dens.append(a.sum().item() / (n * n - n))
    assert len(dens) > 1
    assert float(np.mean(dens)) < 1.0, "a whole corpus at density 1.0 is the bug"
    assert float(np.std(dens)) > 0.0, "zero variance means the matrix is a constant"


@pytest.mark.parametrize("bias_type", ["similarity", "adjacency", "both"])
def test_bias_type_selects_the_right_matrix(corpus, bias_type):
    ds = CMCADataset(json_dir=corpus["json_dir"], n_classes=corpus["n_classes"],
                     bias_type=bias_type, lambda_adj=1.0, lambda_sim=1.0,
                     verbose=False)
    it = ds[0]
    expect = {"similarity": it["attr_similarity"],
              "adjacency": it["adjacency"],
              "both": it["adjacency"] + it["attr_similarity"]}[bias_type]
    assert torch.allclose(it["bias"], expect)


def test_bias_lambdas_are_applied(corpus):
    ds = CMCADataset(json_dir=corpus["json_dir"], n_classes=corpus["n_classes"],
                     bias_type="both", lambda_adj=2.0, lambda_sim=0.5,
                     verbose=False)
    it = ds[0]
    assert torch.allclose(it["bias"], 2.0 * it["adjacency"] + 0.5 * it["attr_similarity"])


# --------------------------------------------------------------------------- #
# attribute views
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("attributes", ["raw", "scaled", "standardized"])
def test_attribute_views_load(corpus, attributes):
    kw = {"norm_stats": corpus["norm_stats"]} if attributes == "standardized" else {}
    ds = CMCADataset(json_dir=corpus["json_dir"], n_classes=corpus["n_classes"],
                     attributes=attributes, verbose=False, **kw)
    assert ds.attr_dim == 4
    assert torch.isfinite(ds[0]["faces_attr"]).all()


def test_standardized_requires_norm_stats(corpus):
    with pytest.raises(ValueError):
        CMCADataset(json_dir=corpus["json_dir"], attributes="standardized",
                    verbose=False)


def test_invalid_options_are_rejected(corpus):
    with pytest.raises(ValueError):
        CMCADataset(json_dir=corpus["json_dir"], attributes="nonsense", verbose=False)
    with pytest.raises(ValueError):
        CMCADataset(json_dir=corpus["json_dir"], bias_type="nonsense", verbose=False)


def test_area_is_never_negative(corpus):
    """OCC returns a tiny negative mass for degenerate faces; a negative area
    survives standardisation as a spurious outlier."""
    ds = CMCADataset(json_dir=corpus["json_dir"], attributes="raw", verbose=False)
    for i in range(len(ds)):
        assert ds[i]["faces_attr"][:, 1].min() >= 0.0


# --------------------------------------------------------------------------- #
# one-hot ordering -- REGRESSION, this one produced silent garbage
# --------------------------------------------------------------------------- #

def test_one_hot_widens_attributes(corpus):
    ds = CMCADataset(json_dir=corpus["json_dir"], n_classes=corpus["n_classes"],
                     attributes="raw", one_hot_type=True, verbose=False)
    assert ds.attr_dim == 11, "8 type indicators + 3 continuous columns"


def test_one_hot_rows_are_valid_indicators(corpus):
    ds = CMCADataset(json_dir=corpus["json_dir"], attributes="raw",
                     one_hot_type=True, verbose=False)
    block = ds[0]["faces_attr"][:, :8]
    assert torch.allclose(block.sum(dim=1), torch.ones(block.shape[0]))
    assert set(block.unique().tolist()) <= {0.0, 1.0}


def test_one_hot_recovers_the_true_type_index(corpus):
    """`_one_hot_type` recovers the index as round(x * 7), which is only valid
    while column 0 still holds type_index / 7. If standardisation ran first the
    recovered index would be garbage that .clamp(0, 7) silently absorbs -- a
    wrong one-hot with no error. Compare against the raw file."""
    raw = json.loads(next(corpus["json_dir"].glob("00000000_features.json")).read_text())
    expected = [int(round(r[0] * 7)) for r in raw["faces_attr_raw"]]

    ds = CMCADataset(json_dir=corpus["json_dir"], model_ids=["00000000"],
                     attributes="standardized", norm_stats=corpus["norm_stats"],
                     one_hot_type=True, verbose=False)
    got = ds[0]["faces_attr"][:, :8].argmax(dim=1).tolist()
    assert got == expected, "one-hot must be taken BEFORE standardisation"


def test_one_hot_standardises_only_the_continuous_columns(corpus):
    """The 8 indicator columns must survive standardisation untouched."""
    ds = CMCADataset(json_dir=corpus["json_dir"], attributes="standardized",
                     norm_stats=corpus["norm_stats"], one_hot_type=True,
                     verbose=False)
    block = ds[0]["faces_attr"][:, :8]
    assert set(block.unique().tolist()) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# split files
# --------------------------------------------------------------------------- #

def test_load_split_returns_the_saved_ids(corpus):
    assert load_split(corpus["split_file"], "train") == corpus["train_ids"]
    assert load_split(corpus["split_file"], "test") == corpus["test_ids"]


def test_load_split_rejects_a_missing_partition(tmp_path):
    """A partition that is not in the file must raise, not return [].

    Under the selection rule every real split file
    HAS a val partition and the trainers now require one, so the fixture will
    grow one; the assertion is therefore made against a file written here.
    """
    train_test_only = tmp_path / "split_train_test_only.json"
    train_test_only.write_text(json.dumps({"train": ["a", "b"],
                                           "test": ["c"]}))
    with pytest.raises(KeyError):
        load_split(train_test_only, "val")


def test_split_partitions_are_disjoint(corpus):
    tr = set(load_split(corpus["split_file"], "train"))
    te = set(load_split(corpus["split_file"], "test"))
    assert not (tr & te)


def test_dataset_honours_model_ids(corpus):
    ds = CMCADataset(json_dir=corpus["json_dir"],
                     model_ids=load_split(corpus["split_file"], "test"),
                     verbose=False)
    assert len(ds) == len(corpus["test_ids"])
    assert {ds[i]["model_id"] for i in range(len(ds))} == set(corpus["test_ids"])


def test_statistics_report_real_density(dataset):
    st = dataset.get_statistics()
    assert 0.0 < st["adjacency_density_mean"] < 1.0
    assert st["attr_dim"] == 4


# --------------------------------------------------------------------------- #
# the `shuffled` control arm
# --------------------------------------------------------------------------- #

def test_shuffled_preserves_every_graph_statistic(corpus):
    """The control is only valid if it differs from the real matrix in exactly
    one respect: which faces the edges connect. Density, degree sequence,
    symmetry and the zero diagonal must all survive, or a gap between the arms
    could be explained by the graph being sparser or better conditioned rather
    than by the topology being right."""
    real = CMCADataset(json_dir=corpus["json_dir"], bias_type="adjacency",
                       verbose=False)
    ctrl = CMCADataset(json_dir=corpus["json_dir"], bias_type="shuffled",
                       verbose=False)
    assert len(real) == len(ctrl)
    for i in range(len(real)):
        a, s = real[i]["bias"], ctrl[i]["bias"]
        n = a.shape[0]
        assert a.sum() == s.sum(), "edge count must match"
        assert torch.allclose(s, s.T), "must stay symmetric"
        assert s.diagonal().abs().max() == 0, "no self-loops"
        assert sorted(a.sum(1).tolist()) == sorted(s.sum(1).tolist()), \
            "degree sequence must be preserved"
        assert set(s.unique().tolist()) <= {0.0, 1.0}, "must stay binary"


def test_shuffled_actually_differs_somewhere(corpus):
    """If it never differed it would be a copy of the adjacency arm, and the
    experiment would silently answer nothing.

    Note a permutation cannot change a complete graph or a 2-face part, so the
    check is that SOME model differs, not every model.
    """
    real = CMCADataset(json_dir=corpus["json_dir"], bias_type="adjacency",
                       verbose=False)
    ctrl = CMCADataset(json_dir=corpus["json_dir"], bias_type="shuffled",
                       verbose=False)
    differ = sum(1 for i in range(len(real))
                 if not torch.equal(real[i]["bias"], ctrl[i]["bias"]))
    assert differ > 0, "the control is identical to the real matrix everywhere"


def test_shuffled_is_fixed_across_epochs(corpus):
    """The permutation is derived from the model id, so repeated access returns
    the same matrix. If it were resampled per epoch this would be a noise
    augmentation, which is a different and much weaker experiment."""
    ds = CMCADataset(json_dir=corpus["json_dir"], bias_type="shuffled",
                     verbose=False)
    first = [ds[i]["bias"].clone() for i in range(len(ds))]
    again = CMCADataset(json_dir=corpus["json_dir"], bias_type="shuffled",
                        verbose=False)
    for i, f in enumerate(first):
        assert torch.equal(f, ds[i]["bias"]), "changed between accesses"
        assert torch.equal(f, again[i]["bias"]), "changed between dataset builds"


def test_shuffled_leaves_the_real_adjacency_untouched(corpus):
    """`adjacency` and `attr_similarity` are emitted alongside `bias`; the
    control must not corrupt them, or a model reading them directly would see
    the permuted graph."""
    ds = CMCADataset(json_dir=corpus["json_dir"], bias_type="shuffled",
                     verbose=False)
    real = CMCADataset(json_dir=corpus["json_dir"], bias_type="adjacency",
                       verbose=False)
    for i in range(len(ds)):
        assert torch.equal(ds[i]["adjacency"], real[i]["adjacency"])


def test_other_arms_do_not_pay_for_the_control(corpus):
    """The permutation is only computed when the arm asks for it."""
    ds = CMCADataset(json_dir=corpus["json_dir"], bias_type="adjacency",
                     verbose=False)
    assert ds.data[0].get("adjacency_shuffled") is None


def test_shuffled_permutation_is_derived_from_the_model_id(corpus):
    """Each model must get its OWN permutation.

    A single permutation shared by every model would still be "wrong topology"
    per part, so the control would not be useless -- but it would be a
    systematic relabelling rather than an independent scramble, and two parts
    with the same face count would receive identical wrong graphs. The seed is
    taken from the model id precisely to avoid that.

    (This test exists because a mutation that removed the per-model seeding
    survived the suite: torch.Generator() has a fixed default seed, so the
    permutation stayed deterministic and "fixed across epochs" still held.)
    """
    from src.data.brep_dataset import _shuffle_adjacency
    import torch as _t

    _t.manual_seed(0)
    n = 9
    a = (_t.rand(n, n) < 0.35).float()
    a = _t.maximum(a, a.T)
    a.fill_diagonal_(0)

    perms = {mid: _shuffle_adjacency(a, mid)
             for mid in ("00000001", "00000002", "00000003", "00000004")}
    distinct = {tuple(v.flatten().tolist()) for v in perms.values()}
    assert len(distinct) > 1, (
        "every model id produced the same permutation -- the seed is not being "
        "derived from the model id")

    # and the same id must still reproduce exactly
    assert _t.equal(perms["00000001"], _shuffle_adjacency(a, "00000001"))


# --------------------------------------------------------------------------- #
# The committed 70/10/20 split files.
#
# These run against the files in datasets/splits/. The test ids of the
# 70/10/20 files must match the test ids of the 80/20 files. They skip
# while those files are absent.
# --------------------------------------------------------------------------- #

from pathlib import Path as _Path  # noqa: E402  (kept local to this section)

SPLITS_DIR = _Path(__file__).resolve().parents[1] / "datasets" / "splits"

# Counts used below are the ones the paper reports.
# section 1, "CADSynth 4,244/606/1,212; MFInstSeg 4,116/588/1,176".
DECISION_B_SPLITS = [
    # corpus,       n_train, n_val, n_test
    ("cadsynth",    4244,    606,   1212),
    ("mfinstseg",   4116,    588,   1176),
]
DECISION_B_IDS = [c[0] for c in DECISION_B_SPLITS]


def _new_split_path(corpus_name):
    return SPLITS_DIR / "{}_70_10_20_seed42.json".format(corpus_name)


def _old_split_path(corpus_name):
    return SPLITS_DIR / "{}_80_20_seed42.json".format(corpus_name)


def _norm_stats_path(corpus_name):
    return SPLITS_DIR / "{}_norm_train.json".format(corpus_name)


def _load_new_split(corpus_name):
    """The 70/10/20 payload, or a clean skip while it does not exist yet."""
    path = _new_split_path(corpus_name)
    if not path.exists():
        pytest.skip(
            "{} not generated yet ; this test "
            "asserts as soon as it exists".format(path.name))
    return json.loads(path.read_text())


def _load_old_split(corpus_name):
    path = _old_split_path(corpus_name)
    assert path.exists(), (
        "{} is missing -- the frozen test partition has no source of truth "
        "any more".format(path.name))
    return json.loads(path.read_text())


@pytest.mark.parametrize("corpus_name", DECISION_B_IDS)
def test_new_split_test_ids_are_identical_to_the_80_20_file(corpus_name):
    """THE non-negotiable one.

    The selection rule carves validation out of the old TRAIN partition. The test
    partition is frozen: same ids, same order, same serialisation. If this ever
    fails, the 70/10/20 results are not comparable with anything and the file
    must be regenerated, never patched by editing the test.
    """
    new = _load_new_split(corpus_name)
    old = _load_old_split(corpus_name)

    assert new["test"] == old["test"], (
        "{}: the new test partition is not element-for-element the old one"
        .format(corpus_name))
    # and byte-identical as serialised, which also catches type drift (an id
    # silently becoming an int, say) that == would let through
    assert json.dumps(new["test"]) == json.dumps(old["test"])


@pytest.mark.parametrize("corpus_name", DECISION_B_IDS)
def test_new_split_val_comes_out_of_the_old_train_partition(corpus_name):
    """Val is carved from old train, so train+val must reconstruct it exactly.

    This is the other half of the freeze: it proves no test id leaked into
    train or val, and that no id was invented or dropped along the way.
    """
    new = _load_new_split(corpus_name)
    old = _load_old_split(corpus_name)

    assert set(new["train"]) | set(new["val"]) == set(old["train"]), (
        "{}: train+val is not the old train partition".format(corpus_name))


@pytest.mark.parametrize("corpus_name", DECISION_B_IDS)
def test_new_split_partitions_are_disjoint_and_complete(corpus_name):
    new = _load_new_split(corpus_name)
    train, val, test = new["train"], new["val"], new["test"]
    s_train, s_val, s_test = set(train), set(val), set(test)

    assert len(s_train) == len(train), "duplicate ids inside train"
    assert len(s_val) == len(val), "duplicate ids inside val"
    assert len(s_test) == len(test), "duplicate ids inside test"

    assert not (s_train & s_val), "train and val overlap"
    assert not (s_train & s_test), "train and test overlap"
    assert not (s_val & s_test), "val and test overlap"

    assert len(train) + len(val) + len(test) == new["n_total"]
    assert len(s_train | s_val | s_test) == new["n_total"]


@pytest.mark.parametrize("corpus_name,n_train,n_val,n_test", DECISION_B_SPLITS)
def test_new_split_counts_match_the_approved_paper_text(
        corpus_name, n_train, n_val, n_test):
    """The setup paragraph quotes these numbers. They are load-bearing text."""
    new = _load_new_split(corpus_name)

    assert len(new["train"]) == n_train
    assert len(new["val"]) == n_val
    assert len(new["test"]) == n_test
    assert new["n_total"] == n_train + n_val + n_test
    assert new["seed"] == 42


@pytest.mark.parametrize("corpus_name", DECISION_B_IDS)
def test_new_split_partitions_are_sorted(corpus_name):
    """make_splits.py writes sorted id lists; a dataset that iterates the file
    in order therefore has a stable order independent of the shuffle."""
    new = _load_new_split(corpus_name)
    for which in ("train", "val", "test"):
        assert new[which] == sorted(new[which]), \
            "{} partition {} is not sorted".format(corpus_name, which)


@pytest.mark.parametrize("corpus_name", DECISION_B_IDS)
def test_load_split_reads_all_three_partitions_of_the_new_file(corpus_name):
    """The loader the trainers call must see a real val partition -- that is
    the whole point of the selection rule early stopping."""
    new = _load_new_split(corpus_name)
    path = str(_new_split_path(corpus_name))
    for which in ("train", "val", "test"):
        assert load_split(path, which) == new[which]


@pytest.mark.parametrize("corpus_name,n_train,n_val,n_test", DECISION_B_SPLITS)
def test_norm_stats_were_refit_on_the_new_train_partition(
        corpus_name, n_train, n_val, n_test):
    """The leak this item exists to close.

    The committed norm stats were fitted on the OLD 4850/4704-model train
    partition, which under the selection rule contains the entire validation set. The
    `split_used` guard cannot catch that: the stale file still says train and
    merely means the OLD train. `n_models` can catch it, so pin it.
    """
    _load_new_split(corpus_name)  # skip in lockstep with the split files

    stats_path = _norm_stats_path(corpus_name)
    assert stats_path.exists(), "{} is missing".format(stats_path.name)
    stats = json.loads(stats_path.read_text())

    assert stats["split_used"] == "train"
    assert stats["n_models"] == n_train, (
        "{}: norm stats were fitted on {} models but the new train partition "
        "has {}. Stale statistics fitted on data that now includes the whole "
        "validation set are a preprocessing leak into the selection data; "
        "refit with runners/compute_norm_stats.py against the 70/10/20 file."
        .format(corpus_name, stats["n_models"], n_train))
    assert stats["n_models_missing"] == 0
    assert _new_split_path(corpus_name).name in str(stats["split_file"]), (
        "{}: norm stats record split_file {!r}, not the 70/10/20 file"
        .format(corpus_name, stats["split_file"]))
    # every measured view must be non-empty and non-degenerate enough to
    # standardise with
    for key in ("faces_attr_raw", "edges_attr_raw", "faces_attr", "edges_attr"):
        view = stats["stats"][key]
        assert view["n"] > 0, "{} has no rows".format(key)
        assert len(view["mean"]) == 4 and len(view["std_safe"]) == 4

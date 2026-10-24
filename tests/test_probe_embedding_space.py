"""Tests for `runners/probe_embedding_space.py` -- the embedding-space probe.

WHY THESE TESTS LOOK THE WAY THEY DO
------------------------------------
The probe exists to answer a question no T3 number reaches: do CMCA and CE
build different face-embedding SPACES, given that they classify faces
indistinguishably (results-T3.md section 8.2.1)? It cannot be validated by
running it, because the studio is stopped and no `.pth` from the batch was ever
pulled home.

So it is validated the way a measuring instrument is validated: **on inputs
whose answer is known before the instrument is applied.**

  * Embeddings built WITH class structure must score far above chance.
  * Embeddings built WITHOUT it must score AT chance -- not at zero, at the
    chance value the label histogram implies, which this suite computes
    independently.
  * The O(N*D) separation identity must equal the brute-force O(N^2) pair
    average it replaces, to 1e-10.
  * mAP must equal a value computed by hand on four unit vectors at stated
    angles.

A metric that only ever runs on real data can be wrong by a factor of two and
nobody finds out. These tests would catch that.

The second half of the file is about the ONE CODE PATH requirement. The four
T3 cells must reach the scorer through code that differs only in how the
embedding is fetched, or a gap between two arms is a gap between two scorers.
`test_ce_and_cmca_share_the_encoder_path` and `test_resnet_arms_share_the_uv
_tower` hold that, and `test_ce_forward_returns_logits_not_embeddings` records
what goes wrong without the dispatch: the CE arm would be scored on its logits.

What `tests/test_models.py` already covers, and what it does not:
`test_ce_encode_returns_embeddings_not_logits` (`:46`) asserts `encode` returns
`d_model` columns, and `test_ce_and_cmca_share_an_identical_trunk` (`:56`)
compares the two arms' state-dict SHAPES. Neither loads one arm's weights into
the other and compares the OUTPUT, and neither goes through the probe's
dispatch -- so neither notices if `graph_face_embeddings` stops calling
`encode`. That is the gap the two tests here close.

The last section trains two real (tiny) models on the synthetic corpus fixture
and runs the whole runner end to end, CMCA arm and CE arm, because a probe that
only ever ran on numpy arrays is a probe whose wiring nothing tested -- which is
exactly the defect class `tests/test_run_integration.py` was written for.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODEBASE = Path(__file__).resolve().parents[1]
if str(CODEBASE) not in sys.path:
    sys.path.insert(0, str(CODEBASE))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, CODEBASE / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe_mod():
    return _load("probe_embedding_space", "runners/probe_embedding_space.py")


# --------------------------------------------------------------------- #
#  Synthetic embeddings with a KNOWN answer                              #
# --------------------------------------------------------------------- #

def _clustered(n_classes=8, per_class=50, dim=32, spread=0.05, seed=0):
    """Embeddings with real class structure: tight blobs, far apart.

    Every measure in the probe must score this near its ceiling. `spread` is
    the isotropic noise added to each class's unit centre; at 0.05 the blobs
    are far tighter than the angle between random centres in 32-d.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_classes, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    emb = (np.repeat(centres, per_class, axis=0)
           + spread * rng.normal(size=(n_classes * per_class, dim)))
    labels = np.repeat(np.arange(n_classes), per_class)
    return emb, labels


def _unstructured(n=1200, n_classes=8, dim=32, seed=1):
    """Embeddings with NO class structure: labels drawn independently.

    Every measure must land at its chance value. This is the half of the
    validation that catches a metric which is high because of how it is
    written rather than because of what is in the data.
    """
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim))
    labels = rng.integers(0, n_classes, size=n)
    return emb, labels


def _brute_force_separation(emb, labels):
    """The O(N^2) definition the probe's O(N*D) identity replaces."""
    x = np.asarray(emb, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    dist = 1.0 - x @ x.T
    same = np.asarray(labels)[:, None] == np.asarray(labels)[None, :]
    off = ~np.eye(len(x), dtype=bool)
    return float(dist[same & off].mean()), float(dist[(~same) & off].mean())


def test_separated_embeddings_score_far_above_chance(probe_mod):
    """Known-separated space: every measure must be near its ceiling."""
    emb, labels = _clustered()
    rep = probe_mod.embedding_space_report(emb, labels, seed=0)

    sep, sil, ret = rep["separation"], rep["silhouette"], rep["retrieval"]
    chance = ret["chance_same_class_rate"]

    assert sep["intra_inter_ratio"] < 0.2, (
        f"intra/inter ratio {sep['intra_inter_ratio']:.4f} on embeddings built "
        f"as tight, far-apart class blobs. 1.0 is no structure at all; this "
        f"construction has nothing BUT structure.")
    assert sil["silhouette_cosine"] > 0.8, sil
    assert ret["knn_accuracy_at_1"] > 0.95, ret
    assert ret["knn_accuracy_at_5"] > 0.95, ret
    assert ret["mean_average_precision"] > 0.95, ret
    assert chance < 0.2, (
        f"chance is {chance:.4f} with 8 balanced classes; it should be near "
        f"0.125. If chance itself is wrong, every score above is unreadable.")


def test_random_embeddings_score_at_chance(probe_mod):
    """No class structure: every measure must land at ITS chance value.

    Not at zero -- at chance. k-NN accuracy on 8 balanced classes is ~0.125 by
    luck alone, and a probe that called that a result would report every
    untrained encoder as mildly successful.
    """
    emb, labels = _unstructured()
    rep = probe_mod.embedding_space_report(emb, labels, seed=0)

    sep, sil, ret = rep["separation"], rep["silhouette"], rep["retrieval"]
    chance = ret["chance_same_class_rate"]

    assert sep["intra_inter_ratio"] == pytest.approx(1.0, abs=0.03), (
        f"intra/inter ratio {sep['intra_inter_ratio']:.4f} on embeddings whose "
        f"labels were drawn independently of their positions. It must be 1.0: "
        f"a face's own class is no closer than any other.")
    assert abs(sil["silhouette_cosine"]) < 0.05, sil
    assert ret["knn_accuracy_at_1"] == pytest.approx(chance, abs=0.05), ret
    assert ret["knn_accuracy_at_5"] == pytest.approx(chance, abs=0.05), ret
    assert ret["mean_average_precision"] == pytest.approx(chance, abs=0.05), ret


def test_separation_ratio_matches_brute_force_pairs(probe_mod):
    """The O(N*D) identity must equal the O(N^2) pair average it replaces.

    `class_separation` never forms the pairwise matrix -- 34,338 test faces is
    1.2e9 pairs. It uses ``sum of cos over ordered pairs of S == ||sum(S)||^2
    - |S|`` for unit rows. If that algebra were wrong the number would still
    look like a ratio, and nothing else in the probe would notice.
    """
    emb, labels = _clustered(n_classes=5, per_class=17, dim=11, spread=0.4,
                             seed=7)
    got = probe_mod.class_separation(emb, labels)
    want_intra, want_inter = _brute_force_separation(emb, labels)

    assert got["mean_intra_cosine_distance"] == pytest.approx(want_intra,
                                                              abs=1e-10)
    assert got["mean_inter_cosine_distance"] == pytest.approx(want_inter,
                                                              abs=1e-10)
    assert got["intra_inter_ratio"] == pytest.approx(want_intra / want_inter,
                                                     abs=1e-10)


def test_knn_excludes_the_query_itself(probe_mod):
    """A query must never retrieve itself -- that would score any space at 1.0.

    Eight unit vectors evenly around a circle with ALTERNATING labels: every
    point's two nearest OTHER points carry the opposite label, so leave-one-out
    k-NN at k=1 is exactly 0.0. Include the query in its own ranking and it
    becomes exactly 1.0, whatever the embeddings are.
    """
    ang = np.deg2rad(np.arange(8) * 45.0)
    emb = np.column_stack([np.cos(ang), np.sin(ang)])
    labels = np.array([0, 1] * 4)

    ret = probe_mod.retrieval(emb, labels, ks=(1,), cap=None,
                              rng=np.random.default_rng(0))
    assert ret["knn_accuracy_at_1"] == 0.0, (
        f"k-NN at 1 is {ret['knn_accuracy_at_1']}, but on this construction "
        f"every nearest OTHER point carries the opposite label. A non-zero "
        f"value means the query is matching itself.")


def test_map_equals_a_hand_computed_value(probe_mod):
    """mAP pinned to arithmetic done by hand, and shown not to be top-1.

    Four unit vectors at 0, 20, 95 and 200 degrees, labels [0, 1, 0, 1]. By
    hand, ranking each query's three others by cosine similarity:

        q0 -> [1, 2, 3]  relevant at rank 2  AP = 1/2
        q1 -> [0, 2, 3]  relevant at rank 3  AP = 1/3
        q2 -> [1, 0, 3]  relevant at rank 2  AP = 1/2
        q3 -> [2, 0, 1]  relevant at rank 3  AP = 1/3

    mAP = 5/12 = 0.41666..., while k-NN at 1 is 0.0 -- not one query's nearest
    neighbour shares its label. A "mAP" that merely re-reported the top-1 hit
    rate would be 0.0 here, so this fixes both the arithmetic and the claim in
    the docstring that mAP reads the WHOLE ranking.
    """
    ang = np.deg2rad([0.0, 20.0, 95.0, 200.0])
    emb = np.column_stack([np.cos(ang), np.sin(ang)])
    labels = np.array([0, 1, 0, 1])

    ret = probe_mod.retrieval(emb, labels, ks=(1,), cap=None,
                              rng=np.random.default_rng(0))
    assert ret["knn_accuracy_at_1"] == 0.0
    assert ret["mean_average_precision"] == pytest.approx(5.0 / 12.0, abs=1e-12)
    assert ret["chance_same_class_rate"] == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_chance_rate_matches_the_label_histogram(probe_mod):
    """Chance is read off the histogram, not assumed to be 1/n_classes.

    Counts 6 / 3 / 1 over 10 faces give
    (6/10)(5/9) + (3/10)(2/9) + (1/10)(0/9) = 0.4, not 1/3. Getting this wrong
    makes an imbalanced split look like a successful one.
    """
    rng = np.random.default_rng(3)
    emb = rng.normal(size=(10, 6))
    labels = np.array([0] * 6 + [1] * 3 + [2])

    ret = probe_mod.retrieval(emb, labels, ks=(1,), cap=None, rng=rng)
    assert ret["chance_same_class_rate"] == pytest.approx(0.4, abs=1e-12)


def test_subsample_is_stratified_seeded_and_keeps_every_class(probe_mod):
    """The cap must not silently delete a rare class or move between runs.

    Asserted as the QUOTA PROPERTY, not as a symptom. An earlier version of
    this test only checked that all three classes survived, and a mutant that
    replaced the stratified draw with a plain uniform one PASSED it: with 200
    of 1000 drawn, the rare class survived by luck at that seed. A test that
    passes against the mutation it exists to catch proves nothing.

    Two things here are exact rather than probable:

      * each class's count is ``min(n_c, max(2, round(cap * n_c / n)))``, the
        stratified quota, computed independently below;
      * the TOTAL is 201, not 200. The floor of two members for the 3-face
        class pushes the sample one over the cap, so any sampler that simply
        returns `cap` indices -- which is what an unstratified draw does --
        fails on the length alone, at every seed.
    """
    labels = np.array([0] * 1000 + [1] * 90 + [2] * 3)
    n, cap = len(labels), 200
    want = {c: min(int((labels == c).sum()),
                   max(2, int(round(cap * int((labels == c).sum()) / n))))
            for c in (0, 1, 2)}
    assert want == {0: 183, 1: 16, 2: 2}, want   # pin the arithmetic itself

    seen = []
    for seed in (0, 7):
        idx = probe_mod.stratified_subsample(labels, cap,
                                             np.random.default_rng(seed))
        u, counts = np.unique(labels[idx], return_counts=True)
        got = dict(zip(u.tolist(), counts.tolist()))
        assert got == want, (
            f"seed {seed}: class counts {got} are not the stratified quota "
            f"{want}. An unstratified draw gives roughly these proportions "
            f"and can drop the 3-face class entirely.")
        assert len(idx) == 201, (
            f"seed {seed}: {len(idx)} indices. The two-member floor on the "
            f"rare class puts this one OVER the cap of {cap}; a sampler that "
            f"returns exactly {cap} is not stratified.")
        seen.append(idx)

    again = probe_mod.stratified_subsample(labels, cap,
                                           np.random.default_rng(0))
    assert np.array_equal(seen[0], again), "same seed gave a different sample"
    assert not np.array_equal(seen[0], seen[1]), (
        "two different seeds gave the identical sample, so the draw is not "
        "random at all -- it is a fixed prefix wearing an rng")
    assert len(seen[0]) < len(labels)


def test_report_is_deterministic_under_a_seed(probe_mod):
    """Same arrays, same seed -> byte-identical JSON, with the cap active."""
    emb, labels = _clustered(n_classes=6, per_class=60, dim=16, seed=11)
    kw = dict(silhouette_cap=120, retrieval_cap=150)

    a = probe_mod.embedding_space_report(emb, labels, seed=4, **kw)
    b = probe_mod.embedding_space_report(emb, labels, seed=4, **kw)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["silhouette"]["subsampled"] is True
    assert a["retrieval"]["subsampled"] is True


def test_single_class_raises_instead_of_scoring_zero(probe_mod):
    """One class is not a separation of zero -- it is no measurement at all."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(20, 8))
    labels = np.zeros(20, dtype=int)

    with pytest.raises(ValueError, match="at least two classes"):
        probe_mod.class_separation(emb, labels)


def test_collapsed_space_raises_instead_of_reporting_one(probe_mod):
    """Every embedding identical is a dead encoder, not "no structure".

    The ratio would be 0/0. Returning 1.0 there would print as "the labels
    explain none of the geometry" when the truth is that there is no geometry.
    """
    emb = np.tile(np.ones(8) / np.sqrt(8.0), (20, 1))
    labels = np.array([0] * 10 + [1] * 10)

    with pytest.raises(ValueError, match="collapsed encoder"):
        probe_mod.class_separation(emb, labels)


def test_zero_length_embeddings_raise(probe_mod):
    """A zero row has no direction; normalising it yields nan, not a score."""
    emb = np.eye(6)[:, :4]
    emb[2] = 0.0
    with pytest.raises(ValueError, match="zero length"):
        probe_mod.unit_rows(emb)


def test_padding_label_minus_one_is_refused(probe_mod):
    """-1 is `collate_cmca`'s padding marker, not a 26th class."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(12, 5))
    labels = np.array([0, 1, -1, 1, 0, 1, 0, 0, 1, -1, 0, 1])

    with pytest.raises(ValueError, match="negative"):
        probe_mod.embedding_space_report(emb, labels, seed=0)


def test_k_larger_than_the_gallery_raises(probe_mod):
    """Padding a short neighbourhood would score every space at 1.0."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(4, 5))
    labels = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="needs 6 faces"):
        probe_mod.retrieval(emb, labels, ks=(1, 5), cap=None, rng=rng)


# --------------------------------------------------------------------- #
#  ONE CODE PATH: the four T3 cells must reach the scorer the same way   #
# --------------------------------------------------------------------- #

def _graph_batch(n_models=2, n_faces=5, n_edges=7, seed=0):
    """A collated batch in the shape `collate_fn_brep` produces."""
    g = torch.Generator().manual_seed(seed)
    return {
        "faces_uv": torch.rand(n_models, n_faces, 7, 8, 8, generator=g),
        "faces_attr": torch.rand(n_models, n_faces, 4, generator=g),
        "edges_attr": torch.rand(n_models, n_edges, 4, generator=g),
        "bias": torch.rand(n_models, n_faces, n_faces, generator=g),
        "face_mask": torch.ones(n_models, n_faces, dtype=torch.bool),
        "edge_mask": torch.ones(n_models, n_edges, dtype=torch.bool),
    }


def test_ce_forward_returns_logits_not_embeddings():
    """The reason `graph_face_embeddings` cannot just call `forward`.

    This is not a test of the probe -- it is the fact the probe's dispatch
    exists for, asserted so the dispatch cannot be "simplified" away later.
    """
    from src.models.ce_encoder import SemanticClassificationModel

    n_classes, d_model = 6, 16
    model = SemanticClassificationModel(
        n_classes=n_classes, d_model=d_model, n_heads=2, n_layers=1,
        dim_feedforward=16, dropout=0.0).eval()
    batch = _graph_batch()

    with torch.no_grad():
        logits = model(batch["faces_uv"], batch["faces_attr"],
                       batch["edges_attr"], batch["bias"],
                       face_mask=batch["face_mask"],
                       edge_mask=batch["edge_mask"])
        emb = model.encode(batch["faces_uv"], batch["faces_attr"],
                           batch["edges_attr"], batch["bias"],
                           face_mask=batch["face_mask"],
                           edge_mask=batch["edge_mask"])

    assert logits.shape[-1] == n_classes
    assert emb.shape[-1] == d_model
    assert logits.shape != emb.shape, (
        "if forward and encode agreed in shape, scoring the CE arm on its "
        "logits would be undetectable")


def test_ce_and_cmca_share_the_encoder_path(probe_mod):
    """Same encoder weights in both arms -> IDENTICAL probe embeddings.

    The CMCA arm's model IS a `BRepEncoder`; the CE arm wraps one and returns
    logits from `forward`. `graph_face_embeddings` dispatches on `encode`, so
    with the same weights the two must be indistinguishable to the scorer. If
    the dispatch were dropped, this fails on shape -- which is the whole point:
    a comparison of two arms must not be a comparison of two tensors.
    """
    from src.models.ce_encoder import SemanticClassificationModel
    from src.models.encoders import SemanticSegmentationModel

    kw = dict(d_model=16, n_heads=2, n_layers=1, dim_feedforward=16,
              dropout=0.0)
    cmca = SemanticSegmentationModel(**kw).eval()
    ce = SemanticClassificationModel(n_classes=6, **kw).eval()
    ce.encoder.load_state_dict(cmca.state_dict())

    batch = _graph_batch(seed=5)
    with torch.no_grad():
        a = probe_mod.graph_face_embeddings(cmca, batch)
        b = probe_mod.graph_face_embeddings(ce, batch)

    assert a.shape == b.shape == (2, 5, 16)
    assert torch.allclose(a, b, atol=1e-6), (
        "the two arms' embeddings differ although their encoders carry "
        "identical weights, so the probe is reading a different tensor in "
        "each arm and any gap it reports is unattributable")


def test_resnet_arms_share_the_uv_tower(probe_mod):
    """Same UV tower weights in both ResNet arms -> identical embeddings."""
    from src.models.resnet_encoder import (CrossModalAlignmentModel,
                                           ResNetClassificationModel)

    cmca = CrossModalAlignmentModel(n_classes=6, embedding_dim=32).eval()
    ce = ResNetClassificationModel(n_classes=6, embedding_dim=32,
                                   dropout=0.0).eval()
    ce.uv_encoder.load_state_dict(cmca.uv_encoder.state_dict())

    uv = torch.rand(3, 7, 8, 8, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        a = probe_mod.resnet_face_embeddings(cmca, uv)
        b = probe_mod.resnet_face_embeddings(ce, uv)

    assert a.shape == b.shape == (3, 32)
    assert torch.allclose(a, b, atol=1e-6)


def test_unknown_model_type_is_refused_not_guessed(probe_mod):
    """A model whose embedding this probe cannot reach must stop the run."""
    cfg = {"model": {"type": "link_predictor"}, "dataset": {}}
    with pytest.raises(ValueError, match="does not know how to reach"):
        probe_mod.collect_test_embeddings(cfg, {}, "cpu")


# --------------------------------------------------------------------- #
#  End to end: a real (tiny) checkpoint, both arms, the whole runner     #
# --------------------------------------------------------------------- #

def _train_config(corpus, out_dir, loss_type):
    """A complete config on the synthetic corpus, in the real configs' shape."""
    return {
        "experiment_name": f"probe_{loss_type}",
        "seed": 0,
        "dataset": {
            "json_dir": str(corpus["json_dir"]),
            "split_file": str(corpus["split_file"]),
            "norm_stats": str(corpus["norm_stats"]),
            "n_classes": corpus["n_classes"],
            "attributes": "standardized",
        },
        "bias": {"type": "both", "lambda_adj": 1.0, "lambda_sim": 1.0},
        "model": {"type": "semantic_segmentation", "version": "v1",
                  "d_model": 16, "n_heads": 2, "n_layers": 1,
                  "dim_feedforward": 16, "dropout": 0.1},
        "label_encoder": {"embedding_dim": 16},
        "loss": {"type": loss_type, "temperature": 0.07,
                 "use_class_weights": False},
        "training": {"max_epochs": 2, "patience": 1, "batch_size": 2,
                     "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0},
        "paths": {"checkpoint": str(out_dir / "ckpt.pth"),
                  "figures": str(out_dir / "figs") + "/"},
        "device": "cpu",
    }


@pytest.fixture(scope="module")
def trained_arms(corpus, tmp_path_factory):
    """One really-trained checkpoint per objective, on the fixture corpus.

    Slow on purpose, in the spirit of `tests/test_run_integration.py`: a probe
    validated only on numpy arrays is a probe whose model loading, dataset
    construction, masking and provenance wiring nothing has ever executed.
    """
    from src.training.train_semantic import train_semantic_model

    arms = {}
    for loss_type in ("prototype_infonce", "cross_entropy"):
        out = tmp_path_factory.mktemp(f"probe_{loss_type}")
        cfg = _train_config(corpus, out, loss_type)
        train_semantic_model(copy.deepcopy(cfg), seed_override=1)
        arms[loss_type] = {"cfg": cfg, "ckpt": out / "ckpt" / "seed1.pth"}
    return arms


@pytest.mark.parametrize("loss_type",
                         ["prototype_infonce", "cross_entropy"])
def test_probe_runs_end_to_end_on_a_real_checkpoint(probe_mod, trained_arms,
                                                    loss_type):
    """The runner loads either arm, embeds the TEST faces and scores them."""
    arm = trained_arms[loss_type]
    assert arm["ckpt"].is_file(), f"no checkpoint at {arm['ckpt']}"

    record = probe_mod.probe(
        copy.deepcopy(arm["cfg"]), "cfg-in-memory", arm["ckpt"],
        seed=1, device="cpu", ks=(1, 5), verbose=False)

    assert record["objective"] == loss_type
    # 18 faces across the 3 held-out models of the fixture split -- the whole
    # test partition and nothing else. A larger number means the probe scored
    # models the checkpoint trained on.
    assert record["n_faces"] == 18, record["n_faces"]
    assert record["embedding_dim"] == 16
    assert record["data_view"]["split"] == "test"
    assert record["data_view"]["n_faces_dropped_unlabelled"] == 0

    ret = record["retrieval"]
    for key in ("knn_accuracy_at_1", "knn_accuracy_at_5",
                "mean_average_precision", "chance_same_class_rate"):
        assert 0.0 <= ret[key] <= 1.0, (key, ret[key])
    assert -1.0 <= record["silhouette"]["silhouette_cosine"] <= 1.0
    assert record["separation"]["intra_inter_ratio"] > 0.0


def test_probe_record_carries_a_code_fingerprint(probe_mod, trained_arms):
    """Provenance in the same shape `eval_metrics.json` uses.

    Without this the probe's output is a loose number: nothing downstream can
    say which tree produced it, which is the whole reason
    `src/utils/provenance.code_fingerprint` exists.
    """
    from src.utils.provenance import code_fingerprint

    arm = trained_arms["prototype_infonce"]
    record = probe_mod.probe(
        copy.deepcopy(arm["cfg"]), "cfg-in-memory", arm["ckpt"],
        seed=1, device="cpu", ks=(1, 5), verbose=False)

    prov = record["provenance"]
    live = code_fingerprint()
    assert prov.get("code_fingerprint") == live, (
        f"training fingerprint {prov.get('code_fingerprint')!r} != live "
        f"{live!r}; the checkpoint was produced by this very tree in this "
        f"test, so they must agree")
    assert prov["evaluated"]["code_fingerprint"] == live
    assert record["probe_run"]["code_fingerprint"] == live
    assert prov["seed"] == 1


def test_probe_record_is_json_serialisable_and_defines_its_measures(
        probe_mod, trained_arms):
    """The output file must round-trip, and must carry its own definitions."""
    arm = trained_arms["cross_entropy"]
    record = probe_mod.probe(
        copy.deepcopy(arm["cfg"]), "cfg-in-memory", arm["ckpt"],
        seed=1, device="cpu", ks=(1, 5), verbose=False)

    reloaded = json.loads(json.dumps(record))
    assert reloaded["probe"] == "embedding_space"

    measures = reloaded["measures"]
    for key in ("intra_inter_ratio", "silhouette_cosine", "knn_accuracy_at_k",
                "mean_average_precision", "chance_same_class_rate", "SCOPE"):
        assert key in measures, f"no written definition of {key}"
    assert "do NOT show" in measures["SCOPE"], (
        "the scope limit must travel with the numbers: this probe can show "
        "two spaces differ, not that the difference is worth having")


def test_probe_is_deterministic_on_a_real_checkpoint(probe_mod, trained_arms):
    """Two runs of the same checkpoint at the same seed give the same scores."""
    arm = trained_arms["prototype_infonce"]
    runs = [probe_mod.probe(copy.deepcopy(arm["cfg"]), "cfg-in-memory",
                            arm["ckpt"], seed=1, device="cpu", ks=(1, 5),
                            verbose=False)
            for _ in range(2)]

    for block in ("separation", "silhouette", "retrieval"):
        assert (json.dumps(runs[0][block], sort_keys=True)
                == json.dumps(runs[1][block], sort_keys=True)), block


# --------------------------------------------------------------------- #
#  The two confounds a CMCA-vs-CE comparison in THIS probe can hide      #
#                                                                       #
#  Added 26 Sep 2026, before the probe had ever been run. Both are       #
#  measured facts, not worries; the command output that established each #
#  is quoted in paper-content/results-PROBE-embedding.md section 6.      #
# --------------------------------------------------------------------- #

def _noise(n, dim, n_classes=25, seed=0):
    """Labels drawn independently of position: the null by construction."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)), rng.integers(0, n_classes, size=n)


def test_silhouette_null_is_measured_not_asserted_to_be_zero(probe_mod):
    """`chance_silhouette: 0.0` is the ASYMPTOTIC value, not this split's.

    On 1,000 faces in 128-d with labels drawn independently of position the
    cosine silhouette sits near -0.027, not 0.0. A probe that prints 0.0 beside
    it reports a well-separated-as-chance space as slightly WORSE than chance,
    by an amount of the same order as the CMCA-vs-CE gap being hunted. The null
    must be measured on these embeddings, at this width, with this histogram.
    """
    emb, labels = _noise(1000, 128, seed=3)
    rep = probe_mod.embedding_space_report(
        emb, labels, seed=0, silhouette_cap=1000, retrieval_cap=600)
    sil = rep["silhouette"]

    observed = sil["silhouette_cosine"]
    null = sil["null_silhouette_label_shuffled"]["mean"]
    assert observed < -0.01, (
        f"expected the finite-sample floor to be visibly below 0, got "
        f"{observed:+.4f}; the premise of this test is gone")
    assert null == pytest.approx(observed, abs=0.006), (
        f"labels were drawn independently of position, so the measured null "
        f"{null:+.4f} must sit on top of the observed score {observed:+.4f}. "
        f"The hard-coded 0.0 is {abs(observed):.4f} away from it.")


def test_silhouette_null_moves_with_embedding_width(probe_mod):
    """The null is width-dependent, so one constant cannot serve both rows.

    The graph cells are d_model=128 and the ResNet cells embedding_dim=512
    (results/*/seed0/config.resolved.yaml). If the probe reports one chance
    value for both, a cross-row reading of the silhouette is confounded.
    """
    nulls = {}
    for dim in (128, 512):
        emb, labels = _noise(1000, dim, seed=3)
        rep = probe_mod.embedding_space_report(
            emb, labels, seed=0, silhouette_cap=1000, retrieval_cap=600)
        nulls[dim] = rep["silhouette"]["null_silhouette_label_shuffled"]["mean"]

    assert abs(nulls[128] - nulls[512]) > 0.005, (
        f"null at d=128 is {nulls[128]:+.4f} and at d=512 {nulls[512]:+.4f}; "
        f"if these were equal a single constant would do and this control "
        f"would be unnecessary")


def test_class_structure_hidden_in_the_norm_is_reported_not_silently_dropped(
        probe_mod):
    """Every cosine measure here discards ||z||. The CE arm's head does not.

    `PrototypeInfoNCELoss` L2-normalises before it compares
    (src/losses/infonce.py:42-43), so the CMCA arm CAN only encode class in
    direction. `SemanticClassificationModel.forward` scores `w_c . z + b_c`
    (src/models/ce_encoder.py:85, :100), which can also use the LENGTH of z.
    This construction puts the class entirely in the length: perfectly
    separated in Euclidean space, invisible to cosine. Without a control the
    probe would report it as below chance and the paper would read that as
    "the CE arm builds no structure".
    """
    rng = np.random.default_rng(5)
    dim, n_classes, per_class = 128, 25, 40
    labels = np.repeat(np.arange(n_classes), per_class)
    u = rng.normal(size=dim)
    u /= np.linalg.norm(u)
    radii = 10.0 + 10.0 * labels
    emb = radii[:, None] * u[None, :] + 0.05 * rng.normal(size=(len(labels), dim))

    rep = probe_mod.embedding_space_report(
        emb, labels, seed=0, silhouette_cap=1000, retrieval_cap=1000)
    sil = rep["silhouette"]

    assert sil["silhouette_cosine"] < 0.0, (
        f"premise gone: cosine silhouette {sil['silhouette_cosine']:+.4f} was "
        f"supposed to miss a structure that lives only in the norm")
    assert sil["silhouette_euclidean_control"] > 0.8, (
        f"the Euclidean control read {sil['silhouette_euclidean_control']} on "
        f"a space whose classes are perfectly separated by length. Without it "
        f"nothing in the output distinguishes 'no structure' from 'structure "
        f"the cosine measures cannot see'.")
    assert rep["embedding_norm"]["class_eta_squared"] > 0.9, (
        f"class_eta_squared is {rep['embedding_norm']['class_eta_squared']}; "
        f"the class label here explains essentially ALL the variance of ||z||, "
        f"which is the one number that says how much the probe threw away")


def test_norm_carries_nothing_when_the_class_is_in_the_direction(probe_mod):
    """The control must also read LOW, or it flags every space and says nothing."""
    emb, labels = _clustered(n_classes=8, per_class=50, dim=32, seed=2)
    rep = probe_mod.embedding_space_report(
        emb, labels, seed=0, silhouette_cap=400, retrieval_cap=400)
    eta = rep["embedding_norm"]["class_eta_squared"]
    assert eta < 0.2, (
        f"class_eta_squared {eta:.4f} on blobs built around unit centres with "
        f"isotropic noise -- the class is in the direction, not the length, so "
        f"discarding the length costs nothing and the number must say so")


def test_model_type_is_recorded_from_the_key_the_scorer_actually_read(
        probe_mod, trained_arms):
    """`collect_test_embeddings` falls back to `semantic_model`; the record did not.

    `collect_test_embeddings` reads `cfg.get("model", cfg.get("semantic_model",
    {}))`, so a config carrying only `semantic_model:` is scored as a graph
    transformer -- while `probe()` wrote `cfg.get("model", {}).get("type")`
    into the record, i.e. null. A provenance field that says null about a run
    that was scored is worse than no field.
    """
    arm = trained_arms["prototype_infonce"]
    cfg = copy.deepcopy(arm["cfg"])
    cfg["semantic_model"] = cfg.pop("model")

    record = probe_mod.probe(cfg, "cfg-in-memory", arm["ckpt"], seed=1,
                             device="cpu", ks=(1, 5), verbose=False)
    assert record["model_type"] == "semantic_segmentation", (
        f"scored as a graph transformer but recorded model_type="
        f"{record['model_type']!r}")

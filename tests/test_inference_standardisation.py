"""
Tests for src/evaluation/inference.py -- the data view it runs a checkpoint on.

The hole these close
--------------------
All three graph paths in `inference.py` built their dataset like this::

    dataset = PreExtractedBRepDataset(
        json_dir=ds_cfg["json_dir"],
        n_classes=ds_cfg.get("n_classes", 25),
        selected_classes=ds_cfg.get("selected_classes"),
    )

-- no ``attributes``, no ``norm_stats``, no ``split_file``, no bias settings.
Those fell through to the `CMCADataset` class defaults, ``attributes="raw"``
and ``bias_type="adjacency"``. Every shipped graph config trains with
``attributes: standardized``, and ``bias.type`` is the arm the paper's main
table compares. So a model trained on standardized inputs under a similarity
bias was run at inference on raw inputs under an adjacency bias.

Nothing raised and nothing warned, because the norm-stats provenance guard in
`src/data/brep_dataset.py` only runs when ``attributes == "standardized"`` --
and this path never asked for standardized, so the guard never fired.

Measured on the fixture corpus with a model trained to loss 0.17 on
standardized inputs: 44 of 59 faces (74.6%) came back with a DIFFERENT
predicted label, in a prediction file that looked entirely ordinary.

`test_the_two_views_really_do_change_what_the_model_sees` is the teeth: it
reproduces the old construction and shows the divergence is real, so the rest
of this file is known not to be asserting a tautology.

Same failure class, same file, two more of them
-----------------------------------------------
* ``ckpt.get("optimal_threshold", 0.5)`` wrote 0.5 into every sample of a
  unified prediction file under the name of a validation-tuned threshold.
* The label encoder was built from the CHECKPOINT's class count and the
  prototypes indexed by the DATASET's, so a shorter ``selected_classes`` list
  silently ran the argmax over a prefix of the prototypes.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch

from src.data.brep_dataset import PreExtractedBRepDataset
from src.data.collate import collate_fn_brep
from src.evaluation import inference as INF
from src.models.encoders import (BottomLabelEncoder, BRepEncoder, LabelEncoder,
                                 SemanticSegmentationModel)
from src.models.link_predictor import GraphLinkPredictor
from src.utils.checkpoint import load_checkpoint, save_checkpoint

D_MODEL = 32


# --------------------------------------------------------------------- #
#  fixtures                                                             #
# --------------------------------------------------------------------- #

@pytest.fixture
def cfg(corpus):
    """A config in the shape every shipped graph config has."""
    return {
        "experiment_name": "inference_test",
        "dataset": {
            "json_dir": str(corpus["json_dir"]),
            "split_file": str(corpus["split_file"]),
            "n_classes": corpus["n_classes"],
            "attributes": "standardized",
            "norm_stats": str(corpus["norm_stats"]),
            "one_hot_type": False,
        },
        "bias": {"type": "similarity", "lambda_adj": 1.0, "lambda_sim": 1.0},
        "model": {"type": "semantic_segmentation", "d_model": D_MODEL,
                  "n_heads": 4, "n_layers": 1, "dim_feedforward": 64,
                  "dropout": 0.0},
        "label_encoder": {"embedding_dim": D_MODEL},
        "device": "cpu",
    }


def _semantic_parts(n_classes, unified=False):
    torch.manual_seed(3)
    cls = BRepEncoder if unified else SemanticSegmentationModel
    model = cls(d_model=D_MODEL, n_heads=4, n_layers=1, dim_feedforward=64,
                dropout=0.0)
    label_encoder = LabelEncoder(n_classes=n_classes, embedding_dim=D_MODEL)
    return model, label_encoder


@pytest.fixture
def semantic_ckpt(tmp_path, corpus):
    model, label_encoder = _semantic_parts(corpus["n_classes"])
    path = tmp_path / "semantic.pth"
    save_checkpoint(path, model, label_encoder=label_encoder,
                    n_classes=corpus["n_classes"], epoch=7)
    return load_checkpoint(path, device="cpu")


@pytest.fixture
def unified_ckpt(tmp_path, corpus):
    model, label_encoder = _semantic_parts(corpus["n_classes"], unified=True)
    path = tmp_path / "unified.pth"
    save_checkpoint(path, model, label_encoder=label_encoder,
                    bottom_encoder=BottomLabelEncoder(embedding_dim=D_MODEL),
                    n_classes=corpus["n_classes"], epoch=7,
                    optimal_threshold=0.3125)
    return load_checkpoint(path, device="cpu")


# --------------------------------------------------------------------- #
#  1. the bug this file replaces -- proof the rest has teeth             #
# --------------------------------------------------------------------- #

def test_the_two_views_really_do_change_what_the_model_sees(cfg, corpus):
    """The old construction vs the config's: different inputs, no complaint.

    Reproduces `inference.py`'s pre-2026-09-22 dataset call verbatim and shows
    that both the attributes and the bias it hands the network differ from what
    the config asks for -- and that the encoder's output moves as a result.
    Without this, every assertion below could be satisfied by a corpus where
    standardisation happened to be a no-op.
    """
    ds_cfg = cfg["dataset"]
    old = PreExtractedBRepDataset(                    # the bug, verbatim
        json_dir=ds_cfg["json_dir"],
        n_classes=ds_cfg.get("n_classes", 25),
        selected_classes=ds_cfg.get("selected_classes"),
        verbose=False,
    )
    new = PreExtractedBRepDataset(
        verbose=False, **INF._resolve_data_view(cfg, where="t"))

    assert old.attributes == "raw" and old.bias_type == "adjacency", (
        "class defaults moved; this test no longer reproduces the old call")
    assert new.attributes == "standardized" and new.bias_type == "similarity"

    # the inputs themselves
    assert not torch.allclose(old[0]["faces_attr"], new[0]["faces_attr"])
    assert (old[0]["faces_attr"] - new[0]["faces_attr"]).abs().max() > 1.0
    assert not torch.equal(old[0]["bias"], new[0]["bias"])

    # and what the encoder makes of them, with ONE set of weights
    model, _ = _semantic_parts(corpus["n_classes"])
    model.eval()
    outs = []
    for ds in (old, new):
        batch = collate_fn_brep([ds[i] for i in range(3)])
        with torch.no_grad():
            outs.append(model(batch["faces_uv"], batch["faces_attr"],
                              batch["edges_attr"], batch["bias"],
                              face_mask=batch["face_mask"],
                              edge_mask=batch["edge_mask"]))
    assert not torch.allclose(outs[0], outs[1], atol=1e-4), (
        "the two views produced identical embeddings; the probe is toothless")


# --------------------------------------------------------------------- #
#  2. the fix: the config's view is the one that is used                 #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["semantic", "unified", "link"])
def test_every_mode_builds_the_dataset_from_the_config(cfg, semantic_ckpt, mode):
    """All three call sites go through one resolver, which reads the config."""
    dataset, meta = INF._build_brep_dataset(
        cfg, semantic_ckpt, where=f"inference(mode={mode})", verbose=False)
    assert dataset.attributes == "standardized"
    assert dataset.bias_type == "similarity"
    assert meta["attributes"] == "standardized"
    assert meta["bias_type"] == "similarity"
    assert meta["norm_stats"] == cfg["dataset"]["norm_stats"]


def test_the_resolved_view_matches_what_the_trainer_resolves(cfg):
    """Inference must reproduce the trainer's kwargs from the same config.

    If these drift apart, a checkpoint is scored on a view it was never fitted
    on -- which is this whole file.
    """
    ds_cfg, bias_cfg = cfg["dataset"], cfg["bias"]
    from src.utils.class_weights import required_bias_type
    trainer_kwargs = dict(                      # src/training/train_semantic.py
        json_dir=ds_cfg["json_dir"],
        n_classes=ds_cfg.get("n_classes", 25),
        selected_classes=ds_cfg.get("selected_classes"),
        attributes=ds_cfg.get("attributes", "scaled"),
        norm_stats=ds_cfg.get("norm_stats"),
        split_file=ds_cfg.get("split_file"),
        bias_type=required_bias_type(bias_cfg, where="train_semantic"),
        lambda_adj=bias_cfg.get("lambda_adj", 1.0),
        lambda_sim=bias_cfg.get("lambda_sim", 1.0),
        one_hot_type=ds_cfg.get("one_hot_type", False),
    )
    assert INF._resolve_data_view(cfg, where="t") == trainer_kwargs


# --------------------------------------------------------------------- #
#  3. refusals -- each of these inputs raises                            #
# --------------------------------------------------------------------- #

def test_standardized_without_norm_stats_raises(cfg, semantic_ckpt):
    """THE headline case: no silent fall back to the raw attribute view."""
    cfg["dataset"].pop("norm_stats")
    with pytest.raises(ValueError, match="norm_stats"):
        INF._infer_semantic_graph(cfg, semantic_ckpt, "cpu")


def test_the_error_says_what_to_do_about_it(cfg):
    cfg["dataset"].pop("norm_stats")
    with pytest.raises(ValueError) as exc:
        INF._resolve_data_view(cfg, where="t")
    msg = str(exc.value)
    assert "norm_train.json" in msg          # where to point it
    assert "attributes: raw" in msg          # how to opt out on purpose


def test_a_missing_bias_block_raises_rather_than_picking_an_arm(cfg):
    """adjacency vs similarity is the paper's claim, not a default."""
    cfg.pop("bias")
    with pytest.raises(KeyError, match="bias.type"):
        INF._resolve_data_view(cfg, where="t")


def test_forwarding_split_file_arms_the_stale_norm_stats_guard(cfg, corpus,
                                                               tmp_path,
                                                               semantic_ckpt):
    """CHECK 2 of the norm-stats guard could never fire on this path before.

    It compares the split the statistics were measured on against the split in
    use, and it is skipped unless the caller passes `split_file` -- which
    inference.py did not.
    """
    stale = json.loads(corpus["norm_stats"].read_text())
    stale["split_file"] = "datasets/splits/cadsynth_80_20_seed42.json"
    path = tmp_path / "stale_norm.json"
    path.write_text(json.dumps(stale))
    cfg["dataset"]["norm_stats"] = str(path)

    with pytest.raises(ValueError, match="STALE NORM STATS"):
        INF._infer_semantic_graph(cfg, semantic_ckpt, "cpu")


def test_fewer_dataset_classes_than_prototypes_raises(cfg, semantic_ckpt):
    """Was an argmax over a prefix of the prototypes, with normal confidences."""
    cfg["dataset"]["selected_classes"] = [0, 1]    # checkpoint trained with 4
    with pytest.raises(ValueError, match="label encoder"):
        INF._infer_semantic_graph(cfg, semantic_ckpt, "cpu")


def test_unified_refuses_a_checkpoint_with_no_tuned_threshold(cfg, unified_ckpt):
    """0.5 is not a tuned threshold."""
    unified_ckpt.pop("optimal_threshold")
    with pytest.raises(ValueError, match="optimal_threshold"):
        INF._infer_unified(cfg, unified_ckpt, "cpu")


def test_unified_uses_the_checkpoints_own_threshold(cfg, unified_ckpt):
    preds, meta = INF._infer_unified(cfg, unified_ckpt, "cpu")
    assert meta["optimal_threshold"] == pytest.approx(0.3125)
    assert all(s["optimal_threshold"] == pytest.approx(0.3125) for s in preds)


def test_split_without_a_split_file_raises(cfg, semantic_ckpt):
    cfg["dataset"].pop("split_file")
    with pytest.raises(ValueError, match="split_file"):
        INF._infer_semantic_graph(cfg, semantic_ckpt, "cpu", split="test")


# --------------------------------------------------------------------- #
#  4. the checkpoint cross-check                                         #
# --------------------------------------------------------------------- #

def test_a_checkpoint_that_names_its_data_view_is_obeyed(cfg, semantic_ckpt):
    """Nothing writes these keys today. If anything ever does, they are fatal.

    This is the cross-check the task asked for, built so that it starts working
    the day `build_provenance` records the data view -- and so that a future
    agent who adds the field cannot also add a mismatch.
    """
    semantic_ckpt["provenance"] = {"attributes": "raw",
                                   "bias_type": "similarity"}
    with pytest.raises(ValueError, match="trained with 'raw'"):
        INF._infer_semantic_graph(cfg, semantic_ckpt, "cpu")


def test_a_matching_recorded_view_is_accepted_and_reported(cfg, semantic_ckpt):
    semantic_ckpt["provenance"] = {"attributes": "standardized",
                                   "bias_type": "similarity"}
    _, meta = INF._build_brep_dataset(cfg, semantic_ckpt, where="t",
                                      verbose=False)
    check = meta["checkpoint_cross_check"]
    assert check["status"] == "verified_from_checkpoint"
    assert check["data_view_recorded_in_checkpoint"] == ["attributes",
                                                         "bias_type"]


def test_the_same_config_hash_proves_the_view_is_the_training_view(cfg,
                                                                   semantic_ckpt):
    from src.utils.provenance import config_hash
    semantic_ckpt["provenance"] = {"config_hash": config_hash(cfg)}
    _, meta = INF._build_brep_dataset(cfg, semantic_ckpt, where="t",
                                      verbose=False)
    assert meta["checkpoint_cross_check"]["status"] == "verified_by_config_hash"


def test_an_unverifiable_view_says_so_instead_of_passing_quietly(cfg,
                                                                 semantic_ckpt):
    """Today's real case: checkpoints record no data view at all.

    `save_checkpoint` writes model / label_encoder / bottom_encoder / criterion
    / history / n_classes / epoch / optimal_threshold, and `build_provenance`
    adds seed, epoch, config_hash, code_fingerprint, gpu, wall time and the
    selection fields. None of them names the attribute view, the
    norm-stats file or the bias type. So the config is the only statement of
    what the model was trained on, and a prediction file has to say that out
    loud rather than imply it was checked.
    """
    assert "provenance" not in semantic_ckpt
    _, meta = INF._build_brep_dataset(cfg, semantic_ckpt, where="t",
                                      verbose=False)
    assert meta["checkpoint_cross_check"]["status"] == (
        "unverifiable_no_provenance")

    semantic_ckpt["provenance"] = {"config_hash": "0000badc0ffee000"}
    _, meta = INF._build_brep_dataset(cfg, semantic_ckpt, where="t",
                                      verbose=False)
    assert meta["checkpoint_cross_check"]["status"] == (
        "unverified_config_differs")
    assert meta["checkpoint_cross_check"]["data_view_recorded_in_checkpoint"] \
        is None


# --------------------------------------------------------------------- #
#  5. what lands in the prediction file                                  #
# --------------------------------------------------------------------- #

def test_the_prediction_file_records_the_inputs_that_produced_it(cfg, tmp_path,
                                                                 corpus):
    """A prediction file that cannot say which inputs made it is unquotable."""
    model, label_encoder = _semantic_parts(corpus["n_classes"])
    ckpt_path = tmp_path / "sem.pth"
    save_checkpoint(ckpt_path, model, label_encoder=label_encoder,
                    n_classes=corpus["n_classes"], epoch=7)
    out = tmp_path / "preds.json"

    INF.run_inference(cfg, ckpt_path, "semantic", out, split="test")
    payload = json.loads(out.read_text())

    view = payload["data_view"]
    assert view["attributes"] == "standardized"
    assert view["bias_type"] == "similarity"
    assert view["one_hot_type"] is False
    assert view["n_models"] == len(corpus["test_ids"])
    assert "test partition" in view["scored"]
    assert view["checkpoint_cross_check"]["status"].startswith(
        ("verified", "unverified", "unverifiable"))


def test_the_default_scope_admits_it_is_the_whole_corpus(cfg, semantic_ckpt,
                                                         corpus):
    """Every prediction carries a ground_truth, so the file is one line away
    from being read as an accuracy -- over models the checkpoint trained on."""
    _, meta = INF._build_brep_dataset(cfg, semantic_ckpt, where="t",
                                      verbose=False)
    assert meta["n_models"] == 10           # the whole fixture corpus
    assert "FULL corpus" in meta["scored"]
    assert "--split test" in meta["scored"]


def test_split_scores_only_that_partition(cfg, semantic_ckpt, corpus):
    preds, meta = INF._infer_semantic_graph(cfg, semantic_ckpt, "cpu",
                                            split="test")
    assert len(preds) == len(corpus["test_ids"])
    assert [p["model_id"] for p in preds] == list(corpus["test_ids"])


def test_predictions_name_the_model_they_came_from(cfg, semantic_ckpt, corpus):
    """`sample_idx` alone is an index into a directory listing."""
    preds, _ = INF._infer_semantic_graph(cfg, semantic_ckpt, "cpu")
    assert [p["model_id"] for p in preds] == sorted(
        corpus["train_ids"] + corpus["val_ids"] + corpus["test_ids"])


# --------------------------------------------------------------------- #
#  6. link mode -- the frozen encoder's view is the one that matters     #
# --------------------------------------------------------------------- #

def test_link_mode_runs_on_the_configs_view(cfg, tmp_path, corpus):
    """The link head consumes the frozen semantic encoder's embeddings, so raw
    inputs corrupt the link predictions through the same bad embeddings."""
    model, label_encoder = _semantic_parts(corpus["n_classes"])
    sem_path = tmp_path / "sem.pth"
    save_checkpoint(sem_path, model, label_encoder=label_encoder,
                    n_classes=corpus["n_classes"], epoch=7)

    torch.manual_seed(5)
    link_path = tmp_path / "link.pth"
    save_checkpoint(link_path, GraphLinkPredictor(
        d_model=D_MODEL, n_heads=4, n_layers=1, dim_feedforward=64,
        dropout=0.0), epoch=7)
    link_ckpt = load_checkpoint(link_path, device="cpu")

    cfg["semantic_model"] = cfg["model"]
    cfg["link_model"] = {"d_model": D_MODEL, "n_heads": 4, "n_layers": 1,
                         "dim_feedforward": 64, "dropout": 0.0}

    preds, meta = INF._infer_link(cfg, link_ckpt, "cpu",
                                  semantic_checkpoint_path=str(sem_path),
                                  split="test")
    assert meta["attributes"] == "standardized"
    assert meta["bias_type"] == "similarity"
    assert "semantic_checkpoint_cross_check" in meta
    assert len(preds) == len(corpus["test_ids"])
    assert preds[0]["links"]


def test_link_mode_refuses_standardized_without_norm_stats(cfg, tmp_path,
                                                           corpus):
    model, label_encoder = _semantic_parts(corpus["n_classes"])
    sem_path = tmp_path / "sem.pth"
    save_checkpoint(sem_path, model, label_encoder=label_encoder,
                    n_classes=corpus["n_classes"], epoch=7)
    torch.manual_seed(5)
    link_path = tmp_path / "link.pth"
    save_checkpoint(link_path, GraphLinkPredictor(
        d_model=D_MODEL, n_heads=4, n_layers=1, dim_feedforward=64,
        dropout=0.0), epoch=7)
    link_ckpt = load_checkpoint(link_path, device="cpu")

    cfg["semantic_model"] = cfg["model"]
    cfg["link_model"] = dict(cfg["model"])
    cfg["dataset"].pop("norm_stats")

    with pytest.raises(ValueError, match="norm_stats"):
        INF._infer_link(cfg, link_ckpt, "cpu",
                        semantic_checkpoint_path=str(sem_path))


# --------------------------------------------------------------------- #
#  7. no silent tolerance was added while fixing this                    #
# --------------------------------------------------------------------- #

def test_no_except_handler_in_this_module_swallows_a_failure():
    """The rule this whole wave is about, checked against the file itself.

    Not "no try/except" -- wrapping an exception to name the file it came from
    and re-raising is good, and `src/data/brep_dataset.py` does exactly that.
    What is banned is an except that RETURNS: a handler with no `raise` in it
    turns a failure into a default, and a default here is a wrong prediction
    with a normal-looking confidence beside it.
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(INF))
    swallowed = [
        h.lineno for h in ast.walk(tree)
        if isinstance(h, ast.ExceptHandler)
        and not any(isinstance(n, ast.Raise) for n in ast.walk(h))
    ]
    assert swallowed == [], (
        f"except handler(s) at line(s) {swallowed} of inference.py do not "
        f"re-raise. An inference failure that becomes a default becomes a "
        f"prediction file that looks fine and is wrong.")


def test_no_path_in_this_module_defaults_the_attribute_view_to_raw(cfg):
    """`attributes` may default only to what the trainers default to.

    The class default is "raw"; the trainers' is "scaled". If this module ever
    resolves a missing key to "raw", it is back to scoring standardized-trained
    weights on unstandardised inputs.
    """
    bare = copy.deepcopy(cfg)
    bare["dataset"].pop("attributes")
    bare["dataset"].pop("norm_stats")
    assert INF._resolve_data_view(bare, where="t")["attributes"] == "scaled"

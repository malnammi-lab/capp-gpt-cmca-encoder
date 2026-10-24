"""Checkpoint round-trip, and the two-stage path collision that destroyed a
trained Stage 1 model."""

import torch
import yaml
from pathlib import Path

from src.models.encoders import LabelEncoder, SemanticSegmentationModel
from src.models.link_predictor import GraphLinkPredictor
from src.utils.checkpoint import load_checkpoint, save_checkpoint

KW = dict(d_model=16, n_heads=2, n_layers=1, dim_feedforward=16, dropout=0.1)


def test_round_trip_preserves_weights(tmp_path):
    model = SemanticSegmentationModel(**KW)
    le = LabelEncoder(n_classes=4, embedding_dim=16)
    p = tmp_path / "ck.pth"
    save_checkpoint(p, model, label_encoder=le, n_classes=4, epoch=7,
                    history={"loss": [1.0, 0.5]}, optimal_threshold=0.61)

    ck = load_checkpoint(p, device="cpu")
    assert ck["n_classes"] == 4 and ck["epoch"] == 7
    assert ck["optimal_threshold"] == 0.61
    assert ck["history"]["loss"] == [1.0, 0.5]

    restored = SemanticSegmentationModel(**KW)
    restored.load_state_dict(ck["model"])
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)


def test_optional_blocks_are_omitted_not_none(tmp_path):
    p = tmp_path / "ck.pth"
    save_checkpoint(p, SemanticSegmentationModel(**KW), epoch=1)
    ck = load_checkpoint(p, device="cpu")
    assert "label_encoder" not in ck
    assert "bottom_encoder" not in ck


def test_extra_keys_survive(tmp_path):
    p = tmp_path / "ck.pth"
    save_checkpoint(p, SemanticSegmentationModel(**KW), epoch=1,
                    extra={"semantic_checkpoint": "results/x.pth", "seed": 42})
    ck = load_checkpoint(p, device="cpu")
    assert ck["semantic_checkpoint"] == "results/x.pth"
    assert ck["seed"] == 42


def test_missing_checkpoint_raises(tmp_path):
    try:
        load_checkpoint(tmp_path / "nope.pth", device="cpu")
    except FileNotFoundError:
        return
    raise AssertionError("a missing checkpoint must raise FileNotFoundError")


def test_legacy_key_names_are_normalised(tmp_path):
    """Old checkpoints wrote *_state_dict; loaders should not have to guess."""
    p = tmp_path / "legacy.pth"
    m = SemanticSegmentationModel(**KW)
    torch.save({"model_state_dict": m.state_dict(),
                "label_encoder_state_dict": LabelEncoder(4, 16).state_dict()}, p)
    ck = load_checkpoint(p, device="cpu")
    assert "model" in ck and "label_encoder" in ck


# --------------------------------------------------------------------------- #
# REGRESSION: Stage 2 overwrote Stage 1
# --------------------------------------------------------------------------- #

def test_train_link_prefers_the_dedicated_checkpoint_key():
    """`train_link.py` read `paths.checkpoint` -- the same key train_semantic.py
    writes Stage 1 to -- so Stage 2 saved the link model straight over the frozen
    semantic model it had just loaded, and never created paths.link_checkpoint.
    The failure surfaced as
        Unexpected key(s) in state_dict: link_graph_layers...
    and cost a full Stage 1 retrain.
    """
    src = Path("src/training/train_link.py").read_text(encoding="utf-8")
    assert 'paths_cfg.get(\n        "link_checkpoint"' in src or \
           '"link_checkpoint"' in src.split("ckpt_path =")[1][:200], \
        "train_link must resolve link_checkpoint before falling back to checkpoint"


def test_two_stage_configs_separate_the_two_checkpoints():
    """Even with the code fixed, a config that points both stages at one file
    would still destroy Stage 1."""
    for name in ("mfinstseg_two_stage.yaml",):
        p = Path("configs") / name
        if not p.exists():
            continue
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        paths = cfg.get("paths", {})
        if "link_checkpoint" in paths:
            assert paths["link_checkpoint"] != paths.get("checkpoint"), \
                f"{name}: stage 1 and stage 2 write to the same file"


def test_semantic_and_link_state_dicts_are_distinguishable():
    """What makes the overwrite detectable: the two models share no key names,
    so loading one into the other fails loudly rather than silently."""
    sem = set(SemanticSegmentationModel(**KW).state_dict())
    link = set(GraphLinkPredictor(**KW).state_dict())
    assert not (sem & link)
    assert any(k.startswith("link_") for k in link)

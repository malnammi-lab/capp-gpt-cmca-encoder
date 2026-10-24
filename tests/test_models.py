"""Model contracts: forward-pass shapes, and the parity that makes the
encoder-vs-objective ablation interpretable.
"""

import pytest
import torch

from src.models.ce_encoder import SemanticClassificationModel
from src.models.encoders import (BottomLabelEncoder, BRepEncoder, LabelEncoder,
                                 SemanticSegmentationModel)

D_MODEL = 16
KW = dict(d_model=D_MODEL, n_heads=2, n_layers=1, dim_feedforward=16, dropout=0.1)


def _forward(model, batch):
    return model(batch["faces_uv"], batch["faces_attr"], batch["edges_attr"],
                 batch["bias"], face_mask=batch["face_mask"],
                 edge_mask=batch["edge_mask"])


@pytest.mark.parametrize("cls", [BRepEncoder, SemanticSegmentationModel])
def test_encoder_forward_shape(cls, batch):
    model = cls(**KW).eval()
    with torch.no_grad():
        out = _forward(model, batch)
    b, n = batch["faces_uv"].shape[:2]
    assert out.shape == (b, n, D_MODEL)
    assert torch.isfinite(out).all()


def test_semantic_segmentation_is_a_brep_encoder():
    """SemanticSegmentationModel is documented as a thin alias. If that ever
    stops being true, configs naming one and code building the other diverge."""
    assert issubclass(SemanticSegmentationModel, BRepEncoder)


def test_ce_model_emits_logits(batch):
    n_classes = 4
    model = SemanticClassificationModel(n_classes=n_classes, **KW).eval()
    with torch.no_grad():
        out = _forward(model, batch)
    b, n = batch["faces_uv"].shape[:2]
    assert out.shape == (b, n, n_classes)
    assert torch.isfinite(out).all()


def test_ce_encode_returns_embeddings_not_logits(batch):
    model = SemanticClassificationModel(n_classes=4, **KW).eval()
    with torch.no_grad():
        emb = model.encode(batch["faces_uv"], batch["faces_attr"],
                           batch["edges_attr"], batch["bias"],
                           face_mask=batch["face_mask"],
                           edge_mask=batch["edge_mask"])
    assert emb.shape[-1] == D_MODEL


def test_ce_and_cmca_share_an_identical_trunk():
    """The 2x2 ablation is only interpretable if the CE and CMCA arms differ in
    exactly one thing -- the objective. The archived CE notebook built its own
    encoder that silently dropped the edge branch; copying it would have made
    any gap unattributable. This pins the trunks together.
    """
    ce = SemanticClassificationModel(n_classes=4, **KW)
    cmca = SemanticSegmentationModel(**KW)

    ce_trunk = {k[len("encoder."):]: tuple(v.shape)
                for k, v in ce.state_dict().items() if k.startswith("encoder.")}
    cmca_trunk = {k: tuple(v.shape) for k, v in cmca.state_dict().items()}
    assert ce_trunk == cmca_trunk, "CE and CMCA encoders have diverged"


def test_ce_head_is_the_only_extra_parameter_block():
    ce = SemanticClassificationModel(n_classes=4, **KW)
    extra = {k for k in ce.state_dict() if not k.startswith("encoder.")}
    assert extra == {"class_head.weight", "class_head.bias"}


def test_edge_features_actually_reach_the_encoder(batch):
    """The archived CE model constructed an edge branch and never called it.
    Perturbing edge attributes must change the output; if it does not, the edge
    branch is dead code again."""
    model = BRepEncoder(**KW).eval()
    with torch.no_grad():
        a = _forward(model, batch)
        perturbed = dict(batch)
        perturbed["edges_attr"] = batch["edges_attr"] + 5.0
        b = _forward(model, perturbed)
    assert not torch.allclose(a, b), "edge attributes do not influence the output"


def test_bias_actually_reaches_the_attention(batch):
    """Same argument for the graph bias: if the attention ignored it, the whole
    topology ablation would be measuring nothing."""
    model = BRepEncoder(**KW).eval()
    with torch.no_grad():
        a = _forward(model, batch)
        perturbed = dict(batch)
        perturbed["bias"] = batch["bias"] * 0.0
        b = _forward(model, perturbed)
    assert not torch.allclose(a, b), "the attention bias has no effect"


def test_label_encoder_shapes():
    le = LabelEncoder(n_classes=4, embedding_dim=D_MODEL)
    out = le(torch.arange(4))
    assert out.shape == (4, D_MODEL)
    assert torch.isfinite(out).all()


def test_bottom_label_encoder_is_binary():
    be = BottomLabelEncoder(embedding_dim=D_MODEL)
    out = be(torch.arange(2))
    assert out.shape == (2, D_MODEL)

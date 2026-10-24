"""Losses: finite, and they actually descend on a problem with a known answer."""

import torch
import torch.optim as optim

from src.losses.bottom_contrastive import BottomContrastiveLoss
from src.losses.infonce import PrototypeInfoNCELoss
from src.losses.instance_loss import HardFocusedInstanceLoss
from src.models.encoders import BottomLabelEncoder, LabelEncoder

D = 16
N_CLASSES = 4


def _separable_problem(seed=0, n_per_class=8):
    """Embeddings that are already clustered by class -- a loss that cannot
    descend on this is broken, not merely untrained."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(N_CLASSES, D, generator=g) * 3.0
    labels = torch.arange(N_CLASSES).repeat_interleave(n_per_class)
    emb = centres[labels] + 0.1 * torch.randn(len(labels), D, generator=g)
    return emb, labels


def test_infonce_is_finite():
    emb, labels = _separable_problem()
    le = LabelEncoder(n_classes=N_CLASSES, embedding_dim=D)
    loss = PrototypeInfoNCELoss()(emb, labels, le, N_CLASSES)
    assert torch.isfinite(loss) and loss.item() > 0


def test_infonce_decreases_on_a_separable_problem():
    emb, labels = _separable_problem()
    le = LabelEncoder(n_classes=N_CLASSES, embedding_dim=D)
    crit = PrototypeInfoNCELoss()
    emb = emb.clone().requires_grad_(True)
    opt = optim.Adam([emb] + list(le.parameters()) + list(crit.parameters()), lr=0.05)

    first = None
    for step in range(60):
        opt.zero_grad()
        loss = crit(emb, labels, le, N_CLASSES)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    assert loss.item() < first, f"loss did not descend: {first:.4f} -> {loss.item():.4f}"


def test_infonce_temperature_is_learnable():
    crit = PrototypeInfoNCELoss(temperature=0.07)
    assert any(p.requires_grad for p in crit.parameters())


def test_infonce_class_weights_change_the_value():
    emb, labels = _separable_problem()
    le = LabelEncoder(n_classes=N_CLASSES, embedding_dim=D)
    torch.manual_seed(0)
    plain = PrototypeInfoNCELoss()(emb, labels, le, N_CLASSES)
    w = torch.tensor([5.0, 1.0, 1.0, 1.0])
    weighted = PrototypeInfoNCELoss(class_weights=w)(emb, labels, le, N_CLASSES)
    assert not torch.isclose(plain, weighted)


def test_bottom_contrastive_is_finite():
    emb, _ = _separable_problem()
    bot = torch.randint(0, 2, (emb.shape[0],))
    be = BottomLabelEncoder(embedding_dim=D)
    loss = BottomContrastiveLoss()(emb, bot, be)
    assert torch.isfinite(loss)


def test_instance_loss_is_finite_and_masked():
    """Instance loss consumes a padded batch; padded faces must not contribute."""
    b, n = 2, 6
    emb = torch.randn(b, n, D)
    inst = torch.zeros(b, n, n)
    inst[:, :3, :3] = 1.0
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, :4] = True
    loss = HardFocusedInstanceLoss()(emb, inst, mask)
    assert torch.isfinite(loss)


def test_instance_loss_ignores_padded_faces():
    """Two real instances, then the same batch with garbage padding appended.
    The loss must not move.

    (An earlier version of this test used an identity affinity matrix, which has
    no positive pairs at all -- the loss was 0 both times and the test passed
    without exercising the mask.)
    """
    b, n_real = 1, 6
    torch.manual_seed(0)
    emb_real = torch.randn(b, n_real, D)
    inst_real = torch.zeros(b, n_real, n_real)
    inst_real[:, :3, :3] = 1.0          # instance A
    inst_real[:, 3:, 3:] = 1.0          # instance B
    mask_real = torch.ones(b, n_real, dtype=torch.bool)

    crit = HardFocusedInstanceLoss()
    a = crit(emb_real, inst_real, mask_real)
    assert a.item() > 0, "fixture must produce a non-trivial loss to be a real test"

    pad = 4
    emb_pad = torch.cat([emb_real, torch.randn(b, pad, D) * 100], dim=1)
    inst_pad = torch.zeros(b, n_real + pad, n_real + pad)
    inst_pad[:, :n_real, :n_real] = inst_real
    mask_pad = torch.zeros(b, n_real + pad, dtype=torch.bool)
    mask_pad[:, :n_real] = True
    c = crit(emb_pad, inst_pad, mask_pad)

    assert torch.allclose(a, c, atol=1e-5), \
        "padded faces changed the loss -- the mask is not being honoured"


def test_instance_loss_returns_a_tensor_with_no_positive_pairs():
    """REGRESSION. `loss` was initialised to the Python float 0.0, so when no
    batch item had a positive pair the function returned a float and
    `loss.backward()` raised AttributeError. It is now a graph-attached zero:
    backward runs and contributes exactly zero gradient."""
    crit = HardFocusedInstanceLoss()
    emb = torch.randn(1, 4, D, requires_grad=True)
    inst = torch.eye(4).unsqueeze(0)            # every face its own instance
    mask = torch.ones(1, 4, dtype=torch.bool)

    out = crit(emb, inst, mask)
    assert isinstance(out, torch.Tensor), "must be a Tensor, not a float"
    assert out.item() == 0.0
    out.backward()
    assert emb.grad is not None and emb.grad.abs().sum() == 0.0

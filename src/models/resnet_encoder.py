"""
ResNet-18 based encoder and cross-modal alignment model.

Separate file because ResNet-18 is a fundamentally different architecture
from the Graph Transformer family (different backbone, different embedding
dim, different dataset class).  Shares almost no code with the Graph
Transformer encoders in ``encoders.py``.

Classes:
    UVGridEncoder: Modified ResNet-18 for 7-channel UV-grid face inputs.
    CrossModalAlignmentModel: Wraps UVGridEncoder + LabelEncoder +
        PrototypeInfoNCELoss for contrastive cross-modal alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .encoders import LabelEncoder
from ..losses.infonce import PrototypeInfoNCELoss


class UVGridEncoder(nn.Module):
    """CNN encoder using a modified ResNet-18 backbone for UV-grid faces.

    Modifications from standard ResNet-18:
        * First convolution accepts 7 input channels (x, y, z, nx, ny, nz,
          validity) instead of 3.
        * BatchNorm after conv1 is re-initialised for 64 channels.
        * ``maxpool`` is replaced with :class:`~torch.nn.Identity` to
          preserve spatial resolution on small grids.
        * Final fully-connected layer is removed; a custom projection MLP
          head maps the 512-d backbone output to ``embedding_dim``.

    Args:
        embedding_dim: Dimensionality of the output face embedding
            (default 512).
    """

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim

        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(
            7, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        base.bn1 = nn.BatchNorm2d(64)
        base.maxpool = nn.Identity()

        # Remove the final FC layer from ResNet-18
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.proj_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, uv_grid):
        """Encode a batch of 7-channel UV-grid face images.

        Args:
            uv_grid: (B, 7, H, W) tensor of UV-grid face representations.

        Returns:
            Face embeddings of shape (B, embedding_dim).
        """
        x = self.backbone(uv_grid)   # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 512)
        embeddings = self.proj_head(x)
        return embeddings


class CrossModalAlignmentModel(nn.Module):
    """Cross-modal contrastive alignment model (ResNet baseline).

    Combines a :class:`UVGridEncoder` geometry tower with a
    :class:`~src.models.encoders.LabelEncoder` label tower, trained via
    :class:`~src.losses.infonce.PrototypeInfoNCELoss`.

    This model is designed for the flat, per-face UV-grid setting (no graph
    structure).  Each face is treated independently.

    Args:
        n_classes: Total number of semantic classes.
        embedding_dim: Shared embedding dimension for both towers
            (default 512).
        temperature: Initial temperature for the InfoNCE loss
            (default 0.07).
        class_weights: Optional (n_classes,) tensor of per-class weights
            for the contrastive loss.  ``None`` means uniform weighting.
    """

    def __init__(
        self,
        n_classes,
        embedding_dim=512,
        temperature=0.07,
        class_weights=None,
    ):
        super().__init__()
        self.n_classes = n_classes

        # Geometry tower
        self.uv_encoder = UVGridEncoder(embedding_dim=embedding_dim)

        # Label tower (reuses LabelEncoder from encoders.py)
        self.label_encoder = LabelEncoder(
            n_classes, embedding_dim=embedding_dim
        )

        # Contrastive loss
        self.contrastive_loss = PrototypeInfoNCELoss(
            temperature=temperature, class_weights=class_weights
        )

    def forward(self, uv_grids, labels, return_embeddings=False):
        """Forward pass: encode UV grids and compute contrastive loss.

        Args:
            uv_grids: (B, 7, H, W) batch of UV-grid face tensors.
            labels: (B,) integer class labels.
            return_embeddings: If ``True``, return ``(uv_embeddings,
                label_embeddings)`` instead of the scalar loss.

        Returns:
            If ``return_embeddings`` is ``False`` (default): scalar loss.
            If ``return_embeddings`` is ``True``: tuple of
            ``(uv_embeddings, label_embeddings)``.
        """
        uv_embeddings = self.uv_encoder(uv_grids)  # (B, embedding_dim)

        if return_embeddings:
            label_embeddings = self.label_encoder(labels)
            return uv_embeddings, label_embeddings

        loss = self.contrastive_loss(
            uv_embeddings, labels, self.label_encoder, self.n_classes
        )
        return loss

    def predict_label(self, uv_grid, all_label_embeddings, label_map=None):
        """Predict the semantic label for a single UV-grid face.

        Uses cosine similarity between the face embedding and precomputed
        label embeddings to find the nearest class.

        Args:
            uv_grid: (7, H, W) or (1, 7, H, W) UV-grid tensor.
            all_label_embeddings: (n_classes, embedding_dim) precomputed
                normalised label embeddings.
            label_map: Optional dict mapping class indices to label values.
                If ``None``, the raw class index is returned.

        Returns:
            Tuple of ``(predicted_label, similarity_scores)`` where
            ``predicted_label`` is an int and ``similarity_scores`` is a
            numpy array of shape ``(n_classes,)``.
        """
        if uv_grid.dim() == 3:
            uv_grid = uv_grid.unsqueeze(0)  # (1, 7, H, W)

        with torch.no_grad():
            uv_emb = self.uv_encoder(uv_grid)        # (1, embedding_dim)
            uv_emb = F.normalize(uv_emb, dim=1)

            all_label_emb = F.normalize(all_label_embeddings, dim=1)
            similarities = torch.matmul(
                uv_emb, all_label_emb.T
            ).squeeze()                                # (n_classes,)

            predicted_idx = torch.argmax(similarities).item()
            predicted_label = (
                label_map[predicted_idx] if label_map else predicted_idx
            )

        return predicted_label, similarities.cpu().numpy()


class ResNetClassificationModel(nn.Module):
    """ResNet-18 UV tower with a linear classification head.

    The cross-entropy counterpart of :class:`CrossModalAlignmentModel`, and the
    fourth cell of the encoder-vs-objective ablation (task 2.2):

        {ResNet-18, Graph Transformer} x {CE head, CMCA}

    Note what the existing baseline actually is. `CrossModalAlignmentModel` pairs
    this same UV tower with a `LabelEncoder` and trains it with
    `PrototypeInfoNCELoss` -- it is ResNet **+ CMCA**, not ResNet + cross-entropy.
    The task list assumed the published ResNet numbers were a plain classifier;
    they were not. So the 2x2 was missing both CE cells, and this supplies the
    second of them.

    As with :class:`~src.models.ce_encoder.SemanticClassificationModel`, the
    geometry tower is *reused*, not reimplemented: the CE and CMCA cells must
    differ in exactly one thing -- the objective -- or a gap between them is
    unattributable.

    Args:
        n_classes: Number of face classes.
        embedding_dim: Width of the UV tower's output, and the classifier's
            input dimension.
        dropout: Applied to the embedding before the head.

    Returns from ``forward``:
        The scalar cross-entropy loss, matching CrossModalAlignmentModel's
        convention that the model owns its loss, so the trainer's loop needs no
        restructuring. With ``return_embeddings=True`` it returns
        ``(uv_embeddings, None)`` -- the ``None`` stands where the label-tower
        embeddings would be, and there is no label tower here.
    """

    def __init__(self, n_classes, embedding_dim=512, dropout=0.3,
                 class_weights=None):
        super().__init__()
        self.n_classes = n_classes
        self.uv_encoder = UVGridEncoder(embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.class_head = nn.Linear(embedding_dim, n_classes)
        self.register_buffer(
            "_class_weights",
            class_weights if class_weights is not None else torch.empty(0),
        )

    def _weights(self):
        w = self._class_weights
        return w if w.numel() else None

    def forward(self, uv_grids, labels, return_embeddings=False):
        uv_embeddings = self.uv_encoder(uv_grids)
        if return_embeddings:
            return uv_embeddings, None
        logits = self.class_head(self.dropout(uv_embeddings))
        return F.cross_entropy(logits, labels, weight=self._weights())

    def predict(self, uv_grids):
        """Class predictions for a batch of UV grids."""
        return self.class_head(self.uv_encoder(uv_grids)).argmax(dim=1)

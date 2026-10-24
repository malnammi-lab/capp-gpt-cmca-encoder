"""
Graph Transformer encoder family and label encoders.

Consolidates the Graph Transformer geometry tower and label projection
modules into a single file:

- BRepEncoder: full Graph Transformer geometry tower (face + edge processing)
- SemanticSegmentationModel: thin wrapper around BRepEncoder for the
  semantic contrastive task (returns face embeddings, no classification head)
- LabelEncoder: projects one-hot semantic labels to shared embedding space
- BottomLabelEncoder: projects binary bottom-face labels to embedding space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    AttributeEncoder,
    Edge1DCNN,
    EdgeEncoder,
    GraphTransformerLayer,
    SurfaceCNN,
)


class BRepEncoder(nn.Module):
    """Graph Transformer geometry tower for B-Rep models.

    Combines per-face processing (SurfaceCNN + AttributeEncoder), per-edge
    processing (Edge1DCNN + EdgeEncoder), projection layers, and a stack of
    GraphTransformerLayers to produce contextualised face embeddings.

    Args:
        d_model: Dimension of the face embeddings throughout the transformer
            stack (default 128).
        n_heads: Number of attention heads in each GraphTransformerLayer.
        n_layers: Number of stacked GraphTransformerLayers.
        dim_feedforward: Hidden size of the feed-forward network inside each
            transformer layer.
        dropout: Dropout probability used in transformer layers.
    """

    def __init__(
        self,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Face processing
        self.surface_cnn = SurfaceCNN(in_channels=7, out_dim=128)
        self.face_attr_encoder = AttributeEncoder(in_dim=4, hidden=64, out_dim=64)

        # Edge processing
        self.edge_1dcnn = Edge1DCNN(in_dim=4, out_dim=32)
        self.edge_mlp_encoder = EdgeEncoder(in_dim=4, hidden=64, out_dim=64)

        # Projection layers
        self.face_proj = nn.Linear(192, d_model)   # 128 (CNN) + 64 (attr)
        self.edge_proj = nn.Linear(96, d_model)     # 32 (1D CNN) + 64 (MLP)

        # Graph transformer stack
        self.graph_layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, faces_uv, faces_attr, edges_attr, bias,
                face_mask=None, edge_mask=None):
        """Encode a batch of B-Rep graphs into per-face embeddings.

        Args:
            faces_uv: (B, N_faces, 7, H, W) UV-grid tensor for each face.
            faces_attr: (B, N_faces, 4) face attribute vectors.
            edges_attr: (B, N_edges, 4) edge attribute vectors.
            bias: (B, N_faces, N_faces) attention bias matrix (e.g. from
                adjacency).
            face_mask: (B, N_faces) boolean mask -- True for real faces,
                False for padded positions.
            edge_mask: (B, N_edges) boolean mask -- True for real edges,
                False for padded positions.

        Returns:
            Face embeddings of shape (B, N_faces, d_model).
        """
        # --- Face branch ---
        geom_feats = self.surface_cnn(faces_uv)             # (B, N_faces, 128)
        face_attr_feats = self.face_attr_encoder(faces_attr) # (B, N_faces, 64)
        face_feats = torch.cat([geom_feats, face_attr_feats], dim=2)  # (B, N_faces, 192)

        # --- Edge branch ---
        edge_cnn_feats = self.edge_1dcnn(edges_attr, edge_mask)       # (B, 32)
        edge_mlp_feats = self.edge_mlp_encoder(edges_attr, edge_mask) # (B, 64)
        edge_feats = torch.cat([edge_cnn_feats, edge_mlp_feats], dim=1)  # (B, 96)

        # --- Project to d_model ---
        h = self.face_proj(face_feats)           # (B, N_faces, d_model)
        edge_proj = self.edge_proj(edge_feats)   # (B, d_model)
        h = h + edge_proj.unsqueeze(1)           # broadcast edge info to each face

        # --- Graph transformer layers ---
        for layer in self.graph_layers:
            h = layer(h, bias=bias, mask=face_mask)

        return h  # (B, N_faces, d_model)


class SemanticSegmentationModel(BRepEncoder):
    """BRepEncoder configured for Stage-1 semantic contrastive learning.

    This is a thin alias for :class:`BRepEncoder`. It exists so that training
    scripts can refer to the semantic-segmentation encoder by a distinct name,
    making experiment configs self-documenting. The model returns per-face
    embeddings suitable for contrastive alignment with a
    :class:`LabelEncoder`; it has **no** classification head.

    All constructor arguments are forwarded to :class:`BRepEncoder`.
    """

    pass


class LabelEncoder(nn.Module):
    """Project one-hot semantic labels into a shared embedding space.

    Used as the "label tower" in contrastive alignment: each class label is
    converted to a one-hot vector and linearly projected to the same
    dimensionality as the geometry embeddings produced by
    :class:`BRepEncoder`.

    Args:
        n_classes: Total number of semantic classes.
        embedding_dim: Output embedding dimension (should match the geometry
            tower's ``d_model``).
    """

    def __init__(self, n_classes, embedding_dim=128):
        super().__init__()
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.proj = nn.Linear(n_classes, embedding_dim)

    def forward(self, labels):
        """
        Args:
            labels: (B,) or (B, N) integer class labels.

        Returns:
            Label embeddings with the same leading dimensions as *labels*
            and a trailing dimension of ``embedding_dim``.
        """
        one_hot = F.one_hot(labels, num_classes=self.n_classes).float()
        return self.proj(one_hot)


class BottomLabelEncoder(nn.Module):
    """Project binary bottom-face labels into a shared embedding space.

    Analogous to :class:`LabelEncoder` but specialised for the binary
    "is bottom face" label used by the unified model.

    Args:
        embedding_dim: Output embedding dimension (should match the geometry
            tower's ``d_model``).
    """

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proj = nn.Linear(2, embedding_dim)

    def forward(self, labels):
        """
        Args:
            labels: (B,) or (B, N) binary integer labels (0 or 1).

        Returns:
            Label embeddings with the same leading dimensions as *labels*
            and a trailing dimension of ``embedding_dim``.
        """
        one_hot = F.one_hot(labels, num_classes=2).float()
        return self.proj(one_hot)

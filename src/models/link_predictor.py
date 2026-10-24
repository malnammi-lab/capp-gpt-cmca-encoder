"""
Stage 2 link predictor for instance segmentation.

Takes frozen face embeddings from a trained semantic model (Stage 1) and
predicts whether each pair of faces belongs to the same instance.  The
predictor refines embeddings through additional Graph Transformer layers
before feeding pairwise concatenations into an MLP classifier.
"""

import torch
import torch.nn as nn

from .components import GraphTransformerLayer


class GraphLinkPredictor(nn.Module):
    """Stage 2: Link Prediction for Instance Segmentation.

    Process:
        1. Reconstruct graph with node features (frozen face embeddings from
           Stage 1) and edges (topology from the bias / adjacency matrix).
        2. Refine embeddings through a stack of Graph Transformer layers.
        3. For every face pair (Fi, Fj), predict a binary link using an MLP.

    Args:
        d_model: Dimension of the input face embeddings.
        n_heads: Number of attention heads in each Graph Transformer layer.
        n_layers: Number of Graph Transformer layers.
        dim_feedforward: Hidden size of the feed-forward network inside each
            Graph Transformer layer.
        dropout: Dropout probability used in the transformer and the
            link classifier MLP.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.link_graph_layers = nn.ModuleList(
            [
                GraphTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )

        self.link_classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),  # binary: 0 = no link, 1 = link
        )

    def forward(self, face_embeddings, bias=None, mask=None):
        """Predict pairwise links between faces.

        Args:
            face_embeddings: (B, N, D) face embeddings from the frozen
                semantic model (Stage 1).
            bias: (B, N, N) adjacency / attention bias matrix encoding
                the B-Rep topology.
            mask: (B, N) boolean mask — ``True`` for real faces, ``False``
                for padded positions.

        Returns:
            link_logits: (B, N, N, 2) logits for every ordered pair where
                channel 0 → no link, channel 1 → same-instance link.
        """
        B, N, D = face_embeddings.shape

        # Refine embeddings through Graph Transformer layers
        h = face_embeddings
        for layer in self.link_graph_layers:
            h = layer(h, bias=bias, mask=mask)

        # Create all pairwise combinations
        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)

        # Concatenate pairs: (B, N, N, 2*D)
        pair_features = torch.cat([h_i, h_j], dim=-1)

        # Predict link for each pair
        link_logits = self.link_classifier(pair_features)

        return link_logits

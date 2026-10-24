"""
Cross-entropy classification head for the Graph Transformer encoder.

This is the CE arm of the encoder-vs-objective ablation (task 2.1 / 2.2), which
answers the question a reviewer will certainly ask:

    is the gain from the graph encoder, or from the contrastive objective?

Without this arm the question cannot be answered, because every model in the
project so far -- graph and ResNet alike -- is trained contrastively.

This class composes `BRepEncoder` -- the same one the CMCA arm uses, with the
same edge branch, the same masking, the same graph bias -- and adds only a
linear classification head on top of its per-face embeddings. The delta
between the two arms is then one loss function and one head.
"""

import torch.nn as nn

from .encoders import BRepEncoder


class SemanticClassificationModel(nn.Module):
    """Graph Transformer encoder with a linear classification head.

    The cross-entropy counterpart of
    :class:`~src.models.encoders.SemanticSegmentationModel`. Same encoder, same
    inputs, same output embeddings -- but instead of aligning those embeddings
    against a :class:`~src.models.encoders.LabelEncoder` tower, it projects them
    straight onto class logits and trains with ``F.cross_entropy``.

    There is no label tower, so this model cannot be extended to unseen classes
    without retraining the head -- which is precisely the property the CMCA arm
    is claimed to provide, and precisely what this ablation is meant to price.

    Args:
        n_classes: Number of semantic classes (size of the output layer).
        d_model: Encoder width; also the classifier's input dimension.
        n_heads: Attention heads per graph layer.
        n_layers: Number of graph transformer layers.
        dim_feedforward: Feed-forward width inside each graph layer.
        dropout: Dropout probability, applied in the encoder and before the head.

    Shape:
        Input matches :class:`BRepEncoder.forward`.
        Output: ``(B, N_faces, n_classes)`` logits.
    """

    def __init__(self, n_classes, d_model=128, n_heads=8, n_layers=4,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        # The identical encoder the CMCA arm trains. Not a reimplementation --
        # if this line ever diverges from what train_semantic builds for the
        # contrastive arm, the ablation stops being controlled.
        self.encoder = BRepEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.class_head = nn.Linear(d_model, n_classes)

    def forward(self, faces_uv, faces_attr, edges_attr, bias,
                face_mask=None, edge_mask=None):
        """Encode a batch and return per-face class logits.

        Returns:
            Logits of shape ``(B, N_faces, n_classes)``. Padded face positions
            carry meaningless values; the caller masks them before computing the
            loss, exactly as the contrastive arm gathers only valid faces.
        """
        embeddings = self.encoder(
            faces_uv, faces_attr, edges_attr, bias,
            face_mask=face_mask, edge_mask=edge_mask,
        )
        return self.class_head(self.dropout(embeddings))

    def encode(self, faces_uv, faces_attr, edges_attr, bias,
               face_mask=None, edge_mask=None):
        """Per-face embeddings without the classification head.

        Lets a CE-trained encoder be probed with the same prototype-similarity
        evaluation used for the contrastive arm, so the two can be compared in
        embedding space and not only on accuracy.
        """
        return self.encoder(
            faces_uv, faces_attr, edges_attr, bias,
            face_mask=face_mask, edge_mask=edge_mask,
        )

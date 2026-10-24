"""Prototype-based InfoNCE loss for semantic segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeInfoNCELoss(nn.Module):
    """Prototype-based InfoNCE loss reformulated as cross-entropy.

    Pulls face embeddings toward their corresponding class prototype
    (produced by a LabelEncoder) and pushes them away from other prototypes.
    Temperature is learnable, stored in log-space.

    Args:
        temperature: Initial temperature value (default: 0.07).
        class_weights: Optional 1-D tensor of per-class weights passed to
            ``F.cross_entropy``.  When *None*, all classes are weighted equally.
    """

    def __init__(self, temperature: float = 0.07, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.class_weights = class_weights

    def forward(self, uv_embeddings: torch.Tensor, labels: torch.Tensor,
                label_encoder: nn.Module, n_classes: int) -> torch.Tensor:
        """Compute prototype InfoNCE loss.

        Args:
            uv_embeddings: Face embeddings of shape ``(B, D)``.
            labels: Ground-truth class indices of shape ``(B,)``.
            label_encoder: Module that maps class indices to prototype
                embeddings (e.g. ``LabelEncoder``).
            n_classes: Total number of classes.

        Returns:
            Scalar loss tensor.
        """
        all_labels = torch.arange(n_classes, device=labels.device)
        prototypes = F.normalize(label_encoder(all_labels), dim=1)
        uv_emb = F.normalize(uv_embeddings, dim=1)

        logits = torch.matmul(uv_emb, prototypes.T) / torch.exp(self.temperature)

        if self.class_weights is not None:
            weights = self.class_weights.to(labels.device)
            return F.cross_entropy(logits, labels, weight=weights)
        return F.cross_entropy(logits, labels)

"""Binary contrastive loss for bottom-face prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottomContrastiveLoss(nn.Module):
    """Contrastive loss for binary bottom-face classification.

    Uses the same prototype-based InfoNCE formulation as
    :class:`PrototypeInfoNCELoss` but specialised to two classes
    (bottom / non-bottom).  Temperature is learnable and stored in log-space.

    Args:
        temperature: Initial temperature value (default: 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, uv_embeddings: torch.Tensor, bottom_labels: torch.Tensor,
                bottom_encoder: nn.Module) -> torch.Tensor:
        """Compute bottom-face contrastive loss.

        Args:
            uv_embeddings: Face embeddings of shape ``(B, D)``.
            bottom_labels: Binary ground-truth labels of shape ``(B,)``
                (0 = non-bottom, 1 = bottom).
            bottom_encoder: Module mapping binary labels to prototype
                embeddings (e.g. ``BottomLabelEncoder``).

        Returns:
            Scalar loss tensor.
        """
        all_labels = torch.arange(2, device=bottom_labels.device)
        prototypes = F.normalize(bottom_encoder(all_labels), dim=1)
        uv_emb = F.normalize(uv_embeddings, dim=1)

        logits = torch.matmul(uv_emb, prototypes.T) / torch.exp(self.temperature)
        return F.cross_entropy(logits, bottom_labels)

"""Hard-mining contrastive loss for instance segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardFocusedInstanceLoss(nn.Module):
    """Hard-mining contrastive loss for instance segmentation.

    For every pair of face embeddings ``(F_i, F_j)``:

    * Compute cosine similarity scaled by a learnable temperature.
    * Ground truth is given by an instance affinity matrix (1 = same instance,
      0 = different instance).
    * Focus training on *hard positives* (same instance but low similarity) and
      *hard negatives* (different instance but high similarity).

    Args:
        temperature: Initial temperature value (default: 0.07).
        hard_mining_ratio: Fraction of positive/negative pairs to keep as
            "hard" examples (default: 0.3).
    """

    def __init__(self, temperature: float = 0.07, hard_mining_ratio: float = 0.3):
        super().__init__()
        self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.hard_mining_ratio = hard_mining_ratio

    def forward(self, embeddings: torch.Tensor, target_matrix: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Compute hard-focused instance loss.

        Args:
            embeddings: Face embeddings of shape ``(B, N, D)``.
            target_matrix: Ground-truth instance affinity matrix of shape
                ``(B, N, N)`` with values 1 (same instance) or 0 (different).
            mask: Valid-face boolean mask of shape ``(B, N)``.

        Returns:
            Scalar loss tensor averaged over the batch.
        """
        # `num_batches` is the batch size, not the number of contributing
        # items, so items skipped above still divide.
        # `embeddings.sum() * 0` rather than `torch.zeros(())`: numerically the
        # same zero, but attached to the autograd graph, so .backward() works and
        # contributes exactly zero gradient. A detached zero raises
        #     RuntimeError: element 0 of tensors does not require grad
        loss = embeddings.sum() * 0.0
        num_batches = embeddings.shape[0]

        for b in range(num_batches):
            valid = mask[b]
            if valid.sum() < 2:
                continue

            valid_idx = torch.nonzero(valid).squeeze(1)
            emb_valid = F.normalize(embeddings[b][valid_idx], p=2, dim=-1)  # (V, D)

            # Pairwise similarity scaled by temperature
            sim_matrix = torch.matmul(emb_valid, emb_valid.T) / torch.exp(self.temperature)

            # Ground-truth for valid pairs
            target = target_matrix[b][valid_idx[:, None], valid_idx]  # (V, V)

            # Exclude self-pairs (diagonal)
            eye = torch.eye(sim_matrix.shape[0], device=sim_matrix.device).bool()
            sim_no_diag = sim_matrix[~eye]
            target_no_diag = target[~eye]

            pos_mask = (target_no_diag == 1)
            neg_mask = (target_no_diag == 0)

            if pos_mask.sum() == 0:
                continue

            pos_sims = sim_no_diag[pos_mask]
            neg_sims = sim_no_diag[neg_mask]

            num_hard = max(1, int(self.hard_mining_ratio * len(pos_sims)))

            # Hard positives: lowest similarities (should be high)
            if len(pos_sims) > 0:
                hard_pos_sims, _ = torch.topk(
                    pos_sims, min(num_hard, len(pos_sims)), largest=False
                )
            else:
                hard_pos_sims = pos_sims

            # Hard negatives: highest similarities (should be low)
            if len(neg_sims) > 0:
                hard_neg_sims, _ = torch.topk(
                    neg_sims, min(num_hard, len(neg_sims)), largest=True
                )
            else:
                hard_neg_sims = neg_sims

            # InfoNCE-style loss on hard examples
            if len(hard_pos_sims) > 0 and len(hard_neg_sims) > 0:
                pos_exp = torch.exp(hard_pos_sims).sum()
                all_neg_exp = torch.exp(neg_sims).sum()
                loss_b = -torch.log(pos_exp / (pos_exp + all_neg_exp + 1e-8))
                loss += loss_b

        if num_batches > 0:
            return loss / num_batches
        return torch.tensor(0.0, device=embeddings.device)

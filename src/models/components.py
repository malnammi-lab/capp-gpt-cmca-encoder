"""
Shared building blocks used by multiple model variants.

Contains the core neural network components for processing B-Rep geometry:
- SurfaceCNN: 2D convolution on UV-grid face representations
- AttributeEncoder: MLP for face attribute vectors
- EdgeEncoder: MLP for edge attributes with masked mean pooling
- Edge1DCNN: 1D convolution for local edge feature dependencies
- GraphTransformerLayer: multi-head self-attention with optional bias matrix
"""

import math

import torch
import torch.nn as nn


class SurfaceCNN(nn.Module):
    """CNN to process UV-grid faces (7 channels: x, y, z, nx, ny, nz, validity).

    Args:
        in_channels: Number of input channels per grid point (default 7).
        out_dim: Dimensionality of the output face embedding.
    """

    def __init__(self, in_channels=7, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, uv):
        """
        Args:
            uv: (B, N, C, H, W) UV-grid tensor for N faces.

        Returns:
            Face features of shape (B, N, 128).
        """
        B, N, C, H, W = uv.shape
        uv = uv.view(B * N, C, H, W)
        out = self.net(uv)
        return out.view(B, N, -1)


class AttributeEncoder(nn.Module):
    """MLP to encode per-face attribute vectors.

    Args:
        in_dim: Input attribute dimension (default 4).
        hidden: Hidden layer size.
        out_dim: Output embedding dimension.
    """

    def __init__(self, in_dim=4, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, D) face attribute tensor.

        Returns:
            Encoded attributes of shape (B, N, out_dim).
        """
        B, N, D = x.shape
        out = self.net(x.view(B * N, D))
        return out.view(B, N, -1)


class EdgeEncoder(nn.Module):
    """MLP to encode edge attributes with masked mean pooling.

    Processes per-edge attribute vectors through a two-layer MLP, then
    aggregates across the edge dimension using mean pooling (respecting an
    optional mask for padded positions).

    Args:
        in_dim: Input attribute dimension (default 4).
        hidden: Hidden layer size.
        out_dim: Output embedding dimension.
    """

    def __init__(self, in_dim=4, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, E, D) edge attribute tensor.
            mask: (B, E) boolean mask — True for real edges, False for padded.

        Returns:
            Aggregated edge features of shape (B, out_dim).
        """
        if x.size(1) == 0:
            return torch.zeros(
                x.size(0), self.net[-2].out_features, device=x.device
            )

        B, N, D = x.shape
        out = self.net(x.view(B * N, D))
        out = out.view(B, N, -1)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
            out_masked = out * mask_expanded.float()
            count = mask_expanded.float().sum(dim=1, keepdim=True)
            count = torch.clamp(count, min=1.0)
            out_agg = out_masked.sum(dim=1) / count.squeeze(-1)  # (B, out_dim)
        else:
            out_agg = out.mean(dim=1)  # (B, out_dim)

        return out_agg  # (B, out_dim)


class Edge1DCNN(nn.Module):
    """1D CNN for local edge feature dependencies.

    Applies per-edge 1D convolutions followed by masked mean pooling to
    produce a single aggregated edge representation per sample.

    Args:
        in_dim: Number of edge attribute channels (default 4).
        out_dim: Output embedding dimension (default 32).
    """

    def __init__(self, in_dim=4, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 16, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_projection = nn.Linear(32, out_dim)

    def forward(self, edges_attr, edge_mask=None):
        """
        Args:
            edges_attr: (B, E, 4) edge attribute tensor.
            edge_mask: (B, E) boolean mask — True for real edges, False for padded.

        Returns:
            Aggregated edge features of shape (B, out_dim).
        """
        B, E, D = edges_attr.shape

        x = edges_attr.view(B * E, D, 1)
        x = self.net(x)              # (B*E, 32, 1)
        x = x.squeeze(-1)            # (B*E, 32)
        x = self.out_projection(x)   # (B*E, out_dim)
        x = x.view(B, E, -1)         # (B, E, out_dim)

        if edge_mask is not None:
            edge_mask_expanded = edge_mask.unsqueeze(-1)  # (B, E, 1)
            x_masked = x * edge_mask_expanded.float()
            edge_count = edge_mask_expanded.float().sum(dim=1, keepdim=True)
            edge_count = torch.clamp(edge_count, min=1.0)
            x_agg = x_masked.sum(dim=1) / edge_count.squeeze(-1)  # (B, out_dim)
        else:
            x_agg = x.mean(dim=1)  # (B, out_dim)

        return x_agg  # (B, out_dim)


class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer with optional attention bias.

    Implements multi-head self-attention where an additive bias matrix
    (e.g. encoding graph structure) can be injected into the attention
    scores. Includes pre-norm residual connections and a feed-forward
    network.

    Args:
        d_model: Model / embedding dimension.
        nhead: Number of attention heads.
        dim_feedforward: Hidden size of the position-wise FFN.
        dropout: Dropout probability.
    """

    def __init__(self, d_model=128, nhead=8, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bias=None, mask=None):
        """
        Args:
            x: (B, N, d_model) node feature tensor.
            bias: (B, N, N) attention bias (e.g. adjacency-derived).
            mask: (B, N) boolean mask — True for real nodes, False for padded.

        Returns:
            Updated node features of shape (B, N, d_model).
        """
        B, N, _ = x.shape

        Q = self.q_lin(x).view(B, N, self.nhead, self.d_k).transpose(1, 2)
        K = self.k_lin(x).view(B, N, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_lin(x).view(B, N, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if bias is not None:
            scores = scores + bias.unsqueeze(1)

        if mask is not None:
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, N, N)
            scores = scores.masked_fill(~mask_2d.unsqueeze(1), -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out_lin(out)

        x = x + self.dropout(out)
        x = self.norm1(x)

        out = self.ffn(x)
        x = x + self.dropout(out)
        x = self.norm2(x)

        return x

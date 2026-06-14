"""
LatentStitcher: Stage 1 (frozen SVD projection) + Stage 2 (trainable residual MLP).

Forward pass:
  x  ∈ R^{src_dim}    ← Qwen last-token hidden state
  x_pad               ← zero-pad to tgt_dim
  x_coarse            ← W_optimal @ x_pad        (Stage 1, frozen)
  x_final             ← x_coarse + MLP(x_coarse) (Stage 2, trained)
"""

import torch
import torch.nn as nn
from torch import Tensor

from config import StitcherConfig


class SVDProjection(nn.Module):
    """Frozen orthogonal linear map from padded src space to tgt space."""

    def __init__(self, src_dim: int, tgt_dim: int, W_optimal: Tensor):
        super().__init__()
        assert W_optimal.shape == (tgt_dim, tgt_dim), (
            f"Expected W_optimal shape ({tgt_dim},{tgt_dim}), got {W_optimal.shape}"
        )
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        # Store as buffer so it moves with .to(device) but stays grad-free
        self.register_buffer("W", W_optimal)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., src_dim)
        pad = torch.zeros(*x.shape[:-1], self.tgt_dim - self.src_dim,
                          dtype=x.dtype, device=x.device)
        x_pad = torch.cat([x, pad], dim=-1)     # (..., tgt_dim)
        return x_pad @ self.W.T                 # (..., tgt_dim)


class ResidualMLP(nn.Module):
    """
    Depth-stacked residual MLP block.
    x_final = x + MLP(x)
    """

    def __init__(self, dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = dim if i == 0 else hidden_dim
            out_d = dim if i == num_layers - 1 else hidden_dim
            layers += [
                nn.LayerNorm(in_d),
                nn.Linear(in_d, out_d, bias=True),
            ]
            if i < num_layers - 1:
                layers += [nn.GELU()]
                if dropout > 0:
                    layers += [nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class LatentStitcher(nn.Module):
    """
    Two-stage latent stitcher.
    Stage 1: frozen SVD projection.
    Stage 2: trainable residual MLP.
    """

    def __init__(self, cfg: StitcherConfig, W_optimal: Tensor):
        super().__init__()
        self.stage1 = SVDProjection(cfg.src_dim, cfg.tgt_dim, W_optimal)
        for p in self.stage1.parameters():
            p.requires_grad = False

        self.stage2 = ResidualMLP(
            dim=cfg.tgt_dim,
            hidden_dim=cfg.mlp_hidden_dim,
            num_layers=cfg.mlp_num_layers,
            dropout=cfg.mlp_dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_coarse = self.stage1(x)
        x_final = self.stage2(x_coarse)
        return x_final

    def stage1_only(self, x: Tensor) -> Tensor:
        return self.stage1(x)

    def trainable_parameters(self):
        return self.stage2.parameters()

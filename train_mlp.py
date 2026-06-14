"""
Phase 3: Train the residual MLP (Stage 2) with InfoNCE + λ·MSE loss.

Stage 1 (SVD projection) is frozen throughout.
Only Stage 2 MLP parameters are updated.

Usage:
  python train_mlp.py [--data-dir DATA_DIR] [--svd-ckpt SVD_CKPT] [--out-dir OUT_DIR]
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import StitcherConfig
from stitcher_model import LatentStitcher


# ── Dataset ─────────────────────────────────────────────────────────────────

class HiddenStatePairDataset(Dataset):
    def __init__(self, data_dir: str):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")
        X_list, Y_list = [], []
        for f in files:
            d = np.load(f)
            X_list.append(d["X"].astype(np.float32))
            Y_list.append(d["Y"].astype(np.float32))
        self.X = torch.from_numpy(np.concatenate(X_list))
        self.Y = torch.from_numpy(np.concatenate(Y_list))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ── Loss ─────────────────────────────────────────────────────────────────────

def infonce_loss(x: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Symmetric InfoNCE over a batch.
    x, y: (B, D) — both L2-normalised inside this function.
    Diagonal entries are positives; all off-diagonal are negatives.
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = (x @ y.T) / temperature          # (B, B)
    labels = torch.arange(len(x), device=x.device)
    loss_x = F.cross_entropy(logits, labels)
    loss_y = F.cross_entropy(logits.T, labels)
    return (loss_x + loss_y) / 2


def combined_loss(
    x_final: torch.Tensor,
    y: torch.Tensor,
    temperature: float,
    lambda_mse: float,
) -> torch.Tensor:
    return infonce_loss(x_final, y, temperature) + lambda_mse * F.mse_loss(x_final, y)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: StitcherConfig, data_dir: str, svd_ckpt: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.source_device)
    dtype = getattr(torch, cfg.dtype)

    # Load W_optimal
    W_optimal = torch.load(svd_ckpt, map_location="cpu")

    # Build model
    model = LatentStitcher(cfg, W_optimal).to(device).to(dtype)

    # Dataset split
    dataset = HiddenStatePairDataset(data_dir)
    n_val = max(1, int(0.05 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    optimizer = AdamW(model.trainable_parameters(),
                      lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.num_epochs
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                      total_iters=cfg.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - cfg.warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[cfg.warmup_steps])

    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}"):
            X_batch = X_batch.to(device, dtype=dtype)
            Y_batch = Y_batch.to(device, dtype=dtype)

            x_final = model(X_batch)
            loss = combined_loss(x_final, Y_batch, cfg.infonce_temperature, cfg.lambda_mse)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.stage2.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validate ──
        model.eval()
        val_loss = val_cos = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device, dtype=dtype)
                Y_batch = Y_batch.to(device, dtype=dtype)
                x_final = model(X_batch)
                val_loss += combined_loss(x_final, Y_batch,
                                          cfg.infonce_temperature, cfg.lambda_mse).item()
                val_cos += F.cosine_similarity(
                    F.normalize(x_final, dim=-1),
                    F.normalize(Y_batch, dim=-1),
                ).mean().item()

        val_loss /= len(val_loader)
        val_cos  /= len(val_loader)

        print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_cos={val_cos:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(out_dir, "stitcher_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_cos": val_cos,
                "cfg": cfg,
            }, ckpt_path)
            print(f"  ↑ saved best checkpoint → {ckpt_path}")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--svd-ckpt", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = StitcherConfig()
    train(
        cfg,
        data_dir=args.data_dir or cfg.data_dir,
        svd_ckpt=args.svd_ckpt or cfg.svd_checkpoint,
        out_dir=args.out_dir or cfg.output_dir,
    )


if __name__ == "__main__":
    main()

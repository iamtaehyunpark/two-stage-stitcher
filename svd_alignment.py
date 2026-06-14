"""
Phase 2: Closed-form SVD alignment (Stage 1, frozen).

Algorithm:
  1. Load all (X, Y) pairs from data_dir.
  2. Zero-pad X from src_dim → tgt_dim.
  3. Compute cross-covariance A = X_pad^T @ Y  ∈ R^{tgt_dim × tgt_dim}.
  4. SVD(A) → U, S, V^T
  5. W_optimal = U @ V^T   (orthogonal Procrustes solution)
  6. Save W_optimal to checkpoint.

W_optimal is later loaded by the stitcher model with requires_grad=False.
"""

import os
import glob
import argparse
import numpy as np
import torch
from scipy.linalg import svd as scipy_svd
from tqdm import tqdm

from config import StitcherConfig


def load_all_pairs(data_dir: str):
    """Stack all X and Y arrays from .npz files in data_dir."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    X_list, Y_list = [], []
    for path in tqdm(files, desc="Loading pairs"):
        d = np.load(path)
        X_list.append(d["X"].astype(np.float64))
        Y_list.append(d["Y"].astype(np.float64))

    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)


def zero_pad(X: np.ndarray, tgt_dim: int) -> np.ndarray:
    """Pad X from (N, src_dim) → (N, tgt_dim) along the feature axis."""
    N, src_dim = X.shape
    if src_dim == tgt_dim:
        return X
    assert src_dim < tgt_dim, f"src_dim {src_dim} must be ≤ tgt_dim {tgt_dim}"
    pad = np.zeros((N, tgt_dim - src_dim), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def compute_w_optimal(X_pad: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Orthogonal Procrustes: find W = argmin_W ||X_pad @ W - Y||_F
    s.t. W^T W = I

    Solution: A = X_pad^T @ Y, SVD(A) = U S V^T, W = U @ V^T
    Returns W ∈ R^{tgt_dim × tgt_dim}.
    """
    print(f"Computing cross-covariance A = X^T @ Y  ({X_pad.shape[1]}×{X_pad.shape[1]}) …")
    # float64 for numerical precision
    A = X_pad.T @ Y                          # (tgt_dim, tgt_dim)

    print("Running SVD (this may take a minute for 8192×8192) …")
    U, _S, Vt = scipy_svd(A, full_matrices=True, check_finite=False)

    W_optimal = U @ Vt                       # (tgt_dim, tgt_dim)
    return W_optimal.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = StitcherConfig()
    data_dir = args.data_dir or cfg.data_dir
    out_path = args.out or cfg.svd_checkpoint

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    X, Y = load_all_pairs(data_dir)
    print(f"Loaded {X.shape[0]} pairs  X:{X.shape}  Y:{Y.shape}")

    X_pad = zero_pad(X, cfg.tgt_dim)
    W_opt = compute_w_optimal(X_pad, Y)

    # Quick sanity: mean cosine similarity after linear projection
    X_proj = (X_pad @ W_opt)               # (N, tgt_dim)
    cos = (
        (X_proj * Y).sum(-1)
        / (np.linalg.norm(X_proj, axis=-1) * np.linalg.norm(Y, axis=-1) + 1e-9)
    ).mean()
    mse = np.mean((X_proj - Y) ** 2)
    print(f"Stage-1 only   cosine_sim={cos:.4f}   MSE={mse:.6f}")

    torch.save(torch.from_numpy(W_opt), out_path)
    print(f"Saved W_optimal ({W_opt.shape}) → {out_path}")


if __name__ == "__main__":
    main()

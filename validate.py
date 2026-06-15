"""
Phase 4: Ablation & validation harness.

Metrics computed on unseen held-out documents:
  Metric A  — Stage 1 (SVD) only:          MSE_A, Cos_A
  Metric B  — Full pipeline (SVD + MLP):   MSE_B, Cos_B

Additional checks:
  Top-1 retrieval accuracy: X_final vs. a gallery of distractor Y vectors.
  Logs the delta between A and B to confirm MLP contribution.

Usage:
  python validate.py --val-dir VAL_DIR --ckpt CKPT_PATH [--gallery-size N]
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from config import StitcherConfig
from stitcher_model import LatentStitcher


def load_val_pairs(val_dir: str):
    files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files in {val_dir}")
    X_list, Y_list = [], []
    for f in files:
        d = np.load(f)
        X_list.append(d["X"].astype(np.float32))
        Y_list.append(d["Y"].astype(np.float32))
    return (
        torch.from_numpy(np.concatenate(X_list)),
        torch.from_numpy(np.concatenate(Y_list)),
    )


def pairwise_metrics(pred: Tensor, target: Tensor):
    mse = F.mse_loss(pred, target).item()
    cos = F.cosine_similarity(
        F.normalize(pred, dim=-1),
        F.normalize(target, dim=-1),
    ).mean().item()
    return mse, cos


def top1_retrieval_accuracy(queries: Tensor, gallery: Tensor) -> float:
    """
    For each query q_i, rank the gallery by cosine similarity.
    Success = correct target y_i ranks at position 1.
    queries[i] must correspond to gallery[i] (positive pair).
    """
    q = F.normalize(queries, dim=-1)    # (N, D)
    g = F.normalize(gallery, dim=-1)    # (N, D)
    sims = q @ g.T                      # (N, N)
    ranks = sims.argsort(dim=-1, descending=True)   # (N, N)
    labels = torch.arange(len(q), device=q.device)
    top1_correct = (ranks[:, 0] == labels).float().mean().item()
    return top1_correct


def run_ablation(
    model: LatentStitcher,
    X: Tensor,
    Y: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 256,
):
    model.eval()
    all_stage1, all_final = [], []

    for i in range(0, len(X), batch_size):
        xb = X[i: i + batch_size].to(device, dtype=dtype)
        with torch.no_grad():
            s1 = model.stage1_only(xb).float().cpu()
            sf = model(xb).float().cpu()
        all_stage1.append(s1)
        all_final.append(sf)

    stage1_out = torch.cat(all_stage1)
    final_out  = torch.cat(all_final)

    mse_a, cos_a = pairwise_metrics(stage1_out, Y)
    mse_b, cos_b = pairwise_metrics(final_out,  Y)

    top1_a = top1_retrieval_accuracy(stage1_out, Y)
    top1_b = top1_retrieval_accuracy(final_out,  Y)

    return {
        "stage1_mse": mse_a, "stage1_cos": cos_a, "stage1_top1": top1_a,
        "full_mse":   mse_b, "full_cos":   cos_b, "full_top1":   top1_b,
        "delta_mse":  mse_a - mse_b,              # positive = improvement
        "delta_cos":  cos_b - cos_a,              # positive = improvement
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True, help="Dir with unseen .npz pairs")
    parser.add_argument("--ckpt", required=True, help="Path to stitcher_best.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    cfg = StitcherConfig()
    device = torch.device(cfg.source_device)
    dtype = getattr(torch, cfg.dtype)

    print("Loading checkpoint …")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    W_optimal = ckpt["model_state"]["stage1.W"]
    model = LatentStitcher(cfg, W_optimal).to(device).to(dtype)
    model.load_state_dict(ckpt["model_state"])

    print("Loading validation pairs …")
    X, Y = load_val_pairs(args.val_dir)
    print(f"  {len(X)} pairs  X:{X.shape}  Y:{Y.shape}")

    print("\nRunning ablation …")
    results = run_ablation(model, X, Y, device, dtype, args.batch_size)

    print("\n─── Ablation Results ─────────────────────────────────")
    print(f"  Stage 1 only   MSE={results['stage1_mse']:.6f}  "
          f"Cos={results['stage1_cos']:.4f}  Top-1={results['stage1_top1']:.4f}")
    print(f"  Full pipeline  MSE={results['full_mse']:.6f}  "
          f"Cos={results['full_cos']:.4f}  Top-1={results['full_top1']:.4f}")
    print(f"  Delta          ΔMSE={results['delta_mse']:+.6f}  "
          f"ΔCos={results['delta_cos']:+.4f}")

    # Quality gates
    passed = True
    if results["delta_mse"] <= 0:
        print("\n[FAIL] MLP did not reduce MSE over SVD-only baseline.")
        passed = False
    if results["delta_cos"] <= 0:
        print("[FAIL] MLP did not improve cosine similarity.")
        passed = False
    if results["full_top1"] < 1.0:
        print(f"[WARN] Top-1 retrieval accuracy {results['full_top1']:.4f} < 1.0")

    if passed:
        print("\n[PASS] All ablation quality gates satisfied.")
    return results


if __name__ == "__main__":
    main()

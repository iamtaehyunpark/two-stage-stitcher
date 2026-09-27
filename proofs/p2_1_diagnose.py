"""
proofs/p2_1_diagnose.py — offline fidelity audit for Proof 2.1

Two checks, no model loading:

  1. Per-item Tier 2 cosines: refit the held-out ridge map on the saved train
     states, apply to each test doc, and report cos(Ŷ_heldout, Y_ds_true) per
     doc (per-token mean AND global flattened).  Resolves the AMPLIFICATION_FLAG:
       cos_per_doc ≥ 0.95 → map is accurate, amplification is real
       cos_per_doc  0.3–0.7 → map is just bad, amplification flag is an artifact

  2. Resampler round-trip (Trap #1): take DeepSeek's own Y_ds (saved in states),
     resample to the Qwen token length saved alongside it, then resample back.
     Compare cos(round_trip_Y_ds, Y_ds_true) per doc.  If this drops below 0.99
     the resampler is eating signal before any cross-family mapping happens, and
     neither tier's geometry conclusion is clean.

Usage:
    python proofs/p2_1_diagnose.py
    python proofs/p2_1_diagnose.py --states proofs/data/p2_1_states.npz \
                                   --results proofs/data/p2_1.json \
                                   --ridge-lambda 1e3 --tier2-test-n 10
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ── helpers (mirrors p2_1_oracle, no imports needed) ─────────────────────────

def resample_seq(X: torch.Tensor, n_target: int) -> torch.Tensor:
    if X.shape[0] == n_target:
        return X
    X_t = X.T.unsqueeze(0).float()
    X_r = F.interpolate(X_t, size=n_target, mode="linear", align_corners=False)
    return X_r.squeeze(0).T


def fit_heldout(Xs, Ys, ridge_lambda: float) -> torch.Tensor:
    X = torch.cat(Xs, dim=0).double()
    Y = torch.cat(Ys, dim=0).double()
    XtX = X.T @ X
    XtY = X.T @ Y
    lam = ridge_lambda * torch.eye(XtX.shape[0], dtype=XtX.dtype)
    W = torch.linalg.solve(XtX + lam, XtY)
    return W.float()


def load_states(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    states = {}
    for key in data.files:
        if key.endswith("__x"):
            doc_id = key[:-len("__x")]
            states[doc_id] = {
                "x": data[f"{doc_id}__x"],
                "y": data[f"{doc_id}__y"],
                "n_ds": int(data[f"{doc_id}__n_ds"]),
                "n_qw": int(data[f"{doc_id}__n_qw"]),
            }
    return states


def per_token_cos(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Per-row cosine similarity between (N, d) tensors A and B. Returns (N,)."""
    A = A.double()
    B = B.double()
    dot = (A * B).sum(dim=1)
    norm_a = A.norm(dim=1).clamp(min=1e-12)
    norm_b = B.norm(dim=1).clamp(min=1e-12)
    return (dot / (norm_a * norm_b)).float()


def global_cos(A: torch.Tensor, B: torch.Tensor) -> float:
    """Single cosine between two flattened tensors (the current map_fidelity method)."""
    a = A.reshape(-1).double()
    b = B.reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states",        default="proofs/data/p2_1_states.npz")
    ap.add_argument("--results",       default="proofs/data/p2_1.json")
    ap.add_argument("--ridge-lambda",  type=float, default=1e3)
    ap.add_argument("--tier2-test-n",  type=int,   default=10)
    args = ap.parse_args()

    # ── load ─────────────────────────────────────────────────────────────────
    print(f"Loading states from {args.states} …")
    states = load_states(args.states)
    print(f"  {len(states)} docs in states")

    print(f"Loading results from {args.results} …")
    with open(args.results) as f:
        results = json.load(f)

    tier1_records = results.get("tier1_records", [])
    gated_ids = [r["id"] for r in tier1_records]
    n_test = args.tier2_test_n
    test_ids  = set(gated_ids[-n_test:])
    train_ids = set(gated_ids[:-n_test])
    print(f"  split: {len(train_ids)} train / {len(test_ids)} test (last {n_test})")

    # ── Resampler round-trip (Trap #1) ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CHECK 1: Resampler round-trip  (Y_ds → n_qw → n_ds vs Y_ds_true)")
    print("=" * 72)
    print(f"  {'doc_id':30s}  {'n_ds':>6}  {'n_qw':>6}  {'cos_roundtrip':>14}  {'mse_roundtrip':>14}")
    print(f"  {'-'*30}  {'------':>6}  {'------':>6}  {'-------------':>14}  {'-------------':>14}")

    rt_coss = []
    for doc_id in sorted(states.keys()):
        s = states[doc_id]
        Y_ds  = torch.from_numpy(s["y"]).float()   # (n_ds, d_ds)
        n_ds  = s["n_ds"]
        n_qw  = s["n_qw"]

        Y_mid   = resample_seq(Y_ds, n_qw)          # (n_qw, d_ds)
        Y_back  = resample_seq(Y_mid, n_ds)          # (n_ds, d_ds)

        c = global_cos(Y_back, Y_ds)
        mse = float(torch.mean((Y_back - Y_ds) ** 2))
        rt_coss.append(c)
        label = doc_id[:30]
        print(f"  {label:30s}  {n_ds:>6}  {n_qw:>6}  {c:>14.6f}  {mse:>14.2e}")

    print(f"\n  round-trip cos: min={min(rt_coss):.6f}  mean={np.mean(rt_coss):.6f}"
          f"  max={max(rt_coss):.6f}")
    if min(rt_coss) < 0.99:
        print("  *** WARNING: round-trip cos < 0.99 — resampler is losing signal. ***")
        print("  *** Tier 1 and Tier 2 geometry conclusions may not be clean.      ***")
    else:
        print("  [OK] round-trip cos ≥ 0.99 — resampler is not the bottleneck.")

    # ── Tier 2 per-item fidelity (held-out oracle) ────────────────────────────
    print("\n" + "=" * 72)
    print("CHECK 2: Per-item Tier 2 fidelity  (Ŷ_heldout vs Y_ds_true)")
    print("=" * 72)

    train_states = {k: v for k, v in states.items() if k in train_ids}
    test_states  = {k: v for k, v in states.items() if k in test_ids}
    missing_train = train_ids - set(train_states)
    missing_test  = test_ids  - set(test_states)
    if missing_train:
        print(f"  WARNING: {len(missing_train)} train doc(s) missing from states: {missing_train}")
    if missing_test:
        print(f"  WARNING: {len(missing_test)} test doc(s) missing from states: {missing_test}")

    if not train_states:
        print("  No train states — cannot fit ridge map. Abort.")
        return

    # Refit ridge (double precision for stability)
    train_Xs = [torch.from_numpy(s["x"]).float() for s in train_states.values()]
    train_Ys = [torch.from_numpy(s["y"]).float() for s in train_states.values()]
    print(f"  Fitting ridge on {len(train_Xs)} train docs (λ={args.ridge_lambda}) …")
    W = fit_heldout(train_Xs, train_Ys, args.ridge_lambda)
    print(f"  W shape={tuple(W.shape)}, ||W||_F={W.norm().item():.2f}")

    print()
    print(f"  {'doc_id':30s}  {'n_ds':>6}  {'cos_global':>11}  "
          f"{'cos_tok_mean':>13}  {'cos_tok_min':>12}  {'cos_tok_max':>12}  {'cos_tok_std':>12}")
    print(f"  {'-'*30}  {'------':>6}  {'-----------':>11}  "
          f"{'-------------':>13}  {'------------':>12}  {'------------':>12}  {'------------':>12}")

    t2_coss_global = []
    t2_coss_token  = []

    for doc_id in sorted(test_states.keys()):
        s = test_states[doc_id]
        X = torch.from_numpy(s["x"]).float()   # (n_ds, d_qw)
        Y = torch.from_numpy(s["y"]).float()   # (n_ds, d_ds)

        Y_hat = X @ W                           # (n_ds, d_ds)

        c_global = global_cos(Y_hat, Y)
        tok_cos  = per_token_cos(Y_hat, Y)      # (n_ds,)

        t2_coss_global.append(c_global)
        t2_coss_token.append(tok_cos.mean().item())

        label = doc_id[:30]
        print(f"  {label:30s}  {s['n_ds']:>6}  {c_global:>11.6f}  "
              f"{tok_cos.mean():>13.6f}  {tok_cos.min():>12.6f}  "
              f"{tok_cos.max():>12.6f}  {tok_cos.std():>12.6f}")

    print()
    print(f"  held-out cos (global)  : min={min(t2_coss_global):.4f}  "
          f"mean={np.mean(t2_coss_global):.4f}  max={max(t2_coss_global):.4f}")
    print(f"  held-out cos (tok mean): min={min(t2_coss_token):.4f}  "
          f"mean={np.mean(t2_coss_token):.4f}  max={max(t2_coss_token):.4f}")

    print()
    mean_c = np.mean(t2_coss_token)
    if mean_c >= 0.95:
        print("  INTERPRETATION: per-doc token-mean cos ≥ 0.95.")
        print("  The held-out map is geometrically accurate. Recall=0.2 with cos≈1")
        print("  → AMPLIFICATION is real: 68 layers of recompute destroy the margin.")
        print("  → Run Proof 2.4 to quantify the error amplification threshold.")
    elif mean_c >= 0.70:
        print("  INTERPRETATION: per-doc token-mean cos in 0.70–0.95.")
        print("  The map is moderately accurate but not tight. Some amplification")
        print("  may be present, but the map's own error is also a factor.")
        print("  → Do not carry AMPLIFICATION_FLAG forward without a tighter map.")
    else:
        print("  INTERPRETATION: per-doc token-mean cos < 0.70.")
        print("  The held-out map is just bad — global linear does not bridge the")
        print("  geometry. The AMPLIFICATION_FLAG is an artifact of a bad fidelity")
        print("  measurement (map_fidelity was computing global flattened cos,")
        print("  likely dominated by the Tier-1 trivial-fit artifact).")
        print("  → Yellow verdict stands (geometry not linearly bridgeable) but")
        print("     amplification is NOT established. 2.4 can wait.")


if __name__ == "__main__":
    main()

"""
Proof 2.1 — Cross-family geometry (oracle map, no training).

Chain 2, rung 1. Proof 2.0 established that the layer-12 residual is a sufficient
handoff target: recomputing the upper stack from it reproduces the stored-cache
injection to within one judge disagreement of 40. This rung asks the existential
question: is there ANY transformation of Qwen's document representation that DeepSeek
can reason over — injected as a layer-12 residual via the 2.0-validated path?

Two models are active for the first time:
  Qwen2.5-7B  (SLM, the reader)    — produces layer-12 residuals for the document
  DeepSeek-70B (LLM, the reasoner) — receives the mapped residuals, answers the question

The cross-family obstacle: different tokenizers → different sequence lengths (N_qw ≠
N_ds). There is no natural position-to-position correspondence. The oracle bypasses
this by linearly interpolating Qwen's state sequence to DeepSeek's length, then
fitting a linear map from the resampled Qwen states to DeepSeek's true residuals.

Three tiers (cheapest first; each gates the next):
  Tier 1 — per-document oracle (the true falsifier): overfit the map on THIS document's
            (X_qw_resampled, Y_ds_true). If even this fails → geometry unbridgeable →
            HARD KILL of the outsourcing thesis. For short docs (N_ds < d_qw = 3584) the
            system is underdetermined so Y_hat = Y_ds exactly; Tier 1 is mainly a
            plumbing / mechanism sanity check. A Tier 1 FAILURE would be surprising and
            means something fundamental is broken.
  Tier 2 — held-out oracle (the generalization guard): fit ONE ridge map on ~75% of
            gated docs, test on the unseen ~25%. Discriminates "spaces align globally"
            from "per-doc overfit faked it." This is the real geometry test.
  Tier 3 — fidelity-vs-recall logging (free; runs alongside Tiers 1 & 2): for every
            injection, record (cos/MSE between Y_hat and Y_ds_true, recall). The gap is
            the first measurement of error amplification (Proof 2.4 arriving early).

Operating point: identical to 2.0 — layer 12, q-fair capture (PREFILL_PREFIX framing),
think-ON, strict + LLM-judge. Zero-memory gated set (C fails AND A succeeds).

Conditions per item:
  A                    full text prefill                        (ceiling; from gate)
  residual_inject_true DeepSeek's own L12 residual injected     (the map's ceiling)
  tier1_perdoc         per-doc oracle map (Tier 1)
  tier2_heldout        held-out oracle map on test docs (Tier 2; test subset only)

Verdict:
  Tier 1 fail  → HARD KILL (even overfit can't bridge → geometry dead)
  Tier 1 pass, Tier 2 pass  → GREEN_LIGHT_TO_2_2
  Tier 1 pass, Tier 2 fail  → YELLOW (alignment only per-doc; reframe 2.2 as harder)
  high Tier-2 fidelity + low recall → AMPLIFICATION_FLAG (key 2.4 warning)

Usage:
  # gate + Tier 1 (both models):
  CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p2_1_oracle.py --arm synth_multihop \\
      --synth-n 40 --out proofs/data/p2_1.json

  # Tier 2 only (after Tier 1 checkpoint is complete; DeepSeek only):
  CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p2_1_oracle.py --arm synth_multihop \\
      --synth-n 40 --out proofs/data/p2_1.json --tier2-only

  # wire-test (no think, no judge, n=6):
  CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p2_1_oracle.py --arm synth_multihop \\
      --synth-n 6 --no-think --no-judge --out proofs/data/p2_1_test.json

  # re-score / re-verdict a saved run (no GPU):
  python proofs/p2_1_oracle.py --rescore proofs/data/p2_1.json
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from proofs.common import (
    load_deepseek, final_answer, _with_think_control,
)
from core.split_forward import (
    capture_doc_cache, split_forward_generate,
    recompute_doc_cache_from_residual, kv_drift_upper,
)
from proofs.p5_latent_vs_rag import (
    run_gate, load_candidates, score_all, judge_answer, ans_qfair,
    _present_scorers, _set_think, _free_cuda,
    PREFILL_PREFIX, PREFILL_QSUFFIX, MIN_N, GAP,
)

# How much Tier-2 recall can fall below the ceiling before it's a failure
TIER2_FAIL_THRESHOLD = 0.40   # tier2_recall < 40% of ceiling recall → yellow/fail
# Fidelity above this + recall below TIER2_FAIL_THRESHOLD → amplification flag
AMPLIFICATION_FID_FLOOR = 0.90
# Off-diagonal agreement tolerance (inherited from 2.0)
AGREE_TOL = 0.10


# ── Qwen loader ───────────────────────────────────────────────────────────────

def load_qwen(cfg, device=None):
    """Load Qwen2.5-7B onto `device`.

    Avoids `device_map=` (a string device passed to from_pretrained) because newer
    transformers 4.4x versions added `caching_allocator_warmup` which calls
    `torch.cuda.mem_get_info(index)` on every mapped device — and that fails with
    "invalid device ordinal" when the CUDA environment doesn't expose the expected
    GPU index. For a 7B model (≈14 GB bf16), loading on CPU then `.to(device)` is
    safe, bypasses caching_allocator_warmup, and works across all 4.x versions.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = device or cfg.source_device
    dtype = getattr(torch, cfg.dtype)

    # Fail fast: validate the target device before a multi-GB download/load.
    try:
        torch.zeros(1, device=device)
    except (RuntimeError, AssertionError) as e:
        n = torch.cuda.device_count()
        raise RuntimeError(
            f"Qwen target device {device!r} is not available "
            f"(CUDA sees {n} device(s): cuda:0 … cuda:{n-1}). "
            "Pass --qwen-device to one of the DeepSeek shard GPUs (default cuda:0)."
        ) from e

    print(f"Loading Qwen {cfg.source_model} on {device} (CPU → .to(device)) …")
    tok = AutoTokenizer.from_pretrained(cfg.source_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.source_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,   # load weights one module at a time to cap peak RAM
    ).to(device)
    model.eval()
    return tok, model


# ── Qwen residual capture ─────────────────────────────────────────────────────

@torch.no_grad()
def capture_qwen_residual(model, tok, text, layer, max_tokens):
    """Capture the hidden states ENTERING layer `layer` of Qwen for `text`.

    Returns (Y_qw, N_qw):
      Y_qw  — (1, N_qw, d_qw) float32 on CPU (small; keep for later fitting)
      N_qw  — token count

    Uses a pre-hook so the capture is independent of the model's output_hidden_states
    setting and does not store all intermediate activations. The document is tokenized
    with Qwen's tokenizer under the same PREFILL_PREFIX framing used by DeepSeek's
    q-fair capture, so the oracle map bridges the same semantic framing in both spaces.
    """
    captured = {}

    def _hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is not None:
            captured["y"] = hs.detach().float().cpu()

    handle = model.model.layers[layer].register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        device = next(model.parameters()).device
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_tokens).input_ids.to(device)
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        handle.remove()

    Y_qw = captured["y"]       # (1, N_qw, d_qw)
    N_qw = int(Y_qw.shape[1])
    return Y_qw, N_qw


# ── Sequence resampling ───────────────────────────────────────────────────────

def resample_seq(X, n_target):
    """Resample a token sequence X (N_src, d) to (n_target, d) via linear interpolation.

    Treats each feature dimension as a 1-D signal sampled at N_src positions and
    resamples to n_target positions. This is the minimal alignment step for the
    Qwen→DeepSeek tokenizer-length mismatch; it does not assume any semantic
    correspondence between Qwen token i and DeepSeek token i — only that the two
    sequences span the same document proportionally.
    """
    if X.shape[0] == n_target:
        return X
    # F.interpolate expects (batch=1, channels=d, length=N_src)
    X_t = X.T.unsqueeze(0).float()   # (1, d, N_src)
    X_r = F.interpolate(X_t, size=n_target, mode="linear", align_corners=False)
    return X_r.squeeze(0).T           # (n_target, d)


# ── Oracle map fitting ────────────────────────────────────────────────────────

def fit_perdoc(X, Y):
    """Per-document oracle map: min-norm linear map from X (N, d_qw) to Y (N, d_ds).

    For N < d_qw (underdetermined), the min-norm solution gives Y_hat = Y exactly
    when X has full row rank — which is expected for hidden states at N ≤ 500 and
    d_qw = 3584. This is intentional: Tier 1 is the most generous possible test.

    Returns W (d_qw, d_ds) such that X @ W ≈ Y.
    """
    result = torch.linalg.lstsq(X.float(), Y.float(), driver="gelsd")
    return result.solution   # (d_qw, d_ds)


def fit_heldout(Xs, Ys, ridge_lambda):
    """Held-out oracle map: ridge regression on a collection of (X_i, Y_i) pairs.

    Xs: list of (N_i, d_qw) float32 tensors
    Ys: list of (N_i, d_ds) float32 tensors
    Solves: W = (X^T X + λI)^{-1} X^T Y where X, Y are the concatenated stacks.

    Ridge regularisation (λ = ridge_lambda) forces generalisation. Choose λ relative
    to the diagonal of X^T X — for fp16 activations with typical L2 norm ≈ O(10),
    diagonal values ≈ N × d_qw × 100 ≈ 15000 × 3584 × 100 ≈ 5e9, so λ = 1e3 is a
    very mild regulariser. Raise to 1e5–1e6 for stronger shrinkage if Tier 2 overfits.
    """
    X = torch.cat(Xs, dim=0).float()   # (N_total, d_qw)
    Y = torch.cat(Ys, dim=0).float()   # (N_total, d_ds)
    XtX = X.T @ X                       # (d_qw, d_qw)
    XtY = X.T @ Y                       # (d_qw, d_ds)
    lam = ridge_lambda * torch.eye(XtX.shape[0], dtype=XtX.dtype)
    W = torch.linalg.solve(XtX + lam, XtY)  # (d_qw, d_ds)
    return W


# ── Fidelity (Tier 3 — free logging) ─────────────────────────────────────────

def map_fidelity(Y_hat, Y_true):
    """Residual-space fidelity between the mapped estimate Y_hat and the true Y_true.

    Both are (N, d) float32 tensors. The gap between this fidelity and downstream
    recall is the first measurement of error amplification (Proof 2.4 arrives early).
    """
    h = Y_hat.reshape(-1).float()
    t = Y_true.reshape(-1).float()
    cos = F.cosine_similarity(h.unsqueeze(0), t.unsqueeze(0)).item()
    mse = float(torch.mean((Y_hat.float() - Y_true.float()) ** 2))
    return {"cos": round(cos, 6), "mse": mse}


# ── Oracle injection ──────────────────────────────────────────────────────────

def inject_oracle(model, tok, structure_cache, Y_hat, n_ds, q, layer, m):
    """Inject Y_hat (1, N_ds, d_ds) via the 2.0-validated residual path and answer q.

    structure_cache is the DeepSeek DynamicCache from capture_doc_cache — it provides
    the per-layer shape metadata for recompute_doc_cache_from_residual and is NOT used
    as a source of stored KV (it is cleared internally before the recompute).
    """
    resid_cache = recompute_doc_cache_from_residual(model, Y_hat, n_ds, layer,
                                                    structure_cache)
    query = _with_think_control(PREFILL_QSUFFIX.format(question=q))
    txt = split_forward_generate(model, tok, resid_cache, n_ds, query_text=query,
                                 target_layer=layer, max_new_tokens=m)
    del resid_cache
    return final_answer(txt)


# ── States cache (X_resampled, Y_ds per doc) ─────────────────────────────────
# Stored as a .npz alongside the main JSON so Tier 2 can fit the held-out map
# without reloading Qwen or re-running the DeepSeek document forward.

def _states_path(out_path):
    return out_path.replace(".json", "_states.npz")


def save_states(path, states):
    """states: {doc_id: {"x": (N_ds, d_qw) np.float32, "y": (N_ds, d_ds) np.float32,
                          "n_ds": int, "n_qw": int}}"""
    arrays = {}
    for doc_id, s in states.items():
        arrays[f"{doc_id}__x"] = s["x"]
        arrays[f"{doc_id}__y"] = s["y"]
        arrays[f"{doc_id}__n_ds"] = np.array(s["n_ds"], dtype=np.int32)
        arrays[f"{doc_id}__n_qw"] = np.array(s["n_qw"], dtype=np.int32)
    np.savez_compressed(path, **arrays)


def load_states(path):
    """Returns {doc_id: {"x": ..., "y": ..., "n_ds": int, "n_qw": int}} or {}."""
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=False)
    states = {}
    for key in data.files:
        if key.endswith("__x"):
            doc_id = key[: -len("__x")]
            states[doc_id] = {
                "x": data[f"{doc_id}__x"],
                "y": data[f"{doc_id}__y"],
                "n_ds": int(data[f"{doc_id}__n_ds"]),
                "n_qw": int(data[f"{doc_id}__n_qw"]),
            }
    return states


# ── Tier 1 eval (per-doc oracle + residual_inject_true) ──────────────────────

def run_tier1(ds_model, ds_tok, qw_model, qw_tok, gated, args, resume_path, states_path):
    """For each gated doc:
      1. Capture DeepSeek q-fair cache and true L12 residual Y_ds
      2. Capture Qwen L12 residual Y_qw (same framing)
      3. Resample Y_qw to (N_ds, d_qw); store X for Tier 2
      4. Condition residual_inject_true: inject Y_ds via the recompute path (same as 2.0)
      5. Tier 1: fit per-doc map W = lstsq(X, Y_ds); inject Y_hat = X @ W
      6. Log fidelity (cos/MSE between Y_hat and Y_ds) alongside recall

    Returns (tier1_records, states_dict). Both are checkpointed after each item.
    """
    layer = args.layer
    m = args.think_max_new_tokens if args.think else args.max_new_tokens

    records, prev_result = [], {}
    if resume_path and os.path.exists(resume_path):
        try:
            with open(resume_path) as f:
                prev_result = json.load(f)
            records = prev_result.get("tier1_records", [])
        except Exception as e:
            print(f"  [resume] could not read {resume_path}: {e}")
    done_ids = {r["id"] for r in records}
    if records:
        print(f"  [tier1 resume] {len(records)} records loaded; their ids skipped")

    states = load_states(states_path)
    if states:
        print(f"  [tier1 resume] {len(states)} state entries loaded from {states_path}")

    cap = args.max_eval or len(gated)
    todo = [g for g in gated if g["id"] not in done_ids][:max(0, cap - len(records))]
    print(f"  tier1 plan: {len(records)} done, {len(todo)} to run "
          f"(cap {cap}, gated {len(gated)}); think_max={m}, judge={args.judge}")

    def snapshot():
        return {**prev_result,
                "tier1_records": records,
                "layer": layer, "think": args.think, "judged": args.judge}

    for rec in todo:
        doc, q, gold = rec["doc_text"], rec["question"], rec["answer"]
        decoys = rec.get("decoy_values", [])
        prefill_text = PREFILL_PREFIX.format(document=doc)

        # ── DeepSeek: q-fair capture (same framing as p2_0) ──────────────────
        pre_ids = ds_tok(prefill_text, return_tensors="pt",
                         truncation=True, max_length=args.max_doc_tokens).input_ids
        qcache, Y_ds, n_ds = capture_doc_cache(ds_model, pre_ids, layer)
        # Y_ds: (1, N_ds, 8192) on the device of layer `layer` in DeepSeek

        # ── Qwen: L-12 residual capture (same framing) ───────────────────────
        Y_qw, n_qw = capture_qwen_residual(
            qw_model, qw_tok, prefill_text, args.qwen_layer, args.max_doc_tokens)
        # Y_qw: (1, N_qw, 3584) float32 on CPU

        # ── Resample Qwen states to DeepSeek's token length ──────────────────
        X = resample_seq(Y_qw.squeeze(0), n_ds)   # (N_ds, d_qw) float32 CPU
        Y_ds_cpu = Y_ds.squeeze(0).float().cpu()   # (N_ds, d_ds) float32 CPU
        # Keep small arrays for Tier 2 fitting (copy: don't share storage with X/Y_ds_cpu
        # tensors that are deleted at the end of this iteration for GPU memory pressure)
        states[rec["id"]] = {
            "x": X.numpy().copy(),
            "y": Y_ds_cpu.numpy().copy(),
            "n_ds": n_ds,
            "n_qw": n_qw,
        }

        _set_think(args.think)

        # ── Condition: residual_inject_true (DeepSeek's own Y_ds → recompute path) ──
        # This is the ceiling the oracle map aims at; same as p2_0's residual_inject.
        resid_true_cache = recompute_doc_cache_from_residual(
            ds_model, Y_ds, n_ds, layer, qcache)
        drift = kv_drift_upper(qcache, resid_true_cache, layer)
        query = _with_think_control(PREFILL_QSUFFIX.format(question=q))
        ans_true = final_answer(split_forward_generate(
            ds_model, ds_tok, resid_true_cache, n_ds, query_text=query,
            target_layer=layer, max_new_tokens=m))
        del resid_true_cache

        # ── Tier 1: per-doc oracle map ────────────────────────────────────────
        W_perdoc = fit_perdoc(X, Y_ds_cpu)           # (d_qw, d_ds) CPU float32
        Y_hat = X @ W_perdoc                         # (N_ds, d_ds) CPU float32
        fid1 = map_fidelity(Y_hat, Y_ds_cpu)
        Y_hat_t = Y_hat.unsqueeze(0).to(device=Y_ds.device, dtype=Y_ds.dtype)
        ans_t1 = inject_oracle(ds_model, ds_tok, qcache, Y_hat_t, n_ds, q, layer, m)
        del W_perdoc, Y_hat, Y_hat_t

        answers = {"A": rec["a_ans"],
                   "residual_inject_true": ans_true,
                   "tier1_perdoc": ans_t1}
        scores = {c: score_all(a, gold, decoys) for c, a in answers.items()}

        if args.judge:
            for c, a in answers.items():
                scores[c]["judge"] = judge_answer(ds_model, ds_tok, q, gold, a)

        records.append({
            "id": rec["id"], "question": q, "gold": gold,
            "type": rec.get("type", ""), "decoy_values": decoys,
            "n_ds": int(n_ds), "n_qw": int(n_qw),
            "answers": answers, "scores": scores,
            "fidelity": {"tier1_perdoc": fid1},
            "kv_drift": drift,
            "tier2_split": "train",   # tentative; overwritten when Tier 2 runs
        })

        del qcache, Y_ds, Y_qw, X, Y_ds_cpu
        _free_cuda()

        if resume_path:
            with open(resume_path, "w") as f:
                json.dump(snapshot(), f, default=str)
        save_states(states_path, states)

        s = args.headline if args.headline in scores["tier1_perdoc"] else "strict"
        print(f"  tier1 [{len(records)} done / {cap}] {rec['id']}: "
              f"true={int(scores['residual_inject_true'][s])} "
              f"t1={int(scores['tier1_perdoc'][s])} "
              f"fid_cos={fid1['cos']:.4f}", end="\r")

    print()
    return records, states


# ── Tier 2 eval (held-out oracle) ─────────────────────────────────────────────

def _tier2_split(gated, n_test):
    """Deterministic train/test split: first (N - n_test) docs are train, rest test.
    Uses gated item order (which is already deterministic from the synthetic builder).
    """
    n_test = min(n_test, max(1, len(gated) // 4))
    return gated[: len(gated) - n_test], gated[len(gated) - n_test:]


def run_tier2(ds_model, ds_tok, gated, tier1_records, states, args, resume_path):
    """Fit ONE ridge map on the training-set states, then inject on each test doc.

    For each test doc:
      1. Fit W_heldout on all training-set (X_i, Y_i) pairs
      2. Apply: Y_hat = X_test @ W_heldout
      3. Re-capture DeepSeek qcache for the test doc (needed for the recompute path)
      4. Inject Y_hat, answer, score
      5. Log fidelity

    tier1_records is updated in-place (adds tier2_heldout to the matching record).
    Returns the list of tier2 records (same objects as the updated tier1_records).
    """
    if not gated or not states:
        print("  [tier2] no gated items or no states; skip")
        return []

    layer = args.layer
    m = args.think_max_new_tokens if args.think else args.max_new_tokens

    # Tier 2 resume: records that already have tier2_heldout are done
    t2_done = {r["id"] for r in tier1_records
               if "tier2_heldout" in r.get("answers", {})}
    if t2_done:
        print(f"  [tier2 resume] {len(t2_done)} items already have tier2_heldout")

    train_docs, test_docs = _tier2_split(gated, args.tier2_test_n)
    print(f"  tier2 split: {len(train_docs)} train / {len(test_docs)} test "
          f"(--tier2-test-n {args.tier2_test_n})")

    # Label the split in tier1_records (diagnostic, not needed for verdict)
    test_ids = {d["id"] for d in test_docs}
    for r in tier1_records:
        r["tier2_split"] = "test" if r["id"] in test_ids else "train"

    # Build training stack from states
    train_Xs, train_Ys = [], []
    for doc in train_docs:
        s = states.get(doc["id"])
        if s is None:
            print(f"  [tier2] WARNING: no states for train doc {doc['id']}; skip")
            continue
        train_Xs.append(torch.from_numpy(s["x"]).float())
        train_Ys.append(torch.from_numpy(s["y"]).float())

    if not train_Xs:
        print("  [tier2] no training states; skip")
        return []

    print(f"  tier2 fitting ridge map on {len(train_Xs)} train docs "
          f"(λ={args.ridge_lambda}) …")
    W_heldout = fit_heldout(train_Xs, train_Ys, args.ridge_lambda)
    print(f"  W_heldout shape={tuple(W_heldout.shape)}, "
          f"||W||_F={W_heldout.norm().item():.2f}")

    todo_test = [d for d in test_docs if d["id"] not in t2_done]
    print(f"  tier2 eval: {len(t2_done)} done, {len(todo_test)} to run")

    tier2_records = []

    def _save_snapshot():
        if resume_path:
            with open(resume_path) as f:
                saved = json.load(f)
            saved["tier1_records"] = tier1_records
            with open(resume_path, "w") as f:
                json.dump(saved, f, default=str)

    for doc in todo_test:
        doc_id = doc["id"]
        s = states.get(doc_id)
        if s is None:
            print(f"  [tier2] WARNING: no states for test doc {doc_id}; skip")
            continue

        X_test = torch.from_numpy(s["x"]).float()   # (N_ds, d_qw)
        Y_ds_cpu = torch.from_numpy(s["y"]).float()  # (N_ds, d_ds)
        n_ds = s["n_ds"]
        q, gold = doc["question"], doc["answer"]
        decoys = doc.get("decoy_values", [])

        # Apply held-out map
        Y_hat = X_test @ W_heldout                   # (N_ds, d_ds) CPU float32
        fid2 = map_fidelity(Y_hat, Y_ds_cpu)

        # Re-capture DeepSeek qcache for this doc (needed for the recompute path)
        prefill_text = PREFILL_PREFIX.format(document=doc["doc_text"])
        pre_ids = ds_tok(prefill_text, return_tensors="pt",
                         truncation=True, max_length=args.max_doc_tokens).input_ids
        qcache, Y_ds_fresh, _n = capture_doc_cache(ds_model, pre_ids, layer)
        del Y_ds_fresh

        _set_think(args.think)
        Y_hat_t = Y_hat.unsqueeze(0).to(device=next(ds_model.parameters()).device,
                                         dtype=ds_model.model.embed_tokens.weight.dtype)
        ans_t2 = inject_oracle(ds_model, ds_tok, qcache, Y_hat_t, n_ds, q, layer, m)
        del Y_hat_t, Y_hat

        sc_t2 = score_all(ans_t2, gold, decoys)
        if args.judge:
            sc_t2["judge"] = judge_answer(ds_model, ds_tok, q, gold, ans_t2)

        del qcache
        _free_cuda()

        # Merge into the matching tier1 record (in-place update)
        matched = next((r for r in tier1_records if r["id"] == doc_id), None)
        if matched is not None:
            matched["answers"]["tier2_heldout"] = ans_t2
            matched["scores"]["tier2_heldout"] = sc_t2
            matched["fidelity"]["tier2_heldout"] = fid2
            tier2_records.append(matched)

        _save_snapshot()

        s2 = args.headline if args.headline in sc_t2 else "strict"
        print(f"  tier2 [{len(tier2_records)} / {len(todo_test)}] {doc_id}: "
              f"t2={int(sc_t2[s2])} fid_cos={fid2['cos']:.4f}", end="\r")

    print()
    # Include already-done test records
    all_test_ids = {d["id"] for d in test_docs}
    tier2_records = [r for r in tier1_records
                     if r["id"] in all_test_ids and "tier2_heldout" in r.get("answers", {})]
    return tier2_records


# ── Aggregation ────────────────────────────────────────────────────────────────

def _rate(records, cond, scorer):
    vals = [r["scores"][cond][scorer] for r in records
            if scorer in r.get("scores", {}).get(cond, {})]
    return round(sum(vals) / len(vals), 3) if vals else None


def _diff(a, b):
    return None if a is None or b is None else round(a - b, 3)


def _fidelity_summary(records, key):
    cos_vals = [r["fidelity"][key]["cos"] for r in records
                if key in r.get("fidelity", {})]
    mse_vals = [r["fidelity"][key]["mse"] for r in records
                if key in r.get("fidelity", {})]
    if not cos_vals:
        return {"cos_mean": None, "mse_mean": None, "n": 0}
    return {
        "cos_mean": round(sum(cos_vals) / len(cos_vals), 4),
        "mse_mean": sum(mse_vals) / len(mse_vals),
        "n": len(cos_vals),
    }


def aggregate(tier1_records, tier2_records, headline="judge"):
    """Compute per-condition recall rates and headline diffs for the verdict."""
    hs = headline
    if tier1_records:
        t1_scols = _present_scorers(tier1_records)
        if hs not in t1_scols:
            hs = "strict"
    else:
        t1_scols = []

    t2_scols = _present_scorers(tier2_records) if tier2_records else t1_scols

    conds_t1 = ["A", "residual_inject_true", "tier1_perdoc"]
    rates_t1 = {c: {s: _rate(tier1_records, c, s) for s in t1_scols}
                for c in conds_t1} if tier1_records else {}

    conds_t2 = ["A", "residual_inject_true", "tier1_perdoc", "tier2_heldout"]
    rates_t2 = {c: {s: _rate(tier2_records, c, s) for s in t2_scols}
                for c in conds_t2} if tier2_records else {}

    # Tier 1: per-doc oracle vs ceiling (should be ~0 gap for short docs)
    t1_recall = _rate(tier1_records, "tier1_perdoc", hs)
    t1_ceil = _rate(tier1_records, "residual_inject_true", hs)

    # Tier 2: held-out oracle vs ceiling (on test docs only)
    t2_recall = _rate(tier2_records, "tier2_heldout", hs) if tier2_records else None
    t2_ceil = _rate(tier2_records, "residual_inject_true", hs) if tier2_records else None

    return {
        "n_tier1": len(tier1_records),
        "n_tier2": len(tier2_records),
        "headline": hs,
        "rates_t1": rates_t1,
        "rates_t2": rates_t2,
        "tier1": {
            "recall": t1_recall,
            "ceiling": t1_ceil,
            "gap_vs_ceiling": _diff(t1_recall, t1_ceil),
        },
        "tier2": {
            "recall": t2_recall,
            "ceiling": t2_ceil,
            "gap_vs_ceiling": _diff(t2_recall, t2_ceil),
        },
        "fidelity": {
            "tier1_perdoc": _fidelity_summary(tier1_records, "tier1_perdoc"),
            "tier2_heldout": _fidelity_summary(tier2_records, "tier2_heldout"),
        },
    }


# ── Verdict ────────────────────────────────────────────────────────────────────

def verdict(agg):
    n1 = agg["n_tier1"]
    n2 = agg["n_tier2"]
    hs = agg["headline"]
    t1 = agg["tier1"]
    t2 = agg["tier2"]
    fid = agg["fidelity"]

    if n1 < MIN_N:
        return {"status": "UNDERPOWERED",
                "detail": f"n_tier1={n1} < {MIN_N}; raise --synth-n"}

    if t1["ceiling"] is None:
        return {"status": "MISSING_DATA", "detail": "no residual_inject_true scores"}

    # ── Tier 1 ──────────────────────────────────────────────────────────────────
    t1_gap = t1["gap_vs_ceiling"]
    if t1_gap is not None and t1_gap < -GAP:
        # Even per-doc overfit can't bridge → hard kill
        return {
            "status": "TIER1_FAIL_GEOMETRY_UNBRIDGEABLE",
            "detail": (
                f"Per-doc oracle recall ({t1['recall']}) sits {t1_gap} below the "
                f"ceiling ({t1['ceiling']}) — outside the {GAP} tolerance. "
                "Even an overfit per-doc map cannot bridge Qwen→DeepSeek at L12. "
                "HARD KILL: the outsourcing thesis dies here."
            ),
        }

    # Tier 1 passed (expected for short docs; continues to Tier 2)
    if n2 == 0:
        return {
            "status": "TIER1_PASS_TIER2_PENDING",
            "detail": (
                f"Tier 1 ({hs}): recall={t1['recall']} ceiling={t1['ceiling']} "
                f"gap={t1_gap}. "
                "Tier 2 not yet run — re-run with --tier2-only after Tier 1 finishes."
            ),
        }

    # ── Tier 2 ──────────────────────────────────────────────────────────────────
    t2_gap = t2["gap_vs_ceiling"]
    t2_r = t2["recall"]
    t2_c = t2["ceiling"]

    fid2 = fid.get("tier2_heldout") or {}
    cos2 = fid2.get("cos_mean")
    amp_flag = (cos2 is not None and cos2 > AMPLIFICATION_FID_FLOOR
                and (t2_r or 0) < TIER2_FAIL_THRESHOLD * (t2_c or 1.0))

    if t2_r is None:
        status = "TIER2_MISSING"
    elif t2_gap is not None and t2_gap >= -GAP:
        status = "GREEN_LIGHT_TO_2_2"
    elif t2_r is not None and t2_r > TIER2_FAIL_THRESHOLD * (t2_c or 1.0):
        status = "YELLOW_HELDOUT_PARTIAL"   # generalises but not to ceiling
    else:
        status = "YELLOW_HELDOUT_FAIL"      # held-out map doesn't generalise

    if amp_flag:
        status += "_AMPLIFICATION_FLAG"

    return {
        "status": status,
        "detail": (
            f"Tier1 ({hs}): recall={t1['recall']} ceiling={t1['ceiling']} "
            f"gap={t1_gap} | "
            f"Tier2 (n={n2}, {hs}): recall={t2_r} ceiling={t2_c} gap={t2_gap} | "
            f"fid2 cos={cos2}"
        ),
    }


_GLOSS = {
    "TIER1_FAIL_GEOMETRY_UNBRIDGEABLE":
        "   → Even a per-doc overfit map cannot bridge the cross-family geometry at L12.\n"
        "     HARD KILL: Proof 2.2 (learned stitcher) is not worth building for this model pair.",
    "TIER1_PASS_TIER2_PENDING":
        "   → Per-doc oracle passes (expected for short docs; often trivial). "
        "Re-run with --tier2-only to test held-out generalisation.",
    "GREEN_LIGHT_TO_2_2":
        "   → Held-out oracle matches the ceiling within tolerance — the geometry is globally\n"
        "     bridgeable, not just per-doc. GREEN LIGHT to Proof 2.2 (trained SLM stitcher).",
    "YELLOW_HELDOUT_PARTIAL":
        "   → Held-out map generalises partially: recall above the fail threshold but below\n"
        "     the ceiling. The spaces align but imperfectly; 2.2 has a target, though the\n"
        "     stitcher's job is harder than a perfect oracle would imply. Investigate why\n"
        "     the held-out map underperforms the per-doc oracle before training.",
    "YELLOW_HELDOUT_FAIL":
        "   → Held-out map does not generalise: alignment is only per-document, not via a\n"
        "     fixed transformation. YELLOW LIGHT: the cross-family geometry is document-\n"
        "     dependent. Understand why before spending a training run on 2.2.",
    "YELLOW_HELDOUT_FAIL_AMPLIFICATION_FLAG":
        "   → Held-out map fails recall AND shows high fidelity (cos > 0.9): near-correct\n"
        "     residuals still fail after 68 layers of recompute. AMPLIFICATION: the stitcher\n"
        "     will need extreme accuracy. Proof 2.4 should run early.",
    "GREEN_LIGHT_TO_2_2_AMPLIFICATION_FLAG":
        "   → GREEN LIGHT to 2.2, but high-fidelity / low-recall pattern detected —\n"
        "     monitor error amplification carefully in 2.4.",
    "UNDERPOWERED": "   → Too few gated items; raise --synth-n.",
    "MISSING_DATA": "   → No residual_inject_true scores found; cannot compute verdict.",
    "TIER2_MISSING": "   → Tier 2 records missing; run --tier2-only.",
}


# ── Report ─────────────────────────────────────────────────────────────────────

def _fmt(v):
    return "·" if v is None else f"{v:+.3f}"


def report(tier1_records, tier2_records, agg, gate_summary, headline, result):
    layer = result.get("layer", "?")
    think_str = "think-on" if result.get("think") else "think-off"
    print("\n" + "=" * 80)
    print(f"PROOF 2.1 — cross-family geometry (oracle map)  "
          f"(L{layer}, {think_str}, headline={agg['headline']})")
    if gate_summary:
        print(f"  gate: {gate_summary['gated']} gated / {gate_summary['candidates']} "
              f"candidates  (closed-book discard {gate_summary['discard_rate']}, "
              f"A pass {gate_summary['a_pass_rate']})")
    print(f"  eval: n_tier1={agg['n_tier1']}, n_tier2={agg['n_tier2']}")

    # Tier 1 table
    hs = agg["headline"]
    cols_t1 = _present_scorers(tier1_records) if tier1_records else ["strict"]
    print(f"\n  === TIER 1 (per-doc oracle, n={agg['n_tier1']}) ===")
    head = "  " + f"{'condition':<26}" + "".join(f"{c:>10}" for c in cols_t1)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for cond in ["A", "residual_inject_true", "tier1_perdoc"]:
        row = f"  {cond:<26}"
        for c in cols_t1:
            v = agg["rates_t1"].get(cond, {}).get(c)
            row += (f"{v:>10.2f}" if v is not None else f"{'·':>10}")
        print(row)

    t1 = agg["tier1"]
    fid1 = agg["fidelity"]["tier1_perdoc"]
    print(f"\n  Tier 1 headline ({hs}):")
    print(f"    tier1_perdoc − residual_inject_true : {_fmt(t1['gap_vs_ceiling'])}"
          "   ← ≈0 expected (short-doc underdetermined system)")
    print(f"    tier1_perdoc recall                : {t1['recall']}")
    print(f"    ceiling (residual_inject_true)     : {t1['ceiling']}")
    print(f"  Tier 1 fidelity (cos_mean={fid1['cos_mean']}, "
          f"mse_mean={fid1['mse_mean']:.2e}, n={fid1['n']})")
    print("    [expected cos≈1 for short docs; confirms Y_hat=Y_ds for underdetermined maps]")

    # Tier 2 table (if available)
    if tier2_records:
        cols_t2 = _present_scorers(tier2_records)
        print(f"\n  === TIER 2 (held-out oracle, n_test={agg['n_tier2']}) ===")
        head2 = "  " + f"{'condition':<26}" + "".join(f"{c:>10}" for c in cols_t2)
        print(head2)
        print("  " + "-" * (len(head2) - 2))
        for cond in ["A", "residual_inject_true", "tier1_perdoc", "tier2_heldout"]:
            row = f"  {cond:<26}"
            for c in cols_t2:
                v = agg["rates_t2"].get(cond, {}).get(c)
                row += (f"{v:>10.2f}" if v is not None else f"{'·':>10}")
            print(row)

        t2 = agg["tier2"]
        fid2 = agg["fidelity"]["tier2_heldout"]
        print(f"\n  Tier 2 headline ({hs}):")
        print(f"    tier2_heldout − residual_inject_true : {_fmt(t2['gap_vs_ceiling'])}"
              "   ← the generalization test")
        print(f"    tier2_heldout recall                 : {t2['recall']}")
        print(f"    ceiling (residual_inject_true)       : {t2['ceiling']}")
        if fid2:
            print(f"  Tier 2 fidelity (cos_mean={fid2['cos_mean']}, "
                  f"mse_mean={fid2['mse_mean']:.2e}, n={fid2['n']})")
            print("    [gap between cos and recall = early amplification signal for 2.4]")
    else:
        print("\n  === TIER 2 not yet run (use --tier2-only) ===")

    v = verdict(agg)
    print("\n  " + "-" * 76)
    print(f"  VERDICT: {v['status']}")
    print(f"    {v['detail']}")
    if v["status"] in _GLOSS:
        print(_GLOSS[v["status"]])
    return v


# ── Rescore (no GPU) ───────────────────────────────────────────────────────────

def rescore_result(result):
    for r in result.get("tier1_records", []):
        decoys = r.get("decoy_values", [])
        kept_judge = {c: r["scores"][c].get("judge") for c in r["scores"]
                      if "judge" in r["scores"].get(c, {})}
        r["scores"] = {c: score_all(a, r["gold"], decoys)
                       for c, a in r.get("answers", {}).items()}
        for c, jv in kept_judge.items():
            r["scores"][c]["judge"] = jv
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", default=None, metavar="PATH",
                    help="re-score / re-verdict a saved run; no model load")
    ap.add_argument("--tier2-only", action="store_true",
                    help="skip gate+Tier1 (assumes checkpoint exists); run/extend Tier 2 only")
    ap.add_argument("--arm", default="synth_multihop",
                    choices=["synth_multihop", "synth_parity", "hotpot"])
    ap.add_argument("--synth-n", type=int, default=40)
    ap.add_argument("--parity-n", type=int, default=32)
    ap.add_argument("--max-candidates", type=int, default=400, help="hotpot arm only")
    ap.add_argument("--layer", type=int, default=12,
                    help="DeepSeek injection layer (pinned to 12 per chain)")
    ap.add_argument("--qwen-layer", type=int, default=12,
                    help="Qwen residual capture layer (default 12; match DeepSeek's layer)")
    ap.add_argument("--ridge-lambda", type=float, default=1e3,
                    help="ridge regularisation λ for the held-out map. Default 1e3 is mild "
                         "for typical fp16 activation scales; raise to 1e5–1e6 for stronger "
                         "shrinkage if Tier 2 overfits.")
    ap.add_argument("--tier2-test-n", type=int, default=10,
                    help="number of test docs for the held-out oracle (default 10 of 40)")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="suppress reasoning (smoke/wire-test only; operating point is think-ON)")
    ap.set_defaults(think=True)
    ap.add_argument("--no-judge", dest="judge", action="store_false",
                    help="skip inline LLM-judge")
    ap.set_defaults(judge=True)
    ap.add_argument("--headline", default="judge",
                    help="primary scorer for headline/verdict (default judge; "
                         "falls back to strict if judge pass was skipped)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--think-max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-doc-tokens", type=int, default=4096)
    ap.add_argument("--max-eval", type=int, default=None,
                    help="cap gated items evaluated this run (Tier 1; resume-safe)")
    ap.add_argument("--gpus", default="0,1,2",
                    help="logical GPU indices for DeepSeek (select physical GPUs with "
                         "CUDA_VISIBLE_DEVICES; Qwen goes on --qwen-device)")
    ap.add_argument("--qwen-device", default="cuda:0",
                    help="device for Qwen2.5-7B (default cuda:0 — co-locates with "
                         "DeepSeek's first shard; ~33 GB free on each 80 GB H200 "
                         "after DeepSeek-70B sharding, well above Qwen-7B's ~14 GB)")
    
    # Detect available GPUs to adjust defaults dynamically if 1 GPU is selected
    try:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus == 1:
                ap.set_defaults(gpus="0", qwen_device="cuda:0")
    except Exception:
        pass

    ap.add_argument("--max-gpu-memory", default="70GiB",
                    help="maximum memory per GPU (e.g. 70GiB, 140GiB or just 140)")
    ap.add_argument("--gate-cache", default=None,
                    help="path to cache/reuse the gated set (default derived from --out)")
    ap.add_argument("--out", default="proofs/data/p2_1.json")
    args = ap.parse_args()

    max_gpu_mem = args.max_gpu_memory
    if isinstance(max_gpu_mem, str) and max_gpu_mem.isdigit():
        max_gpu_mem = f"{max_gpu_mem}GiB"
    elif isinstance(max_gpu_mem, (int, float)):
        max_gpu_mem = f"{int(max_gpu_mem)}GiB"

    # ── Rescore path (no GPU) ────────────────────────────────────────────────
    if args.rescore:
        with open(args.rescore) as f:
            result = rescore_result(json.load(f))
        t1r = result.get("tier1_records", [])
        t2r = [r for r in t1r if "tier2_heldout" in r.get("answers", {})]
        agg = aggregate(t1r, t2r, args.headline)
        v = report(t1r, t2r, agg, result.get("gate_summary"), agg["headline"], result)
        result["aggregate"], result["verdict"] = agg, v
        with open(args.rescore, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nRe-scored → {args.rescore}")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gate_cache = args.gate_cache or args.out.replace(".json", f"_gated_{args.arm}.json")
    states_path = _states_path(args.out)

    # ── Load gate ────────────────────────────────────────────────────────────
    need_gate = not os.path.exists(gate_cache)
    recs_pre = None
    if need_gate:
        recs_pre = load_candidates(args.arm, args)
        print(f"[gate] resolved {len(recs_pre)} {args.arm} candidates")

    # ── Resume existing checkpoint ──────────────────────────────────────────
    result = {}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                result = json.load(f)
            print(f"[resume] loaded checkpoint from {args.out} "
                  f"({len(result.get('tier1_records', []))} tier1 records)")
        except Exception as e:
            print(f"[resume] could not read {args.out}: {e}")

    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))

    # ── Gate (needs DeepSeek) ────────────────────────────────────────────────
    if not args.tier2_only:
        print(f"Loading DeepSeek-70B across GPUs {devices} (max_memory={max_gpu_mem}) …")
        ds_tok, ds_model = load_deepseek(cfg, devices=devices, max_memory_per_gpu=max_gpu_mem)

        if not need_gate:
            with open(gate_cache) as f:
                cached = json.load(f)
            gated, gate_summary = cached["gated"], cached["summary"]
            print(f"[gate] loaded {len(gated)} gated items from {gate_cache}")
        else:
            print(f"[gate] gating {len(recs_pre)} candidates …")
            gated, gate_summary = run_gate(ds_model, ds_tok, recs_pre, args)
            with open(gate_cache, "w") as f:
                json.dump({"gated": gated, "summary": gate_summary}, f, default=str)
            print(f"[gate] cached → {gate_cache}")

        if not gated:
            print("No gated items; nothing to evaluate.")
            return

        # ── Tier 1 (needs Qwen + DeepSeek) ──────────────────────────────────
        print(f"\nLoading Qwen on {args.qwen_device} …")
        qw_tok, qw_model = load_qwen(cfg, args.qwen_device)

        tier1_records, states = run_tier1(
            ds_model, ds_tok, qw_model, qw_tok, gated, args,
            resume_path=args.out, states_path=states_path)

        # Free Qwen (not needed for Tier 2)
        del qw_model, qw_tok
        gc.collect()
        _free_cuda()
        print("[tier1] Qwen freed")

        result["tier1_records"] = tier1_records
        result["gate_summary"] = gate_summary
        result["arm"] = args.arm
        result["layer"] = args.layer
        result["think"] = args.think

    else:
        # Tier-2-only: load gate + states from checkpoint
        if not os.path.exists(args.out):
            raise SystemExit(f"--tier2-only requires an existing checkpoint at {args.out}")
        tier1_records = result.get("tier1_records", [])
        gate_summary = result.get("gate_summary")

        if not os.path.exists(gate_cache):
            raise SystemExit(
                f"Gate cache not found at {gate_cache}; "
                "run Tier 1 first (without --tier2-only).")
        with open(gate_cache) as f:
            cached = json.load(f)
        gated = cached["gated"]

        states = load_states(states_path)
        if not states:
            raise SystemExit(
                f"States file not found at {states_path}; "
                "Tier 1 must complete before Tier 2.")
        print(f"[tier2-only] {len(tier1_records)} tier1 records, "
              f"{len(states)} state entries loaded")

        print(f"Loading DeepSeek-70B across GPUs {devices} (max_memory={max_gpu_mem}) …")
        ds_tok, ds_model = load_deepseek(cfg, devices=devices, max_memory_per_gpu=max_gpu_mem)

    # ── Tier 2 (needs DeepSeek) ──────────────────────────────────────────────
    tier2_records = run_tier2(
        ds_model, ds_tok, gated, result["tier1_records"], states, args,
        resume_path=args.out)

    # ── Final report + save ──────────────────────────────────────────────────
    t1r = result["tier1_records"]
    agg = aggregate(t1r, tier2_records, args.headline)
    v = report(t1r, tier2_records, agg, result.get("gate_summary"),
               agg["headline"], result)
    result["aggregate"] = agg
    result["verdict"] = v

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {args.out}")
    save_states(states_path, states)
    print(f"States → {states_path}")


if __name__ == "__main__":
    main()

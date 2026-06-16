"""
core/selftest_split_forward.py — CPU pre-flight for the split-forward mechanism.

Proof 0 is the behavioural certificate of the harness, but it needs the 70B. This
self-test certifies the *mechanics* — RoPE offsetting, the causal/doc-visible
masks, the per-layer KV cache across the lower/upper split, and the decode loop —
on a tiny random Llama in a few seconds on CPU. Run it before spending H200 time;
if it fails, the bug is in `split_forward.py`, not in the model or the data.

Two exact invariants (greedy decode must match token-for-token):

  A · no-document reduction.
      With an empty document (n_doc = 0), the split-forward is just an ordinary
      forward of the query. It must equal `model.generate(query)`. Exercises the
      full manual pipeline (all layers, both stages, decode) with no document.

  C · prefix-cache identity (the offset-RoPE / doc-visible path).
      Build the document cache WITHOUT clearing the lower layers, so the document
      is present at every layer, and run the query with position_ids offset to
      N..N+M-1. This is exactly ordinary prefix-cached inference, so it must equal
      `model.generate([doc; query])`. Exercises the hard part: the query reading a
      cached prefix at offset positions, with the cache split across the model's
      layer range and grown one token at a time during decode.

The real split-forward differs from C only by clearing the lower-layer document
cache (document absent below `target_layer`) — strictly *less* attention, no new
RoPE/cache machinery. C passing plus Proof 0 passing pins the mechanism down.

    python core/selftest_split_forward.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.split_forward import (
    capture_doc_cache, split_forward_generate, subset_doc_cache,
)


def _tiny_model(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,      # exercise grouped-query attention
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        eos_token_id=None,          # never stop early → deterministic length
        pad_token_id=0,
    )
    model = LlamaForCausalLM(cfg).eval().to(torch.float32)
    return model


@torch.no_grad()
def _ref_generate(model, input_ids, n_new):
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=n_new,
        do_sample=False,
        use_cache=True,
        pad_token_id=0,
    )
    return out[0, input_ids.shape[1]:].tolist()


def _fake_tok():
    return SimpleNamespace(eos_token_id=None)


def run(target_layer=3, n_new=12, seed=0, verbose=True):
    model = _tiny_model(seed)
    tok = _fake_tok()
    g = torch.Generator().manual_seed(123)
    doc_ids = torch.randint(1, 128, (1, 17), generator=g)
    q_ids = torch.randint(1, 128, (1, 5), generator=g)

    ok = True

    # ── Invariant A — no-document reduction ──────────────────────────────────
    ref_a = _ref_generate(model, q_ids, n_new)
    empty_cache, _, _ = capture_doc_cache(model, doc_ids, target_layer)
    # clear ALL layers so the document is fully absent; n_doc = 0
    from core.split_forward import _clear_cache_layers
    _clear_cache_layers(empty_cache, range(0, len(model.model.layers)))
    got_a = split_forward_generate(
        model, tok, empty_cache, n_doc=0, query_ids=q_ids,
        target_layer=target_layer, max_new_tokens=n_new,
        clear_lower=True, return_ids=True,
    )
    pass_a = got_a == ref_a
    ok &= pass_a
    if verbose:
        print(f"[A] no-doc reduction        : {'PASS' if pass_a else 'FAIL'}")
        if not pass_a:
            print(f"    ref={ref_a}\n    got={got_a}")

    # ── Invariant C — prefix-cache identity (offset RoPE, doc visible) ────────
    joint = torch.cat([doc_ids, q_ids], dim=1)
    ref_c = _ref_generate(model, joint, n_new)
    full_cache, _, N = capture_doc_cache(model, doc_ids, target_layer, clear_lower=False)
    got_c = split_forward_generate(
        model, tok, full_cache, n_doc=N, query_ids=q_ids,
        target_layer=target_layer, max_new_tokens=n_new,
        clear_lower=False, return_ids=True,
    )
    pass_c = got_c == ref_c
    ok &= pass_c
    if verbose:
        print(f"[C] prefix-cache identity   : {'PASS' if pass_c else 'FAIL'}")
        if not pass_c:
            print(f"    ref={ref_c}\n    got={got_c}")

    # ── Invariant D — subset-to-all identity (Proof 3 plumbing) ───────────────
    # Subsetting the document cache to ALL of its positions must be a no-op: it
    # exercises the slice + the n_doc_cached-vs-n_doc decoupling (mask width,
    # physical cache positions) while leaving every key untouched, so it must
    # reproduce the real split-forward token-for-token. This certifies the
    # mechanics that Proof 3's needle subsets rely on; the choice of WHICH
    # positions to keep is a science question for the 70B, not this test.
    real_cache, _, N2 = capture_doc_cache(model, doc_ids, target_layer, clear_lower=True)
    got_real = split_forward_generate(
        model, tok, real_cache, n_doc=N2, query_ids=q_ids,
        target_layer=target_layer, max_new_tokens=n_new,
        clear_lower=True, return_ids=True,
    )
    all_positions = list(range(N2))
    sub_cache = subset_doc_cache(real_cache, all_positions)
    got_sub = split_forward_generate(
        model, tok, sub_cache, n_doc=N2, n_doc_cached=len(all_positions),
        query_ids=q_ids, target_layer=target_layer, max_new_tokens=n_new,
        clear_lower=True, return_ids=True,
    )
    pass_d = got_sub == got_real
    ok &= pass_d
    if verbose:
        print(f"[D] subset-to-all identity  : {'PASS' if pass_d else 'FAIL'}")
        if not pass_d:
            print(f"    real={got_real}\n    sub ={got_sub}")

    if verbose:
        print("=" * 48)
        print("ALL PASS — mechanism is trustworthy" if ok
              else "FAILURE — fix split_forward.py before any H200 run")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)

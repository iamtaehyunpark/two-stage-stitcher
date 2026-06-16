"""
core/split_forward.py — the two-cache split-forward mechanism (the keystone).

This is the single component every proof in the chain imports. Its job: let a
frozen decoder-only LLM reason over a document it never tokenized, by handing the
document to the model as *true layer-`target_layer` states* and running only the
query through the early layers.

────────────────────────────────────────────────────────────────────────────────
The mechanism
────────────────────────────────────────────────────────────────────────────────
Positions:                                                  exists at layers …
    document tokens   d_0 … d_{N-1}   → positions 0 … N-1        target_layer … L-1   (injected, never run through 0..29)
    query tokens      q_0 … q_{M-1}   → positions N … N+M-1      0 … L-1
    generated token k                 → position N+M+k           0 … L-1

The document is absent from the lower stack (layers 0 … target_layer-1) entirely —
that is the whole point: we never pay to read it there. It is present only from
`target_layer` upward, as its *true* representations. The query runs through the
lower stack on its own, **with position_ids offset to N…N+M-1 from the start** so
RoPE places it after the (absent-but-positionally-reserved) document. At
`target_layer` the two streams join and run together to the top.

This means the KV cache is NOT uniform across layers — lower layers hold only
{query, generated}; upper layers hold {document, query, generated}. A single
`generate()` call cannot express that (it assumes one cache length and one set of
position_ids for all layers), which is exactly why this is hand-rolled.

────────────────────────────────────────────────────────────────────────────────
How the document's upper-layer states are obtained (and why it's equivalent to
"inject Y_doc at layer 30 and re-run 30→L")
────────────────────────────────────────────────────────────────────────────────
We run one full document-only forward (layers 0 … L-1, `use_cache=True`) and keep
the resulting KV cache, then **clear the lower layers** so the document survives
only at `target_layer … L-1`. Because the document is causal-first (the query is
always positionally after it), the document's keys/values at the upper layers are
identical whether computed by this full forward or by injecting the captured
layer-`target_layer` input states and re-running just the upper stack — the
document never attends to the query in either case. We take the full-forward route
because it fills the cache in strict layer order (0→L-1), sidestepping every
`DynamicCache` layer-index edge case, and it costs one cheap prefill done once.

The captured layer-`target_layer` *input* states (`Y_doc`) are also returned, for
diagnostics and for the parts of the chain (e.g. Proof 6) that inject a single
produced vector rather than a true cache.

────────────────────────────────────────────────────────────────────────────────
Robustness notes
────────────────────────────────────────────────────────────────────────────────
* Works on a model sharded across GPUs via `device_map`: every per-layer call
  moves hidden / mask / position bookkeeping to that layer's device. Harmless on a
  single device.
* Decoder-layer forward signatures drift across transformers versions (the
  `position_embeddings` kwarg, `past_key_value` vs `past_key_values`). We inspect
  the signature once and pass only what it accepts; if RoPE is not pre-computable
  on `model.model.rotary_emb`, we fall back to letting attention derive it from
  `position_ids`.
* Validate the mechanism on CPU with a tiny random model before spending H200 time:
  `python core/selftest_split_forward.py`.
"""

from __future__ import annotations

import copy
import inspect
from typing import Optional, Tuple

import torch


# ── cache construction ────────────────────────────────────────────────────────

def _new_dynamic_cache():
    try:
        from transformers import DynamicCache
    except ImportError:
        from transformers.cache_utils import DynamicCache
    return DynamicCache()


# DynamicCache's internals changed across transformers versions: the legacy API
# exposes `key_cache` / `value_cache` lists of tensors; the refactored API
# (≥ 4.54) stores `layers[i].keys` / `layers[i].values` on per-layer objects.
# These accessors hide the difference so the split-forward works on both.

def _kv_attr_names(layer_obj):
    for kk, vv in (("keys", "values"), ("key_states", "value_states"),
                   ("key_cache", "value_cache")):
        if hasattr(layer_obj, kk) and hasattr(layer_obj, vv):
            return kk, vv
    return None


def _cache_num_layers(cache) -> int:
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    if hasattr(cache, "layers"):
        return len(cache.layers)
    raise AttributeError(
        "Unrecognised DynamicCache layout (no `key_cache` and no `layers`). "
        "Report your transformers version so split_forward.py can support it."
    )


def _cache_get(cache, i):
    """Return (keys, values) tensors for layer i, or (None, None)."""
    if hasattr(cache, "key_cache"):
        if i < len(cache.key_cache):
            return cache.key_cache[i], cache.value_cache[i]
        return None, None
    layer = cache.layers[i]
    names = _kv_attr_names(layer)
    if names is None:
        raise AttributeError(
            f"Cannot find key/value attributes on {type(layer).__name__}. "
            "Report your transformers version so split_forward.py can support it."
        )
    return getattr(layer, names[0]), getattr(layer, names[1])


def _cache_set(cache, i, k, v):
    if hasattr(cache, "key_cache"):
        cache.key_cache[i] = k
        cache.value_cache[i] = v
        return
    layer = cache.layers[i]
    names = _kv_attr_names(layer)
    setattr(layer, names[0], k)
    setattr(layer, names[1], v)


def _clone_cache(cache):
    """Deep-copy the document cache so a generation call can append query /
    generated KV without mutating the caller's cache (reused across many
    questions). `copy.deepcopy` clones every tensor on its own device and is
    agnostic to the DynamicCache internal layout."""
    return copy.deepcopy(cache)


def capture_doc_cache(model, doc_ids: torch.Tensor, target_layer: int,
                      clear_lower: bool = True):
    """
    Run one full document-only forward and return everything the split-forward
    needs to treat the document as already-read:

      doc_cache : a DynamicCache whose layers `target_layer … L-1` hold the
                  document's TRUE keys/values. With `clear_lower=True` (the real
                  split-forward) the lower layers are cleared so the document is
                  absent below `target_layer`; with `clear_lower=False` the
                  document is kept at every layer (used only by the self-test's
                  prefix-cache identity check).
      Y_doc     : (1, N, D) the true hidden states ENTERING `target_layer`.
      N         : document length in tokens.

    `doc_ids` is (1, N). The forward uses default position_ids 0 … N-1.
    """
    embed_device = model.model.embed_tokens.weight.device
    doc_ids = doc_ids.to(embed_device)

    captured = {}

    def cap_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        captured["y"] = hs.detach()
        return None

    handle = model.model.layers[target_layer].register_forward_pre_hook(
        cap_hook, with_kwargs=True
    )
    cache = _new_dynamic_cache()
    try:
        with torch.no_grad():
            model(
                input_ids=doc_ids,
                attention_mask=torch.ones_like(doc_ids),
                past_key_values=cache,
                use_cache=True,
            )
    finally:
        handle.remove()

    Y_doc = captured["y"]                 # (1, N, D) on target_layer's device
    N = int(doc_ids.shape[1])
    if clear_lower:
        _clear_cache_layers(cache, range(0, target_layer))
    return cache, Y_doc, N


def _clear_cache_layers(cache, layer_indices):
    """Truncate the given layers' KV to length 0, preserving shape metadata."""
    n = _cache_num_layers(cache)
    for i in layer_indices:
        if i >= n:
            continue
        k, v = _cache_get(cache, i)
        if k is not None and k.numel() > 0:
            _cache_set(cache, i, k[:, :, :0, :], v[:, :, :0, :])


def subset_doc_cache(cache, positions):
    """
    Return a deep copy of `cache` keeping only the document key/value pairs at the
    given ORIGINAL `positions` (a list/iterable of int indices into 0..N-1) at every
    populated layer. Already-cleared (empty) layers are left empty.

    Why this preserves the experiment (Proof 3's central correctness condition):
    in Llama-family attention RoPE is applied to the keys *before* they are written
    to the KV cache, so a cached key at original position p carries the rotation for
    p baked in. Slicing the sequence dimension therefore keeps each surviving
    position's ABSOLUTE-position encoding intact — positions are NOT renumbered. The
    query, still asked from offset N (the full document length), sees each needle at
    its true relative distance (N − p), exactly as in the all-N forward. Compacting
    the needles to 0,1,2,… and re-deriving RoPE would corrupt every query→needle
    distance; this function deliberately does not do that.

    `positions` are sorted and de-duplicated so the kept keys stay in causal order;
    the model never attends in a different order than it would have over the full
    document. The caller is responsible for telling `split_forward_generate` the new
    cached count via `n_doc_cached=len(set(positions))` while keeping `n_doc` at the
    full document length so the query offset is unchanged.
    """
    out = _clone_cache(cache)
    idx = sorted(set(int(p) for p in positions))
    n = _cache_num_layers(out)
    for i in range(n):
        k, v = _cache_get(out, i)
        if k is None or k.numel() == 0:
            continue
        sel = torch.tensor(idx, dtype=torch.long, device=k.device)
        _cache_set(out, i, k.index_select(2, sel), v.index_select(2, sel))
    return out


# ── layer-call plumbing (signature- and device-robust) ───────────────────────

def _layer_param_names(model):
    return set(inspect.signature(model.model.layers[0].forward).parameters.keys())


def _supports_position_embeddings(model) -> bool:
    return "position_embeddings" in _layer_param_names(model) \
        and getattr(model.model, "rotary_emb", None) is not None


def _rope(model, ref_hidden: torch.Tensor, position_ids: torch.Tensor):
    """Pre-compute (cos, sin) for `position_ids`, or None if unavailable."""
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        return None
    pos = position_ids.to(ref_hidden.device)
    try:
        return rotary(ref_hidden, pos)
    except TypeError:
        # very old signature: rotary(x, seq_len=...)
        return None


def _to(x, device):
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(_to(t, device) for t in x)
    return x.to(device)


def _run_stage(model, layer_range, hidden, attn_mask, position_ids,
               cache, cache_position, position_embeddings, sig):
    """Run `hidden` through `layer_range`, threading `cache`. Returns the output
    hidden states (input to the next stage / final norm)."""
    layers = model.model.layers
    for idx in layer_range:
        layer = layers[idx]
        dev = next(layer.parameters()).device
        h = _to(hidden, dev)
        kwargs = {}
        if "attention_mask" in sig:
            kwargs["attention_mask"] = _to(attn_mask, dev)
        if "position_ids" in sig:
            kwargs["position_ids"] = _to(position_ids, dev)
        if "past_key_value" in sig:
            kwargs["past_key_value"] = cache
        elif "past_key_values" in sig:
            kwargs["past_key_values"] = cache
        if "use_cache" in sig:
            kwargs["use_cache"] = True
        if "cache_position" in sig and cache_position is not None:
            kwargs["cache_position"] = _to(cache_position, dev)
        if "position_embeddings" in sig and position_embeddings is not None:
            kwargs["position_embeddings"] = _to(position_embeddings, dev)
        out = layer(h, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


def _final_logits(model, hidden_last: torch.Tensor) -> torch.Tensor:
    """norm + lm_head on a single (1, 1, D) hidden state → (1, vocab)."""
    norm = model.model.norm
    h = _to(hidden_last, next(norm.parameters()).device)
    h = norm(h)
    head = model.lm_head
    h = _to(h, next(head.parameters()).device)
    return head(h)[:, -1, :]


# ── attention masks (additive, model dtype) ──────────────────────────────────

def _min(dtype):
    return torch.finfo(dtype).min


def _causal_mask(q_len, dtype, device):
    """Standard lower-triangular causal mask, (1,1,q,q)."""
    m = torch.full((q_len, q_len), _min(dtype), dtype=dtype, device=device)
    m = torch.triu(m, diagonal=1)
    return m[None, None, :, :]


def _doc_query_mask(n_doc, q_len, dtype, device):
    """Mask for the upper-stack query prefill: q_len query rows over
    (n_doc + q_len) keys. Document keys (first n_doc) are fully visible; query
    keys are causal among themselves. (1,1,q_len,n_doc+q_len)."""
    kv = n_doc + q_len
    m = torch.zeros((q_len, kv), dtype=dtype, device=device)
    # causal block over the query portion (columns n_doc .. n_doc+q_len-1)
    qblock = torch.full((q_len, q_len), _min(dtype), dtype=dtype, device=device)
    qblock = torch.triu(qblock, diagonal=1)
    m[:, n_doc:] = qblock
    return m[None, None, :, :]


# ── the split-forward generator ──────────────────────────────────────────────

@torch.no_grad()
def split_forward_generate(
    model,
    tokenizer,
    doc_cache,
    n_doc: int,
    query_text: Optional[str] = None,
    target_layer: int = 30,
    max_new_tokens: int = 256,
    clear_lower: bool = True,
    query_ids: Optional[torch.Tensor] = None,
    return_ids: bool = False,
    n_doc_cached: Optional[int] = None,
):
    """
    Generate an answer given a document supplied as `doc_cache` (from
    `capture_doc_cache`). Greedy decode.

    Provide the query as `query_text` (tokenized here) or as `query_ids` (1, M),
    which bypasses the tokenizer — the self-test uses the latter so token-boundary
    effects can't perturb an exact comparison.

    `clear_lower` must match how `doc_cache` was built: True for the real
    split-forward (document absent below `target_layer`), False for the
    prefix-cache identity check (document present at every layer).

    `n_doc` is the document's FULL length: it sets the query's absolute position
    offset (RoPE), so the query is always asked from N…N+M-1 regardless of how many
    document keys actually remain in the cache. `n_doc_cached` is the number of
    document keys physically present in `doc_cache` at the upper layers — equal to
    `n_doc` for a full cache, but smaller when a subset was kept (Proof 3, via
    `subset_doc_cache`). It sizes the doc-visible attention mask and the physical
    cache positions. Decoupling the two is what lets a sparse handoff keep its
    needles at their original positions while the query still attends from the
    right place. Defaults to `n_doc` (the full-cache case).

    Returns the decoded string, or the list of generated token ids if
    `return_ids=True`.
    """
    # The caller's document cache is reused across questions; never mutate it.
    doc_cache = _clone_cache(doc_cache)

    if n_doc_cached is None:
        n_doc_cached = n_doc

    L = len(model.model.layers)
    dtype = model.model.embed_tokens.weight.dtype
    embed_device = model.model.embed_tokens.weight.device
    sig = _layer_param_names(model)
    use_pe = _supports_position_embeddings(model)
    eos_id = tokenizer.eos_token_id

    # The document occupies `n_doc_cached` key slots at the upper layers; `clear_lower`
    # only decides whether it also occupies the lower-layer cache.
    lower_doc_len = 0 if clear_lower else n_doc_cached

    # ── query prefill ────────────────────────────────────────────────────────
    if query_ids is None:
        query_ids = tokenizer(query_text, return_tensors="pt").input_ids
    q_ids = query_ids.to(embed_device)
    M = int(q_ids.shape[1])
    hidden = model.model.embed_tokens(q_ids)                     # (1, M, D)

    # RoPE/position bookkeeping: absolute query positions are n_doc..n_doc+M-1.
    # This is the FULL-document offset and is independent of how many document keys
    # remain cached — a sparse needle handoff is still queried from position N.
    pos_q = torch.arange(n_doc, n_doc + M, device=embed_device)[None, :]
    pe_q = _rope(model, hidden, pos_q) if use_pe else None

    # Lower stack: cache holds only {query} (length lower_doc_len before write).
    lower_mask = (_doc_query_mask(lower_doc_len, M, dtype, embed_device)
                  if lower_doc_len else _causal_mask(M, dtype, embed_device))
    lower_cache_pos = torch.arange(lower_doc_len, lower_doc_len + M, device=embed_device)
    hidden = _run_stage(model, range(0, target_layer), hidden, lower_mask,
                        pos_q, doc_cache, lower_cache_pos, pe_q, sig)

    # Upper stack: cache holds {document(n_doc_cached)} then appends the query. The
    # mask width and physical cache positions follow the CACHED count, not n_doc.
    upper_mask = _doc_query_mask(n_doc_cached, M, dtype, embed_device)
    upper_cache_pos = torch.arange(n_doc_cached, n_doc_cached + M, device=embed_device)
    hidden = _run_stage(model, range(target_layer, L), hidden, upper_mask,
                        pos_q, doc_cache, upper_cache_pos, pe_q, sig)

    logits = _final_logits(model, hidden[:, -1:, :])            # (1, vocab)
    next_id = int(logits.argmax(dim=-1).item())

    generated = [next_id]
    lower_len = lower_doc_len + M
    upper_len = n_doc_cached + M
    pos = n_doc + M

    # ── decode loop (single token at a time; cache lengths differ per stage) ──
    for _ in range(max_new_tokens - 1):
        if next_id == eos_id:
            break
        tok = torch.tensor([[next_id]], device=embed_device)
        hidden = model.model.embed_tokens(tok)                  # (1, 1, D)
        pos_ids = torch.tensor([[pos]], device=embed_device)
        pe = _rope(model, hidden, pos_ids) if use_pe else None

        # single query token attends to everything cached (mask=None ⇒ q_len==1).
        hidden = _run_stage(model, range(0, target_layer), hidden, None, pos_ids,
                            doc_cache, torch.tensor([lower_len], device=embed_device),
                            pe, sig)
        hidden = _run_stage(model, range(target_layer, L), hidden, None, pos_ids,
                            doc_cache, torch.tensor([upper_len], device=embed_device),
                            pe, sig)

        logits = _final_logits(model, hidden)
        next_id = int(logits.argmax(dim=-1).item())
        generated.append(next_id)
        lower_len += 1
        upper_len += 1
        pos += 1

    if eos_id is not None and generated and generated[-1] == eos_id:
        generated = generated[:-1]
    if return_ids:
        return generated
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

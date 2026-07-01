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


# ── RoPE re-rotation (the renumbered control) ─────────────────────────────────
# The default subset keeps each needle at its ORIGINAL position (its baked-in RoPE
# phase is untouched). The renumbered control deliberately does the opposite: it
# moves kept keys to NEW positions to isolate the position effect from the semantic
# one. Because RoPE is a rotation R(p) applied to the key at position p, a key
# already carrying R(p) can be re-placed at position j by applying the *relative*
# rotation R(j − p): R(j−p)·R(p) = R(j). We apply exactly that delta rotation to the
# cached (already-rotated) keys; values are not rotated, so they are left alone.
#
# Assumption: the rotary embedding's `attention_scaling` is 1.0, so cos/sin are a
# pure rotation and angles add (ω_d·j − ω_d·p = ω_d·(j−p)) — true for Llama-3.x
# "default"/"llama3" rope (the project's DeepSeek-R1-Distill-Llama-70B). On a
# "yarn"/"longrope" model where attention_scaling ≠ 1 the delta would re-apply the
# magnitude factor and would need to be divided back out. The self-test's
# translation-invariance invariant (F) certifies the rotation math on a default-rope
# model.

def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rerotate_keys(model, key, deltas):
    """Apply the relative RoPE rotation `deltas` (1-D, one signed offset per key
    position) to an already-rotated key tensor (1, n_kv_heads, S, head_dim)."""
    cs = _rope(model, key, deltas[None, :].to(key.device))
    if cs is None:
        raise RuntimeError(
            "RoPE re-rotation needs a position-id-driven rotary embedding "
            "(model.model.rotary_emb(x, position_ids)); this model/transformers "
            "version does not expose one. The renumbered control cannot run."
        )
    cos, sin = cs
    # cos/sin arrive as (1, S, head_dim); add the head axis to broadcast over heads.
    # On a sharded model the rotary may emit them on its own device, so pin both to
    # the key's device/dtype before the multiply.
    cos = cos.to(device=key.device, dtype=key.dtype).unsqueeze(1)
    sin = sin.to(device=key.device, dtype=key.dtype).unsqueeze(1)
    return key * cos + _rotate_half(key) * sin


def rerotate_cache_keys(model, cache, deltas):
    """In place over a (cloned) cache: re-rotate every populated layer's keys by the
    per-position `deltas` (a 1-D long/float tensor whose length equals the cache's
    current sequence length). Values are untouched."""
    n = _cache_num_layers(cache)
    for i in range(n):
        k, v = _cache_get(cache, i)
        if k is None or k.numel() == 0:
            continue
        if k.shape[2] != deltas.shape[0]:
            raise ValueError(
                f"deltas length {deltas.shape[0]} != layer {i} cache seq "
                f"length {k.shape[2]}"
            )
        _cache_set(cache, i, _rerotate_keys(model, k, deltas), v)


def subset_doc_cache_renumbered(model, cache, positions, target=None):
    """The renumbered control: keep `positions` (original indices) and then move
    their keys to `target` positions by RoPE delta-rotation. With `target=None` the
    kept positions are compacted to a contiguous 0,1,2,… block (the natural
    "renumbered" placement the text arm gets for free). Returns
    `(new_cache, n_doc_renumbered)` where `n_doc_renumbered = max(target)+1` is the
    query offset the caller should pass as `n_doc` (the query sits just past the
    compacted document).

    This is what proves the position bookkeeping is load-bearing: feed the SAME true
    states the default subset feeds, but at renumbered positions. If recall survives
    the subset and dies here, decimation per se was fine and renumbering was the
    killer (Proof 0's relative-distance geometry, violated on purpose).
    """
    pos = sorted(set(int(p) for p in positions))
    if target is None:
        target = list(range(len(pos)))
    if len(target) != len(pos):
        raise ValueError("target must have one position per kept index")
    sub = subset_doc_cache(cache, pos)        # keys still at original phases, sorted
    deltas = torch.tensor([t - p for p, t in zip(pos, target)], dtype=torch.float32)
    rerotate_cache_keys(model, sub, deltas)
    return sub, int(max(target)) + 1


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


# ── Proof 2.0: residual-inject (recompute the upper cache from the layer residual) ──
# The Chain-1 handoff object is the captured KV cache (layers target_layer..L-1). An SLM
# can realistically produce only the *layer-target_layer residual stream* Y_doc — one
# (1, N, D) tensor — and let the LLM recompute the upper stack from it. This module's
# header (lines 30-45) ASSERTS the two are equivalent for true states, because the
# document is causal-first; it never implements or tests the second route. Proof 2.0 does.
#
# `recompute_doc_cache_from_residual` builds the residual-inject cache using ONLY
# operations already proven elsewhere in this file: `_clone_cache` for a correctly-shaped
# L-layer container, `_clear_cache_layers` to empty every layer (so not one stored key
# survives — the recompute cannot secretly read the reference and pass as a hybrid), and
# `_run_stage` to fill the upper layers by running Y_doc through them — the exact
# concatenate-into-cache path `split_forward_generate` uses to append the query. This
# sidesteps the DynamicCache first-write-at-a-nonzero-layer edge case (the reason
# `capture_doc_cache` takes the full-forward route) without depending on it.

@torch.no_grad()
def recompute_doc_cache_from_residual(model, Y_doc: torch.Tensor, n_doc: int,
                                      target_layer: int, structure_cache):
    """
    Build a document cache whose upper layers `target_layer … L-1` are RECOMPUTED from the
    layer-`target_layer` residual stream `Y_doc` (1, N, D) alone — the small object an SLM
    would produce — rather than captured from a full document forward
    (`capture_doc_cache`, the cache-inject reference). Lower layers stay empty, exactly as
    `clear_lower=True`.

      Y_doc           : (1, N, D) the TRUE hidden states ENTERING `target_layer` — the
                        `Y_doc` that `capture_doc_cache` already returns for the same doc.
      n_doc           : N, the document length (must equal Y_doc's sequence length).
      structure_cache : any DynamicCache with all L layers materialised for this doc —
                        pass the cache-inject cache; it is deep-copied and fully cleared,
                        so its stored keys/values do NOT leak into the recompute (they are
                        only borrowed for their per-layer shape metadata).

    The document is run through the upper stack as N causal tokens at positions 0…N-1 —
    the same positions and default RoPE `capture_doc_cache`'s forward used — so each upper
    layer computes and caches its own K/V from Y_doc flowing upward. Because the document
    never attends to the (positionally-later) query, these recomputed upper K/V are
    identical to the stored ones in theory; Proof 2.0 measures whether they are in fact,
    and self-test invariant G proves the plumbing token-for-token on CPU.
    """
    L = len(model.model.layers)
    N = int(n_doc)
    if int(Y_doc.shape[1]) != N:
        raise ValueError(f"Y_doc length {int(Y_doc.shape[1])} != n_doc {N}")
    dtype = model.model.embed_tokens.weight.dtype
    sig = _layer_param_names(model)
    use_pe = _supports_position_embeddings(model)

    # Correctly-shaped, fully-empty L-layer container (borrow structure, drop all KV).
    cache = _clone_cache(structure_cache)
    _clear_cache_layers(cache, range(0, L))
    # Guard against the "secretly a hybrid" failure: every upper layer must be empty
    # BEFORE the recompute, or the query would attend to leftover stored keys and the
    # residual path would pass for the wrong reason.
    for i in range(target_layer, L):
        k, _v = _cache_get(cache, i)
        if k is not None and k.shape[2] != 0:
            raise RuntimeError(
                f"residual recompute: layer {i} not cleared (len {k.shape[2]}); the "
                "recompute would read stored KV — a hybrid, not a true recompute.")

    dev0 = Y_doc.device
    pos = torch.arange(0, N, device=dev0)[None, :]
    pe = _rope(model, Y_doc, pos) if use_pe else None
    mask = _causal_mask(N, dtype, dev0)                 # doc tokens: causal among themselves
    cache_pos = torch.arange(0, N, device=dev0)
    _run_stage(model, range(target_layer, L), Y_doc, mask, pos, cache, cache_pos, pe, sig)

    # Guard: the recompute actually populated the upper layers with all N keys.
    for i in range(target_layer, L):
        k, _v = _cache_get(cache, i)
        if k is None or k.shape[2] != N:
            got = None if k is None else k.shape[2]
            raise RuntimeError(
                f"residual recompute: layer {i} holds {got} keys, expected {N}; the "
                "upper-stack recompute did not fill the cache.")
    return cache


@torch.no_grad()
def kv_drift_upper(cache_ref, cache_test, target_layer: int):
    """Proof 2.0 diagnostic (#4): how far the recomputed upper K/V (`cache_test`) sit from
    the stored ones (`cache_ref`), per layer `target_layer … L-1`. Returns mean cosine
    similarity and mean squared error over K and V, averaged across the populated upper
    layers. cos ≈ 1.0 / MSE ≈ 0 ⇒ the residual determines the cache, so any behavioural
    gap is NOT numerical drift; a large MSE alongside a behavioural gap points at a
    plumbing/precision bug instead of a structural fact about the handoff object."""
    n = _cache_num_layers(cache_ref)
    cos_k = cos_v = mse_k = mse_v = 0.0
    cnt = 0
    for i in range(target_layer, n):
        kr, vr = _cache_get(cache_ref, i)
        kt, vt = _cache_get(cache_test, i)
        if kr is None or kt is None or kr.numel() == 0 or kt.numel() == 0:
            continue
        kt = kt.to(kr.device)
        vt = vt.to(vr.device)
        cos_k += torch.nn.functional.cosine_similarity(
            kr.float().reshape(-1), kt.float().reshape(-1), dim=0).item()
        cos_v += torch.nn.functional.cosine_similarity(
            vr.float().reshape(-1), vt.float().reshape(-1), dim=0).item()
        mse_k += torch.mean((kr.float() - kt.float()) ** 2).item()
        mse_v += torch.mean((vr.float() - vt.float()) ** 2).item()
        cnt += 1
    if cnt == 0:
        return {"cos_k": None, "cos_v": None, "mse_k": None, "mse_v": None, "layers": 0}
    return {"cos_k": round(cos_k / cnt, 6), "cos_v": round(cos_v / cnt, 6),
            "mse_k": mse_k / cnt, "mse_v": mse_v / cnt, "layers": cnt}

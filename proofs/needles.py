"""
proofs/needles.py — turning a document's answer-bearing span into token positions,
and choosing which positions each Proof-3 condition injects.

Proof 3 asks whether the reading travels *light*: does the large model still recall
the planted fact when handed only the answer-bearing positions, rather than the
whole layer-12 trace? Answering that honestly hinges on two pieces of bookkeeping
the proof spec calls out explicitly, and both live here:

  1. Define the needle span by the TEXT, blind to the model's answer. Each QA item
     in `synthetic_docs.py` already carries a `needle` substring — the clause that
     states the fact ("…recovered by Maren Velloth"). We locate that substring in
     the document and map it to the token indices that cover it, using the
     tokenizer's offset mapping. The span is fixed by the words, never reverse-
     engineered from what made the model succeed, so the random-subset control keeps
     its meaning.

  2. Keep ORIGINAL position identity. The indices returned here are positions into
     the document's own 0..N-1 token sequence — the same indexing the captured
     layer-12 cache uses. `core.split_forward.subset_doc_cache` slices the cache by
     these indices WITHOUT renumbering, so a needle at original positions 40-44 is
     injected as 40-44 and the query (asked from offset N) sees it at the true
     relative distance. This module never compacts indices; doing so would corrupt
     RoPE and silently fake a Path-1 failure.

The condition position-set builders (`needle_positions` / `random_subset_positions`
/ `single_position`) all optionally retain the attention-sink position 0 (the BOS
token). The sink is content-free machinery the model leans on for stable attention;
holding it constant across needles-only, random-subset, and single-position keeps
those three comparable, so the needles-vs-random contrast measures *content*, not
the presence or absence of the sink. It is recorded in the Proof-3 output either
way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── span → token positions (blind, offset-based) ──────────────────────────────
def span_token_positions(tokenizer, document, span, max_doc_tokens=8192):
    """Return the sorted list of token indices (into the document's own 0..N-1
    sequence, special tokens included exactly as `capture_document` tokenizes it)
    whose characters overlap `span` within `document`.

    Uses the fast tokenizer's offset mapping. `span` must be a substring of
    `document`; we take its first occurrence — the synthetic needles are unique
    clauses, so this is unambiguous. Raises if the span is absent (a typo in the
    needle would otherwise silently select nothing and fake a Path-1 failure).
    """
    char_start = document.find(span)
    if char_start < 0:
        raise ValueError(f"needle span not found in document: {span!r}")
    char_end = char_start + len(span)

    enc = tokenizer(
        document,
        return_tensors=None,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_doc_tokens,
    )
    offsets = enc["offset_mapping"]
    positions = []
    for i, (s, e) in enumerate(offsets):
        if s == e:          # special tokens (BOS) report a zero-width span
            continue
        if s < char_end and e > char_start:   # token overlaps the needle span
            positions.append(i)
    if not positions:
        raise ValueError(
            f"needle span {span!r} mapped to zero tokens — check tokenization "
            "(is this a fast tokenizer with offset mapping?)"
        )
    return positions


def token_positions_for_char_span(tokenizer, document, char_start, char_end,
                                  max_doc_tokens=8192):
    """Like `span_token_positions`, but located by a KNOWN character range rather than by
    re-finding the span text. Proof 5 needs this: a HotpotQA gold supporting sentence can
    recur verbatim in a distractor paragraph, so `document.find(span)` would mis-place the
    needle; `hotpot.py` already records each gold sentence's exact (char_start, char_end)
    within its own paragraph, and this maps that range to token indices via the same
    offset mapping. Raises if the range maps to zero tokens (a bookkeeping error that would
    otherwise silently fake a sparse-handoff failure)."""
    enc = tokenizer(document, return_tensors=None, return_offsets_mapping=True,
                    truncation=True, max_length=max_doc_tokens)
    positions = []
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s == e:                                   # special tokens (BOS)
            continue
        if s < char_end and e > char_start:          # token overlaps the char range
            positions.append(i)
    return positions      # may be empty if the span was truncated past max_doc_tokens


# ── condition position-set builders ───────────────────────────────────────────
def needle_positions(needle_idx, keep_sink=True):
    """needles-only: the answer-bearing positions, plus the attention sink if
    requested. Sorted, de-duplicated."""
    pos = set(needle_idx)
    if keep_sink:
        pos.add(0)
    return sorted(pos)


def single_position(needle_idx, keep_sink=True):
    """single-position (the sharp probe): only the LAST token of the needle span —
    the token at or nearest the answer itself — plus the sink if requested. Tests
    whether a single bound position carries the fact or whether it lives smeared
    across the span."""
    pos = {max(needle_idx)}
    if keep_sink:
        pos.add(0)
    return sorted(pos)


def random_subset_positions(n_doc, needle_idx, k, seed, keep_sink=True):
    """random-subset (the control that makes needles-only mean something): `k`
    positions drawn from elsewhere in the document — never the needle span — so the
    count matches needles-only but the content does not.

    Deterministic in `seed` so a run is reproducible and the draw is fixed before
    the model is consulted (no fitting the control to the result). The sink (0) is
    excluded from the random pool and added separately when `keep_sink`, so the
    random condition retains the SAME sink as needles-only and injects the same
    total count; the only difference between them is whether the k content positions
    sit on the needle or off it.

    If the document is too short to supply `k` off-needle positions, returns as many
    as exist (the caller logs the shortfall).
    """
    import random

    forbidden = set(needle_idx)
    forbidden.add(0)                       # never draw the sink into the random pool
    pool = [p for p in range(n_doc) if p not in forbidden]
    rng = random.Random(seed)
    draw = rng.sample(pool, min(k, len(pool)))
    pos = set(draw)
    if keep_sink:
        pos.add(0)
    return sorted(pos), len(pool)

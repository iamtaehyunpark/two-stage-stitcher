"""
proofs/decimate.py — choosing which positions survive a decimation, for both arms.

Experiment 3.1 thins a document two ways and compares them: drop positions as TEXT
(re-tokenize the surviving tokens, prefill normally) vs. drop positions as LATENT
(inject only the surviving positions' true layer-12 states). The one rule that makes
the comparison honest: **the kept-index set must be identical for the text and the
latent arm at every rate.** So selection lives here, once, and both arms consume the
same list — the only variable between them is text-vs-latent.

Two orthogonal choices, per the spec:

  pattern — how the surviving positions are chosen:
      "strided"  keep every Nth position (N = round(1/keep_rate)); deterministic.
      "random"   keep round(keep_rate · pool) positions, sampled with a fixed seed.
                 Random averages out any structure-specific luck in strided keeping.

  variant — how the answer-bearing needle span is treated:
      "needle_protected"  always keep the full needle span; decimate only the
                          surrounding document. Asks: can the handoff thin the
                          CONTEXT around the fact and still recall? (the clean
                          folded-context test). keep_rate applies to the non-needle
                          pool.
      "needle_decimated"  decimate uniformly, needle included. Harder; asks whether
                          latent survives losing needle positions where text loses
                          them too. keep_rate applies to the whole document.

The attention sink (position 0, the BOS) is held by default in BOTH arms' kept set
(the latent arm needs it for stable attention; the text arm re-adds a BOS on
re-tokenization anyway), so it never becomes the hidden variable.

By construction `keep_rate >= 1.0` returns EVERY position — this is the canary the
runner checks first: dec_latent at keep-rate 1 must equal full_latent, or the
position bookkeeping is wrong and every swept number is noise.
"""

import random


def kept_indices(n_doc, needle_idx, keep_rate, pattern, variant, seed,
                 keep_sink=True):
    """Return the sorted list of ORIGINAL position indices that survive decimation.
    The same list feeds the text arm (decoded to a decimated string) and the latent
    arm (injected via subset / renumbered cache)."""
    needle = set(int(p) for p in needle_idx)
    always = set()
    if keep_sink:
        always.add(0)

    if variant == "needle_protected":
        always |= needle
        pool = [p for p in range(n_doc) if p not in needle and p != 0]
    elif variant == "needle_decimated":
        pool = [p for p in range(n_doc) if p != 0]
    else:
        raise ValueError(f"unknown variant {variant!r}")

    if keep_rate >= 1.0:
        kept_pool = list(pool)                      # canary: keep everything
    elif pattern == "strided":
        stride = int(round(1.0 / keep_rate))
        kept_pool = [p for p in pool if p % stride == 0]
    elif pattern == "random":
        k = int(round(keep_rate * len(pool)))
        kept_pool = random.Random(seed).sample(pool, min(k, len(pool)))
    else:
        raise ValueError(f"unknown pattern {pattern!r}")

    return sorted(always | set(kept_pool))


def decimated_text(tokenizer, doc_ids, kept_positions):
    """Decode the kept tokens into the decimated DOCUMENT string for the text arm.

    `doc_ids` is the document's token ids (1, N), tokenized exactly as the latent
    arm captured them (so position indices line up). Position 0 (the captured BOS)
    is dropped — the prefill prompt re-adds its own special tokens — and the
    remaining kept tokens are decoded contiguously. The result is deliberately what
    a human would be handed: a renumbered, gap-collapsed text with no marker of the
    dropped material. That contiguity is exactly the asymmetry the experiment names
    (text renumbers for free; the latent arm must be told NOT to)."""
    ids = doc_ids[0].tolist() if hasattr(doc_ids, "tolist") else list(doc_ids)
    kept = [ids[p] for p in kept_positions if p != 0 and p < len(ids)]
    return tokenizer.decode(kept, skip_special_tokens=True)

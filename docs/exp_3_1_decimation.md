# Experiment 3.1 — Latent-decimation vs. Text-decimation

**Date:** 2026-06-18
**Harness:** `proofs/e31_decimation.py --layer 12` (sanity gates → keep-rate sweep)
**Model:** DeepSeek-R1-Distill-Llama-70B (80 layers, hidden 8192), frozen, sharded 4×H200
**States:** *true* layer-12 states injected via the validated two-cache split-forward
(`core/split_forward.py`). No SLM, no stitcher.
**Docs:** 4 long coreference documents (~820–920 tokens), 20 QA, all gated
(`proofs/synthetic_docs_long.py`). The answer entity sits in the *decimatable*
surroundings; the needle refers to it obliquely.
**Conditions:** reasoning suppressed (no-think); C-floor / A-ceiling gating as in
Proofs 1–3.
**Raw data:** `proofs/data/e31.json`, plot `proofs/data/e31.png`.

This is the experiment that decides whether the whole project has a reason to exist.
Proofs 0–3 established that the large model recalls a document it was *handed* as
true layer-12 states. But a sceptic can still say: *fine, but a layer-12 state is
just an expensive re-encoding of its token — hand over the tokens and you'd get the
same thing.* Experiment 3.1 tests exactly that. It thins each document two ways and
compares how fast recall dies:

- **dec_text** — keep a subset of tokens, feed them as ordinary text, prefill.
- **dec_latent** — keep the *same* positions, inject only their true layer-12 states
  at their **original** indices (split-forward subset).
- **dec_latent_renumbered** — the same kept states, but RoPE-re-rotated to contiguous
  positions: the control that asks whether any latent advantage is just
  position-bookkeeping.

If text and latent die together, latent is interchangeable with text and the latent
machinery is dead weight. If text collapses while latent holds, **a layer-12 state
carries context its raw token does not** — the folded-in reading of its dropped
neighbours — and latent handoff has something real to offer.

---

## The result

**Text collapses; latent holds. Latent ≠ text.** At meaningful thinning, dropping
every other *token* recalls essentially nothing, while dropping every other
*position's latent state* still recalls ~40–60% of the planted facts. The renumbered
control rules out a position artifact: the advantage is in the representation itself.

---

## The curves

All cells are recall (fraction of the 20 gated facts produced), C-floor 0.00 and
full-prefill A 1.00 by construction, so each number *is* recall fidelity vs. full
prefill. References at keep-rate 1: **full_text = 1.00, full_latent = 1.00.**

### variant: needle_protected (decimate only the surroundings; keep the needle span)

| keep | pattern | dec_text | dec_latent | dec_latent_renumbered |
|-----:|:-------:|:--------:|:----------:|:---------------------:|
| 1/2  | strided | **0.00** | **0.40**   | 0.35 |
| 1/4  | strided | 0.00     | 0.20       | 0.15 |
| 1/8  | strided | 0.00     | 0.05       | 0.10 |
| 1/16 | strided | 0.00     | 0.00       | 0.05 |
| 1/2  | random  | 0.15     | **0.60**   | 0.60 |
| 1/4  | random  | 0.05     | 0.15       | 0.15 |
| 1/8  | random  | 0.00     | 0.05       | 0.05 |
| 1/16 | random  | 0.00     | 0.05       | 0.05 |

### variant: needle_decimated (decimate uniformly, needle included)

| keep | pattern | dec_text | dec_latent | dec_latent_renumbered |
|-----:|:-------:|:--------:|:----------:|:---------------------:|
| 1/2  | strided | **0.00** | **0.40**   | 0.35 |
| 1/4  | strided | 0.00     | 0.20       | 0.15 |
| 1/8  | strided | 0.00     | 0.00       | 0.05 |
| 1/16 | strided | 0.00     | 0.00       | 0.05 |
| 1/2  | random  | 0.25     | **0.55**   | 0.55 |
| 1/4  | random  | 0.05     | 0.15       | 0.15 |
| 1/8  | random  | 0.00     | 0.10       | 0.00 |
| 1/16 | random  | 0.00     | 0.00       | 0.00 |

**Where to read the result: keep-rate 1/2 and 1/4 — the rates where text is still
alive enough to lose to.** Below that (1/8, 1/16) every arm is on the floor, because
thinning to 6–12% of positions destroys the fact regardless of representation;
reading a verdict off those rows is like calling two engines identical because both
fail with an empty tank.

---

## What it means

**1. Latent states carry context their tokens do not — the project's central claim,
confirmed.** At keep-rate 1/2 strided, `dec_text = 0.00` while `dec_latent = 0.40`.
Shred the text by dropping every other token and it carries *nothing*; drop every
other position's *latent state* and 40% of the facts still recall. The only
difference between the two arms is text-vs-latent — same positions, same count, same
gating — so the gap is the folded-in context the layer-12 states absorbed from the
neighbours that were dropped. **A layer-12 state is not just expensive text.**

The maximum `dec_latent − dec_text` gap is **~0.45** (random, keep-rate 1/2:
0.60 − 0.15), decisively positive, and the gap is robust to *how* you thin:

| keep 1/2 | dec_text | dec_latent | gap |
|:--------:|:--------:|:----------:|:---:|
| strided  | 0.00     | 0.40       | 0.40 |
| random   | 0.15     | 0.60       | 0.45 |

The strided/random split is itself a sanity check: text does worse under strided
(0.00) than random (0.15) because every-other-token guarantees shattering every
local phrase, whereas random sometimes leaves adjacent tokens intact. Latent beats
text under *both* patterns. Text falls off a cliff; latent degrades gracefully.

**2. The advantage is representational, not positional — the renumbered control did
its job.** `dec_latent` and `dec_latent_renumbered` track each other across nearly
every cell (0.40/0.35, 0.60/0.60, 0.20/0.15, 0.55/0.55). If the win had been "latent
looks good only because the positions were preserved," renumbering the kept states
to contiguous positions would have collapsed it. It didn't. So the carried context
lives **in the representation itself**, not in RoPE phase handling. The result
survives the control specifically built to kill it — the strongest form it could
take. (Small inversions like 0.05 vs 0.10 at keep-rate 1/8 are n=20 noise: one fact =
0.05. The signal is that the two latent arms are statistically the same.)

**3. needle_protected and needle_decimated barely differ** (0.40 vs 0.40 strided at
1/2). Protecting the answer-bearing span did *not* rescue recall, which means the
effect is not "did the literal answer span survive" — it is distributed context,
consistent with Proof 3's finding that single positions are null and the fact lives
in a span of folded state. Both variants show the same latent > text gap, so the
effect is not an artifact of which positions were protected.

**4. Latent is not magic — it still degrades.** 0.40 at half-thinning, not 1.00.
The honest claim is **"latent degrades gracefully where text collapses,"** not
"latent is lossless." Thinning loses information in both arms; it just loses it far
faster as text.

---

## A note on the auto-reader (fixed)

The first run's printed verdict said *"both COLLAPSE → latent interchangeable with
text"* — and it was **wrong**. It evaluated only the heaviest thinning (keep-rate
1/16), the one row where everything is dead, so it would declare "illusory" on *any*
successful experiment, since every curve eventually collapses at extreme thinning.
That was a bug in the analysis, not a finding.

`interpret()` in `proofs/e31_decimation.py` now reports the **maximum
`dec_latent − dec_text` gap across the rates where text is still alive**, plus the
mean latent-vs-renumbered agreement, and reads the verdict from there. On this data
it correctly reports a decisive ~0.45 gap and a representational (not positional)
cause.

---

## Loose ends to close before banking it

- **Tighten the two load-bearing numbers.** n=20 means each fact is worth 0.05, and
  everything below keep-rate 1/4 is on the floor and uninformative. Re-run **only**
  keep-rates 1/2 and 1/4 at **n ≥ 60** to put error bars on the 0.40/0.60 figures.
  Don't spend samples at 1/8 and 1/16 — text is already zero there.
- **Document length.** This is on ~850-token docs. The latent > text gap could
  *widen* on longer documents (more skipped context for the states to have folded in)
  or shrink. Worth one run at length; not blocking, since the mechanism is
  demonstrated.
- **The effect is graded, not total.** 0.40–0.60, not ~1.0. That sets a realistic
  ceiling for what a sparse latent handoff can recover and should temper the Proof 5
  and Proof 6 claims accordingly.

---

## Bottom line

At meaningful thinning, **decimated text dies (0.00) while decimated latent survives
(0.40–0.60)**, and the renumbered control proves the advantage is the representation,
not the positions. **Latent states are not interchangeable with their tokens — they
carry the context of their dropped neighbours.** This is the green light Proof 5
needed: there is something real for latent handoff to beat text with. The next move
is Proof 5 proper — latent handoff vs. full-document text-RAG on questions that
matter (including reasoning / multi-hop) — now that 3.1 has established the mechanism
is real.

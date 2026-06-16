# Layer Sweep — where does the receiver read an injected document?

**Date:** 2026-06-17
**Harness:** `proofs/run_chain.py --layers …` (Proof 0 + Proof 1 across depth; Proof 2
at each run's winning layer)
**Model:** DeepSeek-R1-Distill-Llama-70B (80 layers, hidden 8192), frozen
**States:** *true* layer-`L` states injected via the validated two-cache
split-forward (`core/split_forward.py`). No SLM, no stitcher.
**Conditions:** reasoning suppressed (no-think); P0 = memorized Everest doc (130
tokens, 5 Q); P1/P2 = synthetic fabricated-fact docs (~150 tokens, 25 gated Q).
**Raw data:** [`results/sweep_layers_12-23.json`](results/sweep_layers_12-23.json),
[`results/sweep_layers_16-64.json`](results/sweep_layers_16-64.json),
[`results/sweep_layers_25-34.json`](results/sweep_layers_25-34.json).

This sweep asks the question the project never actually answered: **at which layer
should the document be injected?** Layer 30 was an assumption, never measured. It is
now measured, and it is suboptimal.

---

## The curve

Proof 1 recall (`inject_correct_on_gated`, 25 gated synthetic facts; the C floor is
0.00 and full-prefill A is 1.00 at every layer, so this number *is* the recall
fidelity vs full prefill):

| Layer | depth | P0 (plumbing) | P1 recall | P1 verdict |
|------:|------:|:-------------:|:---------:|:----------:|
| 12 | 15% | PASS (5/5) | **1.00** | PASS |
| 14 | 18% | PASS (5/5) | **0.96** | PASS |
| 16 | 20% | PASS (5/5) | 0.56 | PARTIAL |
| 18 | 22% | PASS (5/5) | 0.56 | PARTIAL |
| 21 | 26% | PASS (5/5) | 0.60 | PARTIAL |
| 22 | 28% | PASS (4/5) | 0.64 | PARTIAL |
| 23 | 29% | PASS (4/5) | 0.76 | PARTIAL |
| 24 | 30% | PASS (4/5) | 0.76 | PARTIAL |
| 25 | 31% | PASS (4/5) | 0.76 | PARTIAL |
| 26 | 33% | PASS (4/5) | 0.76 | PARTIAL |
| 28 | 35% | PASS (4/5) | 0.68 | PARTIAL |
| 30 | 38% | PASS (4/5) | **0.60** | PARTIAL |
| 32 | 40% | FAIL (3/5)¹ | — | (skipped)¹ |
| 34 | 42% | PASS (4/5) | 0.20 | PARTIAL |
| 40 | 50% | PASS (4/5) | **0.00** | FAIL |
| 48 | 60% | PASS (4/5) | 0.00 | FAIL |
| 56 | 70% | PASS (4/5) | 0.00 | FAIL |
| 64 | 80% | PASS (5/5) | 0.00 | FAIL |

¹ Layer 32's P1 was skipped because P0 fell to 3/5 (< the 0.8 plumbing gate) — a
**P0-gate artifact, not an injection failure**. It reproduces in both files that
cover layer 32, and it is flanked by L30 = 0.60 and L34 = 0.20, so it is plainly
inside the declining region. The dip is the memorized-doc P0 brittleness (the
Tenzing-Norgay hedge plus one more), not a property of layer 32.

Shape: **peak at the shallowest layers (12–14 ≈ 1.0), a noisy mid plateau (16–30 ≈
0.56–0.76), a cliff after 30, and dead zero from layer 40 onward.**

---

## What it means

**1. Layer 30 was wrong, and measurably so.** The project's assumed injection point
scores **0.60** — mid-pack, clearly suboptimal. The usable band is **shallow
(≤ ~14)**; recall **dies past ~40**.

**2. The mechanism is "reasoning headroom," not "a magic semantic layer."** Injecting
true layer-`L` states and running `L → 80` gives the model `80 − L` layers of its own
computation *over* the injected content. Inject at 12 → 68 layers of headroom; inject
at 64 → only 16 layers left, and the content arrives already transformed toward
next-token prediction with too little stack remaining to reason over it. Past ~40
there is not enough headroom left at all, so recall collapses to zero. This is a more
general and more defensible claim than "30/80 is special": **the receiver needs enough
stack above the injection point to reason, and shallow injection preserves it.**

**3. The premise is causally confirmed at the winning layers.** Proof 2 (the
wrong-document falsifier) was run at every run's winning layer:

| Winning layer | matched-doc recall | wrong-doc recall | verdict |
|--------------:|:------------------:|:----------------:|:-------:|
| 12 | 1.00 | **0.00** | PASS |
| 24 | 0.76 | **0.00** | PASS |
| 25 | 0.76 | **0.00** | PASS |

Wrong-document injection yields **zero** correct answers while the matched document
recalls up to 1.00 — the clean causal result the original Witcher oracle probe could
never produce. On synthetic facts, with the falsifier clean, the recall is *caused by
the injection*, not by parametric memory. **Proofs 1 and 2 are genuinely confirmed at
the shallow layers; reading is transferable for true states.**

**4. Plumbing is correct across the whole stack.** P0 is non-degenerate (0
degeneracy) at *every* layer 12 → 64, and 4–5/5 correct everywhere except the L32
gate artifact. The split-forward RoPE/position/KV work holds at any injection depth.

---

## The tension this creates (the important part)

**Recall is best exactly where compute savings are worst.** The original pitch was
"skip the expensive prefill." But the best injection point is layer ~12/80 — skipping
only ~15% of the layers, while the model still runs ~85% of its forward pass over the
context positions. The FLOPs you save grow toward zero precisely where recall is high
(shallow), and the FLOPs you save are large precisely where recall is zero (deep,
≥ 40). The recall/savings tradeoff runs the wrong way.

This reframes — but does not kill — the project. Two honest readings:

1. **The value is the cross-family handoff and the KV-cache memory, not the prefill
   FLOPs.** Even at layer 12 the document is gone from the large model's **KV cache**
   (the README's "tens of gigabytes" memory argument survives) and the *reading* is
   outsourced to the cheap SLM. The economic story becomes **memory + outsourcing**,
   stated deliberately rather than discovered in review.
2. **The usable layer may move under two conditions not yet tested** — long documents
   and *translated* (not true) states. Both could shift where the receiver reads best.

---

## Loose ends to close before building on this

- **Re-run the sweep at realistic document lengths (2k / 8k / 16k).** These docs are
  ~130–150 tokens; the entire project is about *long* context. The optimal layer may
  migrate with `N`. This is the single most important follow-up: if the peak does not
  move deeper as context grows, the economic story must shift to memory + outsourcing.
- **Fill the L32 hole** by lowering the P0 gate or hardening the memorized P0 doc; it
  is a gate artifact, not a real failure, but it leaves a visible gap in the curve.
- **Don't over-read the mid plateau.** 0.56 → 0.76 → 0.60 across 16–30 is ±0.1
  sampling noise at n = 25. The *shape* (shallow-high, deep-zero) is robust; the bumps
  are not.
- **True states ≠ translated states.** This sweep injects ground-truth layer-`L`
  states. The layer where the Qwen→DeepSeek translation is *learnable* (Proof 6) may
  differ from where the receiver reasons best; re-survey under the translation
  constraint before fixing a target layer.

---

## Bottom line

The receiver works (P0 clean at every depth; P1/P2 confirmed on synthetic facts with
a passing falsifier), and the assumed layer is wrong: the usable band is **shallow
(≤ ~14, recall ≈ 1.0)**, recall **dies past ~40**, and **layer 30 was mediocre
(0.60)**. The sharp new tension — **recall is best where compute savings are worst** —
should drive the next step: a **length-scaling sweep** to see whether the peak
migrates deeper as context grows. If it does not, the project's economic claim moves
from "skip prefill FLOPs" to "skip KV-cache memory + outsource reading to the SLM."

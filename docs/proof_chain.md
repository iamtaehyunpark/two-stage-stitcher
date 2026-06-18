# Proof Chain — Validating the Receiver Before Building the Sender

The project advances as a chain of falsifiable proofs. Each rung is a precondition
for the next: **if one fails, every rung above it is moot.** They are ordered to kill
the project as cheaply as possible.

**Proofs 0–5 use only the LLM and TRUE captured layer-30 states — no Qwen, no stitcher.**
This is deliberate: validate the *receiver* completely before building any *sender*.
The SLM enters only at Proof 6.

> **Result (2026-06-17):** Proofs 0–2 pass on synthetic facts, and a depth sweep
> shows the assumed layer 30 is suboptimal — recall peaks shallow (≈1.0 at L12–14),
> is mediocre at 30 (0.60), and dies past ~40. The new tension: recall is best where
> compute savings are worst. Full write-up: [`layer_sweep.md`](layer_sweep.md).
>
> **Result (2026-06-18):** Experiment 3.1 (latent- vs. text-decimation at L12)
> answers the project's existential question: **a layer-12 state carries context its
> raw token does not.** Thin a document and decimated *text* dies (≈0.00 at keep-rate
> ½ strided) while decimated *latent* survives (0.40–0.60); max latent−text gap
> ≈0.45. A RoPE-renumbered control tracks the original-position arm, so the advantage
> is **representational, not a position artifact.** Latent ≠ text — the green light
> Proof 5 needed. Full write-up: [`exp_3_1_decimation.md`](exp_3_1_decimation.md).

| # | Proof | Proves | Pass | Fail | Status |
|---|---|---|---|---|---|
| 0 | Split-forward plumbing | harness is correct | split-forward(true) ≈ A | plumbing bug; nothing below interpretable | **PASS** (clean at every layer 12–64) |
| 1 | Injection premise | injected states are read & reasoned over | inject-all-N succeeds where C fails | premise broken → **stop project** | **CONFIRMED** (layer-dependent; ≈1.0 at L12–14) |
| 2 | Wrong-doc falsifier | it's injection, not memory | wrong-doc fails X | "success" was memory → **falsified** | **PASS** (wrong-doc 0.00 at L12/24/25) |
| 3 | Path resolution | full prefix vs sparse needles | (see outcomes) | — | **implemented** (`proofs/p3_path.py`, fixed at layer 12); awaiting run |
| 4 | Length scaling | survives long context | recall holds at 16k/32k+ | dies at length → premise unproven | pending (blocked by 3) |
| 5 | Latent beats text-RAG | reason to exist | latent ≥ text-RAG at lower cost | dead weight vs RAG | **motivated** by Exp 3.1 (latent ≠ text); pending full run |
| 6 | Stitcher reproduces states | the SLM works | stitched recovers facts | translation is the real bottleneck | pending (blocked by 5) |

---

## Proof 0 — The split-forward plumbing is correct
**Claim:** the two-phase forward (document states injected at layer 30; query run
through 0–29 with offset `position_ids`; joined at 30) reproduces normal inference
when fed the *true* states.
**Experiment:** memorized document → capture true layer-30 states → run split-forward
→ compare to Condition A (full prefill) on the *same* document. A plumbing/regression
test, not a science result: its only job is to prove RoPE, masking, KV-cache, and
position bookkeeping are right before the harness is trusted. Use a memorized doc
because the correct answers are known independently.
**Pass:** split-forward(true states) ≈ A. **Fail:** plumbing bug → nothing below is
interpretable. **Gates everything.**

## Proof 1 — The injection premise
**Claim:** injected LLM-native representations are actually read and reasoned over,
not ignored.
**Experiment:** synthetic-fact documents (fabricated, unguessable facts).
- **C (floor):** question only → must **fail** (gates that the fact isn't guessable).
- **A (ceiling):** full prefill of the synthetic doc → must **succeed** (gates that the fact is answerable from the text).
- **Inject-all-N:** true layer-30 states, all positions, split-forward → the test.
**Pass:** inject-all-N succeeds where C fails. **Fail:** premise broken — no stitcher
can ever help — **stop the project.**

## Proof 2 — It's the injection, not parametric memory (the falsifier)
**Claim:** the LLM answers from the *injected document*, not coincidental world knowledge.
**Experiment:** wrong-document control — inject document Y's true states, ask document
X's question.
- Still answers X correctly → injection inert, Proof 1's success was memory → **falsified.**
- Answers Y's content / fails X → injection steers the model → **confirmed.**
The control the Witcher oracle probe lacked; **non-negotiable.** Best on synthetic
facts so coincidental memory is impossible by construction. *Proofs 1 + 2 together
prove the premise (reasoned-over AND not memory).*

## Proof 3 — Path resolution: all-N or just the needles?
**Claim (one of two):** the LLM needs the full document prefix (Path 1), or a sparse
set of answer-bearing positions suffices (Path 2).
**Experiment:** same synthetic-fact setup, add:
- **Inject-needles-only:** true states for *only* the answer-bearing positions.
- *(optional)* **Inject-random-subset:** same count, randomly chosen — controls for "needles specifically vs any N positions."
**Outcome is the result:**
- all-N works, needles work → **Path 2** open (sparse handoff, big economic win).
- all-N works, needles fail → locked to **Path 1** (full-prefix outsourcing, bounded win).
- needles work but random-subset also works → suspicious; content isn't being selected; re-examine.

## Proof 4 — Length scaling
**Claim:** whatever works at ~1.3k tokens still works (or doesn't) at the lengths the
project is for.
**Experiment:** repeat Proofs 1–3 across lengths (1k, 4k, 16k, 32k+), synthetic facts
planted at varying depths. Measure recall vs length and vs needle-depth.
**Why its own rung:** the premise is *long* documents; a method that works at 1.3k and
dies at 16k hasn't proven the thing that matters. Path 1 vs Path 2 economics are
decided here.

## Proof 5 — Latent handoff beats text handoff (reason to exist)
**Claim:** transferring *latent representations* beats handing the LLM the retrieved
*text* as tokens.
**Experiment:** baseline = retriever pulls answer-bearing spans, fed to the LLM as
ordinary text (normal RAG). Compare against latent injection of the same spans.
**Pass:** latent injection matches/beats text-RAG at lower token/compute cost.
**Fail:** latent machinery is dead weight — text-RAG already does it. The proof a
reviewer will demand; it justifies the whole latent approach.

## Proof 6 — The stitcher can reproduce true states (the SLM enters)
**Claim:** an SLM→LLM translation produces layer-30 states close enough to the true
ones that Proofs 1–5 still hold with *stitched* states.
**Experiment:** replace true states with stitcher output in the winning condition from
Proofs 3–4, judged **behaviorally** (answers the synthetic fact, fails the
wrong-document control). Critically: train/evaluate the stitcher on a **behavioral
objective (KL vs. full-prefill logits)**, not vector cosine — the post-mortem proved
cosine certifies nothing.
**Pass:** stitched states recover the synthetic facts. **Fail:** translation is the
bottleneck after all — and now you know it's a *real* one (not the architecture),
worth investing in.

---

## Dependency structure

- **0 gates everything** — no trustworthy harness, no trustworthy results.
- **1 + 2 prove the premise** — reasoned-over AND not memory.
- **3 + 4 size the win** — sparse vs full, and whether it survives long context.
- **5 proves you beat the boring baseline** — latent vs text-RAG.
- **6 is the only rung that involves the SLM** — earned only after 0–5 say the receiver
  works, the win is real, and latent beats text.

The first three rungs (0, 1, 2) are a single afternoon's script — `evaluate/oracle_probe_v2.py`
already implements the synthetic-fact / C-fails-gate / wrong-document control for the
*single-vector* case. Proof 0's split-forward and Proof 1's *all-N* injection are the
remaining pieces needed to run 0–2 end to end on multi-token true states.

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
>
> **Result (2026-06-19):** Proof 4 (length scaling) **passes** and Proof 4.1 (a
> hardened re-test) confirms it. inject-all-N holds at **≈1.0 from 500 → 32k tokens at
> L12**, flat across needle depth 10/50/90% (no lost-in-the-middle for injection), the
> optimal layer does **not** migrate, sparse handoff survives length, and the latent>text
> gap **widens** with length (text decays, latent holds). Proof 4.1 then re-ran the 32k
> point with the evaluation slack removed — strict scoring, q-fair capture symmetry,
> near-miss distractors, reasoning-on — pooled over **30 distinct facts** from the new
> adversarial [`fact_bank.py`](../proofs/fact_bank.py): on the gated set
> `inject_docnaive = A = 0.97` (strict, with distractors), `qfair = 0.93` and the **most
> robust under reasoning** (0.92 vs A 0.90, docnaive 0.83), and on coreference docs
> thinned to collapse, **`dec_latent = 0.61` vs `dec_text = 0.03` (gap +0.58)** under
> strict + distractors. The distractors bite even full-prefill A (~15% decoy-confusion
> at depth 0.9), so "parity with A" is parity with a genuinely stressed ceiling. Caveats
> carried forward: latent recovery is **partial** (~0.6, as in 3.1), and Path-2
> sparse-handoff is the weakest inject condition under stress (0.75). Receiver validated
> end-to-end; **Proof 5 is earned.**
>
> **Result (2026-07-01):** Proof 5 (latent handoff vs. text-RAG) is an **honest stop for
> the budget-matched sparse handoff.** On zero-memory synthetic 2-hop (n=40, L12, think-on,
> LLM-judge), the receiver holds — `latent_all = A = 0.97` (full-document handoff ≈ reading)
> — but the budget-matched **`latent_goldspan = 0.62` vs `text_goldspan = 1.00` (gap −0.375)**:
> the sparse latent handoff loses to *clean* text of the same spans, uniformly across answer
> length (−0.38 at 2–3 tok, −0.33 at 4+). The pre-registered sanity anchor passes on
> coreference docs — **`dec_latent = 0.43` vs `dec_text = 0.00` (+0.433)**, replicating Exp
> 3.1 — so the harness is sound. The pattern: **latent beats *decimated/degraded* text but
> loses to *clean* text of the same content**, and nobody deploys shredded retrieval. On real
> HotpotQA the gap is a near-tie (−0.07), but that is memory (43% closed-book discard) lifting
> the floor — the synthetic control breaks the tie against latent. Scoring caveat: the LLM-
> judge is authoritative (0.62); deterministic scorers bracketed 0.47–0.88 and are unreliable
> on reasoning-aloud answers. Env caveat: **transformers 5.x silently breaks the subset-inject
> path**, pinned to 4.x. Proof 5's bar (latent ≥ text-RAG at equal budget) is **not met**;
> Proof 6 is not motivated by a sparse-handoff win. Full write-up:
> [`proof_5_latent_vs_rag.md`](proof_5_latent_vs_rag.md).

| # | Proof | Proves | Pass | Fail | Status |
|---|---|---|---|---|---|
| 0 | Split-forward plumbing | harness is correct | split-forward(true) ≈ A | plumbing bug; nothing below interpretable | **PASS** (clean at every layer 12–64) |
| 1 | Injection premise | injected states are read & reasoned over | inject-all-N succeeds where C fails | premise broken → **stop project** | **CONFIRMED** (layer-dependent; ≈1.0 at L12–14) |
| 2 | Wrong-doc falsifier | it's injection, not memory | wrong-doc fails X | "success" was memory → **falsified** | **PASS** (wrong-doc 0.00 at L12/24/25) |
| 3 | Path resolution | full prefix vs sparse needles | (see outcomes) | — | **PASS → PATH_2** (`proofs/p3_path.py`, L12: needles-only 0.84, random 0.00) |
| 4 | Length scaling | survives long context | recall holds at 16k/32k+ | dies at length → premise unproven | **PASS** (≈1.0 to 32k at L12, depth-flat; hardened by 4.1 on 30 facts) |
| 5 | Latent beats text-RAG | reason to exist | latent ≥ text-RAG at lower cost | dead weight vs RAG | **HONEST STOP** — sparse handoff loses to *clean* text of the same spans (goldspan 0.62 vs 1.00, judge); beats only *decimated* text (dec +0.43). Receiver holds (`latent_all=A`). Bar not met. See [`proof_5_latent_vs_rag.md`](proof_5_latent_vs_rag.md) |
| 6 | Stitcher reproduces states | the SLM works | stitched recovers facts | translation is the real bottleneck | **not motivated by Proof 5** — no budget-matched sparse win over RAG |

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
**Claim:** whatever works at ~130 tokens still works (or doesn't) at the lengths the
project is for.
**Experiment** (`proofs/p4_length.py`): pad each synthetic fact block to a target
length (500 / 2k / 8k / 16k / 32k) with **inert filler** (entity-free, digit-free
prose — [`long_context_docs.py`](../proofs/long_context_docs.py)) and plant it at
needle depth ~10% / ~50% / ~90%. Re-run the core conditions across the
length × depth grid, **re-sweeping the injection layer at each length** (the optimal
layer may migrate deeper as context grows — don't assume 12 transfers).
**The Proof-4-specific trap — filler contamination:** padding can accidentally carry a
cue, creating a false floor. So inertness is **re-checked behaviourally per length**, not
inherited: each cell adds a `C_filler` gate (prefill the padded doc with the fact
*removed* and confirm the model still can't answer). A cell is gated iff
`C` fails **and** `C_filler` fails **and** `A` succeeds.
**Run it staged — let the first curve tell you where to spend:**
- `--stage curve` (the gate): inject-all-N at layer 12, depth 50%, across lengths.
  Does recall stay ≈1.0 at 2k/8k? If yes, scale up; if it drops, the layer re-sweep
  becomes the priority and you've found the real research question early.
- `--stage relayer` (only where the curve dropped): re-sweep layers {8,12,20,30} at
  that length. Does re-picking the layer rescue recall? Where is the optimal layer vs
  length (a layer that migrates *deeper* is the hoped-for outcome — more prefill
  skipped where it matters).
- `--stage axes` (last): needle-depth {10/50/90%}, sparse handoff (needles-only), and
  latent-vs-text at fixed thinning — the Exp-3.1 gap at length (widen / hold / collapse).
**Why its own rung:** the premise is *long* documents; a method that works at 130 tokens
and dies at 16k hasn't proven the thing that matters. Path 1 vs Path 2 economics are
decided here. **None of the outcomes stop the project** — a decay that survives a layer
re-sweep redirects to the multi-slot / per-chunk handoff, it does not kill the SLM.
**Result (2026-06-19):** the cheapest curve was already decisive — inject-all-N held at
**1.00 from 500 → 32k** at L12, depth-flat at 10/50/90% (≈300 gated cells), so the
relayer stage was moot (nothing dropped, the L12 shelf does not migrate). The axes stage
added: sparse handoff survives length (0.81→0.85), and the latent−text gap **widens** with
length (dec_text 1.00→0.92 as dec_latent holds ≈1.0). A's own retrieval shows the faint
32k lost-in-the-middle dip (0.96); injection does not. **PASS.**

## Proof 4.1 — Hardened single-point confirmation
**Claim:** the Proof-4 headline survives strict measurement, not just lenient scoring on
easy facts. **Experiment** (`proofs/p4_1_hardened.py`): re-test 32k / L12 with the
evaluation slack removed, pooled over the adversarial [`fact_bank.py`](../proofs/fact_bank.py)
(50 facts / 10 docs) for real n — (1) **strict scoring** (gold in the answer clause,
exclusively, unhedged) beside lenient/firstline; (2) **capture/A symmetry** —
`inject_qfair` captured inside A's instruction+document framing vs document-only
`inject_docnaive`; (3) **near-miss distractors** woven in so a correct answer must
*discriminate* the true needle; (4) **reasoning-on** arm (Proof 5's actual path). The
latent-vs-text contrast runs on coreference docs thinned to where text collapses.
**Result (2026-06-19):** discrimination **PARITY_WITH_A** (n=76 from 30 facts:
`docnaive = A = 0.97`, `qfair = 0.93` strict, and qfair the most reasoning-robust);
mechanism **LATENT_BEATS_TEXT** (n=59 from 20 facts: `dec_latent 0.61` vs `dec_text 0.03`,
gap +0.58, strict + distractors). Distractors fool full-prefill A ~15% at depth 0.9, so
parity is against a stressed ceiling. **SHIP_TO_PROOF_5.** Caveats carried forward: latent
recovery is partial (~0.6), sparse handoff weakest under stress (0.75), q-fair capture is
the robust one to use going forward.

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

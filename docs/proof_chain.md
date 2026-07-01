# Proof Chain

The project advances as chains of falsifiable proofs. Each rung is a precondition
for the next: **if one fails, every rung above it is moot.** They are ordered to kill
the project as cheaply as possible.

The work splits into two chains at the model boundary:

- **[Chain 1 — the receiver](#chain-1--validating-the-receiver-frozen) (frozen).**
  Every rung used a single model (DeepSeek) as both reader and reasoner, injecting
  *true* captured states. It proved the receiver: injected LLM-native states are read
  and reasoned over — within a model, in full, to 32k. Chain 1 is **done and its proofs
  are true for what they tested.** It is frozen as the validated foundation: cited, not
  rewritten. Rewriting it would either restate proven results under new numbers or
  retroactively fold in cross-model assumptions the original runs never made — corrupting
  a clean record.
- **[Chain 2 — the outsourcing](#chain-2--reading-is-out-sourceable) (active).**
  A *different, cheaper* reader enters. Two things change the moment the model boundary
  is crossed. (1) The handoff object shifts from the *captured KV cache* to the *layer-12
  residual stream* — an SLM can produce one `(N, d)` tensor per position and let the LLM
  recompute the upper stack, but cannot produce 68 layers of cross-family KV. (2) The
  economic claim shifts from *compression* (ship less — dead, Proof 5) to *outsourcing*
  (ship the full read, but a cheap model produced it).

**Chain 1 proved the receiver. Chain 2 proves the outsourcing.** Chain 1 was always the
prerequisite Chain 2 is built on — not a thing Chain 2 replaces. The current frontier is
**Proof 2.0** (residual-equivalence): the cheap, single-model rung that defines the target
the sender must aim at.

---

# Chain 1 — Validating the Receiver (frozen)

**Proofs 0–5 use only the LLM and TRUE captured layer-30 states — no Qwen, no stitcher.**
This is deliberate: validate the *receiver* completely before building any *sender*.
The SLM enters only at Proof 6 — which Chain 2 reshapes (see Proof 2.2).

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

---

# Chain 2 — Reading is *out*-sourceable

Chain 1 proved reading transfers *within* a model. Chain 2 tests the thesis Chain 1 was
always the prerequisite for: **reading done by a *cheaper, different* model transfers to
the expensive one.** Every Chain 1 proof used DeepSeek as both reader and reasoner. Chain 2
is where the SLM (Qwen) enters as the reader while the LLM stays the reasoner.

Two reframings, forced by everything Chain 1 learned, define this chain:
1. **Object:** the handoff is the **layer-12 residual stream** — one `(N, d)` tensor the
   SLM can realistically produce — not the 68-layer KV cache Chain 1 injected. The LLM
   recomputes layers 12→80 from it.
2. **Claim:** the win is **outsourcing**, not compression. Chain 1's Proof 5 killed
   sparse/compressed handoff (latent loses to *clean* text of the same spans). Chain 2
   ships the *full* read — the bet is that a cheap model produced it, so the expensive
   model never had to.

Every rung is gated on the previous, and each can stop the project cheaply.

| # | Proof | Proves | Pass | Fail | Status |
|---|---|---|---|---|---|
| 2.0 | Residual-equivalence | the small object suffices as the handoff target | `residual_inject ≈ cache_inject ≈ A`, clean per-item agreement | recompute-from-residual loses what the stored cache had → SLM on the hook for a bigger object | **IMPLEMENTED — awaiting run** (`proofs/p2_0_residual.py`; plumbing proven on CPU by self-test invariant G) |
| 2.1 | Cross-family geometry (oracle map) | the SLM→LLM spaces are bridgeable | an *oracle-fit* map from Qwen states lets the LLM recall the fact | even an overfit oracle map fails → geometry fundamentally misaligned at L12; no stitcher saves it → **kills the outsourcing thesis** | pending 2.0 |
| 2.2 | Learned outsourcing (the SLM stitcher) | a *trained* cheap reader bridges it | stitched L12 residuals recover recall at **full** handoff | translation is the real bottleneck (now known to be real, not architectural) | pending 2.1 |
| 2.3 | Honest economics | SLM-reads-then-LLM-reasons costs less | SLM prefill + translate + LLM upper-stack-only < LLM full prefill, at equal accuracy | not cheaper → "outsourceable" is true but pointless (Proof-5-shaped stop, one level up) | pending 2.2 |
| 2.4 | Error amplification | the SLM's approximation survives 68-layer recompute | recall tolerance ≥ the stitcher's achievable injected-state error | approximate L12 state gets amplified through the upper stack → breaks 2.2 in practice | pending 2.2 |
| 2.5+ | Outsourced latent > outsourced text | it beats the boring alternative | latent handoff beats "SLM extracts text, feed the LLM tokens" — at length, across tokenizers | the Proof-5 ghost, one level up: outsourced *text* already does it | frontier |

**The spine:** 2.0 the small object suffices (single-model) → 2.1 the cross-family gap is
bridgeable (oracle, no training) → 2.2 a trained cheap reader bridges it (full handoff,
behavioral loss) → 2.3 it's actually cheaper → 2.4 the error survives recompute → 2.5 it
beats outsourced *text*.

**The symmetry with Chain 1, and why it's honest.** Chain 1 ran 0→…→5 and its final rung
was "does latent beat text" — it didn't, sparse. Chain 2 ends the same way one level up:
"does *outsourced* latent beat *outsourced* text," and that should be expected to be the hard
rung again. The residual-stream + full-handoff + behavioral-objective reframing gives it a
*better* shot than sparse had, but the Proof-5 ghost is real — the thing to beat is still
"just hand over the retrieved text," now produced by the SLM.

---

## Proof 2.0 — Residual-equivalence

**The question.** Chain 1 always injected the captured **KV cache** (layers 12→79, the big
object). An SLM can only realistically produce the **layer-12 residual stream** (one `(N, d)`
tensor) and let the LLM recompute the upper stack. Before any cross-model work: *does
injecting the true layer-12 residual stream, and recomputing layers 12→80, match injecting
the captured cache?* If yes, the SLM has a small, well-defined target. If no, outsourcing is
fighting a harder object than Chain 1's success implied.

**Why it's not trivially yes.** Naively the cache *is* a deterministic function of the
residual stream (K, V are linear projections of it), so recomputing should reproduce it
exactly. Two things can break that:
1. **Correctness.** The captured cache was produced in a forward where each layer's input was
   the *real* previous layer's output. The recompute path injects at 12 and rebuilds 13…79
   from there. For *true* states these should be identical — so a mismatch signals a **bug**
   (position handling, dtype, the attention-sink token, cache bookkeeping), worth catching
   before Chain 2 depends on it.
2. **Numerical.** 68 layers of recompute in fp16 can drift. Probably negligible, but this run
   tells you the floor.

So Proof 2.0 is partly a *correctness check* (do the two paths agree for true states, as
theory says) and partly a *setup* (establishing the residual stream as the target the SLM
will produce). A clean pass means "the small object is sufficient and the plumbing is honest."

**The three conditions**, all single-model (DeepSeek), same frozen operating point (L12,
think-on, q-fair capture, strict + judge), on the existing zero-memory gated set:

- **A** — full-document text prefill. The ceiling. (Already have it.)
- **`cache_inject`** — the Chain-1 mechanism: capture the full layer-12+ KV cache, inject,
  run 12→80 reading the *stored* cache. This is `latent_all` from Proof 5, the ~1.0
  condition. The reference the residual path must match.
- **`residual_inject`** — the new path: capture *only* the true layer-12 residual stream
  `(N, d)`, inject it as the layer-12 input, and let the model **recompute** layers 12→80's
  KV from it, rather than reading a stored cache. Then ask the question.

The comparison is `residual_inject` vs `cache_inject` (and both vs A).

**The one mechanism detail that matters.** In `cache_inject`, the query attends to stored K/V
at every layer 12→79. In `residual_inject`, the query attends to K/V the model *computes on
the fly* at each layer from the injected residual flowing upward. Concretely: inject the
`(N, d)` residual at layer 12's input for the N document positions, then run the forward
normally — layers 12→79 process those positions and produce their own KV, which the query then
attends to. The difference from Chain 1's plumbing is that you're *not* pre-populating the
cache at layers 13→79; you let the forward fill them. Everything else — `position_ids`, the
query offset, the attention sink, think-suppression-off — stays exactly as the validated
split-forward.

**What to measure:**
1. `residual_inject − cache_inject` (judge, strict) — **the headline.** ≈0 = the small object
   suffices.
2. `residual_inject − A` — does the residual path still equal reading (it should, if #1 is ≈0
   and cache ≈ A).
3. **Per-item agreement, not just aggregate.** Because this is partly a correctness check,
   don't only compare pass-rates — check whether the two paths agree *item by item*. If
   aggregates match but individual items disagree (one passes cache, fails residual, and vice
   versa, netting to zero), that's a *different* and more worrying result than clean
   agreement — the paths diverge and happen to average out. Report the confusion: items where
   cache✓/residual✗ and cache✗/residual✓.
4. **Raw-state numerical drift** (optional, cheap): for a few items, capture the layer-13…79
   KV *both* ways (stored vs recomputed-from-residual) and measure cosine/MSE between them.
   Tells you whether any behavioral gap is numerical drift or something structural. Not
   required for the verdict, but it's the diagnostic if #1 comes back non-zero.

**Interpretation, fixed in advance:**
- `residual_inject ≈ cache_inject ≈ A`, clean per-item agreement → **the residual stream is a
  sufficient handoff target.** The SLM's job is well-defined and *small* (produce one `(N, d)`
  tensor at layer 12). Green light to Proof 2.1. The expected and hoped-for outcome.
- `residual_inject < cache_inject` → recomputing the upper stack from the layer-12 residual
  *loses* something the stored cache had. Two sub-cases, distinguished by #4: if the
  recomputed KV is numerically far from stored → a plumbing/precision bug, fix it; if
  numerically close but behavior differs → something genuinely lives in the stored upper-layer
  cache that the layer-12 residual doesn't determine (surprising, needs investigation), and
  the SLM would be on the hook for a bigger object.
- Aggregates match but per-item disagreement is high → the paths aren't equivalent, they're
  differently-lossy; investigate before trusting either.

**Why this must run before Proof 2.1.** Proof 2.1 fits a transformation from Qwen's states to
DeepSeek's *target*. That target has to be defined. If the residual stream is sufficient (2.0
passes), the oracle map aims at `(N, d)` residuals — tractable. If only the full cache works
(2.0 fails), the oracle map would have to produce 68 layers of cross-family, per-layer KV —
effectively hopeless. So Proof 2.0 literally determines whether Proof 2.1 is a reasonable
experiment or a doomed one. **You cannot design the sender until you know the smallest object
the receiver accepts.**

**Scope and cost.** Trivial. Single model, no training, reuses the existing gated set,
scorers, and q-fair capture. Implemented in `proofs/p2_0_residual.py` (mirrors the `p5`
gate → eval → verdict structure; checkpointed and resumable). The one new mechanism is
`core.split_forward.recompute_doc_cache_from_residual`: it takes the layer-12 residual
`Y_doc` that `capture_doc_cache` already returns and runs *only* layers 12→L-1 on it,
filling a genuinely **empty** upper cache — an independent recompute, not the full forward.
Run:

```
CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p2_0_residual.py --arm synth_multihop \
    --synth-n 40 --out proofs/data/p2_0.json
```

The correctness half is provable **without the 70B**: `python core/selftest_split_forward.py`
now includes invariant **G**, which asserts the residual recompute reproduces the
stored-cache split-forward token-for-token on a tiny CPU model. A G failure means the
plumbing is broken — don't spend H200 time on the behavioural run until it passes.

**The one implementation risk to watch.** `residual_inject` must recompute the upper cache
from `Y_doc` with a genuinely **empty** upper cache — the failure mode is accidentally
leaving stored upper-layer KV in place, so you're secretly testing a hybrid and #1 passes for
the wrong reason. `recompute_doc_cache_from_residual` guards this structurally (it clears
every layer and asserts length 0 before the recompute, length N after); the per-item
agreement check (#3) and the KV-drift diagnostic (#4) are the behavioural guards on top.

**Headline to watch:** `residual_inject − cache_inject`, judge, with per-item agreement. ≈0
with clean agreement → the SLM has a small target and Chain 2 proceeds. Anything else → you've
found a structural fact about the handoff object before spending a single training run on it.

## Proof 2.1 — Cross-family geometry exists (oracle map, no training)

**Claim:** there is *some* transformation of the SLM's document representation the LLM can
reason over. **Experiment:** take Qwen's document states and DeepSeek's true layer-12 residuals
(the target 2.0 defined) for the *same* documents; fit an **oracle** map (even overfit, even
per-document) and inject the mapped states via the validated residual path. **Pass:** the
oracle-fit map lets DeepSeek recall the fact → the spaces are bridgeable, translation is a
learnable problem → green light to 2.2. **Fail:** even an overfit oracle map fails → the
cross-family geometry is fundamentally misaligned at layer 12 and no stitcher will save it.
This is the **falsifier for the whole outsourcing thesis**, and it uses **zero training** — the
cheapest way to kill or greenlight Proof 2.2.

## Proof 2.2 — Learned outsourcing (the SLM stitcher, full handoff)

**Claim:** a *trained* SLM→LLM map produces layer-12 residuals good enough that recall
survives — at **full-document** handoff. This is the real Proof 6, reshaped by everything
learned: **full handoff** (sparse is dead — Proof 5), **residual-stream target** (small object
— Proof 2.0), **behavioral objective** — KL between the injected model's logits and
full-prefill's logits, *not* cosine (the lesson the post-mortem proved: cosine certifies
nothing). Trained on the geometry Proof 2.1 proved bridgeable. **Pass:** stitched states
recover the synthetic facts and fail the wrong-document control. **Fail:** translation is the
bottleneck after all — now known to be *real* (not the architecture), worth investing in.

## Proof 2.3 — The honest economics

**Claim:** SLM-reads-then-LLM-reasons actually costs less than LLM-reads-itself, at equal
accuracy. **Experiment:** measure SLM prefill + translation + the LLM's **upper-stack-only**
forward, vs. LLM full prefill — including the cost the residual path incurs by making the LLM
*recompute* layers 12→80 (you skipped 0–11, not the whole read). **Pass:** cheaper at equal
accuracy. **Fail:** "outsourceable" is true but pointless — the same shape of honest stop as
Proof 5, one level up.

## Proof 2.4 — Error amplification (the sender's real risk)

**Claim:** the SLM's approximation error, injected at layer 12, survives 68 layers of recompute.
**Why its own rung:** unique to the residual path — a stitched (approximate) layer-12 state gets
*amplified* through the recomputed upper stack, unlike an injected cache where the approximation
stays local. This is the thing most likely to break Proof 2.2 in practice. **Experiment:**
measure recall vs. injected-state error, find the tolerance, and check whether the stitcher's
achievable accuracy (from 2.2) lands inside it.

## Proofs 2.5+ — The frontier

The old Chain-1 Proofs 7–9, now correctly placed one level up: cross-family **at length** (does
the shallow layer hold across tokenizers at 32k), the honest baseline-to-beat, and whether the
whole thing beats the boring alternative — **SLM extracts text, feed it to the LLM as tokens**.
The Proof-5 ghost, back one level up: does *outsourced latent* beat *outsourced text*? Expect
this to be the hard rung again.

---

## Chain 2 dependency structure

- **2.0 gates the sender's target** — you cannot design the SLM until you know the smallest
  object the receiver accepts. Single-model, cheap, runnable now.
- **2.1 is the outsourcing falsifier** — if an oracle map can't bridge the geometry, no trained
  one will. Zero training.
- **2.2 is the only rung that trains the SLM** — earned only after 2.0 defines the target and
  2.1 proves it reachable.
- **2.3 + 2.4 size and stress the win** — is it cheaper, and does the approximation survive
  recompute.
- **2.5 proves you beat the boring baseline** — outsourced latent vs outsourced text, the
  hard rung.

The first runnable thing is **Proof 2.0**: cheap, single-model, no SLM, no training, reusing
the existing `p5`-style harness. It's the afternoon that tells you whether the entire
outsourcing chain aims at a small target or a hopeless one.

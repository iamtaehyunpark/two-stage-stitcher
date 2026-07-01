# Proof 5 — Latent handoff vs. text-RAG

*Result: 2026-07-01. Verdict: **honest stop for the budget-matched sparse handoff.***

---

## The claim under test

Injecting a document's latent KV beats handing the reasoner retrieved **text**, on
questions that require integrating information retrieval would fragment — **at equal
information budget**. If true, the latent pipeline is justified over plain RAG and the
sender (Proof 6) is worth building. If latent ≈ text, the machinery is an expensive
reimplementation of retrieval.

Operating point (pinned from 4.1): **layer 12, think-ON, q-fair capture, original RoPE.**
Scored three ways — `lenient` / `strict` (deterministic, bracketing) and an **LLM-judge**
(authoritative). Multi-hop is a reasoning claim, so reasoning is enabled.

## The grid (Proof 5.1, the tightened version — [`../proofs/p5_1.py`](../proofs/p5_1.py))

| family | condition | what it hands over |
|---|---|---|
| latent | `latent_all` | full document KV (Path-1 reference, ≈ A) |
| latent | `latent_goldspan` | the gold supporting-sentence positions only (budget-matched anchor) |
| latent | `latent_decimated` | every-other position, keep-rate 0.5 (blind, needle_decimated) |
| text | `text_goldspan` | the gold sentences as text (same spans as `latent_goldspan`) |
| text | `text_decimated` | every-other token, keep-rate 0.5 (same kept tokens as `latent_decimated`) |
| text | `text_rag@best` | a real BGE-large retriever, best of k∈{4, budget} |
| — | `A`, `C` | full-document ceiling / question-only floor |

Data: HotpotQA distractor ([`../proofs/hotpot.py`](../proofs/hotpot.py)) for the real arm,
and zero-memory synthetic 2-hop + single-hop controls
([`../proofs/synth_multihop.py`](../proofs/synth_multihop.py)). Every number is on the
**gated set** (closed-book C fails ∧ full-document A succeeds).

## Results (n=40 zero-memory synthetic 2-hop, judge primary)

```
condition          lenient  strict_fixed  judge
A                     1.00       1.00      1.00
latent_all            0.97       0.97      0.97     ← receiver works: full handoff ≈ reading
latent_goldspan       0.88       0.88      0.62     ← the budget-matched sparse handoff
text_goldspan         1.00       1.00      1.00
latent_decimated      0.12       0.12      0.30
text_decimated        0.00       0.00      0.35
text_rag@4            1.00       1.00      1.00
```

Headline numbers (judge):

1. **Sanity anchor — `dec_latent − dec_text` on coreference docs = +0.433** (`dec_latent`
   0.43, `dec_text` 0.00; Proof 4.1 dec arm, keep-rate 0.5). The harness **reproduces Exp
   3.1** (+0.58): decimated *latent* survives where decimated *text* collapses. The
   instrument is sound — the pre-registered gate passes.
2. **The deficit — `latent_goldspan − text_goldspan` = −0.375** (judge; 0.62 vs 1.00). At
   equal budget, the sparse latent handoff **underperforms clean text of the same spans**.
   Consistent across every validated run (−0.375 / −0.39 / −0.45).
3. `latent_decimated − latent_goldspan` = −0.325 — *not* budget-matched (`latent_decimated`
   keeps ~3.3× more positions), and uninformative on these docs (see caveat 2 below).
4. **The deficit is uniform across answer length** — −0.38 at 2–3 tokens, −0.33 at 4+.
   Not concentrated in single-token answers, so it is **not** a "can't deliver one-token
   facts" artifact; it is a broad ~−0.38 loss.

On the real **HotpotQA** arm the gap is a near-tie (`latent_goldspan − text ≈ −0.07`,
judge) — but that is the memory floor lifting everything (43% closed-book discard even
after gating), which is exactly the ambiguity the zero-memory synthetic control exists to
break. The synthetic control breaks it **against** latent.

## The pattern that resolves it

- **latent beats *degraded* text** — on coref docs where decimation shreds text's
  coreference, `dec_latent` (0.43) ≫ `dec_text` (0.00). The 3.1 result is real and
  replicates.
- **latent loses to *clean* text** — on the same-content gold spans, `latent_goldspan`
  (0.62) < `text_goldspan` (1.00).

Latent's advantage exists **only against crippled text**, not against well-formed text of
the same content. In deployment you hand the reasoner clean retrieved text (RAG), never
shredded text — so the advantage does not translate to a win over the boring baseline.

## Verdict: honest stop

**The receiver is validated** (Proofs 0–4: reading is transferable; `latent_all ≈ A`
here confirms full-document handoff ≈ reading). But **Proof 5's bar — latent ≥ text-RAG at
equal budget — is not met.** The budget-matched sparse handoff loses to clean text of the
same spans by ~0.38 (judge), uniformly across answer lengths, on zero-memory data. Latent
only beats text once text is decimated, and shredded retrieval is not a thing anyone
deploys.

Per the pre-registered decision rule ((1) positive ∧ (2) negative → stop), this is the
clean, falsifiable stop: **latent handoff is transferable but, at equal budget, equivalent-
to-worse than well-formed retrieval on multi-hop reasoning.** The economic case for the
stitcher *over RAG* via a sparse handoff is not there. Proof 6 is **not** motivated by this
result; if the project continues, it is on a different premise (e.g. full-document handoff
economics, `latent_all`), not the budget-matched sparse win.

## Caveats, recorded honestly

1. **Scoring is the hard part, and the judge is the number.** Three deterministic scorers
   bracketed `latent_goldspan` from 0.47 (old strict, too harsh on reasoning-aloud) to 0.88
   (`fixed_strict`, over-crediting like `committed` before it); the LLM-judge (0.62) is the
   authority and is consistent across runs. Report the judge; treat string scorers as
   bracketing diagnostics only.
2. **The decimated grid can't be read on answer-in-span docs.** `needle_decimated` deletes
   the answer's own tokens on synth_multihop, collapsing both `latent_decimated` and
   `text_decimated` to ~0.3 — so (1)/(3) there are structural noise. The 3.1 replication (1)
   must be, and was, run on **coref docs** (answer in the decimatable surroundings).
3. **Adversarial distractors may inflate the magnitude.** The near-miss distractors share
   surnames/affixes, a hard discrimination; the absolute deficit may be smaller with easier
   distractors, but the **direction** (latent < clean text) is robust across scorers, doc
   sets, and sample sizes.
4. **Environment fragility cost several runs.** `core/split_forward.py` uses transformers'
   private `DynamicCache` API; **transformers 5.x silently broke the subset-injection path**
   (`latent_goldspan → 0` while full-inject/text kept working, plus a byte-token decode
   regression), and an env rebuild to python 3.12 broke 32k-context inference (compiled
   accelerate hooks + meta tensors). Every result here is on **transformers 4.x** (pinned in
   `requirements.txt`), re-validated by Proof 0 (clean decode) and Proof 3 (needles-only
   0.84). The subset-vs-full **canary** in `p5_1.py` now aborts loudly on a broken path.

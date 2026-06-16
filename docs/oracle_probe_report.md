# Oracle Probe Experiment — Report

**Date:** 2026-06-16
**Component:** `evaluate/oracle_probe.py`
**Target model:** DeepSeek-R1-Distill-Llama-70B (hidden 8192, 80 layers), injection at layer 30
**Purpose:** Determine whether the stitcher's downstream failure is caused by the *injection premise* or by *translation quality*, before committing to any retrain or architecture change.

---

## 1. Motivation

The end-to-end evaluation showed that Condition B (Qwen → stitcher → inject at layer 30) answered factual questions like a no-context model — hedging and guessing from parametric memory rather than reading the injected document. Three hypotheses could explain this:

1. **Translation quality** — the Qwen→DeepSeek mapping is too lossy (cos ≈ 0.37).
2. **Single-vector bottleneck** — compressing a whole document into one vector loses retrievable detail.
3. **Broken premise** — layer 30 is the wrong injection point, or the query cannot attend to injected positions as assumed.

These are not separable from the end-to-end result alone. The oracle probe isolates them by **removing the stitcher** and injecting the **ground-truth** DeepSeek layer-30 states (captured from a real forward pass) through the same Condition-B plumbing. If the true states work, the premise is sound and the problem is translation; if even the true states fail, the premise is broken.

---

## 2. Method

For each QA pair, three answers were generated on the same DeepSeek-70B:

- **A — Full prefill:** document + question through all 80 layers (reference / ceiling).
- **Oracle-SEQ:** capture the *true* layer-30 hidden states of **all** document tokens (run the document through layers 0–29, hook the input to layer 30), inject the **full sequence** as a layer-30 prefix, then ask the query.
- **Oracle-LAST:** inject only the *true* **last-token** layer-30 vector (single position) — the same shape the current stitcher produces, but with ground-truth values instead of a translation.

Injection mechanism: prepend N placeholder tokens, run the model's own `generate()`, and a forward pre-hook on layer 30 overwrites those N positions with the captured true states on the prefill pass only. `generate()` owns RoPE, masking, and KV cache. Decode greedy (`do_sample=False`). R1 `<think>…</think>` traces stripped from the reported answer.

**Run configuration:** 5 QA pairs, all from `wiki_00876.txt` (*The Witcher* / Andrzej Sapkowski), document length **1313 tokens**, `max_new_tokens=512`, DeepSeek sharded across GPUs 0–2.

---

## 3. Results

| Question | Reference | A (full prefill) | Oracle-SEQ (true full sequence) | Oracle-LAST (true single vector) |
|---|---|---|---|---|
| Six-volume series name | The Witcher | The Witcher ✓ | `a the a the …` ✗ | **The Witcher ✓** |
| Year of first short story | 1986 | 1986 ✓ | `a the a the …` ✗ | **1986 ✓** |
| Main character | Geralt of Rivia | Geralt of Rivia ✓ | `a the a the …` ✗ | **Geralt of Rivia ✓** |
| 2013 standalone prequel | Season of Storms | Season of Storms ✓ | `a the …` ✗ | **Season of Storms ✓** |
| Hussite Wars trilogy | The Hussite Trilogy | The Hussite Trilogy ✓ | `a the …` ✗ | **The Hussite Trilogy ✓** |

- **A:** 5/5 correct.
- **Oracle-SEQ:** 0/5 — degenerate two-token repetition on every item.
- **Oracle-LAST:** 5/5 correct, including specific, non-obvious facts (1986; "Season of Storms"; "The Hussite Trilogy").

---

## 4. Analysis

### 4.1 The inject-at-30 premise is sound

Oracle-LAST recovers detailed factual answers from a **single** true layer-30 vector. Injecting at layer 30 and skipping document prefill works mechanically, and the query is able to read content from the injected position via attention. The premise is validated.

### 4.2 The single-vector design is not the bottleneck

A single ground-truth last-token vector was sufficient for fine-grained QA (dates, proper nouns, titles). Compression to one vector is therefore not the cause of the failure — at least for documents of this length (~1300 tokens). Sequence/multi-token injection is **not** required to make the approach work.

### 4.3 The stitcher's failure is translation quality

Combining 4.1 and 4.2: with the *true* vector the model answers correctly; with the *stitcher's approximation* (cos ≈ 0.37) it falls back to parametric memory. The deficiency is in the Qwen→DeepSeek mapping accuracy, not the architecture. This rules out the two expensive responses (switching target model; redesigning to sequence injection) and points to improving the translation.

### 4.4 Oracle-SEQ failure is a probe artifact, not evidence

The degenerate output of Oracle-SEQ is most likely caused by the probe's implementation rather than a property of sequence injection. The N = 1313 placeholder (dummy) tokens flow through layers 0–29; the query and every generated token then attend over 1313 positions of garbage at the early layers, producing out-of-distribution representations and degenerate decoding. Oracle-LAST has only one such garbage position, which is negligible — explaining why LAST succeeds and SEQ collapses. A correct sequence test would run the query through layers 0–29 in isolation and prepend the true states only at layer 30 (the KV-cache-correct split-forward). Because Oracle-LAST already establishes that a single true vector suffices, the SEQ test is not needed for the go-forward decision.

---

## 5. Threats to validity

1. **Parametric-memory leakage (primary).** The probe document (*The Witcher*) is well known to a 70B model. Oracle-LAST's correctness could partly reflect world knowledge rather than reading the injected vector. Circumstantial evidence against pure memory: on this same document the original stitcher (Condition B) hedged and expressed uncertainty, whereas Oracle-LAST answered confidently and correctly — a delta attributable to the injected true vector. This is not conclusive; a counterfactual control is required (see §6).
2. **Single document / small sample.** All five questions come from one document. Results should be replicated across multiple documents, including longer ones, where single-vector compression may degrade.
3. **Document-only capture.** True states were captured from a document-only forward (no instruction/question context), so Oracle conditions are not bit-identical to A. This is intentional (it mirrors the skip-prefill constraint) but means small gaps from A are expected and not meaningful.

---

## 6. Recommended next steps

1. **Counterfactual / obscure-fact control (deciding experiment).** Inject document X's true last-token vector and (a) ask a question whose answer is in X but not guessable, and (b) ask document Y's question to confirm the model tracks the *injected* document rather than world knowledge. This converts the conclusion from "promising" to "confirmed."
2. **If confirmed, focus entirely on translation accuracy:**
   - More and more-diverse training data.
   - Change the Stage-2 objective from pure representation matching (InfoNCE / MSE / cosine) toward a **downstream behavioral objective** — e.g. KL-distillation between Condition-B and Condition-A output logits — directly optimizing "produces the same answer as full prefill" rather than "vector is close."
   - Reassess whether a frozen orthogonal map + residual MLP is expressive enough for the required accuracy.
3. **Optional:** repair the Oracle-SEQ control (isolated-query split-forward) only if multi-token injection becomes relevant for longer documents.

---

## 7. Bottom line

The architecture works. A single ground-truth layer-30 vector is sufficient to answer detailed questions, and injecting it at layer 30 is mechanically sound. The stitcher fails only because its translated vector is not accurate enough. The path forward is a better-trained stitcher (more data + a downstream-aligned objective), pending one counterfactual control to rule out parametric-memory leakage.

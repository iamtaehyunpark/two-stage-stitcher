# Proofs 0–2 — the receiver-validation chain

One question, three experiments, run in order: *does a frozen DeepSeek-70B reason
over injected layer-30 representations of content it never tokenized — for real,
not as a memory artifact?* No SLM, no stitcher, no translation. Only the LLM and
its own true states. Green across all three licenses the translation work; red at
any rung is a cheap, decisive stop.

See [`../docs/proof_chain.md`](../docs/proof_chain.md) for the full claim/pass/fail
rationale and how these gate Proofs 3–6.

## The keystone

Everything imports [`core/split_forward.py`](../core/split_forward.py) — the
two-cache split-forward that hands the document to the model as true
layer-`target_layer` states and runs only the query through the early layers. The
document is absent from layers `0 … target_layer-1` (we never pay to read it
there) and present from `target_layer` up as its true representations; the query
runs the lower stack alone with its `position_ids` offset to sit after the
document, and the two join at `target_layer`. Because the KV cache length then
differs per layer, this cannot be a single `generate()` call — it is hand-rolled.

## Run order

**1 · Certify the mechanism on CPU first (seconds, no GPU).** Catches RoPE / mask /
cache / decode bugs before any H200 time:

```bash
python core/selftest_split_forward.py
```

Three exact invariants must pass: the no-document reduction (split-forward with an
empty document == ordinary generation), the prefix-cache identity (document present
at every layer, query at offset positions == ordinary prefix-cached generation), and
the subset-to-all identity (subsetting the cache to *every* position is a no-op,
certifying the Proof-3 slice + `n_doc_cached` bookkeeping). If any fails, fix
`core/split_forward.py` — do not proceed.

**2 · Run the chain on the box with the 70B (one model load).**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python proofs/run_chain.py --out-dir proofs/data
```

Proof 0 runs first as a **hard gate**: if the plumbing isn't trustworthy, Proofs
1–2 are not attempted. There is no SLM in Proofs 0–2, so all four GPUs hold
DeepSeek shards (`--gpus 0,1,2,3`).

Individual rungs (each loads the model itself):

```bash
python proofs/p0_plumbing.py   --out proofs/data/p0.json   # memorized doc: SF-true ≈ A
python proofs/p1_premise.py    --out proofs/data/p1.json   # synthetic facts: inject succeeds where C fails
python proofs/p2_falsifier.py  --out proofs/data/p2.json   # wrong-doc control: inject-wrong must fail
python proofs/p3_path.py --layer 12 --out proofs/data/p3.json   # travels light? needles-only vs all-N vs random
```

**3 · Resolve the path (Proof 3) at the winning layer.** Fixed at layer 12 (the
1.00-recall winner). Per gated question it injects the full trace (all-N), only the
answer-bearing token positions (needles-only), the same count of off-needle
positions (random-subset, the control), and the single last needle token. The
needle span is taken from each QA's `needle` text, mapped to its *original* token
positions, and injected at those positions — never renumbered (RoPE correctness).
Add `--curve` to map recall vs #needle-positions; `--no-sink` to drop the attention
sink from the sparse conditions.

**4 · Experiment 3.1 — latent-decimation vs. text-decimation.** Does an injected
layer-12 state carry context its raw token does not? Thin each document two ways and
compare the recall-vs-keep-rate curves: drop positions as **text** (re-tokenize the
survivors, prefill) vs. as **latent** (inject only the survivors' true states). The
divergence is the result — if latent holds while text collapses, latent ≠ text.

```bash
# ALWAYS run the sanity gates first and read them (the canary catches position bugs):
CUDA_VISIBLE_DEVICES=0,1,2,3 python proofs/e31_decimation.py --layer 12 --sanity-only
# then the full sweep + plot:
CUDA_VISIBLE_DEVICES=0,1,2,3 python proofs/e31_decimation.py --layer 12 \
    --out proofs/data/e31.json --plot proofs/data/e31.png
```

Conditions: `dec_text`, `dec_latent`, and `dec_latent_renumbered` (the same kept
states RoPE-re-rotated to contiguous positions — the position-isolation control) ×
`strided`/`random` patterns × `needle_protected`/`needle_decimated` variants, over
keep-rates {1, ½, ¼, ⅛, 1/16}. Long coreference documents
([`synthetic_docs_long.py`](synthetic_docs_long.py)) put the answer entity in the
*decimatable* surroundings and refer to it obliquely in the needle, so decimated
text collapses (the antecedent is gone) while full-context latent states may carry
the resolved coreferent. The canary — `dec_latent` at keep-rate 1 must equal
`full_latent` — is the first thing to check: if it fails, the position/decimation
bookkeeping is wrong and the sweep is noise.

**5 · Proof 4 — length scaling.** Does everything validated at ~130 tokens survive as
documents grow to the lengths the project is for (2k … 32k)? Each synthetic fact block
is padded to a target length with **inert filler** ([`long_context_docs.py`](long_context_docs.py)
— entity-free, digit-free prose) and planted at a chosen needle depth. The trap unique
to Proof 4 is *filler contamination* — padding that accidentally carries a cue — so
inertness is re-checked **per length** by a `C_filler` gate (prefill the padded doc with
the fact *removed*; the model must still fail). A cell is gated iff `C` fails, `C_filler`
fails, and `A` succeeds.

Run it **staged** — the first curve tells you where to spend; don't run the full grid blind:

```bash
# Stage 1 — the gate (cheapest): inject-all-N at L12, depth 50%, across lengths.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python proofs/p4_length.py --stage curve \
    --lengths 500,2000,8000,16000,32000 --layer 12 --out proofs/data/p4_curve.json

# Stage 2 — ONLY at the length where recall first dropped: re-sweep the layer.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python proofs/p4_length.py --stage relayer \
    --lengths 8000,16000 --layers 8,12,20,30 --out proofs/data/p4_relayer.json

# Stage 3 — depth, sparse handoff, latent-vs-text, at the winning layer.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python proofs/p4_length.py --stage axes \
    --lengths 2000,8000,16000 --depths 0.1,0.5,0.9 --layer 12 --out proofs/data/p4_axes.json
```

The headline is **recall vs length**; the verdict (`HOLDS_AT_LENGTH` /
`DROP_AT_LENGTH_RESWEEP_LAYER` / `RESCUED_BY_RELAYER` / `DECAYS_AT_LENGTH`) keys on
inject-all-N staying ≥ 0.8 as length grows.

**Memory at length.** By default the 70B is sharded across **every visible GPU**
(`--gpus` to restrict) with `device_map=balanced_low_0`, which spreads the layers
**evenly** and keeps GPU 0 lightest. This matters: the old `sequential` map fills each
GPU to its cap before using the next, so a ~140 GB bf16 70B packs into the **first two**
80 GB GPUs and leaves GPU 0 no room for the 32k prefill's activation — the cause of the
OOM. On launch the runner prints the per-GPU layer/memory placement so you can confirm
all GPUs are loaded (if only ≤2 are, it warns). It also frees each document's KV cache
before the next capture/prefill and sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to fight fragmentation. Still OOM at
the top length? Add GPUs, drop a per-GPU cap with `--max-mem-per-gpu`, or trim the top
length. `--max-doc-tokens` (default 40000) is the truncation cap for capture/prefill —
keep it above your longest length (and under the model's 128k context). The static filler
selftest runs first (`python proofs/long_context_docs.py`); the behavioural inertness
gate runs per cell.

The bigger bank these draw on lives in [`fact_bank.py`](fact_bank.py): **50 adversarially
authored facts across 10 docs** (30 answer-in-span + 20 coreference), mixing fact types
(name / date / number / multitoken / common-word / relation), with native near-miss
distractors per doc and a forced lexical gap between question and needle. This is the
*generalization* infrastructure Proof 5's "latent beats text-RAG" claim needs (five facts
measured six ways is still five facts); a pure-string `selftest_bank()` enforces every
authoring invariant. Run `python proofs/fact_bank.py` after editing it.

**5.1 · Proof 4.1 — hardened confirmation.** Proof 4's 32k recall of 1.00 is almost too
clean, so before Proof 5 we re-test the most stressful point (32k / L12) with the
evaluation slack removed, **pooled over the span docs × depths for n well past 30
independent facts** (the first run had n=3 — too few to resolve a 33% effect)
([`p4_1_hardened.py`](p4_1_hardened.py)):

- **strict scoring** — `lenient` (containment), `firstline` (gold in the answer clause),
  `strict` (gold in the clause **exclusively** — no competing decoy value, no
  negation/hedge). "Velloth, not Vask" / "either Velloth or Vask" fails strict; a clean
  correct sentence passes. The lenient−strict delta is the inflation.
- **capture/A symmetry** — `inject_docnaive` (document-only capture) vs `inject_qfair`
  (captured inside the same instruction+document framing A sees); honest ceiling
  comparison is `inject_qfair` vs `A`.
- **distractor filler** — near-miss decoys (same surface form, wrong values; native to
  every doc in `fact_bank.py`) woven in at other depths; re-gated (C, C_filler fail; A
  succeeds), so a correct answer must *discriminate* the true needle.
- **reasoning on** — a think-on arm for A / inject (Proof 5's actual path).

Two arms, reported separately: the **discrimination arm** above, and a **latent-vs-text
arm** on the Exp-3.1 coreference docs (answer in the *decimatable* surroundings) thinned
to the keep-rate where `dec_text` **collapses** — the only setting where latent>text is
a meaningful claim. The verdict is split into `discrimination` (`PARITY_WITH_A` /
`RECOVERS_NOT_PARITY` / `COLLAPSED_UNDER_DISTRACTORS` / `UNDERPOWERED`) and `mechanism`
(`LATENT_BEATS_TEXT` / `TEXT_DID_NOT_COLLAPSE` / `LATENT_ALSO_COLLAPSED`), combined into
`SHIP_TO_PROOF_5` / `HOLD`. It refuses to claim parity below n ≥ 30 and refuses to read
the latent−text gap unless text actually collapsed.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python proofs/p4_1_hardened.py \
    --length 32000 --layer 12 --depths 0.1,0.5,0.9 --out proofs/data/p4_1.json
# faster first look: one depth, no think-on, skip the collapse arm
python proofs/p4_1_hardened.py --depths 0.5 --no-think-on --no-dec
# if dec_text doesn't collapse at 0.5, thin harder:
python proofs/p4_1_hardened.py --dec-keep-rate 0.25
# re-score a saved run with the current scorers (no GPU — iterate scorers free):
python proofs/p4_1_hardened.py --rescore proofs/data/p4_1.json
```

The raw per-condition answers are saved in `p4_1.json`, so `--rescore` regenerates the
whole table (per-record decoys keyed by doc) with no model. The number to read first is
**`dec_latent − dec_text` (strict) on the collapse arm** — if the Exp-3.1 mechanism
survives there at n ≥ 30, nothing else in the table can sink the project. **This run is
expensive** (many 32k cells); scope it with `--depths` / `--docs` / `--no-dec` for a
quick pass before the full grid.

**6 · Proof 5 — latent handoff vs. text-RAG.** The project's existential test, on REAL
multi-hop questions (HotpotQA distractor): does injecting a document's latent KV beat
handing the reasoner *retrieved text* at equal information budget? If latent ≈ RAG, the
latent machinery is an expensive reimplementation of retrieval. Operating point pinned
from 4.1: **L12, strict, q-fair, think-ON**. Three new pieces feed the runner
([`p5_latent_vs_rag.py`](p5_latent_vs_rag.py)):

- [`hotpot.py`](hotpot.py) — concatenates each item's 10 paragraphs in order, locates the
  gold supporting *sentences* (recurrence-safe, by recorded char range) → `needle_idx`.
  CPU prep, cached. `python proofs/hotpot.py --mock` checks the span logic with no download.
- [`retriever.py`](retriever.py) — the strong baseline: **BGE-large-en-v1.5** on CPU,
  sentence-aware chunking (size tuned on a held-out slice — tune the baseline to win), a
  k-sweep + budget-matched k, and recall-by-char-containment for failure attribution.
- [`synth_multihop.py`](synth_multihop.py) — the controls: ~20 invented-entity 2-hop items
  (zero memory leakage, breaks a HotpotQA null) **plus** single-hop extraction items where
  latent and text **must tie** (the prompt-asymmetry trap from Proof 4's docnaive).

The gate is the same discipline as everywhere: score only items where closed-book **C
fails** and full-document **A succeeds**. On that set the runner scores the 2×2 of
{gold spans, retrieved spans} × {latent, text} plus the ceilings (A, latent_all), then
reports the four numbers that decide the project — `latent_sparse − text_gold` (1),
`latent_sparse − text_rag@best` raw and retrieval-conditioned (2), `latent_all − A` (3),
and synthetic agreement (4). Word-boundary scoring (not substring) is mandatory here so a
refusal can't score a yes/no answer.

Model placement matches proofs 0–3 / `run_chain`: pick physical GPUs with
`CUDA_VISIBLE_DEVICES` and pass logical `--gpus` (default `0,1,2,3`). Eval is
**checkpointed to `--out` after every item** and resumes from a partial `--out`, so a kill
loses only the in-flight item. `--max-eval` caps how many gated items run this session;
`--think-max-new-tokens` is the per-item cost driver (2-hop answers rarely need 2048).

```bash
# 0. one-time: build the HotpotQA cache on CPU (needs `datasets`) so the GPU runs never
#    import it, and install the retriever dep:  pip install datasets sentence-transformers
python proofs/hotpot.py --max-items 500
# 1. main arm — first verdict overnight: cap at 60 items, 1024-token reasoning budget.
#    Gate is cached (p5_gated_hotpot.json); rerun without --max-eval to extend the eval.
CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p5_latent_vs_rag.py --arm hotpot \
    --max-candidates 400 --max-eval 60 --think-max-new-tokens 1024 --out proofs/data/p5.json
# 2. the synthetic control + the single-hop parity control
CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p5_latent_vs_rag.py --arm synth_multihop --out proofs/data/p5_synth.json
CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p5_latent_vs_rag.py --arm synth_parity   --out proofs/data/p5_parity.json
# cheap end-to-end wire test (no reasoning) before the real run
CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p5_latent_vs_rag.py --arm hotpot --max-candidates 30 --no-think
# re-score saved answers with current scorers (no GPU)
python proofs/p5_latent_vs_rag.py --rescore proofs/data/p5.json
```

The **canary** (inject-all-positions == full inject) must be 0 mismatches before any latent
number is trusted. Run the no-GPU selftests first — `python proofs/hotpot.py --mock`,
`python proofs/synth_multihop.py`, `python proofs/retriever.py` — they gate GPU time the way
`selftest_bank` / `selftest_filler` do for 4.1.

## What each rung decides

| Rung | Document | Decides | PASS | FAIL |
|---|---|---|---|---|
| **0** plumbing | memorized (answers known) | the instrument is trustworthy | SF-true coherent & correct, matching A | degeneration → RoPE/join/cache bug; fluent-but-wrong → query not attending injected positions. **STOP.** |
| **1** premise | synthetic (unguessable) | the receiver reads | inject-all-N succeeds on items where C fails & A succeeds | inject behaves like C → no translation can help. **Stop the project.** |
| **2** falsifier | synthetic, **wrong** doc injected | it's the injection, not memory/leak | wrong-doc fails while matched succeeds | wrong-doc still answers → Proof 1 falsified; find the leak. |

The synthetic-fact bank lives in
[`synthetic_docs.py`](synthetic_docs.py) — fabricated entities/dates so a correct
answer can only come through the injection. Each fact is annotated with its
answer-bearing sentence (`needle`) for Proof 3.

## Notes

- Gates are load-bearing, not decoration: Proof 1 scores **only** items where C
  (no-context) fails *and* A (full prefill) succeeds. A guessable fact (C succeeds)
  or an unanswerable one (A fails) is thrown out, not counted.
- DeepSeek-R1 emits long `<think>` traces; only the post-`</think>` answer is
  scored. Default budget is `--max-new-tokens 512`; raise it if answers truncate
  mid-reasoning.
- Results are written to `proofs/data/*.json` (git-ignored) with full per-item
  transcripts for inspection by eye — the verdict heuristics are a guide, not the
  final word.

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

**5.1 · Proof 4.1 — hardened single-point confirmation.** Proof 4's 32k recall of 1.00
is almost too clean, so before Proof 5 we re-test the single most stressful point
(32k / L12 / depth 0.5) with the evaluation slack removed — one cell, run many ways
([`p4_1_hardened.py`](p4_1_hardened.py)):

- **strict scoring** — report `lenient` (containment), `firstline` (gold in the answer
  clause), and `strict` (gold in the clause, **exclusively** — no competing decoy value,
  no negation/hedge) side by side; the lenient−strict delta is the inflation. Strict is
  the scorer the distractors make meaningful: "Velloth, not Vask" or "either Velloth or
  Vask" fails it, a clean correct sentence passes.
- **capture/A symmetry** — `inject_docnaive` (document-only capture) vs `inject_qfair`
  (capture inside the same instruction+document framing A sees); honest ceiling
  comparison is `inject_qfair` vs `A`.
- **distractor filler** — near-miss decoys (same surface form, wrong values) planted at
  other depths, so a correct answer must *discriminate* the true needle. Re-gated (C and
  C_filler must still fail, A must still succeed).
- **reasoning on** — a think-on arm for A / inject (Proof 5's actual path), alongside
  think-off.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python proofs/p4_1_hardened.py \
    --doc zorvian_codex --length 32000 --layer 12 --out proofs/data/p4_1.json
# --no-think-on to skip the slow reasoning arm
# re-score a saved run with the current scorers (no model, no GPU — iterate scorers free):
python proofs/p4_1_hardened.py --rescore proofs/data/p4_1.json
```

The raw per-condition answers are saved in `p4_1.json`, so `--rescore` regenerates the
whole table without re-running the 32k generation.

Sanity gates run first (subset-to-all no-op; C/C_filler must fail *with* distractors; 5
raw injected answers printed for eyeball). The number to read first is
**`dec_latent − dec_text` under strict scoring with distractors** — if the Exp-3.1
mechanism survives there, nothing else in the table can sink the project. Verdict:
`VINDICATED_HARDENED` / `SCALES_NOT_PARITY` / `EASY_TASK_ARTIFACT` /
`MECHANISM_SCORER_INFLATED`.

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

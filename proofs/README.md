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

Two exact invariants must pass: the no-document reduction (split-forward with an
empty document == ordinary generation) and the prefix-cache identity (document
present at every layer, query at offset positions == ordinary prefix-cached
generation). If either fails, fix `core/split_forward.py` — do not proceed.

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
```

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

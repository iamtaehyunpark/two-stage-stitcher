"""
Proof 4 — Does the reading survive long context?

Proofs 1–3 earned their claims at ~130 tokens: injected layer-12 states are read
(Proof 1), it is the injection and not memory (Proof 2), and the handoff travels
light (Proof 3). Proof 4 asks the only question that decides whether "reading is
transferable" is a claim about LONG documents or an artifact of toy ones: do those
results hold as the document grows to the lengths the project is for (2k … 32k)?

Three things that were free at 130 tokens get expensive at length, and each can
break alone:
  • the shallow-layer shelf may not survive — one layer-12 state has to stand in for
    thousands of positions of processing the model never did; the optimal layer may
    migrate deeper. So we RE-SWEEP the layer at length, never assume 12 transfers.
  • needle depth becomes a variable — a fact at 10% vs 90% depth is a different
    retrieval problem (lost-in-the-middle); injected states may suffer it differently
    than ordinary prefill.
  • the latent>text gap (Exp 3.1) may widen, hold, or COLLAPSE as more skipped
    context has to fold into a single state. That collapse is the decisive unknown.

THE GATES (and the trap specific to Proof 4)
────────────────────────────────────────────
Every cell carries the usual C-floor / A-ceiling gates, PLUS a per-length inertness
gate, because the way Proof 4 inflates its own numbers is FILLER CONTAMINATION: pad a
document to 8k and the padding may, by accident, contain a cue the question matches —
a false floor. So we re-check inertness at every length, never inherit it:

  C        — question only, no document          → must FAIL (fact is unguessable).
  C_filler — prefill the padded doc with the fact REMOVED (filler-only) and ask the
             question                              → must FAIL (the padding is inert).
  A        — full prefill of the padded doc        → must SUCCEED (and, at length, A is
             itself the check that the model can do this retrieval normally — it
             bounds what injection could ever achieve).
A cell is GATED (interpretable) iff C fails AND C_filler fails AND A succeeds.

THE CONDITIONS (per gated cell)
  inject_all_n  — the document handed over as TRUE layer-L states (split-forward).
                  The headline. Re-swept across layers at each length.
  needles_only  — only the answer-bearing positions injected (Proof 3's sparse
                  handoff). Does it survive when the surrounding document is 8k, not 100?
  latent_vs_text— at a fixed keep-rate, dec_latent (inject the survivors' states) vs
                  dec_text (hand the survivors as tokens). The Exp-3.1 contrast at
                  length: does the gap hold?

STAGED — let the first curve tell you where to spend (don't run the full grid blind):
  stage=curve   (cheapest, the GATE)  inject_all_n at one layer (12), depth 0.5,
                across all lengths. Does recall stay ≈1.0 at 2k and 8k?
  stage=relayer (only where curve dropped) re-sweep layers {8,12,20,30} at the
                given length(s). Does re-picking the layer rescue recall? Where is
                the optimal layer as length grows — does the shelf migrate deeper?
  stage=axes    (last) depth {0.1,0.5,0.9} + needles_only + latent_vs_text at the
                winning layer and chosen lengths.

Usage:
    # Stage 1 — the gate. Cheap; tells you if there is even a problem.
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python proofs/p4_length.py --stage curve \
        --lengths 500,2000,8000 --layer 12 --out proofs/data/p4_curve.json

    # Stage 2 — only at the length where recall first dropped, re-sweep the layer.
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python proofs/p4_length.py --stage relayer \
        --lengths 8000,16000 --layers 8,12,20,30 --out proofs/data/p4_relayer.json

    # Stage 3 — depth, sparse handoff, latent-vs-text, at the winning layer.
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python proofs/p4_length.py --stage axes \
        --lengths 2000,8000,16000 --depths 0.1,0.5,0.9 --layer 12 \
        --keep-rate 0.5 --out proofs/data/p4_axes.json

By default the 70B is sharded across EVERY visible GPU (more shards → lower per-GPU
memory, which is what lets the 32k full-prefill fit); pass --gpus to restrict.
"""

import os

# Long-context prefill at 32k fragments the allocator; the expandable-segments arena
# lets a freed cache's blocks be reused by the next, larger allocation instead of
# stranding them. Must be set before the first CUDA init (i.e. before torch loads via
# the imports below), so it lives at the very top.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.split_forward import capture_doc_cache
from proofs.common import (
    load_deepseek, no_context_answer, full_prefill_answer, inject_answer,
    inject_answer_subset, correct,
)
from proofs.needles import span_token_positions, needle_positions
from proofs.decimate import kept_indices, decimated_text
from proofs.long_context_docs import build_padded_doc, selftest_filler
from proofs.synthetic_docs import SYNTHETIC_DOCS
from proofs.synthetic_eval import STRONG_RATE, MIN_GATED

# Recall at/above STRONG_RATE "holds"; reuse the chain-wide bar so Proof 4's verdict
# is on the same scale as Proofs 1–3.
HOLDS = STRONG_RATE


def _report_gpu_placement(model):
    """Print how many decoder layers landed on each GPU and the per-GPU memory now
    held, so you can SEE the 70B is spread across all the GPUs you asked for — not
    packed into the first two (the device_map='sequential' trap that OOMs at 32k)."""
    import torch
    from collections import Counter
    counts = Counter()
    for name, p in model.named_parameters():
        if p.device.type == "cuda":
            counts[p.device.index] += 1
    print("  GPU placement (param tensors / memory):")
    for i in sorted(counts):
        alloc = torch.cuda.memory_allocated(i) / 1e9 if torch.cuda.is_available() else 0
        print(f"    cuda:{i}  {counts[i]:>4} tensors  {alloc:5.1f} GB")
    if len(counts) <= 2:
        print("    !!! only %d GPU(s) hold weights — pass --device-map balanced_low_0 "
              "(weights packed front-first will OOM on long prefill)." % len(counts))


def _free_cuda():
    """Drop dead tensors and return their CUDA blocks to the allocator. Called between
    captures and cells so a long document's KV cache (several GB at 32k) does not
    linger while the next cell's full-prefill peaks — the lingering cache, not the
    model, is what pushes a single shard OOM at the longest lengths."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


# ── one (doc, length, depth) build + its layer-independent gates ───────────────
def prepare_cell(model, tok, base_doc, length, depth, max_doc_tokens, max_new_tokens,
                 c_cache):
    """Build the padded document (with fact) and its filler-only twin, run the three
    layer-independent gates per question, and map each needle to token positions in
    the padded sequence. Returns a dict with everything the per-layer conditions need.

    `c_cache` memoizes Condition C (question only) across cells — it depends on
    neither length nor depth nor layer, so it is computed once per question."""
    padded = build_padded_doc(tok, base_doc, length, depth, max_doc_tokens=max_doc_tokens)
    filler = build_padded_doc(tok, base_doc, length, depth, max_doc_tokens=max_doc_tokens,
                              drop_fact=True)
    print(f"  [{base_doc['name']}] target={length} depth={depth} → "
          f"actual={padded['n_tokens']} tok, fact@depth={padded['depth_actual']}")

    qa_items = []
    for qa in base_doc["qa"]:
        q, gold, span = qa["q"], qa["a"], qa["needle"]

        if q in c_cache:
            ans_c = c_cache[q]
        else:
            ans_c = no_context_answer(model, tok, q, max_new_tokens)
            c_cache[q] = ans_c
        # A and the inertness gate are length+depth specific — never inherited.
        ans_a = full_prefill_answer(model, tok, padded["text"], q, max_new_tokens,
                                    max_length=max_doc_tokens)
        ans_cf = full_prefill_answer(model, tok, filler["text"], q, max_new_tokens,
                                     max_length=max_doc_tokens)

        needle_idx = span_token_positions(tok, padded["text"], span, max_doc_tokens)
        c_ok, cf_ok, a_ok = correct(ans_c, gold), correct(ans_cf, gold), correct(ans_a, gold)
        gated = (not c_ok) and (not cf_ok) and a_ok
        qa_items.append({
            "q": q, "gold": gold, "needle": span, "needle_idx": needle_idx,
            "c": ans_c, "a": ans_a, "c_filler": ans_cf,
            "c_correct": c_ok, "c_filler_correct": cf_ok, "a_correct": a_ok,
            "gated": gated,
        })
        print(f"    {q!r}\n      gold={gold!r}  C={c_ok} C_filler={cf_ok} A={a_ok}  "
              f"gated={gated}")
    return padded, qa_items


# ── the per-cell × per-layer evaluation ────────────────────────────────────────
def evaluate(model, tok, docs, lengths, depths, layers, conditions, keep_rate,
             max_doc_tokens, max_new_tokens):
    """Run the requested `conditions` for every (doc, length, depth, layer, question)
    and return the flat record table. Captures the document cache once per
    (doc, length, depth, layer); the gates are computed once per (doc, length, depth)
    and shared across the layer sweep."""
    records = []
    c_cache = {}
    for base_doc in docs:
        for length in lengths:
            for depth in depths:
                padded, qa_items = prepare_cell(
                    model, tok, base_doc, length, depth, max_doc_tokens,
                    max_new_tokens, c_cache)

                # Tokenize once (ids drive both the capture and the text-decimation
                # arm, so needle indices line up with the cache).
                ids = tok(padded["text"], return_tensors="pt", truncation=True,
                          max_length=max_doc_tokens).input_ids

                for layer in layers:
                    print(f"  capturing {base_doc['name']} @ {padded['n_tokens']}tok "
                          f"depth={depth} layer={layer} …")
                    cache, _Y, n_doc = capture_doc_cache(model, ids, layer)
                    del _Y          # the (1, N, D) layer-input states are unused here

                    for qa in qa_items:
                        rec = {
                            "doc": base_doc["name"], "target_tokens": length,
                            "n_tokens": padded["n_tokens"], "depth": depth,
                            "depth_actual": padded["depth_actual"], "layer": layer,
                            "question": qa["q"], "gold": qa["gold"],
                            "n_doc": n_doc, "k_needle": len(qa["needle_idx"]),
                            "c_correct": qa["c_correct"],
                            "c_filler_correct": qa["c_filler_correct"],
                            "a_correct": qa["a_correct"], "gated": qa["gated"],
                        }

                        if "inject_all_n" in conditions:
                            ans = inject_answer(model, tok, cache, n_doc, qa["q"],
                                                layer, max_new_tokens)
                            rec["inject_all_n"] = ans
                            rec["inject_all_n_correct"] = correct(ans, qa["gold"])

                        if "needles_only" in conditions:
                            pos = needle_positions(qa["needle_idx"], keep_sink=True)
                            ans = inject_answer_subset(model, tok, cache, n_doc, pos,
                                                       qa["q"], layer, max_new_tokens)
                            rec["needles_only"] = ans
                            rec["needles_only_correct"] = correct(ans, qa["gold"])

                        if "latent_vs_text" in conditions:
                            # fixed-thinning Exp-3.1 contrast: keep the same survivor
                            # set for both arms (needle decimated, strided).
                            kept = kept_indices(n_doc, qa["needle_idx"], keep_rate,
                                                "strided", "needle_decimated",
                                                seed=0, keep_sink=True)
                            txt = decimated_text(tok, ids, kept)
                            a_txt = full_prefill_answer(model, tok, txt, qa["q"],
                                                        max_new_tokens,
                                                        max_length=max_doc_tokens)
                            a_lat = inject_answer_subset(model, tok, cache, n_doc, kept,
                                                         qa["q"], layer, max_new_tokens)
                            rec["keep_rate"] = keep_rate
                            rec["kept_count"] = len(kept)
                            rec["dec_text"] = a_txt
                            rec["dec_latent"] = a_lat
                            rec["dec_text_correct"] = correct(a_txt, qa["gold"])
                            rec["dec_latent_correct"] = correct(a_lat, qa["gold"])

                        records.append(rec)
                        _print_rec(rec, conditions)

                    # Release this layer's doc cache before capturing the next layer
                    # (or before the next cell's full-prefill gates), so at most one
                    # long KV cache is resident at a time.
                    del cache
                    _free_cuda()

                del ids
                _free_cuda()
    return records


def _print_rec(rec, conditions):
    bits = []
    if "inject_all_n" in conditions:
        bits.append(f"all_n={rec['inject_all_n_correct']}")
    if "needles_only" in conditions:
        bits.append(f"needles={rec['needles_only_correct']}")
    if "latent_vs_text" in conditions:
        bits.append(f"dec_text={rec['dec_text_correct']} dec_latent={rec['dec_latent_correct']}")
    print(f"    [{rec['doc']}|{rec['n_tokens']}tok|d={rec['depth']}|L{rec['layer']}] "
          f"gated={rec['gated']}  " + "  ".join(bits))


# ── aggregation ────────────────────────────────────────────────────────────────
def _rate(recs, key):
    g = [r for r in recs if r["gated"]]
    return round(sum(r[key] for r in g) / len(g), 3) if g else None


def aggregate(records, conditions):
    """Roll the flat table up into the curves Proof 4 is about: recall vs length (per
    layer), the optimal-layer-vs-length surface, recall vs depth, and the
    latent−text gap vs length — all on gated cells only."""
    lengths = sorted({r["target_tokens"] for r in records})
    depths = sorted({r["depth"] for r in records})
    layers = sorted({r["layer"] for r in records})

    def subset(**kw):
        return [r for r in records if all(r[k] == v for k, v in kw.items())]

    agg = {"lengths": lengths, "depths": depths, "layers": layers,
           "n_gated_total": sum(r["gated"] for r in records)}

    # recall vs length, per layer (and A's own recall — the model's native ceiling).
    by_layer = {}
    for layer in layers:
        row = {}
        for length in lengths:
            cells = subset(layer=layer, target_tokens=length)
            entry = {"n_gated": sum(c["gated"] for c in cells),
                     "a_recall_all": round(sum(c["a_correct"] for c in cells) / len(cells), 3)
                     if cells else None}
            for cond in conditions:
                k = f"{cond}_correct" if cond != "latent_vs_text" else None
                if k:
                    entry[cond] = _rate(cells, k)
            if "latent_vs_text" in conditions:
                entry["dec_text"] = _rate(cells, "dec_text_correct")
                entry["dec_latent"] = _rate(cells, "dec_latent_correct")
            row[length] = entry
        by_layer[layer] = row
    agg["recall_vs_length_by_layer"] = by_layer

    # optimal layer per length (argmax inject_all_n on gated), the shelf-migration surface.
    if "inject_all_n" in conditions and len(layers) > 1:
        opt = {}
        for length in lengths:
            best_layer, best_rate = None, -1.0
            for layer in layers:
                r = by_layer[layer][length].get("inject_all_n")
                if r is not None and r > best_rate:
                    best_rate, best_layer = r, layer
            opt[length] = {"optimal_layer": best_layer, "recall": best_rate}
        agg["optimal_layer_vs_length"] = opt

    # recall vs depth (at each layer × length), the lost-in-the-middle axis.
    if len(depths) > 1 and "inject_all_n" in conditions:
        depth_surface = {}
        for layer in layers:
            depth_surface[layer] = {}
            for length in lengths:
                depth_surface[layer][length] = {
                    d: _rate(subset(layer=layer, target_tokens=length, depth=d),
                             "inject_all_n_correct")
                    for d in depths}
        agg["recall_vs_depth"] = depth_surface

    # latent−text gap vs length (per layer), the economically decisive curve.
    if "latent_vs_text" in conditions:
        gap = {}
        for layer in layers:
            gap[layer] = {}
            for length in lengths:
                cells = subset(layer=layer, target_tokens=length)
                dt = _rate(cells, "dec_text_correct")
                dl = _rate(cells, "dec_latent_correct")
                gap[layer][length] = {
                    "dec_text": dt, "dec_latent": dl,
                    "gap": (round(dl - dt, 3) if dt is not None and dl is not None else None)}
        agg["latent_minus_text_vs_length"] = gap

    return agg


# ── verdict / printout ─────────────────────────────────────────────────────────
def report(agg, conditions, stage):
    print("\n" + "=" * 72)
    print(f"PROOF 4 — length scaling   (stage={stage}, total gated cells = "
          f"{agg['n_gated_total']})")
    lengths = agg["lengths"]

    if "inject_all_n" in conditions:
        print("\n recall vs length — inject_all_N on gated  (A = model's own ceiling)")
        print("   layer │ " + "  ".join(f"{l:>7}" for l in lengths))
        for layer in agg["layers"]:
            cells = [agg["recall_vs_length_by_layer"][layer][l] for l in lengths]
            vals = "  ".join(
                (f"{c['inject_all_n']:>7.2f}" if c.get("inject_all_n") is not None else f"{'·':>7}")
                for c in cells)
            print(f"   {layer:>5} │ {vals}")
        print("   A     │ " + "  ".join(
            (f"{agg['recall_vs_length_by_layer'][agg['layers'][0]][l]['a_recall_all']:>7.2f}"
             if agg['recall_vs_length_by_layer'][agg['layers'][0]][l]['a_recall_all'] is not None
             else f"{'·':>7}") for l in lengths))
        print(f"   n_gtd │ " + "  ".join(
            f"{agg['recall_vs_length_by_layer'][agg['layers'][0]][l]['n_gated']:>7}" for l in lengths))

    if agg.get("optimal_layer_vs_length"):
        print("\n optimal layer vs length (does the shelf migrate deeper at length?)")
        for l in lengths:
            o = agg["optimal_layer_vs_length"][l]
            print(f"   {l:>7} tok → layer {o['optimal_layer']}  (recall {o['recall']:.2f})")

    if agg.get("recall_vs_depth"):
        print("\n recall vs needle depth — inject_all_N  (lost-in-the-middle?)")
        for layer in agg["layers"]:
            print(f"   layer {layer}:")
            for l in lengths:
                row = agg["recall_vs_depth"][layer][l]
                cells = "  ".join(
                    (f"d{d:g}={row[d]:.2f}" if row[d] is not None else f"d{d:g}=·")
                    for d in agg["depths"])
                print(f"     {l:>7} tok: {cells}")

    if "needles_only" in conditions:
        print("\n sparse handoff survival — needles_only on gated vs length")
        for layer in agg["layers"]:
            vals = "  ".join(
                (f"{agg['recall_vs_length_by_layer'][layer][l]['needles_only']:>7.2f}"
                 if agg['recall_vs_length_by_layer'][layer][l].get('needles_only') is not None
                 else f"{'·':>7}") for l in lengths)
            print(f"   layer {layer:>3}: {vals}")

    if agg.get("latent_minus_text_vs_length"):
        print("\n latent − text gap vs length (widen / hold / collapse?)")
        for layer in agg["layers"]:
            print(f"   layer {layer}:")
            for l in lengths:
                g = agg["latent_minus_text_vs_length"][layer][l]
                if g["gap"] is None:
                    continue
                print(f"     {l:>7} tok: dec_text={g['dec_text']:.2f} "
                      f"dec_latent={g['dec_latent']:.2f}  gap={g['gap']:+.2f}")

    # ── headline verdict (driven by inject_all_n, the gate) ───────────────────
    verdict = "N/A"
    if "inject_all_n" in conditions:
        primary = agg["layers"][0]
        curve = [agg["recall_vs_length_by_layer"][primary][l].get("inject_all_n") for l in lengths]
        total_gated = agg["n_gated_total"]
        first_drop = next((lengths[i] for i, v in enumerate(curve)
                           if v is not None and v < HOLDS), None)
        if total_gated < MIN_GATED:
            verdict = "INSUFFICIENT_GATED"
        elif first_drop is None:
            verdict = "HOLDS_AT_LENGTH"
        elif agg.get("optimal_layer_vs_length"):
            rescued = all((agg["optimal_layer_vs_length"][l]["recall"] or 0) >= HOLDS
                          for l in lengths)
            verdict = "RESCUED_BY_RELAYER" if rescued else "DECAYS_AT_LENGTH"
        else:
            verdict = "DROP_AT_LENGTH_RESWEEP_LAYER"

        print("\n" + "-" * 72)
        print(f"  VERDICT: {verdict}")
        if verdict == "HOLDS_AT_LENGTH":
            print(f"   → inject_all_N stays ≥ {HOLDS} at every tested length on layer "
                  f"{primary}. The shelf survives; the method scales. Ship to Proof 5.")
        elif verdict == "DROP_AT_LENGTH_RESWEEP_LAYER":
            print(f"   → recall first drops below {HOLDS} at {first_drop} tok on layer "
                  f"{primary}. Run stage=relayer at that length: re-pick the layer "
                  "before concluding anything.")
        elif verdict == "RESCUED_BY_RELAYER":
            print("   → the fixed layer decayed but a re-swept layer recovers ≥ "
                  f"{HOLDS} at every length. Watch the optimal-layer surface: a layer "
                  "that migrates DEEPER with length is the good outcome (more prefill "
                  "skipped where it matters).")
        elif verdict == "DECAYS_AT_LENGTH":
            print("   → recall decays even after re-sweeping layers. The honest hard "
                  "result: a single injection point can't carry arbitrary length. Not "
                  "a stop — a redirect to multi-slot / per-chunk handoff.")
        else:
            print(f"   → only {total_gated} gated cells (< {MIN_GATED}); inconclusive. "
                  "Check the C/C_filler/A gates — at length the filler-only gate or A "
                  "itself may be disqualifying cells.")
    return verdict


# ── stage presets ──────────────────────────────────────────────────────────────
def stage_config(stage, args):
    """Map a stage name to (lengths, depths, layers, conditions). The CLI flags
    override the defaults so you can target exactly the length the curve flagged."""
    lengths = [int(x) for x in args.lengths.split(",")] if args.lengths else None
    depths = [float(x) for x in args.depths.split(",")] if args.depths else None
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    if stage == "curve":
        return (lengths or [500, 2000, 8000, 16000, 32000],
                depths or [0.5],
                layers or [args.layer],
                ["inject_all_n"])
    if stage == "relayer":
        return (lengths or [8000, 16000],
                depths or [0.5],
                layers or [8, 12, 20, 30],
                ["inject_all_n"])
    if stage == "axes":
        return (lengths or [2000, 8000, 16000],
                depths or [0.1, 0.5, 0.9],
                layers or [args.layer],
                ["inject_all_n", "needles_only", "latent_vs_text"])
    raise ValueError(f"unknown stage {stage!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["curve", "relayer", "axes"], default="curve")
    parser.add_argument("--out", default=None, help="defaults to proofs/data/p4_<stage>.json")
    parser.add_argument("--lengths", default=None, help="comma-separated target token lengths")
    parser.add_argument("--depths", default=None, help="comma-separated needle depths (0..1)")
    parser.add_argument("--layers", default=None, help="comma-separated layers to sweep")
    parser.add_argument("--layer", type=int, default=12,
                        help="fixed injection layer for curve/axes (the L12 winner)")
    parser.add_argument("--keep-rate", type=float, default=0.5,
                        help="thinning keep-rate for the latent-vs-text axis")
    parser.add_argument("--docs", default=None,
                        help="comma-separated synthetic doc names (default: all)")
    parser.add_argument("--max-doc-tokens", type=int, default=40000,
                        help="truncation cap for capture/prefill (raise above your max length)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpus", default=None,
                        help="comma-separated logical GPU indices to shard the 70B "
                             "across; default = ALL visible GPUs. More shards → lower "
                             "per-GPU memory, which is what lets 32k prefill fit.")
    parser.add_argument("--device-map", default="balanced_low_0",
                        help="how to place layers across GPUs. 'balanced_low_0' spreads "
                             "evenly and keeps GPU 0 light (needed for long prefill); "
                             "'sequential' packs front GPUs first (only ~2 GPUs used for "
                             "a 70B — the cause of 32k OOM).")
    parser.add_argument("--max-mem-per-gpu", default="70GiB",
                        help="per-GPU memory cap handed to accelerate when sharding")
    parser.add_argument("--reasoning", action="store_true",
                        help="let R1 emit <think> traces instead of suppressing them")
    args = parser.parse_args()

    if not selftest_filler():
        print("!!! filler is not inert — fix proofs/long_context_docs.py before running.")
        sys.exit(1)

    out = args.out or f"proofs/data/p4_{args.stage}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    import proofs.common as _common
    _common.SUPPRESS_THINK = not args.reasoning

    lengths, depths, layers, conditions = stage_config(args.stage, args)
    docs = SYNTHETIC_DOCS
    if args.docs:
        want = set(args.docs.split(","))
        docs = [d for d in SYNTHETIC_DOCS if d["name"] in want]

    from config import StitcherConfig
    cfg = StitcherConfig()
    if args.gpus:
        devices = tuple(int(x) for x in args.gpus.split(","))
    else:
        import torch
        devices = tuple(range(torch.cuda.device_count()))   # use every visible GPU
        if not devices:
            raise RuntimeError("no CUDA devices visible — set CUDA_VISIBLE_DEVICES")
    print(f"  sharding DeepSeek-70B across {len(devices)} GPU(s): {devices}  "
          f"(device_map={args.device_map})")
    tok, model = load_deepseek(cfg, devices=devices, device_map=args.device_map,
                               max_memory_per_gpu=args.max_mem_per_gpu)
    _report_gpu_placement(model)

    print(f"\n########## PROOF 4 — length scaling (stage={args.stage}) ##########")
    print(f"  lengths={lengths}  depths={depths}  layers={layers}")
    print(f"  conditions={conditions}  keep_rate={args.keep_rate}")
    print(f"  docs={[d['name'] for d in docs]}  max_doc_tokens={args.max_doc_tokens}")

    records = evaluate(model, tok, docs, lengths, depths, layers, conditions,
                       args.keep_rate, args.max_doc_tokens, args.max_new_tokens)
    agg = aggregate(records, conditions)
    verdict = report(agg, conditions, args.stage)

    with open(out, "w") as f:
        json.dump({"stage": args.stage, "lengths": lengths, "depths": depths,
                   "layers": layers, "conditions": conditions,
                   "keep_rate": args.keep_rate, "verdict": verdict,
                   "aggregate": agg, "records": records}, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()

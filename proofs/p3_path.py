"""
Proof 3 — Path resolution: does the reading travel light, or must it travel whole?

Proofs 0–2 established that the large model recalls a document it was *handed* (true
layer-`target_layer` states, injected via the validated split-forward) rather than
one it read. Proof 3 asks how much of that handoff the recall actually needs:

  • Path 1 — the reading travels WHOLE: only the full sequence of layer-12 states
    lets the model answer. The handoff is an all-positions act.
  • Path 2 — the reading travels LIGHT: a few answer-bearing positions suffice. The
    SLM's job collapses to "find the needle and hand over just that."

Fixed from here forward: layer 12 (the 1.00-recall winner from the depth sweep).
Every condition injects at layer 12; the answer-bearing positions are defined
against that layer's captured states. Four conditions per gated question, all on the
same frozen 70B, scored behaviourally, with the same C-floor / A-ceiling gates that
made Proofs 1–2 mean something:

  all-N          — inject EVERY document position. The Path-1 reference (Proof 1's
                   known-good ≈1.0 at this layer). The ceiling for Proof 3.
  needles-only   — inject ONLY the answer-bearing positions (the needle clause's
                   tokens), at their ORIGINAL indices. The Path-2 test.
  random-subset  — inject the same COUNT as needles-only, drawn from elsewhere in
                   the document (never the needle). The control that makes a
                   needles-only success mean "the needle specifically", not "any k
                   positions scaffold a guess." Averaged over a few deterministic
                   draws.
  single-position— inject only the LAST token of the needle span. The sharp probe:
                   how far "travels light" goes — one bound position, or a span?

Verdict (rendered on gated items, where C=0 and A≈1.0 by construction):
  PATH_2     — needles-only succeeds AND random-subset fails. Reading is localizable;
               sparse handoff is real. The strong result.
  SUSPICIOUS — needles-only succeeds AND random-subset also succeeds. The needle
               isn't what's carrying the fact; investigate a leak / reconstruction.
  PATH_1     — needles-only fails while all-N succeeds. The fact needs the whole
               trace; the SLM must re-encode the full document.
  GRADED     — needles-only partially works. There is a minimum sufficient span;
               the recall-vs-#positions curve (run with --curve) is the finding and
               sets Proof 6's compression target.

Usage:
    python proofs/p3_path.py --layer 12 --out proofs/data/p3.json
    python proofs/p3_path.py --layer 12 --curve --out proofs/data/p3_curve.json
"""

import os
import sys
import json
import zlib
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.common import (
    load_deepseek, capture_document, no_context_answer, full_prefill_answer,
    inject_answer, inject_answer_subset, correct,
)
from proofs.needles import (
    span_token_positions, needle_positions, single_position,
    random_subset_positions,
)
from proofs.synthetic_docs import SYNTHETIC_DOCS
from proofs.synthetic_eval import STRONG_RATE, IGNORED_MARGIN, CAUSAL_MARGIN, MIN_GATED


def _seed(doc_name, question, trial):
    """A stable, process-independent seed for a random-subset draw, so the control
    is fixed before the model is consulted and reproducible across runs."""
    return zlib.crc32(f"{doc_name}|{question}|{trial}".encode()) & 0xFFFFFFFF


def evaluate_proof3(model, tokenizer, cfg, target_layer, docs=None, caches=None,
                    random_trials=3, keep_sink=True, max_new_tokens=512,
                    c_a_cache=None, curve=False):
    """Run all-N / needles-only / random-subset / single-position for every QA and
    return the record table."""
    docs = docs or SYNTHETIC_DOCS
    if caches is None:
        caches = {}
        for d in docs:
            print(f"  capturing {d['name']} at layer {target_layer} …")
            caches[d["name"]] = capture_document(model, tokenizer, d["text"], target_layer)

    records = []
    for doc in docs:
        d_cache, d_n = caches[doc["name"]]
        for qa in doc["qa"]:
            q, gold, span = qa["q"], qa["a"], qa["needle"]

            # Span → ORIGINAL token positions, blind to the model's answer.
            needle_idx = span_token_positions(tokenizer, doc["text"], span)
            need_pos = needle_positions(needle_idx, keep_sink=keep_sink)
            single_pos = single_position(needle_idx, keep_sink=keep_sink)
            k = len(needle_idx)   # content positions to match in the random control

            # ── gates (cacheable across layers) ───────────────────────────────
            cache_key = (doc["name"], q)
            if c_a_cache is not None and cache_key in c_a_cache:
                ans_c, ans_a = c_a_cache[cache_key]
            else:
                ans_c = no_context_answer(model, tokenizer, q, max_new_tokens)
                ans_a = full_prefill_answer(model, tokenizer, doc["text"], q, max_new_tokens)
                if c_a_cache is not None:
                    c_a_cache[cache_key] = (ans_c, ans_a)

            # ── conditions ────────────────────────────────────────────────────
            ans_all = inject_answer(model, tokenizer, d_cache, d_n, q,
                                    target_layer, max_new_tokens)
            ans_need = inject_answer_subset(model, tokenizer, d_cache, d_n, need_pos,
                                            q, target_layer, max_new_tokens)
            ans_single = inject_answer_subset(model, tokenizer, d_cache, d_n, single_pos,
                                              q, target_layer, max_new_tokens)

            rand_trials = []
            pool_size = None
            for t in range(random_trials):
                rpos, pool_size = random_subset_positions(
                    d_n, needle_idx, k, seed=_seed(doc["name"], q, t),
                    keep_sink=keep_sink)
                a_r = inject_answer_subset(model, tokenizer, d_cache, d_n, rpos, q,
                                           target_layer, max_new_tokens)
                rand_trials.append({"seed": _seed(doc["name"], q, t),
                                    "n_positions": len(rpos),
                                    "answer": a_r, "correct": correct(a_r, gold)})

            rec = {
                "doc": doc["name"], "question": q, "gold": gold,
                "needle": span, "needle_positions": needle_idx,
                "n_doc": d_n, "k_needles": k, "keep_sink": keep_sink,
                "random_pool_size": pool_size,
                "c": ans_c, "a": ans_a,
                "all_n": ans_all, "needles": ans_need, "single": ans_single,
                "random_trials": rand_trials,
                "c_correct": correct(ans_c, gold),
                "a_correct": correct(ans_a, gold),
                "all_n_correct": correct(ans_all, gold),
                "needles_correct": correct(ans_need, gold),
                "single_correct": correct(ans_single, gold),
                "random_any_correct": any(r["correct"] for r in rand_trials),
                "random_mean_correct": (
                    sum(r["correct"] for r in rand_trials) / len(rand_trials)
                    if rand_trials else 0.0),
            }

            # ── optional granularity curve: last-k tokens of the needle span ──
            if curve:
                ordered = sorted(needle_idx)
                curve_pts = []
                for kk in range(1, len(ordered) + 1):
                    sub = ordered[-kk:]                     # the kk tokens nearest the answer
                    if keep_sink:
                        sub = sorted(set(sub) | {0})
                    a_c = inject_answer_subset(model, tokenizer, d_cache, d_n, sub, q,
                                               target_layer, max_new_tokens)
                    curve_pts.append({"k": kk, "n_positions": len(sub),
                                      "answer": a_c, "correct": correct(a_c, gold)})
                rec["curve"] = curve_pts

            records.append(rec)
            gated = (not rec["c_correct"]) and rec["a_correct"]
            print(f"[{doc['name']}] {q} (layer {target_layer}, k={k}, gated={gated})")
            print(f"   gold={gold!r}  C={rec['c_correct']} A={rec['a_correct']}  "
                  f"all-N={rec['all_n_correct']} needles={rec['needles_correct']} "
                  f"single={rec['single_correct']} "
                  f"random={rec['random_mean_correct']:.2f}(any={rec['random_any_correct']})")
    return records


def _gated(records):
    return [r for r in records if (not r["c_correct"]) and r["a_correct"]]


def verdict_p3(records, verbose=True):
    gated = _gated(records)
    n = len(gated)
    disq_c = sum(r["c_correct"] for r in records)
    disq_a = sum(not r["a_correct"] for r in records)

    def rate(key):
        return (sum(r[key] for r in gated) / n) if n else 0.0

    all_n = rate("all_n_correct")
    needles = rate("needles_correct")
    single = rate("single_correct")
    random_mean = (sum(r["random_mean_correct"] for r in gated) / n) if n else 0.0
    random_any = rate("random_any_correct")

    # Decision logic. The reference (all-N) must itself be strong at this layer, or
    # nothing below is interpretable. needles-only "works" at STRONG_RATE; it is
    # "dead" (≈ the C floor of 0) within IGNORED_MARGIN; random must trail needles
    # by CAUSAL_MARGIN for the needle to be what's carrying the fact.
    separates = (needles - random_mean) >= CAUSAL_MARGIN
    if n < MIN_GATED:
        verdict = "FAIL"
    elif all_n < STRONG_RATE:
        verdict = "INVALID_REFERENCE"          # Path-1 ceiling broken; can't interpret
    elif needles >= STRONG_RATE and separates:
        verdict = "PATH_2"                     # localizable, sparse handoff is real
    elif needles >= STRONG_RATE and not separates:
        verdict = "SUSPICIOUS"                 # random also recovers → investigate
    elif needles < IGNORED_MARGIN:
        verdict = "PATH_1"                     # needle fails entirely; whole trace needed
    else:
        verdict = "GRADED"                     # minimum sufficient span between the two

    summary = {
        "total_items": len(records),
        "gated_items": n,
        "disqualified_c_guessable": int(disq_c),
        "disqualified_a_unanswerable": int(disq_a),
        "all_n_on_gated": round(all_n, 3),
        "needles_only_on_gated": round(needles, 3),
        "single_position_on_gated": round(single, 3),
        "random_subset_mean_on_gated": round(random_mean, 3),
        "random_subset_any_on_gated": round(random_any, 3),
        "needles_minus_random": round(needles - random_mean, 3),
        "strong_threshold": STRONG_RATE,
        "causal_margin": CAUSAL_MARGIN,
        "verdict": verdict,
    }
    if verbose:
        print("\n" + "=" * 60)
        print("PROOF 3 — path resolution (travels light vs. travels whole)")
        print(f"  gated items (C fails & A succeeds): {n}/{len(records)}")
        print(f"  disqualified — C guessable: {disq_c}   A unanswerable: {disq_a}")
        print(f"  all-N (Path-1 reference) : {summary['all_n_on_gated']}  (want ≈1.0)")
        print(f"  needles-only             : {summary['needles_only_on_gated']}")
        print(f"  random-subset (mean)     : {summary['random_subset_mean_on_gated']}"
              f"   (any-of-trials {summary['random_subset_any_on_gated']})")
        print(f"  single-position          : {summary['single_position_on_gated']}")
        print(f"  needles − random         : {summary['needles_minus_random']}"
              f"   (≥ {CAUSAL_MARGIN} ⇒ needle-specific)")
        print(f"  VERDICT: {verdict}")
        if verdict == "PATH_2":
            print("  → the reading travels LIGHT: the bound fact alone, at its own")
            print("    positions, recalls — and random positions do not. Sparse handoff.")
        elif verdict == "SUSPICIOUS":
            print("  → needles AND random both recover; the needle isn't what carries")
            print("    the fact. Find the leak / reconstruction before believing Path 2.")
        elif verdict == "PATH_1":
            print("  → the reading travels WHOLE: the local span is not enough; the")
            print("    model needs the full trace. SLM must re-encode the whole document.")
        elif verdict == "GRADED":
            print("  → a minimum sufficient span exists (more than one token, less than")
            print("    the whole). Run --curve to map recall vs #positions (Proof 6 target).")
        elif verdict == "INVALID_REFERENCE":
            print(f"  → all-N is only {all_n} here — the Path-1 reference is broken at this")
            print("    layer, so needles-only cannot be interpreted. Re-check the layer.")
        else:
            print("  → too few gated items to decide. Need ≥ "
                  f"{MIN_GATED}.")

        # Granularity curve, if present: averaged recall at each suffix length.
        if gated and "curve" in gated[0]:
            maxk = max(len(r["curve"]) for r in gated)
            print("\n  recall vs #needle positions (suffix of span, averaged over gated):")
            for kk in range(1, maxk + 1):
                pts = [r["curve"][kk - 1]["correct"] for r in gated if len(r["curve"]) >= kk]
                if pts:
                    print(f"    k={kk:>2}: {sum(pts)/len(pts):.2f}  (n={len(pts)})")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="proofs/data/p3.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--layer", type=int, default=12,
                        help="injection layer (Proof 3 is fixed at the 1.00-recall winner, 12)")
    parser.add_argument("--random-trials", type=int, default=3,
                        help="random-subset draws per question (the control, averaged)")
    parser.add_argument("--no-sink", action="store_true",
                        help="do NOT retain the attention-sink position 0 in sparse conditions")
    parser.add_argument("--curve", action="store_true",
                        help="also map recall vs #needle positions (the granularity curve)")
    parser.add_argument("--reasoning", action="store_true",
                        help="let R1 emit <think> traces instead of suppressing them")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import proofs.common as _common
    _common.SUPPRESS_THINK = not args.reasoning

    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))
    tok, model = load_deepseek(cfg, devices=devices)

    print(f"\n########## PROOF 3 — path resolution at layer {args.layer} ##########")
    print(f"  keep_sink={not args.no_sink}  random_trials={args.random_trials}  "
          f"curve={args.curve}")

    records = evaluate_proof3(
        model, tok, cfg, target_layer=args.layer,
        random_trials=args.random_trials, keep_sink=not args.no_sink,
        max_new_tokens=args.max_new_tokens, c_a_cache={}, curve=args.curve)
    summary = verdict_p3(records)

    with open(args.out, "w") as f:
        json.dump({"layer": args.layer, "keep_sink": not args.no_sink,
                   "random_trials": args.random_trials,
                   "summary": summary, "records": records}, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

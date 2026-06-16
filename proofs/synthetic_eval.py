"""
proofs/synthetic_eval.py — the shared engine for Proofs 1 and 2.

Capturing each synthetic document's true cache and running the conditions is the
expensive part, and Proofs 1 and 2 read the *same* table from opposite angles:

  Proof 1 (premise)   : on items gated by "C fails AND A succeeds", does
                        inject-matched succeed? (the receiver reads)
  Proof 2 (falsifier) : on those same items, does inject-WRONG-document fail,
                        while inject-matched still succeeds? (it's the injection,
                        not memory or a leak)

So we evaluate once into one record table and render two verdicts from it. Each
record holds four conditions for one (document, question):

  c      — Condition C, no context           (gate: must FAIL)
  a      — Condition A, full prefill          (gate: must SUCCEED)
  inject — split-forward, MATCHED document     (Proof 1 test / Proof 2 dual)
  wrong  — split-forward, a DIFFERENT document (Proof 2 falsifier; want ~0)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.common import (
    capture_document, no_context_answer, full_prefill_answer, inject_answer, correct,
)
from proofs.synthetic_docs import SYNTHETIC_DOCS

PASS_RATE = 0.8      # inject success rate on gated items to call a rung green
MIN_GATED = 5        # minimum gated items for the result to mean anything


def capture_all(model, tokenizer, cfg, docs=None):
    """Capture each document's true split-forward cache once. Returns
    name -> (doc_cache, n_doc)."""
    docs = docs or SYNTHETIC_DOCS
    caches = {}
    for d in docs:
        print(f"  capturing {d['name']} …")
        caches[d["name"]] = capture_document(model, tokenizer, d["text"], cfg.target_layer)
    return caches


def evaluate_synthetic(model, tokenizer, cfg, docs=None, caches=None,
                       max_new_tokens=256, want_wrong=True):
    """Run C / A / inject-matched / (inject-wrong) for every QA and return the
    record table."""
    docs = docs or SYNTHETIC_DOCS
    if caches is None:
        caches = capture_all(model, tokenizer, cfg, docs)
    names = [d["name"] for d in docs]

    records = []
    for di, doc in enumerate(docs):
        wrong_name = names[(di + 1) % len(names)]      # a different document
        d_cache, d_n = caches[doc["name"]]
        w_cache, w_n = caches[wrong_name]
        for qa in doc["qa"]:
            q, gold = qa["q"], qa["a"]
            ans_c = no_context_answer(model, tokenizer, q, max_new_tokens)
            ans_a = full_prefill_answer(model, tokenizer, doc["text"], q, max_new_tokens)
            ans_inj = inject_answer(model, tokenizer, d_cache, d_n, q,
                                    cfg.target_layer, max_new_tokens)
            rec = {
                "doc": doc["name"], "wrong_doc": wrong_name,
                "question": q, "gold": gold,
                "c": ans_c, "a": ans_a, "inject": ans_inj,
                "c_correct": correct(ans_c, gold),
                "a_correct": correct(ans_a, gold),
                "inject_correct": correct(ans_inj, gold),
            }
            if want_wrong:
                # inject the WRONG document's states, ask THIS document's question.
                ans_w = inject_answer(model, tokenizer, w_cache, w_n, q,
                                      cfg.target_layer, max_new_tokens)
                rec["wrong"] = ans_w
                rec["wrong_correct"] = correct(ans_w, gold)
            records.append(rec)
            tail = f"  WRONG={rec.get('wrong_correct')}" if want_wrong else ""
            print(f"[{doc['name']}] {q}\n   gold={gold!r}  C={rec['c_correct']}  "
                  f"A={rec['a_correct']}  INJECT={rec['inject_correct']}{tail}")
    return records


def _gated(records):
    """Items where the fact is genuinely unguessable (C fails) AND recoverable
    from the text (A succeeds). Only these can prove anything."""
    return [r for r in records if (not r["c_correct"]) and r["a_correct"]]


def verdict_p1(records, verbose=True):
    gated = _gated(records)
    n = len(gated)
    disq_c = sum(r["c_correct"] for r in records)          # guessable → thrown out
    disq_a = sum(not r["a_correct"] for r in records)      # not in text → thrown out
    inj = (sum(r["inject_correct"] for r in gated) / n) if n else 0.0
    ok = n >= MIN_GATED and inj >= PASS_RATE
    summary = {
        "total_items": len(records),
        "gated_items": n,
        "disqualified_c_guessable": int(disq_c),
        "disqualified_a_unanswerable": int(disq_a),
        "inject_correct_on_gated": round(inj, 3),
        "pass_rate_threshold": PASS_RATE,
        "verdict": "PASS" if ok else "FAIL",
    }
    if verbose:
        print("\n" + "=" * 60)
        print("PROOF 1 — the injection premise")
        print(f"  gated items (C fails & A succeeds): {n}/{len(records)}")
        print(f"  disqualified — C guessable: {disq_c}   A unanswerable: {disq_a}")
        print(f"  inject-all-N correct on gated:      {summary['inject_correct_on_gated']}")
        print(f"  VERDICT: {summary['verdict']}"
              + ("" if ok else "  → premise unproven; do not build the sender"))
        if ok:
            print("  → injected true states are read & reasoned over on unguessable facts.")
    return summary


def verdict_p2(records, verbose=True):
    gated = _gated(records)
    n = len(gated)
    if any("wrong_correct" not in r for r in gated):
        raise ValueError("Proof 2 needs wrong-document answers; run with want_wrong=True")
    matched = (sum(r["inject_correct"] for r in gated) / n) if n else 0.0
    wrong = (sum(r["wrong_correct"] for r in gated) / n) if n else 0.0
    # Falsifier passes when the wrong document does NOT answer (low) while the
    # matched document does (high).
    ok = n >= MIN_GATED and matched >= PASS_RATE and wrong <= (1.0 - PASS_RATE)
    summary = {
        "gated_items": n,
        "inject_matched_correct_on_gated": round(matched, 3),
        "inject_wrong_correct_on_gated": round(wrong, 3),
        "verdict": "PASS" if ok else "FAIL",
    }
    if verbose:
        print("\n" + "=" * 60)
        print("PROOF 2 — the wrong-document falsifier")
        print(f"  matched-document inject correct: {summary['inject_matched_correct_on_gated']}  (want high)")
        print(f"  wrong-document   inject correct: {summary['inject_wrong_correct_on_gated']}  (want ~0)")
        print(f"  VERDICT: {summary['verdict']}")
        if ok:
            print("  → the injected document causally controls the answer (not memory/leak).")
        elif wrong > (1.0 - PASS_RATE):
            print("  → wrong doc still answers: injection inert; Proof 1 FALSIFIED. Find the leak.")
        else:
            print("  → matched inject too weak to claim causation.")
    return summary

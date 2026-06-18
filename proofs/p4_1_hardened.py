"""
Proof 4.1 — Hardened single-point confirmation (32k, layer 12, depth 0.5).

Proof 4 showed inject_all_N = 1.00 from 500 to 32k tokens. That number is almost too
clean, so before shipping to Proof 5 we re-test the SINGLE most stressful point with
the three sources of evaluation slack removed. This is a confirmation, not a sweep:
one length, one layer, one depth, many hardenings.

  Hardening 1 — strict scoring. The chain's `correct()` is containment ("gold appears
      anywhere in a 256-token answer"). We keep it for continuity but report two
      stricter scorers on the SAME outputs:
        lenient   — gold ∈ normalized answer (the old scorer).
        firstline — gold ∈ the first answer clause (not buried in a restatement).
        strict    — the answer clause IS the gold (modulo a tiny answer-carrier like
                    "the answer is …"). Restating the question's sentence fails.
      The lenient−strict delta is the inflation, measured honestly.

  Hardening 2 — capture/A symmetry. Capture is document-only (question-naive) while A
      is document+question. We run inject two ways:
        inject_docnaive — the clean digest (current behaviour).
        inject_qfair    — capture the document states inside the SAME instruction
                          framing A sees (the prefill prompt's instruction + document),
                          so the injected representation is "diluted" the way A's is.
      Honest ceiling comparison is inject_qfair vs A, not inject_docnaive vs A.

  Hardening 3 — distractor filler. The C_filler gate proves the filler does not ANSWER
      the question; it does not prove it COMPETES. We plant near-miss decoys (same
      surface form, wrong values) at other depths, then re-gate (C and C_filler must
      still fail, A must still succeed). A correct answer now requires discriminating
      the true needle from look-alikes — the realistic task. This is the single most
      important hardening: a 1.00 that survives distractors is real.

  Hardening 4 — reasoning on. Proof 4 suppressed <think>, measuring extraction not
      reasoning. We add a think-ON arm (parse post-</think>, count an unclosed think as
      no-answer) for A / inject_docnaive / inject_qfair — the path Proof 5 will use.

To have the power to resolve these effects (the first run had only n=3 gated, which
can't separate a 33% difference), the cell is POOLED over all distractor-banked docs ×
several needle depths — Proof 4 showed depth is inert, so these are independent
retrieval instances, not a confound. Two arms, reported separately:
  • discrimination arm (A / inject_docnaive / inject_qfair / needles_only) on the
    answer-in-needle synthetic docs WITH distractors — the honest ceiling comparison.
  • latent-vs-text arm on the Exp-3.1 COREFERENCE docs (answer in the decimatable
    surroundings) thinned to the keep-rate where dec_text COLLAPSES — the only setting
    where latent>text is a meaningful claim. The verdict refuses to read the gap unless
    text actually collapsed, and refuses to claim parity below n ≥ 30.

The number to look at first is dec_latent − dec_text under STRICT scoring on the
collapse arm: if the Exp-3.1 mechanism (latent carries what text loses) survives the
hardest, fairest test, nothing else in the table can sink the project.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python proofs/p4_1_hardened.py \
        --length 32000 --layer 12 --depths 0.1,0.5,0.9 --out proofs/data/p4_1.json
    # faster first look: one depth, no think-on, skip the dec arm
    ... python proofs/p4_1_hardened.py --depths 0.5 --no-think-on --no-dec
    # if dec_text doesn't collapse, thin harder:
    ... python proofs/p4_1_hardened.py --dec-keep-rate 0.25
    # re-score a saved run with the current scorers (no GPU):
    ... python proofs/p4_1_hardened.py --rescore proofs/data/p4_1.json
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import proofs.common as _common
from core.split_forward import capture_doc_cache, split_forward_generate
from proofs.common import (
    load_deepseek, normalize, final_answer, no_context_answer, full_prefill_answer,
    inject_answer, inject_answer_subset, _with_think_control,
)
from proofs.needles import span_token_positions, needle_positions
from proofs.decimate import kept_indices, decimated_text
from proofs.long_context_docs import build_distractor_doc, selftest_filler
from proofs.fact_bank import (
    span_docs, coref_docs, doc_by_name, distractors_map, decoy_values_map,
    selftest_bank, n_facts,
)

# Proof 4.1 power / mechanism thresholds. P41_MIN_N is the gated-item floor below which
# a "parity" claim is over-reading noise (the n=3 trap). COLLAPSE/GAP mirror Exp 3.1:
# the dec arm only INFORMS the latent>text claim if text actually collapsed (dec_text ≤
# COLLAPSE); a real advantage means dec_latent − dec_text ≥ GAP.
P41_MIN_N = 30
DEC_MIN_N = 8
COLLAPSE = 0.3
GAP = 0.2


# ── the A-prompt split, for the capture/A-symmetry (Hardening 2) ───────────────
# inject_qfair captures the document INSIDE the same framing A prefills, so the only
# remaining asymmetry is the question itself (which A's document also never attends to,
# since the question is causally after it). Derive the split from common.PREFILL_PROMPT
# so the two stay in sync — prefix is "<instruction>\n\nDocument:\n{document}", the
# suffix is "\n\nQuestion: {question}\nAnswer:".
_PFX, _QPART = _common.PREFILL_PROMPT.split("\n\nQuestion: ")
PREFILL_PREFIX = _PFX
PREFILL_QSUFFIX = "\n\nQuestion: " + _QPART
assert PREFILL_PREFIX + PREFILL_QSUFFIX == _common.PREFILL_PROMPT


# ── distractor banks + decoy values come from the expanded fact_bank ──────────
# Each doc carries native near-miss decoys (same surface form, WRONG value) and the
# wrong-value tokens the strict scorer must rule out. Sourced from proofs/fact_bank.py
# (50 facts across 10 docs, authored adversarially) so Proof 4.1 — and Proof 5 after
# it — generalize across many independent facts, not five hand-picked ones.
DISTRACTORS = distractors_map()
DECOY_VALUES = decoy_values_map()


# Cues that the clause negates / hedges / declines the answer — disqualify strict even
# if the gold string is present.
_NEGATIONS = ["not ", "n't", "rather than", "instead of", "do not know",
              "don t know", "unknown", "unclear", "unsure", "cannot", "no information"]


def first_clause(answer: str) -> str:
    """The answer clause: the first non-empty line, then its first sentence. This is
    where a direct answer lives, as opposed to a later restatement of the question."""
    line = next((l for l in answer.splitlines() if l.strip()), "")
    parts = re.split(r"(?<=[.!?])\s", line.strip())
    return parts[0] if parts else line.strip()


# A fact may have several acceptable surface forms (`alt` in fact_bank) — e.g. a
# spelled-out number gold "ninety-eight feet" and the digit form "98 feet" the model is
# just as likely to produce. Scoring against ANY accepted form removes a pure
# word-vs-digit format mismatch from being read as a wrong answer; without it a correct
# digit reply fails the gate and the fact is silently dropped (the A-unanswerable
# disqualifications seen at n=28).
def _golds(gold, alts):
    return [gold] + list(alts or [])


def score_lenient(answer: str, gold: str, alts=()) -> bool:
    a = normalize(answer)
    return any(normalize(g) in a for g in _golds(gold, alts))


def score_firstline(answer: str, gold: str, alts=()) -> bool:
    clause = normalize(first_clause(answer))
    return any(normalize(g) in clause for g in _golds(gold, alts))


def score_strict(answer: str, gold: str, decoys=(), alts=()) -> bool:
    """An accepted form in the answer clause, no decoy value in it, no negation/hedge."""
    clause = normalize(first_clause(answer))
    if not any(normalize(g) in clause for g in _golds(gold, alts)):
        return False
    if any(normalize(d) in clause for d in decoys):
        return False
    if any(neg in clause for neg in _NEGATIONS):
        return False
    return True


SCORERS = ["lenient", "firstline", "strict"]


def score_all(answer: str, gold: str, decoys=(), alts=()) -> dict:
    return {"lenient": score_lenient(answer, gold, alts),
            "firstline": score_firstline(answer, gold, alts),
            "strict": score_strict(answer, gold, decoys, alts)}


# ── think-mode control ─────────────────────────────────────────────────────────
def _set_think(mode):
    """mode 'off' suppresses <think> (extraction); 'on' lets R1 reason."""
    _common.SUPPRESS_THINK = (mode == "off")


# ── condition runners (each returns the raw final answer string) ───────────────
def ans_A(model, tok, doc_text, q, max_new_tokens, max_doc_tokens):
    return full_prefill_answer(model, tok, doc_text, q, max_new_tokens,
                               max_length=max_doc_tokens)


def ans_docnaive(model, tok, cache, n_doc, q, layer, max_new_tokens):
    return inject_answer(model, tok, cache, n_doc, q, layer, max_new_tokens)


def ans_qfair(model, tok, qcache, n_pre, q, layer, max_new_tokens):
    query = _with_think_control(PREFILL_QSUFFIX.format(question=q))
    txt = split_forward_generate(model, tok, qcache, n_pre, query_text=query,
                                 target_layer=layer, max_new_tokens=max_new_tokens)
    return final_answer(txt)


# ── per-cell helpers ───────────────────────────────────────────────────────────
def _free_cuda():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


def _gates_for_cell(model, tok, doc, filler, base, mnt, max_doc_tokens, c_cache):
    """Per-question, layer-independent gates for one built cell (with distractors when
    the doc carries them). Returns one item per QA with its needle positions and the
    three gate booleans (lenient). C (no-context) is cached across cells by question."""
    items = []
    for qa in base["qa"]:
        q, gold, alts = qa["q"], qa["a"], qa.get("alt", [])
        if q in c_cache:
            c = c_cache[q]
        else:
            c = no_context_answer(model, tok, q, mnt)
            c_cache[q] = c
        a = ans_A(model, tok, doc["text"], q, mnt, max_doc_tokens)
        cf = ans_A(model, tok, filler["text"], q, mnt, max_doc_tokens)
        c_ok = score_lenient(c, gold, alts)
        cf_ok = score_lenient(cf, gold, alts)
        a_ok = score_lenient(a, gold, alts)
        idx = span_token_positions(tok, doc["text"], qa["needle"], max_doc_tokens)
        items.append({"q": q, "gold": gold, "alts": alts, "needle_idx": idx,
                      "c_ok": c_ok, "cf_ok": cf_ok, "a_ok": a_ok,
                      "a_ans": a[:90],   # kept for diagnosing A-unanswerable disqualifications
                      "gated": (not c_ok) and (not cf_ok) and a_ok})
    return items


# ── main evaluation: pool the cell over docs × depths, plus the dec arm ─────────
def run(model, tok, args):
    layer, mnt, tmnt = args.layer, args.max_new_tokens, args.think_max_new_tokens
    modes = ["off"] if args.no_think_on else ["off", "on"]
    depths = [float(x) for x in args.depths.split(",")]

    main_names = ([n.strip() for n in args.docs.split(",")] if args.docs
                  else [d["name"] for d in span_docs()])
    for n in main_names:
        if n not in DISTRACTORS:
            raise SystemExit(f"no distractor bank for {n!r} — add one to DISTRACTORS.")
        # no distractor may contain a true gold (else it stops being a near-MISS)
        base = doc_by_name(n)
        for d in DISTRACTORS[n]:
            for qa in base["qa"]:
                assert normalize(qa["a"]) not in normalize(d), \
                    f"distractor leaks gold {qa['a']!r}: {d!r}"

    c_cache, records, gate_rows, eyeball = {}, [], [], []
    canary_mismatch, canary_checked = 0, 0

    # ───────────── discrimination arm: A / inject_docnaive / inject_qfair ─────────
    print("\n" + "=" * 64)
    print(f"DISCRIMINATION ARM — docs={main_names} depths={depths}")
    for name in main_names:
        base = doc_by_name(name)
        distractors, decoys = DISTRACTORS[name], DECOY_VALUES.get(name, [])
        for depth in depths:
            doc = build_distractor_doc(tok, base, args.length, depth, distractors,
                                       max_doc_tokens=args.max_doc_tokens)
            filler = build_distractor_doc(tok, base, args.length, depth, distractors,
                                          max_doc_tokens=args.max_doc_tokens, drop_fact=True)
            ids = tok(doc["text"], return_tensors="pt", truncation=True,
                      max_length=args.max_doc_tokens).input_ids
            print(f"\n[main] {name} depth={depth}: {doc['n_tokens']} tok, "
                  f"fact@{doc['depth_actual']}, {doc['n_distractors']} distractors")

            items = _gates_for_cell(model, tok, doc, filler, base, mnt,
                                    args.max_doc_tokens, c_cache)
            for it in items:
                gate_rows.append({"arm": "main", "doc": name, "depth": depth,
                                  "question": it["q"], "gold": it["gold"],
                                  "c_ok": it["c_ok"], "cf_ok": it["cf_ok"],
                                  "a_ok": it["a_ok"], "a_ans": it["a_ans"],
                                  "gated": it["gated"]})
            gated = [it for it in items if it["gated"]]
            print(f"   gated {len(gated)}/{len(items)} "
                  f"(disq C={sum(it['c_ok'] for it in items)} "
                  f"C_filler={sum(it['cf_ok'] for it in items)} "
                  f"A_fail={sum(not it['a_ok'] for it in items)})")
            if not gated:
                continue

            cache, _Y, n_doc = capture_doc_cache(model, ids, layer); del _Y
            pre_ids = tok(PREFILL_PREFIX.format(document=doc["text"]),
                          return_tensors="pt", truncation=True,
                          max_length=args.max_doc_tokens).input_ids
            qcache, _Yq, n_pre = capture_doc_cache(model, pre_ids, layer); del _Yq

            # one-time bookkeeping canary: inject-all-positions == full inject
            if canary_checked == 0:
                it = gated[0]
                kept_all = kept_indices(n_doc, it["needle_idx"], 1.0, "strided",
                                        "needle_decimated", seed=0, keep_sink=True)
                _set_think("off")
                a_sub = inject_answer_subset(model, tok, cache, n_doc, kept_all,
                                             it["q"], layer, mnt)
                a_full = inject_answer(model, tok, cache, n_doc, it["q"], layer, mnt)
                canary_mismatch += int(a_sub.strip() != a_full.strip())
                canary_checked = 1
                print(f"   canary subset-to-all no-op: "
                      f"{'OK' if a_sub.strip() == a_full.strip() else 'MISMATCH'}")

            for it in gated:
                q, gold, idx, alts = it["q"], it["gold"], it["needle_idx"], it["alts"]
                rec = {"arm": "main", "doc": name, "depth": depth, "question": q,
                       "gold": gold, "alts": alts, "k_needle": len(idx),
                       "answers": {}, "scores": {}}
                for mode in modes:
                    _set_think(mode)
                    m = mnt if mode == "off" else tmnt
                    outs = {
                        "A": ans_A(model, tok, doc["text"], q, m, args.max_doc_tokens),
                        "inject_docnaive": ans_docnaive(model, tok, cache, n_doc, q, layer, m),
                        "inject_qfair": ans_qfair(model, tok, qcache, n_pre, q, layer, m),
                    }
                    if mode == "off":
                        outs["needles_only"] = inject_answer_subset(
                            model, tok, cache, n_doc, needle_positions(idx, keep_sink=True),
                            q, layer, m)
                    for cond, ans in outs.items():
                        rec["answers"][f"{cond}@{mode}"] = ans
                        rec["scores"][f"{cond}@{mode}"] = score_all(ans, gold, decoys, alts)
                records.append(rec)
                if len(eyeball) < 5:
                    eyeball.append({"doc": name, "q": q, "gold": gold,
                                    "answer": rec["answers"]["inject_docnaive@off"]})
            del cache, qcache
            _free_cuda()

    # ───────────── latent-vs-text arm: coreference docs thinned to collapse ───────
    # Run on the Exp-3.1 coreference docs (answer in the DECIMATABLE surroundings,
    # needle refers to it by anaphora) at the keep-rate where TEXT collapses. On the
    # answer-in-needle discrimination docs, needle_protected keeps the answer and text
    # never fails, so the gap there is meaningless — this is the only arm that can show
    # latent > text honestly. The coreference docs carry native distractors too, so this
    # arm runs UNDER distractors (strict scoring rules out the decoy values).
    dec_records = []
    if not args.no_dec:
        dec_names = ([n.strip() for n in args.dec_docs.split(",")] if args.dec_docs
                     else [d["name"] for d in coref_docs()])
        dec_by_name = {d["name"]: d for d in coref_docs()}
        print("\n" + "=" * 64)
        print(f"LATENT-vs-TEXT ARM — coreference docs={dec_names} "
              f"keep_rate={args.dec_keep_rate} (strided, needle_protected)")
        for name in dec_names:
            base = dec_by_name[name]
            distractors, decoys = DISTRACTORS.get(name, []), DECOY_VALUES.get(name, [])
            for depth in depths:
                doc = build_distractor_doc(tok, base, args.length, depth, distractors,
                                           max_doc_tokens=args.max_doc_tokens)
                filler = build_distractor_doc(tok, base, args.length, depth, distractors,
                                              max_doc_tokens=args.max_doc_tokens,
                                              drop_fact=True)
                ids = tok(doc["text"], return_tensors="pt", truncation=True,
                          max_length=args.max_doc_tokens).input_ids
                print(f"\n[dec ] {name} depth={depth}: {doc['n_tokens']} tok")
                items = _gates_for_cell(model, tok, doc, filler, base, mnt,
                                        args.max_doc_tokens, c_cache)
                for it in items:
                    gate_rows.append({"arm": "dec", "doc": name, "depth": depth,
                                      "question": it["q"], "gold": it["gold"],
                                      "c_ok": it["c_ok"], "cf_ok": it["cf_ok"],
                                      "a_ok": it["a_ok"], "a_ans": it["a_ans"],
                                      "gated": it["gated"]})
                gated = [it for it in items if it["gated"]]
                print(f"   gated {len(gated)}/{len(items)}")
                if not gated:
                    continue
                cache, _Y, n_doc = capture_doc_cache(model, ids, layer); del _Y
                _set_think("off")
                for it in gated:
                    q, gold, idx, alts = it["q"], it["gold"], it["needle_idx"], it["alts"]
                    kept = kept_indices(n_doc, idx, args.dec_keep_rate, "strided",
                                        "needle_protected", seed=0, keep_sink=True)
                    a_txt = ans_A(model, tok, decimated_text(tok, ids, kept), q, mnt,
                                  args.max_doc_tokens)
                    a_lat = inject_answer_subset(model, tok, cache, n_doc, kept, q, layer, mnt)
                    rec = {"arm": "dec", "doc": name, "depth": depth, "question": q,
                           "gold": gold, "alts": alts, "kept_count": len(kept),
                           "keep_rate": args.dec_keep_rate,
                           "answers": {"dec_text@off": a_txt, "dec_latent@off": a_lat},
                           "scores": {"dec_text@off": score_all(a_txt, gold, decoys, alts),
                                      "dec_latent@off": score_all(a_lat, gold, decoys, alts)}}
                    dec_records.append(rec)
                    print(f"     [{name}|d{depth}] {q[:42]!r}  "
                          f"text={rec['scores']['dec_text@off']['strict']} "
                          f"lat={rec['scores']['dec_latent@off']['strict']}")
                del cache
                _free_cuda()

    return {
        "length": args.length, "layer": layer, "depths": depths,
        "main_docs": main_names, "modes": modes,
        "dec_keep_rate": args.dec_keep_rate, "no_dec": args.no_dec,
        "sanity_canary_mismatch": canary_mismatch,
        "gate_rows": gate_rows, "eyeball": eyeball,
        "records": records + dec_records,
    }


# ── aggregation + report ────────────────────────────────────────────────────────
CONDS_MAIN = ["A", "inject_docnaive", "inject_qfair", "needles_only"]
CONDS_DEC = ["dec_text", "dec_latent"]


def _diff(a, b):
    return None if a is None or b is None else round(a - b, 3)


def aggregate(result):
    recs = result["records"]
    modes = result["modes"]
    main = [r for r in recs if r.get("arm") == "main"]
    dec = [r for r in recs if r.get("arm") == "dec"]

    def rate(cond, mode, scorer, pool):
        key = f"{cond}@{mode}"
        vals = [r["scores"][key][scorer] for r in pool if key in r.get("scores", {})]
        return round(sum(vals) / len(vals), 3) if vals else None

    table = {}
    for cond in ["A", "inject_docnaive", "inject_qfair"]:
        table[cond] = {m: {s: rate(cond, m, s, main) for s in SCORERS} for m in modes}
    table["needles_only"] = {"off": {s: rate("needles_only", "off", s, main) for s in SCORERS}}
    for cond in CONDS_DEC:
        table[cond] = {"off": {s: rate(cond, "off", s, dec) for s in SCORERS}}

    a_s = table["A"]["off"]["strict"]
    qf_s = table["inject_qfair"]["off"]["strict"]
    dn_l = table["inject_docnaive"]["off"]["lenient"]
    dn_s = table["inject_docnaive"]["off"]["strict"]
    dl = table["dec_latent"]["off"]["strict"]
    dt = table["dec_text"]["off"]["strict"]
    headline = {
        "qfair_strict_vs_A_strict": _diff(qf_s, a_s),
        "total_slack_lenient_minus_strict_docnaive": _diff(dn_l, dn_s),
        "dec_latent_minus_dec_text_strict": _diff(dl, dt),
    }
    return {
        "n_main": len(main), "n_dec": len(dec),
        "distinct_main": len({(r["doc"], r["question"]) for r in main}),
        "distinct_dec": len({(r["doc"], r["question"]) for r in dec}),
        "table": table, "headline": headline,
    }


def _gate_diag(result):
    """Disqualification breakdown per arm: WHY items fell out of gating."""
    rows = result.get("gate_rows", [])
    out = {}
    for arm in ("main", "dec"):
        a = [r for r in rows if r["arm"] == arm]
        out[arm] = {
            "items": len(a), "gated": sum(r["gated"] for r in a),
            "disq_c_guessable": sum(r["c_ok"] for r in a),
            "disq_filler_leak": sum(r["cf_ok"] for r in a),
            "disq_a_unanswerable": sum(not r["a_ok"] for r in a),
        }
    return out


def _verdicts(agg):
    """Two honest sub-verdicts — discrimination (A vs inject under strict) and mechanism
    (latent vs text on the collapse arm) — plus a combined ship/hold. Neither is allowed
    to claim a result the sample can't support (the n=3 trap), and the mechanism verdict
    refuses to read a gap when text never collapsed."""
    t = agg["table"]
    a_s, qf_s = t["A"]["off"]["strict"], t["inject_qfair"]["off"]["strict"]
    qf_l = t["inject_qfair"]["off"]["lenient"]
    dl, dt = t["dec_latent"]["off"]["strict"], t["dec_text"]["off"]["strict"]

    # discrimination
    if agg["n_main"] < P41_MIN_N:
        disc = "UNDERPOWERED"
    elif qf_l is not None and qf_l <= 0.3:
        disc = "COLLAPSED_UNDER_DISTRACTORS"
    elif a_s is not None and qf_s is not None and qf_s >= a_s - 0.05:
        disc = "PARITY_WITH_A"
    elif qf_l is not None and qf_l >= 0.8 and a_s is not None and qf_s is not None:
        disc = "RECOVERS_NOT_PARITY"
    else:
        disc = "MIXED"

    # mechanism
    if agg["n_dec"] < DEC_MIN_N or dt is None or dl is None:
        mech = "UNDERPOWERED_OR_MISSING"
    elif dt > COLLAPSE:
        mech = "TEXT_DID_NOT_COLLAPSE"          # uninformative — thin harder
    elif dl - dt >= GAP:
        mech = "LATENT_BEATS_TEXT"              # the result that ships
    elif dl <= COLLAPSE:
        mech = "LATENT_ALSO_COLLAPSED"          # the threatening outcome
    else:
        mech = "WEAK_SEPARATION"

    ship = (disc in ("PARITY_WITH_A", "RECOVERS_NOT_PARITY")
            and mech == "LATENT_BEATS_TEXT")
    return {"discrimination": disc, "mechanism": mech,
            "recommend": "SHIP_TO_PROOF_5" if ship else "HOLD"}


def report(result, agg):
    modes = result["modes"]
    diag = _gate_diag(result)
    print("\n" + "=" * 80)
    print(f"PROOF 4.1 — hardened confirmation ({result['length']} tok, L{result['layer']}, "
          f"depths={result['depths']})")
    print(f"  discrimination arm: n={agg['n_main']} gated items from "
          f"{agg['distinct_main']} distinct facts (target n ≥ {P41_MIN_N})")
    print(f"  latent-vs-text arm: n={agg['n_dec']} from {agg['distinct_dec']} facts  "
          f"(keep_rate {result.get('dec_keep_rate')})")
    print(f"  canary mismatches = {result['sanity_canary_mismatch']} (must be 0)")
    for arm in ("main", "dec"):
        d = diag[arm]
        print(f"  [{arm}] disqualified — C guessable {d['disq_c_guessable']}, "
              f"filler-leak {d['disq_filler_leak']}, A-unanswerable {d['disq_a_unanswerable']} "
              f"(of {d['items']})")

    # Show WHY any A-unanswerable items fell out — gold vs what A actually said. This is
    # how you tell a real retrieval failure from a scoring-format mismatch (e.g. the
    # model answering a number in digits when the gold is spelled out).
    bad = [r for r in result.get("gate_rows", []) if not r["a_ok"]]
    if bad:
        print("  A-unanswerable detail (gold → A said):")
        for r in bad[:12]:
            print(f"    [{r['doc']}|d{r['depth']}] gold={r.get('gold')!r} → "
                  f"{(r.get('a_ans') or '')[:70]!r}")

    cols = [(s, m) for m in modes for s in SCORERS]
    head = "  " + f"{'condition':<18}" + "".join(f"{s[:4]+'/'+m:>12}" for s, m in cols)
    print("\n" + head)
    print("  " + "-" * (len(head) - 2))
    for cond in CONDS_MAIN + CONDS_DEC:
        row = f"  {cond:<18}"
        for s, m in cols:
            v = agg["table"][cond].get(m, {}).get(s)
            row += (f"{v:>12.2f}" if v is not None else f"{'·':>12}")
        print(row)

    h = agg["headline"]
    print("\n  headline numbers:")
    print(f"    inject_qfair − A  (strict, think-off)        : "
          f"{_fmt(h['qfair_strict_vs_A_strict'])}   (honest ceiling gap; ≈0 ⇒ parity)")
    print(f"    docnaive lenient − strict (scorer slack)     : "
          f"{_fmt(h['total_slack_lenient_minus_strict_docnaive'])}")
    print(f"    dec_latent − dec_text  (strict, collapse arm): "
          f"{_fmt(h['dec_latent_minus_dec_text_strict'])}   ← the number that matters most")

    v = _verdicts(agg)
    print("\n  " + "-" * 76)
    print(f"  discrimination : {v['discrimination']}")
    print(f"  mechanism      : {v['mechanism']}")
    print(f"  RECOMMENDATION : {v['recommend']}")
    if v["recommend"] == "SHIP_TO_PROOF_5":
        print("   → at n ≥ target, inject reaches A under strict+distractors AND latent")
        print("     beats collapsed text. Proof 4 hardened and vindicated; ship to Proof 5.")
    else:
        if v["discrimination"] == "UNDERPOWERED":
            print(f"   → too few gated items (n={agg['n_main']} < {P41_MIN_N}); add docs/"
                  "depths before reading parity. This is the n=3 guard.")
        if v["mechanism"] == "TEXT_DID_NOT_COLLAPSE":
            print("   → dec_text did not collapse; the latent−text gap is not yet")
            print("     interpretable. Lower --dec-keep-rate until text fails, then re-read.")
        if v["mechanism"] == "LATENT_ALSO_COLLAPSED":
            print("   → text collapsed and latent collapsed with it — latent did NOT carry")
            print("     the folded context. The outcome that threatens the project; investigate.")
        if v["discrimination"] == "COLLAPSED_UNDER_DISTRACTORS":
            print("   → inject collapses once decoys compete: the clean recall was an")
            print("     easy-task artifact. Distractor filler must be standard for Proof 4.")
    return v


def _fmt(v):
    return "·" if v is None else f"{v:+.3f}"


def rescore(result):
    """Re-apply the CURRENT scorers to a saved run's raw answers — no model, no GPU.
    Decoys and accepted answer forms (alts) are taken per record (by doc / from the
    stored `alts`), so a pooled multi-doc run re-scores correctly."""
    for rec in result.get("records", []):
        decoys = DECOY_VALUES.get(rec.get("doc"), ())
        alts = rec.get("alts", [])
        rec["scores"] = {key: score_all(ans, rec["gold"], decoys, alts)
                         for key, ans in rec.get("answers", {}).items()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", default=None, metavar="PATH",
                        help="re-score a saved p4_1.json with the current scorers and "
                             "regenerate the table — no model load. Writes back to --out.")
    parser.add_argument("--docs", default=None,
                        help="comma-separated discrimination docs (default: all with a "
                             "distractor bank)")
    parser.add_argument("--depths", default="0.1,0.5,0.9",
                        help="comma-separated needle depths; pooled to reach n ≥ 30 "
                             "(Proof 4 showed depth is inert, so these are independent "
                             "retrieval instances)")
    parser.add_argument("--dec-docs", default=None,
                        help="comma-separated coreference docs for the latent-vs-text arm "
                             "(default: all of synthetic_docs_long)")
    parser.add_argument("--dec-keep-rate", type=float, default=0.5,
                        help="keep-rate for the collapse arm (strided, needle_protected); "
                             "lower it until dec_text collapses")
    parser.add_argument("--no-dec", action="store_true",
                        help="skip the latent-vs-text (collapse) arm")
    parser.add_argument("--length", type=int, default=32000)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="generation budget for the think-off arms")
    parser.add_argument("--think-max-new-tokens", type=int, default=2048,
                        help="generation budget for the think-on arms (large enough to "
                             "close the trace; an unclosed think scores as no-answer)")
    parser.add_argument("--no-think-on", action="store_true",
                        help="skip the (slow) reasoning-on arm")
    parser.add_argument("--max-doc-tokens", type=int, default=40000)
    parser.add_argument("--gpus", default=None, help="default = all visible GPUs")
    parser.add_argument("--device-map", default="balanced_low_0")
    parser.add_argument("--max-mem-per-gpu", default="70GiB")
    parser.add_argument("--out", default="proofs/data/p4_1.json")
    args = parser.parse_args()

    # ── re-score path: load a saved run, re-apply scorers, regenerate the table ──
    if args.rescore:
        with open(args.rescore) as f:
            result = json.load(f)
        result = rescore(result)
        agg = aggregate(result)
        v = report(result, agg)
        result["aggregate"] = agg
        result["verdict"] = v
        out = args.out if args.out != "proofs/data/p4_1.json" else args.rescore
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nRe-scored → {out}")
        return

    if not selftest_filler():
        print("!!! filler not inert — fix proofs/long_context_docs.py first.")
        sys.exit(1)
    if not selftest_bank():
        print("!!! fact bank malformed — fix proofs/fact_bank.py first.")
        sys.exit(1)
    print(f"fact bank: {n_facts('span')} span + {n_facts('coref')} coref facts available")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from config import StitcherConfig
    cfg = StitcherConfig()
    if args.gpus:
        devices = tuple(int(x) for x in args.gpus.split(","))
    else:
        import torch
        devices = tuple(range(torch.cuda.device_count()))
        if not devices:
            raise RuntimeError("no CUDA devices visible — set CUDA_VISIBLE_DEVICES")
    print(f"sharding DeepSeek-70B across {len(devices)} GPU(s): {devices} "
          f"(device_map={args.device_map})")
    tok, model = load_deepseek(cfg, devices=devices, device_map=args.device_map,
                               max_memory_per_gpu=args.max_mem_per_gpu)

    print(f"\n########## PROOF 4.1 — hardened confirmation "
          f"({args.length} tok / L{args.layer}) ##########")
    result = run(model, tok, args)
    agg = aggregate(result)
    v = report(result, agg)
    result["aggregate"] = agg
    result["verdict"] = v

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

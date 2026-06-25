"""
Proof 5.1 — tightened latent-vs-text on the goldspan / decimated grid.

Proof 5 tested latent_sparse = needle-ONLY (an oracle condition) and never tested the
DEPLOYABLE decimated handoff — the one Exp 3.1 said latent should win. 5.1 fixes the design
to a tight, pre-registered grid with no sweeps and no redundant controls.

Conditions (8), all think-ON, L12, q-fair capture, original RoPE (no renumbered arm):
  Latent family
    latent_all        — full document (Path-1 reference, ≈ A)
    latent_goldspan   — gold supporting-sentence positions only (the Proof-5 loser, anchor)
    latent_decimated  — every-other position, keep-rate 0.5 (blind, needle_decimated)
  Text family
    text_goldspan     — gold sentences as text
    text_decimated    — every-other token, keep-rate 0.5 (the SAME kept tokens as latent_decimated)
    text_rag@best     — real RAG baseline (BGE, best of k∈{4, budget})
  Ceiling / floor
    A (full doc as text)   C (question only)

Headline numbers (judge primary, fixed-strict secondary):
  (1) latent_decimated − text_decimated   — does the 3.1 result replicate here? (sanity
      anchor; expect strongly positive ≈ +0.6. If it doesn't replicate, the harness/scorer
      differs from 3.1 and nothing else is trustworthy.)
  (2) latent_goldspan − text_goldspan      — the Proof-5 deficit, re-confirmed (≈ −0.4).
  (3) latent_decimated − latent_goldspan   — more spread context vs targeted needles. NOT
      budget-matched (decimated keeps ~Nx more positions); logged as such so nobody misreads.
  (4) (2) stratified by answer token length {1 / 2–3 / 4+} — is the deficit concentrated in
      single-token answers?

Interpretation, fixed in advance:
  (1) positive AND (2) negative → latent beats DEGRADED text but loses to CLEAN text of the
      same content. Honest stop: sparse latent doesn't beat well-formed retrieval, only
      shredded retrieval, and nobody deploys shredded retrieval.
  (4) deficit all in 1-token answers → softer stop ("can't deliver one-token facts from
      sparse positions"), not a condemnation for substantive answers.
  (3) latent_decimated ≫ latent_goldspan → distributed context matters more than targeting
      the needle; the gold-span SELECTION is the problem, reopening Path 2 with a new rule.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python proofs/p5_1.py --arm synth_multihop \
        --synth-n 40 --think-max-new-tokens 2048 --out proofs/data/p5_1_synth.json
    CUDA_VISIBLE_DEVICES=0,1,2,3 python proofs/p5_1.py --judge proofs/data/p5_1_synth.json
    python proofs/p5_1.py --rescore proofs/data/p5_1_synth.json     # fixed-strict, no GPU
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.common import (
    load_deepseek, full_prefill_answer, inject_answer_subset, normalize,
)
from core.split_forward import capture_doc_cache
from proofs.needles import needle_positions
from proofs.decimate import kept_indices, decimated_text
from proofs.retriever import (
    make_backend, Retriever, retrieval_recall, tune_chunk_size, budget_matched_k,
)
from proofs import hotpot, synth_multihop
# reuse the validated Proof-5 machinery so 5.1 runs against the SAME harness
from proofs.p5_latent_vs_rag import (
    run_gate, load_candidates, ans_qfair, needle_idx_of, gold_text_of, rag_text_of,
    judge_answer, _free_cuda, _set_think, _preflight, PREFILL_PREFIX,
    _sentences, _TENTATIVE, _wb, _golds, first_clause, score_lenient,
)

MIN_N = 30
GAP = 0.10
HEADLINE_SCORER = "strict_fixed"      # judge becomes primary once a --judge pass adds it


# ════════════════════════════════════════════════════════════════════════════════
# Fixed strict scorer — the committed answer, with the negation-SKIP bug removed.
# The old final-stance scorer skipped ANY sentence containing "not", which failed correct
# discriminating answers like "Oolan Pretsky, not Vessa Pretsky" (gold affirmed, decoy
# rejected). Here a candidate counts only if it is NOT *directly* negated — a negation cue
# in the ~20 chars before the mention — so "X, not Y" credits X and "not X but Y" credits Y.
# ════════════════════════════════════════════════════════════════════════════════
_NEG_CUES = ["not ", "n t ", "never ", "rather than ", "instead of ", "no longer "]


def _directly_negated(clause_norm, cand):
    i = clause_norm.find(cand)
    if i < 0:
        return False
    pre = clause_norm[max(0, i - 20):i]
    return any(cue in pre for cue in _NEG_CUES)


def _yesno_fixed(answer, gold):
    g = normalize(gold)
    for s in reversed(_sentences(answer)):
        ns = normalize(s)
        m = re.search(r"answer (?:is|should be|would be|here is|:)\s+(yes|no)\b", ns)
        if not m and ns in ("yes", "no"):
            m = re.match(r"(yes|no)$", ns)
        if m:
            return m.group(1) == g
    return False


def score_strict_fixed(answer, gold, decoys=(), alts=()):
    """True iff the model's final committed answer is the gold (not a decoy, not unsettled),
    reading the last sentence that affirms a NON-NEGATED candidate."""
    if normalize(gold) in ("yes", "no"):
        return _yesno_fixed(answer, gold)
    golds = [normalize(g) for g in _golds(gold, alts)]
    decs = [normalize(d) for d in decoys]
    for s in reversed(_sentences(answer)):
        ns = normalize(s)
        if any(t in ns for t in _TENTATIVE) or s.strip().endswith("?"):
            continue
        gold_hit = any(_wb(ns, g) and not _directly_negated(ns, g) for g in golds)
        dec_hit = any(_wb(ns, d) and not _directly_negated(ns, d) for d in decs)
        if gold_hit and not dec_hit:
            return True
        if dec_hit and not gold_hit:
            return False
    return False


SCORERS = ["lenient", "strict_fixed"]


def score_all(answer, gold, decoys=(), alts=()):
    return {"lenient": score_lenient(answer, gold, alts),
            "strict_fixed": score_strict_fixed(answer, gold, decoys, alts)}


# ════════════════════════════════════════════════════════════════════════════════
# Eval — the 8-condition grid (checkpointed / resumable, like p5)
# ════════════════════════════════════════════════════════════════════════════════
LATENT = ["latent_all", "latent_goldspan", "latent_decimated"]
TEXT = ["text_goldspan", "text_decimated"]
CONDS = ["A", "C"] + LATENT + TEXT + ["text_rag@best"]
RAG_LABELS = ["4", "budget"]


def _answer_tok_len(tok, gold):
    try:
        return len(tok(gold, add_special_tokens=False).input_ids)
    except TypeError:
        return len(tok(gold)["input_ids"])


def run_eval(model, tok, gated, args, backend, resume_path=None):
    layer = args.layer
    m = args.think_max_new_tokens if args.think else args.max_new_tokens

    records, prev = [], {}
    if resume_path and os.path.exists(resume_path):
        try:
            with open(resume_path) as f:
                prev = json.load(f)
            records = prev.get("records", []) or []
        except Exception as e:
            print(f"  [resume] could not read {resume_path}: {e}")
    done = {r["id"] for r in records}
    if records:
        print(f"  [resume] {len(records)} records loaded; skipped")

    if prev.get("best_chunk"):
        best_chunk = prev["best_chunk"]
    else:
        pool = gated[:args.tune_n] if len(gated) > args.tune_n else gated
        best_chunk, scores = tune_chunk_size(pool, backend, tuple(args.chunk_sizes), k=args.tune_k)
        print(f"  chunk-size tune: best={best_chunk} {scores}")

    def snapshot():
        return {"layer": layer, "best_chunk": best_chunk, "keep_rate": args.keep_rate,
                "dec_variant": args.dec_variant, "rag_labels": RAG_LABELS,
                "records": records}

    cap = args.max_eval or len(gated)
    todo = [g for g in gated if g["id"] not in done][:max(0, cap - len(records))]
    print(f"  eval plan: {len(records)} done, {len(todo)} to run (cap {cap}); "
          f"keep_rate={args.keep_rate} variant={args.dec_variant} think_max={m}")

    for rec in todo:
        doc, q, gold = rec["doc_text"], rec["question"], rec["answer"]
        decoys = rec.get("decoy_values", [])
        needle_idx = needle_idx_of(tok, rec, args.max_doc_tokens)
        if not needle_idx:
            continue
        ids = tok(doc, return_tensors="pt", truncation=True,
                  max_length=args.max_doc_tokens).input_ids
        cache, _Y, n_doc = capture_doc_cache(model, ids, layer); del _Y
        pre_ids = tok(PREFILL_PREFIX.format(document=doc), return_tensors="pt",
                      truncation=True, max_length=args.max_doc_tokens).input_ids
        qcache, _Yq, n_pre = capture_doc_cache(model, pre_ids, layer); del _Yq

        goldspan_pos = needle_positions(needle_idx, keep_sink=True)
        kept = kept_indices(n_doc, needle_idx, args.keep_rate, args.dec_pattern,
                            args.dec_variant, seed=args.dec_seed, keep_sink=True)

        retr = Retriever(backend, best_chunk).index(doc)
        bk = max(1, min(budget_matched_k(rec["gold_sentences"], best_chunk), len(retr.chunks)))
        rag = {}
        for label, k in [("4", 4), ("budget", bk)]:
            got = retr.retrieve(q, k)
            rag[label] = {"recall": retrieval_recall(got, rec["gold_sentences"]),
                          "text": rag_text_of(got), "k": k}

        _set_think(args.think)
        answers = {
            "A": rec["a_ans"],
            "C": rec.get("c_ans", ""),
            "latent_all": ans_qfair(model, tok, qcache, n_pre, q, layer, m),
            "latent_goldspan": inject_answer_subset(model, tok, cache, n_doc, goldspan_pos,
                                                    q, layer, m),
            "latent_decimated": inject_answer_subset(model, tok, cache, n_doc, kept,
                                                     q, layer, m),
            "text_goldspan": full_prefill_answer(model, tok, gold_text_of(rec), q, m,
                                                 max_length=args.max_doc_tokens),
            "text_decimated": full_prefill_answer(model, tok, decimated_text(tok, ids, kept),
                                                  q, m, max_length=args.max_doc_tokens),
        }
        for label in RAG_LABELS:
            answers[f"text_rag@{label}"] = full_prefill_answer(
                model, tok, rag[label]["text"], q, m, max_length=args.max_doc_tokens)

        records.append({
            "id": rec["id"], "question": q, "gold": gold, "type": rec.get("type", ""),
            "decoy_values": decoys,
            "k_goldspan": len(goldspan_pos), "k_decimated": len(kept), "n_doc": n_doc,
            "answer_tok_len": _answer_tok_len(tok, gold),
            "recall_by_k": {label: rag[label]["recall"] for label in RAG_LABELS},
            "answers": answers,
            "scores": {c: score_all(a, gold, decoys) for c, a in answers.items()},
        })
        if resume_path:
            with open(resume_path, "w") as f:
                json.dump(snapshot(), f, default=str)
        sc = records[-1]["scores"]
        print(f"  [{len(records)}/{cap}] {rec['id']}: "
              f"goldspan={int(sc['latent_goldspan']['strict_fixed'])} "
              f"dec={int(sc['latent_decimated']['strict_fixed'])} "
              f"dec_txt={int(sc['text_decimated']['strict_fixed'])}", end="\r")
        del cache, qcache
        _free_cuda()
    print()
    return snapshot()


# ════════════════════════════════════════════════════════════════════════════════
# Aggregate + report
# ════════════════════════════════════════════════════════════════════════════════
def _present_scorers(records):
    for r in records:
        for sd in r.get("scores", {}).values():
            return SCORERS + (["judge"] if "judge" in sd else [])
    return SCORERS


def _rate(records, cond, scorer):
    vals = [r["scores"][cond][scorer] for r in records
            if scorer in r.get("scores", {}).get(cond, {})]
    return round(sum(vals) / len(vals), 3) if vals else None


def _diff(a, b):
    return None if a is None or b is None else round(a - b, 3)


def _best_rag(records, scorer):
    rates = {lab: _rate(records, f"text_rag@{lab}", scorer) for lab in RAG_LABELS}
    rates = {k: v for k, v in rates.items() if v is not None}
    if not rates:
        return None, None
    best = max(rates, key=rates.get)
    return best, rates[best]


def _strata(records, scorer):
    """(2) stratified by answer token length: {1, 2-3, 4+}."""
    buckets = {"1": [], "2-3": [], "4+": []}
    for r in records:
        n = r.get("answer_tok_len", 0)
        key = "1" if n <= 1 else ("2-3" if n <= 3 else "4+")
        buckets[key].append(r)
    out = {}
    for key, rs in buckets.items():
        lg = _rate(rs, "latent_goldspan", scorer)
        tg = _rate(rs, "text_goldspan", scorer)
        out[key] = {"n": len(rs), "latent_goldspan": lg, "text_goldspan": tg,
                    "gap": _diff(lg, tg)}
    return out


def aggregate(result):
    records = result["records"]
    scols = _present_scorers(records)
    primary = "judge" if "judge" in scols else HEADLINE_SCORER
    table = {c: {s: _rate(records, c, s) for s in scols} for c in CONDS[:-1]}
    for lab in RAG_LABELS:
        table[f"text_rag@{lab}"] = {s: _rate(records, f"text_rag@{lab}", s) for s in scols}

    def headline(scorer):
        ld = _rate(records, "latent_decimated", scorer)
        td = _rate(records, "text_decimated", scorer)
        lg = _rate(records, "latent_goldspan", scorer)
        tg = _rate(records, "text_goldspan", scorer)
        return {
            "1_decimated_latent_minus_text": _diff(ld, td),       # sanity anchor (≈+0.6)
            "2_goldspan_latent_minus_text": _diff(lg, tg),        # the deficit (≈−0.4)
            "3_latent_decimated_minus_goldspan": _diff(ld, lg),   # context vs needles (UNMATCHED)
        }

    avg_gold = round(sum(r["k_goldspan"] for r in records) / len(records), 1) if records else 0
    avg_dec = round(sum(r["k_decimated"] for r in records) / len(records), 1) if records else 0
    return {
        "n": len(records), "primary": primary, "scorers": scols,
        "best_rag": _best_rag(records, primary),
        "table": table,
        "headline": headline(primary),
        "headline_strict": headline("strict_fixed"),
        "strata": _strata(records, primary),
        "budget": {"avg_k_goldspan": avg_gold, "avg_k_decimated": avg_dec,
                   "ratio": round(avg_dec / avg_gold, 1) if avg_gold else None},
    }


def _fmt(v):
    return "·" if v is None else f"{v:+.3f}"


def report(result, agg):
    print("\n" + "=" * 80)
    print(f"PROOF 5.1 — goldspan / decimated grid  (L{result['layer']}, think-on, "
          f"keep_rate={result.get('keep_rate')}, primary={agg['primary']})")
    print(f"  n={agg['n']}   budget: goldspan≈{agg['budget']['avg_k_goldspan']} pos, "
          f"decimated≈{agg['budget']['avg_k_decimated']} pos "
          f"({agg['budget']['ratio']}× more — (3) is NOT budget-matched)")
    br = agg["best_rag"]
    print(f"  best RAG = @{br[0]} ({br[1]})" if br[0] else "  best RAG = ·")

    cols = agg["scorers"]
    head = "  " + f"{'condition':<18}" + "".join(f"{c:>13}" for c in cols)
    print("\n" + head + "\n  " + "-" * (len(head) - 2))
    for cond in CONDS[:-1] + [f"text_rag@{l}" for l in RAG_LABELS]:
        row = f"  {cond:<18}"
        for c in cols:
            v = agg["table"].get(cond, {}).get(c)
            row += (f"{v:>13.2f}" if v is not None else f"{'·':>13}")
        print(row)

    h, hs = agg["headline"], agg["headline_strict"]
    print(f"\n  headline ({agg['primary']}; strict_fixed in parens):")
    print(f"    (1) latent_decimated − text_decimated : "
          f"{_fmt(h['1_decimated_latent_minus_text'])} ({_fmt(hs['1_decimated_latent_minus_text'])})"
          "   ← 3.1 replication / sanity anchor (want ≫0)")
    print(f"    (2) latent_goldspan − text_goldspan   : "
          f"{_fmt(h['2_goldspan_latent_minus_text'])} ({_fmt(hs['2_goldspan_latent_minus_text'])})"
          "   ← the Proof-5 deficit")
    print(f"    (3) latent_decimated − latent_goldspan: "
          f"{_fmt(h['3_latent_decimated_minus_goldspan'])} ({_fmt(hs['3_latent_decimated_minus_goldspan'])})"
          "   ← context vs needles (UNMATCHED budget)")
    print("    (4) (2) by answer token length:")
    for key in ("1", "2-3", "4+"):
        s = agg["strata"][key]
        print(f"          len {key:<4} n={s['n']:<3}  latent_goldspan={_fmt(s['latent_goldspan'])[1:]} "
              f"text_goldspan={_fmt(s['text_goldspan'])[1:]}  gap={_fmt(s['gap'])}")

    _interpret(agg)


def _interpret(agg):
    h = agg["headline"]
    n = agg["n"]
    one, two = h["1_decimated_latent_minus_text"], h["2_goldspan_latent_minus_text"]
    print("\n  " + "-" * 76)
    if n < MIN_N:
        print(f"  READ: UNDERPOWERED (n={n} < {MIN_N}).")
        return
    if one is None or two is None:
        print("  READ: incomplete.")
        return
    if one < GAP:
        print(f"  READ: SANITY-ANCHOR FAILED — (1)={one:+.3f} did not replicate 3.1's latent>text "
              "under decimation. The harness/scorer differs from 3.1; fix before trusting (2).")
    elif one >= GAP and two <= -GAP:
        print(f"  READ: HONEST STOP — latent beats DEGRADED text ((1)={one:+.3f}) but loses to "
              f"CLEAN text of the same content ((2)={two:+.3f}). Sparse latent doesn't beat "
              "well-formed retrieval, only shredded retrieval. Check (4): if the deficit is all "
              "in 1-token answers, the stop softens.")
    elif two > -GAP:
        print(f"  READ: deficit closed ((2)={two:+.3f}) — goldspan latent ≈ text. Re-examine.")
    else:
        print(f"  READ: MIXED — (1)={one:+.3f} (2)={two:+.3f}.")


# ════════════════════════════════════════════════════════════════════════════════
# Rescore + judge (reuse p5's judge primitive)
# ════════════════════════════════════════════════════════════════════════════════
def rescore(result):
    for r in result["records"]:
        decoys = r.get("decoy_values", [])
        r["scores"] = {c: score_all(a, r["gold"], decoys) for c, a in r["answers"].items()}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", default=None, metavar="PATH")
    ap.add_argument("--judge", default=None, metavar="PATH")
    ap.add_argument("--arm", default="synth_multihop",
                    choices=["hotpot", "synth_multihop", "synth_parity"])
    ap.add_argument("--max-candidates", type=int, default=400)
    ap.add_argument("--synth-n", type=int, default=40)
    ap.add_argument("--parity-n", type=int, default=32)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.set_defaults(think=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--think-max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-doc-tokens", type=int, default=4096)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--keep-rate", type=float, default=0.5)
    ap.add_argument("--dec-variant", default="needle_decimated",
                    choices=["needle_decimated", "needle_protected"])
    ap.add_argument("--dec-pattern", default="strided", choices=["strided", "random"])
    ap.add_argument("--dec-seed", type=int, default=0)
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--tune-n", type=int, default=20)
    ap.add_argument("--tune-k", type=int, default=4)
    ap.add_argument("--rag-device", default="cpu")
    ap.add_argument("--rag-backend", default="bge", choices=["bge", "hash"])
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--gate-cache", default=None)
    ap.add_argument("--out", default="proofs/data/p5_1.json")
    args = ap.parse_args()

    if args.rescore:
        with open(args.rescore) as f:
            result = rescore(json.load(f))
        agg = aggregate(result)
        report(result, agg)
        result["aggregate"] = agg
        with open(args.rescore, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nRe-scored → {args.rescore}")
        return

    if args.judge:
        global HEADLINE_SCORER
        with open(args.judge) as f:
            result = rescore(json.load(f))
        from config import StitcherConfig
        cfg = StitcherConfig()
        devices = tuple(int(x) for x in args.gpus.split(","))
        print(f"sharding DeepSeek-70B across GPUs {devices} for judging")
        tok, model = load_deepseek(cfg, devices=devices)
        conds = [c for c in CONDS if c != "text_rag@best"] + [f"text_rag@{l}" for l in RAG_LABELS]
        for i, rec in enumerate(result["records"]):
            for cond in conds:
                ans = rec["answers"].get(cond)
                if ans is None:
                    continue
                rec["scores"].setdefault(cond, {})["judge"] = judge_answer(
                    model, tok, rec["question"], rec["gold"], ans)
            print(f"  judged [{i+1}/{len(result['records'])}]", end="\r")
        print()
        # judge sanity: A should be high; if it collapses the judge env is broken
        a_j = _rate(result["records"], "A", "judge")
        a_s = _rate(result["records"], "A", "strict_fixed")
        if a_j is not None and a_s and a_s >= 0.8 and a_j < 0.5:
            print(f"  !!! JUDGE UNRELIABLE: A strict={a_s} but judge={a_j}; re-judge.")
        agg = aggregate(result)
        report(result, agg)
        result["aggregate"] = agg
        with open(args.judge, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nJudged → {args.judge}")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gate_cache = args.gate_cache or args.out.replace(".json", f"_gated_{args.arm}.json")
    need_gate = not os.path.exists(gate_cache)
    _preflight(args, need_gate)
    recs = load_candidates(args.arm, args) if need_gate else None

    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))
    print(f"sharding DeepSeek-70B across GPUs {devices}")
    tok, model = load_deepseek(cfg, devices=devices)

    if not need_gate:
        with open(gate_cache) as f:
            cached = json.load(f)
        gated = cached["gated"]
        print(f"[gate] loaded {len(gated)} gated items from {gate_cache}")
    else:
        gated, summary = run_gate(model, tok, recs, args)
        with open(gate_cache, "w") as f:
            json.dump({"gated": gated, "summary": summary}, f, default=str)
        print(f"[gate] cached → {gate_cache}")
    if not gated:
        print("No gated items.")
        return

    backend = make_backend(args.rag_backend, args.rag_device)
    result = run_eval(model, tok, gated, args, backend, resume_path=args.out)
    result["arm"] = args.arm
    agg = aggregate(result)
    report(result, agg)
    result["aggregate"] = agg
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

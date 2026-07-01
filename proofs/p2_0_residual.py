"""
Proof 2.0 — Residual-equivalence (Chain 2, rung 0).

Chain 1 always injected the captured KV cache (layers target_layer..L-1, the big object).
An SLM can only realistically produce the layer-`target_layer` RESIDUAL STREAM Y_doc — one
(1, N, D) tensor — and let the LLM recompute the upper stack from it. Before any cross-model
work: does injecting the true residual and recomputing 12→80 match injecting the stored
cache? If yes, the SLM has a small, well-defined target and Chain 2 proceeds. If no,
outsourcing is fighting a harder object than Chain 1's success implied.

This is single-model (DeepSeek), no SLM, no training. It is partly a CORRECTNESS check (do
the two paths agree for true states, as theory says — a mismatch is a plumbing bug) and
partly a SETUP (establishing the residual stream as the handoff target Proof 2.1's oracle
map, and Proof 2.2's stitcher, will aim at).

The CPU half of the correctness check is already proven without the 70B: self-test
invariant G (`python core/selftest_split_forward.py`) shows the residual recompute
reproduces the stored-cache split-forward token-for-token on a tiny model. This script is
the BEHAVIOURAL half on real facts at the pinned operating point.

Operating point (pinned from 4.1 / Proof 5, non-negotiable): layer 12, q-fair capture,
think-ON, strict + LLM-judge. Zero-memory synthetic gated set (C fails AND A succeeds), so
recall can only come from the injected document, never parametric memory.

Three conditions per gated item:
    A               full-document text prefill                        (the ceiling; reused from gate)
    cache_inject    q-fair capture, STORED upper cache injected       (the Chain-1 mechanism, ≈A per Proof 5)
    residual_inject q-fair capture, upper cache RECOMPUTED from Y_doc (the new small-object path)

The numbers that decide the rung:
  (1) residual_inject − cache_inject (judge)  — THE HEADLINE. ≈0 ⇒ the small object suffices.
  (2) residual_inject − A                     — does the residual path still equal reading.
  (3) cache_inject − A                        — sanity: the stored path still ≈ reading (Proof 5).
  (4) per-item agreement / confusion          — do the two paths agree ITEM BY ITEM, or only
                                                on aggregate (differently-lossy, netting to ~0)?
  (5) upper-KV drift (cos / MSE)              — is any gap numerical (tiny) or structural (large)?

Usage (pick physical GPUs with CUDA_VISIBLE_DEVICES, exactly like p5 / run_chain; eval is
checkpointed to --out after every item and resumes from a partial --out):
    CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p2_0_residual.py --arm synth_multihop \
        --synth-n 40 --think-max-new-tokens 1024 --out proofs/data/p2_0.json
    # cheap wire-test of the whole pipeline before committing GPU hours:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p2_0_residual.py --arm synth_multihop \
        --synth-n 6 --no-think --no-judge
    # re-score / re-verdict a saved run (no GPU):
    python proofs/p2_0_residual.py --rescore proofs/data/p2_0.json
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import proofs.common as _common
from proofs.common import (
    load_deepseek, final_answer, _with_think_control,
    capture_doc_cache, split_forward_generate,
    recompute_doc_cache_from_residual, kv_drift_upper,
)
# Reuse p5's gate, scoring, judge, and q-fair framing verbatim — same operating point.
from proofs.p5_latent_vs_rag import (
    run_gate, load_candidates, score_all, judge_answer, ans_qfair,
    _present_scorers, _set_think, _free_cuda,
    PREFILL_PREFIX, PREFILL_QSUFFIX, MIN_N, GAP,
)

# Item-by-item agreement floor: how much cache/residual disagreement is tolerable before
# "aggregates match" stops meaning "the paths are equivalent". ≤10% off-diagonal is clean.
AGREE_TOL = 0.10

CONDS = ["A", "cache_inject", "residual_inject"]


# ── the residual-inject condition (q-fair capture, recomputed upper cache) ────────
def ans_qfair_residual(model, tok, qcache, Yq, n_pre, q, layer, m):
    """residual_inject at the q-fair operating point: recompute the upper cache from the
    q-fair capture's layer-`layer` residual `Yq` (not the stored `qcache` KV) and answer the
    q-fair question suffix. Returns (answer, upper-KV drift vs the stored cache)."""
    resid_cache = recompute_doc_cache_from_residual(model, Yq, n_pre, layer, qcache)
    drift = kv_drift_upper(qcache, resid_cache, layer)
    query = _with_think_control(PREFILL_QSUFFIX.format(question=q))
    txt = split_forward_generate(model, tok, resid_cache, n_pre, query_text=query,
                                 target_layer=layer, max_new_tokens=m)
    del resid_cache
    return final_answer(txt), drift


# ════════════════════════════════════════════════════════════════════════════════
# Stage: eval (A / cache_inject / residual_inject on the gated set)
# ════════════════════════════════════════════════════════════════════════════════
def run_eval(model, tok, gated, args, resume_path=None):
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
    done_ids = {r["id"] for r in records}
    if records:
        print(f"  [resume] {len(records)} eval records loaded; their ids are skipped")

    def snapshot():
        return {"layer": layer, "think": args.think, "judged": args.judge,
                "records": records}

    cap = args.max_eval or len(gated)
    todo = [g for g in gated if g["id"] not in done_ids][:max(0, cap - len(records))]
    print(f"  eval plan: {len(records)} done, {len(todo)} to run this session "
          f"(cap {cap}, gated {len(gated)}); think_max={m}, judge={args.judge}")

    for rec in todo:
        doc, q, gold = rec["doc_text"], rec["question"], rec["answer"]
        decoys = rec.get("decoy_values", [])

        # q-fair capture: document states captured INSIDE A's instruction+document framing,
        # so the only residual asymmetry vs A is the (causally-later) question. Y_pre is the
        # residual-inject handoff object; qcache is the cache-inject reference.
        pre_ids = tok(PREFILL_PREFIX.format(document=doc), return_tensors="pt",
                      truncation=True, max_length=args.max_doc_tokens).input_ids
        qcache, Y_pre, n_pre = capture_doc_cache(model, pre_ids, layer)

        _set_think(args.think)
        cache_ans = ans_qfair(model, tok, qcache, n_pre, q, layer, m)
        resid_ans, drift = ans_qfair_residual(model, tok, qcache, Y_pre, n_pre, q, layer, m)

        answers = {"A": rec["a_ans"], "cache_inject": cache_ans, "residual_inject": resid_ans}
        scores = {c: score_all(a, gold, decoys) for c, a in answers.items()}
        if args.judge:
            for c, a in answers.items():
                scores[c]["judge"] = judge_answer(model, tok, q, gold, a)

        records.append({
            "id": rec["id"], "question": q, "gold": gold, "type": rec.get("type", ""),
            "decoy_values": decoys, "n_pre": int(n_pre),
            "answers": answers, "scores": scores, "kv_drift": drift,
        })
        if resume_path:
            with open(resume_path, "w") as f:
                json.dump(snapshot(), f, default=str)
        s = args.headline if args.headline in scores["residual_inject"] else "strict"
        print(f"  eval [{len(records)} done / {cap}] {rec['id']}: "
              f"cache={int(scores['cache_inject'][s])} "
              f"resid={int(scores['residual_inject'][s])} "
              f"cos_k={drift.get('cos_k')}", end="\r")
        del qcache, Y_pre
        _free_cuda()
    print()
    return snapshot()


# ════════════════════════════════════════════════════════════════════════════════
# Aggregation: rates, headline diffs, per-item confusion, KV drift
# ════════════════════════════════════════════════════════════════════════════════
def _rate(records, cond, scorer):
    vals = [r["scores"][cond][scorer] for r in records
            if scorer in r.get("scores", {}).get(cond, {})]
    return round(sum(vals) / len(vals), 3) if vals else None


def _diff(a, b):
    return None if a is None or b is None else round(a - b, 3)


def confusion(records, scorer):
    """Per-item agreement between cache_inject and residual_inject. The headline number is
    an aggregate; this checks whether the two paths agree ITEM BY ITEM or merely average to
    the same rate (differently-lossy). `both`/`neither` are agreement; the off-diagonal
    (`cache_only`, `resid_only`) is disagreement."""
    c = {"both": 0, "neither": 0, "cache_only": 0, "resid_only": 0, "n": 0}
    for r in records:
        sc = r["scores"].get("cache_inject", {})
        sr = r["scores"].get("residual_inject", {})
        if scorer not in sc or scorer not in sr:
            continue
        ci, ri = bool(sc[scorer]), bool(sr[scorer])
        c["n"] += 1
        if ci and ri:
            c["both"] += 1
        elif not ci and not ri:
            c["neither"] += 1
        elif ci and not ri:
            c["cache_only"] += 1
        else:
            c["resid_only"] += 1
    n = c["n"] or 1
    c["agree"] = round((c["both"] + c["neither"]) / n, 3)
    c["disagree"] = round((c["cache_only"] + c["resid_only"]) / n, 3)
    return c


def drift_summary(records):
    keys = ["cos_k", "cos_v", "mse_k", "mse_v"]
    out = {}
    for k in keys:
        vals = [r["kv_drift"][k] for r in records
                if r.get("kv_drift", {}).get(k) is not None]
        out[k] = round(sum(vals) / len(vals), 6) if vals else None
    out["n"] = len([r for r in records if r.get("kv_drift", {}).get("cos_k") is not None])
    return out


def aggregate(result, headline):
    records = result["records"]
    scols = _present_scorers(records)
    table = {c: {s: _rate(records, c, s) for s in scols} for c in CONDS}

    def _head(scorer):
        rc = _rate(records, "cache_inject", scorer)
        rr = _rate(records, "residual_inject", scorer)
        ra = _rate(records, "A", scorer)
        return {"resid_minus_cache": _diff(rr, rc),   # (1) THE HEADLINE
                "resid_minus_A": _diff(rr, ra),       # (2)
                "cache_minus_A": _diff(rc, ra)}       # (3) sanity: stored path ≈ reading

    hs = headline if headline in scols else "strict"
    head = _head(hs)
    head["scorer"] = hs
    head["strict_reference"] = _head("strict")
    return {
        "n": len(records),
        "distinct": len({r["id"] for r in records}),
        "table": table,
        "headline": head,
        "confusion": confusion(records, hs),
        "kv_drift": drift_summary(records),
    }


def verdict(agg):
    n = agg["n"]
    h = agg["headline"]
    conf = agg["confusion"]
    drift = agg["kv_drift"]
    d = h["resid_minus_cache"]
    ra = h["resid_minus_A"]

    if n < MIN_N:
        return {"status": "UNDERPOWERED",
                "detail": f"n={n} < {MIN_N}; add gated items (raise --synth-n) before "
                          "reading the gaps."}
    if d is None:
        return {"status": "MISSING", "detail": "no scored records for the headline."}

    disagree = conf["disagree"]
    equal = abs(d) <= GAP
    clean = disagree <= AGREE_TOL

    if d <= -GAP:
        # residual < cache: recomputing the upper stack loses what the stored cache had.
        # Distinguish plumbing/precision (large KV MSE) from structural (KV ≈ equal but
        # behaviour differs — something lives in the stored cache the residual doesn't fix).
        mse = drift.get("mse_k")
        if mse is not None and mse > 1e-2:
            status = "RESIDUAL_LOSSY_NUMERICAL"   # recomputed KV far from stored → fix the plumbing
        else:
            status = "RESIDUAL_LOSSY_STRUCTURAL"  # KV close but behaviour worse → SLM on the hook for more
    elif equal and clean:
        status = "RESIDUAL_SUFFICIENT"            # the hoped-for outcome — green light to Proof 2.1
    elif equal and not clean:
        status = "DIFFERENTLY_LOSSY"              # aggregates match, per-item diverges — investigate
    else:  # d >= GAP: residual BEATS cache (unexpected — the recompute should not add signal)
        status = "RESIDUAL_EXCEEDS_CACHE"
    return {"status": status,
            "detail": f"(1) resid−cache {d}  (2) resid−A {ra}  (3) cache−A "
                      f"{h['cache_minus_A']}  | agree {conf['agree']} "
                      f"(cache-only {conf['cache_only']}, resid-only {conf['resid_only']})  "
                      f"| cos_k {drift.get('cos_k')} mse_k {drift.get('mse_k')}"}


_GLOSS = {
    "RESIDUAL_SUFFICIENT":
        "   → the layer-residual reproduces the stored cache; the SLM's target is small "
        "and well-defined (one (N,d) tensor). GREEN LIGHT to Proof 2.1.",
    "RESIDUAL_LOSSY_NUMERICAL":
        "   → recomputed upper KV is numerically far from stored: a plumbing/precision bug "
        "(position_ids, mask, dtype, cache fill). Fix split_forward before trusting either.",
    "RESIDUAL_LOSSY_STRUCTURAL":
        "   → recomputed KV ≈ stored but behaviour is worse: something in the stored upper "
        "cache the layer-residual doesn't determine. Surprising; the SLM would owe a bigger "
        "object. Investigate before Proof 2.1.",
    "DIFFERENTLY_LOSSY":
        "   → aggregates match but the paths disagree item-by-item: they are not equivalent, "
        "they are differently-lossy and happen to average out. Investigate the off-diagonal.",
    "RESIDUAL_EXCEEDS_CACHE":
        "   → residual scores ABOVE the stored cache — unexpected (the recompute cannot add "
        "signal). Likely noise (small n) or a scoring artifact; inspect before trusting.",
    "UNDERPOWERED": "   → too few gated items to read the gaps; raise --synth-n.",
    "MISSING": "   → no scored records; nothing to conclude.",
}


# ════════════════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════════════════
def _fmt(v):
    return "·" if v is None else f"{v:+.3f}"


def report(result, agg, gate_summary, headline):
    print("\n" + "=" * 80)
    print(f"PROOF 2.0 — residual-equivalence  (L{result['layer']}, "
          f"{'think-on' if result.get('think') else 'think-off'}, headline={headline})")
    if gate_summary:
        print(f"  gate: {gate_summary['gated']} gated / {gate_summary['candidates']} "
              f"candidates  (closed-book discard {gate_summary['discard_rate']}, "
              f"A pass {gate_summary['a_pass_rate']})")
    print(f"  eval n={agg['n']} (distinct {agg['distinct']})")

    cols = _present_scorers(result["records"])
    head = "  " + f"{'condition':<18}" + "".join(f"{c:>11}" for c in cols)
    print("\n" + head)
    print("  " + "-" * (len(head) - 2))
    for cond in CONDS:
        row = f"  {cond:<18}"
        for c in cols:
            v = agg["table"].get(cond, {}).get(c)
            row += (f"{v:>11.2f}" if v is not None else f"{'·':>11}")
        print(row)

    h = agg["headline"]
    hs = h.get("strict_reference", {})
    print(f"\n  headline ({h.get('scorer')}; strict in parens):")
    print(f"    (1) residual_inject − cache_inject : "
          f"{_fmt(h['resid_minus_cache'])} ({_fmt(hs.get('resid_minus_cache'))})"
          "   ← THE HEADLINE — ≈0 ⇒ small object suffices")
    print(f"    (2) residual_inject − A            : "
          f"{_fmt(h['resid_minus_A'])} ({_fmt(hs.get('resid_minus_A'))})"
          "   ← residual path still equals reading")
    print(f"    (3) cache_inject − A               : "
          f"{_fmt(h['cache_minus_A'])} ({_fmt(hs.get('cache_minus_A'))})"
          "   ← sanity: stored path ≈ reading (Proof 5)")

    c = agg["confusion"]
    print(f"\n  per-item agreement ({h.get('scorer')}): agree {c['agree']} "
          f"(both {c['both']}, neither {c['neither']}); disagree {c['disagree']} "
          f"(cache-only {c['cache_only']}, resid-only {c['resid_only']}) of n={c['n']}")

    d = agg["kv_drift"]
    print(f"  upper-KV drift (stored vs recomputed, n={d['n']}): "
          f"cos_k {d['cos_k']}  cos_v {d['cos_v']}  mse_k {d['mse_k']}  mse_v {d['mse_v']}")

    v = verdict(agg)
    print("\n  " + "-" * 76)
    print(f"  VERDICT: {v['status']}")
    print(f"    {v['detail']}")
    if v["status"] in _GLOSS:
        print(_GLOSS[v["status"]])
    return v


# ════════════════════════════════════════════════════════════════════════════════
# Inspector — items where cache and residual disagree (the load-bearing diagnostic)
# ════════════════════════════════════════════════════════════════════════════════
def show_disagreements(result, headline, n=12):
    recs = result.get("records", [])
    hs = headline
    print("\n" + "=" * 80)
    print(f"INSPECT — cache/residual disagreements ({hs}), of {len(recs)} items")
    shown = 0
    for r in recs:
        sc = r["scores"].get("cache_inject", {})
        sr = r["scores"].get("residual_inject", {})
        if hs not in sc or hs not in sr or bool(sc[hs]) == bool(sr[hs]):
            continue
        shown += 1
        who = "cache✓/resid✗" if sc[hs] else "cache✗/resid✓"
        print(f"\n  [{r['id']}] gold={r['gold']!r}  {who}  "
              f"drift cos_k={r.get('kv_drift', {}).get('cos_k')}")
        print(f"    cache_inject    : {r['answers']['cache_inject'][:240]!r}")
        print(f"    residual_inject : {r['answers']['residual_inject'][:240]!r}")
        if shown >= n:
            break
    if not shown:
        print("  (none — cache and residual agree on every item for this scorer)")


# ════════════════════════════════════════════════════════════════════════════════
# Rescore (no GPU)
# ════════════════════════════════════════════════════════════════════════════════
def rescore(result):
    for r in result["records"]:
        decoys = r.get("decoy_values", [])
        judged = {c: r["scores"][c].get("judge") for c in r["scores"]
                  if "judge" in r["scores"].get(c, {})}
        r["scores"] = {c: score_all(a, r["gold"], decoys) for c, a in r["answers"].items()}
        for c, jv in judged.items():           # preserve any saved judge verdicts
            r["scores"][c]["judge"] = jv
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", default=None, metavar="PATH",
                    help="re-score / re-verdict a saved run with current scorers; no model load")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="with --rescore: print up to N cache/residual DISAGREEMENT items")
    ap.add_argument("--arm", default="synth_multihop",
                    choices=["synth_multihop", "synth_parity", "hotpot"],
                    help="gated set; default synth_multihop = zero-memory (no parametric recall)")
    ap.add_argument("--synth-n", type=int, default=40,
                    help="synthetic item count (≥30 to clear the verdict power floor)")
    ap.add_argument("--parity-n", type=int, default=32)
    ap.add_argument("--max-candidates", type=int, default=400, help="hotpot arm only")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="suppress reasoning (smoke only; the operating point is think-ON)")
    ap.set_defaults(think=True)
    ap.add_argument("--no-judge", dest="judge", action="store_false",
                    help="skip the inline LLM-judge (deterministic scorers only)")
    ap.set_defaults(judge=True)
    ap.add_argument("--headline", default="judge",
                    help="primary scorer for the headline/verdict (default judge; falls back "
                         "to strict if the judge pass was skipped)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--think-max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-doc-tokens", type=int, default=4096)
    ap.add_argument("--max-eval", type=int, default=None,
                    help="cap gated items evaluated this run (resume-safe; default = all)")
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="logical GPU indices for DeepSeek shards (pick physical GPUs with "
                         "CUDA_VISIBLE_DEVICES, exactly like p5 / run_chain)")
    ap.add_argument("--gate-cache", default=None,
                    help="path to cache/reuse the gated set (default derived from --out)")
    ap.add_argument("--out", default="proofs/data/p2_0.json")
    args = ap.parse_args()

    if args.rescore:
        with open(args.rescore) as f:
            result = rescore(json.load(f))
        agg = aggregate(result, args.headline)
        v = report(result, agg, result.get("gate_summary"), agg["headline"]["scorer"])
        result["aggregate"], result["verdict"] = agg, v
        if args.show:
            show_disagreements(result, agg["headline"]["scorer"], args.show)
        with open(args.rescore, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nRe-scored → {args.rescore}")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gate_cache = args.gate_cache or args.out.replace(".json", f"_gated_{args.arm}.json")
    need_gate = not os.path.exists(gate_cache)

    recs = None
    if need_gate:
        recs = load_candidates(args.arm, args)
        print(f"[gate] resolved {len(recs)} {args.arm} candidates (pre-load)")

    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))
    print(f"sharding DeepSeek-70B across GPUs {devices} "
          "(select physical GPUs with CUDA_VISIBLE_DEVICES)")
    tok, model = load_deepseek(cfg, devices=devices)

    if not need_gate:
        with open(gate_cache) as f:
            cached = json.load(f)
        gated, gate_summary = cached["gated"], cached["summary"]
        print(f"[gate] loaded {len(gated)} gated items from {gate_cache}")
    else:
        print(f"[gate] gating {len(recs)} {args.arm} candidates (think-on)…")
        gated, gate_summary = run_gate(model, tok, recs, args)
        with open(gate_cache, "w") as f:
            json.dump({"gated": gated, "summary": gate_summary}, f, default=str)
        print(f"[gate] cached → {gate_cache}")

    if not gated:
        print("No gated items — nothing to evaluate. (All memorized or all A-unanswerable.)")
        return

    result = run_eval(model, tok, gated, args, resume_path=args.out)
    result["arm"] = args.arm
    result["gate_summary"] = gate_summary
    agg = aggregate(result, args.headline)
    v = report(result, agg, gate_summary, agg["headline"]["scorer"])
    result["aggregate"], result["verdict"] = agg, v

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

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

The number to look at first is dec_latent − dec_text under STRICT scoring WITH
distractors: if the Exp-3.1 mechanism (latent carries what text loses) survives the
hardest, fairest test, nothing else in the table can sink the project.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python proofs/p4_1_hardened.py \
        --doc zorvian_codex --length 32000 --layer 12 --depth 0.5 \
        --out proofs/data/p4_1.json
    # skip the (slow) think-on arm:
    ... python proofs/p4_1_hardened.py --no-think-on
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
from proofs.synthetic_docs import doc_by_name


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


# ── distractor bank (Hardening 3): same surface form, WRONG values ─────────────
# Hedged decoys ("some chroniclers insist", "a disputed pamphlet"), so the plainly
# stated true needle should still win for A — but every question gets ≥1 wrong-value
# competitor it must discriminate against. None contains a true gold (asserted below).
DISTRACTORS = {
    "zorvian_codex": [
        "Some chroniclers insist the Zorvian Codex was first catalogued in the year "
        "1602 by the explorer Toren Vask, who is said to have raised it from the ruins "
        "of Antial.",
        "A competing tradition holds that the codex contains exactly 2,118 verses and "
        "was set down by the philosopher Esca Morrow during his long exile.",
        "According to one disputed pamphlet, the scholar Halvard Crane produced the "
        "first complete translation in 1889, well before any rival attempt.",
        "It is occasionally claimed that the manuscript was attributed to the poet Sela "
        "Brunn and that it numbers some 4,000 stanzas in all.",
        "An old rumor maintains the work was recovered by Dalen Roost and first entered "
        "the catalogues in the year 1450.",
    ],
}


# ── the three scorers (Hardening 1) ────────────────────────────────────────────
_CARRIERS = ["", "the answer is ", "it is ", "it was ", "answer ", "answer is ",
             "this is ", "that is ", "the answer is the "]


def first_clause(answer: str) -> str:
    """The answer clause: the first non-empty line, then its first sentence. This is
    where a direct answer lives, as opposed to a later restatement of the question."""
    line = next((l for l in answer.splitlines() if l.strip()), "")
    parts = re.split(r"(?<=[.!?])\s", line.strip())
    return parts[0] if parts else line.strip()


def score_lenient(answer: str, gold: str) -> bool:
    """The chain's containment scorer: gold appears anywhere in the answer."""
    return normalize(gold) in normalize(answer)


def score_firstline(answer: str, gold: str) -> bool:
    """Gold appears in the answer CLAUSE (first sentence), not buried in a restatement."""
    return normalize(gold) in normalize(first_clause(answer))


def score_strict(answer: str, gold: str) -> bool:
    """The answer clause IS the gold (modulo a tiny answer-carrier phrase). A reply that
    restates the question's whole sentence — 'The codex was recovered by Maren Velloth' —
    fails; a direct 'Maren Velloth' passes. This is the harsh end of the ladder; the
    lenient−strict gap is the measurement of how much containment was inflating."""
    core = normalize(first_clause(answer))
    g = normalize(gold)
    return any(core == (c + g).strip() for c in _CARRIERS)


SCORERS = {"lenient": score_lenient, "firstline": score_firstline, "strict": score_strict}


def score_all(answer: str, gold: str) -> dict:
    return {name: fn(answer, gold) for name, fn in SCORERS.items()}


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


# ── main evaluation ────────────────────────────────────────────────────────────
def run(model, tok, args):
    base = doc_by_name(args.doc)
    distractors = DISTRACTORS.get(args.doc)
    if not distractors:
        raise SystemExit(f"no distractor bank authored for doc {args.doc!r} — add one "
                         "to DISTRACTORS before running Proof 4.1 on it.")

    # No distractor may contain a true gold (else it stops being a near-MISS).
    for qa in base["qa"]:
        for d in distractors:
            assert normalize(qa["a"]) not in normalize(d), \
                f"distractor leaks gold {qa['a']!r}: {d!r}"

    layer, mnt = args.layer, args.max_new_tokens
    tmnt = args.think_max_new_tokens
    modes = ["off"] if args.no_think_on else ["off", "on"]

    # Build the 32k distractor document (with fact) and its filler-only twin (distractors
    # remain, fact removed) — the C_filler-with-distractors gate.
    doc = build_distractor_doc(tok, base, args.length, args.depth, distractors,
                               max_doc_tokens=args.max_doc_tokens)
    filler = build_distractor_doc(tok, base, args.length, args.depth, distractors,
                                  max_doc_tokens=args.max_doc_tokens, drop_fact=True)
    print(f"\nbuilt {args.doc}: {doc['n_tokens']} tok, fact@depth={doc['depth_actual']}, "
          f"{doc['n_distractors']} distractors  (filler-only twin: {filler['n_tokens']} tok)")

    # Tokenize once: ids drive the capture and the text-decimation arm.
    ids = tok(doc["text"], return_tensors="pt", truncation=True,
              max_length=args.max_doc_tokens).input_ids

    # Needle positions per question (true needle only; decoys have different wording).
    for qa in base["qa"]:
        qa["needle_idx"] = span_token_positions(tok, doc["text"], qa["needle"],
                                                args.max_doc_tokens)

    # ── captures (document-only and q-fair) ───────────────────────────────────
    print(f"capturing docnaive cache @ layer {layer} ({doc['n_tokens']} tok) …")
    cache, _Y, n_doc = capture_doc_cache(model, ids, layer)
    del _Y

    pre_text = PREFILL_PREFIX.format(document=doc["text"])
    pre_ids = tok(pre_text, return_tensors="pt", truncation=True,
                  max_length=args.max_doc_tokens).input_ids
    print(f"capturing q-fair cache @ layer {layer} ({pre_ids.shape[1]} tok, "
          "instruction+document framing) …")
    qcache, _Yq, n_pre = capture_doc_cache(model, pre_ids, layer)
    del _Yq

    # ── SANITY GATES (run + eyeball before trusting anything) ─────────────────
    _set_think("off")
    print("\n" + "=" * 64)
    print("SANITY GATES")
    # 1. subset-to-all no-op: inject all positions == full inject (bookkeeping check)
    canary_mismatch = 0
    for qa in base["qa"]:
        kept_all = kept_indices(n_doc, qa["needle_idx"], 1.0, "strided",
                                "needle_decimated", seed=0, keep_sink=True)
        a_sub = inject_answer_subset(model, tok, cache, n_doc, kept_all, qa["q"],
                                     layer, mnt)
        a_full = inject_answer(model, tok, cache, n_doc, qa["q"], layer, mnt)
        if a_sub.strip() != a_full.strip():
            canary_mismatch += 1
    print(f"  1. subset-to-all no-op : {'OK' if not canary_mismatch else f'{canary_mismatch} MISMATCH'}"
          "  (inject-all-positions must equal full inject)")

    # 2/3. C and C_filler must FAIL with distractors present; eyeball 5 raw injects.
    print("  2. C / C_filler (with distractors) must FAIL (lenient), A must SUCCEED:")
    gate = {}
    for qa in base["qa"]:
        q, gold = qa["q"], qa["a"]
        c = no_context_answer(model, tok, q, mnt)
        cf = ans_A(model, tok, filler["text"], q, mnt, args.max_doc_tokens)
        a = ans_A(model, tok, doc["text"], q, mnt, args.max_doc_tokens)
        c_ok, cf_ok, a_ok = score_lenient(c, gold), score_lenient(cf, gold), score_lenient(a, gold)
        gated = (not c_ok) and (not cf_ok) and a_ok
        gate[q] = {"c": c, "c_filler": cf, "a": a, "c_ok": c_ok, "cf_ok": cf_ok,
                   "a_ok": a_ok, "gated": gated}
        print(f"     {q!r}\n       C={c_ok} C_filler={cf_ok} A={a_ok}  gated={gated}"
              f"   C_filler→{first_clause(cf)[:60]!r}")
    print("  3. eyeball — 5 raw 32k injected (docnaive, think-off) answers vs gold:")
    eyeball = []
    for qa in base["qa"]:
        a_inj = ans_docnaive(model, tok, cache, n_doc, qa["q"], layer, mnt)
        eyeball.append({"q": qa["q"], "gold": qa["a"], "answer": a_inj})
        print(f"     gold={qa['a']!r}\n       inj → {a_inj[:90]!r}")

    gated_qs = [qa for qa in base["qa"] if gate[qa["q"]]["gated"]]
    print(f"\n  → gated questions (C&C_filler fail, A succeeds): {len(gated_qs)}/{len(base['qa'])}")

    # ── the conditions, every which way, on the gated set ─────────────────────
    records = []
    for qa in gated_qs:
        q, gold, idx = qa["q"], qa["a"], qa["needle_idx"]
        kept = kept_indices(n_doc, idx, args.keep_rate, "strided", "needle_protected",
                            seed=0, keep_sink=True)
        rec = {"question": q, "gold": gold, "k_needle": len(idx),
               "kept_count": len(kept), "keep_rate": args.keep_rate, "answers": {}, "scores": {}}

        for mode in modes:
            _set_think(mode)
            m = mnt if mode == "off" else tmnt
            outs = {
                "A": ans_A(model, tok, doc["text"], q, m, args.max_doc_tokens),
                "inject_docnaive": ans_docnaive(model, tok, cache, n_doc, q, layer, m),
                "inject_qfair": ans_qfair(model, tok, qcache, n_pre, q, layer, m),
            }
            if mode == "off":
                # sparse handoff + the latent-vs-text contrast: extraction path only.
                outs["needles_only"] = inject_answer_subset(
                    model, tok, cache, n_doc, needle_positions(idx, keep_sink=True),
                    q, layer, m)
                outs["dec_text"] = ans_A(
                    model, tok, decimated_text(tok, ids, kept), q, m, args.max_doc_tokens)
                outs["dec_latent"] = inject_answer_subset(
                    model, tok, cache, n_doc, kept, q, layer, m)
            for cond, ans in outs.items():
                rec["answers"][f"{cond}@{mode}"] = ans
                rec["scores"][f"{cond}@{mode}"] = score_all(ans, gold)
        records.append(rec)
        print(f"  scored {q!r}")

    del cache, qcache

    return {
        "doc": args.doc, "length": args.length, "n_tokens": doc["n_tokens"],
        "layer": layer, "depth": args.depth, "depth_actual": doc["depth_actual"],
        "keep_rate": args.keep_rate, "n_distractors": doc["n_distractors"],
        "modes": modes, "sanity_canary_mismatch": canary_mismatch,
        "gate": gate, "eyeball": eyeball, "gated_n": len(gated_qs),
        "records": records,
    }


# ── aggregation + report ────────────────────────────────────────────────────────
CONDS_BOTH = ["A", "inject_docnaive", "inject_qfair"]
CONDS_OFF = ["needles_only", "dec_text", "dec_latent"]


def aggregate(result):
    recs = result["records"]
    n = len(recs)
    modes = result["modes"]
    table = {}

    def rate(cond, mode, scorer):
        key = f"{cond}@{mode}"
        vals = [r["scores"][key][scorer] for r in recs if key in r["scores"]]
        return round(sum(vals) / len(vals), 3) if vals else None

    for cond in CONDS_BOTH + CONDS_OFF:
        cond_modes = modes if cond in CONDS_BOTH else ["off"]
        table[cond] = {m: {s: rate(cond, m, s) for s in SCORERS} for m in cond_modes}

    # The headline numbers, computed once.
    a_strict = table["A"]["off"]["strict"]
    qfair_strict = table["inject_qfair"]["off"]["strict"]
    qfair_lenient = table["inject_qfair"]["off"]["lenient"]
    dl = table["dec_latent"]["off"]["strict"]
    dt = table["dec_text"]["off"]["strict"]
    headline = {
        "qfair_strict_vs_A_strict": (None if a_strict is None or qfair_strict is None
                                     else round(qfair_strict - a_strict, 3)),
        "total_slack_lenient_minus_strict_docnaive": (
            None if table["inject_docnaive"]["off"]["lenient"] is None
            else round(table["inject_docnaive"]["off"]["lenient"]
                       - table["inject_docnaive"]["off"]["strict"], 3)),
        "dec_latent_minus_dec_text_strict": (None if dl is None or dt is None
                                             else round(dl - dt, 3)),
    }
    return {"n_gated": n, "table": table, "headline": headline}


def report(result, agg):
    modes = result["modes"]
    print("\n" + "=" * 78)
    print(f"PROOF 4.1 — hardened single-point confirmation "
          f"({result['n_tokens']} tok, L{result['layer']}, depth {result['depth_actual']}, "
          f"{result['n_distractors']} distractors)")
    print(f"  gated questions = {agg['n_gated']}   canary mismatches = "
          f"{result['sanity_canary_mismatch']}  (must be 0)")

    # header
    cols = [(s, m) for m in modes for s in SCORERS]
    head = "  " + f"{'condition':<18}" + "".join(f"{s[:4]+'/'+m:>12}" for s, m in cols)
    print("\n" + head)
    print("  " + "-" * (len(head) - 2))
    for cond in CONDS_BOTH + CONDS_OFF:
        row = f"  {cond:<18}"
        for s, m in cols:
            v = agg["table"][cond].get(m, {}).get(s)
            row += (f"{v:>12.2f}" if v is not None else f"{'·':>12}")
        print(row)

    h = agg["headline"]
    print("\n  headline numbers:")
    print(f"    inject_qfair − A  (strict, think-off)        : "
          f"{_fmt(h['qfair_strict_vs_A_strict'])}   (honest ceiling gap; ≈0 ⇒ parity)")
    print(f"    docnaive lenient − strict (the scorer slack) : "
          f"{_fmt(h['total_slack_lenient_minus_strict_docnaive'])}")
    print(f"    dec_latent − dec_text  (strict, distractors) : "
          f"{_fmt(h['dec_latent_minus_dec_text_strict'])}   ← the number that matters most")

    # fixed-in-advance interpretation
    print("\n  " + "-" * 74)
    a_s = agg["table"]["A"]["off"]["strict"]
    qf_s = agg["table"]["inject_qfair"]["off"]["strict"]
    qf_len = agg["table"]["inject_qfair"]["off"]["lenient"]
    c_floor = 0.0  # gated by construction
    gap_dl = h["dec_latent_minus_dec_text_strict"]
    verdict = "SEE_TABLE"
    if qf_s is not None and a_s is not None:
        if qf_s >= a_s - 0.05 and (gap_dl is not None and gap_dl >= 0.05):
            verdict = "VINDICATED_HARDENED"
        elif qf_len is not None and qf_len >= 0.8 and qf_s < a_s - 0.05:
            verdict = "SCALES_NOT_PARITY"
        elif qf_len is not None and qf_len <= 0.3:
            verdict = "EASY_TASK_ARTIFACT"
    if gap_dl is not None and gap_dl <= 0.0 and verdict != "EASY_TASK_ARTIFACT":
        verdict = "MECHANISM_SCORER_INFLATED"
    print(f"  VERDICT: {verdict}")
    if verdict == "VINDICATED_HARDENED":
        print("   → inject_qfair reaches A under strict scoring WITH distractors, and the")
        print("     latent>text gap survives strict. Proof 4 fully vindicated; ship to Proof 5.")
    elif verdict == "SCALES_NOT_PARITY":
        print("   → inject stays well above the floor but below A under strict. Honest and")
        print("     strong — reframe the claim from 'matches prefill' to 'recovers most of")
        print("     prefill at a fraction of the cost.'")
    elif verdict == "EASY_TASK_ARTIFACT":
        print("   → inject collapses toward C once decoys compete. The 32k 1.00 was an")
        print("     easy-task artifact; distractor filler must become standard for Proof 4.")
    elif verdict == "MECHANISM_SCORER_INFLATED":
        print("   → dec_latent no longer beats dec_text under strict+distractors. The 3.1")
        print("     mechanism was scorer-inflated — the one outcome that threatens the")
        print("     project. Investigate before any further build.")
    else:
        print("   → read the table; the automatic verdict did not fire cleanly.")
    return verdict


def _fmt(v):
    return "·" if v is None else f"{v:+.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="zorvian_codex")
    parser.add_argument("--length", type=int, default=32000)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--keep-rate", type=float, default=0.5)
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

    if not selftest_filler():
        print("!!! filler not inert — fix proofs/long_context_docs.py first.")
        sys.exit(1)
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
          f"({args.length} tok / L{args.layer} / depth {args.depth}) ##########")
    result = run(model, tok, args)
    agg = aggregate(result)
    verdict = report(result, agg)
    result["aggregate"] = agg
    result["verdict"] = verdict

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

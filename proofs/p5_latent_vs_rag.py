"""
Proof 5 — Latent handoff vs. text-RAG (HotpotQA distractor, reasoning path).

The claim under test: injecting a document's latent KV beats handing the reasoner
retrieved text, on questions that require integrating information retrieval would
fragment — at EQUAL information budget. If true, the latent pipeline is justified over
plain RAG. If latent ≈ RAG, the project is an expensive reimplementation of retrieval and
Proof 6 isn't worth building.

Operating point (pinned from 4.1, non-negotiable): layer 12, strict scoring, q-fair
capture, think-ON (multi-hop is a reasoning claim, so reasoning is enabled).

Stages (each cacheable / resumable; gating is the expensive part):

  gate — every candidate run closed-book (C, question only) and full-document (A). Discard
         C-successes (memorized — they test nothing) and A-failures (unanswerable for this
         model). Gated set = C fails AND A succeeds; every Proof-5 number is on it.
  rag  — build a real BGE-large retriever per gated doc, tune chunk size on a held-out
         slice (tune the baseline to win), k-sweep, log retrieval recall.
  eval — the 2×2 of {gold spans, retrieved spans} × {latent, text} plus the two ceilings:
           A              full document as text                       (ceiling)
           C              question only                               (floor, =0 by constr.)
           text_gold      gold supporting sentences as text           (oracle retrieval)
           text_rag@k     top-k retrieved chunks as text              (the real baseline)
           latent_all     full document KV injected (q-fair capture)  (whole-read handoff)
           latent_sparse  only the gold-sentence positions injected   (budget-matched H2H)

The numbers that decide the project:
  (1) latent_sparse − text_gold (strict)            — representation vs text at equal oracle
      budget; Exp 3.1 predicts positive. ≈0 ⇒ representation adds nothing at multi-hop.
  (2) latent_sparse − text_rag@best (strict, raw)   — the deployment comparison, reported
      raw AND conditioned on retrieval-success (the fair reasoning-vs-reasoning subset).
  (3) latent_all − A (strict)                       — whole-document handoff cost (≈0 hoped).
  (4) synthetic agreement                           — does (1) replicate with zero memory.
  (5) single-hop parity                             — latent and text MUST tie on extraction;
      a gap there means the prompt framing flatters latent (the docnaive bug), and the
      multi-hop numbers aren't trustworthy until fixed.

Usage (pick physical GPUs with CUDA_VISIBLE_DEVICES, like proofs 0–3 / run_chain; eval
is checkpointed to --out after every item and resumes from a partial --out):
    CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p5_latent_vs_rag.py --arm hotpot \
        --max-candidates 400 --max-eval 60 --think-max-new-tokens 1024 --out proofs/data/p5.json
    # the synthetic control + the parity control:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p5_latent_vs_rag.py --arm synth_multihop --out proofs/data/p5_synth.json
    CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p5_latent_vs_rag.py --arm synth_parity   --out proofs/data/p5_parity.json
    # wire-test the whole pipeline cheaply before committing GPU hours:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p5_latent_vs_rag.py --arm hotpot --max-candidates 30 --no-think
    # re-score a saved run with the current scorers (no GPU):
    python proofs/p5_latent_vs_rag.py --rescore proofs/data/p5.json
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
from proofs.common import (
    load_deepseek, no_context_answer, full_prefill_answer, inject_answer,
    inject_answer_subset, final_answer, _with_think_control, normalize, generate_plain,
)
from core.split_forward import capture_doc_cache, split_forward_generate
from proofs.needles import token_positions_for_char_span, needle_positions
from proofs.decimate import kept_indices
from proofs.retriever import (
    make_backend, Retriever, retrieval_recall, tune_chunk_size, budget_matched_k,
)
from proofs import hotpot, synth_multihop
from proofs.hotpot import YESNO

# Parity-claim floor (the n=3 guard inherited from 4.1's P41_MIN_N): below this a
# "tie"/"parity" verdict over-reads noise. GAP is the strict-accuracy margin that counts
# as a real latent advantage (mirrors Exp 3.1 / Proof 4.1).
MIN_N = 30
GAP = 0.10

# q-fair capture split (identical to p4_1): capture the document inside the SAME framing A
# prefills, so the only residual asymmetry vs A is the (causally-later) question.
_PFX, _QPART = _common.PREFILL_PROMPT.split("\n\nQuestion: ")
PREFILL_PREFIX = _PFX
PREFILL_QSUFFIX = "\n\nQuestion: " + _QPART
assert PREFILL_PREFIX + PREFILL_QSUFFIX == _common.PREFILL_PROMPT


# ════════════════════════════════════════════════════════════════════════════════
# Scoring — word-boundary containment (HotpotQA answers are short spans; substring
# matching marks a refusal "I don't know" correct for gold "no" because "no" ⊂ "know",
# which would corrupt the C-fails gate). Strict adds the 4.1 negation/decoy guards. EM/F1
# are HotpotQA's canonical metrics, reported alongside for external credibility.
# ════════════════════════════════════════════════════════════════════════════════
_NEGATIONS = ["not ", "n't", "rather than", "instead of", "do not know", "dont know",
              "don t know", "unknown", "unclear", "unsure", "cannot", "no information",
              "not present", "not mentioned", "not stated"]


def first_clause(answer: str) -> str:
    line = next((l for l in answer.splitlines() if l.strip()), "")
    parts = re.split(r"(?<=[.!?])\s", line.strip())
    return parts[0] if parts else line.strip()


def _wb(text_norm: str, gold: str) -> bool:
    """Word-boundary containment of normalized `gold` in already-normalized `text_norm`."""
    g = normalize(gold)
    if not g:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", text_norm) is not None


def _golds(gold, alts):
    return [gold] + list(alts or [])


def score_lenient(answer, gold, alts=()):
    a = normalize(answer)
    return any(_wb(a, g) for g in _golds(gold, alts))


def score_firstline(answer, gold, alts=()):
    c = normalize(first_clause(answer))
    return any(_wb(c, g) for g in _golds(gold, alts))


def score_strict(answer, gold, decoys=(), alts=()):
    c = normalize(first_clause(answer))
    if not any(_wb(c, g) for g in _golds(gold, alts)):
        return False
    if any(_wb(c, d) for d in decoys):
        return False
    if any(neg in c for neg in _NEGATIONS):
        return False
    return True


# HotpotQA SQuAD-style normalization (drop articles/punct) for EM/F1.
def _squad_norm(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def score_em(answer, gold, alts=()):
    c = _squad_norm(first_clause(answer))
    return any(_squad_norm(g) == c for g in _golds(gold, alts))


def score_f1(answer, gold, alts=()):
    pred = _squad_norm(first_clause(answer)).split()
    best = 0.0
    for g in _golds(gold, alts):
        gt = _squad_norm(g).split()
        if not pred or not gt:
            best = max(best, float(pred == gt))
            continue
        common = {}
        for t in pred:
            if t in gt:
                common[t] = min(pred.count(t), gt.count(t))
        n = sum(common.values())
        if n == 0:
            continue
        p, r = n / len(pred), n / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return round(best, 3)


# ── final-stance scoring (the think-ON fix) ──────────────────────────────────────
# think-ON R1 reasons aloud IN the answer, self-corrects, and oscillates: "I do not know.
# Wait, no, the document says … So the answer should be X. Wait, no …". first_clause grabs
# "I do not know" and fails a correct answer; a fixed last-clause / answer-marker heuristic
# is just as brittle the other way. So instead of guessing a position, we read the model's
# FINAL STANCE: walk the sentences from the END and take the first that AFFIRMS a candidate
# (gold or a near-miss decoy) — affirm = contains it as a word in a sentence that is not
# itself negated/hedged. This is position-independent and marker-free; it captures what the
# model CONCLUDED. A truncated/oscillating answer that never settles → no stance → wrong,
# which is the honest verdict (the fix for those is more think budget, not a cleverer regex).
_TENTATIVE = ["maybe", "perhaps", "might be", "possibly", "not sure", "i guess",
              "could be", "i think it"]


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip()) if s.strip()]


def _yesno_stance(answer, gold):
    """yes/no answers naturally contain 'not' ('they are not the same, so the answer is
    no'), so the generic negation-skip would wrongly drop a correct 'no'. Instead read the
    last explicit yes/no determination — an 'answer is/should be yes|no' assertion, or a
    standalone 'Yes.'/'No.' — and compare to gold."""
    g = normalize(gold)
    for s in reversed(_sentences(answer)):
        ns = normalize(s)
        m = re.search(r"answer (?:is|should be|would be|here is|:)\s+(yes|no)\b", ns)
        if not m and ns in ("yes", "no"):
            m = re.match(r"(yes|no)$", ns)
        if m:
            return ("gold" if m.group(1) == g else "decoy"), s
    return None, ""


def final_stance(answer, gold, decoys=(), alts=()):
    """Return ('gold'|'decoy'|None, deciding_sentence) — the model's last affirmed stance."""
    if normalize(gold) in ("yes", "no"):
        return _yesno_stance(answer, gold)
    golds = [normalize(g) for g in _golds(gold, alts)]
    decs = [normalize(d) for d in decoys]
    for s in reversed(_sentences(answer)):
        ns = normalize(s)
        # a sentence that is negated, hedged, tentative, or a question is not a commitment
        if (any(neg in ns for neg in _NEGATIONS) or any(t in ns for t in _TENTATIVE)
                or s.strip().endswith("?")):
            continue
        has_gold = any(_wb(ns, g) for g in golds)
        has_dec = any(_wb(ns, d) for d in decs)
        if has_gold and not has_dec:
            return "gold", s
        if has_dec and not has_gold:
            return "decoy", s              # final stance is a decoy → wrong
    return None, ""


def score_committed(answer, gold, decoys=(), alts=()):
    """True iff the model's FINAL STANCE is the gold answer (not a decoy, not unsettled)."""
    return final_stance(answer, gold, decoys, alts)[0] == "gold"


def committed_clause(answer, gold="", decoys=(), alts=()):
    """The deciding sentence (for --show display); the last sentence if nothing decided."""
    _, sent = final_stance(answer, gold, decoys, alts)
    if sent:
        return sent
    sents = _sentences(answer)
    return sents[-1] if sents else ""


SCORERS = ["lenient", "firstline", "strict", "committed", "em"]


def score_all(answer, gold, decoys=(), alts=()):
    return {"lenient": score_lenient(answer, gold, alts),
            "firstline": score_firstline(answer, gold, alts),
            "strict": score_strict(answer, gold, decoys, alts),
            "committed": score_committed(answer, gold, decoys, alts),
            "em": score_em(answer, gold, alts),
            "f1": score_f1(answer, gold, alts)}


def _set_think(on: bool):
    _common.SUPPRESS_THINK = (not on)


# ── LLM-judge (the authoritative, non-heuristic scorer) ──────────────────────────
# The deterministic scorers above are transparent and validated, but any string rule has
# blind spots on free-form reasoning. The field-standard cross-check is to let the model
# itself judge whether a response's FINAL answer matches the reference — robust to
# phrasing, self-correction, oscillation and yes/no. Run on SAVED answers (no re-generation
# of the expensive main eval), so it is cheap to apply to a finished run.
JUDGE_PROMPT = (
    "You are grading a model's answer to a question. The response often shows its reasoning "
    "and may correct itself or say \"I do not know\" before committing — judge ONLY whether "
    "its FINAL answer is correct.\n\n"
    "Question: {q}\nReference (correct) answer: {gold}\n\n"
    "Model response:\n{ans}\n\n"
    "Does the model's final answer match the reference (same entity / number / yes-no), "
    "ignoring wording? Reply with exactly one word: CORRECT or INCORRECT."
)


def judge_answer(model, tok, q, gold, ans, max_new_tokens=8):
    """One short think-off verdict. Returns True/False (defaults False on an unparseable
    reply, so the judge never silently inflates)."""
    prev = _common.SUPPRESS_THINK
    _common.SUPPRESS_THINK = True
    try:
        out = generate_plain(model, tok,
                             JUDGE_PROMPT.format(q=q, gold=gold, ans=(ans or "")[:2000]),
                             max_new_tokens)
    finally:
        _common.SUPPRESS_THINK = prev
    u = out.strip().upper()
    if "INCORRECT" in u:
        return False
    return "CORRECT" in u


# ════════════════════════════════════════════════════════════════════════════════
# Condition helpers
# ════════════════════════════════════════════════════════════════════════════════
def ans_qfair(model, tok, qcache, n_pre, q, layer, m):
    query = _with_think_control(PREFILL_QSUFFIX.format(question=q))
    txt = split_forward_generate(model, tok, qcache, n_pre, query_text=query,
                                 target_layer=layer, max_new_tokens=m)
    return final_answer(txt)


def gold_text_of(rec):
    """The gold supporting sentences in document order, handed over as oracle-retrieved
    text (the cleanest text baseline: same exact spans as latent_sparse, as tokens)."""
    gs = sorted(rec["gold_sentences"], key=lambda g: g["char_start"])
    return " ".join(g["text"].strip() for g in gs)


def rag_text_of(chunks):
    """Retrieved chunks concatenated in document order (so the model reads them in the
    order they'd appear in the source, not in score order)."""
    ordered = sorted(chunks, key=lambda c: c["char_start"])
    return "\n".join(c["text"].strip() for c in ordered)


def needle_idx_of(tok, rec, max_doc_tokens):
    """Union of token positions covering every gold supporting sentence — `needle_idx`
    for latent_sparse, located by recorded char range (recurrence-safe)."""
    idx = set()
    for g in rec["gold_sentences"]:
        idx |= set(token_positions_for_char_span(tok, rec["doc_text"], g["char_start"],
                                                 g["char_end"], max_doc_tokens))
    return sorted(idx)


def _free_cuda():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════════
# Stage: gate (C fails AND A succeeds)
# ════════════════════════════════════════════════════════════════════════════════
def run_gate(model, tok, recs, args):
    _set_think(args.think)
    m = args.think_max_new_tokens if args.think else args.max_new_tokens
    c_cache, gated = {}, []
    n_c_pass = n_a_fail = 0
    for i, rec in enumerate(recs):
        q, gold = rec["question"], rec["answer"]
        decoys = rec.get("decoy_values", [])
        c = c_cache.get(q)
        if c is None:
            c = no_context_answer(model, tok, q, m)
            c_cache[q] = c
        a = full_prefill_answer(model, tok, rec["doc_text"], q, m,
                                max_length=args.max_doc_tokens)
        c_ok = score_lenient(c, gold)
        a_ok = score_lenient(a, gold)
        n_c_pass += int(c_ok)
        n_a_fail += int(not a_ok)
        item = {**{k: rec[k] for k in ("id", "question", "answer", "type",
                                       "doc_text", "gold_sentences")},
                "decoy_values": decoys,
                "c_ans": c, "a_ans": a,
                "c_ok": c_ok, "a_ok": a_ok,
                "a_scores": score_all(a, gold, decoys),
                "gated": (not c_ok) and a_ok}
        if item["gated"]:
            gated.append(item)
        print(f"  gate [{i+1}/{len(recs)}] C={'pass' if c_ok else 'fail'} "
              f"A={'pass' if a_ok else 'fail'} → "
              f"{'GATED' if item['gated'] else 'drop'}  ({len(gated)} kept)", end="\r")
    print()
    n = len(recs)
    summary = {"candidates": n, "discard_memorized": n_c_pass,
               "discard_rate": round(n_c_pass / n, 3) if n else None,
               "a_pass_rate": round((n - n_a_fail) / n, 3) if n else None,
               "gated": len(gated)}
    print(f"  gate: {len(gated)} gated / {n} candidates "
          f"(closed-book discard {summary['discard_rate']}, A pass {summary['a_pass_rate']})")
    return gated, summary


# ════════════════════════════════════════════════════════════════════════════════
# Stage: eval (the 2×2 grid + ceilings, on the gated set)
# ════════════════════════════════════════════════════════════════════════════════
def run_eval(model, tok, gated, args, backend, resume_path=None):
    layer = args.layer
    m = args.think_max_new_tokens if args.think else args.max_new_tokens
    ks = sorted(set(args.rag_ks))
    # rag conditions are STRING-labeled so the per-item, budget-matched k (a "generous k
    # that roughly matches the gold-fact token budget" — the spec's information-matched
    # point) sits in the sweep beside the fixed ks without making the columns ragged.
    rag_labels = [str(k) for k in ks] + ["budget"]

    # ── resume: each item is checkpointed to `resume_path`, so a killed run loses at most
    # the in-flight item, not the whole eval (this run already cost a day; never again). ──
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

    # tune the baseline once (reuse across resumes): chunk size maximizing recall on a slice
    if prev.get("best_chunk"):
        best_chunk, chunk_scores = prev["best_chunk"], prev.get("chunk_scores", {})
        print(f"  chunk-size reuse: best={best_chunk}")
    else:
        tune_pool = gated[:args.tune_n] if len(gated) > args.tune_n else gated
        best_chunk, chunk_scores = tune_chunk_size(
            tune_pool, backend, candidate_sizes=tuple(args.chunk_sizes), k=args.tune_k)
        print(f"  chunk-size tune: best={best_chunk}  scores={chunk_scores}")

    canary_mismatch = prev.get("canary_mismatch", 0)
    canary_checked = bool(records)       # a prior session already ran the canary
    dropped = prev.get("dropped_no_needle", 0)

    def snapshot():
        return {"layer": layer, "best_chunk": best_chunk, "chunk_scores": chunk_scores,
                "ks": ks, "rag_labels": rag_labels, "canary_mismatch": canary_mismatch,
                "dropped_no_needle": dropped, "records": records}

    cap = args.max_eval or len(gated)
    todo = [g for g in gated if g["id"] not in done_ids][:max(0, cap - len(records))]
    n_gens = (4 if args.with_docnaive else 3) + len(rag_labels)
    print(f"  eval plan: {len(records)} done, {len(todo)} to run this session "
          f"(cap {cap}, gated {len(gated)}); ~{n_gens} gens/item, think_max={m}")

    for rec in todo:
        doc, q, gold = rec["doc_text"], rec["question"], rec["answer"]
        decoys = rec.get("decoy_values", [])
        needle_idx = needle_idx_of(tok, rec, args.max_doc_tokens)
        if not needle_idx:
            dropped += 1
            continue

        ids = tok(doc, return_tensors="pt", truncation=True,
                  max_length=args.max_doc_tokens).input_ids
        cache, _Y, n_doc = capture_doc_cache(model, ids, layer); del _Y
        pre_ids = tok(PREFILL_PREFIX.format(document=doc), return_tensors="pt",
                      truncation=True, max_length=args.max_doc_tokens).input_ids
        qcache, _Yq, n_pre = capture_doc_cache(model, pre_ids, layer); del _Yq

        # one-time bookkeeping canary: injecting ALL positions == full inject
        if not canary_checked:
            _set_think(False)
            kept_all = kept_indices(n_doc, needle_idx, 1.0, "strided",
                                    "needle_decimated", seed=0, keep_sink=True)
            a_sub = inject_answer_subset(model, tok, cache, n_doc, kept_all, q, layer,
                                         args.max_new_tokens)
            a_full = inject_answer(model, tok, cache, n_doc, q, layer, args.max_new_tokens)
            canary_mismatch += int(a_sub.strip() != a_full.strip())
            canary_checked = True
            print(f"  canary subset-all==full: "
                  f"{'OK' if a_sub.strip() == a_full.strip() else 'MISMATCH'}")

        # retriever (tuned chunk size) + k-sweep, plus the budget-matched k
        retr = Retriever(backend, best_chunk).index(doc)
        bk = max(1, min(budget_matched_k(rec["gold_sentences"], best_chunk),
                        len(retr.chunks)))
        sweep = [(str(k), k) for k in ks] + [("budget", bk)]
        rag = {}
        for label, k in sweep:
            got = retr.retrieve(q, k)
            rag[label] = {"recall": retrieval_recall(got, rec["gold_sentences"]),
                          "text": rag_text_of(got), "k": k}

        _set_think(args.think)
        # docnaive is a diagnostic (q-fair is the robust capture per 4.1) — off by default
        # so it doesn't cost a generation per item.
        answers = {"A": rec["a_ans"]}     # reuse the gate's full-document answer
        if args.with_docnaive:
            answers["latent_all_docnaive"] = inject_answer(model, tok, cache, n_doc, q, layer, m)
        answers["latent_all_qfair"] = ans_qfair(model, tok, qcache, n_pre, q, layer, m)
        answers["latent_sparse"] = inject_answer_subset(
            model, tok, cache, n_doc, needle_positions(needle_idx, keep_sink=True),
            q, layer, m)
        answers["text_gold"] = full_prefill_answer(model, tok, gold_text_of(rec), q, m,
                                                   max_length=args.max_doc_tokens)
        for label, _k in sweep:
            answers[f"text_rag@{label}"] = full_prefill_answer(
                model, tok, rag[label]["text"], q, m, max_length=args.max_doc_tokens)

        records.append({
            "id": rec["id"], "question": q, "gold": gold, "type": rec.get("type", ""),
            "decoy_values": decoys, "k_needle": len(needle_idx),
            "recall_by_k": {label: rag[label]["recall"] for label, _k in sweep},
            "rag_k_used": {label: rag[label]["k"] for label, _k in sweep},
            "answers": answers,
            "scores": {c: score_all(a, gold, decoys) for c, a in answers.items()},
        })
        if resume_path:                  # checkpoint after every item
            with open(resume_path, "w") as f:
                json.dump(snapshot(), f, default=str)
        print(f"  eval [{len(records)} done / {cap}] {rec['id']}: "
              f"sparse={records[-1]['scores']['latent_sparse']['strict']} "
              f"gold={records[-1]['scores']['text_gold']['strict']}", end="\r")
        del cache, qcache
        _free_cuda()
    print()
    return snapshot()


# ════════════════════════════════════════════════════════════════════════════════
# Aggregation, failure modes, headline, verdict
# ════════════════════════════════════════════════════════════════════════════════
CONDS = ["A", "latent_all_docnaive", "latent_all_qfair", "latent_sparse", "text_gold"]


def _rate(records, cond, scorer):
    vals = [r["scores"][cond][scorer] for r in records
            if scorer in r.get("scores", {}).get(cond, {})]
    return round(sum(vals) / len(vals), 3) if vals else None


def _diff(a, b):
    return None if a is None or b is None else round(a - b, 3)


def _best_rag_k(records, labels, scorer="strict"):
    rates = {lab: _rate(records, f"text_rag@{lab}", scorer) for lab in labels}
    rates = {lab: v for lab, v in rates.items() if v is not None}
    if not rates:
        return None, {}
    return max(rates, key=rates.get), rates


def _retrieved_subset(records, label):
    """Items where the top-`label` retriever fetched ALL gold supporting sentences — the
    fair reasoning-vs-reasoning subset for the deployment comparison."""
    return [r for r in records if r["recall_by_k"].get(label, {}).get("full")]


# Primary scorer for the headline / verdict. `committed` is the think-ON-fair metric (the
# model's final assertion, not its first clause); `strict` is kept in the table for
# reference so the reasoning-aloud artifact is visible, not hidden.
HEADLINE_SCORER = "committed"


def failure_modes(records, best_k, scorer=HEADLINE_SCORER):
    """Bucket each method's failures (the carried risk from 4.1), judged on the COMMITTED
    clause so a reasoning-aloud answer that reaches the right conclusion isn't miscounted
    as a 'blank'."""
    latent, textr = {"blank": 0, "distractor_grab": 0, "hallucinate": 0, "n_fail": 0}, \
                    {"retrieval_miss": 0, "reasoning_fail": 0, "n_fail": 0}
    for r in records:
        # latent_sparse failure taxonomy via final stance
        if not r["scores"]["latent_sparse"][scorer]:
            latent["n_fail"] += 1
            ans, gold = r["answers"]["latent_sparse"], r["gold"]
            decoys, alts = r.get("decoy_values", []), r.get("alts", [])
            stance, _ = final_stance(ans, gold, decoys, alts)
            if stance == "decoy":
                latent["distractor_grab"] += 1            # committed to a near-miss
            elif score_lenient(ans, gold, alts):
                latent["blank"] += 1                      # had the gold but never settled (oscillation/truncation)
            else:
                latent["hallucinate"] += 1                # gold never appears
        # text_rag@best failure taxonomy
        if best_k is not None and not r["scores"][f"text_rag@{best_k}"][scorer]:
            textr["n_fail"] += 1
            if not r["recall_by_k"].get(str(best_k), {}).get("full"):
                textr["retrieval_miss"] += 1
            else:
                textr["reasoning_fail"] += 1
    return {"latent_sparse": latent, "text_rag_best": textr}


def _present_scorers(records):
    """Base scorers plus 'judge' when an LLM-judge pass has added it — so the table and
    the headline pick it up automatically without hard-coding the column set."""
    extra = []
    for r in records:
        for sd in r.get("scores", {}).values():
            if "judge" in sd:
                extra = ["judge"]
            break
        if extra:
            break
    return SCORERS + extra


def aggregate(result):
    records = result["records"]
    labels = result.get("rag_labels") or [str(k) for k in result.get("ks", [])]
    best_k, rag_rates = _best_rag_k(records, labels, HEADLINE_SCORER)
    scols = _present_scorers(records) + ["f1"]
    table = {c: {s: _rate(records, c, s) for s in scols} for c in CONDS}
    for lab in labels:
        table[f"text_rag@{lab}"] = {s: _rate(records, f"text_rag@{lab}", s) for s in scols}

    retr_sub = _retrieved_subset(records, best_k) if best_k is not None else []

    def _headline(scorer):
        sp = _rate(records, "latent_sparse", scorer)
        tg = _rate(records, "text_gold", scorer)
        la = _rate(records, "latent_all_qfair", scorer)
        a = _rate(records, "A", scorer)
        rb = _rate(records, f"text_rag@{best_k}", scorer) if best_k is not None else None
        sp_sub = _rate(retr_sub, "latent_sparse", scorer)
        rb_sub = _rate(retr_sub, f"text_rag@{best_k}", scorer) if best_k is not None else None
        return {
            "repr__sparse_minus_textgold": _diff(sp, tg),                     # (1)
            "deploy_raw__sparse_minus_ragbest": _diff(sp, rb),                # (2) raw
            "deploy_retrieved__sparse_minus_ragbest": _diff(sp_sub, rb_sub),  # (2) fair subset
            "handoff__latentall_minus_A": _diff(la, a),                       # (3)
        }

    headline = _headline(HEADLINE_SCORER)
    headline["scorer"] = HEADLINE_SCORER
    headline["strict_reference"] = _headline("strict")
    headline["best_rag_k"] = best_k
    headline["rag_strict_by_k"] = rag_rates
    headline["n_retrieved_subset"] = len(retr_sub)
    return {
        "n": len(records),
        "distinct": len({r["id"] for r in records}),
        "table": table, "headline": headline,
        "failure_modes": failure_modes(records, best_k, HEADLINE_SCORER),
    }


def verdict(agg, arm):
    h = agg["headline"]
    n = agg["n"]
    repr_gap = h["repr__sparse_minus_textgold"]
    deploy_gap = h["deploy_raw__sparse_minus_ragbest"]

    if n < MIN_N:
        return {"status": "UNDERPOWERED",
                "detail": f"n={n} < {MIN_N}; add candidates before reading the gaps."}

    # parity arm: latent and text must TIE on single-hop extraction
    if arm == "synth_parity":
        if repr_gap is None:
            status = "MISSING"
        elif abs(repr_gap) <= GAP:
            status = "PARITY_OK"
        else:
            status = "PROMPT_ASYMMETRY"     # framing flatters one side — invalidates multihop
        return {"status": status,
                "detail": f"latent_sparse − text_gold = {repr_gap} (|·| ≤ {GAP} ⇒ tie)"}

    # multi-hop arms (hotpot / synth_multihop)
    repr_pos = repr_gap is not None and repr_gap >= GAP
    repr_zero = repr_gap is not None and abs(repr_gap) < GAP
    deploy_pos = deploy_gap is not None and deploy_gap >= GAP

    if repr_pos and deploy_pos:
        status = "VINDICATED"               # build Proof 6
    elif repr_zero:
        status = "LATENT_EQUALS_RETRIEVAL"  # honest stop
    elif deploy_pos and repr_zero:
        status = "RETRIEVAL_HARDNESS_ONLY"  # win is retrieval difficulty, not representation
    elif repr_pos and not deploy_pos:
        status = "REPR_WINS_NOT_DEPLOYABLE"  # interesting, retrieval erases the margin
    else:
        status = "MIXED"
    return {"status": status,
            "detail": f"(1) repr {repr_gap}  (2) deploy_raw {deploy_gap}  "
                      f"(2-fair) {h['deploy_retrieved__sparse_minus_ragbest']}  "
                      f"(3) handoff {h['handoff__latentall_minus_A']}"}


# ════════════════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════════════════
def _fmt(v):
    return "·" if v is None else f"{v:+.3f}"


def report(result, agg, gate_summary, arm):
    print("\n" + "=" * 80)
    print(f"PROOF 5 — latent vs text-RAG  [arm={arm}]  (L{result['layer']}, think-on, "
          f"headline={HEADLINE_SCORER})")
    if gate_summary:
        print(f"  gate: {gate_summary['gated']} gated / {gate_summary['candidates']} "
              f"candidates  (closed-book discard {gate_summary['discard_rate']}, "
              f"A pass {gate_summary['a_pass_rate']})")
    print(f"  eval n={agg['n']} (distinct {agg['distinct']}); "
          f"chunk={result['best_chunk']} {result['chunk_scores']}; "
          f"canary mismatches={result['canary_mismatch']} (must be 0); "
          f"dropped(no-needle)={result['dropped_no_needle']}")

    labels = result.get("rag_labels") or [str(k) for k in result.get("ks", [])]
    cols = _present_scorers(result["records"]) + ["f1"]
    head = "  " + f"{'condition':<22}" + "".join(f"{c:>10}" for c in cols)
    print("\n" + head)
    print("  " + "-" * (len(head) - 2))
    for cond in CONDS + [f"text_rag@{lab}" for lab in labels]:
        row = f"  {cond:<22}"
        for c in cols:
            v = agg["table"].get(cond, {}).get(c)
            row += (f"{v:>10.2f}" if v is not None else f"{'·':>10}")
        print(row)

    h = agg["headline"]
    hs = h.get("strict_reference", {})
    print(f"\n  headline numbers ({h.get('scorer', 'committed')}; strict in parens):")
    print(f"    (1) latent_sparse − text_gold              : "
          f"{_fmt(h['repr__sparse_minus_textgold'])} ({_fmt(hs.get('repr__sparse_minus_textgold'))})"
          "   ← representation vs text, equal oracle budget")
    print(f"    (2) latent_sparse − text_rag@{h['best_rag_k']} (raw)      : "
          f"{_fmt(h['deploy_raw__sparse_minus_ragbest'])} ({_fmt(hs.get('deploy_raw__sparse_minus_ragbest'))})"
          "   ← deployment comparison")
    print(f"    (2) … conditioned on retrieval success     : "
          f"{_fmt(h['deploy_retrieved__sparse_minus_ragbest'])} "
          f"({_fmt(hs.get('deploy_retrieved__sparse_minus_ragbest'))})   "
          f"(fair reasoning-vs-reasoning, n={h['n_retrieved_subset']})")
    print(f"    (3) latent_all − A                         : "
          f"{_fmt(h['handoff__latentall_minus_A'])} ({_fmt(hs.get('handoff__latentall_minus_A'))})"
          "   ← whole-document handoff cost (≈0 hoped)")
    print(f"    best RAG k = {h['best_rag_k']}  strict-by-k = {h['rag_strict_by_k']}")

    fm = agg["failure_modes"]
    print("\n  failure modes:")
    ls = fm["latent_sparse"]
    print(f"    latent_sparse ({ls['n_fail']} fails): blank {ls['blank']}, "
          f"distractor-grab {ls['distractor_grab']}, hallucinate {ls['hallucinate']}")
    tr = fm["text_rag_best"]
    print(f"    text_rag@best ({tr['n_fail']} fails): retrieval-miss {tr['retrieval_miss']} "
          f"(not a reasoning loss), reasoning-fail {tr['reasoning_fail']} (the fair loss)")

    # judge sanity: if the judge scores the A ceiling far below its strict rate, the judge
    # pass is broken (e.g. empty generations defaulting to INCORRECT) — its numbers, and any
    # judge-based verdict, are not trustworthy for this run.
    if HEADLINE_SCORER == "judge":
        a_j, a_s = agg["table"]["A"].get("judge"), agg["table"]["A"].get("strict")
        if a_j is not None and a_s is not None and a_s >= 0.8 and a_j < 0.5:
            print(f"\n  !!! JUDGE UNRELIABLE: A scores strict={a_s} but judge={a_j} "
                  "(the judge returned no parseable verdict — likely a broken generation "
                  "env). Ignore the judge column and verdict for this run; re-judge.")

    v = verdict(agg, arm)
    print("\n  " + "-" * 76)
    print(f"  VERDICT [{arm}]: {v['status']}")
    print(f"    {v['detail']}")
    _verdict_gloss(v["status"])
    return v


def _verdict_gloss(status):
    g = {
        "VINDICATED": "   → latent beats text at equal budget AND beats real RAG; "
                      "representation is the edge. Build Proof 6.",
        "LATENT_EQUALS_RETRIEVAL": "   → latent doesn't beat text of the same spans; "
                                   "honest stop — 'transferable but equivalent to retrieval'.",
        "RETRIEVAL_HARDNESS_ONLY": "   → latent only 'wins' because retrieval is hard, not "
                                   "because the representation is better. Redirect to retrieval.",
        "REPR_WINS_NOT_DEPLOYABLE": "   → representation helps but real retrieval erases the "
                                    "margin; scientifically interesting, not deployable. Report honestly.",
        "PARITY_OK": "   → latent and text tie on extraction; the prompt framing is fair, "
                     "the multi-hop numbers are trustworthy.",
        "PROMPT_ASYMMETRY": "   → latent ≠ text on PURE EXTRACTION: the framing flatters one "
                            "side (the docnaive bug). Fix prompt parity before trusting multihop.",
        "UNDERPOWERED": "   → too few gated items to read the gaps; add candidates.",
    }
    if status in g:
        print(g[status])


# ════════════════════════════════════════════════════════════════════════════════
# Inspector — eyeball WHY latent_sparse scores the way it does
# ════════════════════════════════════════════════════════════════════════════════
def show_disagreements(result, n, cond="latent_sparse"):
    """Print items where `cond` has the gold answer somewhere (lenient) but fails strict —
    the hedge/format cases — alongside the clean text_gold answer. This is how you tell a
    REPRESENTATION failure (no gold, hallucination) from a SCORING artifact (gold present
    but the model hedged in the first clause, or the <think> truncated to empty). The first
    is fatal to the project; the second is fixable scoring/budget."""
    recs = result.get("records", [])
    print("\n" + "=" * 80)
    print(f"INSPECT [{cond}] — gold present (lenient) but strict-fail, of {len(recs)} items")
    n_settled = n_unsettled = n_clean = 0
    shown = []
    for r in recs:
        sc = r["scores"].get(cond, {})
        ans = r["answers"].get(cond, "")
        if sc.get("lenient") and not sc.get("strict"):
            stance, sent = final_stance(ans, r["gold"], r.get("decoy_values", []),
                                        r.get("alts", []))
            if stance == "gold":
                n_settled += 1               # final stance IS gold (first_clause/strict mis-scored)
            else:
                n_unsettled += 1             # gold present but never committed (oscillation/truncation)
            if len(shown) < n:
                shown.append((r, ans, sent, stance))
        elif sc.get("strict"):
            n_clean += 1
    print(f"  first-clause strict-pass: {n_clean} | gold-present-but-strict-fail: "
          f"{n_settled + n_unsettled}  (final-stance=gold {n_settled}, "
          f"unsettled/oscillating {n_unsettled})")
    for r, ans, sent, stance in shown:
        print(f"\n  [{r['id']}] gold={r['gold']!r}  final-stance={stance}")
        print(f"    {cond} deciding   : {sent[:160]!r}")
        print(f"    {cond} full(≤300) : {ans[:300]!r}")
    if not shown:
        print("  (none — lenient and strict agree for this condition)")


def dump_condition(result, cond, n):
    """Print the first N raw answers for `cond` regardless of score, beside gold and the
    needle count — for diagnosing a condition that fails everywhere (e.g. latent_sparse =
    0.00 on parity, where --show finds nothing because lenient is 0). Shows whether the
    gold is absent (injection not transferring) vs present-but-mis-scored."""
    recs = result.get("records", [])
    print("\n" + "=" * 80)
    print(f"DUMP [{cond}] — first {n} raw answers of {len(recs)} items")
    for r in recs[:n]:
        ans = r["answers"].get(cond, "<none>")
        print(f"\n  [{r['id']}] gold={r['gold']!r}  k_needle={r.get('k_needle')}  "
              f"lenient={r['scores'].get(cond, {}).get('lenient')}")
        print(f"    {ans[:320]!r}")


# ════════════════════════════════════════════════════════════════════════════════
# Candidate loading per arm
# ════════════════════════════════════════════════════════════════════════════════
def load_candidates(arm, args):
    if arm == "hotpot":
        return hotpot.prep(max_items=args.max_candidates)
    if arm == "synth_multihop":
        return synth_multihop.build_multihop(args.synth_n)
    if arm == "synth_parity":
        return synth_multihop.build_parity(args.parity_n)
    raise SystemExit(f"unknown arm {arm!r}")


# ════════════════════════════════════════════════════════════════════════════════
# Rescore (no GPU)
# ════════════════════════════════════════════════════════════════════════════════
def rescore(result):
    for r in result["records"]:
        decoys = r.get("decoy_values", [])
        r["scores"] = {c: score_all(a, r["gold"], decoys) for c, a in r["answers"].items()}
    return result


def _preflight(args, need_gate):
    """Fail fast on missing optional deps, with the exact pip line, BEFORE loading the
    70B. `datasets` is only needed when the hotpot arm has to build its prep cache;
    `sentence-transformers` is needed for the BGE retriever in eval."""
    missing = []
    if args.rag_backend == "bge":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            missing.append("sentence-transformers")
    if need_gate and args.arm == "hotpot" and not os.path.exists(hotpot.CACHE_PATH):
        try:
            import datasets  # noqa: F401
        except ImportError:
            missing.append("datasets")
    if missing:
        pip = sys.executable.replace("/bin/python", "/bin/pip") \
            if "/bin/python" in sys.executable else "pip"
        raise SystemExit(
            f"missing dependencies for this run: {', '.join(missing)}\n"
            f"  install into THIS env:  {pip} install {' '.join(missing)}\n"
            "  (or pre-build the HotpotQA cache once on CPU: python proofs/hotpot.py)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", default=None, metavar="PATH",
                    help="re-score a saved p5 run with current scorers; no model load")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="with --rescore: print up to N latent_sparse answers where the gold "
                         "is present (lenient) but strict fails — eyeball hedge vs truncation "
                         "vs real failure")
    ap.add_argument("--judge", default=None, metavar="PATH",
                    help="LLM-judge a SAVED run: load the model, judge each saved answer's "
                         "FINAL answer vs the reference (no main re-generation), add a 'judge' "
                         "scorer and make it the headline. The authoritative cross-check.")
    ap.add_argument("--dump-cond", default=None, metavar="COND",
                    help="with --rescore: print raw answers for a condition (e.g. "
                         "latent_sparse) regardless of score — diagnose a 0.00 condition")
    ap.add_argument("--arm", default="hotpot",
                    choices=["hotpot", "synth_multihop", "synth_parity"])
    ap.add_argument("--max-candidates", type=int, default=400,
                    help="HotpotQA candidate pool to gate through (oversample for attrition)")
    ap.add_argument("--synth-n", type=int, default=40,
                    help="synthetic multihop item count (≥30 to clear the verdict's power floor)")
    ap.add_argument("--parity-n", type=int, default=32,
                    help="single-hop parity item count (≥30 to clear the power floor)")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="suppress reasoning (smoke only; Proof 5 is think-ON)")
    ap.set_defaults(think=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--think-max-new-tokens", type=int, default=2048,
                    help="reasoning budget per generation; the per-item cost driver. "
                         "Lower (e.g. 1024) to roughly halve wall-clock — 2-hop answers "
                         "rarely need 2048, and a truncated <think> already scores as "
                         "no-answer so over-truncation is visible, not silent.")
    ap.add_argument("--max-doc-tokens", type=int, default=4096)
    ap.add_argument("--max-eval", type=int, default=None,
                    help="cap gated items evaluated this run (resume-safe; default = all). "
                         "Use e.g. 60 for a first verdict overnight, then rerun to extend.")
    ap.add_argument("--with-docnaive", action="store_true",
                    help="also run latent_all_docnaive (diagnostic; off by default — q-fair "
                         "is the robust capture per 4.1 — so it doesn't cost a gen/item)")
    # default to a single tuned k (+ the budget-matched k) so eval runs ~2 RAG gens/item,
    # not 4. Pass --rag-ks 2 4 8 for the full sweep when compute allows.
    ap.add_argument("--rag-ks", type=int, nargs="+", default=[4])
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--tune-n", type=int, default=20, help="held-out slice for chunk tuning")
    ap.add_argument("--tune-k", type=int, default=4, help="k used during chunk tuning")
    ap.add_argument("--rag-device", default="cpu", help="device for the BGE retriever")
    ap.add_argument("--rag-backend", default="bge", choices=["bge", "hash"])
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="logical GPU indices for DeepSeek shards (pick physical GPUs with "
                         "CUDA_VISIBLE_DEVICES, exactly like proofs 0–3 / run_chain)")
    ap.add_argument("--gate-cache", default=None,
                    help="path to cache/reuse the gated set (default derived from --out)")
    ap.add_argument("--out", default="proofs/data/p5.json")
    args = ap.parse_args()

    if args.rescore:
        with open(args.rescore) as f:
            result = json.load(f)
        result = rescore(result)
        agg = aggregate(result)
        v = report(result, agg, result.get("gate_summary"), result.get("arm", "hotpot"))
        result["aggregate"] = agg
        result["verdict"] = v
        if args.show:
            show_disagreements(result, args.show)
        if args.dump_cond:
            dump_condition(result, args.dump_cond, args.show or 12)
        out = args.out if args.out != "proofs/data/p5.json" else args.rescore
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nRe-scored → {out}")
        return

    # ── --judge: load the model, judge SAVED answers, add a 'judge' scorer, re-report ──
    if args.judge:
        global HEADLINE_SCORER
        with open(args.judge) as f:
            result = rescore(json.load(f))          # ensure deterministic scorers present
        from config import StitcherConfig
        cfg = StitcherConfig()
        devices = tuple(int(x) for x in args.gpus.split(","))
        print(f"sharding DeepSeek-70B across GPUs {devices} for judging "
              "(select physical GPUs with CUDA_VISIBLE_DEVICES)")
        tok, model = load_deepseek(cfg, devices=devices)
        labels = result.get("rag_labels") or [str(k) for k in result.get("ks", [])]
        conds = CONDS + [f"text_rag@{lab}" for lab in labels]
        recs = result["records"]
        for i, rec in enumerate(recs):
            for cond in conds:
                ans = rec["answers"].get(cond)
                if ans is None:
                    continue
                rec["scores"].setdefault(cond, {})["judge"] = judge_answer(
                    model, tok, rec["question"], rec["gold"], ans)
            print(f"  judged [{i+1}/{len(recs)}]", end="\r")
        print()
        HEADLINE_SCORER = "judge"
        agg = aggregate(result)
        v = report(result, agg, result.get("gate_summary"), result.get("arm", "hotpot"))
        result["aggregate"] = agg
        result["verdict"] = v
        if args.show:
            show_disagreements(result, args.show)
        with open(args.judge, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nJudged → {args.judge}")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gate_cache = args.gate_cache or args.out.replace(".json", f"_gated_{args.arm}.json")
    need_gate = not os.path.exists(gate_cache)

    # ── pre-flight: surface missing deps and resolve the candidate set on CPU BEFORE the
    # 30s+ 70B load, so a missing `datasets` / `sentence-transformers` (or a HotpotQA
    # download) fails fast instead of after a wasted model load. ──
    _preflight(args, need_gate)
    recs = None
    if need_gate:
        recs = load_candidates(args.arm, args)
        print(f"[gate] resolved {len(recs)} {args.arm} candidates (pre-load)")

    # Model placement copied verbatim from proofs 0–3 / run_chain: --gpus are logical GPU
    # indices (pick physical GPUs with CUDA_VISIBLE_DEVICES), load_deepseek's default
    # device_map="sequential" — right for HotpotQA's short docs, and it never trips the
    # balanced_memory KeyError.
    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))
    print(f"sharding DeepSeek-70B across GPUs {devices} "
          "(select physical GPUs with CUDA_VISIBLE_DEVICES)")
    tok, model = load_deepseek(cfg, devices=devices)

    # ── gate (resumable) ──
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

    # ── eval (checkpointed to --out after every item; resumes from a partial --out) ──
    backend = make_backend(args.rag_backend, args.rag_device)
    print(f"[eval] retriever backend={args.rag_backend} on {args.rag_device}")
    result = run_eval(model, tok, gated, args, backend, resume_path=args.out)
    result["arm"] = args.arm
    result["gate_summary"] = gate_summary
    agg = aggregate(result)
    v = report(result, agg, gate_summary, args.arm)
    result["aggregate"] = agg
    result["verdict"] = v

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

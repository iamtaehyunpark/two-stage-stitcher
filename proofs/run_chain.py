"""
proofs/run_chain.py — run Proofs 0 → 1 → 2 in one session, one model load.

The recommended entry point. Loads DeepSeek-70B once and runs the full receiver-
validation gate. Proof 0 runs first and is a HARD GATE: if the plumbing isn't
trustworthy, Proofs 1–2 are not even attempted (their numbers would be
uninterpretable). Proofs 1 and 2 are then rendered from a single synthetic-fact
evaluation pass.

Before any of this, certify the mechanism on CPU — it costs seconds and catches
RoPE/mask/cache bugs without touching a GPU:

    python core/selftest_split_forward.py

Then, on the box with the 70B:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python proofs/run_chain.py --out-dir proofs/data
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.common import (
    load_deepseek, capture_document, full_prefill_answer, inject_answer, correct,
)
from proofs import p0_plumbing
from proofs.synthetic_eval import (
    capture_all, evaluate_synthetic, verdict_p1, verdict_p2,
)


def run_p0(model, tok, cfg, max_new_tokens):
    """Proof 0 on the built-in memorized document. Returns (summary, records)."""
    document, qa = p0_plumbing.DEFAULT_DOC, p0_plumbing.DEFAULT_QA
    print("\n########## PROOF 0 — split-forward plumbing ##########")
    doc_cache, n_doc = capture_document(model, tok, document, cfg.target_layer)
    print(f"  memorized document = {n_doc} tokens")

    records = []
    for item in qa:
        q, gold = item["q"], item["a"]
        ans_a = full_prefill_answer(model, tok, document, q, max_new_tokens)
        ans_sf = inject_answer(model, tok, doc_cache, n_doc, q, cfg.target_layer, max_new_tokens)
        a_ok, sf_ok = correct(ans_a, gold), correct(ans_sf, gold)
        degen = p0_plumbing.looks_degenerate(ans_sf)
        records.append({"question": q, "gold": gold, "answer_a": ans_a,
                        "answer_sf_true": ans_sf, "a_correct": a_ok,
                        "sf_correct": sf_ok, "sf_degenerate": degen})
        print(f"\nQ: {q}\n  gold={gold!r}")
        print(f"  A      [{'ok' if a_ok else 'XX'}]: {ans_a[:140]}")
        print(f"  SF-true[{'ok' if sf_ok else 'XX'}{' DEGEN' if degen else ''}]: {ans_sf[:140]}")

    n = len(qa)
    sf_correct = sum(r["sf_correct"] for r in records)
    sf_degen = sum(r["sf_degenerate"] for r in records)
    all_pass = sf_degen == 0 and (sf_correct / n) >= p0_plumbing.P0_PASS_RATE
    summary = {
        "n_questions": n, "doc_tokens": n_doc,
        "a_correct": sum(r["a_correct"] for r in records),
        "sf_correct": sf_correct,
        "sf_degenerate": sf_degen,
        "verdict": "PASS" if all_pass else "FAIL",
    }
    print("\n" + "=" * 60)
    print(f"PROOF 0 — SF-true correct {summary['sf_correct']}/{summary['n_questions']}, "
          f"degenerate {summary['sf_degenerate']}  →  VERDICT: {summary['verdict']}")
    return summary, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="proofs/data")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpus", default="0,1,2,3",
                        help="logical GPU indices (no SLM in Proofs 0–2 → use all 4)")
    parser.add_argument("--force", action="store_true",
                        help="run Proofs 1–2 even if Proof 0 fails (debugging only)")
    parser.add_argument("--reasoning", action="store_true",
                        help="let R1 emit <think> traces instead of suppressing them")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import proofs.common as _common
    _common.SUPPRESS_THINK = not args.reasoning

    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))
    tok, model = load_deepseek(cfg, devices=devices)

    # ── Proof 0 — hard gate ──────────────────────────────────────────────────
    s0, r0 = run_p0(model, tok, cfg, args.max_new_tokens)
    with open(os.path.join(args.out_dir, "p0.json"), "w") as f:
        json.dump({"summary": s0, "records": r0}, f, indent=2)

    if s0["verdict"] != "PASS" and not args.force:
        print("\n!!! Proof 0 FAILED — the harness is not trustworthy.")
        print("    Stopping before Proofs 1–2 (their numbers would be meaningless).")
        print("    Fix core/split_forward.py (run core/selftest_split_forward.py),")
        print("    or re-run with --force to inspect downstream behaviour anyway.")
        return

    # ── Proofs 1 & 2 — one synthetic-fact pass, two verdicts ─────────────────
    print("\n########## PROOFS 1 & 2 — premise + falsifier ##########")
    caches = capture_all(model, tok, cfg)
    records = evaluate_synthetic(model, tok, cfg, caches=caches,
                                 max_new_tokens=args.max_new_tokens, want_wrong=True)
    s1 = verdict_p1(records)
    s2 = verdict_p2(records)
    with open(os.path.join(args.out_dir, "p1p2.json"), "w") as f:
        json.dump({"proof1": s1, "proof2": s2, "records": records}, f, indent=2)

    # ── chain verdict ────────────────────────────────────────────────────────
    chain_green = (s0["verdict"] == "PASS" and s1["verdict"] == "PASS"
                   and s2["verdict"] == "PASS")
    print("\n" + "#" * 60)
    print(f"  Proof 0 (plumbing)  : {s0['verdict']}")
    print(f"  Proof 1 (premise)   : {s1['verdict']}")
    print(f"  Proof 2 (falsifier) : {s2['verdict']}")
    print("#" * 60)
    if chain_green:
        print("  RECEIVER VALIDATED — the project is alive. Translation work is licensed.")
        print("  Next: Proof 3 (all-N vs needles), then 4, 5, and only then the SLM (6).")
    else:
        print("  CHAIN RED — a cheap, early, decisive stop. Read the failing rung above.")
    print(f"\nSaved → {args.out_dir}/p0.json, {args.out_dir}/p1p2.json")


if __name__ == "__main__":
    main()

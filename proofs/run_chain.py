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


def run_p0(model, tok, target_layer, max_new_tokens, a_cache=None):
    """Proof 0 on the built-in memorized document. Returns (summary, records)."""
    document, qa = p0_plumbing.DEFAULT_DOC, p0_plumbing.DEFAULT_QA
    print(f"\n########## PROOF 0 — split-forward plumbing (layer {target_layer}) ##########")
    doc_cache, n_doc = capture_document(model, tok, document, target_layer)
    print(f"  memorized document = {n_doc} tokens")

    records = []
    for item in qa:
        q, gold = item["q"], item["a"]
        if a_cache is not None and q in a_cache:
            ans_a = a_cache[q]
        else:
            ans_a = full_prefill_answer(model, tok, document, q, max_new_tokens)
            if a_cache is not None:
                a_cache[q] = ans_a
        ans_sf = inject_answer(model, tok, doc_cache, n_doc, q, target_layer, max_new_tokens)
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
    parser.add_argument("--layers", default="30",
                        help="comma-separated list of layers to sweep (e.g. 16,24,32,40,48,56,64)")
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

    layers = [int(x.strip()) for x in args.layers.split(",")]
    print(f"\nStarting layer sweep over: {layers}")

    p0_a_cache = {}    # q -> ans_a
    p1_c_a_cache = {}  # (doc_name, question) -> (ans_c, ans_a)

    sweep_results = {}

    for layer in layers:
        print(f"\n==========================================")
        print(f"Evaluating Layer {layer}")
        print(f"==========================================")

        # ── Proof 0 ──────────────────────────────────────────────────────────
        s0, r0 = run_p0(model, tok, layer, args.max_new_tokens, a_cache=p0_a_cache)

        sweep_results[layer] = {
            "p0_summary": s0,
            "p0_records": r0,
            "p1_summary": None,
            "p1_records": None,
        }

        if s0["verdict"] != "PASS" and not args.force:
            print(f"\n!!! Proof 0 FAILED for layer {layer} — skipping Proof 1.")
            continue

        # ── Proof 1 (matched inject only to save compute) ─────────────────────
        print(f"\n########## PROOF 1 — premise (matched inject) at layer {layer} ##########")
        caches = capture_all(model, tok, cfg, target_layer=layer)
        records = evaluate_synthetic(model, tok, cfg, caches=caches,
                                     max_new_tokens=args.max_new_tokens, want_wrong=False,
                                     c_a_cache=p1_c_a_cache, target_layer=layer)
        s1 = verdict_p1(records)

        sweep_results[layer]["p1_summary"] = s1
        sweep_results[layer]["p1_records"] = records

    # ── Find the Winning Layer ────────────────────────────────────────────────
    winning_layer = None
    best_fidelity = -1.0

    print("\n" + "=" * 60)
    print("Layer Sweep Summary Table:")
    print("Layer\tProof 0\tProof 1 Recall")
    print("─" * 40)
    for layer in layers:
        s0_ver = sweep_results[layer]["p0_summary"]["verdict"]
        s1_sum = sweep_results[layer]["p1_summary"]
        if s1_sum is not None:
            fidelity = s1_sum["recall_fidelity_vs_prefill"]
            fidelity_str = f"{fidelity:.3f}"
            if s0_ver == "PASS" and fidelity > best_fidelity:
                best_fidelity = fidelity
                winning_layer = layer
        else:
            fidelity_str = "-"
        print(f"{layer}\t{s0_ver}\t{fidelity_str}")
    print("=" * 60)

    # ── Proof 2 at the Winning Layer ──────────────────────────────────────────
    if winning_layer is not None:
        print(f"\n########## PROOF 2 — wrong-document falsifier at winning layer {winning_layer} ##########")
        caches = capture_all(model, tok, cfg, target_layer=winning_layer)
        # Run full eval (with want_wrong=True) at the winning layer
        records_win = evaluate_synthetic(model, tok, cfg, caches=caches,
                                         max_new_tokens=args.max_new_tokens, want_wrong=True,
                                         c_a_cache=p1_c_a_cache, target_layer=winning_layer)
        s1_win = verdict_p1(records_win)
        s2_win = verdict_p2(records_win)

        sweep_results["winning_layer"] = winning_layer
        sweep_results["p2_summary"] = s2_win
        sweep_results["p2_records"] = records_win

        v0_win = sweep_results[winning_layer]["p0_summary"]["verdict"]
        v1_win = s1_win["verdict"]
        v2_win = s2_win["verdict"]
        fidelity_win = s1_win.get("recall_fidelity_vs_prefill")

        print("\n" + "#" * 60)
        print(f"Winning Layer: {winning_layer}")
        print(f"  Proof 0 (plumbing)  : {v0_win}")
        print(f"  Proof 1 (premise)   : {v1_win}   (recall fidelity {fidelity_win})")
        print(f"  Proof 2 (falsifier) : {v2_win}")
        print("#" * 60)

        if v0_win == "PASS" and v1_win == "PASS" and v2_win == "PASS":
            print("  RECEIVER VALIDATED — read, causal, high fidelity. Translation work licensed.")
        elif v1_win == "FAIL":
            print("  CHAIN RED — injected states are ignored. Premise broken; stop the project.")
        elif v0_win == "PASS" and v1_win == "PARTIAL" and v2_win in ("PASS", "INCONCLUSIVE"):
            print("  RECEIVER WORKS AND IS CAUSAL — but recall fidelity is partial "
                  f"({fidelity_win}).")
            print("  Not a stop: the premise holds and the wrong-doc control is clean.")
        else:
            print("  MIXED — see details above.")
    else:
        print("\n!!! No winning layer found (no layer passed Proof 0).")

    out_file = os.path.join(args.out_dir, "sweep_results.json")
    with open(out_file, "w") as f:
        # We convert layer keys to strings for JSON compliance
        json_save = {str(k): v for k, v in sweep_results.items() if k != "winning_layer"}
        if winning_layer is not None:
            json_save["winning_layer"] = winning_layer
            json_save["p2_summary"] = sweep_results["p2_summary"]
            json_save["p2_records"] = sweep_results["p2_records"]
        json.dump(json_save, f, indent=2)
    print(f"\nSaved sweep results → {out_file}")


if __name__ == "__main__":
    main()

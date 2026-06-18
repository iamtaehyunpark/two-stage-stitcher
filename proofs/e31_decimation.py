"""
Experiment 3.1 — Latent-decimation vs. Text-decimation.

Is an injected layer-12 state interchangeable with its raw token, or does it carry
context its token alone does not? Decimate each document two ways and watch how fast
recall of the planted fact degrades:

  dec_text             — keep a subset of tokens, feed as ordinary text, prefill.
  dec_latent           — keep the SAME positions, inject their true layer-12 states
                         at their ORIGINAL indices (split-forward subset).
  dec_latent_renumbered— same kept states, but RoPE-re-rotated to contiguous
                         positions (the position-isolation control).

with the references at keep-rate 1:

  full_text            — whole document as tokens (== Condition A).
  full_latent          — all positions injected (== Proof-1 all-N).

The deliverable is two curves per variant — recall vs keep-rate for dec_text and
dec_latent (plus the renumbered control). The DIVERGENCE is the result:

  dec_latent holds while dec_text collapses → latent states carry skipped-neighbour
        context; latent ≠ text. The project's mechanism is real. Proceed to Proof 5.
  both collapse together → latent is interchangeable with text under thinning; the
        latent advantage is illusory. Reconsider the premise honestly.
  dec_latent holds only when the needle span is protected, collapses when it is
        decimated → carried context lives in the needle, not the thinned
        surroundings; sparse handoff, weaker edge.
  dec_latent holds but dec_latent_renumbered collapses → it was the POSITIONS that
        mattered; decimation per se is fine, renumbering is the killer.

Run order — DO THE SANITY GATES FIRST and read them before the sweep:

    # certify the mechanism (incl. re-rotation invariants) on CPU
    python core/selftest_split_forward.py

    # sanity gates only: full refs pass, C-floor ~0, dec_latent@1 == full_latent
    CUDA_VISIBLE_DEVICES=0,1,2,3 python proofs/e31_decimation.py --layer 12 --sanity-only

    # if the canary is green, the full sweep
    CUDA_VISIBLE_DEVICES=0,1,2,3 python proofs/e31_decimation.py --layer 12 \
        --out proofs/data/e31.json --plot proofs/data/e31.png

The canary (dec_latent at keep-rate 1 == full_latent) is the one to watch: if it
fails, the decimation / position code is wrong and every swept number is noise.
Catch it there, not after a full sweep.
"""

import os
import sys
import json
import zlib
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.split_forward import capture_doc_cache
from proofs.common import (
    load_deepseek, no_context_answer, full_prefill_answer, inject_answer,
    inject_answer_subset, inject_answer_renumbered, correct,
)
from proofs.needles import span_token_positions
from proofs.decimate import kept_indices, decimated_text
from proofs.synthetic_docs_long import SYNTHETIC_DOCS_LONG

DEFAULT_KEEP_RATES = [0.5, 0.25, 0.125, 0.0625]
ALL_CONDITIONS = ["dec_text", "dec_latent", "dec_latent_renumbered"]
HOLD = 0.6        # recall at/above this "holds"
COLLAPSE = 0.3    # recall at/below this "collapses"


def _seed(doc, q, variant, keep_rate, trial):
    return zlib.crc32(f"{doc}|{q}|{variant}|{keep_rate}|{trial}".encode()) & 0xFFFFFFFF


# ── preparation: tokenize once, capture once, map needles once ─────────────────
def prepare(model, tok, target_layer, docs, max_doc_tokens=4096):
    prepared = []
    for d in docs:
        ids = tok(d["text"], return_tensors="pt", truncation=True,
                  max_length=max_doc_tokens).input_ids
        print(f"  capturing {d['name']} ({ids.shape[1]} tokens) at layer {target_layer} …")
        cache, _Y, n_doc = capture_doc_cache(model, ids, target_layer)
        qas = []
        for qa in d["qa"]:
            needle_idx = span_token_positions(tok, d["text"], qa["needle"], max_doc_tokens)
            qas.append({**qa, "needle_idx": needle_idx})
        prepared.append({"name": d["name"], "text": d["text"], "ids": ids,
                         "n_doc": n_doc, "cache": cache, "qa": qas})
    return prepared


# ── gates + references ─────────────────────────────────────────────────────────
def run_gates(model, tok, prepared, target_layer, max_new_tokens):
    items = []
    for d in prepared:
        for qa in d["qa"]:
            q, gold = qa["q"], qa["a"]
            c = no_context_answer(model, tok, q, max_new_tokens)
            a = full_prefill_answer(model, tok, d["text"], q, max_new_tokens)
            fl = inject_answer(model, tok, d["cache"], d["n_doc"], q,
                               target_layer, max_new_tokens)
            rec = {
                "doc": d["name"], "question": q, "gold": gold,
                "needle": qa["needle"], "needle_idx": qa["needle_idx"],
                "n_doc": d["n_doc"], "k_needle": len(qa["needle_idx"]),
                "full_text": a, "full_latent": fl, "c": c,
                "full_text_correct": correct(a, gold),     # == Condition A
                "full_latent_correct": correct(fl, gold),  # == all-N
                "c_correct": correct(c, gold),
            }
            rec["gated"] = (not rec["c_correct"]) and rec["full_text_correct"]
            items.append(rec)
            print(f"[{d['name']}] {q}\n   gold={gold!r}  C={rec['c_correct']} "
                  f"full_text(A)={rec['full_text_correct']} "
                  f"full_latent={rec['full_latent_correct']}  gated={rec['gated']}")
    return items


def sanity_gates(model, tok, prepared, items, target_layer, max_new_tokens,
                 keep_sink=True, verbose=True):
    """The three checks that must pass before any sweep number is trustworthy."""
    n = len(items)
    gated = [r for r in items if r["gated"]]
    full_text = sum(r["full_text_correct"] for r in items) / n if n else 0.0
    full_latent = sum(r["full_latent_correct"] for r in items) / n if n else 0.0
    c_floor = sum(r["c_correct"] for r in items) / n if n else 0.0

    # Canary: dec_latent at keep-rate 1 (keep ALL positions) must equal full_latent.
    by_doc = {d["name"]: d for d in prepared}
    canary_mismatch = []
    for r in gated:
        d = by_doc[r["doc"]]
        kept = kept_indices(r["n_doc"], r["needle_idx"], 1.0, "strided",
                            "needle_decimated", seed=0, keep_sink=keep_sink)
        dl1 = inject_answer_subset(model, tok, d["cache"], d["n_doc"], kept,
                                   r["question"], target_layer, max_new_tokens)
        if correct(dl1, r["gold"]) != r["full_latent_correct"]:
            canary_mismatch.append({"doc": r["doc"], "question": r["question"],
                                    "full_latent": r["full_latent"], "dec_latent@1": dl1})

    summary = {
        "n_items": n, "n_gated": len(gated),
        "full_text_rate": round(full_text, 3),
        "full_latent_rate": round(full_latent, 3),
        "c_floor": round(c_floor, 3),
        "canary_mismatches": len(canary_mismatch),
    }
    ok = (full_text >= HOLD and full_latent >= HOLD and c_floor <= COLLAPSE
          and len(canary_mismatch) == 0 and len(gated) >= 5)
    summary["sanity_pass"] = ok
    if verbose:
        print("\n" + "=" * 64)
        print("EXP 3.1 — SANITY GATES (read these before the sweep)")
        print(f"  gated items                : {len(gated)}/{n}")
        print(f"  full_text (A) pass rate    : {summary['full_text_rate']}  (want ≥ {HOLD})")
        print(f"  full_latent  pass rate     : {summary['full_latent_rate']}  (want ≥ {HOLD})")
        print(f"  C floor                    : {summary['c_floor']}  (want ≤ {COLLAPSE})")
        print(f"  canary dec_latent@1==full  : "
              f"{'OK' if not canary_mismatch else f'{len(canary_mismatch)} MISMATCH'}"
              f"  (must be 0)")
        if canary_mismatch:
            print("    !!! position/decimation bookkeeping is WRONG — fix before trusting any sweep:")
            for m in canary_mismatch[:5]:
                print(f"      [{m['doc']}] {m['question']}")
                print(f"        full_latent : {m['full_latent'][:80]}")
                print(f"        dec_latent@1: {m['dec_latent@1'][:80]}")
        print(f"  SANITY: {'PASS — proceed to sweep' if ok else 'FAIL — do not trust the sweep'}")
        if not ok and full_latent < HOLD:
            print("    (full_latent low at this length → harness/doc problem; fix first.)")
        if not ok and c_floor > COLLAPSE:
            print("    (C floor leaks on the longer docs → re-gate; some facts are guessable.)")
    return summary, canary_mismatch


# ── the sweep ──────────────────────────────────────────────────────────────────
def run_sweep(model, tok, prepared, items, target_layer, max_new_tokens,
              keep_rates, patterns, variants, conditions, random_trials=1,
              keep_sink=True):
    by_doc = {d["name"]: d for d in prepared}
    gated = [r for r in items if r["gated"]]
    records = []
    for r in gated:
        d = by_doc[r["doc"]]
        for variant in variants:
            for pattern in patterns:
                trials = random_trials if pattern == "random" else 1
                for kr in keep_rates:
                    cell = {"doc": r["doc"], "question": r["question"], "gold": r["gold"],
                            "variant": variant, "pattern": pattern, "keep_rate": kr}
                    # average over trials (only random varies; strided is fixed)
                    agg = {c: [] for c in conditions}
                    kept_counts = []
                    for t in range(trials):
                        kept = kept_indices(r["n_doc"], r["needle_idx"], kr, pattern,
                                            variant, seed=_seed(r["doc"], r["question"],
                                                                variant, kr, t),
                                            keep_sink=keep_sink)
                        kept_counts.append(len(kept))
                        if "dec_text" in conditions:
                            txt = decimated_text(tok, d["ids"], kept)
                            a = full_prefill_answer(model, tok, txt, r["question"], max_new_tokens)
                            agg["dec_text"].append(correct(a, r["gold"]))
                        if "dec_latent" in conditions:
                            a = inject_answer_subset(model, tok, d["cache"], d["n_doc"],
                                                     kept, r["question"], target_layer,
                                                     max_new_tokens)
                            agg["dec_latent"].append(correct(a, r["gold"]))
                        if "dec_latent_renumbered" in conditions:
                            a = inject_answer_renumbered(model, tok, d["cache"], kept,
                                                         r["question"], target_layer,
                                                         max_new_tokens)
                            agg["dec_latent_renumbered"].append(correct(a, r["gold"]))
                    cell["kept_count"] = round(sum(kept_counts) / len(kept_counts), 1)
                    for c in conditions:
                        cell[c] = sum(agg[c]) / len(agg[c]) if agg[c] else None
                    records.append(cell)
                    msg = "  ".join(f"{c}={cell[c]:.2f}" for c in conditions if cell[c] is not None)
                    print(f"[{r['doc']}|{variant}|{pattern}|keep={kr}] kept≈{cell['kept_count']}  {msg}")
    return records


# ── aggregation, table, plot ─────────────────────────────────────────────────
def aggregate(records, items, conditions, keep_rates, patterns, variants):
    """Mean recall over gated items, indexed [variant][pattern][keep_rate][condition]."""
    n_gated = sum(1 for r in items if r["gated"])
    refs = {
        "full_text": round(sum(r["full_text_correct"] for r in items if r["gated"]) / n_gated, 3)
        if n_gated else 0.0,
        "full_latent": round(sum(r["full_latent_correct"] for r in items if r["gated"]) / n_gated, 3)
        if n_gated else 0.0,
    }
    table = {}
    for variant in variants:
        table[variant] = {}
        for pattern in patterns:
            table[variant][pattern] = {}
            for kr in keep_rates:
                cells = [c for c in records if c["variant"] == variant
                         and c["pattern"] == pattern and c["keep_rate"] == kr]
                row = {}
                for cond in conditions:
                    vals = [c[cond] for c in cells if c.get(cond) is not None]
                    row[cond] = round(sum(vals) / len(vals), 3) if vals else None
                table[variant][pattern][kr] = row
    return {"n_gated": n_gated, "references": refs, "table": table}


def print_table(agg, conditions, keep_rates, patterns, variants):
    print("\n" + "=" * 72)
    print(f"EXP 3.1 — recall vs keep-rate   (n gated = {agg['n_gated']})")
    print(f"references @ keep-rate 1:  full_text(A)={agg['references']['full_text']}  "
          f"full_latent={agg['references']['full_latent']}")
    for variant in variants:
        print(f"\n── variant: {variant} " + "─" * (60 - len(variant)))
        for pattern in patterns:
            print(f"  pattern: {pattern}")
            header = "    keep   kept   " + "  ".join(f"{c:>22}" for c in conditions)
            print(header)
            # reference row
            ref_cells = "  ".join(
                f"{(agg['references']['full_text'] if c=='dec_text' else agg['references']['full_latent']):>22.3f}"
                for c in conditions)
            print(f"    1.000   all  {ref_cells}")
            for kr in keep_rates:
                row = agg["table"][variant][pattern][kr]
                cells = "  ".join(
                    (f"{row[c]:>22.3f}" if row[c] is not None else f"{'-':>22}")
                    for c in conditions)
                print(f"    {kr:<6.4f}      {cells}")


def make_plot(agg, conditions, keep_rates, patterns, variants, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable — skipping plot: {e})")
        return
    style = {"dec_text": "tab:red", "dec_latent": "tab:green",
             "dec_latent_renumbered": "tab:orange"}
    pat_ls = {"strided": "-", "random": "--"}
    xs_all = [1.0] + list(keep_rates)
    fig, axes = plt.subplots(1, len(variants), figsize=(7 * len(variants), 5), squeeze=False)
    for ax, variant in zip(axes[0], variants):
        for pattern in patterns:
            for cond in conditions:
                ref = agg["references"]["full_text"] if cond == "dec_text" else agg["references"]["full_latent"]
                ys = [ref] + [agg["table"][variant][pattern][kr][cond] for kr in keep_rates]
                xs = [x for x, y in zip(xs_all, ys) if y is not None]
                ysf = [y for y in ys if y is not None]
                label = f"{cond} ({pattern})"
                ax.plot(xs, ysf, pat_ls[pattern], marker="o", color=style.get(cond, "gray"),
                        label=label, alpha=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs_all)
        ax.set_xticklabels([f"{x:g}" for x in xs_all])
        ax.set_xlabel("keep-rate")
        ax.set_ylabel("recall (planted fact)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"variant: {variant}")
        ax.axhline(HOLD, color="gray", lw=0.5, ls=":")
        ax.axhline(COLLAPSE, color="gray", lw=0.5, ls=":")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
    fig.suptitle("Exp 3.1 — latent-decimation vs text-decimation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"  saved plot → {out_path}")


GAP_MARGIN = 0.2   # max (dec_latent − dec_text) gap that counts as a real divergence


def interpret(agg, conditions, keep_rates, variants):
    """Read the verdict from the MAXIMUM dec_latent − dec_text gap across rates,
    NOT from the heaviest thinning.

    Every curve eventually collapses at extreme thinning (you destroy the fact
    regardless of representation), so reading the verdict off the most-thinned row
    declares "illusory" on any successful experiment — that was the original bug. The
    signal lives where text is still alive enough to lose to: the largest gap between
    the latent and text curves. We report that gap, where it occurs, and — separately
    — whether the latent advantage survives renumbering (representational vs.
    positional)."""
    if not keep_rates:
        return
    print("\n" + "=" * 72)
    print("EXP 3.1 — reading (max latent−text gap across rates; floor rows ignored)")
    for variant in variants:
        # find the (pattern, keep_rate) with the largest dec_latent − dec_text gap
        best = None   # (gap, pattern, kr, dt, dl)
        renum_diffs = []
        renum_collapse = False
        for pattern in agg["table"][variant]:
            for kr in keep_rates:
                row = agg["table"][variant][pattern].get(kr, {})
                dt, dl, rn = row.get("dec_text"), row.get("dec_latent"), row.get("dec_latent_renumbered")
                if dt is not None and dl is not None:
                    gap = dl - dt
                    if best is None or gap > best[0]:
                        best = (gap, pattern, kr, dt, dl)
                if dl is not None and rn is not None:
                    # only compare where latent is alive enough for the test to mean anything
                    if dl > COLLAPSE:
                        renum_diffs.append(abs(dl - rn))
                        if rn <= COLLAPSE and dl >= HOLD:
                            renum_collapse = True

        if best is None:
            print(f"  [{variant}] no comparable cells")
            continue
        gap, pat, kr, dt, dl = best
        print(f"  [{variant}] max gap {gap:+.2f} at keep={kr:g} ({pat}): "
              f"dec_text={dt:.2f} dec_latent={dl:.2f}")
        if gap >= GAP_MARGIN:
            print("      → latent ≠ text: layer-12 states carry context their tokens do not")
        elif dl <= COLLAPSE:
            print("      → both arms collapse everywhere; no rate shows latent surviving → "
                  "thin less / lengthen docs (or latent truly interchangeable)")
        else:
            print("      → graded / weak separation → see the curve")

        # representational vs positional
        if renum_diffs:
            mean_diff = sum(renum_diffs) / len(renum_diffs)
            if renum_collapse and mean_diff > GAP_MARGIN:
                print(f"      → renumbered collapses (mean |Δ|={mean_diff:.2f}) → POSITION is "
                      "what matters, not the representation")
            else:
                print(f"      → renumbered tracks latent (mean |Δ|={mean_diff:.2f}) → advantage "
                      "is REPRESENTATIONAL, not a position artifact")
    print("  (needle_protected isolates folded context; needle_decimated is the harder, "
          "needle-loss test — compare the two.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="proofs/data/e31.json")
    parser.add_argument("--plot", default=None, help="path to save the curves PNG")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--layer", type=int, default=12,
                        help="injection layer (validated winner is 12)")
    parser.add_argument("--keep-rates", default="0.5,0.25,0.125,0.0625")
    parser.add_argument("--patterns", default="strided,random")
    parser.add_argument("--variants", default="needle_protected,needle_decimated")
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    parser.add_argument("--random-trials", type=int, default=1,
                        help="random-pattern draws per cell, averaged (≥2 reduces draw luck)")
    parser.add_argument("--no-sink", action="store_true",
                        help="do NOT retain the attention-sink position 0")
    parser.add_argument("--sanity-only", action="store_true",
                        help="run gates + canary and stop (do this first)")
    parser.add_argument("--reasoning", action="store_true")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import proofs.common as _common
    _common.SUPPRESS_THINK = not args.reasoning

    keep_rates = [float(x) for x in args.keep_rates.split(",")]
    patterns = [p.strip() for p in args.patterns.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    keep_sink = not args.no_sink

    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))
    tok, model = load_deepseek(cfg, devices=devices)

    print(f"\n########## EXP 3.1 — decimation at layer {args.layer} ##########")
    print(f"  keep_sink={keep_sink}  random_trials={args.random_trials}")
    print(f"  keep_rates={keep_rates}  patterns={patterns}")
    print(f"  variants={variants}  conditions={conditions}")

    prepared = prepare(model, tok, args.layer, SYNTHETIC_DOCS_LONG)
    items = run_gates(model, tok, prepared, args.layer, args.max_new_tokens)
    s_summary, canary = sanity_gates(model, tok, prepared, items, args.layer,
                                     args.max_new_tokens, keep_sink=keep_sink)

    result = {"layer": args.layer, "keep_sink": keep_sink,
              "sanity": s_summary, "gate_items": items}

    if args.sanity_only:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved sanity result → {args.out}")
        if not s_summary["sanity_pass"]:
            print("Sanity did NOT pass — fix before running the sweep.")
        return

    if not s_summary["sanity_pass"]:
        print("\n!!! Sanity gates did not pass. Running the sweep anyway would produce "
              "noise. Re-run with --sanity-only, fix the flagged issue, then sweep.")
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return

    records = run_sweep(model, tok, prepared, items, args.layer, args.max_new_tokens,
                        keep_rates, patterns, variants, conditions,
                        random_trials=args.random_trials, keep_sink=keep_sink)
    agg = aggregate(records, items, conditions, keep_rates, patterns, variants)
    print_table(agg, conditions, keep_rates, patterns, variants)
    interpret(agg, conditions, keep_rates, variants)

    result["aggregate"] = agg
    result["sweep_records"] = records
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {args.out}")

    if args.plot:
        make_plot(agg, conditions, keep_rates, patterns, variants, args.plot)


if __name__ == "__main__":
    main()

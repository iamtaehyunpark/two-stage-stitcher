"""
Proof 2.4 — Content-token error tolerance (Chain 2, rung 2.4).

Runs BEFORE Proof 2.2 (the trained stitcher). The question: how accurate must the
stitcher's content-token residuals be for recall to survive the 68-layer recompute?
This gives Proof 2.2 a concrete threshold instead of training blind. The cliff shape
also resolves the amplification question: steep = 68-layer recompute amplifies content
error; gentle = content is intrinsically hard but the recompute doesn't worsen it.

Single-model (DeepSeek only), no Qwen, no training. Reuses the same gated set and
residual path as Proofs 2.0 / 2.1.

Design:
  Start from the true layer-12 residual (the 0.95-ceiling condition, p2_0 confirmed).
  For each gated item identify:
    content_pos  — token positions of the answer-bearing span (needle_idx: the union
                   of gold-sentence char spans, same as p5_latent_vs_rag)
    struct_pos   — all other document positions (N - n_content)

  Content sweep (main experiment):
    Perturb ONLY content-token positions to a target per-token cosine ∈ COS_TARGETS;
    structural positions hold true. Measure recall → tolerance curve.

  Structural control:
    Perturb ONLY structural positions to each cos ∈ CONTROL_COS; content holds true.
    If structural perturbation barely dents recall while content cliffs → content
    fidelity is the sole bottleneck, and the threshold is clean.

  The perturbation is orthogonal noise calibrated to exact per-token cos = target:
    for each position i: y_i_perturbed = y_i + t_i * n_orth_i
    where n_orth_i ⊥ y_i and t_i = ||y_i|| * sqrt(1 - cos²) / (cos * ||n_orth_i||)
    → cos(y_i_perturbed, y_i) = cos exactly.

Verdict:
  THRESHOLD_X: "stitcher must reach content-token cos ≥ X for recall ≥ 90% of ceiling."
  Cliff diagnosis (steep / gentle): max recall drop between adjacent cos levels.

Usage:
  # full run (reusing the p2_1 gate to skip re-gating):
  CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p2_4_tolerance.py --arm synth_multihop \\
      --synth-n 40 --out proofs/data/p2_4.json \\
      --gate-cache proofs/data/p2_1_gated_synth_multihop.json

  # fresh gate (if no p2_1 gate exists):
  CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p2_4_tolerance.py --arm synth_multihop \\
      --synth-n 40 --out proofs/data/p2_4.json

  # wire-test (no think, no judge, n=4, skip structural control):
  CUDA_VISIBLE_DEVICES=4,5,6,7 python proofs/p2_4_tolerance.py --arm synth_multihop \\
      --synth-n 4 --no-think --no-judge --no-struct-control

  # re-score a saved run (no GPU):
  python proofs/p2_4_tolerance.py --rescore proofs/data/p2_4.json
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import argparse
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from proofs.common import (
    load_deepseek, final_answer, _with_think_control,
    capture_doc_cache, split_forward_generate,
    recompute_doc_cache_from_residual,
)
from proofs.p5_latent_vs_rag import (
    run_gate, load_candidates, score_all, judge_answer,
    _present_scorers, _set_think, _free_cuda,
    PREFILL_PREFIX, PREFILL_QSUFFIX, MIN_N, GAP,
    needle_idx_of,
)

# ── experiment parameters ─────────────────────────────────────────────────────
# Content sweep: these are the per-content-token cosine targets to sweep.
COS_TARGETS = [0.99, 0.97, 0.95, 0.90, 0.80, 0.70, 0.60]

# Structural control: a subset of the same cos range.  Run all by default but
# --no-struct-control skips it entirely.  The control is less critical than the
# content sweep, so a subset is fine if runtime is a concern.
CONTROL_COS = [0.99, 0.90, 0.80, 0.70, 0.60]

# A recall level is "OK" if it is ≥ this fraction of the ceiling recall.
RECALL_FLOOR_FRAC = 0.90

# Condition name helpers
_TRUE_COND = "residual_inject_true"


def _content_cond(cos):
    return f"perturb_content_{cos:.2f}"


def _struct_cond(cos):
    return f"perturb_struct_{cos:.2f}"


# ── perturbation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def perturb_residual(Y, positions, target_cos, seed):
    """Return a copy of Y (1, N, d) with the given token positions perturbed so
    that each has per-token cosine = target_cos with its true value. All other
    positions are unchanged.

    Uses orthogonal Gaussian noise: for each position i,
      n_orth = n - (n·y_i / ||y_i||^2) y_i      (n_orth ⊥ y_i)
      t_i    = ||y_i|| * sqrt(1-cos²) / (cos * ||n_orth||)
      result = y_i + t_i * n_orth
    This gives cos(result, y_i) = target_cos exactly (up to fp32 precision).

    target_cos == 1.0 returns an exact clone (no perturbation).
    """
    if target_cos >= 1.0 or not positions:
        return Y.clone()

    Y_out = Y.clone()
    # Work in float32 on CPU to avoid dtype issues; move back at the end.
    Y_cpu = Y.squeeze(0).float().cpu()  # (N, d)

    cos = float(target_cos)
    sin = math.sqrt(max(0.0, 1.0 - cos * cos))

    rng = torch.Generator()
    rng.manual_seed(seed)

    for pos in positions:
        y = Y_cpu[pos].clone()          # (d,)
        norm_y = float(y.norm())
        if norm_y < 1e-12:
            continue

        n = torch.randn(y.shape[0], generator=rng)   # (d,) iid Gaussian
        # Remove the component parallel to y
        n_orth = n - (float(n @ y) / (norm_y ** 2)) * y
        norm_n = float(n_orth.norm())
        if norm_n < 1e-12:
            # Degenerate: sample again with a different seed offset
            n = torch.randn(y.shape[0], generator=rng)
            n_orth = n - (float(n @ y) / (norm_y ** 2)) * y
            norm_n = float(n_orth.norm())
            if norm_n < 1e-12:
                continue

        # Scale so cos(y + t*n_orth, y) = cos exactly
        # cos = ||y|| / sqrt(||y||^2 + t^2 * ||n_orth||^2)  →  t = ||y||*sin / (cos*||n_orth||)
        t = norm_y * sin / (cos * norm_n)
        y_perturbed = y + t * n_orth
        Y_out[0, pos] = y_perturbed.to(dtype=Y.dtype, device=Y.device)

    return Y_out


def _verify_per_token_cos(Y_pert, Y_true, positions):
    """Debug helper: compute actual per-token cosine at the perturbed positions."""
    Y_p = Y_pert.squeeze(0).float().cpu()
    Y_t = Y_true.squeeze(0).float().cpu()
    coss = []
    for pos in positions:
        a, b = Y_p[pos], Y_t[pos]
        c = float(torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-12))
        coss.append(c)
    return coss


# ── answer under a perturbed residual ────────────────────────────────────────

def ans_perturbed(model, tok, qcache, Y_perturbed, n_pre, q, layer, m):
    """Inject the perturbed residual via the 2.0-validated recompute path and answer q."""
    resid_cache = recompute_doc_cache_from_residual(model, Y_perturbed, n_pre, layer, qcache)
    query = _with_think_control(PREFILL_QSUFFIX.format(question=q))
    txt = split_forward_generate(model, tok, resid_cache, n_pre, query_text=query,
                                 target_layer=layer, max_new_tokens=m)
    del resid_cache
    return final_answer(txt)


# ════════════════════════════════════════════════════════════════════════════════
# Stage: eval
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
        print(f"  [resume] {len(records)} records loaded; their ids skipped")

    cap = args.max_eval or len(gated)
    todo = [g for g in gated if g["id"] not in done_ids][:max(0, cap - len(records))]

    struct_cos = [] if args.no_struct_control else CONTROL_COS
    content_conds = [_content_cond(c) for c in COS_TARGETS]
    struct_conds  = [_struct_cond(c)  for c in struct_cos]
    all_conds = [_TRUE_COND] + content_conds + struct_conds

    print(f"  conditions per item: {_TRUE_COND} + {len(content_conds)} content + "
          f"{len(struct_conds)} struct = {len(all_conds)} total")
    print(f"  eval plan: {len(records)} done, {len(todo)} to run "
          f"(cap {cap}, gated {len(gated)}); think_max={m}, judge={args.judge}")

    def snapshot():
        return {"layer": layer, "think": args.think, "judged": args.judge,
                "cos_targets": COS_TARGETS, "control_cos": struct_cos,
                "records": records}

    for rec in todo:
        doc, q, gold = rec["doc_text"], rec["question"], rec["answer"]
        decoys = rec.get("decoy_values", [])

        # Identify content vs. structural positions using the needle span.
        # content_pos = answer-bearing token positions (needle_idx)
        # struct_pos  = all other document positions
        content_pos = needle_idx_of(tok, rec, args.max_doc_tokens)
        if not content_pos:
            print(f"  WARNING: {rec['id']} has no needle positions; skip")
            continue
        content_set = set(content_pos)

        # q-fair capture: same framing as p2_0 / p2_1
        pre_ids = tok(PREFILL_PREFIX.format(document=doc), return_tensors="pt",
                      truncation=True, max_length=args.max_doc_tokens).input_ids
        qcache, Y_pre, n_pre = capture_doc_cache(model, pre_ids, layer)
        # Y_pre: (1, N_ds, d_ds) on the device of layer `layer`

        struct_pos = [i for i in range(n_pre) if i not in content_set]

        _set_think(args.think)

        answers, scores = {}, {}

        # ── residual_inject_true ────────────────────────────────────────────
        # Plain recompute from the true residual — the ceiling condition.
        resid_cache = recompute_doc_cache_from_residual(model, Y_pre, n_pre, layer, qcache)
        query = _with_think_control(PREFILL_QSUFFIX.format(question=q))
        true_txt = final_answer(split_forward_generate(
            model, tok, resid_cache, n_pre, query_text=query,
            target_layer=layer, max_new_tokens=m))
        del resid_cache
        answers[_TRUE_COND] = true_txt
        scores[_TRUE_COND] = score_all(true_txt, gold, decoys)
        if args.judge:
            scores[_TRUE_COND]["judge"] = judge_answer(model, tok, q, gold, true_txt)

        # ── content sweep ───────────────────────────────────────────────────
        for i, cos_t in enumerate(COS_TARGETS):
            cond = _content_cond(cos_t)
            seed = hash((rec["id"], "content", cos_t)) & 0xFFFFFFFF
            Y_pert = perturb_residual(Y_pre, content_pos, cos_t, seed)
            ans = ans_perturbed(model, tok, qcache, Y_pert, n_pre, q, layer, m)
            del Y_pert
            answers[cond] = ans
            scores[cond] = score_all(ans, gold, decoys)
            if args.judge:
                scores[cond]["judge"] = judge_answer(model, tok, q, gold, ans)

        # ── structural control ──────────────────────────────────────────────
        for cos_t in struct_cos:
            cond = _struct_cond(cos_t)
            seed = hash((rec["id"], "struct", cos_t)) & 0xFFFFFFFF
            Y_pert = perturb_residual(Y_pre, struct_pos, cos_t, seed)
            ans = ans_perturbed(model, tok, qcache, Y_pert, n_pre, q, layer, m)
            del Y_pert
            answers[cond] = ans
            scores[cond] = score_all(ans, gold, decoys)
            if args.judge:
                scores[cond]["judge"] = judge_answer(model, tok, q, gold, ans)

        del qcache, Y_pre
        _free_cuda()

        records.append({
            "id": rec["id"], "question": q, "gold": gold,
            "type": rec.get("type", ""), "decoy_values": decoys,
            "n_pre": int(n_pre), "n_content": len(content_pos),
            "n_struct": len(struct_pos),
            "answers": answers, "scores": scores,
        })
        if resume_path:
            with open(resume_path, "w") as f:
                json.dump(snapshot(), f, default=str)

        hs = args.headline if args.headline in scores[_TRUE_COND] else "strict"
        s_true = scores[_TRUE_COND].get(hs)
        s_mid = scores.get(_content_cond(0.80), {}).get(hs)
        s_low = scores.get(_content_cond(0.60), {}).get(hs)
        print(f"  eval [{len(records)} done / {cap}] {rec['id']}: "
              f"n_content={len(content_pos)} "
              f"true={s_true} cos0.80={s_mid} cos0.60={s_low}", end="\r")
    print()
    return snapshot()


# ════════════════════════════════════════════════════════════════════════════════
# Aggregation
# ════════════════════════════════════════════════════════════════════════════════

def _rate(records, cond, scorer):
    vals = [r["scores"][cond][scorer] for r in records
            if scorer in r.get("scores", {}).get(cond, {})]
    return round(sum(vals) / len(vals), 3) if vals else None


def aggregate(result, headline):
    records = result.get("records", [])
    if not records:
        return {"n": 0}

    scols = _present_scorers(records)
    hs = headline if headline in scols else "strict"

    cos_targets = result.get("cos_targets", COS_TARGETS)
    control_cos = result.get("control_cos", CONTROL_COS)

    ceiling = {s: _rate(records, _TRUE_COND, s) for s in scols}

    content_curve = []
    for cos_t in cos_targets:
        cond = _content_cond(cos_t)
        row = {"cos": cos_t, "cond": cond}
        for s in scols:
            row[s] = _rate(records, cond, s)
        # recall drop vs ceiling (headline scorer)
        r = row.get(hs)
        c = ceiling.get(hs)
        row["drop_vs_ceiling"] = round(r - c, 3) if (r is not None and c is not None) else None
        content_curve.append(row)

    struct_curve = []
    for cos_t in control_cos:
        cond = _struct_cond(cos_t)
        row = {"cos": cos_t, "cond": cond}
        for s in scols:
            row[s] = _rate(records, cond, s)
        r = row.get(hs)
        c = ceiling.get(hs)
        row["drop_vs_ceiling"] = round(r - c, 3) if (r is not None and c is not None) else None
        struct_curve.append(row)

    return {
        "n": len(records),
        "distinct": len({r["id"] for r in records}),
        "headline": hs,
        "ceiling": ceiling,
        "content_curve": content_curve,
        "struct_curve": struct_curve,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Verdict
# ════════════════════════════════════════════════════════════════════════════════

def verdict(agg):
    n = agg.get("n", 0)
    if n < MIN_N:
        return {"status": "UNDERPOWERED",
                "detail": f"n={n} < {MIN_N}; raise --synth-n"}

    hs = agg["headline"]
    ceiling_recall = agg["ceiling"].get(hs)
    if ceiling_recall is None:
        return {"status": "MISSING_DATA", "detail": "no ceiling recall scores"}
    if ceiling_recall < 1e-6:
        return {"status": "CEILING_ZERO",
                "detail": "ceiling recall is 0 — gated set or operating point broken"}

    recall_floor = RECALL_FLOOR_FRAC * ceiling_recall

    # Find the threshold: the highest cos_t where recall is still ≥ floor.
    # Equivalently: the lowest cos_t we can afford.
    # Walk from high cos (easy) to low cos (hard) and find where it first drops below floor.
    curve = agg["content_curve"]
    threshold = None         # the required minimum cos
    cliff_drop = None        # biggest single-step drop in the curve
    prev_recall = ceiling_recall
    drops = []

    for row in curve:                # curve is ordered high→low cos
        r = row.get(hs)
        if r is None:
            continue
        drop = prev_recall - r
        drops.append(drop)
        prev_recall = r
        if r < recall_floor and threshold is None:
            # First failure: the required minimum is the cos ONE step above this.
            # Find the cos of the previous row.
            idx = curve.index(row)
            if idx > 0:
                threshold = curve[idx - 1]["cos"]
            else:
                threshold = 1.0     # fails even at the highest tested cos

    cliff_drop = max(drops) if drops else None

    # If recall never dropped below floor, the stitcher can afford cos = min(COS_TARGETS)
    if threshold is None:
        threshold = min(row["cos"] for row in curve if row.get(hs) is not None)
        status = f"FLOOR_BELOW_MIN_TESTED"
    else:
        status = f"THRESHOLD_{threshold:.2f}"

    # Cliff diagnosis
    if cliff_drop is not None:
        cliff = "STEEP" if cliff_drop > 0.10 else "GENTLE"
        # Steep: max single-step drop > 10pp → 68-layer recompute amplifies content error
        # Gentle: error accumulates smoothly → content tokens intrinsically hard but no amplification
    else:
        cliff = "UNKNOWN"

    # Structural control: does it barely dent recall?
    struct_clean = True
    struct_detail = []
    for row in agg["struct_curve"]:
        r = row.get(hs)
        if r is not None and r < recall_floor:
            struct_clean = False
            struct_detail.append(f"cos={row['cos']:.2f} recall={r:.3f}")

    return {
        "status": status,
        "threshold": threshold,
        "cliff": cliff,
        "cliff_drop": cliff_drop,
        "recall_floor": round(recall_floor, 3),
        "ceiling_recall": ceiling_recall,
        "struct_clean": struct_clean,
        "struct_failures": struct_detail,
        "detail": (
            f"threshold {threshold:.2f} (recall must stay ≥ {recall_floor:.2f} = "
            f"{RECALL_FLOOR_FRAC*100:.0f}% of ceiling {ceiling_recall:.3f}) | "
            f"cliff={cliff} (max step drop {cliff_drop:.3f}) | "
            f"struct_clean={struct_clean}"
        ),
    }


_GLOSS = {
    "FLOOR_BELOW_MIN_TESTED":
        "   → Recall held at ≥ 90% of ceiling even at the lowest tested cos (0.60).\n"
        "     The stitcher has a very generous tolerance. Content error does NOT\n"
        "     amplify through the 68-layer recompute. GREEN LIGHT to Proof 2.2 with\n"
        "     a wide target band.",
    "UNDERPOWERED":
        "   → Too few gated items; raise --synth-n to ≥ 30.",
    "MISSING_DATA":
        "   → No scored records; check the run completed correctly.",
    "CEILING_ZERO":
        "   → Ceiling recall is 0; the gated set or operating point is broken.",
}


# ════════════════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════════════════

def _fmt(v, w=8):
    return f"{'·':>{w}}" if v is None else f"{v:>{w}.3f}"


def report(result, agg, gate_summary, headline):
    print("\n" + "=" * 80)
    print(f"PROOF 2.4 — content-token error tolerance  "
          f"(L{result.get('layer', '?')}, "
          f"{'think-on' if result.get('think') else 'think-off'}, "
          f"headline={agg.get('headline', headline)})")
    if gate_summary:
        print(f"  gate: {gate_summary['gated']} gated / {gate_summary['candidates']} "
              f"candidates  (closed-book discard {gate_summary['discard_rate']}, "
              f"A pass {gate_summary['a_pass_rate']})")
    n = agg.get("n", 0)
    print(f"  eval n={n} (distinct {agg.get('distinct', 0)})")

    if n == 0:
        print("  (no records to report)")
        return verdict(agg)

    records = result.get("records", [])
    scols = _present_scorers(records)
    hs = agg.get("headline", "strict")

    # Content-token sweep table
    print(f"\n  === Content-token perturbation sweep ===")
    col_w = 10
    head = f"  {'cos_target':>10}" + "".join(f"{c:>{col_w}}" for c in scols) + f"  drop({hs})"
    print(head)
    print("  " + "-" * (len(head) - 2))

    ceiling = agg.get("ceiling", {})
    ceil_str = "  " + f"{'1.00 (true)':>10}" + "".join(
        _fmt(ceiling.get(c), col_w) for c in scols) + f"  {0.000:>+8.3f}"
    print(ceil_str)

    for row in agg.get("content_curve", []):
        drop = row.get("drop_vs_ceiling")
        drop_str = f"  {drop:>+8.3f}" if drop is not None else f"  {'·':>8}"
        line = f"  {row['cos']:>10.2f}" + "".join(_fmt(row.get(c), col_w) for c in scols) + drop_str
        print(line)

    # Structural control table
    if agg.get("struct_curve"):
        print(f"\n  === Structural-token control (content held true) ===")
        print(head)
        print("  " + "-" * (len(head) - 2))
        print(ceil_str)
        for row in agg.get("struct_curve", []):
            drop = row.get("drop_vs_ceiling")
            drop_str = f"  {drop:>+8.3f}" if drop is not None else f"  {'·':>8}"
            line = f"  {row['cos']:>10.2f}" + "".join(_fmt(row.get(c), col_w) for c in scols) + drop_str
            print(line)

    # n_content stats
    cnt_ns = [r["n_content"] for r in records if "n_content" in r]
    if cnt_ns:
        print(f"\n  needle sizes: min={min(cnt_ns)} mean={sum(cnt_ns)/len(cnt_ns):.1f} max={max(cnt_ns)}")

    v = verdict(agg)
    print("\n  " + "-" * 76)
    print(f"  VERDICT: {v['status']}")
    print(f"    {v['detail']}")
    if v.get("threshold") is not None:
        print(f"\n  THRESHOLD FOR PROOF 2.2: content-token cos ≥ {v['threshold']:.2f}")
        if v["cliff"] == "STEEP":
            print("  CLIFF=STEEP → 68-layer recompute AMPLIFIES content error. "
                  "The stitcher must hit the threshold precisely; there is little margin.")
        else:
            print("  CLIFF=GENTLE → content tokens are intrinsically hard but the recompute "
                  "does not amplify. The stitcher has a smooth target.")
        if v["struct_clean"]:
            print("  STRUCT=CLEAN → structural perturbation barely dents recall; "
                  "content fidelity is the sole target. Threshold is clean.")
        else:
            print(f"  STRUCT=DIRTY → structural perturbation also hurts recall at: "
                  f"{', '.join(v['struct_failures'])}. "
                  "Both content and structural fidelity matter.")
    if v["status"] in _GLOSS:
        print(_GLOSS[v["status"]])
    return v


# ════════════════════════════════════════════════════════════════════════════════
# Rescore (no GPU)
# ════════════════════════════════════════════════════════════════════════════════

def rescore(result):
    for r in result.get("records", []):
        decoys = r.get("decoy_values", [])
        judged = {c: r["scores"][c].get("judge") for c in r["scores"]
                  if "judge" in r["scores"].get(c, {})}
        r["scores"] = {c: score_all(a, r["gold"], decoys)
                       for c, a in r.get("answers", {}).items()}
        for c, jv in judged.items():
            r["scores"][c]["judge"] = jv
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", default=None, metavar="PATH",
                    help="re-score / re-verdict a saved run; no model load")
    ap.add_argument("--arm", default="synth_multihop",
                    choices=["synth_multihop", "synth_parity", "hotpot"])
    ap.add_argument("--synth-n", type=int, default=40)
    ap.add_argument("--parity-n", type=int, default=32)
    ap.add_argument("--max-candidates", type=int, default=400, help="hotpot arm only")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="suppress reasoning (smoke/wire-test only; operating point is think-ON)")
    ap.set_defaults(think=True)
    ap.add_argument("--no-judge", dest="judge", action="store_false",
                    help="skip inline LLM-judge")
    ap.set_defaults(judge=True)
    ap.add_argument("--no-struct-control", action="store_true",
                    help="skip the structural perturbation control (saves ~30%% of runtime)")
    ap.add_argument("--headline", default="judge",
                    help="primary scorer for headline/verdict (default judge; "
                         "falls back to strict if judge pass was skipped)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--think-max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-doc-tokens", type=int, default=4096)
    ap.add_argument("--max-eval", type=int, default=None,
                    help="cap gated items evaluated this run (resume-safe; default all)")
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="logical GPU indices for DeepSeek shards (select physical GPUs "
                         "with CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--max-gpu-memory", default="70GiB",
                    help="max memory per GPU for DeepSeek sharding")
    ap.add_argument("--gate-cache", default=None,
                    help="path to cache/reuse the gated set (default derived from --out)")
    ap.add_argument("--out", default="proofs/data/p2_4.json")
    args = ap.parse_args()

    max_gpu_mem = args.max_gpu_memory
    if isinstance(max_gpu_mem, str) and max_gpu_mem.isdigit():
        max_gpu_mem = f"{max_gpu_mem}GiB"

    # ── Rescore path (no GPU) ────────────────────────────────────────────────
    if args.rescore:
        with open(args.rescore) as f:
            result = rescore(json.load(f))
        agg = aggregate(result, args.headline)
        v = report(result, agg, result.get("gate_summary"), agg["headline"])
        result["aggregate"], result["verdict"] = agg, v
        with open(args.rescore, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nRe-scored → {args.rescore}")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gate_cache = args.gate_cache or args.out.replace(".json", f"_gated_{args.arm}.json")
    need_gate = not os.path.exists(gate_cache)

    recs = None
    if need_gate:
        recs = load_candidates(args.arm, args)
        print(f"[gate] resolved {len(recs)} {args.arm} candidates (pre-load)")

    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))
    print(f"Loading DeepSeek-70B across GPUs {devices} (max_memory={max_gpu_mem}) …")
    tok, model = load_deepseek(cfg, devices=devices, max_memory_per_gpu=max_gpu_mem)

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
        print("No gated items — nothing to evaluate.")
        return

    result = run_eval(model, tok, gated, args, resume_path=args.out)
    result["arm"] = args.arm
    result["gate_summary"] = gate_summary
    agg = aggregate(result, args.headline)
    v = report(result, agg, gate_summary, agg["headline"])
    result["aggregate"], result["verdict"] = agg, v

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

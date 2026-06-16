"""
Proof 0 — the split-forward plumbing is correct.

A regression test, not a science result. Its only job: certify that RoPE, the
causal/doc-visible masks, position bookkeeping, and the two-cache decode in
`core.split_forward` reproduce ordinary inference *before* any later number is
trusted. The original project hit exactly these bugs (wrong RoPE shape, missing
KV cache) once; this is where we catch them against a known answer.

Why a MEMORIZED document. Here memory is the feature. We use a document whose
answers we know independently (and the model knows too), so any deviation is
attributable to the plumbing, not to ambiguity about what "correct" means. This
proof does NOT test content transfer — that needs synthetic facts (Proof 1).

Conditions, same frozen DeepSeek-70B, same questions:
  A       — full prefill: document + question as ordinary tokens (the reference).
  SF-true — split-forward: capture the document's TRUE layer-`target_layer` states,
            hand them over as the document, run only the query through 0..29.

They will not be bit-identical (SF-true's lower layers never saw the question), but
they must be the SAME answer: same facts, same correctness, no degeneration.

PASS  : SF-true coherent, correct, matching A.
FAIL-degenerate : `a the a the`, repetition, gibberish → RoPE/join/cache bug. STOP.
FAIL-wrong      : fluent but wrong → query not attending injected positions (a mask
                  or offset bug that silently isolates the query). STOP.

Usage:
    python proofs/p0_plumbing.py --out proofs/data/p0.json
    python proofs/p0_plumbing.py --document /data/tpark45/docs/wiki_00876.txt \
        --qa proofs/data/p0_witcher_qa.json        # use a real long memorized doc
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.common import (
    load_deepseek, capture_document, full_prefill_answer, inject_answer, correct,
)

# A compact, universally memorized passage. Every answer is independently known.
DEFAULT_DOC = (
    "Mount Everest is Earth's highest mountain above sea level, located in the "
    "Mahalangur Himal sub-range of the Himalayas. Its peak rises to 8,849 metres "
    "and sits on the border between Nepal and the Tibet Autonomous Region of China. "
    "The mountain was named in 1865 after Sir George Everest, a former Surveyor "
    "General of India. The first confirmed ascent to the summit was made on 29 May "
    "1953 by Edmund Hillary of New Zealand and Tenzing Norgay, a Sherpa of Nepal. "
    "In Nepali the mountain is known as Sagarmatha, and in Tibetan as Chomolungma."
)
# Facts where the model's memory and the document AGREE and the answer is
# unambiguous. (Deliberately NOT the height: 8,848 vs 8,849 is a real-world
# revision the model's prior disagrees with the text on — a confound for a test
# whose whole point is that memory and document coincide.)
DEFAULT_QA = [
    # gold "Himalaya" matches both "Himalayas" and "Himalayan"; each question is
    # self-contained (no cross-question coreference — every item is asked standalone
    # with only the injected document, so a dangling "him" has no antecedent).
    {"q": "In which mountain range does Mount Everest lie?", "a": "Himalaya"},
    {"q": "Who was the first to reach the summit of Mount Everest, in 1953?", "a": "Hillary"},
    {"q": "Who accompanied Edmund Hillary on the first ascent of Mount Everest?", "a": "Tenzing Norgay"},
    {"q": "After whom was Mount Everest named?", "a": "George Everest"},
    {"q": "What is Mount Everest called in Tibetan?", "a": "Chomolungma"},
]

# Plumbing certificate: non-degenerate and substantively matching A on at least
# this fraction. Not literal 5/5 — SF-true is intentionally not bit-identical to A
# (its lower layers never saw the question), so a stray phrasing/near-tie miss must
# not read as a plumbing bug. Degeneration or fluent-but-wrong is the real failure.
P0_PASS_RATE = 0.8


def looks_degenerate(text: str) -> bool:
    """Heuristic for the `a the a the` / repetition failure mode."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 4:
        return False
    uniq_ratio = len(set(words)) / len(words)
    if uniq_ratio < 0.35:
        return True
    # any single token dominating, or a tight bigram loop
    from collections import Counter
    if Counter(words).most_common(1)[0][1] > max(6, 0.5 * len(words)):
        return True
    bigrams = list(zip(words, words[1:]))
    if bigrams and Counter(bigrams).most_common(1)[0][1] > 5:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", help="path to a memorized document (default: built-in Everest passage)")
    parser.add_argument("--qa", help="path to JSON [{q,a}, ...] matching --document")
    parser.add_argument("--out", default="proofs/data/p0.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpus", default="0,1,2,3", help="logical GPU indices for DeepSeek shards")
    parser.add_argument("--reasoning", action="store_true",
                        help="let R1 emit <think> traces instead of suppressing them")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import proofs.common as _common
    _common.SUPPRESS_THINK = not args.reasoning

    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))

    if args.document:
        with open(args.document) as f:
            document = f.read()
        with open(args.qa) as f:
            qa = json.load(f)
    else:
        document, qa = DEFAULT_DOC, DEFAULT_QA

    tok, model = load_deepseek(cfg, devices=devices)

    # Capture the document's true split-forward cache once; reuse for all questions.
    print("Capturing true layer-%d states …" % cfg.target_layer)
    doc_cache, n_doc = capture_document(model, tok, document, cfg.target_layer)
    print(f"  document = {n_doc} tokens")

    records = []
    for item in qa:
        q, gold = item["q"], item["a"]
        ans_a = full_prefill_answer(model, tok, document, q, args.max_new_tokens)
        ans_sf = inject_answer(model, tok, doc_cache, n_doc, q, cfg.target_layer,
                               args.max_new_tokens)

        a_ok = correct(ans_a, gold)
        sf_ok = correct(ans_sf, gold)
        degen = looks_degenerate(ans_sf)
        agree = a_ok and sf_ok

        records.append({
            "question": q, "gold": gold,
            "answer_a": ans_a, "answer_sf_true": ans_sf,
            "a_correct": a_ok, "sf_correct": sf_ok,
            "sf_degenerate": degen, "agree": agree,
        })
        print(f"\nQ: {q}\n  gold={gold!r}")
        print(f"  A      [{ 'ok' if a_ok else 'XX'}]: {ans_a[:160]}")
        print(f"  SF-true[{ 'ok' if sf_ok else 'XX'}{' DEGEN' if degen else ''}]: {ans_sf[:160]}")

    n = len(qa)
    sf_correct = sum(r["sf_correct"] for r in records)
    sf_degen = sum(r["sf_degenerate"] for r in records)
    all_pass = sf_degen == 0 and (sf_correct / n) >= P0_PASS_RATE
    summary = {
        "n_questions": n,
        "doc_tokens": n_doc,
        "a_correct": sum(r["a_correct"] for r in records),
        "sf_correct": sf_correct,
        "sf_degenerate": sf_degen,
        "agree_with_a": sum(r["agree"] for r in records),
        "pass_rate_threshold": P0_PASS_RATE,
        "verdict": "PASS" if all_pass else "FAIL",
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"A correct:        {summary['a_correct']}/{n}")
    print(f"SF-true correct:  {summary['sf_correct']}/{n}  (degenerate {sf_degen})")
    print(f"SF-true agrees with A: {summary['agree_with_a']}/{n}")
    print(f"VERDICT: {summary['verdict']}")
    if not all_pass:
        print("  → plumbing is NOT trustworthy. Do not run Proofs 1–2 until SF-true")
        print("    matches A (coherent + correct) on every memorized question.")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

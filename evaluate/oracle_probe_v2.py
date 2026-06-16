"""
Oracle probe v2 — decisive single-vector test, free of the memory confound.

The v1 probe was uninterpretable: it ran on a memorized document (*The Witcher*),
so Oracle-LAST's "success" was the 70B reciting world knowledge, not reading the
injected vector (the tell: it confabulated the Hussite Trilogy's author). This
version fixes the *experiment*, not the plumbing:

  1. SYNTHETIC-FACT documents — fabricated entities/numbers the model cannot know,
     so any correct answer must come from the injected vector.
  2. C-FAILS-FIRST gate — only score items where Condition C (no context) is wrong.
     If C already knows it, the item proves nothing.
  3. WRONG-DOCUMENT control (the falsifier) — inject document Y's vector but ask
     document X's question. A faithful injection should FAIL here; if it still
     answers X, the "signal" was memory, not the vector.
  4. Two single-vector summaries — MEAN-pooled (a real summary) and LAST-token
     (what the current stitcher emits). v1 only tried the weakest one (last token).

Plumbing note: single-vector injection uses the existing generate_with_injection
hook. For N=1 the placeholder-contamination flaw is negligible (one garbage
position the model ignores), so this is a fair test of the single vector. A fair
MULTI-token (sequence) test needs a proper two-cache split-forward and is out of
scope here — and moot, since the stitcher emits a single vector.

Usage:
    python evaluate/oracle_probe_v2.py --out evaluate/data/oracle_probe_v2.json
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_conditions import load_deepseek
from inference import generate_with_injection
from oracle_probe import capture_layer30, strip_think

# ── Synthetic documents — every fact is fabricated and ungoogleable ───────────
# Answers are specific strings/numbers the base model has no way to know.
SYNTHETIC_DOCS = [
    {
        "name": "zorvian_codex",
        "text": (
            "The Zorvian Codex is a manuscript first catalogued in the year 1487 by the "
            "explorer Maren Velloth, who recovered it from the flooded cellars of Khaldros. "
            "The codex contains exactly 3,412 verses, all composed in the extinct Tannic "
            "language. For centuries it was considered untranslatable, until the scholar "
            "Idris Pell produced the first complete translation in 1923. Pell attributed the "
            "work to the philosopher Banu Castreth, who is believed to have written it while "
            "imprisoned on the island of Sethry. The codex is currently held in the Varn "
            "Athenaeum, where it occupies a sealed vault designated Chamber 9."
        ),
        "qa": [
            {"q": "Who recovered the Zorvian Codex?", "a": "Maren Velloth"},
            {"q": "In what year was the Zorvian Codex first catalogued?", "a": "1487"},
            {"q": "How many verses does the Zorvian Codex contain?", "a": "3412"},
            {"q": "Who produced the first complete translation, and in what year?", "a": "Idris Pell"},
            {"q": "To whom is the codex attributed?", "a": "Banu Castreth"},
        ],
    },
    {
        "name": "harnel_engine",
        "text": (
            "The Harnel rotary engine was designed in 1956 by the engineer Lucia Pendran for "
            "the airship Calistra. It produced 1,840 horsepower and ran on a fuel mixture "
            "known as blue naphtha. The engine was notable for its seven-chamber compression "
            "cycle, an arrangement Pendran patented under the name the Vossler ring. Only "
            "four Harnel engines were ever built; the last surviving unit is displayed at the "
            "Tindall Institute in the city of Brassmoor. The Calistra itself was retired in "
            "1971 after completing 212 transcontinental flights."
        ),
        "qa": [
            {"q": "Who designed the Harnel rotary engine?", "a": "Lucia Pendran"},
            {"q": "How much horsepower did the Harnel engine produce?", "a": "1840"},
            {"q": "What fuel did the Harnel engine run on?", "a": "blue naphtha"},
            {"q": "What was Pendran's patented compression arrangement called?", "a": "Vossler ring"},
            {"q": "How many transcontinental flights did the Calistra complete?", "a": "212"},
        ],
    },
    {
        "name": "marsh_of_olden",
        "text": (
            "The Marsh of Olden is a wetland region governed since 1604 by the Pell Concord, "
            "an assembly of nine elected wardens. Its largest settlement, Quenby, sits on "
            "stilts above the water and houses roughly 8,700 residents. The marsh is famous "
            "for the greyfin eel, a species harvested only during the month locals call "
            "Sothmark. In 1889 a flood known as the Verrin Surge destroyed two thirds of "
            "Quenby, after which the wardens commissioned the great levee designed by the "
            "architect Hollis Drane."
        ),
        "qa": [
            {"q": "What assembly governs the Marsh of Olden?", "a": "Pell Concord"},
            {"q": "How many wardens are in the governing assembly?", "a": "nine"},
            {"q": "What is the largest settlement in the Marsh of Olden?", "a": "Quenby"},
            {"q": "What species is the marsh famous for harvesting?", "a": "greyfin eel"},
            {"q": "Who designed the great levee?", "a": "Hollis Drane"},
        ],
    },
    {
        "name": "tovic_protocol",
        "text": (
            "The Tovic Protocol is a set of navigation rules established in 1742 by the "
            "cartographer Selma Aurich for crossing the Ashen Strait. It mandates that ships "
            "travel in convoys of no more than five vessels, each carrying a marker lantern "
            "called a corden. The protocol was adopted after the loss of the merchant fleet "
            "Brae, which sank with 64 crew aboard. Aurich's original charts are kept in the "
            "Lormont Registry under catalogue number K-318."
        ),
        "qa": [
            {"q": "Who established the Tovic Protocol?", "a": "Selma Aurich"},
            {"q": "In what year was the Tovic Protocol established?", "a": "1742"},
            {"q": "What is the maximum number of vessels allowed in a convoy?", "a": "five"},
            {"q": "What is the marker lantern called?", "a": "corden"},
            {"q": "How many crew were lost on the merchant fleet Brae?", "a": "64"},
        ],
    },
]

QUERY_PROMPT = (
    "You are a helpful assistant. Answer the question based on the preceding document, "
    "using only the information it contains. If the answer is not present, say you do not know.\n\n"
    "Question: {question}\nAnswer:"
)

NO_CONTEXT_PROMPT = (
    "You are a helpful assistant. Answer the question as best you can.\n\n"
    "Question: {question}\nAnswer:"
)


def normalize(s: str) -> str:
    s = s.lower()
    s = s.replace(",", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def correct(answer: str, gold: str) -> bool:
    return normalize(gold) in normalize(answer)


def generate_plain(model, tokenizer, prompt, max_new_tokens):
    import torch
    dev = model.model.embed_tokens.weight.device
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(dev)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return strip_think(tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))


def main():
    import torch
    from config import StitcherConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="evaluate/data/oracle_probe_v2.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = StitcherConfig()
    llama_tok, llama_model = load_deepseek(cfg)
    first_device = next(llama_model.parameters()).device

    # Pre-capture each document's true layer-30 states once.
    doc_vectors = {}   # name -> {"mean": (1,1,D), "last": (1,1,D)}
    for doc in SYNTHETIC_DOCS:
        ids = llama_tok(doc["text"], return_tensors="pt",
                        truncation=True, max_length=8192).input_ids.to(first_device)
        Y = capture_layer30(llama_model, ids, cfg.target_layer)   # (1, N, D)
        doc_vectors[doc["name"]] = {
            "mean": Y.mean(dim=1, keepdim=True),
            "last": Y[:, -1:, :],
            "ntok": int(Y.shape[1]),
        }

    records = []
    doc_names = [d["name"] for d in SYNTHETIC_DOCS]

    for di, doc in enumerate(SYNTHETIC_DOCS):
        wrong_doc = doc_names[(di + 1) % len(doc_names)]   # a different document
        for qa in doc["qa"]:
            q, gold = qa["q"], qa["a"]
            query_text = QUERY_PROMPT.format(question=q)

            ans_c = generate_plain(llama_model, llama_tok,
                                   NO_CONTEXT_PROMPT.format(question=q), args.max_new_tokens)
            ans_mean = generate_with_injection(
                llama_model, llama_tok, doc_vectors[doc["name"]]["mean"].squeeze(1),
                query_text, cfg.target_layer, max_new_tokens=args.max_new_tokens)
            ans_last = generate_with_injection(
                llama_model, llama_tok, doc_vectors[doc["name"]]["last"].squeeze(1),
                query_text, cfg.target_layer, max_new_tokens=args.max_new_tokens)
            ans_wrong = generate_with_injection(
                llama_model, llama_tok, doc_vectors[wrong_doc]["mean"].squeeze(1),
                query_text, cfg.target_layer, max_new_tokens=args.max_new_tokens)

            rec = {
                "doc": doc["name"], "wrong_doc": wrong_doc,
                "question": q, "gold": gold,
                "c_no_context": strip_think(ans_c),
                "inject_mean":  strip_think(ans_mean),
                "inject_last":  strip_think(ans_last),
                "inject_wrong": strip_think(ans_wrong),
                "c_correct":     correct(ans_c, gold),
                "mean_correct":  correct(ans_mean, gold),
                "last_correct":  correct(ans_last, gold),
                "wrong_correct": correct(ans_wrong, gold),
            }
            records.append(rec)
            print(f"[{doc['name']}] {q}")
            print(f"   gold={gold!r}  C={rec['c_correct']}  MEAN={rec['mean_correct']}  "
                  f"LAST={rec['last_correct']}  WRONG={rec['wrong_correct']}")

    # ── Aggregate on the C-fails gate ────────────────────────────────────────
    gated = [r for r in records if not r["c_correct"]]
    n = len(gated)

    def rate(key):
        return sum(r[key] for r in gated) / n if n else 0.0

    summary = {
        "total_items": len(records),
        "c_fails_gated_items": n,
        "c_baseline_correct_overall": sum(r["c_correct"] for r in records),
        "on_gated": {
            "inject_mean_correct":  round(rate("mean_correct"), 3),
            "inject_last_correct":  round(rate("last_correct"), 3),
            "inject_wrong_correct": round(rate("wrong_correct"), 3),
        },
    }

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Items: {summary['total_items']}   C-fails (gated): {n}")
    print(f"  inject MEAN  correct on gated: {summary['on_gated']['inject_mean_correct']}")
    print(f"  inject LAST  correct on gated: {summary['on_gated']['inject_last_correct']}")
    print(f"  inject WRONG correct on gated: {summary['on_gated']['inject_wrong_correct']}  (should be ~0)")
    print("=" * 60)
    print("Verdict:")
    print("  MEAN/LAST high AND WRONG ~0  → single-vector injection genuinely works")
    print("  MEAN/LAST ~ WRONG            → no real signal; 'success' was memory/leakage")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

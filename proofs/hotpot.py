"""
proofs/hotpot.py — HotpotQA (distractor) prep and the memory/answerability gate's data
layer for Proof 5.

Proof 5 asks the project's existential question on REAL multi-hop questions: does
injecting a document's latent KV beat handing the reasoner retrieved text? HotpotQA's
distractor setting is the honest arena — 2 gold paragraphs scattered among 8 distractors,
with sentence-level supporting-fact annotations that do double duty: they define the
multi-hop structure AND mark the needle positions for budget-matched sparse latent.

This module is pure data prep (CPU, no model). It:

  • concatenates each item's 10 paragraphs IN THEIR GIVEN ORDER into one document (gold
    facts scattered among distractors — the realistic, retrieval-must-work setting);
  • locates each gold supporting SENTENCE inside the concatenated document by exact
    substring match *within its own paragraph's char range* (so a sentence that happens
    to recur in a distractor can't mis-locate the needle);
  • emits prepped records the runner turns into `needle_idx` (via
    `needles.span_token_positions`) for latent_sparse and into oracle text for text_gold.

The one rule that keeps the comparison honest: the document the latent arm captures, the
text arm prefills, and the needle spans are located in are ALL the same string, built
once here. Everything downstream indexes into it.

Run `python3 proofs/hotpot.py` to prep + cache + selftest (needs `datasets`); the gold
schema is also validated offline against a hand-built mock so the span logic is checked
without a download. `python3 proofs/hotpot.py --mock` runs only the offline check.
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_PATH = "proofs/data/hotpot_prepped.json"

# HotpotQA answers that are not extractable spans — present in neither the document nor a
# gold sentence. For these the "answer ∈ doc" invariant does not apply (the model must
# infer yes/no), but they remain perfectly good gated items.
YESNO = {"yes", "no"}


# ── document construction ───────────────────────────────────────────────────────
def _paragraph_block(title, sentences):
    """One paragraph as it appears in the concatenated document: a title header line
    followed by the paragraph's sentences joined with no extra separator (HotpotQA
    sentence strings already carry their own trailing spaces, so joining verbatim keeps
    each gold sentence an exact substring of the block)."""
    return f"{title}\n" + "".join(sentences)


def build_document(context):
    """Concatenate the 10 paragraphs in given order. Returns (doc_text, para_spans)
    where para_spans[p] = (block_char_start, sentences_char_start) — the offset of the
    paragraph block and the offset where its sentences (past the title line) begin, both
    into `doc_text`. The sentences offset is what gold-sentence location searches from."""
    titles, sents = context["title"], context["sentences"]
    blocks, para_spans, cursor = [], [], 0
    for p, (title, para) in enumerate(zip(titles, sents)):
        block = _paragraph_block(title, para)
        sent_start = cursor + len(f"{title}\n")
        para_spans.append((cursor, sent_start))
        blocks.append(block)
        cursor += len(block) + 2          # the "\n\n" paragraph separator
    return "\n\n".join(blocks), para_spans


def gold_sentences(example, doc_text, para_spans):
    """Resolve each (title, sent_id) supporting fact to its sentence text and locate it
    in `doc_text`, searching only inside the owning paragraph's char range so a sentence
    repeated in a distractor cannot capture the needle. Returns a list of
    {text, char_start, char_end, title, sent_id}; silently drops a fact whose sent_id is
    out of range or whose text cannot be found (the caller logs the drop)."""
    titles, sents = example["context"]["title"], example["context"]["sentences"]
    title_to_p = {t: p for p, t in enumerate(titles)}     # first paragraph per title
    out = []
    sf = example["supporting_facts"]
    for title, sid in zip(sf["title"], sf["sent_id"]):
        p = title_to_p.get(title)
        if p is None or sid < 0 or sid >= len(sents[p]):
            continue
        text = sents[p][sid]
        if not text.strip():
            continue
        para_start, sent_start = para_spans[p]
        para_end = para_start + len(_paragraph_block(titles[p], sents[p]))
        idx = doc_text.find(text, sent_start, para_end)
        if idx < 0:
            continue
        out.append({"text": text, "char_start": idx, "char_end": idx + len(text),
                    "title": title, "sent_id": int(sid)})
    return out


def prep_example(example, min_gold=2):
    """Turn one raw HotpotQA item into a prepped record, or None if it cannot supply at
    least `min_gold` locatable gold sentences (a multi-hop item must have ≥2 supporting
    facts; an item we can't fully locate is dropped rather than silently half-needled).
    `min_gold=1` is used by the synthetic single-hop parity control, which is genuinely
    one-sentence by design."""
    doc_text, para_spans = build_document(example["context"])
    gs = gold_sentences(example, doc_text, para_spans)
    if len(gs) < min_gold:
        return None
    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "type": example.get("type", ""),       # bridge | comparison
        "level": example.get("level", ""),
        "doc_text": doc_text,
        "gold_sentences": gs,
    }


# ── prep driver (streams the dev distractor split) ───────────────────────────────
def prep(max_items=None, cache_path=CACHE_PATH, force=False):
    """Build (or load) the prepped HotpotQA records. Streams the dev distractor split,
    keeps items with ≥2 locatable gold sentences, caps at `max_items`, caches to JSON."""
    if os.path.exists(cache_path) and not force:
        with open(cache_path) as f:
            recs = json.load(f)
        if max_items is None or len(recs) >= max_items:
            return recs[:max_items] if max_items else recs

    from datasets import load_dataset
    print("Loading hotpot_qa/distractor validation split …")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True,
                      trust_remote_code=True)
    recs, seen, dropped = [], 0, 0
    for ex in ds:
        seen += 1
        rec = prep_example(ex)
        if rec is None:
            dropped += 1
            continue
        recs.append(rec)
        if max_items and len(recs) >= max_items:
            break
    print(f"  prepped {len(recs)} items (saw {seen}, dropped {dropped} un-locatable)")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(recs, f)
    print(f"  cached → {cache_path}")
    return recs


def load_prepped(cache_path=CACHE_PATH):
    with open(cache_path) as f:
        return json.load(f)


# ── selftests ────────────────────────────────────────────────────────────────────
def _check_record(rec):
    """Pure-string invariants on one prepped record."""
    doc = rec["doc_text"]
    assert len(rec["gold_sentences"]) >= 2, f"{rec['id']}: <2 gold sentences"
    for g in rec["gold_sentences"]:
        assert doc[g["char_start"]:g["char_end"]] == g["text"], \
            f"{rec['id']}: gold span does not match recorded offsets"
        assert g["text"] in doc, f"{rec['id']}: gold sentence not in doc"
    ans = rec["answer"].strip().lower()
    if ans not in YESNO:
        # extractable answers should appear somewhere in the document (else A could
        # never succeed and the item would just be A-unanswerable noise)
        assert rec["answer"].lower() in doc.lower(), \
            f"{rec['id']}: extractable answer {rec['answer']!r} absent from doc"
    return True


def selftest_mock():
    """Validate document construction + gold-sentence location WITHOUT a download, on a
    hand-built example mimicking the HF hotpot_qa schema (two gold paras, distractors,
    and a decoy sentence repeated in a distractor to exercise the per-paragraph search)."""
    repeated = "It was founded in 1850. "
    example = {
        "id": "mock-1",
        "question": "Who founded the school that opened in 1850?",
        "answer": "Mara Quill",
        "type": "bridge", "level": "hard",
        "supporting_facts": {"title": ["Ashford School", "Mara Quill"],
                             "sent_id": [1, 0]},
        "context": {
            "title": ["Ashford School", "Mara Quill", "Distractor One", "Distractor Two"],
            "sentences": [
                ["Ashford School is in Vale. ", repeated],
                ["Mara Quill was an educator. ", "She lived in Vale. "],
                ["Some other place exists. ", repeated],          # repeats gold sentence
                ["Nothing relevant here. "],
            ],
        },
    }
    rec = prep_example(example)
    assert rec is not None, "mock item dropped unexpectedly"
    # the repeated gold sentence must be located in Ashford School (p0), not Distractor One
    g0 = next(g for g in rec["gold_sentences"] if g["text"] == repeated)
    p0_block = _paragraph_block("Ashford School", example["context"]["sentences"][0])
    doc = rec["doc_text"]
    assert g0["char_start"] < doc.find(p0_block) + len(p0_block), \
        "repeated gold sentence mis-located into a distractor paragraph"
    _check_record(rec)
    print("selftest_mock: OK (document build + per-paragraph gold location)")
    return True


def selftest_hotpot(n=50):
    """Run the string invariants over the first `n` prepped real items."""
    recs = prep(max_items=n)
    for rec in recs:
        _check_record(rec)
    yesno = sum(r["answer"].strip().lower() in YESNO for r in recs)
    print(f"selftest_hotpot: OK on {len(recs)} items ({yesno} yes/no, "
          f"{len(recs) - yesno} extractable)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="run only the offline mock selftest (no download)")
    ap.add_argument("--max-items", type=int, default=500,
                    help="cap prepped items (oversample for the gate's attrition)")
    ap.add_argument("--force", action="store_true", help="rebuild the cache")
    ap.add_argument("--selftest-n", type=int, default=50)
    args = ap.parse_args()

    selftest_mock()
    if args.mock:
        return
    prep(max_items=args.max_items, force=args.force)
    selftest_hotpot(n=args.selftest_n)


if __name__ == "__main__":
    main()

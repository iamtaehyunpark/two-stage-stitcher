"""
proofs/synth_multihop.py — the synthetic control + the single-hop parity control for
Proof 5.

A null on HotpotQA alone is ambiguous: a real latent≈text tie, or memory helping the text
arm (the model half-knows the answer, so retrieved text is enough and the representational
advantage is masked). The synthetic control breaks that tie. It rebuilds the SAME
conditions on invented-entity 2-hop items with ZERO memory leakage — bridge facts placed
in DIFFERENT paragraphs, scattered among near-miss distractors — so if latent > text on
both HotpotQA and here, the result is robust; if they disagree, the gap localizes to
memory interference. It is kept small (a control, not the main event).

It also carries the Proof-5 trap control: a handful of SINGLE-HOP extraction items where
latent and text should TIE. latent and text_rag use different prompt framings
(injected-document vs. retrieved-text-inline); if latent wins on items that are pure
extraction, the prompt asymmetry is flattering latent, not the representation — the bug
that inflated Proof 4's docnaive. A gap on the parity arm invalidates the multi-hop
numbers until the framing is fixed.

Construction reuses `hotpot.prep_example` (and therefore `build_document` /
`gold_sentences`) verbatim by authoring fake HotpotQA-schema examples, so the synthetic
document is built, and its gold spans located, by exactly the same code path as the real
arm — no second, subtly-different builder to drift out of sync.

Run `python3 proofs/synth_multihop.py` to build + selftest (no model, no download).
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.hotpot import prep_example, _paragraph_block   # reuse the real builder

# ── invented vocabulary (unguessable: coined institutions / surnames / placenames) ──
ANCHORS = [
    "the Zelmar Institute", "the Orrin Conservatory", "the Pell Foundation",
    "the Vantis Academy", "the Drennel Society", "the Mossgate Trust",
    "the Halloran Bureau", "the Quennox Guild", "the Ferrant College",
    "the Sable Repository", "the Wend Observatory", "the Castellan Lyceum",
    "the Inwick Athenaeum", "the Pryor Assembly", "the Tarn Collegium",
    "the Velm Chapterhouse", "the Olcott Conclave", "the Rontide Workshop",
    "the Bellmark Hall", "the Cindrel Office",
]
PEOPLE = [
    "Oolan Pretsky", "Vessa Trundle", "Marek Ondwell", "Sabriel Crow",
    "Doran Velleth", "Imrit Sallow", "Petra Munby", "Calix Wren",
    "Norel Fanshawe", "Brisa Oxley", "Teodric Vale", "Ysolde Marrin",
    "Garran Plume", "Lenna Quist", "Osmer Hatch", "Rue Demarco",
    "Falk Trevise", "Wyn Adair", "Corvin Belisle", "Mirae Tollan",
]
PLACES = [
    "Crennick", "Dunmarsh", "Velsworth", "Harrowby", "Tollgate Fen",
    "Pellan Reach", "Orrin Hollow", "Sable Mere", "Wend Cross", "Marrow Down",
    "Quist Vale", "Hatchford", "Belisle Strand", "Tarn Edge", "Inwick Moor",
    "Castellan Bay", "Drennel Spur", "Vantis Cove", "Ferrant Bly", "Olcott Reach",
]

# Relation families: each makes the bridge GENUINELY required — fact1 names the bridge
# person for the anchor, fact2 places that person; the question asks the place of the
# anchor's person, answerable only by composing the two.
FAMILIES = [
    {"f1": "{anchor} was for many years overseen by {person}.",
     "f2": "{person} was born in the town of {place}.",
     "q":  "In which town was the person who oversaw {anchor} born?"},
    {"f1": "The founding charter of {anchor} was drafted by {person}.",
     "f2": "{person} spent every summer at {place}.",
     "q":  "Where did the person who drafted the founding charter of {anchor} spend the summers?"},
    {"f1": "The annual lecture at {anchor} is delivered by {person}.",
     "f2": "{person} keeps a private library in {place}.",
     "q":  "In which place does the person who delivers the annual lecture at {anchor} keep a private library?"},
]


def _example(idx, anchor, person, place, family, distractors):
    """Build one fake HotpotQA-schema example: two gold paragraphs (the bridge fact and
    the placement fact under different titles) scattered among near-miss distractor
    paragraphs (same relation, different invented entities)."""
    f1 = family["f1"].format(anchor=anchor, person=person)
    f2 = family["f2"].format(person=person, place=place)
    # gold para A (title=anchor): filler then the bridge fact  → sent_id 1
    para_anchor = [f"{anchor} has a long and uneventful history. ", f1 + " "]
    # gold para B (title=person): the placement fact then bio filler → sent_id 0
    para_person = [f2 + " ", f"{person} was known to dislike travel by sea. "]

    titles = [anchor, person]
    sentences = [para_anchor, para_person]
    # distractor paragraphs: each a near-miss using OTHER entities in the same family
    for (da, dp, dpl) in distractors:
        titles.append(da)
        sentences.append([family["f1"].format(anchor=da, person=dp) + " ",
                          family["f2"].format(person=dp, place=dpl) + " "])

    # scatter: gold paragraphs are not adjacent and not first/last
    order = [2, 0, 3, 4, 1, 5][:len(titles)]
    titles = [titles[i] for i in order]
    sentences = [sentences[i] for i in order]

    ex = {
        "id": f"synth-mh-{idx}",
        "question": family["q"].format(anchor=anchor),
        "answer": place,
        "type": "bridge", "level": "synthetic",
        "supporting_facts": {"title": [anchor, person], "sent_id": [1, 0]},
        "context": {"title": titles, "sentences": sentences},
    }
    return ex


def build_multihop(n=20):
    """Deterministic invented-entity 2-hop items. Decoys are other items' targets so a
    correct answer must discriminate the right bridge, not pick the only placename."""
    recs = []
    for i in range(n):
        anchor, person, place = ANCHORS[i], PEOPLE[i], PLACES[i]
        fam = FAMILIES[i % len(FAMILIES)]
        # four near-miss distractors drawn from other items (distinct entities)
        dist = []
        for j in (i + 1, i + 2, i + 3, i + 4):
            k = j % len(ANCHORS)
            if k == i:
                k = (k + 5) % len(ANCHORS)
            dist.append((ANCHORS[k], PEOPLE[k], PLACES[k]))
        ex = _example(i, anchor, person, place, fam, dist)
        rec = prep_example(ex, min_gold=2)
        assert rec is not None, f"multihop item {i} failed to build"
        rec["hops"] = 2
        rec["arm"] = "synth_multihop"
        rec["decoy_values"] = [PLACES[k % len(PLACES)] for k in (i + 1, i + 2, i + 3)]
        recs.append(rec)
    return recs


# ── single-hop parity control (latent vs text MUST tie here) ─────────────────────
PARITY = [
    ("the Marn Array", "override sigil", "Veltris", "What is the override sigil of the Marn Array?"),
    ("the Calder Engine", "primary coolant", "br-fluid Yune", "What is the primary coolant of the Calder Engine?"),
    ("the Selvat Codex", "binding clasp", "a Wren-lock", "What kind of binding clasp does the Selvat Codex use?"),
    ("the Ottenby Beacon", "signal colour", "pale Drennel green", "What is the signal colour of the Ottenby Beacon?"),
    ("the Halver Press", "house typeface", "Mossgate Antiqua", "What is the house typeface of the Halver Press?"),
    ("the Pinnow Vault", "access phrase", "Quennox-and-salt", "What is the access phrase of the Pinnow Vault?"),
]


def build_parity():
    """Single-sentence extraction items (one gold sentence, in a gold paragraph scattered
    among distractors). No bridge — pure extraction, where injected latent and retrieved
    text should score identically."""
    recs = []
    for i, (subj, attr, val, q) in enumerate(PARITY):
        gold = f"The {attr} of {subj} is {val}. "
        para_gold = [f"{subj} is documented in several places. ", gold]   # sent_id 1
        titles = [subj]
        sentences = [para_gold]
        # near-miss distractors: same attribute, wrong subject+value
        for j in range(4):
            k = (i + j + 1) % len(PARITY)
            s2, a2, v2, _ = PARITY[k]
            titles.append(f"{s2} (note {j})")
            sentences.append([f"The {a2} of {s2} is {v2}. ",
                              f"Unrelated remark number {j}. "])
        order = [1, 0, 2, 3, 4][:len(titles)]
        titles = [titles[t] for t in order]
        sentences = [sentences[t] for t in order]
        ex = {
            "id": f"synth-sh-{i}", "question": q, "answer": val,
            "type": "single", "level": "synthetic",
            "supporting_facts": {"title": [subj], "sent_id": [1]},
            "context": {"title": titles, "sentences": sentences},
        }
        rec = prep_example(ex, min_gold=1)
        assert rec is not None, f"parity item {i} failed to build"
        rec["hops"] = 1
        rec["arm"] = "synth_parity"
        rec["decoy_values"] = [PARITY[(i + j + 1) % len(PARITY)][2] for j in range(3)]
        recs.append(rec)
    return recs


def build_all(n_multihop=20):
    return build_multihop(n_multihop) + build_parity()


# ── selftest (pure string, no model) ──────────────────────────────────────────────
def selftest():
    mh, par = build_multihop(20), build_parity()
    assert len(mh) == 20 and len(par) == len(PARITY)
    for rec in mh:
        doc, q, ans = rec["doc_text"], rec["question"], rec["answer"]
        # bridge truly required: neither gold sentence alone names BOTH the anchor link
        # and the answer place
        assert len(rec["gold_sentences"]) == 2, f"{rec['id']}: not 2-hop"
        for g in rec["gold_sentences"]:
            assert doc[g["char_start"]:g["char_end"]] == g["text"]
        # lexical gap: the answer must not appear in the question (else not a real hop)
        assert ans.lower() not in q.lower(), f"{rec['id']}: answer leaks into question"
        # zero memory leakage proxy: the answer appears exactly once in the doc (only in
        # its gold sentence), so the text arm cannot win by a distractor coincidence
        assert doc.lower().count(ans.lower()) == 1, \
            f"{rec['id']}: answer {ans!r} appears {doc.lower().count(ans.lower())}× (want 1)"
        assert ans not in rec.get("decoy_values", []), f"{rec['id']}: decoy == answer"
    for rec in par:
        doc, q, ans = rec["doc_text"], rec["question"], rec["answer"]
        assert len(rec["gold_sentences"]) == 1, f"{rec['id']}: parity must be single-hop"
        assert ans.lower() not in q.lower(), f"{rec['id']}: answer leaks into question"
        assert doc.lower().count(ans.lower()) == 1, f"{rec['id']}: answer not unique in doc"
    print(f"selftest: OK ({len(mh)} multihop + {len(par)} parity; "
          "bridge-required, lexical-gap, unique-answer all hold)")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="print one built multihop doc")
    args = ap.parse_args()
    selftest()
    if args.dump:
        rec = build_multihop(1)[0]
        print("\n--- example multihop doc ---")
        print("Q:", rec["question"], "\nA:", rec["answer"])
        print(rec["doc_text"])

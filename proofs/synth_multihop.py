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
# Compositional vocab so the control can scale past n=30 (the verdict's power floor) while
# staying unique per item — the answer (a PLACE) must appear exactly once per doc, so we
# need many distinct invented places. Two-part composition gives 60–110 distinct tokens
# each from short stem lists; `selftest` asserts distinctness and the per-doc invariants.
_INST_STEM = ["Zelmar", "Orrin", "Pell", "Vantis", "Drennel", "Mossgate", "Halloran",
              "Quennox", "Ferrant", "Sable", "Wend", "Castellan", "Inwick", "Pryor",
              "Tarn", "Velm", "Olcott", "Rontide", "Bellmark", "Cindrel"]
_INST_KIND = ["Institute", "Conservatory", "Foundation", "Society"]
ANCHORS = [f"the {s} {k}" for k in _INST_KIND for s in _INST_STEM]            # 80

_FIRST = ["Oolan", "Vessa", "Marek", "Sabriel", "Doran", "Imrit", "Petra", "Calix",
          "Norel", "Brisa", "Teodric", "Ysolde", "Garran", "Lenna", "Osmer", "Rue"]
_LAST = ["Pretsky", "Trundle", "Ondwell", "Crow", "Velleth", "Sallow"]
PEOPLE = [f"{f} {l}" for l in _LAST for f in _FIRST]                          # 96

_PLACE_A = ["Cren", "Dun", "Vels", "Harrow", "Toll", "Pellan", "Orrin", "Sable",
            "Wend", "Marrow", "Quist", "Hatch", "Belisle", "Tarn", "Inwick", "Castel"]
_PLACE_B = ["nick", "marsh", "worth", "by", "gate", " holt", "mere"]
PLACES = [f"{a}{b}" for b in _PLACE_B for a in _PLACE_A]                      # 112

# Relation families: each makes the bridge GENUINELY required — fact1 names the bridge
# person for the anchor, fact2 places that person; the question asks the place of the
# anchor's person, answerable only by composing the two.
FAMILIES = [
    {"f1": "{anchor} was for many years overseen by {person}.",
     "f2": "{person} was born in the town of {place}.",
     "q":  "In which town was the person who oversaw {anchor} born?",
     "q1": "Who oversaw {anchor} for many years?"},
    {"f1": "The founding charter of {anchor} was drafted by {person}.",
     "f2": "{person} spent every summer at {place}.",
     "q":  "Where did the person who drafted the founding charter of {anchor} spend the summers?",
     "q1": "Who drafted the founding charter of {anchor}?"},
    {"f1": "The annual lecture at {anchor} is delivered by {person}.",
     "f2": "{person} keeps a private library in {place}.",
     "q":  "In which place does the person who delivers the annual lecture at {anchor} keep a private library?",
     "q1": "Who delivers the annual lecture at {anchor}?"},
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
# This is the FIRST HOP of a multihop item (anchor → person), asked directly. The answer is
# a COINED person the model cannot confabulate from priors, so it is forced to read the
# injection — isolating single-hop extraction from multi-hop binding. The earlier framing
# ("the primary coolant of X is …") invited world-knowledge confabulation ("coolant →
# water") that the *sparse* injection couldn't override, which conflated prior-strength with
# the extraction test and made latent_sparse fail for the wrong reason. Reusing the multihop
# entities keeps the no-prior property identical across the two arms.
def build_parity(n=32):
    recs = []
    for i in range(n):
        anchor, person = ANCHORS[i], PEOPLE[i]
        fam = FAMILIES[i % len(FAMILIES)]
        f1 = fam["f1"].format(anchor=anchor, person=person)
        q = fam["q1"].format(anchor=anchor)
        para_gold = [f"{anchor} has a long and well-documented history. ", f1 + " "]  # sent_id 1
        titles, sentences = [anchor], [para_gold]
        # near-miss distractors: same relation, other anchors → other (coined) persons
        for j in range(4):
            k = (i + j + 1) % len(ANCHORS)
            if k == i:
                k = (k + 5) % len(ANCHORS)
            titles.append(ANCHORS[k])
            sentences.append([fam["f1"].format(anchor=ANCHORS[k], person=PEOPLE[k]) + " ",
                              f"{ANCHORS[k]} keeps no other records of note. "])
        order = [1, 0, 2, 3, 4][:len(titles)]
        titles = [titles[t] for t in order]
        sentences = [sentences[t] for t in order]
        ex = {
            "id": f"synth-sh-{i}", "question": q, "answer": person,
            "type": "single", "level": "synthetic",
            "supporting_facts": {"title": [anchor], "sent_id": [1]},
            "context": {"title": titles, "sentences": sentences},
        }
        rec = prep_example(ex, min_gold=1)
        assert rec is not None, f"parity item {i} failed to build"
        rec["hops"] = 1
        rec["arm"] = "synth_parity"
        rec["decoy_values"] = [PEOPLE[(i + j + 1) % len(PEOPLE)] for j in range(3)]
        recs.append(rec)
    return recs


def build_all(n_multihop=40, n_parity=32):
    return build_multihop(n_multihop) + build_parity(n_parity)


# ── selftest (pure string, no model) ──────────────────────────────────────────────
def selftest():
    # vocab must be distinct so per-doc answer uniqueness holds at scale
    for name, pool in [("ANCHORS", ANCHORS), ("PEOPLE", PEOPLE), ("PLACES", PLACES)]:
        assert len(set(pool)) == len(pool), f"{name} has duplicates"
    assert len(PLACES) >= 40 and len(PEOPLE) >= 32, "not enough unique answers to scale"
    mh, par = build_multihop(40), build_parity(32)
    assert len(mh) == 40 and len(par) == 32
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
        # coined answer (a person) the model can't confabulate, absent from the question,
        # unique in the doc, and not equal to a decoy — a clean no-prior extraction control
        assert ans.lower() not in q.lower(), f"{rec['id']}: answer leaks into question"
        assert doc.lower().count(ans.lower()) == 1, f"{rec['id']}: answer not unique in doc"
        assert ans not in rec.get("decoy_values", []), f"{rec['id']}: decoy == answer"
        g = rec["gold_sentences"][0]
        assert ans in doc[g["char_start"]:g["char_end"]], \
            f"{rec['id']}: needle span must contain the answer (else sparse can't transfer it)"
    print(f"selftest: OK ({len(mh)} multihop + {len(par)} parity; "
          "no-prior coined answers, lexical-gap, unique-answer, needle-covers-answer all hold)")
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

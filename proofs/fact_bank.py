"""
proofs/fact_bank.py — the expanded, adversarially-authored synthetic fact bank.

Proofs 1–4 ran on 25 facts in 5 hand-authored documents. That is enough to *resolve*
within-fact comparisons (qfair vs A, latent vs text on the SAME fact), but it is not
enough to support a GENERALIZATION claim — and Proof 5 ("latent handoff beats text
RAG") is exactly that. Five facts measured at six depths is still five facts: if those
happen to be easy (distinctive proper nouns, no near-synonyms), every depth inherits the
easiness and "n=30" is really "n=5 measured 6 ways" — tight error bars around a possibly
biased point. So this bank exists to give Proofs 4.1 / 5 / 6 enough INDEPENDENT facts
that the central claim is not an anecdote.

It is authored against the failure modes the chain already taught us:

  • Varied fact type — not all proper-noun substitution. Each doc mixes `name`, `date`,
    `number` (often spelled-out, which stresses strict scoring), `multitoken` coined
    terms, `relation`, and `common_word` answers (a common word in an unusual role,
    where substring luck breaks).
  • Native distractors — every doc carries near-miss decoys for its OWN facts (same
    surface form, WRONG value), so the C_filler / A gates and the
    distractor-discrimination requirement are intrinsic, not bolted on. `decoy_values`
    lists the wrong tokens the strict scorer must rule out.
  • Lexical gap — the question is phrased to AVOID reusing the needle's answer-bearing
    words, so the model must bind meaning, not match strings. (`selftest_bank` checks
    the answer never appears in the question and the question is not a verbatim slice of
    the needle.)
  • Coreference subset — `kind="coref"` docs (imported from `synthetic_docs_long`) put
    the answer in the DECIMATABLE surroundings and refer to it by anaphora in the
    needle. Without these, dec_latent and dec_text tie at 1.0 forever and the latent>text
    mechanism (Exp 3.1, the project's reason to exist) can never show itself. These get
    native distractors here too, so the latent-vs-text arm runs UNDER distractors.

Schema (per doc):
    {name, kind: "span"|"coref", text, distractors: [str], decoy_values: [str],
     qa: [{q, a, needle, type}]}
  span : answer is IN the needle clause (answer-in-span).
  coref: answer is in the surroundings; the needle refers to it by anaphora
         (answer NOT in needle) — the Exp-3.1 design.

Run `python proofs/fact_bank.py` after any edit: the selftest enforces every invariant
with pure string checks (no model), so authoring mistakes are caught before GPU time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.synthetic_docs_long import SYNTHETIC_DOCS_LONG, doc_by_name as _long_by_name  # noqa: E402


FACT_TYPES = {"name", "date", "number", "multitoken", "relation", "common_word"}


# ════════════════════════════════════════════════════════════════════════════════
# SPAN DOCUMENTS — answer is in the needle clause. The discrimination arm of Proof 4.1
# and the bulk of Proofs 1–3's premise live here.
# ════════════════════════════════════════════════════════════════════════════════
SPAN_DOCS = [
    {
        "name": "verdant_ledger", "kind": "span",
        "text": (
            "The Verdant Ledger is a merchant register first opened in 1503 in the "
            "trading port of Hesperance by a wool-factor named Oswin Brake. For nearly a "
            "century it recorded every cargo that cleared the port's customs house. The "
            "book is bound in green calf, which gave it the name it still carries, and "
            "its pages run to one thousand one hundred and forty-four numbered folios. "
            "After its founder died the work was carried on by the clerk Mathilde Brake, "
            "his granddaughter, whose index of shipmasters is the volume's most consulted "
            "section. The single most cited entry concerns a shipment of dye called "
            "crozier blue, the trade in which made the port briefly rich. The original "
            "is kept today in the muniment room at Hesperance under the shelf-mark "
            "V-twelve."
        ),
        "distractors": [
            "Some catalogues insist the register was first opened in 1487 by a "
            "salt-factor named Edrin Vole.",
            "A competing note holds that the book runs to two thousand and eight folios "
            "in all.",
            "According to one disputed account the work was carried on after its founder "
            "by a clerk named Sabine Orr.",
            "It is occasionally claimed the ledger's famous entry concerns a dye called "
            "wyvern green.",
            "An old guide places the volume in the muniment room at Calderon rather than "
            "Hesperance.",
        ],
        "decoy_values": ["Edrin Vole", "1487", "two thousand and eight", "Sabine Orr",
                         "wyvern green", "Calderon"],
        "qa": [
            {"q": "Who first opened the green-bound register at the port?",
             "a": "Oswin Brake", "type": "name",
             "needle": "by a wool-factor named Oswin Brake"},
            {"q": "In what year did the register begin?", "a": "1503", "type": "date",
             "needle": "first opened in 1503"},
            {"q": "How many leaves are numbered in the book?",
             "a": "one thousand one hundred and forty-four", "alt": ["1,144", "1144"], "type": "number",
             "needle": "run to one thousand one hundred and forty-four numbered folios"},
            {"q": "Who carried on the register after its founder died?",
             "a": "Mathilde Brake", "type": "relation",
             "needle": "carried on by the clerk Mathilde Brake, his granddaughter"},
            {"q": "Which dye is named in the volume's most consulted entry?",
             "a": "crozier blue", "type": "common_word",
             "needle": "a shipment of dye called crozier blue"},
        ],
    },
    {
        "name": "pellan_lighthouse", "kind": "span",
        "text": (
            "The Pellan Lighthouse marks the reef at the mouth of the Sorrel estuary. It "
            "was raised in 1729 at the order of a harbour-master named Crispin Dowe, "
            "after a winter in which several grain ships broke on the rocks. The tower "
            "stands ninety-eight feet from its base to the lamp, and for its first "
            "century it burned a whale-oil flame thrown out by a ring of polished "
            "mirrors that the keepers called the Marl array. Its most celebrated keeper "
            "was a woman named Aurelia Fenn, who held the light for forty-one years "
            "without letting a single night go dark. Since the lamp was automated the "
            "station has been looked after by the trust at Sorrel Haven."
        ),
        "distractors": [
            "Some records claim the tower was raised in 1701 at the order of a pilot "
            "named Edmund Crale.",
            "A competing survey gives the height from base to lamp as one hundred and "
            "forty feet.",
            "It is sometimes said the ring of mirrors was known instead as the Ferris "
            "array.",
            "According to one disputed log the longest-serving keeper was a man named "
            "Tobias Reeve.",
            "An old chart sets the lighthouse at the mouth of the Calder estuary.",
        ],
        "decoy_values": ["Edmund Crale", "1701", "one hundred and forty", "Ferris array",
                         "Tobias Reeve", "Calder"],
        "qa": [
            {"q": "On whose orders was the lighthouse raised?", "a": "Crispin Dowe",
             "type": "name", "needle": "at the order of a harbour-master named Crispin Dowe"},
            {"q": "In what year was the tower built?", "a": "1729", "type": "date",
             "needle": "It was raised in 1729"},
            {"q": "How tall is the structure from its base to the lamp?",
             "a": "ninety-eight feet", "alt": ["98 feet", "98 ft"], "type": "number",
             "needle": "stands ninety-eight feet from its base to the lamp"},
            {"q": "What name did the keepers give the ring of mirrors?", "a": "Marl array",
             "type": "multitoken", "needle": "the keepers called the Marl array"},
            {"q": "Which keeper held the light for over four decades?", "a": "Aurelia Fenn",
             "type": "name",
             "needle": "a woman named Aurelia Fenn, who held the light for forty-one years"},
        ],
    },
    {
        "name": "mossgrove_vineyard", "kind": "span",
        "text": (
            "Mossgrove is the oldest working vineyard on the Adran slopes, planted in "
            "1684 by a dispossessed weaver named Hester Crowe, who turned to vines after "
            "the cloth trade failed. The estate is best known for a single grape, a "
            "thick-skinned black variety called the friar's mark, which ripens late and "
            "survives the frosts that ruin its neighbours. In a good year the slopes "
            "yield seventy-two casks, no more, and the wine is sold only by subscription. "
            "The cellars were cut into the hillside by Crowe's son-in-law, the mason "
            "Pelegrin Ashe, whose vaulted galleries keep an even cool through every "
            "season. The estate has stayed in the same family for eleven generations and "
            "is run today from the old press-house at Adran Cross."
        ),
        "distractors": [
            "Some almanacs claim the vineyard was planted in 1651 by a dyer named "
            "Maris Vane.",
            "A competing record names its signature grape the abbot's seal.",
            "It is occasionally said the slopes yield as many as one hundred and ten "
            "casks in a strong year.",
            "According to one disputed deed the cellars were dug by a mason named "
            "Corvin Slate.",
            "An old survey reports the estate has passed through seven generations.",
        ],
        "decoy_values": ["Maris Vane", "1651", "abbot's seal", "one hundred and ten",
                         "Corvin Slate", "seven generations"],
        "qa": [
            {"q": "Who established the vineyard on the Adran slopes?", "a": "Hester Crowe",
             "type": "name", "needle": "planted in 1684 by a dispossessed weaver named Hester Crowe"},
            {"q": "In what year were the vines first set?", "a": "1684", "type": "date",
             "needle": "planted in 1684"},
            {"q": "What is the estate's signature grape called?", "a": "friar's mark",
             "type": "common_word", "needle": "a thick-skinned black variety called the friar's mark"},
            {"q": "How many casks does a good harvest produce?", "a": "seventy-two", "alt": ["72"],
             "type": "number", "needle": "the slopes yield seventy-two casks"},
            {"q": "Who dug the hillside cellars?", "a": "Pelegrin Ashe", "type": "relation",
             "needle": "cut into the hillside by Crowe's son-in-law, the mason Pelegrin Ashe"},
        ],
    },
    {
        "name": "tarn_aqueduct", "kind": "span",
        "text": (
            "The Tarn Aqueduct carries water down from the high lakes to the dry city of "
            "Vesh, a distance the engineers of the day thought impossible. It was "
            "completed in 1812 to a design by the hydraulic engineer Lucinda Marr, the "
            "first woman admitted to the Vesh college of works. Its boldest stretch is a "
            "tiered stone bridge of forty-one arches across the Orsin gorge. To stop the "
            "channel silting, Marr lined it with a glazed tile of her own recipe that the "
            "masons nicknamed the kingfisher glaze for its blue sheen. The works were "
            "paid for by a salt magnate named Tobias Venn, whose name the city pointedly "
            "left off every plaque. Water has run along it without a single full stoppage "
            "for over two hundred years."
        ),
        "distractors": [
            "Some chronicles date the aqueduct's completion to 1788 and credit its design "
            "to an engineer named Halvard Stane.",
            "A rival account gives the great bridge sixty-three arches.",
            "It is sometimes claimed the channel lining was known as the heron glaze.",
            "According to one disputed ledger the works were funded by a timber magnate "
            "named Doran Frey.",
            "An old plaque names the lead engineer as a man called Edric Pollard.",
        ],
        "decoy_values": ["1788", "Halvard Stane", "sixty-three", "heron glaze",
                         "Doran Frey", "Edric Pollard"],
        "qa": [
            {"q": "Who designed the aqueduct that supplies the dry city?",
             "a": "Lucinda Marr", "type": "name",
             "needle": "to a design by the hydraulic engineer Lucinda Marr"},
            {"q": "When was the water-channel finished?", "a": "1812", "type": "date",
             "needle": "completed in 1812"},
            {"q": "How many arches span the gorge on its boldest stretch?",
             "a": "forty-one", "alt": ["41"], "type": "number",
             "needle": "a tiered stone bridge of forty-one arches across the Orsin gorge"},
            {"q": "What did the masons nickname the channel's lining tile?",
             "a": "kingfisher glaze", "type": "multitoken",
             "needle": "the masons nicknamed the kingfisher glaze for its blue sheen"},
            {"q": "Who paid for the construction?", "a": "Tobias Venn", "type": "relation",
             "needle": "paid for by a salt magnate named Tobias Venn"},
        ],
    },
    {
        "name": "halvern_mint", "kind": "span",
        "text": (
            "The Halvern Mint struck the coinage of the river states for the better part "
            "of two centuries. It was chartered in 1571 under a master-moneyer named "
            "Idris Calloway, who had learned the trade abroad and returned with a press "
            "no rival could match. The mint's reputation rested on a tamper-proof milled "
            "edge, a technique Calloway guarded jealously and the guild recorded only as "
            "the wolf's tooth. At its height the works employed three hundred and sixteen "
            "hands and struck coin through the night. Its downfall came in 1744, when a "
            "cache of false dies was traced to a foreman named Garric Stoll, and the "
            "charter was revoked within the year. The surviving press is displayed today "
            "at the civic hall in Halvern."
        ),
        "distractors": [
            "Some histories say the mint was chartered in 1602 under a master-moneyer "
            "named Pelham Roe.",
            "A competing account calls the milled-edge technique the lion's claw.",
            "It is occasionally claimed the works employed four hundred and ninety hands "
            "at their height.",
            "According to one disputed record the false dies were traced to a foreman "
            "named Wymar Teld.",
            "An old chronicle dates the loss of the charter to 1719.",
        ],
        "decoy_values": ["1602", "Pelham Roe", "lion's claw", "four hundred and ninety",
                         "Wymar Teld", "1719"],
        "qa": [
            {"q": "Under which master-moneyer was the mint chartered?", "a": "Idris Calloway",
             "type": "name", "needle": "under a master-moneyer named Idris Calloway"},
            {"q": "In what year did the mint receive its charter?", "a": "1571", "type": "date",
             "needle": "It was chartered in 1571"},
            {"q": "What did the guild call the secret milled-edge technique?",
             "a": "wolf's tooth", "type": "common_word",
             "needle": "the guild recorded only as the wolf's tooth"},
            {"q": "How many workers did the mint employ at its peak?",
             "a": "three hundred and sixteen", "alt": ["316"], "type": "number",
             "needle": "employed three hundred and sixteen hands"},
            {"q": "Which foreman was the cache of false dies traced to?", "a": "Garric Stoll",
             "type": "relation", "needle": "a cache of false dies was traced to a foreman named Garric Stoll"},
        ],
    },
    {
        "name": "brae_funicular", "kind": "span",
        "text": (
            "The Brae Funicular hauls passengers up the cliff between the harbour town of "
            "Lonan and the clifftop village above it. It opened in 1896, the work of an "
            "engineer named Senga Pryce, who solved the problem of the unstable cliff by "
            "counterbalancing two cars on a single cable. The line climbs a gradient of "
            "one in three, the steepest of any public railway in the region. Its cars "
            "are still drawn by the original water-balance system, in which a tank "
            "beneath the upper car is filled until its weight pulls the lower car up, an "
            "arrangement the line's guides call the tipping cradle. The funicular was "
            "saved from closure in 1969 by a preservation society and now carries more "
            "visitors in a summer than it ever did commuters. It is operated from the "
            "winding house at Lonan Head."
        ),
        "distractors": [
            "Some guidebooks date the funicular's opening to 1871 and name its engineer "
            "as a man called Roderic Vane.",
            "A competing account gives the gradient as one in five.",
            "It is sometimes said the water-balance arrangement is known as the rocking "
            "berth.",
            "According to one disputed plaque the line was rescued from closure in 1981.",
            "An old timetable credits the design to an engineer named Marda Quist.",
        ],
        "decoy_values": ["1871", "Roderic Vane", "one in five", "rocking berth", "1981",
                         "Marda Quist"],
        "qa": [
            {"q": "Who engineered the cliff railway between the town and the village?",
             "a": "Senga Pryce", "type": "name",
             "needle": "the work of an engineer named Senga Pryce"},
            {"q": "In what year did the cliff railway open?", "a": "1896", "type": "date",
             "needle": "It opened in 1896"},
            {"q": "How steep is the line at its sharpest?", "a": "one in three", "alt": ["1 in 3", "1:3"],
             "type": "number", "needle": "a gradient of one in three"},
            {"q": "What do the guides call the water-balance arrangement?",
             "a": "tipping cradle", "type": "multitoken",
             "needle": "an arrangement the line's guides call the tipping cradle"},
            {"q": "When was the line rescued from closure?", "a": "1969", "type": "date",
             "needle": "saved from closure in 1969 by a preservation society"},
        ],
    },
]


# ════════════════════════════════════════════════════════════════════════════════
# COREFERENCE DOCUMENTS — answer is in the DECIMATABLE surroundings; the needle refers
# to it by anaphora. Reused from synthetic_docs_long (vetted by its own selftest:
# answer ∉ needle), with native distractors + decoy_values attached here so the
# latent-vs-text arm runs under distractors. These are the docs that can SHOW latent >
# text; a bank of only answer-in-span facts would let the two tie at 1.0 forever.
# ════════════════════════════════════════════════════════════════════════════════
_COREF_META = {
    "kessler_observatory": {
        "distractors": [
            "Some accounts insist the observatory was founded by a lens-grinder named "
            "Corvin Thane.",
            "A rival tradition calls the great disc the Harrow lens.",
            "It is occasionally claimed the comet's orbit was fixed there in 1879.",
            "One disputed memoir names the second director as the astronomer Sela Hox.",
            "An old guide places the dome on the slopes of Mount Carrow.",
        ],
        "decoy_values": ["Corvin Thane", "Harrow lens", "1879", "Sela Hox", "Mount Carrow"],
        "types": ["name", "multitoken", "date", "name", "name"],
    },
    "harlan_bridge": {
        "distractors": [
            "Some histories credit the bridge's design to a surveyor named Doran Whitlock.",
            "A competing account calls the locking rivet the Whitlock joint.",
            "According to one disputed record, construction began in 1853.",
            "It is sometimes said the span opened to traffic in 1872.",
            "An old register holds that the designer's papers went to the school at "
            "Marrender.",
        ],
        "decoy_values": ["Doran Whitlock", "Whitlock joint", "1853", "1872", "Marrender"],
        "types": ["name", "multitoken", "date", "date", "name"],
    },
    "saltwell_press": {
        "distractors": [
            "Some chronicles say the press was founded by a chandler named Edmund Roke.",
            "A competing account credits the Tidal Atlas to a philosopher named "
            "Coralie Venn.",
            "According to one disputed catalogue the atlas first appeared in 1808.",
            "It is sometimes claimed the flexible spine was called the Marlow binding.",
            "An old guide places the surviving press in the museum at Pellmire.",
        ],
        "decoy_values": ["Edmund Roke", "Coralie Venn", "1808", "Marlow binding", "Pellmire"],
        "types": ["name", "name", "date", "multitoken", "name"],
    },
    "vantroy_engine": {
        "distractors": [
            "Some enthusiasts insist the engine was designed by a mechanic named "
            "Idris Vale.",
            "A rival account calls the low-slung arrangement the Vale truck.",
            "According to one disputed plate the locomotive first ran in 1876.",
            "It is occasionally said the mining line closed in 1923.",
            "An old timetable keeps the engine at a museum line at Corrin.",
        ],
        "decoy_values": ["Idris Vale", "Vale truck", "1876", "1923", "Corrin"],
        "types": ["name", "multitoken", "date", "date", "name"],
    },
}


def _build_coref_docs():
    docs = []
    for name, meta in _COREF_META.items():
        base = _long_by_name(name)
        types = meta["types"]
        qa = [{"q": q["q"], "a": q["a"], "needle": q["needle"],
               "type": types[i] if i < len(types) else "name"}
              for i, q in enumerate(base["qa"])]
        docs.append({"name": name, "kind": "coref", "text": base["text"],
                     "distractors": meta["distractors"],
                     "decoy_values": meta["decoy_values"], "qa": qa})
    return docs


COREF_DOCS = _build_coref_docs()
FACT_DOCS = SPAN_DOCS + COREF_DOCS


# ── accessors ───────────────────────────────────────────────────────────────────
def span_docs():
    return [d for d in FACT_DOCS if d["kind"] == "span"]


def coref_docs():
    return [d for d in FACT_DOCS if d["kind"] == "coref"]


def doc_by_name(name):
    for d in FACT_DOCS:
        if d["name"] == name:
            return d
    raise KeyError(name)


def distractors_map():
    return {d["name"]: d["distractors"] for d in FACT_DOCS}


def decoy_values_map():
    return {d["name"]: d["decoy_values"] for d in FACT_DOCS}


def n_facts(kind=None):
    return sum(len(d["qa"]) for d in FACT_DOCS if kind is None or d["kind"] == kind)


# ── static selftest (run after any edit; pure strings, no model) ────────────────
def _norm(s):
    import re
    s = s.lower().replace(",", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def selftest_bank(verbose=True):
    """Enforce every authoring invariant with pure string checks:
      1. answer ∈ text                       (A can succeed)
      2. needle ∈ text                        (span→token map works)
      3. span: answer ∈ needle / coref: answer ∉ needle
      4. answer ∉ question                    (no lexical giveaway)
      5. question is not a verbatim slice of the needle (forces a lexical gap)
      6. every distractor: no true gold inside it (it stays a near-MISS)
      7. every distractor ∉ text              (decoys live in the filler, not the doc)
      8. every decoy_value ∉ text             (else A reading the true doc hits a decoy)
      9. every decoy_value appears in ≥1 distractor (consistency)
     10. type ∈ FACT_TYPES
    A failure here means a fact is malformed — fix the prose, do not run on GPU."""
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        if verbose:
            print(f"  ✗ {msg}")

    for d in FACT_DOCS:
        name, kind, text = d["name"], d["kind"], d["text"]
        tnorm = _norm(text)
        golds = [qa["a"] for qa in d["qa"]]

        for qa in d["qa"]:
            q, a, needle, typ = qa["q"], qa["a"], qa["needle"], qa.get("type")
            if a not in text:
                fail(f"[{name}] answer {a!r} not in text")
            if needle not in text:
                fail(f"[{name}] needle {needle!r} not in text")
            if kind == "span" and a not in needle:
                fail(f"[{name}] span: answer {a!r} not in needle {needle!r}")
            if kind == "coref" and a in needle:
                fail(f"[{name}] coref: answer {a!r} IS in needle (mechanism void)")
            if _norm(a) and _norm(a) in _norm(q):
                fail(f"[{name}] answer {a!r} leaks into question {q!r}")
            if _norm(q) and _norm(q) in _norm(needle):
                fail(f"[{name}] question {q!r} is a verbatim slice of the needle")
            if typ not in FACT_TYPES:
                fail(f"[{name}] unknown fact type {typ!r}")

        for dec in d["distractors"]:
            if dec in text:
                fail(f"[{name}] distractor appears in the document text: {dec[:50]!r}")
            for g in golds:
                if _norm(g) in _norm(dec):
                    fail(f"[{name}] distractor leaks gold {g!r}: {dec[:50]!r}")

        for dv in d["decoy_values"]:
            if _norm(dv) in tnorm:
                fail(f"[{name}] decoy_value {dv!r} appears in the true text")
            if not any(_norm(dv) in _norm(dec) for dec in d["distractors"]):
                fail(f"[{name}] decoy_value {dv!r} not present in any distractor")

    if verbose:
        n_span, n_coref = n_facts("span"), n_facts("coref")
        by_type = {}
        for d in FACT_DOCS:
            for qa in d["qa"]:
                by_type[qa["type"]] = by_type.get(qa["type"], 0) + 1
        print(f"{'BANK WELL-FORMED' if ok else 'PROBLEMS FOUND'} — "
              f"{len(FACT_DOCS)} docs, {n_span} span + {n_coref} coref = "
              f"{n_span + n_coref} facts")
        print(f"  fact types: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest_bank() else 1)

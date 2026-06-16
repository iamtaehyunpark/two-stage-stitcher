"""
proofs/synthetic_docs.py — the fabricated-fact document bank.

Every entity, date, and relationship here is invented. No model has seen these
facts, so a correct answer cannot come from parametric memory — it can only come
through the injected representations. This is the move that makes Proof 1 mean
something and Proof 2's wrong-document control airtight.

Each QA item carries a `needle`: the exact answer-bearing sentence from the
document. Proof 3 ("all-N vs needles") will map these substrings to token
positions; Proofs 1–2 only need (q, a).

These documents are intentionally short (~120–180 words). Length scaling is
Proof 4's job, not these. They are the one piece of authored content the whole
chain reuses.
"""

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
            {"q": "Who recovered the Zorvian Codex?", "a": "Maren Velloth",
             "needle": "first catalogued in the year 1487 by the explorer Maren Velloth"},
            {"q": "In what year was the Zorvian Codex first catalogued?", "a": "1487",
             "needle": "first catalogued in the year 1487"},
            {"q": "How many verses does the Zorvian Codex contain?", "a": "3412",
             "needle": "contains exactly 3,412 verses"},
            {"q": "Who produced the first complete translation?", "a": "Idris Pell",
             "needle": "the scholar Idris Pell produced the first complete translation in 1923"},
            {"q": "To whom is the codex attributed?", "a": "Banu Castreth",
             "needle": "attributed the work to the philosopher Banu Castreth"},
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
            {"q": "Who designed the Harnel rotary engine?", "a": "Lucia Pendran",
             "needle": "designed in 1956 by the engineer Lucia Pendran"},
            {"q": "How much horsepower did the Harnel engine produce?", "a": "1840",
             "needle": "It produced 1,840 horsepower"},
            {"q": "What fuel did the Harnel engine run on?", "a": "blue naphtha",
             "needle": "ran on a fuel mixture known as blue naphtha"},
            {"q": "What was Pendran's patented compression arrangement called?", "a": "Vossler ring",
             "needle": "patented under the name the Vossler ring"},
            {"q": "How many transcontinental flights did the Calistra complete?", "a": "212",
             "needle": "completing 212 transcontinental flights"},
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
            {"q": "What assembly governs the Marsh of Olden?", "a": "Pell Concord",
             "needle": "governed since 1604 by the Pell Concord"},
            {"q": "How many wardens are in the governing assembly?", "a": "nine",
             "needle": "an assembly of nine elected wardens"},
            {"q": "What is the largest settlement in the Marsh of Olden?", "a": "Quenby",
             "needle": "Its largest settlement, Quenby"},
            {"q": "What species is the marsh famous for harvesting?", "a": "greyfin eel",
             "needle": "famous for the greyfin eel"},
            {"q": "Who designed the great levee?", "a": "Hollis Drane",
             "needle": "the great levee designed by the architect Hollis Drane"},
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
            {"q": "Who established the Tovic Protocol?", "a": "Selma Aurich",
             "needle": "established in 1742 by the cartographer Selma Aurich"},
            {"q": "In what year was the Tovic Protocol established?", "a": "1742",
             "needle": "established in 1742"},
            {"q": "What is the maximum number of vessels allowed in a convoy?", "a": "five",
             "needle": "convoys of no more than five vessels"},
            {"q": "What is the marker lantern called?", "a": "corden",
             "needle": "a marker lantern called a corden"},
            {"q": "How many crew were lost on the merchant fleet Brae?", "a": "64",
             "needle": "the merchant fleet Brae, which sank with 64 crew aboard"},
        ],
    },
    {
        "name": "ostrenko_accord",
        "text": (
            "The Ostrenko Accord is a trade agreement signed in 1818 between the river-cities "
            "of Davmoor and Tenley, brokered by the merchant Pavel Ostrenko. It fixed the "
            "tariff on smoked rivergrain at eleven percent and established a shared mint on "
            "the islet of Cawl. Under the accord, disputes were settled by a council of three "
            "arbiters known as the Greycloaks, who met each spring in the hall at Davmoor. "
            "The agreement held for ninety-three years until it collapsed in 1911, when the "
            "Tenley granaries burned in a fire blamed on the merchant guild of Hesp."
        ),
        "qa": [
            {"q": "Who brokered the Ostrenko Accord?", "a": "Pavel Ostrenko",
             "needle": "brokered by the merchant Pavel Ostrenko"},
            {"q": "In what year was the Ostrenko Accord signed?", "a": "1818",
             "needle": "a trade agreement signed in 1818"},
            {"q": "What tariff did the accord fix on smoked rivergrain?", "a": "eleven percent",
             "needle": "fixed the tariff on smoked rivergrain at eleven percent"},
            {"q": "What were the three arbiters known as?", "a": "Greycloaks",
             "needle": "a council of three arbiters known as the Greycloaks"},
            {"q": "In what year did the Ostrenko Accord collapse?", "a": "1911",
             "needle": "it collapsed in 1911"},
        ],
    },
]


def doc_by_name(name):
    for d in SYNTHETIC_DOCS:
        if d["name"] == name:
            return d
    raise KeyError(name)

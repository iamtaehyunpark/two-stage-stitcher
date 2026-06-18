"""
proofs/synthetic_docs_long.py — long, coreference-loaded fabricated documents for
Experiment 3.1 (latent-decimation vs. text-decimation).

The Proof-1/2/3 bank (`synthetic_docs.py`) is ~130 tokens and states each fact in a
self-contained clause: decimate the text and the answer token usually survives in
the kept span, so text-decimation never visibly fails and the latent-vs-text
contrast is invisible. These documents are built for the opposite property.

THE DESIGN (read before editing — it is the whole experiment):

  • Each document is long (~800-1400 tokens) and low-redundancy: the fact is stated
    once, not echoed.
  • The ANSWER string lives in the *surrounding* prose — the part decimation is
    allowed to drop. It is the proper name / number you must recover.
  • The NEEDLE sentence states the fact but refers to the answer ONLY by anaphora
    ("she", "he", "it", "that year", "the device", "there"). The answer string is
    deliberately NOT inside the needle span.

Why this makes the contrast visible:
  full text          → the model resolves the anaphor to the named antecedent and
                        answers (the A-ceiling gate must pass).
  decimated TEXT, needle protected → the kept text is the needle alone ("It was she
                        who first lifted it from the cellars"); the antecedent was
                        dropped, so there is no name to give → recall collapses.
  decimated LATENT, needle protected → the needle tokens' layer-12 states were
                        computed while attending to the whole document, so the
                        resolved coreferent ("she" → the named explorer) may be
                        folded into those states → recall may hold. THAT is the test.

INVARIANTS each QA item must satisfy (checked by `selftest_docs()` below, and worth
re-checking whenever you edit):
  1. `answer`  appears in `text`           (else A cannot succeed).
  2. `needle`  appears in `text`            (else span→token mapping fails).
  3. `answer`  does NOT appear in `needle`  (else decimated text keeps the answer and
                                             the coreference test is void).
  4. the answer is fabricated/unguessable   (so the C floor stays ~0).

If `selftest_docs()` flags an item, the experiment for that item is meaningless —
fix the prose, do not paper over it.
"""

SYNTHETIC_DOCS_LONG = [
    {
        "name": "kessler_observatory",
        "text": (
            "The Kessler Observatory stands on the windward shoulder of Mount Haldane, "
            "a basalt peak that rises above the fishing town of Orrin. The land was "
            "purchased in the spring of 1871 by a reclusive instrument-maker named "
            "Aldous Wren, who had made a modest fortune grinding lenses for the naval "
            "academy at Pell. Wren disliked cities and disliked committees more, and so "
            "he built his observatory far from both, hauling timber and brass up the "
            "mountain on the backs of mules. Construction took the better part of four "
            "years. When it was finished, the dome alone weighed as much as a small "
            "ship.\n\n"
            "The building's reputation rests almost entirely on a single instrument: a "
            "refracting telescope fitted with what the records of the town call the "
            "Marrow lens, an enormous disc of flawless glass nearly a metre across. It "
            "was she, the lens, around which the entire dome was designed — every "
            "girder and gear in the place exists only to point that great lens at the "
            "sky and hold it steady through the night. No comparable optic was cast "
            "again for thirty years, and rival astronomers travelled from as far as the "
            "southern republics merely to look through it once.\n\n"
            "The observatory's most celebrated night came in the autumn of 1893, when a "
            "faint smear of light was caught crossing the constellation the locals call "
            "the Plough. The object was the comet that would later carry the name "
            "Verel, and it was in that year that the instrument first turned its full "
            "aperture upon it and fixed its orbit. The measurement made the place "
            "famous, and for a decade afterwards the observatory's letters were answered "
            "by every learned society on the continent.\n\n"
            "Aldous Wren did not enjoy the fame for long. He grew ill the following "
            "winter and withdrew from the work almost entirely, and the running of the "
            "observatory passed to his assistant of many years, the astronomer Petra "
            "Voss. It was she who succeeded him as director, and under her hand the "
            "observatory turned from a single man's obsession into a working "
            "institution with students, a library, and a printed annual of "
            "observations. She held the post for twenty-six years.\n\n"
            "Life on the mountain was never comfortable. The road up from the harbour "
            "was passable only in the dry months, and through the long winters the "
            "staff were cut off for weeks at a stretch, living on salt fish and the "
            "vegetables they could coax from a walled garden behind the dome. Water "
            "had to be carried from a spring lower down the slope, and the wind, which "
            "came off the sea without anything to break it, found every gap in the "
            "shutters. More than one assistant gave up the work after a single season "
            "and went back down to the warmth of the town, and the logs of the early "
            "years are full of complaints about frozen ink and instruments too cold to "
            "touch with a bare hand. Those who stayed did so out of a kind of "
            "stubbornness that the founder seems to have prized above any talent for "
            "mathematics.\n\n"
            "The observatory kept careful watch not only on the sky but on the harbour "
            "below, for the same clear nights that made the seeing good also made the "
            "coast dangerous, and the staff fell into the habit of signalling ships "
            "with a shuttered lamp from the dome. The fishermen of the town came to "
            "rely on it, and for many years the relationship between the learned men on "
            "the mountain and the unlettered ones on the water was warmer than either "
            "would have predicted. It is said that the comet logs survive at all only "
            "because a fisherman carried them down through a storm the year the road "
            "washed out.\n\n"
            "Today the dome is weathered and the brass is green, but the mountain still "
            "draws visitors. They climb its slopes each summer to stand beneath the "
            "old lens, which has not been moved from the spot where its maker first "
            "set it. The town below keeps the observatory's records in the hall at "
            "Orrin, where the founding deed and the comet logs may still be read by "
            "anyone who asks."
        ),
        "qa": [
            {"q": "Who founded the Kessler Observatory?", "a": "Aldous Wren",
             "needle": "the running of the observatory passed to his assistant of many years"},
            {"q": "What is the name of the observatory's great lens?", "a": "Marrow lens",
             "needle": "It was she, the lens, around which the entire dome was designed"},
            {"q": "In what year did the observatory fix the orbit of the Verel comet?", "a": "1893",
             "needle": "it was in that year that the instrument first turned its full aperture upon it and fixed its orbit"},
            {"q": "Who succeeded the founder as director of the observatory?", "a": "Petra Voss",
             "needle": "It was she who succeeded him as director"},
            {"q": "On what mountain does the Kessler Observatory stand?", "a": "Mount Haldane",
             "needle": "They climb its slopes each summer to stand beneath the old lens"},
        ],
    },
    {
        "name": "harlan_bridge",
        "text": (
            "The crossing at Harlan Gorge is one of those works that outlived the "
            "purpose it was built for. The gorge itself is a narrow, vertiginous cut in "
            "the limestone, three hundred feet from rim to river, and for most of "
            "recorded history the only way across was a rope ferry that drowned someone "
            "almost every spring. The matter was finally settled by the county "
            "surveyor, an exacting and unpopular woman named Edda Cole, who drew up the "
            "plans for a single-span iron bridge in the years after the great flood of "
            "1849.\n\n"
            "She fought for the design for most of a decade. The county council "
            "preferred a cheaper timber trestle, and it was only after a second flood "
            "carried off the ferry entirely that they relented and let her build the "
            "thing in iron. Construction began at last in 1861. The span she had drawn "
            "was audacious for its day — a single arch leaping the whole gorge without a "
            "pier in the riverbed, because any pier, she argued, would simply be "
            "knocked down by the next flood.\n\n"
            "The arch was cast in sections at a foundry downriver and floated up on "
            "barges, and the ironwork was bolted together using a locking rivet of her "
            "own devising. She called the fastening the Cole joint, and it is the "
            "reason the structure has never needed its rivets replaced; the joint "
            "tightens rather than loosens under load, and engineers still study it. The "
            "bridge opened to traffic in the autumn of 1864 to no ceremony at all, "
            "because its designer refused to attend a celebration she considered "
            "premature.\n\n"
            "For eighty years the crossing carried farm carts, then motor traffic, then "
            "the heavy lorries of the quarrying companies. It was the quarry traffic "
            "that nearly destroyed it. By the 1940s the deck was cracking under loads "
            "no one had imagined in 1864, and the county proposed to demolish the arch "
            "and replace it with concrete. What saved it was an inspection that found "
            "the original ironwork sound — it was the deck, not the arch, that had "
            "failed — and so the deck alone was rebuilt and the old span left standing.\n\n"
            "The building of the bridge left its own small history in the valley. The "
            "foundry that cast the arch employed half the men of the downriver "
            "villages for three years, and when the work ended the sudden idleness "
            "caused such hardship that the parish had to open a relief fund. The "
            "barges that floated the iron sections upriver could only move on the "
            "spring freshets, so the assembly proceeded in fits and starts, a few "
            "sections a year, with the half-built arch left jutting over the gorge "
            "through each long winter like the bones of some enormous animal. "
            "Travellers wrote about the sight; one of them, a painter passing through "
            "on his way to the coast, made a series of watercolours of the unfinished "
            "span that hang today in a provincial gallery and are the only "
            "contemporary images of the work in progress.\n\n"
            "There were accidents, as there always were on such works. A scaffold "
            "collapsed in the second year and two men fell into the river, though both "
            "were pulled out alive a mile downstream; after that the designer insisted "
            "on rope harnesses for anyone working above the water, a precaution almost "
            "unheard of at the time and one the labourers resented as much as they "
            "were grateful for it. The local people, who had expected the iron monster "
            "to fall into the gorge at any moment, were astonished when it held, and "
            "for a generation afterwards a popular toast in the riverside taverns was "
            "simply to 'the arch that stayed up.'\n\n"
            "The bridge is named, as it happens, not for its builder but for the gorge, "
            "and most people who drive across it have never heard of the surveyor who "
            "spent ten years of her life forcing it into being. Her papers, including "
            "the original drawings of the joint, were left to the engineering school at "
            "Brennan, where they remain. The school keeps them in a glass case near its "
            "library, beneath a small portrait of the woman who drew them."
        ),
        "qa": [
            {"q": "Who designed the bridge at Harlan Gorge?", "a": "Edda Cole",
             "needle": "She fought for the design for most of a decade"},
            {"q": "What is the name of the locking rivet used in the bridge?", "a": "Cole joint",
             "needle": "it is the reason the structure has never needed its rivets replaced"},
            {"q": "In what year did construction of the iron bridge begin?", "a": "1861",
             "needle": "Construction began at last"},
            {"q": "In what year did the bridge open to traffic?", "a": "1864",
             "needle": "because its designer refused to attend a celebration she considered premature"},
            {"q": "To which school were the designer's papers left?", "a": "Brennan",
             "needle": "where they remain. The school keeps them in a glass case near its library"},
        ],
    },
    {
        "name": "saltwell_press",
        "text": (
            "Of all the small printing houses that once crowded the river district of "
            "Garrow, only one is still remembered by name, and that is largely the "
            "doing of a single book. The press was established above a chandlery in "
            "1799 by a former ship's clerk named Tobias Renn, who had learned his "
            "letters at sea and arrived in the city with little more than a "
            "second-hand frame and a case of worn type. He set up under the sign of a "
            "saltwell, an old word for the brine pits that had once made the district "
            "wealthy, and the name stuck to the business ever after.\n\n"
            "For twenty years it printed the ordinary work of any city press — "
            "handbills, ships' manifests, the occasional book of sermons. Its fortunes "
            "changed with a single commission. A natural philosopher named Iseult "
            "Marsh brought the press a manuscript no one else would touch: a treatise "
            "on the tides illustrated with hundreds of folding plates, each one "
            "engraved with a precision that made the work ruinously expensive. It was "
            "she who insisted on the foldouts, and it was that book, the Tidal Atlas, "
            "that made the little press famous when it appeared in 1822.\n\n"
            "The Atlas was a sensation, and not only for its contents. To bind the "
            "enormous folding plates without tearing them, the press's foreman had "
            "devised a flexible cloth spine reinforced with linen tape, a method the "
            "trade soon called the Garrow binding after the district. The binding "
            "outlived the book; for the rest of the century it was the standard way to "
            "bind any volume of maps or plates, and a great many printers used it "
            "without ever knowing where it had come from.\n\n"
            "The making of the book nearly ruined the press before it saved it. The "
            "engraving of the plates took three years and the labour of a dozen hands, "
            "and the cost of the copper alone exhausted what little capital the "
            "business had. For a time the printer was reduced to borrowing against the "
            "frame itself, and the chandler downstairs, who held the lease, more than "
            "once threatened to put the whole concern out onto the street. What carried "
            "it through was a list of subscribers — gentlemen of the philosophical "
            "societies who paid in advance for their copies — assembled by the author, "
            "who proved as relentless in raising money as in correcting proofs. She is "
            "said to have visited every learned man within fifty miles, atlas pages "
            "under her arm, and to have left none of them in peace until they "
            "subscribed.\n\n"
            "When the book at last appeared it was reviewed in every journal that "
            "mattered, and the small printing house found itself, for one strange "
            "season, the most talked-of address in the city. Visitors came merely to "
            "see the room where the plates had been pulled. The trade, of course, was "
            "less interested in the tides than in the binding, and within a year "
            "printers in three countries were imitating the flexible spine without "
            "the least idea whom to credit for it. The foreman who had devised it "
            "received nothing for the invention but the satisfaction of seeing his "
            "work outlast him.\n\n"
            "Tobias Renn died in 1831 and left the press to his daughter, who ran it "
            "competently for another forty years without ever again printing anything "
            "the world remembered. The frame on which the Atlas had been printed was "
            "kept all that time in a back room, more relic than tool. When the business "
            "finally closed, the family gave the old press to the civic museum at "
            "Garrow, where it stands today in the entrance hall. Beside it the museum "
            "displays a first edition of the Atlas, open to one of the great folding "
            "plates, so that the visitor sees at once the machine and the thing that "
            "made it famous."
        ),
        "qa": [
            {"q": "Who founded the Saltwell Press?", "a": "Tobias Renn",
             "needle": "He set up under the sign of a saltwell"},
            {"q": "Who was the author of the Tidal Atlas?", "a": "Iseult Marsh",
             "needle": "It was she who insisted on the foldouts"},
            {"q": "In what year was the Tidal Atlas published?", "a": "1822",
             "needle": "that made the little press famous when it appeared"},
            {"q": "What was the press's flexible binding method called?", "a": "Garrow binding",
             "needle": "for the rest of the century it was the standard way to bind any volume of maps or plates"},
            {"q": "To which museum was the old press given?", "a": "Garrow",
             "needle": "the family gave the old press to the civic museum"},
        ],
    },
    {
        "name": "vantroy_engine",
        "text": (
            "The locomotive known to enthusiasts simply as Number Nine was not, in "
            "fact, the ninth of anything; the number was painted on it by mistake at "
            "the works and never corrected. The Vantroy ironworks had its busiest "
            "season in 1888, and it was in that year that the locomotive first rolled "
            "out onto the mountain line between Cold Harbour and the mining town "
            "of Skel, a route so steep that ordinary engines lost their footing on the "
            "grades. The problem of the grades was solved by the works' chief "
            "engineer, a taciturn Welshman named Gareth Pyne, who had spent years on "
            "rack railways in the Alps before coming to Vantroy.\n\n"
            "His solution was unusual. Rather than add a toothed rack rail, which the "
            "company could not afford, he designed the engine to carry water in tanks "
            "slung low between the wheels, so that its weight sat almost on the rails "
            "themselves and its driving wheels could not slip. The arrangement made the "
            "locomotive look squat and ugly, and the drivers hated it at first, but it "
            "climbed the grades to Skel without a rack and it never once lost its grip "
            "in twenty years of service. He called the low-slung tank design the "
            "Pyne truck, and two other railways copied it before the decade was out.\n\n"
            "The engine ran the Skel line until the mines closed in 1911. By then it "
            "was the last of its kind still working, the others having been scrapped "
            "for their metal during a shortage some years before. When the line shut, "
            "the company meant to scrap this one too, but the railwaymen of Cold "
            "Harbour, who had grown fond of the ugly machine, raised the money to buy "
            "it themselves and keep it in the engine shed.\n\n"
            "The line it had served was a hard one in every season. In summer the "
            "rails ran through cuttings where the heat stood still and the crews "
            "worked stripped to the waist; in winter the same cuttings filled with "
            "drifting snow, and the engine spent as many days pushing a plough as "
            "hauling ore. The grades were so severe that loaded trains descended "
            "under braking the whole way down, and the smell of hot iron brakes is "
            "the thing the oldest townspeople still remember first when the line is "
            "mentioned. Derailments were common enough that the company kept a crane "
            "and a gang of men permanently stationed halfway up, and the wages of "
            "that gang were for years the largest single item in the railway's "
            "accounts.\n\n"
            "The miners themselves rode the same trains to and from the workings, "
            "there being no road worth the name, and so the fortunes of the engine "
            "and the fortunes of the town were bound together from the start. When "
            "the ore began to run thin the traffic fell away year by year, and the "
            "crews who had once worked the line in shifts around the clock were let go "
            "a few at a time, until at the end a single driver took the last train "
            "down with no ceremony and a handful of passengers who had come simply to "
            "say that they had.\n\n"
            "There it sat, oiled and cold, for the better part of forty years. A "
            "preservation society finally restored it to steam in the 1950s, and it now "
            "runs a few miles of the old route each summer for visitors. The society "
            "keeps it not at Cold Harbour, where the sheds were demolished, but at a "
            "small museum line in the valley at Tern, which maintains the boiler and "
            "sells tickets to ride behind it. The mistake in its number has never been "
            "painted out, on the grounds that it is now part of the engine's history."
        ),
        "qa": [
            {"q": "Who designed the Number Nine locomotive?", "a": "Gareth Pyne",
             "needle": "His solution was unusual"},
            {"q": "What was the low-slung tank arrangement called?", "a": "Pyne truck",
             "needle": "two other railways copied it before the decade was out"},
            {"q": "In what year was the locomotive built?", "a": "1888",
             "needle": "it was in that year that the locomotive first rolled out onto the mountain line"},
            {"q": "In what year did the Skel mining line close?", "a": "1911",
             "needle": "The engine ran the Skel line until the mines closed"},
            {"q": "At which museum line is the locomotive now kept?", "a": "Tern",
             "needle": "but at a small museum line in the valley"},
        ],
    },
]


def doc_by_name(name):
    for d in SYNTHETIC_DOCS_LONG:
        if d["name"] == name:
            return d
    raise KeyError(name)


def selftest_docs(verbose=True):
    """Check the four authoring invariants on every QA item. Pure string checks —
    no model, no tokenizer — so it runs anywhere and should be run after any edit to
    the prose. Returns True iff every item is well-formed."""
    ok = True
    for d in SYNTHETIC_DOCS_LONG:
        text = d["text"]
        for qa in d["qa"]:
            q, a, needle = qa["q"], qa["a"], qa["needle"]
            problems = []
            if a not in text:
                problems.append("answer NOT in text (A can't succeed)")
            if needle not in text:
                problems.append("needle NOT a substring of text (span map fails)")
            if a.lower() in needle.lower():
                problems.append("answer IS in needle (coreference test void)")
            if problems:
                ok = False
                if verbose:
                    print(f"[{d['name']}] {q!r}\n    a={a!r} needle={needle!r}")
                    for p in problems:
                        print(f"    ✗ {p}")
    if verbose:
        n_docs = len(SYNTHETIC_DOCS_LONG)
        n_qa = sum(len(d["qa"]) for d in SYNTHETIC_DOCS_LONG)
        print(f"{'ALL WELL-FORMED' if ok else 'PROBLEMS FOUND'} "
              f"— {n_docs} docs, {n_qa} QA items")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest_docs() else 1)

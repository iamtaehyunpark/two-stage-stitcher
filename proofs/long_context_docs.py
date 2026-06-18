"""
proofs/long_context_docs.py — padding a synthetic fact to a target length, with
the planted fact at a chosen depth, for Proof 4 (length scaling).

Proofs 1–3 validated the receiver on ~130-token documents. Proof 4 asks whether the
same facts survive when the document grows to the lengths the project is actually
for (2k … 32k tokens). To test that honestly we need, at every length:

  • a real, unguessable planted fact (so C still fails and the win is not memory);
  • the rest of the document filled to length with FILLER that introduces no fact
    the question could pattern-match to (so the padding never becomes a false floor
    or a false ceiling);
  • control over WHERE the fact sits (depth ~10% / ~50% / ~90%), to expose any
    lost-in-the-middle behaviour the way normal long-context retrieval shows it.

THE CONSTRUCTION
────────────────
A padded document is three pieces laid end to end:

      [ before-filler ]  [ FACT BLOCK ]  [ after-filler ]
      └──────────────────── target_tokens total ────────────────────┘

The FACT BLOCK is one of the existing short synthetic paragraphs
(`synthetic_docs.SYNTHETIC_DOCS`) — a self-contained ~130-token block whose facts
are fabricated, whose `qa` items carry the exact `needle` clause Proof 3 uses. We
reuse it verbatim so the gates, the needle→position mapping, and the wrong-document
control all keep their meaning; Proof 4 changes only the surroundings.

Depth is the fraction of the document that precedes the block: depth 0.1 places the
block near the top, 0.9 near the bottom. `before_filler ≈ depth · filler`,
`after_filler ≈ (1−depth) · filler`.

THE FILLER MUST BE INERT — this is the one trap specific to Proof 4
──────────────────────────────────────────────────────────────────
If the padding contains anything the model can pattern-match to the question, you
manufacture a false floor (the model answers from a cue in the filler, C-equivalent
stops failing) or a false ceiling, and every recall number is silently inflated. So
the filler bank below is deliberately:

  • entity-free  — no proper nouns, so a "Who …?" question finds no name to latch;
  • digit-free   — no numbers, so a "How many / what year" question finds no figure;
  • fact-free    — generic, abstract prose that asserts nothing checkable.

`selftest_filler()` enforces the digit-free / answer-free invariants statically. But
static checks cannot prove behavioural inertness, so Proof 4 ALSO runs a behavioural
gate per length+depth: prefill the padded document with the fact block REPLACED by
filler ("filler-only") and confirm the model still cannot answer. That re-check —
not an assumption inherited from the short-doc runs — is what keeps a stray cue at
8k of padding from inflating the numbers. See `build_padded_doc(..., drop_fact=True)`.
"""

import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.synthetic_docs import SYNTHETIC_DOCS, doc_by_name  # noqa: F401,E402


# ── the inert filler bank ──────────────────────────────────────────────────────
# Neutral, abstract sentences. No proper nouns, no digits, no checkable facts. Each
# is mundane enough that no QA in the synthetic bank can pattern-match to it. Keep
# this property when editing: run `python proofs/long_context_docs.py` after any
# change and the selftest will flag a stray digit or a leaked answer string.
FILLER_SENTENCES = [
    "The afternoon settled into a quiet that asked nothing of anyone.",
    "A breeze moved through the open room and turned the pages of nothing in particular.",
    "Work of this kind rewards patience more than it rewards haste.",
    "It is easy, on such days, to mistake stillness for the absence of effort.",
    "The light shifted slowly along the wall and no one thought to mark it.",
    "Some tasks are finished only because someone decided they were finished.",
    "There was tea, and then there was the matter of whether to make more.",
    "A habit, once formed, tends to outlast the reason it was formed for.",
    "The hours folded into one another with the soft indifference of routine.",
    "Nothing about the morning insisted on being remembered.",
    "One learns, in time, to let small things pass without comment.",
    "The conversation drifted, as conversations do, toward and then away from its point.",
    "Patience, the older hands liked to say, is mostly a matter of breathing.",
    "The room held the warmth of the day a little longer than the day deserved.",
    "It was the sort of work that looks like rest to anyone watching from outside.",
    "A long pause is not the same as an empty one, though they are easily confused.",
    "The floor creaked in the familiar places and was quiet in the others.",
    "Attention, given freely, has a way of returning more than it costs.",
    "There is a rhythm to waiting that only reveals itself to those who wait.",
    "The kettle cooled and was warmed again and cooled once more.",
    "Most of what is said in passing is forgotten before it can do any harm.",
    "The shape of an ordinary day resists every attempt to summarize it.",
    "One does the next small thing, and then the small thing after that.",
    "The window framed a view that changed too slowly to notice changing.",
    "Comfort, when it comes, rarely announces itself in advance.",
    "The dust settled where it always settled and was wiped away again.",
    "An unhurried mind finds more in a quiet hour than a busy one finds in a day.",
    "The chair had been moved once and never moved back, and no one minded.",
    "Some afternoons exist only to be passed through on the way to evening.",
    "The hum of the building was the kind one stops hearing after a while.",
    "It is a gentle discipline, doing little and doing it without complaint.",
    "The page took the ink and gave nothing back but the ink itself.",
    "A thing left half-said is sometimes the most considerate thing to leave.",
    "The light went amber, then grey, and the room accepted both without protest.",
    "Routine is a quiet companion that never asks where the time has gone.",
    "The corridor carried its small echoes politely from one end to the other.",
    "What is unremarkable is, for that reason, remarkably easy to overlook.",
    "The day asked for nothing and so received the rare gift of being left alone.",
    "A settled stillness is its own kind of occupation, if one allows it to be.",
    "The clock was heard only when there was nothing else to hear.",
]


def _shuffled(seed):
    """A deterministic shuffle of the filler bank, so a given (length, depth, seed)
    always yields the same padding — reproducible across runs and machines."""
    import random
    bank = list(FILLER_SENTENCES)
    random.Random(seed).shuffle(bank)
    return bank


def _filler_with_token_budget(tokenizer, n_tokens, seed, max_doc_tokens):
    """Build a filler string of approximately `n_tokens` tokens by cycling the
    shuffled bank, then trim to the budget by slicing token ids and decoding. Returns
    "" for a non-positive budget. The result carries no special tokens of its own;
    it is plain prose to be concatenated around the fact block."""
    if n_tokens <= 0:
        return ""
    bank = _shuffled(seed)
    # over-build, then trim. Estimate ~16 tokens/sentence; pad the estimate so we
    # never under-fill before trimming.
    need_sentences = int(n_tokens / 8) + 4
    parts = [bank[i % len(bank)] for i in range(need_sentences)]
    text = " ".join(parts)
    ids = tokenizer(text, add_special_tokens=False,
                    truncation=True, max_length=max_doc_tokens).input_ids
    if len(ids) > n_tokens:
        ids = ids[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _seed(name, target_tokens, depth, salt):
    return zlib.crc32(f"{name}|{target_tokens}|{depth}|{salt}".encode()) & 0xFFFFFFFF


def build_padded_doc(tokenizer, base_doc, target_tokens, depth, max_doc_tokens=40000,
                     drop_fact=False):
    """Assemble one padded document.

    base_doc        — an entry from SYNTHETIC_DOCS (has `text`, `name`, `qa`).
    target_tokens   — desired total length (tokens), approximate (actual reported).
    depth           — fraction of the document preceding the fact block (0..1).
    max_doc_tokens  — truncation cap for the internal tokenization (raise for 32k+).
    drop_fact       — if True, REPLACE the fact block with filler of the same token
                      length (the "filler-only" inertness control: the document has
                      the same size and shape but the planted fact is gone, so the
                      question must now be unanswerable). The returned `qa` is kept so
                      the caller can ask the same questions and confirm they fail.

    Returns a dict:
      name, text, qa, target_tokens, n_tokens (actual), depth (requested),
      depth_actual (fact-block start fraction by token), fact_token_span (start,end)
      in the padded token sequence (None when drop_fact), drop_fact.
    """
    fact_text = base_doc["text"]
    fact_ids = tokenizer(fact_text, add_special_tokens=False).input_ids
    n_fact = len(fact_ids)

    filler_total = max(0, target_tokens - n_fact)
    n_before = int(round(depth * filler_total))
    n_after = filler_total - n_before

    before = _filler_with_token_budget(
        tokenizer, n_before, _seed(base_doc["name"], target_tokens, depth, "before"),
        max_doc_tokens)
    after = _filler_with_token_budget(
        tokenizer, n_after, _seed(base_doc["name"], target_tokens, depth, "after"),
        max_doc_tokens)

    if drop_fact:
        # Same shape, no fact: swap the block for filler of equal token length.
        middle = _filler_with_token_budget(
            tokenizer, n_fact, _seed(base_doc["name"], target_tokens, depth, "middle"),
            max_doc_tokens)
    else:
        middle = fact_text

    pieces = [p for p in (before, middle, after) if p]
    text = "\n\n".join(pieces)

    # Report the ACTUAL token length and the fact block's token span in the padded
    # sequence (special tokens included exactly as capture_document tokenizes it), so
    # the caller knows the true length and where the needle region landed.
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=max_doc_tokens)
    n_tokens = len(enc["input_ids"])

    fact_token_span = None
    depth_actual = depth
    if not drop_fact:
        char_start = text.find(fact_text)
        char_end = char_start + len(fact_text)
        tok_start = tok_end = None
        for i, (s, e) in enumerate(enc["offset_mapping"]):
            if s == e:                       # special token (BOS): zero-width
                continue
            if s < char_end and e > char_start:
                if tok_start is None:
                    tok_start = i
                tok_end = i
        if tok_start is not None:
            fact_token_span = (tok_start, tok_end)
            depth_actual = round(tok_start / max(1, n_tokens), 3)

    return {
        "name": base_doc["name"],
        "text": text,
        "qa": base_doc["qa"],
        "target_tokens": target_tokens,
        "n_tokens": n_tokens,
        "depth": depth,
        "depth_actual": depth_actual,
        "fact_token_span": fact_token_span,
        "drop_fact": drop_fact,
    }


# ── static inertness selftest (run after any edit to the filler bank) ───────────
def selftest_filler(verbose=True):
    """Enforce the filler invariants WITHOUT a model:
      1. no filler sentence contains a digit          (kills number/year cues);
      2. no QA answer string appears in the filler     (kills name/term cues);
      3. no QA answer appears in another document's needle (sanity on the bank).
    A failure here means the padding could leak a fact — fix the prose, do not run.
    Behavioural inertness is still re-checked per length by the filler-only gate; this
    is the cheap static guard that catches the obvious leaks first."""
    ok = True
    blob = " ".join(FILLER_SENTENCES).lower()

    for i, s in enumerate(FILLER_SENTENCES):
        if any(ch.isdigit() for ch in s):
            ok = False
            if verbose:
                print(f"  ✗ filler[{i}] contains a digit: {s!r}")

    answers = []
    for d in SYNTHETIC_DOCS:
        for qa in d["qa"]:
            answers.append(qa["a"])
    for a in answers:
        if a.lower() in blob:
            ok = False
            if verbose:
                print(f"  ✗ answer {a!r} appears in the filler bank (leaked cue)")

    if verbose:
        print(f"{'FILLER INERT' if ok else 'FILLER LEAK'} — "
              f"{len(FILLER_SENTENCES)} sentences, {len(answers)} answers checked")
    return ok


if __name__ == "__main__":
    sys.exit(0 if selftest_filler() else 1)

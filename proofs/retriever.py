"""
proofs/retriever.py — the "strong" text-RAG baseline for Proof 5.

A latent win is only meaningful against a real, well-tuned retriever; a strawman RAG
proves nothing. So this builds a current dense retriever (BGE-large-en-v1.5), chunks each
document at a TUNED granularity, and runs a k-sweep — text_rag is reported at its best
config, the honest baseline. It also logs retrieval recall (did the top-k actually fetch
the gold supporting sentences?), which is the signal Proof 5's failure-mode attribution
needs: a RAG miss because the retriever whiffed is a retrieval failure, not evidence that
latent reasons better.

Design choices that keep the comparison fair and cheap:

  • Recall is computed by CHAR-SPAN CONTAINMENT against the gold spans `hotpot.py` already
    recorded — a gold sentence counts as retrieved iff some retrieved chunk's char range
    fully contains it. This is robust to sentence-splitter quirks (no dependence on the
    chunker reproducing HotpotQA's exact sentence boundaries) and makes "retrieval success"
    an objective per-item fact, not a fuzzy string match.

  • Chunks are sentence-aware with char offsets and a one-sentence overlap (sliding), so a
    gold sentence near a boundary still lands wholly inside some chunk. Chunk size is a
    target token budget (approximated as chars/`CHARS_PER_TOK`); `tune_chunk_size` picks
    the size that maximizes recall on a held-out slice — tuning the baseline to win.

  • The BGE backend is lazy and runs on CPU by default so it never contends with the 70B's
    GPU memory (tiny model, latency irrelevant at this n). A `hash` backend (deterministic
    bag-of-words) lets the retrieve/recall/chunk logic selftest offline without the model.

Run `python3 proofs/retriever.py` for the offline selftest (hash backend, no download).
"""

import re
import sys
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CHARS_PER_TOK = 4          # crude tokens≈chars/4 for sizing chunks without a tokenizer
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ── sentence-aware chunking with char offsets ────────────────────────────────────
def _sentence_spans(doc):
    """Contiguous (start, end) spans tiling `doc`, split at sentence terminators while
    keeping the trailing whitespace with each piece so the spans cover the doc with no
    gaps (needed for exact char-containment recall)."""
    spans, start, n = [], 0, len(doc)
    for m in re.finditer(r"[.!?]+[\s)\"']*\s+", doc):
        end = m.end()
        if end > start:
            spans.append((start, end))
            start = end
    if start < n:
        spans.append((start, n))
    return spans


def chunk_document(doc, chunk_tokens=128, overlap_sentences=1):
    """Group consecutive sentences into chunks of ~`chunk_tokens` tokens, with
    `overlap_sentences` of sliding overlap. Returns [{text, char_start, char_end}] whose
    char ranges are contiguous-per-chunk slices of `doc` (so a chunk that covers a gold
    sentence's span contains that sentence verbatim)."""
    sents = _sentence_spans(doc)
    if not sents:
        return [{"text": doc, "char_start": 0, "char_end": len(doc)}]
    budget = max(1, chunk_tokens) * CHARS_PER_TOK
    chunks, i = [], 0
    while i < len(sents):
        cs = sents[i][0]
        j, ce = i, sents[i][1]
        while j + 1 < len(sents) and (sents[j + 1][1] - cs) <= budget:
            j += 1
            ce = sents[j][1]
        chunks.append({"text": doc[cs:ce], "char_start": cs, "char_end": ce})
        if j + 1 >= len(sents):
            break
        i = max(j + 1 - overlap_sentences, i + 1)     # slide, always make progress
    return chunks


# ── recall by char-span containment ──────────────────────────────────────────────
def _covers(chunk, gold):
    return chunk["char_start"] <= gold["char_start"] and chunk["char_end"] >= gold["char_end"]


def retrieval_recall(retrieved_chunks, gold_sentences):
    """Fraction of gold supporting sentences wholly contained in some retrieved chunk,
    plus a `full` flag (all gold sentences covered — the retrieval-success condition the
    fair reasoning-vs-reasoning headline is computed on)."""
    if not gold_sentences:
        return {"recall": 0.0, "full": False, "n_gold": 0}
    covered = sum(any(_covers(c, g) for c in retrieved_chunks) for g in gold_sentences)
    return {"recall": covered / len(gold_sentences),
            "full": covered == len(gold_sentences), "n_gold": len(gold_sentences)}


# ── embedding backends ───────────────────────────────────────────────────────────
def _dot(a, b):
    """Inner product that works for both python lists (hash backend) and numpy rows
    (BGE backend), so `retriever.py` needs no numpy import of its own."""
    return sum(x * y for x, y in zip(a, b))


class _HashBackend:
    """Deterministic bag-of-words embedder for offline selftests only. Not used in any
    real run — it exists so chunk/retrieve/recall logic is exercised without sbert or
    numpy. Returns plain python list vectors."""
    dim = 4096

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[hash(w) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v

    def encode_passages(self, texts):
        return [self._vec(t) for t in texts]

    def encode_queries(self, texts):
        return self.encode_passages(texts)


class _BGEBackend:
    """BAAI/bge-large-en-v1.5 via sentence-transformers. Query side prepends the BGE
    retrieval instruction; passage side is raw. Normalized embeddings → cosine = dot."""

    def __init__(self, device="cpu", model_name="BAAI/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def encode_passages(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    def encode_queries(self, texts):
        q = [BGE_QUERY_INSTRUCTION + t for t in texts]
        return self.model.encode(q, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


def make_backend(kind="bge", device="cpu"):
    return _HashBackend() if kind == "hash" else _BGEBackend(device=device)


# ── retriever ─────────────────────────────────────────────────────────────────────
class Retriever:
    """Per-document dense retriever. Chunk once, embed once, then `retrieve(question, k)`.
    Designed for HotpotQA's small 10-paragraph docs: re-index per item is cheap."""

    def __init__(self, backend, chunk_tokens=128, overlap_sentences=1):
        self.backend = backend
        self.chunk_tokens = chunk_tokens
        self.overlap_sentences = overlap_sentences

    def index(self, doc):
        self.chunks = chunk_document(doc, self.chunk_tokens, self.overlap_sentences)
        self.emb = self.backend.encode_passages([c["text"] for c in self.chunks])
        return self

    def retrieve(self, question, k):
        q = self.backend.encode_queries([question])[0]
        scored = sorted(range(len(self.chunks)),
                        key=lambda i: _dot(self.emb[i], q), reverse=True)
        return [self.chunks[i] for i in scored[:k]]


def budget_matched_k(gold_sentences, chunk_tokens):
    """A k that roughly matches the gold-fact token budget: enough chunks to hold the
    gold supporting sentences' tokens. Gives the k-sweep an information-matched point so
    text_rag isn't penalized purely on budget."""
    gold_chars = sum(g["char_end"] - g["char_start"] for g in gold_sentences)
    gold_toks = gold_chars / CHARS_PER_TOK
    return max(1, math.ceil(gold_toks / max(1, chunk_tokens)))


# ── chunk-size tuning (tune the baseline to win) ─────────────────────────────────
def tune_chunk_size(items, backend, candidate_sizes=(64, 128, 256), k=4,
                    overlap_sentences=1):
    """Pick the chunk size maximizing mean recall@k over `items` (each
    {doc_text, question, gold_sentences}). Returns (best_size, {size: mean_recall})."""
    scores = {}
    for size in candidate_sizes:
        rec = []
        for it in items:
            r = Retriever(backend, size, overlap_sentences).index(it["doc_text"])
            got = r.retrieve(it["question"], k)
            rec.append(retrieval_recall(got, it["gold_sentences"])["recall"])
        scores[size] = sum(rec) / len(rec) if rec else 0.0
    best = max(scores, key=scores.get)
    return best, scores


# ── selftest (offline, hash backend) ─────────────────────────────────────────────
def selftest():
    doc = ("Intro about nothing in particular. The reactor was sealed in 1962. "
           "Filler sentence that says little. The override code is named Halcyon. "
           "More irrelevant prose follows here. Yet another distractor sentence. ")
    gold_text_a = "The reactor was sealed in 1962. "
    gold_text_b = "The override code is named Halcyon. "
    cs_a = doc.find(gold_text_a)
    cs_b = doc.find(gold_text_b)
    gold = [{"char_start": cs_a, "char_end": cs_a + len(gold_text_a), "text": gold_text_a},
            {"char_start": cs_b, "char_end": cs_b + len(gold_text_b), "text": gold_text_b}]

    chunks = chunk_document(doc, chunk_tokens=8)
    # every gold sentence must fall wholly inside some chunk (contiguity invariant)
    assert all(any(_covers(c, g) for c in chunks) for g in gold), "chunking lost a gold span"

    backend = make_backend("hash")
    r = Retriever(backend, chunk_tokens=8).index(doc)
    got = r.retrieve("When was the reactor sealed and what is the override code?", k=4)
    rec = retrieval_recall(got, gold)
    assert rec["full"], f"hash retriever failed to fetch gold (recall={rec['recall']})"

    best, scores = tune_chunk_size(
        [{"doc_text": doc, "question": "reactor sealed year override code", "gold_sentences": gold}],
        backend, candidate_sizes=(8, 32), k=4)
    assert best in (8, 32)
    print(f"selftest: OK (chunk contiguity, recall@4 full, tune→{best} {scores})")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hash", choices=["hash", "bge"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.backend == "bge":
        b = make_backend("bge", args.device)
        print("BGE backend loaded:", type(b).__name__)
    selftest()

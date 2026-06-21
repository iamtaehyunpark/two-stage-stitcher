"""
Side experiment — visualize a model's attention as a real 3D token graph.

Not a receiver-validation proof; a diagnostic/illustration that reuses the
project's own models (`StitcherConfig.source_model` by default — Qwen2.5-7B, the
stitcher's *source* side, which fits on one GPU). One forward pass exposes both
attentions and hidden states, and three view styles render them:

  graph   (default) — the REAL 3D view: stopwords/punctuation/specials are dropped,
            the surviving content tokens are clustered into meaningful units
            (attention-graph communities, or k-means on hidden states), and each
            token is projected to a point in 3D (PCA — or UMAP — of its layer hidden
            state). Same-sentence tokens are pulled together and the full sentence
            floats as a label at each sentence's centroid. Attention is drawn as
            LINES between points (width/opacity ∝ score); nodes are colored by
            cluster. Distinct highlight lines connect the SAME token (gold) or a
            same-meaning token (magenta) recurring across different sentences. Self-
            contained interactive HTML: drag to rotate, hover a node for its
            token+cluster+sentence, CLICK a node to isolate just its links. One 3D
            scene per requested layer.
  surface — the attention matrix as a 3D terrain (key × query × score).
  arc     — a flat bertviz-style arc diagram.

Why "eager". SDPA / FlashAttention fuse the softmax and never materialize the
[heads, q, k] weight matrix, so `output_attentions=True` silently returns None
under them. We force `attn_implementation="eager"` — slower, but it is the only
implementation that hands back the weights we are here to look at.

What gets omitted (the "meaningless attention" to drop):
  --drop-first/--keep-first   the attention *sink* on token 0 (dropped by default).
  --drop-self/--keep-self     the i→i self-loops (kept only in the surface terrain).
  --threshold T               any edge below score T is dropped (absolute cut).
  --top-k K                   per query keep only its K strongest keys (great for
                              decluttering the 3D graph).
Heads are mean-aggregated (`--heads mean`); pass a subset (`--heads 3 7 12`) or
`--per-head` for one scene/panel per head of a single layer.

Usage:
    # interactive 3D token graph (default): stopwords dropped, attention-community
    # clustering, click a node to isolate its links — open the .html in a browser
    python proofs/attention_map.py \
        --text "The Eiffel Tower is in Paris. It was completed in 1889." \
        --layers 6 14 24 --top-k 3 --out proofs/data/attn_graph.html

    # cluster by semantic similarity instead, with UMAP layout
    python proofs/attention_map.py --cluster embedding --proj umap \
        --layers 14 --out proofs/data/g.html

    # the matrix views instead
    python proofs/attention_map.py --style surface --layers 4 14 24 --out s.html
    python proofs/attention_map.py --style arc --layers 14 --out a.png
"""

from __future__ import annotations   # `str | None` annotations on any 3.7+ interp

import os
import sys
import re
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")                      # headless GPU box — no display
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401  (registers 3d projection)

from config import StitcherConfig


DEFAULT_TEXT = (
    "The Eiffel Tower is located in Paris, the capital of France. "
    "It was completed in 1889 and is named after the engineer Gustave Eiffel."
)


# ── model loading ───────────────────────────────────────────────────────────────
def load_model(cfg: StitcherConfig, which: str, model_name: str | None, device: str):
    """Load `source` (Qwen, single GPU) or `target` (DeepSeek-70B, sharded) with
    attentions exposed. Eager attention is mandatory — see module docstring.

    `device` is the single GPU for the source model. It is NOT inherited from
    `cfg.source_device` (hardcoded to cuda:3 for the 4-GPU stitcher layout) — this
    standalone experiment puts the SLM on whatever you expose, default cuda:0. So
    `CUDA_VISIBLE_DEVICES=6 python … ` just works (GPU 6 is logical cuda:0)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype = getattr(torch, cfg.dtype)
    name = model_name or (cfg.source_model if which == "source" else cfg.target_model)
    tokenizer = AutoTokenizer.from_pretrained(name)

    if which == "source":
        device_map = device
        max_memory = None
    else:
        # mirror the env-faithful sharded placement the proofs use
        device_map = "sequential"
        max_memory = {i: "70GiB" for i in cfg.llama_devices}

    print(f"Loading {name} (attn_implementation=eager, device_map={device_map}) …")
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        attn_implementation="eager",        # the whole point — return the weights
    )
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def get_outputs(model, tokenizer, text: str, max_length: int):
    """One forward pass. Returns (labels, attentions, hiddens):
      attentions — tuple len n_layers, each [n_heads, seq, seq] on CPU float32.
      hiddens    — tuple len n_layers+1, each [seq, hidden_dim] on CPU float32
                   (index 0 is the embedding output; layer L's output is index L+1).
                   These are what the graph view projects to 3D positions."""
    first_device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(first_device) for k, v in enc.items()}
    out = model(**enc, output_attentions=True, output_hidden_states=True)

    ids = enc["input_ids"][0].tolist()
    labels = clean_token_labels(tokenizer, ids)
    attns = tuple(a[0].float().cpu() for a in out.attentions)        # drop batch dim
    hiddens = tuple(h[0].float().cpu() for h in out.hidden_states)
    return labels, attns, hiddens


def clean_token_labels(tokenizer, ids):
    """Human-readable per-token strings: strip the BPE space markers (Ġ / ▁) and
    render whitespace/newlines visibly so the y-axis stays legible."""
    labels = []
    for tid in ids:
        tok = tokenizer.convert_ids_to_tokens(tid)
        tok = tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n")
        tok = tok.replace("\n", "\\n")
        labels.append(tok if tok.strip() else repr(tok))
    return labels


# ── stopwords + clustering ──────────────────────────────────────────────────────
STOPWORDS = set("""
a an the this that these those of in on at to for from by with about as into over
after before under above between out off up down and or but nor so yet if then else
than is are was were be been being am do does did doing have has had having will would
shall should can could may might must not no nor i you he she it we they me him her us
them my your his its our their mine yours hers ours theirs who whom which what whose
when where why how there here all any both each few more most other some such only own
same too very s t can just don now also which while because
""".split())


def _is_dropped_token(label, drop_stopwords):
    """A token is dropped if it's a special/BOS token, punctuation/whitespace only, or
    (when enabled) a stopword. Subword fragments of content words are kept — stopwords
    are whole tokens, so matching the cleaned label is enough."""
    t = label.strip()
    if not t:
        return True
    if t.startswith("<") and t.endswith(">"):      # <|im_start|>, <|endoftext|>, …
        return True
    if t.startswith("\\n"):                          # rendered newline tokens
        return True
    if not any(ch.isalnum() for ch in t):            # punctuation only
        return True
    if drop_stopwords and t.lower() in STOPWORDS:
        return True
    return False


def keep_indices(labels, drop_stopwords):
    """Indices of tokens to visualize. Token 0 (BOS / attention sink) is always
    dropped — it's never a content word and is the projection outlier."""
    keep = [i for i, lab in enumerate(labels)
            if i != 0 and not _is_dropped_token(lab, drop_stopwords)]
    return keep or list(range(len(labels)))          # never return empty


_SENT_END = ".!?…"


def sentence_ids(labels):
    """Assign each token (full, unfiltered sequence) a sentence index, and return the
    reconstructed sentence strings. A token ends a sentence when it terminates in
    .!?… AND the next token starts a new word (leading space) or is end-of-text —
    that lookahead avoids splitting decimals/abbreviations whose continuation tokens
    carry no leading space ('8', '.', '849' stays one 'sentence')."""
    n = len(labels)
    sids, texts, buf, cur = [], [], [], 0
    for i, lab in enumerate(labels):
        sids.append(cur)
        buf.append(lab)
        t = lab.strip()
        nxt = labels[i + 1] if i + 1 < n else ""
        ends = bool(t) and t[-1] in _SENT_END
        boundary = ends and (i == n - 1 or nxt[:1].isspace() or nxt.startswith("\\n"))
        if boundary:
            texts.append(_join_sentence(buf)); buf = []; cur += 1
    if buf:
        texts.append(_join_sentence(buf))
    while len(texts) <= (max(sids) if sids else -1):
        texts.append("")
    return sids, texts


def _join_sentence(buf):
    s = "".join(buf)
    s = re.sub(r"\\n", " ", s)
    s = re.sub(r"\s+([.,!?;:…])", r"\1", s)          # no space before punctuation
    return re.sub(r"\s+", " ", s).strip()


def _wrap(text, max_chars):
    """Truncate a sentence for the floating label and soft-wrap with <br>."""
    text = text if len(text) <= max_chars else text[:max_chars - 1].rstrip() + "…"
    words, lines, line = text.split(" "), [], ""
    for w in words:
        if len(line) + len(w) + 1 > 32 and line:
            lines.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    return "<br>".join(lines)


def sentence_pull(coords, sids, alpha):
    """Pull each token toward its sentence centroid so same-sentence tokens sit next
    to each other (alpha in [0,1]; 0 = no grouping). Convex blend keeps coords in
    the cube and preserves intra-sentence spread."""
    if alpha <= 0:
        return coords
    out = coords.copy()
    sids = np.asarray(sids)
    for s in np.unique(sids):
        m = sids == s
        out[m] = (1.0 - alpha) * coords[m] + alpha * coords[m].mean(0, keepdims=True)
    return out


# coreference highlight kinds: 0 = same token (gold), 1 = same meaning (magenta)
COREF_STYLE = [("rgba(240,170,20,0.95)", 5.0), ("rgba(220,70,200,0.85)", 3.0)]


def coreference_links(labels, sids, H, mode, sim_thresh):
    """Cross-sentence highlight links between separate nodes that mean the same:
      kind 0 (same token) — identical content token recurring in a different
        sentence; consecutive occurrences are chained.
      kind 1 (same meaning) — different token whose layer hidden state is most
        similar (cosine ≥ sim_thresh) in another sentence; one partner per token.
    Returns deduped undirected [i, j, kind]."""
    n = len(labels)
    norm = [re.sub(r"[^a-z0-9]", "", l.strip().lower()) for l in labels]
    links = []

    if mode in ("same", "both"):
        groups = {}
        for i, key in enumerate(norm):
            if key:
                groups.setdefault(key, []).append(i)
        for idxs in groups.values():
            prev = None
            for i in idxs:
                if prev is not None and sids[i] != sids[prev]:
                    links.append([prev, i, 0])
                prev = i

    if mode in ("meaning", "both") and H is not None and n > 1:
        X = H.numpy().astype("float64")
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        S = Xn @ Xn.T
        np.fill_diagonal(S, -1.0)
        for i in range(n):
            for j in np.argsort(-S[i])[:5]:
                if S[i, j] < sim_thresh:
                    break
                if sids[i] == sids[int(j)] or norm[i] == norm[int(j)]:
                    continue                       # same sentence or same token → skip
                links.append([i, int(j), 1])
                break

    seen, out = set(), []
    for i, j, k in links:
        a, b = (i, j) if i < j else (j, i)
        if (a, b, k) in seen:
            continue
        seen.add((a, b, k))
        out.append([a, b, k])
    return out


def cluster_tokens(attn_2d, H, mode, n_clusters, seed=0):
    """Group the kept tokens into meaningful units → returns int label per token.
      attention — community detection (weighted label-propagation) on the symmetrized
                  attention graph: tokens that attend to each other land together.
      embedding — k-means on the layer hidden states: semantically related words.
    Both are dependency-free."""
    n = (attn_2d.shape[0] if mode == "attention" else H.shape[0])
    if n <= 2:
        return np.zeros(n, dtype=int)
    if mode == "embedding":
        return _kmeans(H.numpy().astype("float64"), n_clusters or _auto_k(n), seed)
    return _label_prop(attn_2d.numpy().astype("float64"))


def _auto_k(n):
    return max(2, min(8, n // 5))


def _kmeans(X, k, seed=0, iters=50):
    """Plain Lloyd k-means with k-means++ init (numpy only, deterministic by seed)."""
    n = X.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    # k-means++ seeding
    centers = [int(rng.integers(n))]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - X[None, centers, :]) ** 2).sum(-1), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        centers.append(int(rng.choice(n, p=probs)))
    C = X[centers].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                C[c] = X[m].mean(0)
    _u, inv = np.unique(labels, return_inverse=True)
    return inv


def _label_prop(A, iters=30):
    """Weighted async label-propagation community detection on adjacency A. Each node
    adopts the label carrying the most attention weight among its neighbours; ties
    break to the lowest label for determinism. Isolated nodes stay singletons."""
    n = A.shape[0]
    W = A + A.T
    np.fill_diagonal(W, 0.0)
    lab = np.arange(n)
    for _ in range(iters):
        changed = False
        for i in range(n):
            nz = np.nonzero(W[i])[0]
            if nz.size == 0:
                continue
            sums = {}
            for j in nz:
                sums[lab[j]] = sums.get(lab[j], 0.0) + W[i, j]
            best = max(sums.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            if best != lab[i]:
                lab[i] = best
                changed = True
        if not changed:
            break
    _u, inv = np.unique(lab, return_inverse=True)
    return inv


# ── edge selection (the "omit meaningless attention" logic) ─────────────────────
def select_edges(attn_2d, threshold, top_k, drop_first, no_self):
    """attn_2d: [seq, seq] aggregated scores (rows=query i, cols=key j, causal so
    j<=i). Returns parallel arrays (qi, kj, score) for the edges worth drawing."""
    seq = attn_2d.shape[0]
    a = attn_2d.clone()

    if drop_first:
        a[:, 0] = 0.0                      # kill the attention-sink column
    if no_self:
        a.fill_diagonal_(0.0)

    if top_k is not None and top_k > 0:
        kept = torch.zeros_like(a, dtype=torch.bool)
        for i in range(seq):
            row = a[i]
            k = min(top_k, int((row > 0).sum().item()))
            if k > 0:
                idx = torch.topk(row, k).indices
                kept[i, idx] = True
        a = torch.where(kept, a, torch.zeros_like(a))

    qi, kj = torch.where(a > threshold)
    scores = a[qi, kj]
    return qi.tolist(), kj.tolist(), scores.tolist()


# ── 3D token projection + graph ─────────────────────────────────────────────────
def project_3d(H, method="pca", seed=0):
    """Project token hidden states H [seq, dim] → coords [seq, 3].

    Robust by design: LLMs park a massive-activation outlier on the first/sink
    token, and an ordinary PCA lets that one token define PC1 — everyone else then
    gets ~0 on PC1 and collapses into the PC2-PC3 plane (the "looks 2D" bug). So we
    fit the principal axes on INLIER tokens only (median/MAD gate), project all
    tokens onto those axes, then scale each axis by a robust spread and clip — one
    extreme point lands at the cube edge instead of squashing the rest flat."""
    Hn = H.numpy().astype("float64")
    n = Hn.shape[0]
    if method == "umap":
        try:
            import umap
        except ImportError:
            raise SystemExit("--proj umap needs umap-learn (`pip install umap-learn`); "
                             "or use --proj pca (no dependency)")
        coords = umap.UMAP(n_components=3, random_state=seed,
                           n_neighbors=min(15, max(2, n - 1))).fit_transform(Hn)
        return _robust_cube(coords)

    center = np.median(Hn, axis=0, keepdims=True)
    Xc = Hn - center
    dist = np.linalg.norm(Xc, axis=1)
    med = np.median(dist)
    mad = np.median(np.abs(dist - med)) + 1e-9
    inliers = dist <= med + 4.0 * 1.4826 * mad          # ~4σ robust gate
    if inliers.sum() < max(4, int(0.5 * n)):            # keep enough to fit 3 axes
        inliers = np.ones(n, dtype=bool)
    Xin = Xc[inliers]
    _U, _S, Vt = np.linalg.svd(Xin - Xin.mean(0, keepdims=True), full_matrices=False)
    coords = Xc @ Vt[:3].T                              # project ALL onto inlier axes
    return _robust_cube(coords)


def _robust_cube(coords):
    """Center on the median, scale each axis by a robust spread (90th-pct |dev|),
    clip to ±2.5 so an outlier can't stretch an axis, then map into a [0,1] cube."""
    c = np.asarray(coords, dtype="float64")
    c = c - np.median(c, axis=0, keepdims=True)
    scale = np.percentile(np.abs(c), 90, axis=0)
    scale[scale == 0] = 1.0
    c = np.clip(c / scale, -2.5, 2.5)
    c = c - c.min(0, keepdims=True)
    span = c.max(0, keepdims=True)
    span[span == 0] = 1.0
    return c / span


def project_graph(H, sids, labels, sent_weight=1.0, knn=8, sim_thresh=0.3):
    """Projection that MINIMIZES the distance between tokens that mean the same and/or
    share a sentence — Laplacian Eigenmaps on a graph whose edge weights encode
    "should be near":
      • semantic — cosine kNN of the layer hidden states (same-meaning tokens attract);
      • same surface token — a strong fixed boost;
      • same sentence — `sent_weight` added to every within-sentence pair.
    The spectral embedding y minimizes Σ wᵢⱼ‖yᵢ−yⱼ‖² (subject to a scale constraint),
    so heavily-weighted (same-meaning / same-sentence) pairs are pulled together. This
    replaces variance-preserving PCA, which has no notion of which tokens belong close.
    """
    Hn = H.numpy().astype("float64")
    n = Hn.shape[0]
    if n < 5:
        return project_3d(H, "pca")

    Xn = Hn / (np.linalg.norm(Hn, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -1.0)

    # semantic kNN affinity (symmetrized by max; negative/low sims dropped)
    W = np.zeros((n, n))
    k = min(knn, n - 1)
    for i in range(n):
        for j in np.argpartition(-S[i], k)[:k]:
            w = max(0.0, S[i, j] - sim_thresh)
            W[i, j] = W[j, i] = max(W[i, j], w)

    nrm = [re.sub(r"[^a-z0-9]", "", l.strip().lower()) for l in labels]
    sa = np.asarray(sids)
    same_tok = np.equal.outer(nrm, nrm) & np.array([[bool(x)] for x in nrm])
    same_sent = np.equal.outer(sa, sa)
    W = np.maximum(W, same_tok.astype(float))            # same token → strong pull
    W = W + sent_weight * same_sent                       # same sentence → pull
    np.fill_diagonal(W, 0.0)
    W += 1e-6                                              # keep the graph connected
    np.fill_diagonal(W, 0.0)

    # normalized-Laplacian eigenmaps: smallest non-trivial eigenvectors
    dinv = 1.0 / np.sqrt(W.sum(1) + 1e-12)
    Lsym = np.eye(n) - (dinv[:, None] * W * dinv[None, :])
    _vals, vecs = np.linalg.eigh(Lsym)                    # ascending
    Y = dinv[:, None] * vecs[:, 1:4]                      # drop trivial vec 0
    return _robust_cube(Y)


NBINS = 4   # edge strength buckets (controls line boldness, and click-rebuild)


def _cluster_rgba(cids):
    """One distinct color per CLUSTER (golden-angle hue spacing) as explicit rgba
    strings, so co-clustered tokens share a color and the click handler can grey-out
    non-neighbours and restore them by swapping colors."""
    uniq = sorted(set(int(c) for c in cids))
    hue = {c: (i * 0.61803) % 1.0 for i, c in enumerate(uniq)}
    out = []
    for c in cids:
        r, g, b = _hsl_to_rgb(hue[int(c)], 0.62, 0.55)
        out.append(f"rgba({r},{g},{b},0.95)")
    return out


def _hsl_to_rgb(h, s, l):
    def f(n):
        k = (n + h * 12) % 12
        x = l - s * min(l, 1 - l) * max(-1, min(k - 3, 9 - k, 1))
        return int(round(255 * x))
    return f(0), f(8), f(4)


def draw_graph3d_plotly(fig, row, col, labels, coords, attn_2d, cids, sids,
                        sent_texts, coref, args, smax):
    """Add one interactive 3D node-link scene and return its (metadata dict, n_edges).
    Nodes are colored by cluster (cids); same-sentence tokens are pulled together and
    the full sentence floats at each sentence centroid. Always emits the node trace +
    NBINS edge traces (some may start empty) so the click handler has a fixed set of
    traces to repopulate per bin."""
    import plotly.graph_objects as go
    seq = len(labels)
    qi, kj, scores = select_edges(attn_2d, args.threshold, args.top_k,
                                  args.drop_first, args.no_self)

    indeg = np.zeros(seq)
    for j, s in zip(kj, scores):
        indeg[j] += s
    msize = (4.0 + 9.0 * (indeg / indeg.max())) if indeg.max() > 0 else np.full(seq, 4.0)
    colors = _cluster_rgba(cids)

    node_idx = len(fig.data)
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text", text=list(labels), textposition="top center",
        textfont=dict(size=9), marker=dict(size=msize, color=colors,
                                           opacity=0.95, line=dict(width=0)),
        customdata=np.stack([np.arange(seq), np.asarray(cids), np.asarray(sids)], axis=1),
        hovertemplate="#%{customdata[0]} · cluster %{customdata[1]} · sentence "
                      "%{customdata[2]}: <b>%{text}</b><extra></extra>",
        showlegend=False), row=row, col=col)

    # bucket edges by strength; record (query, key, bin) for the click handler
    edges = []
    binned = [[] for _ in range(NBINS)]
    for i, j, s in zip(qi, kj, scores):
        b = min(NBINS - 1, int(NBINS * min(s / smax, 0.999))) if smax > 0 else 0
        binned[b].append((i, j))
        edges.append([int(i), int(j), int(b)])

    bin_idx, origX, origY, origZ = [], [], [], []
    for b in range(NBINS):
        frac = (b + 1) / NBINS
        xs, ys, zs = [], [], []
        for i, j in binned[b]:
            xs += [float(coords[j, 0]), float(coords[i, 0]), None]   # key → query
            ys += [float(coords[j, 1]), float(coords[i, 1]), None]
            zs += [float(coords[j, 2]), float(coords[i, 2]), None]
        bin_idx.append(len(fig.data))
        origX.append(xs); origY.append(ys); origZ.append(zs)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(width=1.0 + 6.0 * frac,
                      color=f"rgba(70,110,200,{0.08 + 0.72 * frac:.3f})"),
            hoverinfo="skip", showlegend=False), row=row, col=col)

    # distinct cross-sentence highlight links (same token / same meaning): one trace
    # per kind, always present so the click handler can repopulate them.
    coref_idx, cX, cY, cZ = [], [], [], []
    for kind, (kcolor, kwidth) in enumerate(COREF_STYLE):
        xs, ys, zs = [], [], []
        for i, j, k in coref:
            if k != kind:
                continue
            xs += [float(coords[i, 0]), float(coords[j, 0]), None]
            ys += [float(coords[i, 1]), float(coords[j, 1]), None]
            zs += [float(coords[i, 2]), float(coords[j, 2]), None]
        coref_idx.append(len(fig.data))
        cX.append(xs); cY.append(ys); cZ.append(zs)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(width=kwidth, color=kcolor),
            hoverinfo="skip", showlegend=False), row=row, col=col)

    # sentences: a faint THREAD through each sentence's tokens in reading order, and
    # the full-sentence label anchored at the sentence MEDOID (a real token, never
    # the empty geometric centroid that left labels floating in the middle of nowhere)
    if args.show_sentences:
        sa = np.asarray(sids)
        tx, ty, tz = [], [], []                       # thread segments
        lx, ly, lz, lt = [], [], [], []               # label anchors
        for s in np.unique(sa):
            idx = np.where(sa == s)[0]
            for a, b in zip(idx[:-1], idx[1:]):       # reading-order thread
                tx += [float(coords[a, 0]), float(coords[b, 0]), None]
                ty += [float(coords[a, 1]), float(coords[b, 1]), None]
                tz += [float(coords[a, 2]), float(coords[b, 2]), None]
            pts = coords[idx]                          # medoid = most central token
            med = idx[int(np.linalg.norm(pts[:, None] - pts[None], axis=-1).sum(1).argmin())]
            lx.append(float(coords[med, 0])); ly.append(float(coords[med, 1]))
            lz.append(float(coords[med, 2]) + 0.05)
            lt.append("<i>" + _wrap(sent_texts[int(s)], args.sentence_chars) + "</i>")
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz, mode="lines",
            line=dict(width=1.4, color="rgba(120,120,120,0.35)"),
            hoverinfo="skip", showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="text", text=lt,
            textfont=dict(size=10, color="rgba(30,30,30,0.9)"),
            hoverinfo="skip", showlegend=False), row=row, col=col)

    meta = dict(node=node_idx, bins=bin_idx, edges=edges,
                coords=np.round(coords, 4).tolist(), colors=colors,
                labels=list(labels), origX=origX, origY=origY, origZ=origZ,
                coref=coref_idx, corefEdges=[[int(i), int(j), int(k)] for i, j, k in coref],
                corefOrigX=cX, corefOrigY=cY, corefOrigZ=cZ)
    return meta, len(scores)


# JS injected into the HTML. NOTE: Plotly's scatter3d (gl3d/WebGL) traces do NOT
# emit `plotly_click` — only hover events work in 3D. So we track the hovered node
# via plotly_hover and fire the selection from a real DOM mouseup, distinguishing a
# click from a rotate-drag by mouse travel. Click a node → show only its incident
# edges + label its neighbours, grey the rest; click it again, click empty space, or
# double-click → reset. Per-panel via the PANELS trace-index map.
_CLICK_JS = """
var __divs = document.querySelectorAll('div.plotly-graph-div');
var gd = __divs[__divs.length - 1];
var PANELS = __PANELS__;
var lastHover = null, downXY = null;

function findPanel(cn) {
  for (var k = 0; k < PANELS.length; k++) if (PANELS[k].node === cn) return PANELS[k];
  return null;
}
function rebuild(p, traceIdx, edgeList, sel) {
  // set each trace to only the edges (from edgeList, tagged by 3rd field) incident
  // to `sel`; sel<0 means show all. tag t maps to traceIdx[t].
  for (var t = 0; t < traceIdx.length; t++) {
    var xs = [], ys = [], zs = [];
    edgeList.forEach(function (e) {
      if (e[2] !== t) return;
      if (sel >= 0 && e[0] !== sel && e[1] !== sel) return;
      var a = p.coords[e[0]], d = p.coords[e[1]];
      xs.push(a[0], d[0], null); ys.push(a[1], d[1], null); zs.push(a[2], d[2], null);
    });
    Plotly.restyle(gd, {x: [xs], y: [ys], z: [zs]}, [traceIdx[t]]);
  }
}
function resetPanel(p) {
  Plotly.restyle(gd, {'marker.color': [p.colors.slice()], 'text': [p.labels.slice()]}, [p.node]);
  for (var b = 0; b < p.bins.length; b++)
    Plotly.restyle(gd, {x: [p.origX[b]], y: [p.origY[b]], z: [p.origZ[b]]}, [p.bins[b]]);
  for (var c = 0; c < p.coref.length; c++)
    Plotly.restyle(gd, {x: [p.corefOrigX[c]], y: [p.corefOrigY[c]], z: [p.corefOrigZ[c]]}, [p.coref[c]]);
  p.sel = null;
}
function select(p, clicked) {
  if (p.sel === clicked) { resetPanel(p); return; }     // toggle off
  p.sel = clicked;
  var nb = {}; nb[clicked] = 1;
  p.edges.forEach(function (e) {
    if (e[0] === clicked) nb[e[1]] = 1;
    if (e[1] === clicked) nb[e[0]] = 1;
  });
  p.corefEdges.forEach(function (e) {                   // coref partners count too
    if (e[0] === clicked) nb[e[1]] = 1;
    if (e[1] === clicked) nb[e[0]] = 1;
  });
  var colors = p.colors.map(function (c, i) {
    return (i === clicked) ? 'rgba(220,30,30,1)' : (nb[i] ? c : 'rgba(200,200,200,0.06)');
  });
  var text = p.labels.map(function (t, i) { return nb[i] ? t : ''; });
  Plotly.restyle(gd, {'marker.color': [colors], 'text': [text]}, [p.node]);
  rebuild(p, p.bins, p.edges, clicked);
  rebuild(p, p.coref, p.corefEdges, clicked);
}
gd.on('plotly_hover', function (e) { if (e.points && e.points.length) lastHover = e.points[0]; });
gd.on('plotly_unhover', function () { lastHover = null; });
gd.addEventListener('mousedown', function (ev) { downXY = [ev.clientX, ev.clientY]; });
gd.addEventListener('mouseup', function (ev) {
  if (!downXY) return;
  var moved = Math.abs(ev.clientX - downXY[0]) + Math.abs(ev.clientY - downXY[1]);
  downXY = null;
  if (moved > 6) return;                         // a drag/rotate, not a click
  if (lastHover) {
    var p = findPanel(lastHover.curveNumber);
    if (p) { select(p, lastHover.pointNumber); return; }
  }
  PANELS.forEach(resetPanel);                     // clicked empty space → reset
});
gd.on('plotly_doubleclick', function () { PANELS.forEach(resetPanel); });
"""


def render_graph3d(labels, attns, hiddens, args):
    """Interactive 3D token graph → HTML: every token projected to a point in 3D
    (PCA/UMAP of its layer hidden state), attention drawn as lines between points.
    Nodes are clickable — click one to isolate just its connections."""
    try:
        import plotly.graph_objects as go      # noqa: F401
        from plotly.subplots import make_subplots
    except ImportError:
        raise SystemExit("the 3D token graph needs plotly — `pip install plotly`")

    # sentence index per token is computed on the FULL sequence (so boundaries are
    # right) then carried through the keep mask.
    sids_full, sent_texts = sentence_ids(labels)

    # 1) drop stopwords / punctuation / specials / sink, then slice attentions and
    #    hidden states down to the surviving content tokens (same set across layers).
    keep = keep_indices(labels, args.drop_stopwords)
    kt = torch.tensor(keep, dtype=torch.long)
    flabels = [labels[i] for i in keep]
    fsids = [sids_full[i] for i in keep]
    fattns = tuple(a.index_select(1, kt).index_select(2, kt) for a in attns)
    fhiddens = tuple(h.index_select(0, kt) for h in hiddens)
    print(f"kept {len(keep)}/{len(labels)} content tokens after stopword removal; "
          f"{len(set(fsids))} sentences")

    panels = build_panels(fattns, args)         # (title, attn_2d) with layer in title
    n = len(panels)
    ncols = min(n, args.max_cols)
    nrows = (n + ncols - 1) // ncols
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    smax = max((float(omit(m, args).max()) for _, m in panels), default=1.0) or 1.0
    specs = [[{"type": "scatter3d"} for _ in range(ncols)] for _ in range(nrows)]
    fig = make_subplots(rows=nrows, cols=ncols, specs=specs,
                        subplot_titles=[t for t, _ in panels])

    metas, total, ncl, ncoref = [], 0, 0, 0
    for idx, (title, mat) in enumerate(panels):
        L = resolve_layer(args.layers[0] if args.per_head else args.layers[idx],
                          len(fattns))
        # 1b) project to 3D. 'graph' minimizes distance between same-meaning /
        #     same-sentence tokens directly (Laplacian eigenmaps); pca/umap don't.
        if args.proj == "graph":
            coords = project_graph(fhiddens[L + 1], fsids, flabels,
                                   sent_weight=args.graph_sent_weight)
        else:
            coords = project_3d(fhiddens[L + 1], args.proj)
        # 2) gently pull same-sentence tokens together (the graph projection already
        #    groups them, so a lighter pull avoids collapsing sentences to points)
        pull = min(args.sentence_pull, 0.25) if args.proj == "graph" else args.sentence_pull
        coords = sentence_pull(coords, fsids, pull)
        # 3) cluster the surviving tokens into meaningful units
        if args.cluster == "none":
            cids = np.arange(len(flabels))
        else:
            cids = cluster_tokens(omit(mat, args), fhiddens[L + 1],
                                  args.cluster, args.n_clusters)
        ncl = max(ncl, len(set(int(c) for c in cids)))
        # 4) cross-sentence highlight links (same token / same meaning)
        coref = ([] if args.coref == "none" else
                 coreference_links(flabels, fsids, fhiddens[L + 1],
                                   args.coref, args.synonym_sim))
        ncoref += len(coref)
        meta, ne = draw_graph3d_plotly(fig, idx // ncols + 1, idx % ncols + 1,
                                       flabels, coords, mat, cids, fsids,
                                       sent_texts, coref, args, smax)
        metas.append(meta)
        total += ne

    fig.update_scenes(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                      xaxis=dict(showticklabels=False),
                      yaxis=dict(showticklabels=False),
                      zaxis=dict(showticklabels=False), aspectmode="cube")
    clu = "off" if args.cluster == "none" else f"{args.cluster} (~{ncl} clusters)"
    sub = ("  —  click a node to isolate its links · gold = same token across "
           "sentences, magenta = same meaning" if args.coref != "none"
           else "  —  click a node to isolate its links")
    fig.update_layout(
        title=(args.title or f"3D token graph · {args.model} · {args.proj.upper()} proj")
              + sub,
        height=560 * nrows, width=640 * ncols, margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="white")

    click_js = _CLICK_JS.replace("__PANELS__", json.dumps(metas))
    fig.write_html(args.out, include_plotlyjs=True, full_html=True, post_script=click_js)
    print(f"saved {args.out}  ({n} scenes, {total} edges, {ncoref} coref links, "
          f"{len(flabels)} tokens, proj={args.proj}, clustering={clu}) — "
          f"drag to rotate, click to isolate")


# ── rendering ───────────────────────────────────────────────────────────────────
def draw_panel(ax, labels, attn_2d, title, args, smax_global=None):
    """One arc panel: tokens listed top→bottom on both sides; an arc from key j
    (left) to query i (right). Line width and alpha scale with the score, so
    stronger attention literally reads as a bolder line."""
    seq = len(labels)
    qi, kj, scores = select_edges(
        attn_2d, args.threshold, args.top_k, args.drop_first, args.no_self
    )

    ys = list(range(seq))
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(seq - 0.5, -0.5)           # token 0 at top
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["key", "query"], fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    ax.tick_params(length=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)

    # endpoint dots
    ax.scatter([0.0] * seq, ys, s=6, color="0.6", zorder=2)
    ax.scatter([1.0] * seq, ys, s=6, color="0.6", zorder=2)

    if not scores:
        ax.text(0.5, seq / 2, "(no edges above threshold)",
                ha="center", va="center", fontsize=8, color="0.5")
        return 0

    # Normalize against the strongest edge so width/alpha are comparable across
    # panels when a global max is supplied (a per-layer max would make a weak
    # layer look as confident as a strong one).
    smax = smax_global if smax_global else max(scores)
    segs, widths, alphas = [], [], []
    for i, j, s in zip(qi, kj, scores):
        segs.append([(0.0, j), (1.0, i)])
        norm = min(s / smax, 1.0) if smax > 0 else 0.0
        widths.append(0.2 + 3.3 * norm)            # bold ∝ score
        alphas.append(0.06 + 0.84 * norm)          # faint weak edges out

    cmap = plt.get_cmap("viridis")
    colors = [(*cmap(min(s / smax, 1.0))[:3], a) for s, a in zip(scores, alphas)]
    lc = LineCollection(segs, linewidths=widths, colors=colors, zorder=1)
    ax.add_collection(lc)
    return len(scores)


def build_panels(attns, args):
    """Shared panel list: one (title, [seq,seq] matrix) per requested layer, or one
    per head of a single layer with --per-head."""
    n_layers = len(attns)
    if args.per_head:
        if len(args.layers) != 1:
            raise SystemExit("--per-head needs exactly one --layers value")
        L = resolve_layer(args.layers[0], n_layers)
        heads = attns[L]                          # [n_heads, seq, seq]
        return [(f"L{L} · head {h}", heads[h]) for h in range(heads.shape[0])]
    panels = []
    for spec in args.layers:
        L = resolve_layer(spec, n_layers)
        panels.append((f"layer {L}", aggregate_heads(attns[L], args.heads)))
    return panels


def omit(mat, args):
    """Apply the 'meaningless attention' omissions in-place on a clone: zero the
    token-0 sink column and/or the self-diagonal per the flags. Shared by both
    styles so the arc and surface views show the SAME matrix."""
    a = mat.clone()
    if args.drop_first:
        a[:, 0] = 0.0
    if args.no_self:
        a.fill_diagonal_(0.0)
    return a


def draw_surface_panel(ax, labels, attn_2d, title, args, zmax):
    """One 3D word→word panel: key tokens on X, query tokens on Y, attention score
    as HEIGHT (Z). Local attention shows as a diagonal ridge, an anchor/sink token
    as a wall along its key column, induction as off-diagonal stripes."""
    seq = len(labels)
    Z = omit(attn_2d, args).numpy()               # Z[query, key]
    X, Y = np.meshgrid(np.arange(seq), np.arange(seq))
    ax.plot_surface(X, Y, Z, cmap="viridis", vmin=0.0, vmax=zmax,
                    rstride=1, cstride=1, linewidth=0.0, antialiased=True)
    ax.set_zlim(0.0, zmax)
    ax.set_title(title, fontsize=10, pad=0)

    step = max(1, seq // args.max_ticks)          # thin labels so axes stay legible
    ticks = list(range(0, seq, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], fontsize=6, rotation=90,
                       va="center", ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=6)
    ax.set_xlabel("key (attended-to)", fontsize=8, labelpad=10)
    ax.set_ylabel("query (attending)", fontsize=8, labelpad=10)
    ax.set_zlabel("attn", fontsize=8)
    ax.tick_params(labelsize=6, pad=-1)
    ax.view_init(elev=args.elev, azim=args.azim)


def render(labels, attns, hiddens, args):
    """Dispatch on style/backend:
      --style graph  → interactive 3D token node-link graph (tokens projected to
                       points in 3D, attention as lines between them) — the 'real 3D'.
      --style surface/arc → matrix views (3D terrain or 2D arcs); .html out (or
                       --backend plotly) gives the interactive terrain, else a PNG."""
    if args.style == "graph":
        return render_graph3d(labels, attns, hiddens, args)
    use_plotly = (args.backend == "plotly" or
                  (args.backend == "auto" and args.out.lower().endswith(".html")))
    if use_plotly:
        return render_plotly(labels, attns, args)
    return render_matplotlib(labels, attns, args)


def render_plotly(labels, attns, args):
    """Interactive 3D word→word surface(s) written to a self-contained HTML. Drag to
    rotate, scroll to zoom, and HOVER a peak to read the exact (query word, key word,
    score) — the part a static image can't give you. One scene per layer/head."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise SystemExit("interactive HTML needs plotly — `pip install plotly`, or "
                         "write a static image with --out …png")

    panels = build_panels(attns, args)
    n = len(panels)
    ncols = min(n, args.max_cols)
    nrows = (n + ncols - 1) // ncols
    seq = len(labels)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    vals = np.concatenate([omit(m, args).numpy().ravel() for _, m in panels])
    pos = vals[vals > 0]
    zmax = max(float(np.percentile(pos, args.zmax_pct)) if pos.size else 1.0, 1e-6)

    # per-cell token strings for the hover box (query = row i, key = col j)
    lab = np.array(labels, dtype=object)
    qlab = np.tile(lab.reshape(seq, 1), (1, seq))     # [i,j] -> labels[i]
    klab = np.tile(lab.reshape(1, seq), (seq, 1))     # [i,j] -> labels[j]
    customdata = np.dstack([qlab, klab])
    hover = ("query: <b>%{customdata[0]}</b><br>"
             "key:   <b>%{customdata[1]}</b><br>"
             "attn:  %{z:.3f}<extra></extra>")

    specs = [[{"type": "surface"} for _ in range(ncols)] for _ in range(nrows)]
    fig = make_subplots(rows=nrows, cols=ncols, specs=specs,
                        subplot_titles=[t for t, _ in panels])
    for idx, (title, mat) in enumerate(panels):
        Z = omit(mat, args).numpy()
        fig.add_trace(
            go.Surface(z=Z, customdata=customdata, colorscale="Viridis",
                       cmin=0.0, cmax=zmax, showscale=(idx == 0),
                       colorbar=dict(title="attn", len=0.6),
                       hovertemplate=hover),
            row=idx // ncols + 1, col=idx % ncols + 1,
        )

    step = max(1, seq // args.max_ticks)              # thin axis ticks
    ticks = list(range(0, seq, step))
    ticktext = [labels[i] for i in ticks]
    # all scenes share the same token set, so one update covers every panel
    fig.update_scenes(
        xaxis=dict(title="key (attended-to)", tickmode="array",
                   tickvals=ticks, ticktext=ticktext),
        yaxis=dict(title="query (attending)", tickmode="array",
                   tickvals=ticks, ticktext=ticktext),
        zaxis=dict(title="attn", range=[0.0, zmax]),
        aspectmode="cube",
    )
    fig.update_layout(
        title=args.title or f"Attention · {args.model} model · interactive 3D",
        height=520 * nrows, width=620 * ncols, margin=dict(l=0, r=0, t=60, b=0),
    )
    # embed plotly.js so the file opens offline after you scp it off the cluster
    fig.write_html(args.out, include_plotlyjs=True, full_html=True)
    print(f"saved {args.out}  ({n} interactive panels, "
          f"z-scale@{args.zmax_pct}%={zmax:.3f}, seq_len={seq}) — open in a browser")


def render_matplotlib(labels, attns, args):
    """Build a STATIC figure. --style arc → 2D arc panels; --style surface → 3D
    word→word terrain panels. One panel per requested layer (or per head)."""
    panels = build_panels(attns, args)
    n = len(panels)
    ncols = min(n, args.max_cols)
    nrows = (n + ncols - 1) // ncols
    seq = len(labels)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.style == "surface":
        # robust global z-scale: a high percentile of the displayed (post-omission)
        # mass so one residual spike can't flatten every other panel.
        vals = np.concatenate([omit(m, args).numpy().ravel() for _, m in panels])
        pos = vals[vals > 0]
        zmax = float(np.percentile(pos, args.zmax_pct)) if pos.size else 1.0
        zmax = max(zmax, 1e-6)
        fig = plt.figure(figsize=(args.panel_width * 1.7 * ncols,
                                  args.panel_width * 1.5 * nrows))
        for idx, (title, mat) in enumerate(panels):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
            draw_surface_panel(ax, labels, mat, title, args, zmax)
        note = f"{n} panels, z-scale@{args.zmax_pct}%={zmax:.3f}, seq_len={seq}"
    else:
        smax_global = max((float(omit(m, args).max()) for _, m in panels), default=0.0)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(args.panel_width * ncols, max(3.0, 0.22 * seq) * nrows),
            squeeze=False,
        )
        total_edges = 0
        for idx, (title, mat) in enumerate(panels):
            ax = axes[idx // ncols][idx % ncols]
            total_edges += draw_panel(ax, labels, mat, title, args, smax_global)
        for idx in range(n, nrows * ncols):        # blank unused cells
            axes[idx // ncols][idx % ncols].axis("off")
        note = f"{n} panels, {total_edges} edges, seq_len={seq}"

    fig.suptitle(args.title or f"Attention · {args.model} model · {args.style}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {args.out}  ({note})")


def aggregate_heads(layer_attn, heads_spec):
    """layer_attn: [n_heads, seq, seq] → [seq, seq]. mean over all heads, or over
    an explicit subset list."""
    if heads_spec == "mean":
        return layer_attn.mean(0)
    idx = [int(h) for h in heads_spec]
    return layer_attn[idx].mean(0)


def resolve_layer(spec, n_layers):
    L = int(spec)
    if L < 0:
        L += n_layers
    if not (0 <= L < n_layers):
        raise SystemExit(f"layer {spec} out of range (model has {n_layers} layers)")
    return L


# ── cli ──────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", default=None, help="input string to visualize")
    src.add_argument("--text-file", default=None, help="read input from a file")
    p.add_argument("--model", choices=["source", "target"], default="source",
                   help="source=Qwen (1 GPU), target=DeepSeek-70B (sharded)")
    p.add_argument("--model-name", default=None, help="override the HF model id")
    p.add_argument("--device", default="cuda:0",
                   help="GPU for the source SLM (default cuda:0; expose it with "
                        "CUDA_VISIBLE_DEVICES). Ignored for --model target.")
    p.add_argument("--layers", type=int, nargs="+", default=[0, -1],
                   help="layer indices to plot (negatives count from the end)")
    p.add_argument("--heads", nargs="+", default=["mean"],
                   help="'mean' (all heads) or an explicit head-index subset")
    p.add_argument("--per-head", action="store_true",
                   help="plot every head of a single layer instead of layer panels")
    p.add_argument("--style", choices=["graph", "surface", "arc"], default="graph",
                   help="graph=interactive 3D token node-link graph (default); "
                        "surface=3D matrix terrain; arc=2D arc diagram")
    p.add_argument("--proj", choices=["graph", "pca", "umap"], default="graph",
                   help="3D layout for --style graph. graph (default): Laplacian "
                        "eigenmaps that MINIMIZE distance between same-meaning and "
                        "same-sentence tokens; pca: variance-preserving; umap: "
                        "neighborhood embedding (needs umap-learn)")
    p.add_argument("--graph-sent-weight", type=float, default=1.0,
                   help="--proj graph: attraction added to every same-sentence pair "
                        "(higher → sentences pack tighter; default 1.0)")
    p.add_argument("--cluster", choices=["attention", "embedding", "none"],
                   default="attention",
                   help="group surviving tokens into meaningful units, colored per "
                        "cluster: attention=community detection on the attention "
                        "graph; embedding=k-means on hidden states; none=off")
    p.add_argument("--n-clusters", type=int, default=0,
                   help="k for --cluster embedding (0 = auto by token count)")
    p.add_argument("--drop-stopwords", dest="drop_stopwords", action="store_true",
                   default=True, help="drop stopwords before the graph (default on)")
    p.add_argument("--keep-stopwords", dest="drop_stopwords", action="store_false",
                   help="keep stopwords/function words in the graph")
    p.add_argument("--sentence-pull", type=float, default=0.6,
                   help="0..1 cohesion pulling same-sentence tokens together so each "
                        "sentence reads as a tight group (0 = off, default 0.6)")
    p.add_argument("--show-sentences", dest="show_sentences", action="store_true",
                   default=True, help="float the full sentence text on the map (on)")
    p.add_argument("--no-sentences", dest="show_sentences", action="store_false",
                   help="hide the floating sentence labels")
    p.add_argument("--sentence-chars", type=int, default=80,
                   help="max chars of each sentence shown in its floating label")
    p.add_argument("--coref", choices=["same", "meaning", "both", "none"],
                   default="both",
                   help="distinct cross-sentence highlight links: same=identical "
                        "token recurring in another sentence (gold); meaning=most "
                        "similar token in another sentence (magenta); both; none")
    p.add_argument("--synonym-sim", type=float, default=0.6,
                   help="cosine threshold for --coref meaning/both (0..1, default 0.6)")
    p.add_argument("--backend", choices=["auto", "plotly", "matplotlib"],
                   default="auto",
                   help="surface/arc only: auto = .html→plotly, else static image")
    p.add_argument("--elev", type=float, default=35.0, help="3D view elevation")
    p.add_argument("--azim", type=float, default=-60.0, help="3D view azimuth")
    p.add_argument("--zmax-pct", type=float, default=99.0,
                   help="3D height scale = this percentile of attention mass "
                        "(lower → taller, more sensitive terrain)")
    p.add_argument("--max-ticks", type=int, default=24,
                   help="max token labels per 3D axis (thinned if seq is longer)")
    p.add_argument("--threshold", type=float, default=0.02,
                   help="omit edges with score below this (default 0.02)")
    p.add_argument("--top-k", type=int, default=None,
                   help="additionally keep only each query's K strongest keys")
    # sink/diagonal omission. Defaults are style-aware (resolved below): the sink is
    # dropped in both styles (it flattens the height scale / clutters arcs); the
    # self-diagonal is KEPT in 3D (local attention is the clearest terrain pattern)
    # but dropped in arcs. Either can be forced on/off explicitly.
    p.add_argument("--drop-first", dest="drop_first", action="store_const", const=True,
                   default=None, help="force-drop the token-0 attention sink")
    p.add_argument("--keep-first", dest="drop_first", action="store_const", const=False,
                   help="force-keep the token-0 attention sink")
    p.add_argument("--drop-self", dest="no_self", action="store_const", const=True,
                   default=None, help="force-drop the i→i self-diagonal")
    p.add_argument("--keep-self", dest="no_self", action="store_const", const=False,
                   help="force-keep the i→i self-diagonal")
    p.add_argument("--max-length", type=int, default=128,
                   help="truncate input to this many tokens (plots get dense fast)")
    p.add_argument("--max-cols", type=int, default=4)
    p.add_argument("--panel-width", type=float, default=3.2)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--title", default=None)
    p.add_argument("--out", default="proofs/data/attention_map.html",
                   help="output path; .html → interactive 3D, .png → static image")
    args = p.parse_args()

    if args.text_file:
        args.text = Path(args.text_file).read_text()
    if not args.text:
        args.text = DEFAULT_TEXT
    if args.heads == ["mean"]:
        args.heads = "mean"                       # all heads
    else:
        args.heads = [int(h) for h in args.heads]

    # style-aware omission defaults (None == user didn't force it)
    if args.drop_first is None:
        args.drop_first = True                     # sink clutters every view
    if args.no_self is None:
        # the self-diagonal is real terrain in the surface view, but a meaningless
        # zero-length edge in the graph and clutter in arcs — keep it only for surface
        args.no_self = (args.style != "surface")

    cfg = StitcherConfig()
    tokenizer, model = load_model(cfg, args.model, args.model_name, args.device)
    labels, attns, hiddens = get_outputs(model, tokenizer, args.text, args.max_length)
    print(f"{len(attns)} layers · {attns[0].shape[0]} heads · seq_len {len(labels)}")
    render(labels, attns, hiddens, args)


if __name__ == "__main__":
    main()

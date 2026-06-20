"""
Side experiment — visualize a model's attention as a real 3D token graph.

Not a receiver-validation proof; a diagnostic/illustration that reuses the
project's own models (`StitcherConfig.source_model` by default — Qwen2.5-7B, the
stitcher's *source* side, which fits on one GPU). One forward pass exposes both
attentions and hidden states, and three view styles render them:

  graph   (default) — the REAL 3D view: every token is projected to a point in 3D
            space (PCA — or UMAP — of its layer hidden state, so related tokens sit
            near each other), and attention is drawn as LINES between those points,
            each line's width/opacity growing with the score. Written as a
            self-contained interactive HTML: drag to rotate, scroll to zoom, hover a
            node for its token. One 3D scene per requested layer.
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
    # interactive 3D token graph (default) — open the .html in a browser
    python proofs/attention_map.py \
        --text "The Eiffel Tower is in Paris. It was completed in 1889." \
        --layers 6 14 24 --top-k 3 --out proofs/data/attn_graph.html

    # cleaner clusters with UMAP (pip install umap-learn)
    python proofs/attention_map.py --proj umap --layers 14 --out proofs/data/g.html

    # the matrix views instead
    python proofs/attention_map.py --style surface --layers 4 14 24 --out s.html
    python proofs/attention_map.py --style arc --layers 14 --out a.png
"""

from __future__ import annotations   # `str | None` annotations on any 3.7+ interp

import os
import sys
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
    """Project token hidden states H [seq, dim] → coords [seq, 3]. Dependency-free
    PCA (centered SVD, top-3 components) by default; UMAP if --proj umap and the
    package is installed (better cluster separation, needs `pip install umap-learn`)."""
    Hn = H.numpy().astype("float64")
    if method == "umap":
        try:
            import umap
        except ImportError:
            raise SystemExit("--proj umap needs umap-learn (`pip install umap-learn`); "
                             "or use --proj pca (no dependency)")
        n = Hn.shape[0]
        reducer = umap.UMAP(n_components=3, random_state=seed,
                            n_neighbors=min(15, max(2, n - 1)))
        coords = reducer.fit_transform(Hn)
    else:
        X = Hn - Hn.mean(0, keepdims=True)
        # right singular vectors give the principal axes; project onto top 3
        _U, _S, Vt = np.linalg.svd(X, full_matrices=False)
        coords = X @ Vt[:3].T
    # normalize each axis to a comparable cube so panels look consistent
    coords = coords - coords.min(0, keepdims=True)
    span = coords.max(0, keepdims=True)
    span[span == 0] = 1.0
    return coords / span


def draw_graph3d_plotly(fig, row, col, labels, coords, attn_2d, args, smax, first):
    """One interactive 3D node-link scene: tokens as points at their projected
    coords, attention as lines between them. Bold/opaque line ∝ score."""
    import plotly.graph_objects as go
    seq = len(labels)
    qi, kj, scores = select_edges(attn_2d, args.threshold, args.top_k,
                                  args.drop_first, args.no_self)

    # nodes: marker size ∝ attention received (in-degree), colored by token order
    indeg = np.zeros(seq)
    for j, s in zip(kj, scores):
        indeg[j] += s
    msize = 4.0 + 9.0 * (indeg / indeg.max() if indeg.max() > 0 else indeg)
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text", text=labels, textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=msize, color=list(range(seq)), colorscale="Turbo",
                    opacity=0.9, line=dict(width=0)),
        customdata=np.arange(seq),
        hovertemplate="#%{customdata}: <b>%{text}</b><extra></extra>",
        showlegend=False), row=row, col=col)

    if not scores:
        return 0
    # Variable line boldness: plotly width is per-trace, so bucket edges into a few
    # strength bins — one line trace per bin, segments separated by None.
    nbins = 4
    bins = [[] for _ in range(nbins)]
    for i, j, s in zip(qi, kj, scores):
        b = min(nbins - 1, int(nbins * min(s / smax, 0.999))) if smax > 0 else 0
        bins[b].append((i, j))
    for b, pairs in enumerate(bins):
        if not pairs:
            continue
        frac = (b + 1) / nbins
        xs, ys, zs = [], [], []
        for i, j in pairs:
            xs += [coords[j, 0], coords[i, 0], None]      # key → query
            ys += [coords[j, 1], coords[i, 1], None]
            zs += [coords[j, 2], coords[i, 2], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(width=1.0 + 6.0 * frac,
                      color=f"rgba(70,110,200,{0.08 + 0.72 * frac:.3f})"),
            hoverinfo="skip", showlegend=False), row=row, col=col)
    return len(scores)


def render_graph3d(labels, attns, hiddens, args):
    """Interactive 3D token graph → HTML: every token projected to a point in 3D
    (PCA/UMAP of its layer hidden state), attention drawn as lines between points."""
    try:
        import plotly.graph_objects as go      # noqa: F401
        from plotly.subplots import make_subplots
    except ImportError:
        raise SystemExit("the 3D token graph needs plotly — `pip install plotly`")

    panels = build_panels(attns, args)          # (title, attn_2d) with layer in title
    n = len(panels)
    ncols = min(n, args.max_cols)
    nrows = (n + ncols - 1) // ncols
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    smax = max((float(omit(m, args).max()) for _, m in panels), default=1.0) or 1.0
    specs = [[{"type": "scatter3d"} for _ in range(ncols)] for _ in range(nrows)]
    fig = make_subplots(rows=nrows, cols=ncols, specs=specs,
                        subplot_titles=[t for t, _ in panels])

    total = 0
    for idx, (title, mat) in enumerate(panels):
        L = resolve_layer(args.layers[0] if args.per_head else args.layers[idx],
                          len(attns))
        coords = project_3d(hiddens[L + 1], args.proj)     # layer-L output states
        total += draw_graph3d_plotly(fig, idx // ncols + 1, idx % ncols + 1,
                                     labels, coords, mat, args, smax, idx == 0)

    fig.update_scenes(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                      xaxis=dict(showticklabels=False),
                      yaxis=dict(showticklabels=False),
                      zaxis=dict(showticklabels=False), aspectmode="cube")
    fig.update_layout(
        title=args.title or f"3D token graph · {args.model} · {args.proj.upper()} proj",
        height=560 * nrows, width=640 * ncols, margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="white")
    fig.write_html(args.out, include_plotlyjs=True, full_html=True)
    print(f"saved {args.out}  ({n} 3D scenes, {total} edges, {len(labels)} tokens, "
          f"proj={args.proj}) — drag to rotate, hover a node for its token")


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
    p.add_argument("--proj", choices=["pca", "umap"], default="pca",
                   help="how to place tokens in 3D for --style graph "
                        "(pca: no deps; umap: better clusters, needs umap-learn)")
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

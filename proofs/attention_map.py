"""
Side experiment — visualize the attention map at chosen layers.

Not a receiver-validation proof; a diagnostic/illustration that reuses the
project's own models (`StitcherConfig.source_model` by default — Qwen2.5-7B, the
stitcher's *source* side, which fits on one GPU). For a given input string it
runs one forward pass with attentions exposed, then draws, per requested layer,
a bertviz-style arc diagram: the token sequence down both sides, an arc from key
token j to query token i whose **line weight and opacity grow with the attention
score**. Weak, structural, or sink attention is omitted so only the edges that
actually carry signal remain visible.

Why "eager". SDPA / FlashAttention fuse the softmax and never materialize the
[heads, q, k] weight matrix, so `output_attentions=True` silently returns None
under them. We force `attn_implementation="eager"` — slower, but it is the only
implementation that hands back the weights we are here to look at.

What gets omitted (the "meaningless attention" the prompt asks to drop):
  --drop-first   the attention *sink*: almost every head dumps mass on token 0
                 (here, the BOS/first token). Kept it would dominate every plot
                 and hide the content-bearing edges. On by default.
  --no-self      the diagonal i→i self-loops. On by default.
  --threshold T  any edge below score T is dropped (absolute cut).
  --top-k K      additionally, per query keep only its K strongest remaining keys.
Aggregation across heads is mean by default (`--heads mean`); pass an explicit
list (`--heads 3 7 12`) to average a subset, or `--per-head` to draw every head
of a single layer as its own panel instead.

Usage:
    python proofs/attention_map.py \
        --text "The Eiffel Tower is in Paris. It was completed in 1889." \
        --layers 0 14 27 --out proofs/data/attn_qwen.png

    # the target (DeepSeek-70B) side instead — needs the sharded GPUs:
    python proofs/attention_map.py --model target --layers 20 40 60 \
        --text-file some_doc.txt --out proofs/data/attn_deepseek.png

    # every head of one layer, side by side:
    python proofs/attention_map.py --layers 14 --per-head --out proofs/data/heads.png
"""

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

from config import StitcherConfig


DEFAULT_TEXT = (
    "The Eiffel Tower is located in Paris, the capital of France. "
    "It was completed in 1889 and is named after the engineer Gustave Eiffel."
)


# ── model loading ───────────────────────────────────────────────────────────────
def load_model(cfg: StitcherConfig, which: str, model_name: str | None):
    """Load `source` (Qwen, single GPU) or `target` (DeepSeek-70B, sharded) with
    attentions exposed. Eager attention is mandatory — see module docstring."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype = getattr(torch, cfg.dtype)
    name = model_name or (cfg.source_model if which == "source" else cfg.target_model)
    tokenizer = AutoTokenizer.from_pretrained(name)

    if which == "source":
        device_map = cfg.source_device
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
        output_attentions=True,
    )
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def get_attentions(model, tokenizer, text: str, max_length: int):
    """One forward pass. Returns (token_labels, attentions) where attentions is a
    tuple of length n_layers, each tensor [n_heads, seq, seq] on CPU float32."""
    first_device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(first_device) for k, v in enc.items()}
    out = model(**enc, output_attentions=True)

    ids = enc["input_ids"][0].tolist()
    labels = clean_token_labels(tokenizer, ids)
    attns = tuple(a[0].float().cpu() for a in out.attentions)   # drop batch dim
    return labels, attns


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


def render(labels, attns, args):
    """Build the figure: one panel per requested layer, or — with --per-head — one
    panel per head of the single requested layer."""
    n_layers = len(attns)

    if args.per_head:
        if len(args.layers) != 1:
            raise SystemExit("--per-head needs exactly one --layers value")
        L = resolve_layer(args.layers[0], n_layers)
        heads = attns[L]                          # [n_heads, seq, seq]
        panels = [(f"L{L} · head {h}", heads[h]) for h in range(heads.shape[0])]
    else:
        panels = []
        for spec in args.layers:
            L = resolve_layer(spec, n_layers)
            panels.append((f"layer {L}", aggregate_heads(attns[L], args.heads)))

    # one global scale so "bolder = stronger" is comparable across panels
    smax_global = 0.0
    for _, mat in panels:
        a = mat.clone()
        if args.drop_first:
            a[:, 0] = 0.0
        if args.no_self:
            a.fill_diagonal_(0.0)
        smax_global = max(smax_global, float(a.max()))

    n = len(panels)
    ncols = min(n, args.max_cols)
    nrows = (n + ncols - 1) // ncols
    seq = len(labels)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(args.panel_width * ncols, max(3.0, 0.22 * seq) * nrows),
        squeeze=False,
    )
    total_edges = 0
    for idx, (title, mat) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        total_edges += draw_panel(ax, labels, mat, title, args, smax_global)
    for idx in range(n, nrows * ncols):            # blank unused cells
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(args.title or f"Attention map · {args.model} model", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {args.out}  ({n} panels, {total_edges} edges, seq_len={seq})")


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
    p.add_argument("--layers", type=int, nargs="+", default=[0, -1],
                   help="layer indices to plot (negatives count from the end)")
    p.add_argument("--heads", nargs="+", default=["mean"],
                   help="'mean' (all heads) or an explicit head-index subset")
    p.add_argument("--per-head", action="store_true",
                   help="plot every head of a single layer instead of layer panels")
    p.add_argument("--threshold", type=float, default=0.02,
                   help="omit edges with score below this (default 0.02)")
    p.add_argument("--top-k", type=int, default=None,
                   help="additionally keep only each query's K strongest keys")
    p.add_argument("--keep-first", dest="drop_first", action="store_false",
                   help="keep the token-0 attention sink (omitted by default)")
    p.add_argument("--keep-self", dest="no_self", action="store_false",
                   help="keep the i→i self-loops (omitted by default)")
    p.add_argument("--max-length", type=int, default=128,
                   help="truncate input to this many tokens (plots get dense fast)")
    p.add_argument("--max-cols", type=int, default=4)
    p.add_argument("--panel-width", type=float, default=3.2)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--title", default=None)
    p.add_argument("--out", default="proofs/data/attention_map.png")
    p.set_defaults(drop_first=True, no_self=True)
    args = p.parse_args()

    if args.text_file:
        args.text = Path(args.text_file).read_text()
    if not args.text:
        args.text = DEFAULT_TEXT
    if args.heads == ["mean"]:
        args.heads = "mean"                       # all heads
    else:
        args.heads = [int(h) for h in args.heads]

    cfg = StitcherConfig()
    tokenizer, model = load_model(cfg, args.model, args.model_name)
    labels, attns = get_attentions(model, tokenizer, args.text, args.max_length)
    print(f"{len(attns)} layers · {attns[0].shape[0]} heads · seq_len {len(labels)}")
    render(labels, attns, args)


if __name__ == "__main__":
    main()

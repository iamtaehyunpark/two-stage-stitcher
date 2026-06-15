"""
Phase 1: Extract paired hidden states from Qwen2.5-7B and Llama-70B.

For each document, generates progressive cumulative chunks:
  [chunk_1], [chunk_1 + chunk_2], [chunk_1 + chunk_2 + chunk_3], ...

Extracts the last-token hidden state at:
  - Qwen: final attention layer  (shape: src_dim)
  - Llama: layer `target_layer`  (shape: tgt_dim)

Saves pairs to data_dir as .npz files, one per document.
"""

import os
import argparse
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List, Tuple

from config import StitcherConfig


def load_source_model(cfg: StitcherConfig):
    dtype = getattr(torch, cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.source_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.source_model,
        torch_dtype=dtype,
        device_map=cfg.source_device,
        output_hidden_states=True,
    )
    model.eval()
    return tokenizer, model


def load_target_model(cfg: StitcherConfig):
    dtype = getattr(torch, cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
    max_memory = {i: "70GiB" for i in cfg.llama_devices}
    model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model,
        torch_dtype=dtype,
        device_map="sequential",
        max_memory=max_memory,
        output_hidden_states=True,
    )
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def extract_qwen_hidden(
    model,
    tokenizer,
    text: str,
    device: str,
) -> Tensor:
    """Returns last-token hidden state from Qwen's final layer. Shape: (src_dim,)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=131072)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    last_layer_hs = out.hidden_states[-1]       # (1, seq_len, src_dim)
    return last_layer_hs[0, -1, :].float().cpu()


@torch.inference_mode()
def extract_llama_hidden(
    model,
    tokenizer,
    text: str,
    target_layer: int,
) -> Tensor:
    """Returns last-token hidden state from Llama at `target_layer`. Shape: (tgt_dim,)."""
    first_device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    out = model(**inputs)
    hs_at_layer = out.hidden_states[target_layer + 1]   # (1, seq_len, tgt_dim)
    return hs_at_layer[0, -1, :].float().cpu()


def progressive_chunks(tokens: List[int], chunk_size: int, max_chunks: int) -> List[List[int]]:
    """Returns cumulative progressive token windows."""
    chunks = [tokens[i * chunk_size: (i + 1) * chunk_size] for i in range(max_chunks)]
    chunks = [c for c in chunks if c]
    cumulative = []
    acc = []
    for c in chunks:
        acc = acc + c
        cumulative.append(acc)
    return cumulative


def process_document(
    doc_text: str,
    qwen_tok, qwen_model,
    llama_tok, llama_model,
    cfg: StitcherConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns X (N, src_dim) and Y (N, tgt_dim) for one document.
    N = number of progressive chunks.
    """
    token_ids = qwen_tok.encode(doc_text)
    windows = progressive_chunks(token_ids, cfg.chunk_size, cfg.max_chunks_per_doc)

    X, Y = [], []
    for window_ids in windows:
        chunk_text = qwen_tok.decode(window_ids, skip_special_tokens=True)

        x = extract_qwen_hidden(qwen_model, qwen_tok, chunk_text, cfg.source_device)
        y = extract_llama_hidden(llama_model, llama_tok, chunk_text, cfg.target_layer)

        X.append(x.numpy())
        Y.append(y.numpy())

    return np.stack(X), np.stack(Y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", nargs="+", required=True,
                        help="Paths to plain-text document files")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = StitcherConfig()
    out_dir = args.out_dir or cfg.data_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Qwen model …")
    qwen_tok, qwen_model = load_source_model(cfg)

    print("Loading Llama model …")
    llama_tok, llama_model = load_target_model(cfg)

    for doc_path in tqdm(args.documents, desc="Documents"):
        with open(doc_path) as f:
            text = f.read()

        X, Y = process_document(text, qwen_tok, qwen_model, llama_tok, llama_model, cfg)

        stem = os.path.splitext(os.path.basename(doc_path))[0]
        out_path = os.path.join(out_dir, f"{stem}.npz")
        np.savez_compressed(out_path, X=X, Y=Y)
        print(f"  saved {X.shape[0]} pairs → {out_path}")


if __name__ == "__main__":
    main()

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
def extract_qwen_hidden_batch(
    model,
    tokenizer,
    texts: List[str],
    device: str,
) -> Tensor:
    """
    Batch forward pass through Qwen. Returns last-token hidden states, shape (N, src_dim).
    Chunks are padded to the longest sequence in the batch.
    """
    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                       max_length=131072, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    last_layer_hs = out.hidden_states[-1]           # (N, seq_len, src_dim)
    # Last non-padding token per sequence
    seq_lens = inputs["attention_mask"].sum(dim=1) - 1   # (N,)
    last_tokens = last_layer_hs[torch.arange(len(texts)), seq_lens]  # (N, src_dim)
    return last_tokens.float().cpu()


@torch.inference_mode()
def extract_llama_hidden_batch(
    model,
    tokenizer,
    texts: List[str],
    target_layer: int,
) -> Tensor:
    """
    Batch forward pass through Llama. Returns hidden states at target_layer, shape (N, tgt_dim).
    """
    first_device = next(model.parameters()).device
    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                       max_length=8192, padding=True)
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    out = model(**inputs)
    hs_at_layer = out.hidden_states[target_layer + 1]    # (N, seq_len, tgt_dim)
    seq_lens = inputs["attention_mask"].sum(dim=1) - 1   # (N,)
    last_tokens = hs_at_layer[torch.arange(len(texts)), seq_lens]    # (N, tgt_dim)
    return last_tokens.float().cpu()


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

    texts = [qwen_tok.decode(ids, skip_special_tokens=True) for ids in windows]

    X = extract_qwen_hidden_batch(qwen_model, qwen_tok, texts, cfg.source_device)
    Y = extract_llama_hidden_batch(llama_model, llama_tok, texts, cfg.target_layer)

    return X.numpy(), Y.numpy()


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

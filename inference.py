"""
Downstream inference: inject translated hidden state into Llama at `target_layer`,
bypassing native text prefill for the document context.

The injection exploits a split-forward approach:
  1. Document context  → Qwen → LatentStitcher → x_final  (precomputed, cached)
  2. Query tokens      → Llama layers [0 .. target_layer]  (cheap, short text)
  3. Concatenate x_final as a prefix at layer `target_layer`
  4. Continue Llama layers [target_layer .. 80] → generate

This saves running Llama's first `target_layer` layers over the entire document
and eliminates the KV cache memory for document tokens.

Usage:
  python inference.py --ckpt CKPT --document DOC.txt --query "Summarise section 3."
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast

from config import StitcherConfig
from stitcher_model import LatentStitcher
from collect_hidden_states import extract_qwen_hidden, load_source_model


# ── Llama split-forward helpers ───────────────────────────────────────────────

def _rope(model, hidden, position_ids):
    """Compute RoPE (cos, sin) — compatible with transformers >= 4.40."""
    return model.model.rotary_emb(hidden, position_ids)


def llama_embed_and_early_layers(model, input_ids, target_layer: int):
    """
    Run Llama embedding + layers 0…target_layer-1.
    Returns hidden states at layer `target_layer`.
    """
    hidden = model.model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
    position_embeddings = _rope(model, hidden, position_ids)

    for i, layer in enumerate(model.model.layers):
        if i >= target_layer:
            break
        layer_out = layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        hidden = layer_out[0]

    return hidden   # (1, seq_len_query, tgt_dim)


def llama_late_layers_and_generate(
    model,
    tokenizer,
    hidden: torch.Tensor,        # (1, total_seq, tgt_dim)  after prefix injection
    target_layer: int,
    max_new_tokens: int = 512,
):
    """
    Run Llama layers target_layer…80 and decode output tokens.
    `hidden` already contains the document prefix + query hidden states
    concatenated at layer `target_layer`.
    """
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
    position_embeddings = _rope(model, hidden, position_ids)

    for i in range(target_layer, len(model.model.layers)):
        layer_out = model.model.layers[i](
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        hidden = layer_out[0]

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)     # (1, seq_len, vocab)

    generated = []
    next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)   # (1,)
    generated.append(next_token.item())

    for _ in range(max_new_tokens - 1):
        embed = model.model.embed_tokens(next_token.unsqueeze(0))   # (1, 1, D)
        pos = torch.tensor([[seq_len + len(generated) - 1]], device=hidden.device)
        pe = _rope(model, embed, pos)
        for layer in model.model.layers[target_layer:]:
            out = layer(embed, attention_mask=None, position_ids=pos,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        position_embeddings=pe)
            embed = out[0]
        embed = model.model.norm(embed)
        logits = model.lm_head(embed)
        next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
        token_id = next_token.item()
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_inference(
    document_text: str,
    query: str,
    cfg: StitcherConfig,
    stitcher: LatentStitcher,
    qwen_tok, qwen_model,
    llama_tok, llama_model,
) -> str:
    dtype = getattr(torch, cfg.dtype)
    stitcher_device = torch.device(cfg.source_device)
    llama_device = next(llama_model.parameters()).device

    # Step 1: Translate document context to Llama's layer `target_layer` space
    print("Encoding document with Qwen …")
    x_qwen = extract_qwen_hidden(qwen_model, qwen_tok, document_text, cfg.source_device)
    x_qwen = x_qwen.unsqueeze(0).to(stitcher_device, dtype=dtype)

    stitcher.eval()
    with torch.no_grad():
        x_final = stitcher(x_qwen)                    # (1, tgt_dim)
    x_final = x_final.to(llama_device)                # move to Llama shards

    # Step 2: Encode query and run early Llama layers
    print("Running Llama early layers on query …")
    query_ids = llama_tok(query, return_tensors="pt").input_ids.to(llama_device)
    query_hidden = llama_embed_and_early_layers(llama_model, query_ids, cfg.target_layer)
    # query_hidden: (1, query_len, tgt_dim)

    # Step 3: Prepend x_final as a single-token soft prefix at layer `target_layer`
    prefix = x_final.unsqueeze(1)                     # (1, 1, tgt_dim)
    combined = torch.cat([prefix, query_hidden], dim=1)  # (1, 1+query_len, tgt_dim)

    # Step 4: Run late layers and generate
    print("Generating …")
    response = llama_late_layers_and_generate(
        llama_model, llama_tok, combined, cfg.target_layer
    )
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--document", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    cfg = StitcherConfig()

    from collect_hidden_states import load_target_model
    print("Loading Qwen …")
    qwen_tok, qwen_model = load_source_model(cfg)
    print("Loading Llama …")
    llama_tok, llama_model = load_target_model(cfg)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    W_optimal = ckpt["model_state"]["stage1.W"]
    dtype = getattr(torch, cfg.dtype)
    stitcher = LatentStitcher(cfg, W_optimal)
    stitcher.load_state_dict(ckpt["model_state"])
    stitcher = stitcher.to(cfg.source_device).to(dtype)

    with open(args.document) as f:
        document_text = f.read()

    response = run_inference(
        document_text, args.query, cfg, stitcher,
        qwen_tok, qwen_model, llama_tok, llama_model,
    )
    print("\n── Response ────────────────────────────────────")
    print(response)


if __name__ == "__main__":
    main()

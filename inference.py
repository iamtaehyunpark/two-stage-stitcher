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

def generate_with_injection(
    model,
    tokenizer,
    x_final: torch.Tensor,       # (1, tgt_dim) — document representation in layer-`target_layer` space
    query: str,
    target_layer: int,
    max_new_tokens: int = 256,
) -> str:
    """
    Inject `x_final` at `target_layer` and let the model's own generate() do the rest.

    Mechanism:
      1. Prepend ONE dummy token to the query so every layer sees a uniform
         sequence length (keeps the KV cache + attention mask consistent —
         transformers handles all of that for us).
      2. A forward-pre-hook on `model.model.layers[target_layer]` overwrites the
         dummy position's hidden state with `x_final` on the prefill pass only
         (seq_len > 1). From layer `target_layer` onward, position 0 carries the
         document representation, and it lands in the KV cache like a normal token.
      3. During decoding (seq_len == 1) the hook is a no-op, so each generated
         token attends to the full [x_final, query, generated…] context via the
         cache — no context loss, no manual RoPE, no version drift.
    """
    embed_device = model.model.embed_tokens.weight.device

    query_ids = tokenizer(query, return_tensors="pt").input_ids
    dummy_id = tokenizer.bos_token_id
    if dummy_id is None:
        dummy_id = tokenizer.eos_token_id
    dummy = torch.tensor([[dummy_id]], dtype=query_ids.dtype)
    input_ids = torch.cat([dummy, query_ids], dim=1).to(embed_device)
    attention_mask = torch.ones_like(input_ids)

    x = x_final.reshape(1, -1)   # (1, tgt_dim)
    layer = model.model.layers[target_layer]

    def pre_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is None or hs.shape[1] <= 1:      # decode step → leave untouched
            return None
        hs = hs.clone()
        hs[:, 0, :] = x.to(hs.device, hs.dtype)
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hs
            return args, kwargs
        return (hs,) + tuple(args[1:]), kwargs

    handle = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    generated = out[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


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

    # Step 2: Inject at `target_layer` and generate via the model's own forward
    print("Generating …")
    response = generate_with_injection(
        llama_model, llama_tok, x_final, query, cfg.target_layer
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

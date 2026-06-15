"""
Oracle / upper-bound probe for the layer-30 injection premise.

Removes the stitcher entirely and injects the *ground-truth* DeepSeek layer-30
hidden states (captured from a real forward pass) using the SAME injection
plumbing as condition B. This isolates the premise from the translation.

For each QA pair, three answers on the same DeepSeek-70B:

  A           — full prefill                          (ceiling / reference)
  Oracle-SEQ  — inject the TRUE layer-30 states of ALL document tokens
                (multi-token prefix) then ask the query
  Oracle-LAST — inject only the TRUE last-token layer-30 vector
                (single token — mirrors the current stitcher's shape)

Verdicts:
  SEQ ≈ A                  → premise + plumbing sound; failure is compression+translation
  SEQ ≈ A but LAST fails   → single-vector bottleneck is the killer; sequence is mandatory
  SEQ also fails           → inject-at-30 premise itself is broken (wrong layer / attention path)

Usage:
    python evaluate/oracle_probe.py \
        --qa evaluate/data/qa_pairs.json \
        --num 5 \
        --out evaluate/data/oracle_probe.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_conditions import load_deepseek, CONDITION_A_PROMPT, generate_answers

# Query-only prompt; the document is supplied as injected layer-30 states, so the
# instruction references the (latent) preceding context rather than inline text.
ORACLE_QUERY_PROMPT = """\
You are a helpful assistant. Answer the question based on the preceding document.

Question: {question}
Answer:"""


def capture_layer30(model, input_ids, target_layer):
    """Run a forward pass and capture the hidden states ENTERING `target_layer`."""
    import torch

    captured = {}

    def cap_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        captured["y"] = hs.detach()
        return None

    handle = model.model.layers[target_layer].register_forward_pre_hook(
        cap_hook, with_kwargs=True
    )
    try:
        with torch.no_grad():
            model(input_ids=input_ids,
                  attention_mask=torch.ones_like(input_ids),
                  use_cache=False)
    finally:
        handle.remove()

    return captured["y"]   # (1, seq_len, tgt_dim) on target_layer's device


def generate_with_sequence_injection(model, tokenizer, Y_seq, query_text,
                                     target_layer, max_new_tokens=512):
    """
    Inject a SEQUENCE of layer-30 states `Y_seq` (1, N, D) as a prefix.

    Prepend N placeholder tokens (overwritten at `target_layer` on the prefill
    pass), then the query. The query sits at positions N.. and is causally
    allowed to attend back to all N injected positions.
    """
    import torch

    embed_device = model.model.embed_tokens.weight.device
    N = Y_seq.shape[1]

    q_ids = tokenizer(query_text, return_tensors="pt").input_ids
    dummy_id = tokenizer.bos_token_id
    if dummy_id is None:
        dummy_id = tokenizer.eos_token_id
    dummy = torch.full((1, N), dummy_id, dtype=q_ids.dtype)
    input_ids = torch.cat([dummy, q_ids], dim=1).to(embed_device)
    attention_mask = torch.ones_like(input_ids)

    layer = model.model.layers[target_layer]

    def pre_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is None or hs.shape[1] <= 1:      # decode step → no-op
            return None
        hs = hs.clone()
        hs[:, :N, :] = Y_seq.to(hs.device, hs.dtype)
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

    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def strip_think(text: str) -> str:
    """R1 wraps answers in <think>…</think>; keep only the final answer for readability."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def main():
    import torch
    from config import StitcherConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--qa",  default="evaluate/data/qa_pairs.json")
    parser.add_argument("--num", type=int, default=5, help="number of QA pairs to probe")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", default="evaluate/data/oracle_probe.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = StitcherConfig()
    with open(args.qa) as f:
        qa_pairs = json.load(f)[: args.num]
    print(f"Probing {len(qa_pairs)} QA pairs")

    llama_tok, llama_model = load_deepseek(cfg)
    first_device = next(llama_model.parameters()).device

    results = []
    for i, qa in enumerate(qa_pairs):
        doc = qa["document"][:6000]
        q = qa["question"]
        print(f"\n{'='*70}\n[{i+1}/{len(qa_pairs)}] {qa['doc_name']}\nQ: {q}")

        # ── A: full prefill ───────────────────────────────────────────────
        prompt_a = CONDITION_A_PROMPT.format(document=doc, question=q)
        ans_a = generate_answers(llama_model, llama_tok, [prompt_a],
                                 max_new_tokens=args.max_new_tokens)[0]

        # ── Capture TRUE layer-30 states of the document ──────────────────
        doc_ids = llama_tok(doc, return_tensors="pt",
                            truncation=True, max_length=8192).input_ids.to(first_device)
        Y_true = capture_layer30(llama_model, doc_ids, cfg.target_layer)  # (1, N, D)
        query_text = ORACLE_QUERY_PROMPT.format(question=q)

        # ── Oracle-SEQ: inject the full true sequence ─────────────────────
        ans_seq = generate_with_sequence_injection(
            llama_model, llama_tok, Y_true, query_text,
            cfg.target_layer, max_new_tokens=args.max_new_tokens,
        )

        # ── Oracle-LAST: inject only the true last-token vector ───────────
        Y_last = Y_true[:, -1:, :]   # (1, 1, D)
        ans_last = generate_with_sequence_injection(
            llama_model, llama_tok, Y_last, query_text,
            cfg.target_layer, max_new_tokens=args.max_new_tokens,
        )

        rec = {
            "doc_name": qa["doc_name"],
            "question": q,
            "reference_answer": qa.get("reference_answer", ""),
            "doc_tokens": int(Y_true.shape[1]),
            "answer_a":           strip_think(ans_a),
            "answer_oracle_seq":  strip_think(ans_seq),
            "answer_oracle_last": strip_think(ans_last),
        }
        results.append(rec)

        print(f"  doc_tokens: {rec['doc_tokens']}")
        print(f"  A          : {rec['answer_a'][:200]}")
        print(f"  Oracle-SEQ : {rec['answer_oracle_seq'][:200]}")
        print(f"  Oracle-LAST: {rec['answer_oracle_last'][:200]}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")
    print("\nInterpretation:")
    print("  SEQ ≈ A                → premise sound; failure is compression+translation")
    print("  SEQ ≈ A but LAST fails → single-vector bottleneck; sequence is mandatory")
    print("  SEQ also fails         → inject-at-30 premise itself is broken")


if __name__ == "__main__":
    main()

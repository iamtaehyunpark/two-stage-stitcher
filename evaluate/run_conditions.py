"""
Step 2: Run all three conditions and collect answers.

  Condition A — Full prefill  : document + question → DeepSeek-70B (transformers)
  Condition B — Stitcher      : Qwen-7B → stitcher → inject into DeepSeek-70B
  Condition C — No context    : question only → DeepSeek-70B (same model, no doc)

All three conditions use the same DeepSeek-70B model loaded once via transformers.
This is a pure ablation: A tests full context, B tests the stitcher, C is the floor.

Usage:
    python evaluate/run_conditions.py \
        --qa evaluate/data/qa_pairs.json \
        --ckpt checkpoints/stitcher_best.pt \
        --out evaluate/data/condition_results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CONDITION_A_PROMPT = """\
You are a helpful assistant. Answer the question based on the document below.

Document:
{document}

Question: {question}
Answer:"""

CONDITION_C_PROMPT = """\
You are a helpful assistant. Answer the question as best you can.

Question: {question}
Answer:"""


def load_deepseek(cfg):
    """Load DeepSeek-70B once; shared by conditions A, B, and C."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype = getattr(torch, cfg.dtype)
    max_memory = {i: "70GiB" for i in cfg.llama_devices}
    print(f"Loading {cfg.target_model} …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model,
        torch_dtype=dtype,
        device_map="sequential",
        max_memory=max_memory,
    )
    model.eval()
    return tokenizer, model


def generate_answers(model, tokenizer, prompts: list, max_new_tokens: int = 256) -> list:
    import torch

    first_device = next(model.parameters()).device
    answers = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=8192).to(first_device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[-1]:]
        answers.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return answers


def run_conditions_abc(qa_pairs: list, ckpt_path: str, skip_b: bool = False, answers_a: list = None):
    """
    Run all three conditions sharing a single DeepSeek-70B load.

    A — full document prefill
    B — stitcher injection (Qwen-7B → MLP → inject at layer 30)
    C — question only, no context (lower bound)
    """
    import torch
    from config import StitcherConfig
    from collect_hidden_states import load_source_model, load_target_model, extract_qwen_hidden
    from inference import llama_embed_and_early_layers, llama_late_layers_and_generate

    cfg = StitcherConfig()
    dtype = getattr(torch, cfg.dtype)

    llama_tok, llama_model = load_deepseek(cfg)

    # ── Condition A ──────────────────────────────────────────────────────────
    if answers_a is None:
        print("\n[A] Full prefill …")
        prompts_a = [
            CONDITION_A_PROMPT.format(document=qa["document"][:6000], question=qa["question"])
            for qa in qa_pairs
        ]
        answers_a = []
        for i, prompt in enumerate(prompts_a):
            print(f"  A [{i+1}/{len(prompts_a)}] …", end="\r")
            answers_a += generate_answers(llama_model, llama_tok, [prompt])
        print(f"\n  Done. {len(answers_a)} answers.")
    else:
        print(f"\n[A] Skipped — using {len(answers_a)} preloaded answers.")

    # ── Condition B ──────────────────────────────────────────────────────────
    if skip_b:
        answers_b = ["[skipped]"] * len(qa_pairs)
    else:
        print("\n[B] Stitcher injection …")
        from stitcher_model import LatentStitcher
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        W_optimal = ckpt["model_state"]["stage1.W"]
        stitcher = LatentStitcher(cfg, W_optimal).to(cfg.source_device).to(dtype)
        stitcher.load_state_dict(ckpt["model_state"])
        stitcher.eval()

        qwen_tok, qwen_model = load_source_model(cfg)
        llama_device = next(llama_model.parameters()).device

        answers_b = []
        for i, qa in enumerate(qa_pairs):
            print(f"  B [{i+1}/{len(qa_pairs)}] …", end="\r")
            x_qwen = extract_qwen_hidden(
                qwen_model, qwen_tok, qa["document"][:6000], cfg.source_device
            )
            x_qwen = x_qwen.unsqueeze(0).to(cfg.source_device, dtype=dtype)
            with torch.no_grad():
                x_final = stitcher(x_qwen).to(llama_device)
            query_ids = llama_tok(qa["question"], return_tensors="pt").input_ids.to(llama_device)
            query_hidden = llama_embed_and_early_layers(llama_model, query_ids, cfg.target_layer)
            prefix = x_final.unsqueeze(1)
            combined = torch.cat([prefix, query_hidden], dim=1)
            answer = llama_late_layers_and_generate(
                llama_model, llama_tok, combined, cfg.target_layer, max_new_tokens=256
            )
            answers_b.append(answer)
        print(f"\n  Done. {len(answers_b)} answers.")

    # ── Condition C ──────────────────────────────────────────────────────────
    print("\n[C] No context (question only) …")
    prompts_c = [
        CONDITION_C_PROMPT.format(question=qa["question"])
        for qa in qa_pairs
    ]
    answers_c = []
    for i, prompt in enumerate(prompts_c):
        print(f"  C [{i+1}/{len(prompts_c)}] …", end="\r")
        answers_c += generate_answers(llama_model, llama_tok, [prompt])
    print(f"\n  Done. {len(answers_c)} answers.")

    return answers_a, answers_b, answers_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa",      default="evaluate/data/qa_pairs.json")
    parser.add_argument("--ckpt",    default="checkpoints/stitcher_best.pt")
    parser.add_argument("--out",     default="evaluate/data/condition_results.json")
    parser.add_argument("--skip-a",  action="store_true",
                        help="Load existing answers_a from --out instead of rerunning condition A")
    parser.add_argument("--skip-b",  action="store_true",
                        help="Skip condition B (stitcher) for a quick A/C baseline run")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.qa) as f:
        qa_pairs = json.load(f)
    print(f"Loaded {len(qa_pairs)} QA pairs")

    if args.skip_a:
        if not os.path.exists(args.out):
            raise FileNotFoundError(
                f"--skip-a requires existing results at {args.out} (run condition A first)"
            )
        with open(args.out) as f:
            prev = json.load(f)
        answers_a = [r["answer_a"] for r in prev]
        print(f"[A] Loaded {len(answers_a)} existing answers from {args.out}")
    else:
        answers_a = None

    answers_a, answers_b, answers_c = run_conditions_abc(
        qa_pairs, args.ckpt, skip_b=args.skip_b, answers_a=answers_a
    )

    results = []
    for qa, a, b, c in zip(qa_pairs, answers_a, answers_b, answers_c):
        results.append({
            "doc_name":         qa["doc_name"],
            "question":         qa["question"],
            "reference_answer": qa["reference_answer"],
            "answer_a":         a,   # DeepSeek-70B full prefill  (gold)
            "answer_b":         b,   # DeepSeek-70B via stitcher
            "answer_c":         c,   # DeepSeek-70B no context    (lower bound)
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results → {args.out}")


if __name__ == "__main__":
    main()

"""
Step 2: Run all three conditions and collect answers.

  Condition A — Full prefill  : document + question → DeepSeek-70B (transformers)
  Condition B — Stitcher      : Qwen-7B → stitcher → inject into DeepSeek-70B
  Condition C — No context    : question only → Qwen-72B-AWQ (vLLM)

A and B share the same DeepSeek-70B model loaded via transformers.
C uses Qwen-72B-AWQ via vLLM as a separate process.

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

QWEN_72B_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"

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


def run_condition_a(qa_pairs: list) -> list:
    """
    Condition A: standard full-prefill generation with DeepSeek-70B.
    Uses the same model and layer setup as the stitcher target, so the
    comparison is apples-to-apples.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from config import StitcherConfig

    cfg = StitcherConfig()
    dtype = getattr(torch, cfg.dtype)
    max_memory = {i: "70GiB" for i in cfg.llama_devices}

    print(f"\nLoading {cfg.target_model} for condition A …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model,
        torch_dtype=dtype,
        device_map="sequential",
        max_memory=max_memory,
    )
    model.eval()
    first_device = next(model.parameters()).device

    answers_a = []
    for i, qa in enumerate(qa_pairs):
        print(f"  Condition A [{i+1}/{len(qa_pairs)}] …", end="\r")
        prompt = CONDITION_A_PROMPT.format(
            document=qa["document"][:6000],
            question=qa["question"],
        )
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=8192).to(first_device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[-1]:]
        answers_a.append(tokenizer.decode(generated, skip_special_tokens=True).strip())

    print(f"\n  Done. {len(answers_a)} answers collected.")
    return answers_a


def run_condition_b(qa_pairs: list, ckpt_path: str) -> list:
    """Condition B: stitcher injection into DeepSeek-70B."""
    import torch
    from config import StitcherConfig
    from stitcher_model import LatentStitcher
    from collect_hidden_states import load_source_model, load_target_model, extract_qwen_hidden
    from inference import llama_embed_and_early_layers, llama_late_layers_and_generate

    cfg = StitcherConfig()
    dtype = getattr(torch, cfg.dtype)

    print("\nLoading stitcher checkpoint …")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    W_optimal = ckpt["model_state"]["stage1.W"]
    stitcher = LatentStitcher(cfg, W_optimal).to(cfg.source_device).to(dtype)
    stitcher.load_state_dict(ckpt["model_state"])
    stitcher.eval()

    print("Loading Qwen-7B …")
    qwen_tok, qwen_model = load_source_model(cfg)

    print("Loading DeepSeek-70B …")
    llama_tok, llama_model = load_target_model(cfg)
    llama_device = next(llama_model.parameters()).device

    answers_b = []
    for i, qa in enumerate(qa_pairs):
        print(f"  Condition B [{i+1}/{len(qa_pairs)}] …", end="\r")

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

    print(f"\n  Done. {len(answers_b)} answers collected.")
    return answers_b


def run_condition_c(qa_pairs: list) -> list:
    """Condition C: no context, Qwen-72B-AWQ via vLLM."""
    from vllm import LLM, SamplingParams

    print(f"\nLoading {QWEN_72B_MODEL} for condition C …")
    llm = LLM(
        model=QWEN_72B_MODEL,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        tensor_parallel_size=1,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop=['\n\nQuestion:', '\n\nAnswer:'],
    )

    prompts = [CONDITION_C_PROMPT.format(question=qa["question"]) for qa in qa_pairs]
    print(f"  Running condition C ({len(prompts)} prompts) …")
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text.strip() for o in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa",      default="evaluate/data/qa_pairs.json")
    parser.add_argument("--ckpt",    default="checkpoints/stitcher_best.pt")
    parser.add_argument("--out",     default="evaluate/data/condition_results.json")
    parser.add_argument("--skip-b",  action="store_true",
                        help="Skip condition B (stitcher) for a quick A/C baseline run")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.qa) as f:
        qa_pairs = json.load(f)
    print(f"Loaded {len(qa_pairs)} QA pairs")

    # A and B both use DeepSeek-70B — run together to load the model once
    answers_a = run_condition_a(qa_pairs)

    if args.skip_b:
        answers_b = ["[skipped]"] * len(qa_pairs)
    else:
        answers_b = run_condition_b(qa_pairs, args.ckpt)

    # C uses a different model — separate session
    answers_c = run_condition_c(qa_pairs)

    results = []
    for qa, a, b, c in zip(qa_pairs, answers_a, answers_b, answers_c):
        results.append({
            "doc_name":         qa["doc_name"],
            "question":         qa["question"],
            "reference_answer": qa["reference_answer"],
            "answer_a":         a,   # DeepSeek-70B full prefill  (gold)
            "answer_b":         b,   # DeepSeek-70B via stitcher
            "answer_c":         c,   # Qwen-72B no context        (lower bound)
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results → {args.out}")


if __name__ == "__main__":
    main()

"""
Step 1: Generate QA pairs from held-out documents using Qwen2.5-72B-AWQ.

For each document, prompts the model to produce 5 factual questions
that require reading the document to answer. Saves results to qa_pairs.json.

Usage:
    python evaluate/generate_qa.py \
        --docs-dir /data/tpark45/docs/eval \
        --num-docs 30 \
        --out evaluate/data/qa_pairs.json
"""

import os
import json
import argparse
import random
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"

QA_PROMPT = """\
Read the following document carefully, then generate exactly 5 factual questions \
that can only be answered correctly by someone who has read this document. \
For each question also provide the correct answer based on the document.

Respond in JSON with this exact format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Document:
{document}

JSON output:"""


def truncate(text: str, max_chars: int = 6000) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--num-docs", type=int, default=30)
    parser.add_argument("--out", default="evaluate/data/qa_pairs.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    random.seed(args.seed)
    all_docs = list(Path(args.docs_dir).glob("*.txt"))
    if not all_docs:
        raise FileNotFoundError(f"No .txt files in {args.docs_dir}")
    docs = random.sample(all_docs, min(args.num_docs, len(all_docs)))
    print(f"Selected {len(docs)} documents")

    from vllm import LLM, SamplingParams
    print(f"Loading {MODEL_ID} …")
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(temperature=0.3, max_tokens=1024)

    prompts, doc_texts, doc_names = [], [], []
    for doc_path in docs:
        text = doc_path.read_text().strip()
        prompts.append(QA_PROMPT.format(document=truncate(text)))
        doc_texts.append(text)
        doc_names.append(doc_path.name)

    print(f"Generating QA pairs for {len(prompts)} documents …")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    skipped = 0
    for doc_name, doc_text, output in zip(doc_names, doc_texts, outputs):
        raw = output.outputs[0].text.strip()
        try:
            # extract JSON array even if model wraps it in markdown
            start = raw.find("[")
            end = raw.rfind("]") + 1
            pairs = json.loads(raw[start:end])
            for pair in pairs:
                if "question" in pair and "answer" in pair:
                    results.append({
                        "doc_name": doc_name,
                        "document": doc_text,
                        "question": pair["question"],
                        "reference_answer": pair["answer"],
                    })
        except (json.JSONDecodeError, ValueError):
            skipped += 1

    print(f"Generated {len(results)} QA pairs  (skipped {skipped} docs with parse errors)")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()

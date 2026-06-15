"""
Step 3: LLM-as-judge scoring.

For each question, Qwen2.5-72B judges condition B (stitcher) and
condition C (no context) against condition A (full prefill) as the reference.

Scoring criteria (1–5 each):
  - Faithfulness : does the answer agree with / not contradict A?
  - Completeness : does the answer cover the key information in A?

Final score per condition = mean across all questions.

Usage:
    python evaluate/judge.py \
        --results evaluate/data/condition_results.json \
        --out evaluate/data/scores.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"

JUDGE_PROMPT = """\
You are an objective evaluator. A reference answer (Answer A) was produced by a \
model that read the full document. You must score a candidate answer against it.

Question: {question}

Answer A (reference — full document access): {answer_a}

Candidate Answer ({label}): {candidate}

Score the candidate on two criteria from 1 to 5:
  Faithfulness  — Does the candidate agree with A and avoid contradicting it? \
(1=contradicts A, 5=fully consistent)
  Completeness  — Does the candidate cover the key information in A? \
(1=misses almost everything, 5=covers everything)

Respond in JSON only:
{{"faithfulness": <1-5>, "completeness": <1-5>, "reason": "<one sentence>"}}

JSON:"""


def parse_score(raw: str) -> dict:
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        d = json.loads(raw[start:end])
        return {
            "faithfulness": int(d.get("faithfulness", 0)),
            "completeness": int(d.get("completeness", 0)),
            "reason": d.get("reason", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {"faithfulness": 0, "completeness": 0, "reason": "parse error"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="evaluate/data/condition_results.json")
    parser.add_argument("--out",     default="evaluate/data/scores.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results")

    from vllm import LLM, SamplingParams
    print(f"Loading {JUDGE_MODEL} …")
    llm = LLM(
        model=JUDGE_MODEL,
        dtype="float16",
        quantization="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=2,
    )
    params = SamplingParams(temperature=0.0, max_tokens=128)

    # Build judge prompts for B and C
    prompts_b, prompts_c = [], []
    for r in results:
        prompts_b.append(JUDGE_PROMPT.format(
            question=r["question"],
            answer_a=r["answer_a"],
            label="B — stitcher injection",
            candidate=r["answer_b"],
        ))
        prompts_c.append(JUDGE_PROMPT.format(
            question=r["question"],
            answer_a=r["answer_a"],
            label="C — no context",
            candidate=r["answer_c"],
        ))

    print(f"Judging condition B ({len(prompts_b)} items) …")
    out_b = llm.generate(prompts_b, params)
    scores_b = [parse_score(o.outputs[0].text) for o in out_b]

    print(f"Judging condition C ({len(prompts_c)} items) …")
    out_c = llm.generate(prompts_c, params)
    scores_c = [parse_score(o.outputs[0].text) for o in out_c]

    # Attach scores to results
    scored = []
    for r, sb, sc in zip(results, scores_b, scores_c):
        scored.append({**r, "score_b": sb, "score_c": sc})

    # Aggregate
    def mean_scores(scores):
        valid = [s for s in scores if s["faithfulness"] > 0]
        if not valid:
            return {"faithfulness": 0.0, "completeness": 0.0}
        return {
            "faithfulness": round(sum(s["faithfulness"] for s in valid) / len(valid), 3),
            "completeness": round(sum(s["completeness"] for s in valid) / len(valid), 3),
        }

    agg_b = mean_scores(scores_b)
    agg_c = mean_scores(scores_c)

    print("\n══════════════════════════════════════════════════")
    print("  Evaluation Results (vs Condition A — full prefill)")
    print("══════════════════════════════════════════════════")
    print(f"  Condition B (stitcher)   "
          f"faithfulness={agg_b['faithfulness']:.3f}  "
          f"completeness={agg_b['completeness']:.3f}")
    print(f"  Condition C (no context) "
          f"faithfulness={agg_c['faithfulness']:.3f}  "
          f"completeness={agg_c['completeness']:.3f}")
    print(f"\n  Δ faithfulness  B−C = {agg_b['faithfulness'] - agg_c['faithfulness']:+.3f}")
    print(f"  Δ completeness  B−C = {agg_b['completeness'] - agg_c['completeness']:+.3f}")
    print("══════════════════════════════════════════════════")

    output = {
        "num_questions": len(results),
        "aggregate": {"B_stitcher": agg_b, "C_no_context": agg_c},
        "per_question": scored,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed scores → {args.out}")


if __name__ == "__main__":
    main()

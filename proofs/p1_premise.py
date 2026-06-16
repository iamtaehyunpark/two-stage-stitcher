"""
Proof 1 — the injection premise: injected representations are reasoned over.

The central premise of the project. On fabricated, unguessable facts (so a correct
answer cannot come from memory), three conditions per question:

  C (floor)        — question only. MUST FAIL, else the fact was guessable → item
                     disqualified.
  A (ceiling)      — full prefill of the synthetic document. MUST SUCCEED, else the
                     fact isn't recoverable from the text → item disqualified.
  inject-all-N     — the document handed over as TRUE layer-`target_layer` states
                     via the validated Proof-0 split-forward. THE TEST.

PASS: on items where C fails and A succeeds, inject-all-N succeeds. The model
answered a fact it could not have known, from injected states it never tokenized.
FAIL: inject behaves like C while A succeeds → the receiver doesn't read injected
states; no translation can rescue it → stop the project (cheaply).

Note: a clean pass here is necessary but not sufficient — Proof 2's wrong-document
control is what rules out a leak. Run `p2_falsifier.py` (or `run_chain.py`) next.

Usage:
    python proofs/p1_premise.py --out proofs/data/p1.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.common import load_deepseek
from proofs.synthetic_eval import evaluate_synthetic, verdict_p1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="proofs/data/p1.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))
    tok, model = load_deepseek(cfg, devices=devices)

    records = evaluate_synthetic(model, tok, cfg,
                                 max_new_tokens=args.max_new_tokens, want_wrong=False)
    summary = verdict_p1(records)

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

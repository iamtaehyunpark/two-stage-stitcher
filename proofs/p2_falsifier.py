"""
Proof 2 — the falsifier: it's the injection, not anything else.

Proof 1 shows the model answers correctly WHEN the right document is injected.
Proof 2 shows it answers INCORRECTLY when the wrong document is injected. Together
they nail causation: the injected content controls the answer. This is the control
the original oracle probe never ran — its absence is why that probe's "5/5" was
uninterpretable. It is non-negotiable.

Wrong-document injection: inject document Y's true states, ask document X's
question. On the gated items (C fails, A succeeds):

  matched (inject X, ask X)  — should SUCCEED  (this is Proof 1's dual)
  wrong   (inject Y, ask X)  — should FAIL / hedge / surface Y's content

PASS: matched high AND wrong ~0 → the injection is causally responsible; combined
with Proof 1 the premise is CONFIRMED.
FAIL: wrong still answers X regardless of which document is injected → injection
inert, Proof 1 falsified retroactively → stop and find the leak.

Usage:
    python proofs/p2_falsifier.py --out proofs/data/p2.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.common import load_deepseek
from proofs.synthetic_eval import evaluate_synthetic, verdict_p1, verdict_p2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="proofs/data/p2.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from config import StitcherConfig
    cfg = StitcherConfig()
    devices = tuple(int(x) for x in args.gpus.split(","))
    tok, model = load_deepseek(cfg, devices=devices)

    # want_wrong=True also yields the matched/gate columns, so we render Proof 1's
    # verdict too for context — Proof 2 is only interpretable alongside it.
    records = evaluate_synthetic(model, tok, cfg,
                                 max_new_tokens=args.max_new_tokens, want_wrong=True)
    s1 = verdict_p1(records)
    s2 = verdict_p2(records)

    with open(args.out, "w") as f:
        json.dump({"proof1": s1, "proof2": s2, "records": records}, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

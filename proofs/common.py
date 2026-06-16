"""
proofs/common.py — shared scaffolding for the receiver-validation proofs.

Everything here is environment-faithful to the existing experiment code: the same
`StitcherConfig`, the same DeepSeek loader (`run_conditions.load_deepseek`, sharded
across the logical GPUs in `cfg.llama_devices`), the same `<think>` stripping. The
proofs differ from the old oracle probes only in the *mechanism* they test — the
correct two-cache `core.split_forward`, not the dummy-token prefix that produced
the degenerate `a the a the` artifact.

Three answer-producing primitives, all on one frozen DeepSeek-70B:

  no_context_answer  — Condition C: question only, no document.  (the floor / gate)
  full_prefill_answer— Condition A: document + question as tokens. (the ceiling / gate)
  inject_answer      — the test: document handed over as TRUE layer-`target_layer`
                       states via the split-forward; only the query is tokenized.
"""

import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "evaluate"))     # reuse the env-faithful loaders

from run_conditions import load_deepseek          # noqa: E402  (env-faithful loader)
from oracle_probe import strip_think               # noqa: E402
from core.split_forward import capture_doc_cache, split_forward_generate  # noqa: E402


# ── prompts ───────────────────────────────────────────────────────────────────
# Condition A: the document is inline text.
PREFILL_PROMPT = (
    "You are a helpful assistant. Answer the question based on the document below, "
    "using only the information it contains. If the answer is not present, say you "
    "do not know.\n\nDocument:\n{document}\n\nQuestion: {question}\nAnswer:"
)
# Condition C: no document at all.
NO_CONTEXT_PROMPT = (
    "You are a helpful assistant. Answer the question as best you can.\n\n"
    "Question: {question}\nAnswer:"
)
# Inject conditions: the document is supplied as injected states, not text, so the
# instruction refers to the (latent) preceding document.
QUERY_PROMPT = (
    "You are a helpful assistant. Answer the question based on the preceding "
    "document, using only the information it contains. If the answer is not "
    "present, say you do not know.\n\nQuestion: {question}\nAnswer:"
)

# DeepSeek-R1 is a reasoning model. For this factual-recall eval we suppress the
# <think> trace (the standard R1 no-think trick: begin the assistant turn with an
# already-closed, empty think block) so every condition emits one short, directly
# scorable answer. This avoids two failure modes: a long trace truncating before
# the answer, and a gold token (e.g. "five", "64") matching by chance inside the
# reasoning and corrupting the C-fails gate. Set SUPPRESS_THINK=False (CLI
# --reasoning) to let it reason instead.
SUPPRESS_THINK = True
THINK_SKIP = "<think>\n\n</think>\n\n"


def _with_think_control(prompt: str) -> str:
    return prompt + THINK_SKIP if SUPPRESS_THINK else prompt


# ── scoring (substring match after light normalisation) ───────────────────────
def normalize(s: str) -> str:
    s = s.lower().replace(",", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def correct(answer: str, gold: str) -> bool:
    return normalize(gold) in normalize(answer)


# ── answer primitives ─────────────────────────────────────────────────────────
def _device(model):
    return next(model.parameters()).device


def generate_plain(model, tokenizer, prompt, max_new_tokens):
    import torch
    prompt = _with_think_control(prompt)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=8192).to(_device(model))
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return strip_think(text)


def no_context_answer(model, tokenizer, question, max_new_tokens=256):
    return generate_plain(model, tokenizer,
                          NO_CONTEXT_PROMPT.format(question=question), max_new_tokens)


def full_prefill_answer(model, tokenizer, document, question, max_new_tokens=256):
    return generate_plain(model, tokenizer,
                          PREFILL_PROMPT.format(document=document, question=question),
                          max_new_tokens)


def capture_document(model, tokenizer, document, target_layer, max_doc_tokens=8192):
    """Tokenize `document` and capture its true split-forward cache once. Returns
    (doc_cache, n_doc) to be reused across every question for that document."""
    ids = tokenizer(document, return_tensors="pt", truncation=True,
                    max_length=max_doc_tokens).input_ids
    doc_cache, _Y, n_doc = capture_doc_cache(model, ids, target_layer)
    return doc_cache, n_doc


def inject_answer(model, tokenizer, doc_cache, n_doc, question, target_layer,
                  max_new_tokens=256):
    """The test condition: answer `question` with the document supplied only as
    injected true layer-`target_layer` states (split-forward)."""
    text = split_forward_generate(
        model, tokenizer, doc_cache, n_doc,
        query_text=_with_think_control(QUERY_PROMPT.format(question=question)),
        target_layer=target_layer, max_new_tokens=max_new_tokens,
    )
    return strip_think(text)

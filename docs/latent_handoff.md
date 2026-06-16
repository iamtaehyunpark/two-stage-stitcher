# Latent Handoff: Reading as a Transferable Act

*Project vision — Phase Two*

---

## The thesis

A document, once read, leaves a trace. When a transformer processes a context, it doesn't merely store the tokens — it builds an internal representation in which the facts of the document have been *located, bound, and made available* for whatever comes next. The reading has happened. The understanding sits somewhere in the model's activations, position by position, layer by layer.

The question this project asks is deceptively simple: **once reading has happened somewhere, must it happen again?**

The prevailing answer in every long-context system is *yes*. Each model that needs to reason over a document pays, in full, to read it — even if the same document was read a moment ago by a different model, even if a cheaper model could have read it just as accurately. Reading is treated as inseparable from the reader. The work cannot be saved, lent, or inherited. Every reasoner re-reads from scratch.

We suspect this is a contingent fact about how systems are built, not a necessary fact about how reasoning works. The premise of this project is that **reading is a transferable act** — that the result of comprehension can be handed from the model that did the reading to the model that will do the reasoning, so the second model recalls the document rather than re-reading it.

## What this is not

It is tempting to call this compression, and we want to refuse that word deliberately.

Compression asks: *how small can the document be made?* It produces a gist, a summary, a smaller smell of the original. Compression is lossy by design — it trades fidelity for size, and accepts that fine detail will be sacrificed to fit a budget. A compressed document tells you what the document was *about*. It cannot tell you the date in the third paragraph, the name buried on page nine, the exact figure that the question happens to turn on.

This project is not compression because **it does not want to discard.** It wants to *relocate.* The distinction is the entire philosophy. We are not trying to make the document smaller; we are trying to move the act of reading it from an expensive reader to a cheap one, while preserving the document's full, retrievable specificity. The handoff carries facts, not impressions — the needle, not the haystack's silhouette.

This is why the small model matters in the way it does. Small models are not chosen because they summarize well. They are chosen because they *read* well — they are genuinely strong at finding the needle in the haystack, at recalling exact information from heavy context. The division of labor is precise: **the small model reads, the large model reasons.** Reading is the act of locating and binding what is in the text; reasoning is the act of thinking with it. These are different competencies, and there is no law requiring the same model to perform both. We outsource the reading to the model that is cheap and good at it, and reserve the expensive model for the thing only it can do.

## The shape of the claim

The economic argument follows from the division of labor, but the *interesting* claim is the one underneath it.

A transformer's reasoning operates over its own internal representations — the keys and values it attends to, the residual stream it propagates. For the large model to reason over a document "as if it had read it," there must exist, somewhere inside it, representations its own attention can read. The naive way to create those representations is to run the document through the model's layers. But those representations are just *vectors in a space* — and a space can be reached by more than one road. If a cheaper model can produce vectors that land in the same place, the large model cannot tell the difference between a document it read and a document it was handed. The reading was done elsewhere; the recall is its own.

This is the soul of the project: **the trace of reading is portable.** Not the tokens, not the text, but the internal state that reading produces — the document already metabolized into the form reasoning consumes. If that state can be manufactured cheaply and injected faithfully, the large model inherits a reading it never performed.

## The discipline the pivot imposes

This project arrives at its current form through a failure, and the failure taught it humility about its own evidence.

The first attempt tried to hand over a single vector — the whole document collapsed to one point — and measured success by how close that point landed to its target. The vectors landed close. The method did not work. The model, handed its single vector, answered from memory and ignored the document entirely. The lesson was not "translate better." The lesson was epistemic: **proximity in representation space certifies nothing about whether reasoning can read the representation.** A vector can be correct by every geometric measure and useless by the only measure that matters — whether the model, given it, answers as though it had read.

So the pivot is not only architectural; it is methodological. The project now refuses to trust any proxy for the thing it actually wants. It does not ask "is the vector close?" It asks "does the model answer the question?" And it does not let itself be fooled by documents the model already knows — because a model reciting from memory looks identical to a model reading from a handoff, until you give it a fact it cannot have memorized and watch which one still knows it.

This is why the work proceeds as a chain of proofs before a line of the translation is built. First: can the large model reason over its *own* true reading, handed back to it, on facts no model could guess? Only if reading is recallable in principle is there any point in making it cheap to produce. The receiver must be proven before the sender is built. We earn each claim before we are allowed to assume it.

## Why it matters if it's true

If reading is genuinely transferable, the consequence is larger than a faster pipeline. It means comprehension is not bound to the comprehender — that the expensive, repeated act of ingesting context can be performed once, by whoever is cheapest, and inherited by whoever is wisest. It means a small model's fluency at *finding* and a large model's depth at *thinking* can be composed rather than collapsed into one model that must be excellent at both. It suggests that the internal states of language models are not private, model-specific artifacts but points in a shared space that can be reached, handed over, and recalled.

The honest version of the project does not assume any of this. It suspects it, and sets out to find the document length, the layer, the representation, and the objective at which the suspicion either holds or breaks. The wager is that reading, once done, need not be done again — and the work is the patient, falsifiable test of whether that wager pays.

---

## Where Phase Two stands

The manifesto above sets the standard; the code is where it gets earned. Phase Two is the **receiver-first proof** — establishing that a handed-over reading is recallable *in principle* before any cheap sender (the stitcher) is trusted to produce one.

- **Phase One (closed):** translate a document to a single layer-30 vector and inject it. Measured by representation proximity (cos ≈ 0.37, Top-1 99.3%). The proxy passed; the act failed — Condition B answered from memory. This is the failure that disciplines everything after it. See [`oracle_probe_report.md`](oracle_probe_report.md).
- **Phase Two (current):** stop trusting proxies. Hand the large model its *own* true layer-30 reading and ask whether it can answer — on **synthetic facts no model could have memorized**, gated on the no-context baseline failing first, with a **wrong-document control** as the falsifier. Implemented in [`../evaluate/oracle_probe_v2.py`](../evaluate/oracle_probe_v2.py).

The decision rule is the manifesto made executable: *injection genuinely works* only if the right document's reading lets the model answer a fact it otherwise cannot, **and** the wrong document's reading does not. Proximity is no longer admissible as evidence. The receiver must be proven before the sender is built.

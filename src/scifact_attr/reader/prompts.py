"""Prompts shared by inference, training, and perturbation tests."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

CLOSED_BOOK = """You are a scientific claim verifier. Decide whether the following claim is supported,
or refuted by the scientific literature, or whether there is not enough information
to decide. Answer with one label only: SUPPORT, REFUTE, or NOT_ENOUGH_INFO.

Claim: {claim}
Label:"""

GROUNDED_LABEL_ONLY = """You are a scientific claim verifier. Use only the evidence below. Answer
with one label only: SUPPORT, REFUTE, or NOT_ENOUGH_INFO.

{context}

Claim: {claim}
Label:"""

GROUNDED_WITH_DEFINITIONS = """You are a scientific claim verifier. Use only the evidence below.
Choose SUPPORT if it directly supports the claim, REFUTE if it directly contradicts
the claim, or NOT_ENOUGH_INFO if it does neither, even if it discusses the same topic.
Answer with one label only.

{context}

Claim: {claim}
Label:"""

COT_WITH_CITATIONS = """Compare the claim with the numbered evidence sentences.

{context}

Claim: {claim}

Start with the relevant evidence IDs and a colon. Format: E1: one short sentence
explaining how the evidence relates to the claim. Use the actual IDs from above.
If no sentence addresses the claim, write "No relevant evidence." """

TEMPLATES = {
    "closed_book": CLOSED_BOOK,
    "grounded_label_only": GROUNDED_LABEL_ONLY,
    "grounded_with_definitions": GROUNDED_WITH_DEFINITIONS,
    "cot_with_citations": COT_WITH_CITATIONS,
}


def render_abstracts(abstracts: Sequence[Dict[str, Any]]) -> str:
    """Abstracts with per-sentence ids, for C1 (no sentence selection)."""
    blocks = []
    for doc in abstracts:
        head = f"[{doc['doc_id']}] {doc.get('title', '')}".strip()
        lines = [f"  [{doc['doc_id']}#{i}] {s}" for i, s in enumerate(doc["sentences"])]
        blocks.append(head + "\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def render_sentences(sentences: Sequence[Dict[str, Any]]) -> str:
    """Selected sentences with alias ids E1..En, for C2..C5.

    The alias is what the model is asked to cite; the map back to (doc_id, sent_idx) is
    passed to the parser, so a citation is never resolved by string matching.
    """
    lines = []
    for i, sent in enumerate(sentences, start=1):
        lines.append(f"E{i}. [{sent['doc_id']}#{sent.get('sent_idx', '?')}] {sent['text']}")
    return "\n".join(lines)


def build_prompt(template: str, claim: str, abstracts: Optional[Sequence[Dict[str, Any]]] = None,
                 sentences: Optional[Sequence[Dict[str, Any]]] = None, note: str = "") -> str:
    if template not in TEMPLATES:
        raise KeyError(f"unknown template {template!r}; available: {sorted(TEMPLATES)}")
    if template == "closed_book":
        return TEMPLATES[template].format(claim=claim)
    if sentences is not None:
        context = render_sentences(sentences)
    elif abstracts is not None:
        context = render_abstracts(abstracts)
    else:
        raise ValueError("grounded templates need abstracts or sentences")
    if note:
        context += f"\n\nEvidence assessment (check against the sentences): {note}"
    return TEMPLATES[template].format(claim=claim, context=context)

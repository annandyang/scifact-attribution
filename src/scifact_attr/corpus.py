"""Corpus preparation: deterministic, abbreviation-aware sentence splitting.

Sentence boundaries must be identical in every condition. If C1 sees fifteen sentences in
an abstract and C2 sees twelve, the difference between them is no longer the selector, and
the attribution is broken before it starts. Hence a rule-based splitter with a fixed
abbreviation list, rather than a model-based sentence tokeniser whose output can drift with
the installed version.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Abbreviations whose trailing period does not end a sentence.
ABBREVIATIONS = {
    "al", "e.g", "i.e", "vs", "dr", "prof", "fig", "figs", "eq", "eqs", "ref", "refs",
    "no", "approx", "ca", "cf", "resp", "sec", "suppl", "vol", "pp", "wt", "i.e",
    "u.s", "u.k", "e.u", "st", "ltd", "inc", "corp", "univ", "dept", "min", "max",
    "s.d", "s.e", "c.i", "rpm", "kg", "mg", "ml", "dl", "cm", "mm", "nm", "hz",
}

_SENT_END = re.compile(r"(?<=[.!?])\s+")
_INITIALS = re.compile(r"^[A-Z]\.$")
_NUMBERED = re.compile(r"^(\(?\d+|[a-z])\.$")


def _ends_mid_sentence(fragment: str) -> bool:
    """True if the fragment's last token cannot end a sentence."""
    token = fragment.rsplit(" ", 1)[-1]
    bare = token.rstrip(".!?").lower().lstrip("(")
    if bare in ABBREVIATIONS:
        return True
    if _INITIALS.match(token):
        return True
    # "3." / "(3)." / "a." at the head of a list item, and decimals such as "3.5"
    if _NUMBERED.match(token):
        return True
    return False


def split_sentences(text: str, min_chars: int = 3) -> List[str]:
    """Split abstract text into sentences without breaking on abbreviations or decimals.

    Deterministic: the same input always yields the same list, on any platform.
    """
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return []

    fragments = _SENT_END.split(cleaned)
    sentences: List[str] = []
    buffer = ""
    for fragment in fragments:
        buffer = f"{buffer} {fragment}".strip() if buffer else fragment
        if _ends_mid_sentence(buffer):
            continue
        if len(buffer) >= min_chars:
            sentences.append(buffer)
            buffer = ""
    if buffer and len(buffer) >= min_chars:
        sentences.append(buffer)
    return sentences


def sentence_key(doc_id: int, sent_idx: int) -> str:
    """Stable identifier for a sentence, used in prompts and prediction rows."""
    return f"{doc_id}#{sent_idx}"


def build_sentence_table(corpus: List[Dict[str, Any]], dedupe: bool = True
                         ) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], str]]:
    """Flatten abstracts into sentences for sentence-level indexing and selection.

    Returns the sentence records and a mapping from (doc_id, sentence index) to sentence id.
    With dedupe on, identical sentence texts share one id; this prevents a sentence that
    appears in twenty abstracts from being scored twenty times for the same claim.
    """
    records: List[Dict[str, Any]] = []
    lookup: Dict[Tuple[int, int], str] = {}
    seen: Dict[str, str] = {}

    for doc in corpus:
        for idx, sent in enumerate(doc["sentences"]):
            key = sentence_key(doc["doc_id"], idx)
            digest = re.sub(r"\W+", "", sent.lower())
            if dedupe and digest and digest in seen:
                lookup[(doc["doc_id"], idx)] = seen[digest]
                continue
            if dedupe and digest:
                seen[digest] = key
            lookup[(doc["doc_id"], idx)] = key
            records.append({
                "sid": key,
                "doc_id": doc["doc_id"],
                "sent_idx": idx,
                "text": sent,
                "n_tokens": len(sent.split()),
            })
    return records, lookup


def numbered_abstract(doc: Dict[str, Any]) -> str:
    """Render an abstract with sentence ids, so the prompt can cite them."""
    lines = [f"[{sentence_key(doc['doc_id'], i)}] {s}" for i, s in enumerate(doc["sentences"])]
    header = f"[{doc['doc_id']}] {doc['title']}" if doc.get("title") else f"[{doc['doc_id']}]"
    return header + "\n" + "\n".join(lines)

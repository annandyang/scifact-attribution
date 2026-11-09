"""Parsing model output into a label, cited sentence ids and a rationale.

Parsing is treated as part of the measurement, not as plumbing. A parser that silently
defaults to SUPPORT when it cannot find a label would inflate accuracy in exactly the
condition where the model failed to follow the format, and the inflation would be
invisible. Unparseable output is therefore recorded as its own category and counted.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .. import LABEL_ALIASES

_LABEL_PATTERNS = [
    re.compile(r"^\s*LABEL\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*(?:final\s+)?(?:answer|label)\s*[:\-]\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
]
_ALIAS_IN_TEXT = re.compile(r"\bE(\d{1,2})\b")
_DOC_SENT = re.compile(r"\b(\d+#\d+)\b")
_RATIONALE = re.compile(r"^\s*RATIONALE\s*:\s*(.*)$", re.MULTILINE | re.IGNORECASE | re.DOTALL)
_EVIDENCE_BLOCK = re.compile(r"^\s*EVIDENCE\s*:\s*(.*?)(?:^\s*(?:LABEL|RATIONALE)\s*:|\Z)",
                             re.MULTILINE | re.IGNORECASE | re.DOTALL)
_CANON = re.compile(r"NOT[\s_]?ENOUGH[\s_]?(?:INFO|INFORMATION)|SUPPORT(?:ED)?|REFUTE(?:D)?|CONTRACT(?:ED)?")


def normalise_label(raw: str) -> Optional[str]:
    """Map a free-text label surface form onto SUPPORT / REFUTE / NEI."""
    if not raw:
        return None
    upper = raw.strip().upper().replace("-", "_").replace(" ", "_").strip(".:;,")
    if upper in LABEL_ALIASES:
        return LABEL_ALIASES[upper]
    match = _CANON.search(upper)
    if not match:
        return None
    token = match.group(0).replace(" ", "_").replace("SUPPORTED", "SUPPORT")
    token = token.replace("REFUTED", "REFUTE").replace("CONTRACTED", "CONTRACT")
    token = "NOT_ENOUGH_INFO" if token.startswith("NOT") else token
    return LABEL_ALIASES.get(token)


def _clean_citations(token_source: str, sid_by_alias: Dict[str, str]) -> List[str]:
    """Resolve citations to sentence ids, preferring explicit doc#sent over E-aliases."""
    allowed = set(sid_by_alias.values())
    found = [sid for sid in _DOC_SENT.findall(token_source) if not allowed or sid in allowed]
    for alias_idx in _ALIAS_IN_TEXT.findall(token_source):
        key = f"E{int(alias_idx)}"
        if key in sid_by_alias:
            found.append(sid_by_alias[key])
    return sorted(set(found))


def parse_output(text: str, sid_by_alias: Optional[Dict[str, str]] = None,
                 expect_reasoning: bool = False) -> Dict[str, Any]:
    """Return label, cited ids, rationale, and a parse status.

    status is one of: ok, label_unparseable, format_violation (reasoning condition with
    no EVIDENCE or LABEL block), truncated.
    """
    sid_by_alias = sid_by_alias or {}
    result: Dict[str, Any] = {
        "label": None, "cited_ids": [], "rationale": "", "status": "ok",
    }
    if not text or not text.strip():
        result["status"] = "empty"
        return result

    raw_label = None
    for pattern in _LABEL_PATTERNS:
        match = pattern.search(text)
        if match:
            raw_label = match.group(1)
            break
    if raw_label is None and not expect_reasoning:
        # fall back to the last canonical label token anywhere in the output
        tokens = _CANON.findall(text.upper())
        raw_label = tokens[-1] if tokens else None

    if raw_label and expect_reasoning:
        key = re.sub(r"[\s-]+", "_", raw_label.strip(" *`.:;,").upper())
        result["label"] = LABEL_ALIASES.get(key)
    elif raw_label:
        result["label"] = normalise_label(raw_label)
    if result["label"] is None:
        result["status"] = "label_unparseable"

    if expect_reasoning:
        evidence = _EVIDENCE_BLOCK.search(text)
        rationale = _RATIONALE.search(text)
        if evidence:
            result["cited_ids"] = _clean_citations(evidence.group(1), sid_by_alias)
            if result["status"] == "ok" and (
                not rationale or (result["label"] in {"SUPPORT", "REFUTE"} and not result["cited_ids"])
            ):
                result["status"] = "format_violation"
        elif result["status"] == "ok":
            result["status"] = "format_violation"
        if rationale:
            result["rationale"] = rationale.group(1).strip()[:600]
    else:
        # in label-only conditions a stray citation means the model ignored the instruction
        if _DOC_SENT.search(text):
            result["status"] = "format_violation"
            result["cited_ids"] = sorted(set(_DOC_SENT.findall(text)))

    if "\ufffd" in text or text.rstrip().endswith(("EVIDENCE:", "RATIONALE:")):
        result["status"] = "truncated"
    return result


def is_usable(parsed: Dict[str, Any]) -> bool:
    return parsed.get("label") is not None and parsed.get("status") in {"ok", "truncated"}


def parse_evidence_assessment(text: str, sid_by_alias: Dict[str, str]) -> tuple[list[str], str, str]:
    if not sid_by_alias:
        return [], "No evidence sentences passed selection.", "ok"
    rationale = " ".join(text.split())[:600]
    cited = _clean_citations(rationale, sid_by_alias)
    ids = {f"E{int(index)}" for index in _ALIAS_IN_TEXT.findall(rationale)}
    unknown = ids - sid_by_alias.keys()
    if unknown or (not cited and rationale.strip('".').casefold() != "no relevant evidence"):
        return [], rationale, "uncited_reasoning"
    return cited, rationale, "ok"


def citation_metrics(predicted_ids: Sequence[str], gold_ids: Sequence[str]) -> Dict[str, float]:
    """Citation precision, recall and F1 against the gold rationale sentences.

    Precision is the quantity a reader cares about (how many of the cited sentences are
    genuinely supporting), and it is the one a plausible-but-irrelevant rationale fails.
    """
    gold = {str(g) for g in gold_ids}
    pred = {str(p) for p in predicted_ids}
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_cited": 0.0}
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_cited": float(len(pred))}
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_cited": float(len(pred))}

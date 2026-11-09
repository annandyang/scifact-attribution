"""Load and verify the official SciFact release."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen

from . import LABEL_ALIASES
from .corpus import sentence_key, split_sentences
from .utils.io import read_jsonl

EXPECTED = {
    "claims_train": 809,
    "claims_dev": 300,
    "claims_test": 300,
    "corpus": 5183,
}
SPLITS = ("train", "dev", "test")
OFFICIAL_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
OFFICIAL_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"


def normalise_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    key = str(raw).strip().upper().replace(" ", "_")
    if key == "CONTRADICT":
        return "REFUTE"
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    raise ValueError(f"unrecognised label {raw!r}; known aliases: {sorted(LABEL_ALIASES)}")


def normalise_document(row: dict) -> dict:
    abstract = row.get("abstract")
    if isinstance(abstract, list):
        # SciFact evidence indices refer directly to the released sentence list.
        sentences = abstract
        text = " ".join(sentences)
    else:
        text = (row.get("text") or abstract or "").strip()
        sentences = split_sentences(text)
    return {
        "doc_id": int(row["doc_id"]),
        "title": (row.get("title") or "").strip(),
        "text": text,
        "sentences": sentences,
        "n_sentences": len(sentences),
    }


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    """Return one record per abstract, with sentences and a stable sentence index."""
    docs = [normalise_document(row) for row in read_jsonl(path)]
    if not docs:
        raise ValueError(f"no abstracts read from {path}")
    return docs


def load_claims(path: Path, split: str) -> List[Dict[str, Any]]:
    """Normalize labels and keep both the alternative rationale sets and their union."""
    claims: List[Dict[str, Any]] = []
    for row in read_jsonl(path):
        evidence = row.get("evidence")
        labels = set()
        doc_ids = []
        gold_sets = []
        indices = []
        if evidence:
            if "doc_id" in evidence:
                legacy = evidence
                evidence = {legacy["doc_id"]: [
                    {"label": legacy["label"], "sentences": sentences}
                    for sentences in (legacy.get("rationales") or [legacy.get("sentences", [])])
                ]}
            for doc_id, rationales in sorted(evidence.items(), key=lambda item: int(item[0])):
                doc_ids.append(int(doc_id))
                for rationale in rationales:
                    labels.add(normalise_label(rationale["label"]))
                    sentence_indices = [int(idx) for idx in rationale["sentences"]]
                    indices.append(sentence_indices)
                    gold_sets.append([sentence_key(int(doc_id), idx) for idx in sentence_indices])
            if len(labels) != 1 or None in labels:
                raise ValueError(f"claim {row['id']}: expected one consistent evidence label")

        label = next(iter(labels), "NEI") if evidence is not None else None

        claims.append({
            "claim_id": str(row["id"]),
            "claim": row["claim"].strip(),
            "split": split,
            "label": label,
            "gold_doc_ids": doc_ids,
            "gold_sentence_ids": list(dict.fromkeys(sid for group in gold_sets for sid in group)),
            "gold_evidence_sets": gold_sets,
            "gold_sentences": sorted({idx for group in indices for idx in group}),
            "rationales": indices,
            "cited_doc_ids": [int(d) for d in (row.get("cited_doc_ids") or [])],
        })
    if not claims:
        raise ValueError(f"no claims read from {path}")
    return claims


def verify(corpus: List[Dict[str, Any]],
           claims: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compare loaded counts against the published ones and report gaps."""
    report: Dict[str, Any] = {"corpus": len(corpus), "claims": {k: len(v) for k, v in claims.items()}}
    problems: List[str] = []

    if len(corpus) != EXPECTED["corpus"]:
        problems.append(f"corpus: expected {EXPECTED['corpus']}, found {len(corpus)}")
    for split in SPLITS:
        expected = EXPECTED[f"claims_{split}"]
        found = len(claims.get(split, []))
        if found != expected:
            problems.append(f"claims_{split}: expected {expected}, found {found}")

    sentence_counts = {d["doc_id"]: d["n_sentences"] for d in corpus}
    if len(sentence_counts) != len(corpus):
        problems.append("duplicate document IDs")
    claim_ids = [claim["claim_id"] for group in claims.values() for claim in group]
    if len(set(claim_ids)) != len(claim_ids):
        problems.append("duplicate claim IDs within or across splits")

    report["label_counts"] = {}
    for split, group in claims.items():
        counts = {label: 0 for label in ("SUPPORT", "REFUTE", "NEI", "unlabeled")}
        missing_docs = 0
        invalid_sentences = 0
        for claim in group:
            counts[claim["label"] or "unlabeled"] += 1
            missing_docs += sum(doc_id not in sentence_counts for doc_id in claim["gold_doc_ids"])
            for sentence_id in claim["gold_sentence_ids"]:
                doc_id, idx = map(int, sentence_id.split("#"))
                invalid_sentences += not 0 <= idx < sentence_counts.get(doc_id, 0)
        report["label_counts"][split] = counts
        report[f"{split}_gold_docs_missing_from_corpus"] = missing_docs
        report[f"{split}_invalid_gold_sentences"] = invalid_sentences
        if missing_docs or invalid_sentences:
            problems.append(f"{split}: {missing_docs} missing evidence documents, "
                            f"{invalid_sentences} invalid evidence sentence IDs")
        if split in ("train", "dev") and counts["unlabeled"]:
            problems.append(f"{split}: {counts['unlabeled']} unlabeled claims")

    report["abstracts_without_title"] = sum(not d["title"] for d in corpus)
    report["abstracts_without_sentences"] = sum(d["n_sentences"] == 0 for d in corpus)
    nei_empty = sum(1 for c in claims.get("train", [])
                    if c["label"] == "NEI" and not c["gold_sentences"])
    nei_total = sum(1 for c in claims.get("train", []) if c["label"] == "NEI")
    report["nei_train_total"] = nei_total
    report["nei_train_without_evidence"] = nei_empty
    report["problems"] = problems
    report["ok"] = not problems
    return report


def load_all(raw_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]],
                                     Dict[str, Any]]:
    raw_dir = Path(raw_dir)
    corpus = load_corpus(raw_dir / "corpus.jsonl")
    claims = {}
    for split in SPLITS:
        path = raw_dir / f"claims_{split}.jsonl"
        claims[split] = load_claims(path, split) if path.exists() else []
    report = verify(corpus, claims)
    if not report["ok"]:
        raise ValueError("SciFact validation failed: " + "; ".join(report["problems"]))
    return corpus, claims, report


def acquire_official(raw_dir: Path, archive: Optional[Path] = None) -> None:
    """Validate the official tarball before replacing any raw inputs."""
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=raw_dir.parent) as tmp:
        staging = Path(tmp)
        if archive is None:
            archive = staging / "data.tar.gz"
            with urlopen(OFFICIAL_URL, timeout=60) as response, archive.open("wb") as target:
                shutil.copyfileobj(response, target)
        with archive.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if checksum != OFFICIAL_SHA256:
            raise ValueError(f"SciFact archive checksum mismatch: {checksum}")

        with tarfile.open(archive, "r:gz") as tar:
            for name in EXPECTED:
                member = tar.getmember(f"data/{name}.jsonl")
                if not member.isfile():
                    raise ValueError(f"SciFact archive member is not a file: {member.name}")
                with tar.extractfile(member) as source, (staging / f"{name}.jsonl").open("wb") as target:
                    shutil.copyfileobj(source, target)
        load_all(staging)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name in EXPECTED:
            (staging / f"{name}.jsonl").replace(raw_dir / f"{name}.jsonl")

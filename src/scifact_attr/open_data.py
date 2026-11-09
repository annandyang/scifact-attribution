"""SciFact-Open annotations and a disk-backed corpus lookup."""

import json
import tarfile
from functools import lru_cache
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from .corpus import sentence_key
from .data import normalise_document, normalise_label
from .utils.io import read_jsonl, write_jsonl

ARCHIVE_URL = "https://scifact.s3.us-west-2.amazonaws.com/scifact-open/latest/scifact_open.tar.gz"
SCOPES = ("full", "candidates")


def normalise_open_claim(row):
    evidence = row["evidence"]
    labels = {normalise_label(item["label"]) for item in evidence.values()}
    gold_sets = [
        [sentence_key(int(doc_id), int(idx)) for idx in item["sentences"]]
        for doc_id, item in evidence.items() if item["provenance"] == "citation"
    ]
    return {
        "claim_id": str(row["id"]),
        "claim": row["claim"].strip(),
        "split": "open",
        "label": "NEI" if not labels else next(iter(labels)) if len(labels) == 1 else None,
        "gold_doc_ids": sorted(int(doc_id) for doc_id in evidence),
        "gold_evidence_sets": gold_sets,
        "gold_sentence_ids": list(dict.fromkeys(sid for group in gold_sets for sid in group)),
    }


def read_archive_member(archive, filename):
    candidates = [item for item in archive.getmembers()
                  if PurePosixPath(item.name).parts == ("data", filename)]
    if len(candidates) != 1 or not candidates[0].isfile():
        raise ValueError(f"Expected data/{filename} in the official SciFact-Open archive")
    with archive.extractfile(candidates[0]) as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def prepare_open(archive_path, destination, scope, official_test):
    corpus_name = "corpus.jsonl" if scope == "full" else "corpus_candidates.jsonl"
    original = {row["claim_id"]: row["claim"] for row in read_jsonl(official_test)}
    destination.mkdir(parents=True, exist_ok=True)
    with (tarfile.open(archive_path, "r:gz") as archive,
          TemporaryDirectory(dir=destination) as tmp):
        staging = Path(tmp)
        claims = [normalise_open_claim(row) for row in read_archive_member(archive, "claims.jsonl")]
        claim_ids = {row["claim_id"] for row in claims}
        if len(claims) != 279 or len(claim_ids) != len(claims):
            raise ValueError("Expected 279 unique SciFact-Open claims")
        if any(original.get(row["claim_id"]) != row["claim"] for row in claims):
            raise ValueError("SciFact-Open claims must match the official SciFact test claims")

        gold_docs = {doc_id for row in claims for doc_id in row["gold_doc_ids"]}
        seen = set()

        def corpus_rows():
            for raw_doc in read_archive_member(archive, corpus_name):
                doc = normalise_document(raw_doc)
                doc_id = doc["doc_id"]
                if doc_id in seen:
                    raise ValueError(f"Duplicate SciFact-Open document ID: {doc_id}")
                seen.add(doc_id)
                yield doc

        n_docs = write_jsonl(staging / "corpus.jsonl", corpus_rows())
        if (scope == "full" and n_docs != 500000) or not n_docs:
            raise ValueError(f"Unexpected SciFact-Open {scope} corpus size: {n_docs}")
        if not gold_docs.issubset(seen):
            raise ValueError(f"{len(gold_docs - seen)} evidence documents missing from {scope}")
        write_jsonl(staging / "claims_open.jsonl", claims)
        manifest = {
            "scope": scope,
            "n_docs": n_docs,
            "n_claims": len(claims),
            "n_ambiguous_claims": sum(row["label"] is None for row in claims),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for name in ("corpus.jsonl", "claims_open.jsonl", "manifest.json"):
            (staging / name).replace(destination / name)
    return manifest


class DiskCorpus:
    def __init__(self, path):
        self.path = path
        self.offsets = {}
        with path.open("rb") as source:
            while line := source.readline():
                doc_id = int(json.loads(line)["doc_id"])
                self.offsets[doc_id] = source.tell() - len(line)

    def __contains__(self, doc_id):
        return doc_id in self.offsets

    @lru_cache(maxsize=2048)
    def __getitem__(self, doc_id):
        with self.path.open("rb") as source:
            source.seek(self.offsets[doc_id])
            return json.loads(source.readline())


class DiskSentences:
    def __init__(self, corpus, limit):
        self.corpus = corpus
        self.limit = limit

    def get(self, doc_id, default):
        if doc_id not in self.corpus:
            return default
        return [
            {"doc_id": doc_id, "sent_idx": idx, "text": text}
            for idx, text in enumerate(self.corpus[doc_id]["sentences"][:self.limit])
        ]

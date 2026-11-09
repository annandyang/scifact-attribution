"""Build the sparse and dense document indexes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.config import REPO_ROOT, load_base
from scifact_attr.open_data import SCOPES
from scifact_attr.retrieval.bm25 import BM25Index
from scifact_attr.utils.io import get_logger, load_jsonl_as_list, read_jsonl

LOG = get_logger("index")


def build_open_dense(corpus_path, n_docs, encoder, settings, indexes, seed, scope):
    import numpy as np

    from scifact_attr.retrieval.dense import IVFIndex, build_flat

    params = settings["ivfpq"]
    approximate = None
    if scope == "full":
        approximate = IVFIndex(dim=int(settings["dim"]), nlist=int(params["nlist"]),
                               m=int(params["m"]), nbits=int(params["nbits"]))
    with TemporaryDirectory(dir=indexes) as tmp:
        staging = Path(tmp)
        vector_path = staging / "dense_open.npy"
        vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype=np.float32,
                                            shape=(n_docs, int(settings["dim"])))
        ids = []
        docs = read_jsonl(corpus_path)
        while batch := list(islice(docs, 512)):
            flat = build_flat(batch, encoder)
            start = len(ids)
            vectors[start:start + len(batch)] = flat.vectors
            ids.extend(flat.ids)
            if len(ids) % 8192 == 0:
                LOG.info("encoded %d/%d documents", len(ids), n_docs)
        if len(ids) != n_docs:
            raise ValueError("SciFact-Open corpus changed since preparation")
        (staging / "dense_open.ids.json").write_text(json.dumps(ids), encoding="utf-8")
        vectors.flush()
        if approximate is not None:
            LOG.info("training the IVF-PQ index")
            approximate.fit_large(vectors, ids, seed, int(params.get("nprobe", 32)))
            approximate.save(staging / "ivf_open")
        del vectors
        for path in staging.iterdir():
            path.replace(indexes / path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "all"])
    parser.add_argument("--corpus", default="processed", choices=["processed", "open"])
    parser.add_argument("--open-scope", default="full", choices=SCOPES)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_base(args.config)
    processed = REPO_ROOT / cfg["paths"]["processed"]
    indexes = REPO_ROOT / cfg["paths"]["indexes"]
    tag = "scifact"
    if args.corpus == "open":
        tag = "open"
        processed = processed / tag / args.open_scope
        indexes = indexes / tag / args.open_scope
        manifest = processed / "manifest.json"
        if not manifest.exists():
            raise SystemExit("Run scripts/11_scifact_open.py prepare for the selected scope.")
        n_docs = json.loads(manifest.read_text(encoding="utf-8"))["n_docs"]
    else:
        corpus = load_jsonl_as_list(processed / "corpus.jsonl")
        n_docs = len(corpus)
    corpus_path = processed / "corpus.jsonl"
    LOG.info("%s corpus: %d abstracts", tag, n_docs)
    indexes.mkdir(parents=True, exist_ok=True)

    bm25_cfg = cfg["retrieval"]["bm25"]
    dense_cfg = cfg["retrieval"]["dense"]
    if args.retriever in ("bm25", "all"):
        started = time.perf_counter()
        docs = ((str(doc["doc_id"]), f"{doc['title']}. {doc['text']}") for doc in read_jsonl(corpus_path))
        index = BM25Index(k1=float(bm25_cfg["k1"]), b=float(bm25_cfg["b"])).fit(docs)
        path = indexes / f"bm25_{tag}.pkl"
        index.save(path)
        del index
        LOG.info("BM25 built in %.1fs -> %s", time.perf_counter() - started, path)

    if args.retriever in ("dense", "all"):
        from scifact_attr.retrieval.dense import Encoder, build_flat

        started = time.perf_counter()
        encoder = Encoder(dense_cfg["model"], device=args.device,
                          batch_size=int(dense_cfg["batch_size"]), revision=dense_cfg.get("revision"))
        if tag == "open":
            build_open_dense(corpus_path, n_docs, encoder, dense_cfg, indexes,
                             int(cfg["seed"]), args.open_scope)
        else:
            build_flat(corpus, encoder).save(indexes / f"dense_{tag}")
        LOG.info("Dense index built in %.1fs", time.perf_counter() - started)

    meta = {"tag": tag, "n_docs": n_docs, "retriever": args.retriever,
            "bm25": bm25_cfg, "dense_model": dense_cfg["model"], "revision": dense_cfg.get("revision")}
    with (indexes / f"index_meta_{tag}.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Indexes written to {indexes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run one condition and save per-claim predictions."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.config import adapter_dir, load_condition
from scifact_attr.open_data import DiskCorpus, DiskSentences, SCOPES
from scifact_attr.reader.infer import Reader
from scifact_attr.selection.selector import CrossEncoderSelector, OverlapSelector
from scifact_attr.system import (BM25Retriever, DenseRetriever, HybridRetriever, RunContext,
                                 RerankedRetriever, run_condition)
from scifact_attr.utils.io import get_logger, load_jsonl_as_list, write_jsonl
from scifact_attr.utils.io import set_seed

LOG = get_logger("run")


def build_retriever(cfg: Dict[str, Any], processed: Path, indexes: Path, mode: str,
                    open_scope=None, exact_index=False):
    corpus_path = processed / "corpus.jsonl"
    corpus = DiskCorpus(corpus_path) if open_scope else {
        int(d["doc_id"]): d for d in load_jsonl_as_list(corpus_path)
    }
    tag = "open" if open_scope else "scifact"
    bm25_path = indexes / f"bm25_{tag}.pkl"
    if not bm25_path.exists():
        raise SystemExit(f"BM25 index missing: {bm25_path}. Run scripts/02_build_index.py first.")

    from scifact_attr.retrieval.bm25 import BM25Index

    bm25 = BM25Retriever(BM25Index.load(bm25_path))

    if mode == "bm25":
        return bm25, corpus

    from scifact_attr.retrieval.dense import Encoder, FlatIndex, IVFIndex

    if open_scope == "full" and not exact_index:
        dense_prefix = indexes / "ivf_open"
        if not dense_prefix.with_suffix(".faiss").exists():
            raise SystemExit(f"IVF index missing: {dense_prefix}.faiss")
        dense_index = IVFIndex.load(dense_prefix)
    else:
        dense_prefix = indexes / f"dense_{tag}"
        if not dense_prefix.with_suffix(".npy").exists():
            raise SystemExit(f"Dense index missing: {dense_prefix}.npy")
        dense_index = FlatIndex.load(dense_prefix)

    dense_cfg = cfg["retrieval"]["dense"]
    dense = DenseRetriever(dense_index,
                           Encoder(dense_cfg["model"], revision=dense_cfg.get("revision")))
    fuse = cfg["retrieval"]["fuse"]
    retriever = HybridRetriever(bm25, dense, k=int(fuse["k"]), candidate_depth=int(fuse["top_k"]))

    if mode == "rerank":
        rerank_cfg = cfg["retrieval"]["rerank"]
        reranker = CrossEncoderSelector(rerank_cfg["model"], revision=rerank_cfg.get("revision"))
        retriever = RerankedRetriever(retriever, reranker, corpus,
                                      top_n=int(cfg["retrieval"]["rerank"]["top_n"]))
    return retriever, corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train", "open"])
    parser.add_argument("--open-scope", default="full", choices=SCOPES)
    parser.add_argument("--exact-index", action="store_true", help="use exact search for SciFact-Open")
    parser.add_argument("--retriever", default=None, choices=["bm25", "hybrid", "rerank"])
    parser.add_argument("--seeds", default=None, help="comma separated; overrides the config")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="no model; exercises the pipeline")
    parser.add_argument("--out", default=None)
    parser.add_argument("--adapter-base", help="adapter path before the _sSEED suffix; requires --out")
    parser.add_argument("--precision", choices=("qlora", "bf16_lora"), default="qlora")
    parser.add_argument("--require-evidence", action="store_true",
                        help="C2 only: return NEI for an empty selection; requires --out")
    parser.add_argument("--smoke-adapter", action="store_true", help="load the separate smoke adapter")
    args = parser.parse_args()

    cfg = load_condition(args.condition)
    if cfg["condition"] == "c5":
        if args.split == "open":
            raise SystemExit("Run C4 first, then apply the frozen gate with scripts/11_scifact_open.py score.")
        raise SystemExit("C5 reuses C4 predictions. Run python scripts/06_calibrate.py instead.")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.adapter_base and not args.out:
        parser.error("--adapter-base requires --out")
    if args.adapter_base:
        cfg["adapter_path"] = args.adapter_base
    if args.precision == "bf16_lora":
        if not args.adapter_base or not args.out or args.dry_run or not cfg["components"].get("adapter"):
            parser.error("bf16_lora requires a real adapter condition, --adapter-base and --out")
        cfg["reader"]["quant"]["load_in_4bit"] = False
    if args.require_evidence:
        if cfg["condition"] != "c2" or not args.out or args.dry_run:
            parser.error("--require-evidence requires real C2 inference and --out")
        cfg["require_evidence"] = True
    processed = cfg["paths"]["processed"]
    indexes = cfg["paths"]["indexes"]
    predictions = cfg["paths"]["predictions"]
    open_scope = args.open_scope if args.split == "open" else None
    if open_scope:
        processed = processed / "open" / open_scope
        indexes = indexes / "open" / open_scope
        predictions = predictions / "open" / open_scope
    elif args.exact_index:
        parser.error("--exact-index applies only to SciFact-Open")
    if args.require_evidence or args.precision != "qlora":
        canonical = predictions / f"{cfg['condition']}_{args.split}.jsonl"
        out_path = Path(args.out)
        if (out_path.resolve() == canonical.resolve()
                or out_path.with_suffix(".meta.json").resolve() == canonical.with_suffix(".meta.json").resolve()):
            parser.error("An experimental option cannot overwrite the original condition cache.")

    claims = load_jsonl_as_list(processed / f"claims_{args.split}.jsonl")
    if args.limit:
        claims = claims[: args.limit]
    if not claims:
        raise SystemExit(f"no claims in {processed / f'claims_{args.split}.jsonl'}")

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg.get("seeds", [0]))
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("--seeds must contain distinct nonnegative seeds")
    mode = args.retriever or ("bm25" if args.dry_run else cfg["retrieval"]["mode"])

    retriever = None
    corpus: Dict[int, Dict[str, Any]] = {}
    if cfg["components"].get("retrieval"):
        retriever, corpus = build_retriever(cfg, processed, indexes, mode, open_scope,
                                           args.exact_index)

    sentence_limit = int(cfg["corpus"]["max_sentences"])
    sentences_by_doc = {}
    if cfg["components"].get("selection"):
        if open_scope:
            sentences_by_doc = DiskSentences(corpus, sentence_limit)
        else:
            for sent in load_jsonl_as_list(processed / "sentences.jsonl"):
                if sent["sent_idx"] < sentence_limit:
                    sentences_by_doc.setdefault(int(sent["doc_id"]), []).append(sent)
            for doc in sentences_by_doc.values():
                doc.sort(key=lambda s: s["sent_idx"])

    if cfg["components"].get("selection"):
        if args.dry_run:
            # a dry run must work on a machine with no GPU and no encoder installed
            selector = OverlapSelector()
            LOG.warning("dry run: lexical OverlapSelector instead of the cross-encoder")
        else:
            rerank_cfg = cfg["retrieval"]["rerank"]
            selector = CrossEncoderSelector(rerank_cfg["model"], batch_size=16,
                                            revision=rerank_cfg.get("revision"))
    else:
        selector = OverlapSelector()

    if args.dry_run:
        LOG.warning("dry run: labels are placeholders, metrics are meaningless")

    ctx = RunContext(corpus_by_id=corpus, sentences_by_doc=sentences_by_doc, retriever=retriever,
                     selector=selector, config=cfg)

    started = time.perf_counter()
    rows = []
    smoke = args.dry_run or args.limit > 0 or args.smoke_adapter
    for seed in seeds:
        set_seed(int(cfg.get("seed", 42)) + seed, with_torch=not args.dry_run)
        reader = None
        if not args.dry_run:
            path = adapter_dir(cfg, seed, smoke=args.smoke_adapter) if cfg["components"].get("adapter") else None
            if path and not (path / "adapter_config.json").is_file():
                raise SystemExit(f"Adapter missing: {path}. Train seed {seed} with script 05 first.")
            reader = Reader(cfg["reader"]["model"], quant=cfg["reader"]["quant"],
                            adapter_path=str(path) if path else None,
                            revision=cfg["reader"].get("revision"))
            reader.torch.cuda.reset_peak_memory_stats()
        seed_rows = run_condition(claims, ctx, reader, seeds=[seed])
        peak = reader.torch.cuda.max_memory_allocated() / 1024 ** 2 if reader else 0
        for row in seed_rows:
            row["peak_mem_mb"] = round(peak, 1)
            row["mode"] = "smoke" if smoke else "experiment"
        rows.extend(seed_rows)
        if reader:
            torch = reader.torch
            del reader
            gc.collect()
            torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started

    name = f"{cfg['condition']}_{args.split}.jsonl"
    if smoke:
        predictions = predictions / "smoke"
    out = Path(args.out) if args.out else predictions / name
    n = write_jsonl(out, rows)

    meta = {
        "condition": cfg["condition"],
        "condition_name": cfg["condition_name"],
        "split": args.split,
        "open_scope": open_scope,
        "open_dense_index": ("exact" if args.exact_index or open_scope == "candidates" else "ivfpq")
        if open_scope and mode != "bm25" and retriever else None,
        "seeds": seeds,
        "retriever": mode if retriever else "none",
        "n_rows": n,
        "dry_run": bool(args.dry_run),
        "smoke_adapter": bool(args.smoke_adapter),
        "limit": args.limit,
        "reader_revision": cfg["reader"].get("revision"),
        "reasoning_protocol": "base_assessment_then_adapted_verdict"
        if cfg["components"].get("reasoning") else None,
        "wall_clock_s": round(elapsed, 1),
        "sec_per_claim": round(elapsed / max(n, 1), 2),
        "components": cfg["components"],
        **({"reader_precision": args.precision} if args.adapter_base else {}),
        **({"require_evidence": True} if args.require_evidence else {}),
    }
    with open(out.with_suffix(".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    LOG.info("wrote %d rows to %s in %.1fs", n, out, elapsed)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

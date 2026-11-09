"""Train one adapter per seed; smoke adapters are saved separately."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.config import adapter_dir, load_condition
from scifact_attr.reader.train_lora import build_sft_examples, train
from scifact_attr.retrieval.bm25 import BM25Index
from scifact_attr.selection.selector import OverlapSelector, select
from scifact_attr.utils.io import get_logger, load_jsonl_as_list, set_seed, write_jsonl

LOG = get_logger("train")


def training_evidence(claims, sentence_rows, config, neutralise):
    by_id = {s["sid"]: s for s in sentence_rows}
    by_doc = {}
    for s in sentence_rows:
        by_doc.setdefault(s["doc_id"], []).append(s)

    settings = config["selection"]
    limit = settings["top_sentences"]
    cap = settings["max_per_abstract"]
    if neutralise:
        index = BM25Index.load(config["paths"]["indexes"] / "bm25_scifact.pkl")
        selector = OverlapSelector()

    result = {}
    for claim in claims:
        candidates = [by_id[sid] for sid in claim["gold_sentence_ids"]]
        scores = [1.0] * len(candidates)
        if neutralise and claim["label"] == "NEI":
            excluded = set(claim["cited_doc_ids"])
            candidates = [
                sentence
                for doc_id, _ in index.search(claim["claim"], top_k=8)
                if int(doc_id) not in excluded
                for sentence in by_doc.get(int(doc_id), [])[:12]
            ]
            scores = selector.score(claim["claim"], [s["text"] for s in candidates])
            if not candidates:
                raise ValueError(f"No control evidence for claim {claim['claim_id']}")
        result[claim["claim_id"]] = select(scores, candidates, top_n=limit, max_per_doc=cap)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default="c3", choices=("c3", "c3_prime"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume the last epoch checkpoint with the same settings")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--rank", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--precision", choices=("qlora", "bf16_lora"), default="qlora")
    parser.add_argument("--epochs", type=float, help="override training epochs for a separate pilot")
    args = parser.parse_args()
    if args.resume and args.smoke:
        parser.error("smoke runs do not save checkpoints")

    cfg = load_condition(args.condition)
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("--seeds must contain distinct nonnegative seeds")
    cfg["reader"]["lora"]["r"] = args.rank
    cfg["reader"]["quant"]["load_in_4bit"] = args.precision == "qlora"
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error("--epochs must be positive")
        cfg["reader"]["train"]["epochs"] = args.epochs
    if args.rank != 16 or args.precision != "qlora" or args.epochs is not None:
        base = Path(cfg["adapter_path"])
        pilot = f"_e{args.epochs:g}" if args.epochs is not None else ""
        cfg["adapter_path"] = str(base.with_name(f"{base.name}_r{args.rank}_{args.precision}{pilot}"))

    processed = cfg["paths"]["processed"]
    claims = load_jsonl_as_list(processed / "claims_train.jsonl")
    sentences = load_jsonl_as_list(processed / "sentences.jsonl")
    evidence = training_evidence(claims, sentences, cfg, args.condition == "c3_prime")
    examples = build_sft_examples(claims, evidence)
    example_path = processed / f"sft_{args.condition}.jsonl"
    write_jsonl(example_path, examples)
    LOG.info("%d training examples (%d with evidence)", len(examples),
             sum(example["has_evidence"] for example in examples))

    for seed in seeds:
        effective_seed = int(cfg["seed"]) + seed
        set_seed(effective_seed)
        path = adapter_dir(cfg, seed, smoke=args.smoke)
        LOG.info("training seed %d -> %s", seed, path)
        train(cfg, example_path, path, seed=effective_seed, smoke_test=args.smoke, resume=args.resume)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

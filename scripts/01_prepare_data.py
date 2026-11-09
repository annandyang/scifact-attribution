"""Acquire, verify and prepare the official SciFact data."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.config import load_base, REPO_ROOT
from scifact_attr.corpus import build_sentence_table  # noqa: E402
from scifact_attr.data import OFFICIAL_URL, acquire_official, load_all
from scifact_attr.utils.io import get_logger, write_jsonl  # noqa: E402

LOG = get_logger("prepare")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="all", choices=["all", "dev"],
                        help="kept for compatibility; prepares all three splits")
    parser.add_argument("--archive", type=Path, help="path to the official SciFact data.tar.gz")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--with-open", action="store_true", help="show the SciFact-Open command")
    args = parser.parse_args()

    cfg = load_base(args.config)
    raw = REPO_ROOT / cfg["paths"]["raw"]
    processed = REPO_ROOT / cfg["paths"]["processed"]

    needed = ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]
    present = all((raw / f).exists() for f in needed)

    try:
        if args.archive or args.force or not present:
            LOG.info("acquiring the official SciFact release")
            acquire_official(raw, args.archive)
        corpus, claims, report = load_all(raw)
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise SystemExit(f"{exc}\nDownload {OFFICIAL_URL} and pass --archive <file>.") from None

    print("\nloaded")
    print(f"  abstracts           {report['corpus']}")
    for split in ("train", "dev", "test"):
        print(f"  claims_{split:<10} {report['claims'].get(split, 0)}")
        print("    " + ", ".join(f"{label}: {count}"
                               for label, count in report["label_counts"][split].items()))

    print("\nstructure")
    print(f"  dev gold docs missing from corpus : {report['dev_gold_docs_missing_from_corpus']}")
    print(f"  abstracts without title           : {report['abstracts_without_title']}")
    print(f"  abstracts without sentences       : {report['abstracts_without_sentences']}")
    print(f"  NEI training claims               : {report['nei_train_total']}")
    print(f"  ... with empty evidence field     : {report['nei_train_without_evidence']}"
          f"  ({report['nei_train_without_evidence'] / max(report['nei_train_total'], 1):.1%})")
    sentences, _ = build_sentence_table(corpus, dedupe=False)
    print(f"\nsentences              {len(sentences)}")

    processed.mkdir(parents=True, exist_ok=True)
    write_jsonl(processed / "sentences.jsonl", sentences)
    for split in ("train", "dev", "test"):
        write_jsonl(processed / f"claims_{split}.jsonl", claims[split])
    write_jsonl(processed / "corpus.jsonl", corpus)
    with open(processed / "verify_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    LOG.info("wrote %s", processed)
    if args.with_open:
        print("SciFact-Open: python scripts/11_scifact_open.py prepare --archive scifact_open.tar.gz")

    print("\nnext: python scripts/02_build_index.py --retriever bm25")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SciFact Attribution

A MIAZ01 course project measuring what retrieval, sentence selection, adapter training,
reasoning and abstention contribute to scientific claim verification.

See [Windows setup and demo](docs/QUICKSTART.zh.md) for commands and
[experiment protocol](docs/REPRODUCE.md) for splits, metrics and run order.
See [local run results](docs/RESULTS.zh.md) for the original run and follow-up measurements.
See [the sentence selection audit](docs/SELECTION_STUDY.zh.md) for the C1–C2 comparison and the optional empty-evidence rule.
See [the English results draft](docs/REPORT_DRAFT.md) for the experiment rationale and measured results.
See [project status in Chinese](docs/STATUS.zh.md) for the completed and pending work.
See [course concept mapping in Chinese](docs/COURSE_MAP.zh.md) for the methods behind each experiment.
The browser demo reads the checked-in prediction cache and does not load a model.
Run the dataset setup once to make the evidence sentences available. The separate
full run ZIP also contains the prepared data and trained adapters.

| Condition | Change |
|---|---|
| C0 | Qwen2.5-1.5B sees the claim only |
| C1 | Add retrieved abstracts |
| C2 | Use selected sentences from those abstracts |
| C3 | Attach a seed-specific QLoRA adapter |
| C4 | Add a cited evidence assessment before the same adapted label scorer |
| C5 | Apply a frozen conformal gate to the C4 output |
| C3-prime | Retrain C3 with retrieved distractors for NEI training claims |

The official test release has no public labels. Label metrics use a fixed held-out half of dev.
The other half calibrates C5. Test inference writes predictions only.
Smoke caches and adapters have separate paths.

The project uses the [official SciFact release](https://github.com/allenai/scifact),
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
[BGE small](https://huggingface.co/BAAI/bge-small-en-v1.5), and the
[BGE reranker](https://huggingface.co/BAAI/bge-reranker-base).
Revisions are pinned in configs/base.yaml. Dataset and model files retain their own terms.

Raw corpora, rebuilt indexes and model weights are stored outside Git. Run
`python scripts/01_prepare_data.py` to download and prepare the official SciFact release.
The script also accepts `--archive` with a local copy. The archive checksum and split
are recorded in [the experiment protocol](docs/REPRODUCE.md). Prediction records,
tables and plots from the measured run are checked in under `outputs/`.

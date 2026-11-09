# Experiment protocol

1. **Data.** Run python scripts/01_prepare_data.py. The official archive has 5,183 abstracts and 809 / 300 / 300 train/dev/test claims. Preserve original sentence positions, all gold documents and alternative evidence sets. Test labels are unreleased; empty train/dev evidence is NEI.
2. **E1.** Build BM25 and dense indexes with script 02, then run python scripts/03_retrieval_study.py --with-rerank. Retrieval metrics use the 188 dev claims with annotated evidence. The archived BEIR numbers use a different protocol. Hybrid fuses 100 candidates per retriever. C1–C4 use the configured top eight hybrid abstracts.
3. **E2.** Run C0–C4 with script 04. C3/C4 use matching adapters from script 05. Only answer tokens contribute to training loss. For C4, base Qwen writes a short evidence assessment with the adapter disabled, then the same adapted label scorer as C3 scores the claim. The label prompt includes the note only when it cites an available sentence or explicitly identifies a lack of relevant evidence. The run records raw assessment text, valid citation IDs, and uncited outputs. The initial direct prompt pilot is retained in outputs/pilots/direct_cot.
4. **E5.** Run script 06 after C4. For each seed, 150 dev IDs calibrate 1-p(gold); the other 150 measure results. C5 answers when its calibrated label set is a singleton that agrees with a valid C4 output. Marginal label-set coverage under exchangeability and observed error on answered claims are different quantities. Script 08 uses the same held-out IDs for every rung. Smoke caches are excluded.
5. **E3, E4, E7, E8.** C3-prime retains the training labels and inserts BM25 distractors for NEI. Script 07 reports matched held-out comparisons. Script 09 compares cited and uncited interventions on the same seed and claim. The Windows ablation stage compares ranks 8, 16 and 32 at the same precision. Script 10 exports evidence and blank human-review fields.
6. **Test.** Generate predictions after freezing prompts, revisions and calibration. Script 06 reuses dev thresholds for test. The official test labels are not available locally. For E6, use the separate SciFact-Open corpus-shift protocol.

The archive used for the local run has SHA-256
11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be.
Keep exact package versions, condition YAML, adapter metadata and per-claim JSONL.
A confidence interval that overlaps zero does not resolve the direction of an adjacent-rung
difference. A negative change is a valid result. Evidence overlap metrics use the union of
annotated rationale sentences. Pooled SciFact-Open highlights are not gold sentence annotations.

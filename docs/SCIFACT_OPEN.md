# E6: SciFact-Open transfer

Use the same C0–C4 prompts, retrieval mode and adapter seeds as the small-corpus run.
Complete the C4 development run and freeze the C5 thresholds with script 06 first.
Then download the [official SciFact-Open archive](https://github.com/dwadden/scifact-open#data).
The full corpus contains 500,000 documents. Its BM25 index and IVF-PQ build take
substantially more disk, RAM and time than the 5,183-document experiments.

Run these commands from the project root with the same $py and model cache as the dev run.
This example uses seed 0.

```powershell
& $py -m pip install faiss-cpu
& $py scripts/11_scifact_open.py prepare --archive path/to/scifact_open.tar.gz
& $py scripts/02_build_index.py --corpus open --retriever all --device cuda
& $py scripts/11_scifact_open.py index-check --device cuda

foreach ($condition in 'c0','c1','c2','c3','c4') {
    & $py scripts/04_run_condition.py --condition $condition --split test --seeds 0
    if ($LASTEXITCODE -ne 0) { throw "Test inference failed for $condition." }
    & $py scripts/04_run_condition.py --condition $condition --split open --seeds 0
    if ($LASTEXITCODE -ne 0) { throw "Open inference failed for $condition." }
}
& $py scripts/11_scifact_open.py score
```

If you ran the small corpus with an explicit retrieval mode or seed list, pass the
same values to both runs. Script 11 applies the saved per-seed C5 thresholds directly
and writes the comparison to `outputs/tables/e6_transfer_full.csv`.
The packaged seed-0 test predictions can be reused if the prompts, adapter and retrieval
settings stay frozen.

The report compares the same 279 claims using SciFact-Open's pooled annotations for both
corpora. Claims with both annotated support and refutation are excluded from label metrics
and included in retrieval metrics. Pooled model-selected highlights are excluded from gold
citation evaluation. Hit rate measures recovery of any annotated evidence document.
Accuracy given a hit measures verdict correctness among labeled claims with a hit.
Set coverage under corpus shift is empirical.

For a smaller exploratory run, add `--scope candidates` to the two script 11
commands for preparation and scoring, and `--open-scope candidates` to scripts 02
and 04 when using Open. This uses the released candidate pool and exact dense search.
Skip the IVF-PQ overlap check in this mode. The scope is recorded separately in the
index, prediction and result paths.

On a machine with limited RAM, build only the full dense index with
`scripts/02_build_index.py --corpus open --retriever dense --device cuda`.
Use script 15 to precompute exact BM25 results for the 279 fixed queries instead of
building a general-purpose Python BM25 index:

```powershell
$open = 'data/processed/open/full'
& $py scripts/15_open_bm25_check.py --scope full --claims "$open/claims_open.jsonl" `
    --corpus "$open/corpus.jsonl" --out "$open/bm25_check.json" `
    --ranking-depth 100 --rankings-dir "$open/rankings"
```

For each full Open hybrid inference run with retrieval enabled, pass
`--frozen-bm25 "$open/rankings/bm25_full.jsonl"` to script 04. The ranking depth must
match the hybrid candidate depth in the config. C0 does not use this argument.
The frozen ranking file contains only these benchmark queries and cannot be used for new queries.

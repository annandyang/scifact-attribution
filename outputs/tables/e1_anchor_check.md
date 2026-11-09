# E1 retrieval study

SciFact dev: 300 claims, 188 with annotated evidence documents.
Recall@k is the fraction of evidence-bearing claims with at least one gold document
in the top k. Full recall@k averages the fraction of gold documents retrieved per claim.
nDCG uses binary document relevance.

Latency uses 20 seeded, single-claim queries per cutoff after warmup,
including query encoding and ranking. See the CSV for median and p95 in milliseconds.

## Published context

[Chen, Sun & Brown (2023)](https://doi.org/10.69987/JACS.2023.30102) report BEIR
SciFact test qrels (300 queries and 339 relevant pairs), BM25 k1=1.2/b=0.75,
a SciFact-trained mean-embedding encoder and a logistic regression reranker.
This study uses official SciFact annotations, BM25 k1=0.9/b=0.4 by default,
a pretrained BGE encoder and a pretrained BGE cross-encoder. These protocol
differences prevent interpreting the gaps as reproduction errors.

| configuration | metric | measured here | published |
|---|---|---:|---:|
| bm25 | full_recall@100 | 0.953 | 0.883 |
| bm25 | ndcg@10 | 0.815 | 0.662 |
| dense | ndcg@10 | 0.860 | 0.537 |
| hybrid_rrf | full_recall@100 | 0.995 | 0.923 |
| hybrid_rrf+rerank | ndcg@10 | 0.858 | 0.659 |

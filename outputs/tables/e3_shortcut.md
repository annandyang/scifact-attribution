# E3: Empty-evidence shortcut audit

Prediction comparison: held-out dev, seed 0.

- NEI with empty gold evidence: 304/304.
- SUPPORT/REFUTE with empty gold evidence: 0/505.

| Top retrieval score range | n | NEI rate |
|---|---:|---:|
| 0.02229 to 0.03227 | 32 | 0.312 |
| 0.03227 to 0.03279 | 35 | 0.314 |
| 0.03279 to 0.03279 | 83 | 0.060 |

Rate span: 0.254. This association does not establish that the reader used the evidence.

150 matched claims.

| Metric | C3 | C3-prime | C3 minus C3-prime |
|---|---:|---:|---:|
| macro_f1 | 0.628 | 0.445 | +0.183 |
| nei_f1 | 0.565 | 0.371 | +0.194 |
| support_f1 | 0.706 | 0.524 | +0.182 |
| refute_f1 | 0.613 | 0.440 | +0.173 |

C3-prime receives BM25 distractors for NEI training rows. Their irrelevance is an assumption, not a human judgment.

# E2 evidence selection audit

This audit pairs seed 0 C1 and C2 predictions for the same SciFact dev claims. It checks that retrieved documents and scores match and that selected sentence IDs, selection budgets, and label probabilities are consistent.

C1 supplies abstracts capped at 12 sentences and 256 words each. A truncated sentence counts only if its full text reaches the prompt. C2 supplies up to 4 sentences with up to 2 from one abstract. A complete alternative set requires every sentence from at least one annotated rationale.

## Calibration half

| Metric | Count |
|---|---:|
| C1 correct | 74/150 |
| C2 correct | 79/150 |
| Correct in C1 only | 15 |
| Correct in C2 only | 20 |
| Gold document retrieved | 92/97 |
| C1 contains a complete gold sentence | 81/97 |
| C2 contains a complete gold sentence | 45/97 |
| C1 contains at least one complete alternative evidence set | 80/97 |
| C2 contains at least one complete alternative evidence set | 40/97 |
| Gold document retrieved but no gold sentence selected | 47 |
| Correct in C1 only, with gold evidence annotated | 12 |
| Of those, C2 contains no gold sentence | 8 |
| Of those, C1 has a complete alternative set and C2 contains no gold sentence | 7 |
| Of those, C2 contains at least one complete alternative set | 4 |
| Of those, C2 predicts NEI | 12 |
| Empty C2 context | 24/150 |
| Empty C2 context with a correct verdict | 9/24 |

Empty C2 context predictions were 13 SUPPORT, 0 REFUTE, 11 NEI.

| Label | Gold count | C1 predicted | C2 predicted | C1 correct | C2 correct | C1 only | C2 only |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUPPORT | 63 | 110 | 84 | 56 | 48 | 12 | 4 |
| REFUTE | 34 | 0 | 0 | 0 | 0 | 0 | 0 |
| NEI | 53 | 40 | 66 | 18 | 31 | 3 | 16 |

## Held-out evaluation half

| Metric | Count |
|---|---:|
| C1 correct | 84/150 |
| C2 correct | 71/150 |
| Correct in C1 only | 24 |
| Correct in C2 only | 11 |
| Gold document retrieved | 87/91 |
| C1 contains a complete gold sentence | 81/91 |
| C2 contains a complete gold sentence | 49/91 |
| C1 contains at least one complete alternative evidence set | 81/91 |
| C2 contains at least one complete alternative evidence set | 48/91 |
| Gold document retrieved but no gold sentence selected | 38 |
| Correct in C1 only, with gold evidence annotated | 18 |
| Of those, C2 contains no gold sentence | 11 |
| Of those, C1 has a complete alternative set and C2 contains no gold sentence | 9 |
| Of those, C2 contains at least one complete alternative set | 7 |
| Of those, C2 predicts NEI | 18 |
| Empty C2 context | 26/150 |
| Empty C2 context with a correct verdict | 7/26 |

Empty C2 context predictions were 19 SUPPORT, 0 REFUTE, 7 NEI.

| Label | Gold count | C1 predicted | C2 predicted | C1 correct | C2 correct | C1 only | C2 only |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUPPORT | 61 | 108 | 83 | 57 | 44 | 15 | 2 |
| REFUTE | 30 | 3 | 0 | 3 | 0 | 3 | 0 |
| NEI | 59 | 39 | 67 | 24 | 27 | 6 | 9 |

Reproduce after running C1 and C2 on dev with `python scripts/13_audit_selection.py`.

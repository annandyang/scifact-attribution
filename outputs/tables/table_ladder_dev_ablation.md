# Ladder results on held-out SciFact dev

Shared evaluation claims: 150 / 150.
Official test predictions are unlabeled. Each row reports one run seed.
Accuracy and F1 count abstentions and invalid outputs as missed labels. Selective accuracy uses valid answers.
ECE uses valid answered rows with finite probabilities. Evidence metrics use annotated claims and the union of gold rationale sentences.
Retrieval hit@8 is the fraction of annotated claims with any gold abstract in the top 8.
C5 citations are inherited from C4 and remain visible when the gate abstains.
Use E5 for the effect of C5 on answer rate and error among answers.

| condition | seed | accuracy | 95% CI | macro-F1 | NEI F1 | answer rate | selective accuracy | ECE | hit@8 | selected evidence recall | citation F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rank8 | 0 | 0.653 | [0.573, 0.727] | 0.633 | 0.565 | 1.000 | 0.653 | 0.332 | 0.956 | 0.411 | n/a |
| c3 | 0 | 0.647 | [0.567, 0.720] | 0.628 | 0.565 | 1.000 | 0.647 | 0.338 | 0.956 | 0.411 | n/a |
| rank32 | 0 | 0.660 | [0.587, 0.733] | 0.645 | 0.565 | 1.000 | 0.660 | 0.320 | 0.956 | 0.411 | n/a |
| bf16_r8 | 0 | 0.673 | [0.600, 0.747] | 0.656 | 0.565 | 1.000 | 0.673 | 0.300 | 0.956 | 0.411 | n/a |
| bf16_r16 | 0 | 0.660 | [0.587, 0.733] | 0.638 | 0.565 | 1.000 | 0.660 | 0.308 | 0.956 | 0.411 | n/a |
| bf16_r32 | 0 | 0.667 | [0.593, 0.740] | 0.646 | 0.565 | 1.000 | 0.667 | 0.319 | 0.956 | 0.411 | n/a |

## Paired accuracy differences

Paired bootstrap intervals resample claims. McNemar p-values are unadjusted.

| comparison | baseline seed | seed | delta | 95% CI | McNemar p | verdict |
|---|---|---|---|---|---|---|
| rank8 - c3 | 0 | 0 | +0.007 | [+0.000, +0.020] | 1.000 | inconclusive |
| rank32 - c3 | 0 | 0 | +0.013 | [+0.000, +0.033] | 0.500 | inconclusive |
| bf16_r16 - c3 | 0 | 0 | +0.013 | [+0.000, +0.033] | 0.500 | inconclusive |
| bf16_r8 - rank8 | 0 | 0 | +0.020 | [+0.000, +0.047] | 0.250 | inconclusive |
| bf16_r32 - rank32 | 0 | 0 | +0.007 | [-0.013, +0.033] | 1.000 | inconclusive |
| bf16_r8 - bf16_r16 | 0 | 0 | +0.013 | [-0.013, +0.040] | 0.625 | inconclusive |
| bf16_r32 - bf16_r16 | 0 | 0 | +0.007 | [+0.000, +0.020] | 1.000 | inconclusive |

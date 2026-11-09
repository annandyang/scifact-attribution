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
| c0 | 0 | 0.420 | [0.340, 0.500] | 0.291 | 0.308 | 1.000 | 0.420 | 0.403 | n/a | n/a | n/a |
| c1 | 0 | 0.560 | [0.480, 0.640] | 0.449 | 0.490 | 1.000 | 0.560 | 0.198 | 0.956 | n/a | n/a |
| c2 | 0 | 0.473 | [0.393, 0.553] | 0.347 | 0.429 | 1.000 | 0.473 | 0.319 | 0.956 | 0.411 | n/a |
| c3 | 0 | 0.647 | [0.567, 0.720] | 0.628 | 0.565 | 1.000 | 0.647 | 0.338 | 0.956 | 0.411 | n/a |
| c4 | 0 | 0.567 | [0.487, 0.647] | 0.522 | 0.351 | 1.000 | 0.567 | 0.364 | 0.956 | 0.411 | 0.327 |
| c5 | 0 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.956 | 0.411 | 0.327 |
| c3_prime | 0 | 0.447 | [0.367, 0.527] | 0.445 | 0.371 | 1.000 | 0.447 | 0.480 | 0.956 | 0.411 | n/a |

## Paired accuracy differences

Paired bootstrap intervals resample claims. McNemar p-values are unadjusted.

| comparison | baseline seed | seed | delta | 95% CI | McNemar p | verdict |
|---|---|---|---|---|---|---|
| c1 - c0 | 0 | 0 | +0.140 | [+0.073, +0.213] | 0.000 | improvement |
| c2 - c1 | 0 | 0 | -0.087 | [-0.167, -0.013] | 0.041 | regression |
| c3 - c2 | 0 | 0 | +0.173 | [+0.067, +0.280] | 0.003 | improvement |
| c4 - c3 | 0 | 0 | -0.080 | [-0.133, -0.033] | 0.004 | regression |
| c3_prime - c3 | 0 | 0 | -0.200 | [-0.313, -0.080] | 0.001 | regression |

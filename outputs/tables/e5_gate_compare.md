# E5 conformal gate on held-out SciFact dev

Calibration and evaluation use disjoint halves of the official dev split.
The label set has marginal coverage at least 1 minus alpha under exchangeability.
The answer rate and error among answered claims are empirical.

| seed | rule | calibration n | evaluation n | nominal set coverage | empirical set coverage | answer rate | selective error |
|---|---|---|---|---|---|---|---|
| 0 | conformal, alpha=0.10 | 150 | 150 | 0.900 | 0.887 | 0.000 | n/a |
| 0 | conformal, alpha=0.20 | 150 | 150 | 0.800 | 0.780 | 0.067 | 0.000 |
| 0 | conformal, alpha=0.30 | 150 | 150 | 0.700 | 0.687 | 0.620 | 0.366 |
| 0 | fixed confidence >= 0.70 | - | 150 | - | - | 0.920 | 0.413 |
| 0 | confidence sweep AURC 0.296 | - | 150 | - | - | - | - |

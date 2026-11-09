# E8 error review on held-out SciFact dev

C4 seed 0, 150 of 150 held-out claims.
Matched C5 predictions available for 150 reviewed claims.

| Observable flag | Flagged claims |
|---|---|
| Annotated evidence document absent from retrieval | 4 |
| Gold document retrieved, no annotated sentence selected | 38 |
| Wrong polarity with an annotated sentence selected | 3 |
| NEI emitted for an annotated SUPPORT or REFUTE claim | 2 |
| C5 withheld an otherwise correct C4 label | 85 |
| No valid C4 label | 0 |

Flags may overlap. Citation and evidence flags use the released annotations.
Review the actual sentences before coding causes. A model can emit NEI for several reasons.
Use the manual category field to record retrieval miss, selection error, contradiction miss, shortcut-driven NEI, over-refusal, reasoning drift, other, or no error.
The second-coder sample marks 60 claims. Record that coding independently before comparing agreement.

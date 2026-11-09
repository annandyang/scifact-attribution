# What improves scientific claim verification on SciFact?

This results draft separates experiments we ran from the remaining work. The linked tables are the source of the reported numbers.

## Research question and setup

We ask which parts of a small retrieval-augmented system improve scientific claim verification: retrieving abstracts, selecting sentences, adapting the reader, asking for a cited assessment, or letting the system abstain. The verifier predicts SUPPORT, REFUTE, or NOT_ENOUGH_INFO for each claim. We use [SciFact](https://github.com/allenai/scifact), Qwen2.5-1.5B-Instruct, BM25 and BGE retrieval, and a BGE cross-encoder for sentence ranking. A 4-bit QLoRA adapter is trained on 809 labeled claims.

The official development set has 300 claims. A fixed split of the claim IDs assigns 150 to calibration and 150 to evaluation. Classification scores below come from the evaluation half. We generated 1,800 predictions across the six conditions for the official test claims. The public test file has no labels, so those predictions are not included in the reported accuracy.

## Component results

| Condition | Change | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| C0 | Claim only | 42.0% | 0.291 |
| C1 | Read excerpts from retrieved abstracts | 56.0% | 0.449 |
| C2 | Read up to four selected sentences | 47.3% | 0.347 |
| C3 | Add the rank-16 QLoRA adapter, seed 0 | 64.7% | 0.628 |
| C4 | Use a cited assessment before scoring the label, seed 0 | 56.7% | 0.522 |

On the same evaluation claims, C1 improved on C0 by 14.0 percentage points. C2 then dropped by 8.7 points (paired bootstrap 95% interval: −16.7 to −1.3). C3 improved on C2 by 17.3 points (interval: +6.7 to +28.0). C4 provided traceable citations but scored lower than C3. [Main results and paired comparisons](../outputs/tables/table_ladder_dev.md)

Three independently trained C3 adapters with the same settings gave 97, 98, and 97 correct predictions out of 150. Their macro-F1 scores were 0.628, 0.634, and 0.623. Averaging the three adapters' label probabilities gave 97 correct predictions and did not improve accuracy over the original seed. They shared 50 errors, including 35 on claims whose reference label was NOT_ENOUGH_INFO.

C4 accuracy was 56.7%, 61.3%, and 63.3% across the three seeds, below the corresponding C3 score in each case. [Multiseed results](../outputs/tables/e2_multiseed_summary.json)

## Why selection can hurt

C1 and C2 retrieved identical document rankings. Of 91 evaluation claims with annotated evidence, 87 had an evidence document in the top eight. C1 passed a complete annotated evidence set to the reader for 81 of those 91 claims; C2 did so for 48. C2 also answered some claims incorrectly with the complete evidence present. The two prompts differ in how they present titles and sentences, so the C1–C2 difference cannot be attributed to the number of sentences alone.

We then tested changes to the selector and to empty-context handling. Removing the selector score cutoff raised C2 accuracy to 56.7%. A separate cached replay that returned NOT_ENOUGH_INFO when C2 selected no evidence raised accuracy to 58.7%, with macro-F1 of 0.436, below C1's 0.449. For the adapted C3 reader, a wider selector increased complete evidence from 48 to 62 evaluation claims without changing the number of correct verdicts (97/150). These are follow-up diagnostics on a previously used development split. [Selection audit and replay](SELECTION_STUDY.zh.md)

## Shortcuts, citations, and abstention

All 304 NOT_ENOUGH_INFO training claims had empty annotated evidence, while none of the 505 SUPPORT and REFUTE claims did. The C3-prime control gave NOT_ENOUGH_INFO examples retrieved distractors. Its macro-F1 fell from 0.628 to 0.445, showing sensitivity to the training input pattern. Some distractors may still be relevant, so the full difference cannot be assigned to a single shortcut. [Shortcut audit](../outputs/tables/e3_shortcut.md)

Replacing cited sentences changed 20 of 80 verdicts. Replacing the same number of uncited sentences with the same distractors changed 9 of 80. Scoring the label before generating the assessment gave 64.7% accuracy, compared with 56.7% when the assessment came first. A citation can help account for a verdict, but these tests do not make every rationale faithful. [Evidence replacement table](../outputs/tables/e4_evidence_perturb_replace.csv)

C5 uses a conformal label set calibrated on separate claims and answers only when the set has one label. For seed 0, at alpha 0.10 it answered none of 150 evaluation claims; at 0.20 it answered 10 and all 10 were correct; at 0.30 it answered 93 with 34 errors. At alpha 0.10 the other seeds answered 8 and 0 claims. The conformal guarantee concerns label-set coverage under exchangeability. These answer rates and errors are empirical. [Abstention results by seed](../outputs/tables/e5_conformal_multiseed.csv)

## Transfer and limitations

Using the same 206 evidence-bearing [SciFact-Open](https://aclanthology.org/2022.findings-emnlp.347/) claims, BM25 found at least one annotated document in the top eight for 176 claims in the original 5,183-document corpus and 152 in the full 500,000-document corpus. This isolates the retrieval stage. The pooled relevance annotations are not exhaustive, and the two corpora also differ in composition. [Transfer retrieval report](../outputs/tables/e6_open_retrieval.zh.md)

The comparison separates aggregate classification accuracy from evidence retention and abstention. More evidence did not always yield more correct verdicts, and an explanation did not always improve the label. The final course report should give the per-seed measurements, identify which transfer and adapter ablations were completed, and keep AI-assisted error examples separate from independent human coding. [Current project status](STATUS.zh.md)

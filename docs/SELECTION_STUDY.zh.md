# 选句为什么有时会降低准确率

这里把“只保留选中的数据”理解为 C2：两组检索到同样的 8 篇摘要，C1 读带标题的摘要前部，C2 改读最多 4 条选中的句子。C2 的输入更短，也可能失去关键句、研究对象或前后文。

## 检查结果

逐条检查 300 条开发声明，C1/C2 的论文编号、顺序和检索分数一致；C2 的句子编号都能对回语料，并遵守每篇最多 2 句的限制。校准半组 C2 比 C1 多对 5 题，评估半组少对 13 题。评估半组中，91 题有标注证据，C1 在 81 题的实际输入中保留至少一组完整证据，C2 只有 48 题。C2 在整个开发集没有输出过 REFUTE。详见[配对审计](../outputs/tables/e2_selection_audit.md)和[逐题例子](SELECTION_CASES.zh.md)。

## 空证据规则

用原始 C2 的预测回放：选出 0 句时直接返回 NEI（所提供的证据不足），否则保留阅读模型的输出。复算不重新检索、不调用模型，也不改动原始 C2。空证据触发的判定是规则，输出里的 1.0 是规则权重，不能当作模型置信度。

| 固定比较 | 校准组正确 / 150 | 评估组正确 / 150 | 评估组宏平均 F1 |
|---|---:|---:|---:|
| C1，读摘要 | 74 | 84 | 0.449 |
| C2，读选句 | 79 | 71 | 0.347 |
| C2，空证据直接判 NEI | 86 | 88 | 0.436 |

评估组有 26 条空证据；规则改变了 19 条预测，纠正 18 条、改错 1 条，净增 17 条（+11.3 个百分点，配对 95% 区间 +6.0 到 +17.3）。规则后的准确率为 58.7%，宏平均 F1 仍低于 C1，也没有补回丢掉的证据或提升 REFUTE 识别。这是看到原始结果后做的诊断，不能视为新的独立测试集成绩。

可用缓存或真实推理复跑。以下命令在已经准备好 SciFact 数据的仓库根目录执行：

```powershell
python scripts/13_audit_selection.py
python scripts/14_selection_study.py --empty-rule-only --out outputs/tables/e2_empty_rule
python scripts/04_run_condition.py --condition c2 --split dev --require-evidence --out outputs/predictions/c2_guard_dev.jsonl
```

## 选句参数对照

用同一批检索结果和同一个阅读模型，比较原规则、取消最低相关性分数（仍最多 4 句）、放宽到 8 句（每篇最多 4 句，并考虑摘要第 12 句以后的候选）和补邻句（保留原选句，再加同一摘要紧邻句，总计最多 8 句）。先在 150 条校准声明上比较宏平均 F1，然后只在另外 150 条上跑选中的方案和原规则。C2 与 C3 分开比较，避免把微调与选句的作用混在一起。

```powershell
python scripts/14_selection_study.py --condition c2 --out outputs/tables/e2_selection_c2
python scripts/14_selection_study.py --condition c3 --out outputs/tables/e2_selection_c3
```

`wide8` 同时改了候选范围和两种句数上限，因此只能比较整套策略。仅凭它的结果不能判断是哪一个参数起了作用。

## 英文报告表述

On the held-out SciFact development subset, replacing retrieved abstract excerpts with up to four selected sentences reduced accuracy from 56.0% to 47.3%. Document rankings were identical. Among the 91 claims with annotated evidence, the abstract condition presented at least one complete annotated evidence set for 81 claims; sentence selection reduced this number to 48. Some predictions remained incorrect even when a complete set was preserved, so the difference also reflects how the reader uses the available context.

The base reader sometimes predicted SUPPORT when no sentence passed the selector threshold. In a post-hoc replay, requiring an empty context to return NOT_ENOUGH_INFO raised held-out accuracy to 58.7%. This rule neither recovers missing evidence nor improves REFUTE detection. The guarded system still had lower macro-F1 than the abstract condition (0.436 vs. 0.449).

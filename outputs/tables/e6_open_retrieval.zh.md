# SciFact-Open 检索实验

用 SciFact-Open 的官方标注，在同一批 206 条有证据的声明上对比三种语料库。BM25 的参数固定为 `k1=0.9` 和 `b=0.4`，每条声明检索前 8 篇摘要。

| 检索语料 | 摘要数 | 前 8 篇至少命中一篇证据 | 平均证据文档召回率 @8 |
| --- | ---: | ---: | ---: |
| 原始 SciFact 小库 | 5,183 | 176 / 206（85.44%） | 69.70% |
| SciFact-Open 官方候选库 | 12,236 | 161 / 206（78.16%） | 63.62% |
| SciFact-Open 全库 | 500,000 | 152 / 206（73.79%） | 60.80% |

小库和全库均命中 140 条、均未命中 18 条。另有 36 条仅小库命中、12 条仅全库命中。用 `scifact_attr.eval.bootstrap.paired_delta_ci` 对每条声明的命中差值进行 10,000 次有放回配对重抽样，随机种子为 0。全库减小库的命中率差为 −11.65 个百分点，重抽样差值第 2.5 和第 97.5 百分位给出的 95% 置信区间为 −17.96 至 −5.34 个百分点。

命中率统计有多少声明找到了至少一篇标注证据。召回率先计算每条声明找回的标注证据比例，再对声明求平均。候选库来自发布方筛选出的候选文档，与小库重叠 1,198 篇，语料组成也有变化。这里测量的是检索表现，端到端的 C0–C5 迁移实验需要另行评估判定与引用。 [SciFact-Open 数据说明](https://github.com/dwadden/scifact-open/blob/main/doc/data.md)

## 数据和运行记录

- 官方下载归档为 287,369,773 字节，SHA-256 为 `8f4dc238b2eff422bab5c7918fcbddb6996b29d37cb887942ef6ee1a16fb3415`。完整读取通过 gzip 校验。 [官方数据下载脚本](https://github.com/dwadden/scifact-open/blob/main/script/get_data.sh)
- `scripts/11_scifact_open.py prepare` 的 `candidates` 和 `full` 模式均处理了 279 条与原始测试集 ID 和文本匹配的声明。两种模式生成的声明文件 SHA-256 相同，为 `827c336f2a1b78465204102366988061186e3c94c4aac41e2c637b1de3ff133f`。其中 101 条支持、90 条反驳、73 条没有标注证据、15 条有相互冲突的证据。冲突声明计入检索实验。
- 候选库的 BM25 索引由 `scripts/02_build_index.py --corpus open --open-scope candidates --retriever all --device cpu` 构建，耗时 2.3 秒。索引 `bm25_open.pkl` 的 SHA-256 为 `b54f1c9f3953f178fc6456860c143229d9c53e36253a37bbbfc4d96606a99c57`。
- 全库检索诊断只为这批固定查询的词项把倒排列表写入磁盘，使用全库的文档频次和平均文档长度，沿用标准索引的 BM25 公式。在候选库上核对 206 条查询、小库上核对 279 条查询，完整分数向量均逐项一致。全库默认诊断用时 74 秒，峰值进程内存为 106 MiB，临时倒排文件约 167 MB。另一次为全部 279 条声明生成排序，用时 87 秒，复现了相同的命中率和召回率。

## 复跑

用 `scripts/11_scifact_open.py prepare` 准备候选库和全库后，在项目根目录运行以下命令。复跑小库时，用相同的 Open 声明文件，语料路径改为 `data/processed/corpus.jsonl`，并把 `--scope` 改为 `small`。

```powershell
$open = 'data/processed/open'
python scripts/15_open_bm25_check.py --scope full `
    --claims "$open/full/claims_open.jsonl" --corpus "$open/full/corpus.jsonl" `
    --out data/processed/bm25_full_check.json
```

用 `--reference-index` 指定同一语料库的标准 BM25 索引即可核对全部分数。用 `--rankings-dir` 可以把 279 条声明各自的排序保存到本地。该文件含有数据集文本，聚合 JSON 才是共享结果。

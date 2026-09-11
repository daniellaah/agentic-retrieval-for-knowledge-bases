# P0 当前基线：2026-09-10

**本轮已经建立可离线复算的当前代码基线，但没有证据支持直接更换检索器或模型。** 默认 4B Agent 的 v1 成功率仍为 77.50%；与历史同一总分相比，逐题有两项改善、两项退化。因此后续评估必须保留逐题差值和错误原因，不能只看总分。

## 实验与有效性

按 [P0 预注册协议](experiments/p0-protocol.json)，在代码 commit `4568acf` 的源文件冻结副本和独立 schema v2 快照上执行：40 个 v1 case × 4 种固定检索方法，以及默认 `qwen3.5:4b` 的 40 × 3 次新对话。Agent 开启 thinking、8 轮上限，保持生产 Runtime 决策路径。

- 40 篇原始 example notes、160 chunks；chunk 512 / overlap 64，embedding context 8192。
- embedding 为 `qwen3-embedding:0.6b`，1024 维；模型 digest、provider 版本、源文件哈希、锁文件和 reranker identity 随制品保存。
- baseline 返回 top-10 chunks，再按首次出现折叠 source；candidate 与 rerank candidate 深度均为 20。它不是补足到 10 个文档的 doc@10。
- 120 条 Agent、160 条 baseline 的矩阵齐全；逐条离线评分与保存结果一致。
- Qdrant 完整向量快照在前后均验证通过，语料、模型 digest 与快照指纹未漂移。
- baseline 无运行错误；Agent 的 6 次运行错误全部保留并计为失败，没有重试或删除。

完整原始记录、冻结源码、语料、SQLite 快照和校验和在约 2.19 MB 的 [P0 压缩制品](experiments/artifacts/p0-current-20260910-r2.tar.gz)，[制品 manifest](experiments/artifacts/p0-current-20260910-r2.tar.manifest.json) 保存 SHA-256。仓库内的 [机器摘要](experiments/p0-current-20260910-summary.json) 可直接检查分母、分组和失败明细。

第一次启动 `p0-current-20260910` 在索引构建后的快照读取处因返回值解包错误退出，尚未执行任何评测 trial。错误已修正，原失败目录保留；本报告只使用另一个新目录 `p0-current-20260910-r2` 的完整实验，不合并两次尝试。

## 当前测量

| 固定检索 | Source Recall from top-10 chunks | Source nDCG(exp) from top-10 chunks | 平均请求耗时 |
| --- | ---: | ---: | ---: |
| BM25 | 30.39% | 0.2872 | 0.22 ms |
| Semantic | 82.79% | 0.8464 | 79.29 ms |
| Hybrid | 84.02% | 0.8545 | 75.67 ms |
| Hybrid + Rerank | 87.30% | 0.8687 | 2,839.38 ms |

每组来源指标分母为 34 个 retrieval case；另 6 个 no-retrieval case 明确为不适用。延迟包含本次请求的 embedding / reranking，不含初始化；缓存与系统负载未作独立控制，不能据此下 warm/cold 性能结论。BM25 在这组以中文问题检索英文笔记的数据上较低，不代表所有 lexical 检索场景。

| 默认 4B Agent | 当前结果 | 分母 |
| --- | ---: | ---: |
| v1 source/tool 约束成功率 | 77.50%（93 成功 / 27 失败） | 120 trials / 40 cases |
| 来源覆盖率 | 86.72% | 102 retrieval trials |
| 最大轮数停止 | 10.00%（12 次） | 120 trials |
| 运行错误 | 5.00%（6 次） | 120 trials |
| 无需检索却调用检索 | 0.00% | 18 no-retrieval trials |
| 平均工具请求数 | 3.10 | 120 trials |
| 平均端到端耗时 | 13.59 s | 120 trials |

三次 trial 在每个 case 上的成功/失败结果一致，但这不证明未来确定性。120 次执行不能解释为 120 个独立信息需求。Agent 有多次工具调用、read 与最终生成，预算与单次 baseline 不同，不能把上两表直接当作公平的架构胜负。

| 任务 | 成功率 | trials |
| --- | ---: | ---: |
| exact | 83.33% | 18 |
| semantic | 100.00% | 18 |
| direct read | 100.00% | 18 |
| exploratory | 40.00% | 30 |
| knowledge QA | 66.67% | 18 |
| no retrieval | 100.00% | 18 |

这些仍是 **v1 来源、工具与停止约束**，不含答案正确性、事实完整性或引用支持判断。100% 的小切片也不能解释为生产可靠性保证。

## 总分掩盖的变化

对照 [历史 4B 结果](phase1-agent-model-ablation-results.md)，同为 93/120 成功，但 `semantic_005`、`explore_005` 从三次失败变为三次成功；`explore_010`、`qa_004` 从三次成功变为三次失败。其余 case 的二元结果相同。

历史与当前实验的代码/索引身份及服务环境并非单一变量控制，这里仅报告测量差异，不能归因为停止提醒、模型能力或某一个代码改动。

## 失败诊断与下一步实验

| 观测事实 | 下一步应隔离的因素 |
| --- | --- |
| `exact_001` 三次均只覆盖 2/5 个期望来源并正常结束 | occurrence limit 与 distinct-source 枚举、截断反馈、完整性核对；不能只提高语义检索 K |
| `explore_001/002` 来源不足且耗尽轮数；`explore_007` 来源不足但正常结束 | 首轮候选、后续 query 改写及 evidence/facet 增量；区分检索不到与过早停止 |
| `explore_004/010` 来源覆盖已为 1.0，仍耗尽轮数 | 在实际证据与输出 rubric 复核后做停止策略干预，不能直接把所有后续调用算浪费 |
| `explore_006` 三次 read 字符范围越界 | 工具参数、正文长度反馈与错误恢复；原始错误为 `Character range is outside the current document body.` |
| `qa_004` 三次 read 的 document_id 不是合法 64 位小写 SHA-256 | 工具身份参数可用性与校验错误恢复；来源虽已覆盖，执行仍失败 |
| `qa_006` 三次来源覆盖为 0.5 并结束 | 先复核必要事实/替代证据标签，再区分 candidate、rank、实际返回片段的缺失 |

当前最明确的工程诊断方向是 **工具参数错误、停止失败与 exact 枚举**；证据层 pilot 和独立复核用于判断这些失败如何对应真实任务质量。本轮保持被测算法不变，避免把修复后的结果混入基线。

## 离线复算

```sh
mkdir -p /tmp/arkb-p0-replay
tar -xzf evaluation/experiments/artifacts/p0-current-20260910-r2.tar.gz -C /tmp/arkb-p0-replay
.venv/bin/python evaluation/experiments/verify_bundle.py \
  /tmp/arkb-p0-replay/p0-current-20260910-r2
```

校验先检查制品哈希，再自动使用归档内冻结的 Python 源文件重算，不访问 Ollama、Qdrant 或实时 example_notes。新环境应先按归档锁文件准备依赖；复算证明计算与记录一致，不能替代 gold 审阅。

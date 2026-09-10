# Deterministic Retrieval Baselines

Agent Evaluation v1 的固定检索 control group，直接使用同一个
`evaluation/data/agent_v1.jsonl` 和 `expected_sources`。执行路径为：

`原始 query → RetrievalEngine.search → 原始 chunks → source 去重 → ranking_metrics → JSONL / summary / Markdown`

## 架构与修改范围

| 文件 | 职责 |
| --- | --- |
| `src/arkb/evaluation/models.py` | 增加 `BaselineEvalConfig`、`BaselineEvalResult`、`BaselineEvalRun` 和固定模式映射 |
| `src/arkb/evaluation/baselines.py` | 单次检索执行、source 归一化、调用已有 metrics、整体和 task_type 汇总、Markdown 报告 |
| `src/arkb/evaluation/agent_runner.py` | 现有 runner 增加 `run_baseline_evaluation` 和 `run_evaluation` 分发；复用数据解析、JSON 序列化、知识库/源码指纹和产物生命周期 |
| `tests/evaluation/test_baselines.py` | 40 项新增确定性检查，包含完整 40-case 本地集成 |
| `README.md`、`evaluation/README.md`、本文 | 入口说明、配置语义和比较边界 |

没有新增 retrieval algorithm。现有 Agent 执行和指标入口继续工作，产品 CLI
仍使用现有顶层命令；baseline 参数属于已有 evaluation 模块入口。

| Baseline | Engine mode | rerank |
| --- | --- | --- |
| `bm25` | `bm25` | `False` |
| `semantic` | `semantic` | `False` |
| `hybrid` | `hybrid` | `False` |
| `hybrid_rerank` | `hybrid` | `True` |

Runtime 在选定的 SQLite 快照上准备所需能力，随后复用一个 RetrievalEngine。
Hybrid 内部的 BM25、semantic、RRF，以及固定 Qwen3 reranker 都使用原实现。
每个适用 case / baseline 只调用一次 `engine.search`，传入未经改写的原始 query；
不执行 Agent Loop、Agent tools、read、generation、额外搜索或重试。
方法顺序按 case 轮换，执行始终串行。

## 输入与公平性

- 复用 `parse_agent_eval_dataset`，保存输入 JSONL 的原始字节和 SHA-256。
- `--index-version` 可指定已发布快照；省略时捕获运行开始的 active snapshot。
  后续检索不会跟随 active pointer 变化。所有方法使用同一份快照记录。
- 预检验证 vault、知识库目录、标签来源和 live document revisions 与快照一致。
  文件读取仅用于输入验证和指纹，不是 case 的 read 工具执行。
- 当前 `AgentEvalCase` 没有独立 source-filter 字段，因此使用同一个完整 vault
  范围并传 `filters=None`。`expected_sources`、允许/禁止工具和 read 限制都是
  evaluation 标签，不用于筛选候选或调整 query。也不从 query 中抽取文件名生成过滤器。
- 默认 `exact=False` 与当前 Agent 检索一致；可显式 `--exact` 使用 Qdrant exact。
  固定 pipeline 表示固定执行策略，浮点数、embedding provider 和 ANN 不承诺逐位一致。
- 运行末尾再次保存知识库指纹。检测到漂移时保留结果，并标记 `invalidated` /
  `knowledge_changed=true`，不把它当成固定输入的有效 control group。

## Source 排名与指标

`top_k` 是 Engine 返回的 chunk/result 上限，不保证 K 个不同 source。例如：

```text
返回 chunks: rag.md, agent.md, rag.md, memory.md
source 排名: rag.md, agent.md, memory.md
```

去重沿用现有 retrieval evaluation 的 `dict.fromkeys` 规则：首次出现决定相对
顺序，去重后连续编号。每行同时保留原始 `response`、可重复的 `retrieved_sources`
和去重后的 `ranked_sources`；不追加请求补满 source 数量。

所有公式调用已有 `evaluation.metrics.ranking_metrics`。当前标签为 binary relevance：

- 默认 `top_k=10`，输出 Recall@1/3/5/10、nDCG@1/3/5/10，以及 MRR。
- MRR 使用该次返回的完整 source 排名，最多来自 top_k chunks；不是整个索引上的无限深度 MRR。
- nDCG 沿用现有 gain / discount / ideal ranking 定义，`expected_sources` 的 grade 均为 1。
- 自定义 top_k 时，仅报告不超过该限制的标准 cutoff，并加入 top_k 本身。
  例如 top_k=2 输出 @1、@2；不会将未检索到的 @10 标成已测量。
- `no_retrieval` 标记 `not_applicable`，不发起检索；ranking metrics 和 latency 为 null。
- 有效空结果为 `ok`，ranking metrics 为 0。检索失败为 `error`，指标与 source 排名为
  null，保留错误类型、错误消息和耗时；失败不会转换为有效空结果。

整体与每个 task_type / baseline 都保存总 case 数、适用数、成功执行数、失败数、
不适用数，以及每个指标的实际分母。Ranking means 仅使用有定义的成功执行结果；
latency mean 包括失败请求、排除不适用 case。表中的错误数必须与排名均值一起解读。

单次 latency 只包围 `engine.search`，包含 embedding 和 reranking 推理调用，排除
数据加载、source 去重、指标计算与产物写入。模型/引擎初始化耗时单独记录为 `setup_ms`。

## 与 Agent 指标的比较边界

Baseline source recall 可以与 Agent 整个 trajectory 发现的 unique-source recall
比较，但必须对齐 case 集合、快照和有效分母，并明确单次 top_k 与多轮累计证据的预算差别。
该比较不能直接推导答案质量。

Agent task success 还检查工具限制、read 覆盖、调用次数和停止原因。本阶段不制造
baseline task success，不重新定义 Agent success，也不为 Agent 累计证据伪造单次排名。
MRR/nDCG 只用于 baseline；Agent 的 tool calls、turns、token usage 和生成质量属于
独立的行为/结果维度。本阶段未添加读取历史 Agent 报告后自动合并指标的功能。

## 运行

Python API 默认运行全部四种方法：

```python
from pathlib import Path
from arkb.config import RetrievalConfig, RuntimeConfig
from arkb.evaluation.agent_runner import run_evaluation
from arkb.evaluation.models import BaselineEvalConfig

config = BaselineEvalConfig(
    dataset_path=Path('evaluation/data/agent_v1.jsonl'),
    db=Path('.obsidian-rag/index.sqlite'),
    notes_dir=Path('example_notes'),
    vault_id='default',
    top_k=10,
    retrieval_config=RetrievalConfig(reranker_cache='.obsidian-rag/models'),
    runtime_config=RuntimeConfig(offline=True),
)
run = run_evaluation(config)
# 单独运行：BaselineEvalConfig(baselines=('bm25',), ...)
```

现有 evaluation runner 的模块入口：

```sh
# 单个 baseline，只需要匹配的 SQLite 知识库快照。
uv run --locked python -m arkb.evaluation.agent_runner \
  --baselines bm25 --top-k 10 --output /tmp/arkb-bm25-new

# 全部 baseline：本机需运行 embedding/Qdrant，并准备 Qwen reranker 缓存。
uv run --locked --extra rerank python -m arkb.evaluation.agent_runner \
  --baselines bm25 semantic hybrid hybrid_rerank --top-k 10 \
  --reranker-cache .obsidian-rag/models --offline \
  --output /tmp/arkb-baselines-new
```

通过 `--db`、`--notes-dir`、`--vault-id`、`--index-version`、`--qdrant-url` 指向
Agent Evaluation 使用的同一套输入。输出目录必须是新目录；省略时生成
`evaluation/results/baselines-<timestamp>-<id>`。Baseline 固定一次执行，不接受
`--num-trials` 大于 1。省略 `--baselines` 时仍执行原来的 Agent Evaluation。
Agent generation 参数与 baseline 参数混用会明确报错，避免误以为配置已生效。

产物沿用原 runner 的命名：`cases.jsonl`、`results.jsonl`、`summary.json`、
`report.md`、`run_metadata.json`。每完成一行立即 flush；单 case 失败继续执行，
初始化/配置错误 fail fast，中断或写盘失败会在 metadata 标记未完成。

旧 `evaluation.retrieval baseline` 入口继续用于独立的 graded relevance benchmark
数据格式；Agent v1 control group 使用上述 runner，不转换或改写 v1 dataset。

## 验证与后续接口问题

```sh
.venv/bin/python -m pytest -q
git diff --check
```

全量结果：**957 passed，63 skipped**。新增 40 项检查不依赖真实 LLM；完整本地
集成使用真实 SQLite、Qdrant Local、BM25、Semantic、Hybrid、RRF 和 Reranker，
embedding/scorer 为固定测试替身。原有 Agent metrics、runner 与 ablation 测试通过。

当前 schema/API 足够完成本 milestone。后续若需要每个 case 的 source filter 或
graded relevance，应在共享 dataset schema 中显式声明，而不能由 baseline 从标签推导。
现有 Agent runner 每次 ask 捕获 active snapshot，baseline 则固定整个 run；进行
后续对照时应避免并发重新发布索引，并核对双方的指纹。没有实现 Search Mode Ownership
ablation 或其他后续 Agent 能力。

真实同源运行已完成，见 [实测结果](deterministic-baseline-results.md)。

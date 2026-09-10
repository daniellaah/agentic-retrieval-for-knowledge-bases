# Deterministic Retrieval Baselines：实测结果

2026-09-09 完成同一 Agent Evaluation v1 数据集上的四条固定检索 pipeline。
**40 cases × 4 methods = 160 条结果；136 次实际检索，24 条不适用，0 retrieval errors。**

已回放全部结构化结果的指标，并验证 query、binary ground truth、dataset SHA-256 和知识库
快照与已完成的 4B/9B/27B Agent Model Ablation 三组输入完全一致。

## 固定配置

- 数据集：`evaluation/data/agent_v1.jsonl`；SHA-256 `8babda40f2a74506c3fefe5159a9094a91c86e4e5562443a09c7b35790a0e8af`。
- 已发布快照：`facaea4136d1421389ca640051e1e55c`；40 文档、160 chunks；运行前后知识库指纹一致。
- Embedding：`qwen3-embedding:0.6b`（Qwen3-Embedding-0.6B），1024 维；沿用已存向量及原 query instruction。
- Reranker：`Qwen/Qwen3-Reranker-0.6B`，固定 revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`。
- Engine top_k=10，Hybrid 每路 candidate_k=20，RRF k=60，rerank_candidates=20。
- 默认 Qdrant 检索配置 `exact=False`，与之前 Agent 实验一致；无 query 改写、额外检索、Agent 或 generation。
- CPU float32 reranker，512-token 输入、batch size 16，测试进程 4 个 PyTorch CPU 线程。
- Engine/model setup 单独计时 1656.27 ms；不计入下表请求延迟。

## Overall

每个方法的排名指标分母均为 **34**；另外 6 个 no_retrieval case 不检索、指标为 null。
Recall 表示百分比，MRR/nDCG 为 0–1。K 是一次 top-10 chunks 去重后的 source cutoff。

| Method | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | Mean latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 21.42% | 24.80% | 29.80% | 30.39% | 0.3211 | 0.2872 | 0.21 |
| semantic | 56.13% | 77.89% | 81.13% | 82.79% | 0.9559 | 0.8464 | 99.25 |
| hybrid | 58.48% | 77.94% | 82.35% | 84.61% | 0.9471 | 0.8581 | 77.27 |
| hybrid_rerank | 54.66% | 80.10% | 85.10% | 87.30% | 0.9412 | 0.8687 | 2815.57 |

## 按 task_type 的 R@10

| Task type | Cases | BM25 | Semantic | Hybrid | Hybrid + Rerank |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_read | 6 | 16.67% | 100.00% | 100.00% | 100.00% |
| exact_lookup | 6 | 96.67% | 88.33% | 100.00% | 100.00% |
| exploratory_retrieval | 10 | 15.33% | 68.83% | 68.00% | 68.83% |
| knowledge_qa | 6 | 33.33% | 91.67% | 91.67% | 100.00% |
| no_retrieval | 6 | N/A | N/A | N/A | N/A |
| semantic_discovery | 6 | 0.00% | 74.44% | 74.44% | 80.00% |

在这批原始 query 上，BM25 的覆盖主要来自 exact_lookup；semantic_discovery 为 0。
Hybrid + Rerank 相对 Hybrid 的总体 R@10 增加 2.70 个百分点，MRR 从 0.9471 降至 0.9412，
同时增加了请求耗时。重排对不同 cutoff 的影响不同；本次单次运行不建立统计显著性或通用性能排名。
尤其是 Semantic / Hybrid 的小幅时间差可能受 provider 缓存、加载和运行顺序影响。

Agent trajectory 的 source recall 可以在相同 case 和有效分母下单独对照，需注明多轮累计证据
与单次 top_k 的预算差别。这里未生成 Agent MRR/nDCG 或 baseline task success。

## 产物与复现

- [完整生成报告（含逐任务分组）](results/deterministic-baselines-20260909/report.md)
- [逐 case 原始结果](results/deterministic-baselines-20260909/results.jsonl)
- [结构化汇总](results/deterministic-baselines-20260909/summary.json)
- [运行元数据](results/deterministic-baselines-20260909/run_metadata.json)
- [回放与同源验证](results/deterministic-baselines-20260909/validation.json)
- [测量时源码存档](results/deterministic-baselines-20260909/measured-source.zip)

测量后只补充 CLI 参数混用的拒绝检查；该次测量调用的 Python API executor 和指标代码未变。
源码存档与运行元数据中的指纹已逐一校验。依赖版本及 CPU 线程数另存 `execution_environment.json`。

```sh
OMP_NUM_THREADS=4 uv run --locked --extra rerank python -m arkb.evaluation.agent_runner \
  --baselines bm25 semantic hybrid hybrid_rerank --top-k 10 \
  --db evaluation/results/integration-20260909T190858Z-089e02/index.sqlite \
  --notes-dir example_notes --vault-id agent-eval-integration \
  --index-version facaea4136d1421389ca640051e1e55c \
  --qdrant-url http://127.0.0.1:32776 --reranker-cache .obsidian-rag/models --offline \
  --output /tmp/arkb-baselines-new
```

输出必须使用新目录。完整配置、去重规则、错误分母和接口限制见
[实现说明](deterministic-retrieval-baselines.md)。

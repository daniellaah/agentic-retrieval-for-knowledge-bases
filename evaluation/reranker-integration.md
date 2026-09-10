# Qwen3 Reranker 集成验证

2026-09-09 已取消 reranker A/B 消融实验。生产检索与检索评估统一使用
`Qwen/Qwen3-Reranker-0.6B`，固定 revision
`e61197ed45024b0ed8a2d74b80b4d909f1255473`。

## 审查和清理

- 将 Runtime、CLI 检索和 benchmark 的旧 MiniLM 接线替换为 `QwenRerankerScorer`。
  删除 `CrossEncoderScorer`、模型/版本配置字段及相应 CLI 参数；保留候选数、
  输入长度和缓存目录设置。通过现有 `--rerank` 开关启用重排。
- 使用原生 Transformers 的 [官方 Qwen 模板](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)，
  在截断前预留前后缀，左侧补齐，在最后位置计算 `logit(yes) - logit(no)`。
  默认 CPU float32、512 tokens、batch size 16；返回证据原文不截断。
- 删除消融 runner、专用测试和协议文档、未完成实验及预检/试跑产物、旧 MiniLM
  benchmark 示例结果与两处本地 MiniLM 模型缓存。Qwen 权重保存在
  `.arkb/models`。
- 移除 `sentence-transformers` 及 6 个闲置传递依赖，更新 lockfile、安装环境和
  benchmark 依赖版本记录。已完成的 Agent Model Ablation 历史结果单独保留。

## 实测结果

真实集成测试 **4 passed，0 failed，0 skipped，14.18 秒**。
离线回归 **917 passed**，另外 63 项真实模型/服务测试按 marker 排除。
`uv pip check` 与 `git diff --check` 均通过。

| 检查 | 结果 |
| --- | --- |
| 冻结候选评分 | Paris 相关证据从第 2 位提升至第 1 位；长文本评分输入限制为 512 tokens，返回原文完整 |
| 模板与批处理 | 左侧补齐、末尾评分位置、空候选、重复运行与分批/单条评分一致性通过 |
| BM25 + Qwen | 20 个候选 → 5 个结果；来源分数、排序、原文、字符范围和 source 过滤通过 |
| Semantic + Qwen | 同上；真实 Ollama embedding + Qdrant Server 检索通过 |
| Hybrid + Qwen | 同上；保留 RRF 来源信息 |
| 索引完整性 | SQLite 文件 SHA-256 和已发布快照元数据在测试前后保持一致 |

三条检索路径测试问题为 “How does Reciprocal Rank Fusion combine lexical and
semantic rankings?”，首位均为 `13_hybrid_rank_fusion.md`。测试记录同一次重排
实际收到的候选池，精确核对来源分数；独立 embedding 请求可能存在微小浮点差异。

测试复用 40 文档、160 chunks 的已发布快照
`facaea4136d1421389ca640051e1e55c`，embedding 为 `qwen3-embedding:0.6b`
（1024 维）。环境为 Apple M4 Max、PyTorch 2.14.0、Transformers 5.16.1；
测试期间 PyTorch 使用 4 个 CPU 线程。此测试验证实际接线和结果契约，覆盖一个
知识库检索问题；没有运行 Agent 生成或模型优劣比较。

本地 [机器可读报告](results/reranker-integration-20260909/report.json) 包含模型、
依赖、快照和源文件指纹；同目录的 `frozen.json`、`bm25.json`、`semantic.json`、
`hybrid.json` 保留原始候选和输出，`junit.xml` 保留测试结果。

## 复现

在仓库根目录运行以下命令。`ARKB_RERANKER_TEST_DB` 必须指向 `example_notes`
的现有快照，vault 与 Qdrant 地址应与其匹配；本机本次使用以下配置：

```sh
uv sync --locked --extra rerank
ARKB_RUN_RERANKER_TESTS=1 \
ARKB_RERANKER_OFFLINE=1 \
ARKB_RERANKER_CACHE=.arkb/models \
ARKB_RERANKER_TEST_DB=evaluation/results/integration-20260909T190858Z-089e02/index.sqlite \
ARKB_RERANKER_TEST_VAULT=agent-eval-integration \
ARKB_RERANKER_TEST_QDRANT_URL=http://127.0.0.1:32776 \
  .venv/bin/python -m pytest -q tests/retrieval/integration/test_qwen_reranker_model.py

.venv/bin/python -m pytest -q -m 'not integration'
```

未提供快照环境变量时，只运行冻结候选真实模型测试，三条 Runtime 集成检查跳过。
模型首次下载可使用 `ARKB_RERANKER_OFFLINE=0`；本次所有 reranker 权重均离线读取。

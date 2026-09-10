# Agent 重复搜索：基线修复与验证

日期：2026-09-09。分支：`refactor`。本次仅处理真实模型基线失败，后续架构重构暂停。

## 失败与原因

`tests/agent/integration/test_qwen_agent.py` 中的“帮我找一些写 Agent Memory 的素材”
在未改动的 `7bbf4ce` 和已提交前四项重构的代码上均用尽默认的 8 轮预算。
两份笔记中只有一份包含 Agent Memory 的简短定义。完整轨迹为：

```text
search -> read(memory.md) -> search × 6 -> max_turns
```

后六次搜索改写了关键词，但返回的证据与首次搜索完全相同。工具返回正常，
模型响应均正常结束，没有输出截断、SDK 丢失调用或索引内容缺失的证据。
这是有限语料上的检索停止决策失败：模型不断尝试补足笔记里没有的实现细节。

原有静态提示已经要求避免重复检索。两版加强静态提示的对照仍存在预算耗尽，
或者直到第 8 轮才回答、重复搜索没有减少，因此没有采用。

## 最小修改

修改仅位于 `src/arkb/agent/loop.py`。从现有 tool observations 中比较 search
返回的完整证据对象；忽略查询措辞和结果排序，不增加另一份持久化证据状态。
允许一次没有新增证据的后续搜索。再遇到没有新增证据的后续搜索时，在下一次
模型请求前加入 system 提醒，要求停止泛化或改写搜索、利用已有材料说明缺口，
并明确允许为用户要求的事实继续读取具体来源或关联文档。

判断考虑同一模型轮次中的全部 search 调用。新 chunk、字符范围、正文或 revision
均视为新增证据，即使来自同一文件。首次空搜索和一次后续空搜索仍允许探索。
提醒记录在完整 messages 中，trace 仍从原消息派生；evaluation 的源码哈希继续
覆盖实际 loop 与提示实现。

工具声明、实际 observations、检索算法、模型、`think=True`、`temperature=0`
与 8 轮预算均保持原样。Loop 不强制 final、不屏蔽工具、不添加额外模型请求。
模型忽略提醒时，仍按原协议返回 `response=None, stop_reason="max_turns"`。

## 为什么允许一次没有新增结果的后续搜索

首个实现一遇到重复结果就提醒，修复了 Agent 冒烟用例，却使真实 CLI 的
“Read a.md and explain why cache vectors.” 从第 5 轮正常回答变成预算耗尽。
在相同持久化索引上的对照确认：前四个 HTTP 请求体 JSON 逐字段相同，第五个
请求体只多了一条提醒。它打断了模型原本的收尾，不能作为合格修复提交。

最终条件保留一次查询或策略调整，仅在后续搜索持续没有新增证据时介入。
这使正常的短探索保留原来的收尾机会，同时对已复现的重复搜索提供明确反馈。

另外核对了当前 [Ollama Qwen3.5 renderer 源码](https://github.com/ollama/ollama/blob/v0.33.2/model/renderers/qwen35.go)：
它会序列化非首条 system 消息。本次没有改动 provider 消息协议或删除 thinking。

## 验证范围

常规回归通过公开 `run_agent` 入口，使用真实 BM25 和文档工具，仅脚本化模型
响应。覆盖不同查询返回相同证据、空结果、结果子集、同轮多次调用、新片段、
继续 read、完整 trace，以及模型忽略提醒时仍严格遵守原预算。
原有 CLI snapshot/live 一致性断言保持原样。

真实集成使用本机 `qwen3.5:4b`、`qwen3-embedding:0.6b`、Ollama `0.33.2`
及独立 Docker Qdrant `1.19.0`。模型摘要和完整请求保存在本地报告中。
测试没有写入默认知识库；临时向量集合由各 fixture 清理。

最终代码验证：

- 常规测试：**972 通过，0 失败**。新增进度提醒回归用例对原始 loop 仍明确失败。
- 原失败场景在最终代码上重复 **3/3 完成，均为第 6 轮**，工具轨迹为 search、
  read、search × 3、final。前四个请求体 JSON 与基线相同，随后才出现提醒。
  仍有重复搜索，但已能在原预算内报告已有素材和资料缺口。
- CLI 同索引对照恢复为 **第 5 轮完成**；全部请求体 JSON 和完整 AgentResult
  与原始 loop 相同，正常收尾没有触发提醒。
- 真实集成：**59 通过，0 失败，0 跳过**，覆盖 Agent、CLI 生命周期、持久化
  Qdrant、生成/引用、embedding 和 tokenizer。另 4 个可选 reranker 用例与本次
  Agent loop 修改无关，没有纳入本轮；未放宽任何已有真实测试的预算或断言。
- 默认思考模式的跨文档重复对照：**2/2 事实完整**；CLI 和独立 Runtime 场景也
  返回实际“30 天”及随机口令。关闭思考的对照 **0/2 事实完整**，与已有已知
  行为一致，不将它的协议断言通过当成任务成功。
- 冻结的 retrieval/context/tool schema JSON 与重构前基线逐字节一致。
- `git diff --check` 通过。

本地原始证据保存在被 Git 忽略的
`.arkb/agent-stopping/refactor-baseline-20260909/`：
[常规测试 JUnit](../.arkb/agent-stopping/refactor-baseline-20260909/final-unit.xml)、
[真实集成 JUnit](../.arkb/agent-stopping/refactor-baseline-20260909/final-integration.xml)、
[模型与环境](../.arkb/agent-stopping/refactor-baseline-20260909/environment.json)、
[原始失败轨迹](../.arkb/agent-stopping/refactor-baseline-20260909/baseline-1.json)、
[最终轨迹](../.arkb/agent-stopping/refactor-baseline-20260909/final-1.json)、
[最终同索引对照](../.arkb/agent-stopping/refactor-baseline-20260909/cache-comparison-final/progress.json)。
保留的 SQLite 是诊断证据；临时 Qdrant 集合清理后需重新索引才能执行向量查询。

这是一条依据精确返回值的进度提醒，不是语义新颖度评分，也不保证所有模型
或查询都能在预算内完成。`final` 仅表示协议结束；跨文档测试另行检查随机口令
和期限。显式 `think=False` 仍保留为对照，其协议通过不代表事实完整。

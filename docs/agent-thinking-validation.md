# Agent thinking 配置与 CLI 验证

日期：2026-09-09。真实模型为本机 `qwen3.5:4b`，embedding 为
`qwen3-embedding:0.6b`，Qdrant Server 为 `1.19.0`。

## 为什么再次出现提前结束

此前诊断已通过冻结请求和完整 Runtime 对照定位到硬编码的 `think=False`，
但当时明确只完成诊断，没有修改运行配置。CLI 改为调用 Agent Runtime 后，
仍然继承了这一设置，因此跨文档任务再次出现相同行为。此次排查最初没有先
复用已有报告，重复了已完成的诊断。

[原始诊断与对照证据](../.arkb/agent-diagnosis/run-20260909-01/report.md)
保留原状；这里记录后续配置落地和验证，避免把“已定位”误当作“已修复”。

## 已落地的调用关系

```text
arkb ask [--think | --no-think]
  -> Runtime.ask(think=...)
  -> Runtime.run_agent(think=...)
  -> arkb.agent.run_agent(think=...)
  -> Ollama Client.chat(think=...) on every model turn
```

`DEFAULT_AGENT_THINK=True` 是当前 Agent 默认值。Python 两个 Runtime 入口和
底层 loop 均接受显式布尔参数；非布尔值报错，不按真值隐式转换。
`--no-think` 可用于对照或使用不支持 thinking 的模型。没有模型名称判断、
失败时自动切换模式或 query 文本路由。

```sh
arkb ask "帮我核对关联笔记中的事实" --think --trace
arkb ask "帮我核对关联笔记中的事实" --no-think --json
```

`--json` 只序列化相同 AgentResult；`--trace` 将工具调用轨迹写到 stderr，
最终正文或 JSON 写到 stdout。这两个输出开关都不影响 thinking 或执行。
正常正文和轨迹不打印 provider 的 thinking 文本，完整 JSON state 保留 provider
消息字段。固定 RAG Python generation API 仍使用原有 `think=False` 及配套
token 计数配置。

## 最终 CLI contract

| 命令 | Runtime 入口 / capability | 主要参数 |
| --- | --- | --- |
| `match <pattern>` | `Runtime.match` → live `ExactRetriever` | `--source`、`--top-k`、`--notes-dir` |
| `search <query>` | `Runtime.search` → `RetrievalEngine` | `--mode bm25\|semantic\|hybrid`（默认 semantic）、`--top-k`、`--source`，保留已有 rerank 等参数 |
| `ask <query>` | `Runtime.ask` → Agent tools / loop | `--think` / `--no-think`、`--max-turns`、`--trace`、`--generation-model`；无 `--mode` |
| `index` | `Runtime.index` → `build_index` | `--notes-dir`、`--force` 及已有 embedding / chunking / Qdrant 参数 |
| `status` | `Runtime.status` → SQLite manifests | `--db`、`--vault-id` |

五个命令均支持 `--json`、`--db`、`--vault-id`。旧 `query` 已移除，纯检索迁移
至 `search`，Agent 检索回答使用 `ask`，固定 RAG 保留 Python API。
console entrypoint 仍是 `arkb.interfaces.cli:main`。search 固定使用 index snapshot；
match/read 读取实时文件。status 保留已有索引元信息展示，没有增加文件 freshness
扫描或服务健康探测。

主要业务修改位于 `src/arkb/config.py`、`src/arkb/interfaces/cli.py`、
`src/arkb/runtime.py`、`src/arkb/agent/loop.py`、`src/arkb/agent/tools.py`。
相应 CLI、Runtime、Agent、SDK contract 和真实集成测试位于 `tests/interfaces/`、
`tests/test_runtime.py`、`tests/agent/`；固定 generation 集成测试迁移到 Python API。
README 与 `benchmarks/retrieval.md` 同步命令迁移说明。

## 验证方法与首轮发现

离线测试不调用真实 LLM；使用 fake runtime / scripted model 检查参数透传、
多轮工具调度、JSON 和 trace 的执行等价性、错误处理与索引行为。实际 Ollama SDK
通过 MockTransport 检查 HTTP JSON 中每轮的 `think`。

真实测试建立独立 SQLite 和临时 Docker Qdrant，无宿主目录挂载，使用真实 CLI
index、Ollama embedding、Retrieval Engine、Agent tools 和模型。跨文档测试将
随机验收口令与“30 天”仅写入关联规范，检查回答及 tool observation 中的证据，
不能仅以 `stop_reason="final"` 判定任务成功。另检查增量索引、三个 search 模式、
实时 match 与 snapshot 差异、CLI JSON/trace、turn limit 和固定 RAG Python API。

同一 snapshot 上分别运行两次 `think=False` / `True`，记录实际 HTTP 参数、
工具数量、耗时以及每次 search 新增的 source 和精确片段。
`new_snippets` 按 document/chunk ID、位置和正文去重；它不是语义新颖度指标。
关闭组是诊断对照，测试协议检查通过不代表 `facts_complete=True`。

首轮真实测试 **23 通过、1 失败**。跨文档开启组 2/2 完整，关闭组 0/2 完整。
开启组每次 search 1 次、read 2 次，约 16.9–17.5 秒；关闭组 search 1 次、read
1 次，约 3.7–4.4 秒。简单相关性查询开启后 search 3 次、read 2 次，约 15.1–15.7
秒，其中一次 search 没有新增片段；关闭时 search 1 次，约 2.2–2.8 秒。

失败用例是只配置 BM25 的模型冒烟测试：工具 mode enum 已限制为 BM25，但
系统指令和 search 描述仍笼统列出 semantic/hybrid，模型选择了未配置的 semantic。
随后修正提示与工具声明的一致性：系统只要求按可用 mode 选择，mode 描述按
实际 engine capability 列出策略及含义。保留严格报错，不将错误策略替换为默认值。
新增真实 SDK 边界回归检查，先复现受限 engine 的声明冲突，再验证修正。

首轮证据：[汇总](../.arkb/agent-thinking/run-f6cnxc5m/summary.json)、
[JUnit](../.arkb/agent-thinking/run-f6cnxc5m/junit.xml)。

## 最终代码的测试结果

- `.venv/bin/python -m pytest -q -m 'not integration'`：**784 通过，60 个 integration
  用例未选中**，2.67 秒。`git diff --check` 通过，安装后的 `arkb ask --help`
  正确展示 `--think` / `--no-think`。
- 真实集成组合：**23 通过、1 失败**，150.83 秒。包括 5 条 BM25 Agent 冒烟、
  17 条持久化 Runtime/CLI/think 对照、1 条 CLI 索引生命周期、1 条固定 RAG API
  测试；没有把关闭组事实不完整误报为任务成功。

| 同一 hybrid snapshot 的场景（各重复两次） | think | 事实完整 | search / read 次数 | 耗时 |
| --- | --- | --- | --- | --- |
| 跨文档核对期限及随机口令 | False | 0/2 | 1 / 1 | 4.07–4.69 秒 |
| 跨文档核对期限及随机口令 | True | 2/2 | 1 / 2 | 9.51–10.14 秒 |
| RAG 相关材料查询 | False | 未做独立相关性评分 | 1 / 0 | 3.81–4.40 秒 |
| RAG 相关材料查询 | True | 未做独立相关性评分 | 1 / 0 | 8.08–8.10 秒 |

这组最终提示下，相关材料查询没有重复 search；首轮和此前诊断仍表明过度检索
可能发生。上述变化不能单独归因于 thinking：两轮之间还修正了工具模式描述，
运行时间也受本机缓存等因素影响。

`arkb ask` 的默认模式真实 CLI 跨文档用例也通过，轨迹为
`search → read(memory.md) → read(memory-policy.md) → final`，耗时 11.91 秒，
返回本次实际的“30 天”和随机口令。实际 HTTP 对照确认每一轮使用选定的 think。

唯一失败是 CLI 生命周期中的 Agent 子步骤，显式指定 `--max-turns 4`：

```text
read(a.md)
search("cache vectors store reuse", mode="semantic")
read(b.md)
search("why cache vectors performance efficiency", mode="hybrid")
max_turns
```

CLI 正确返回 exit 1、`response=null` 和 `stop_reason="max_turns"`；模型未能在
该预算内完成简单解释，质量断言保持失败。没有提高测试预算或放宽断言来隐藏它。
这一轮生命周期测试在该断言处中断，因此后续增量索引步骤没有执行；首轮真实
生命周期完整通过，相关 indexing/search 业务代码在两轮间没有变化。

随后用相同的两份笔记建立独立临时索引，对同一查询直接运行真实 CLI 对照：
4 轮预算复现相同轨迹并返回 `max_turns`（7.78 秒）；8 轮预算保持相同的四次工具
调用，在第 5 轮返回 `final`（9.08 秒）。这证明默认预算下该场景可以完成，
也确认最终回答自身占用一轮；它不消除多余检索的质量问题。
[预算对照原始结果](../.arkb/agent-thinking/budget-c95zf1rm/summary.json)
与两次完整 JSON 均保留，临时集合和容器已清理。

最终证据：[汇总](../.arkb/agent-thinking/run-bduu7xn1/summary.json)、
[完整测试输出](../.arkb/agent-thinking/run-bduu7xn1/pytest.log)、
[JUnit](../.arkb/agent-thinking/run-bduu7xn1/junit.xml)、
[真实 CLI 结果](../.arkb/agent-thinking/run-bduu7xn1/agent/CLI-materials.json)。
两轮均确认临时 Qdrant 集合清空、独立容器停止并自动删除，真实测试没有写入
默认知识库数据库。最终运行环境为 Ollama 0.33.2。

## 后续仍需评估

- Thinking 能改善已复现的跨文档场景，但可能增加检索次数和延迟。
  `--max-turns` 仅限制模型请求次数；不是工具总数或耗时上限。
- `stop_reason="final"` 是模型协议结束。事实完整性、相关性、引用支持仍需评估。
- top-k 是片段数量，可能集中于同一文档；多个长 ID 仍可能被模型混淆。
  当前 source 选择器可读取已知文件，本次不改变证据结构或强制检索顺序。
- 本次验证时模型或工具异常会抛出，CLI `--trace` 仅格式化正常返回的轨迹。
  后续最小 AgentTrace 支持已补充 `result.trace`，并在原异常允许附加属性时通过
  `error.agent_result.trace` 保留部分轨迹；CLI `--trace` 也会显示该调用摘要。
- 本次是小规模、同一任务的重复对照，不是独立查询集成功率或通用性能基准。

原始数据保存在本地被 Git 忽略的 `.arkb/agent-thinking/` 中；本文件及
测试代码可随本次变更提交。保留的 SQLite 用于诊断，临时向量集合清理后需重新索引
才能再次执行向量查询。

# P2 预算、运行观测与输出评审：2026-09-10

**P2 的工程首版与真实服务诊断已完成；可信的输出质量结论仍待独立人工标签与 judge 校准。** 本轮 36 次 Agent 试验、8 次固定生成和 4 次 judge 请求均已结算，原始记录及参考 token 计数可离线复算。发布检查会明确拒绝把这些开发数据当成已验证的质量结果。

## 本轮实现

- `AgentObserver` 是可选、每次任务新建的观测对象。记录原始模型请求/响应、provider 原生 usage、模型/工具耗时和错误阶段；工具状态区分 requested、success、error、skipped，并分别标明是否已尝试执行、进入对话、提交给下一次模型请求。
- `AgentBudget` 在线限制工具总次数、查询型工具次数、read 次数和累计交付的证据 tokens。重复内容重复计费；超限的整条工具观察保留在审计中，但不进入对话，不偷偷裁剪、不额外生成答案，明确返回 `budget`。
- 时间预算在操作边界检查，并拒绝迟到结果。**它不能取消已经发出的同步请求**，实际墙钟时间可能超限；它不是硬实时截止。预算计时范围为 Agent loop，初始化和 runner 总耗时另外记录。
- v2 证据评分支持缺少范围元数据、但与对应 revision 的完整正文逐字相等的结果。无坐标的局部文本仍拒绝。原 P1 归档保留旧评分源码和测量，未回写历史结果。
- 最终答案、query/rubric、实际提交证据与停止状态绑定为 `packet_sha256`。改动任一部分后，旧人工评分会被拒绝。没有人工判定时，正常最终回答的语义正确率与 grounded success 保持 null；运行/预算失败的交付效用为 0。
- Judge 分成 correctness 与 support 两次请求。前者看参考要点，后者只看候选回答实际取得的证据；保留所有原始判断与解析错误，不重试挑选好分数。能解析 JSON、引用 ID 存在或 quote 出现，都不自动证明语义判断正确。
- 校准工具报告混淆矩阵、错误答案被放行的 false-accept rate、Wilson 区间和 judge coverage；拒绝调试/校准之间的 family 泄漏与重复 family 伪独立样本。

默认不传 observer 时，模型消息、工具选择路径和原 `AgentResult` / `AgentTrace` 结构保持兼容。启用后使用 `ObservedAgentResult` 保存附加记录。单元测试比较了启用纯观测前后的请求和轨迹。

## 冻结实验

见 [P2 协议](experiments/p2-protocol.json) 与 [机器摘要](experiments/p2-budget-20260910-summary.json)。从已见的 provisional pilot 预先选择 12 条查询，属于 **9 个 family**，覆盖语义发现、探索、多跳、exact、read、证据缺口与无需检索。每种配置只运行一次，不是独立 test 样本。

三组均使用 `qwen3.5:4b`、thinking、8 轮、总工具尝试 ≤12、read ≤6、协作式 120 秒截止；只改变 query-call 或 evidence 限额。预算不通过 gold 设置，也不向 Agent 增加预算专用提示。查询型调用为 match + search。顺序按 case 轮换，三组 Agent 全部结束后才切到固定生成与 9B judge。

| 预算配置 | 正常结束 | 预算停止 | 耗尽轮数 | 平均已执行工具尝试 | 平均交付证据 tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| query ≤6 / evidence ≤4,000 | 9/12 | 2/12 | 1/12 | 3.33 | 634.0 |
| query ≤1 / evidence ≤4,000 | 5/12 | 7/12 | 0/12 | 1.25 | 160.8 |
| query ≤6 / evidence ≤2,000 | 9/12 | 2/12 | 1/12 | 3.33 | 634.0 |

36 次中有 23 次正常结束、11 次因查询上限停止、2 次耗尽轮数、0 次运行异常。**正常结束不是回答正确**，这 23 条输出均等待人工审阅。

参考 token 使用仓库固定的 embedding tokenizer 作为跨配置统一的文本计量器，归档保存完整 tokenizer 和 fingerprint。只对每条 evidence 的 content 计入证据上限；工具 schema、JSON metadata 和对话模板不在这个限额中，另外保留请求 JSON 的参考 token 数与 provider 原生计数。JSON 参考计数不冒充模型真实 chat-template 长度。

Agent 原生 prompt/output tokens 在三组的平均值分别为 **17,826.6 / 644.2、3,879.3 / 334.5、17,826.6 / 644.2**，每组分母均为 12。这里是所有 Agent chat 请求的 provider 报告值之和，包含多轮重复输入；**embedding 推理的 token usage 未单独采集**。整体单次均时为 10.61 / 4.66 / 10.48 秒，缓存/负载未控制，不能声称生产加速或等质量优化。

## 分母会怎样误导结论

12 条问题中 6 条有可计算的必需证据标签。只统计“正常结束且适用”的问题时，三组 evidence coverage 都是 100%；对应分母却是 **5、1、5**。

保留全部 6 条适用问题，并把预算/轮数失败记为 0 evidence delivery utility，三组分别是 **83.33%、16.67%、83.33%**。这仍只是临时证据标签的交付诊断，不是答案正确率，但说明过滤失败任务会掩盖预算造成的交付损失。

这批样本未触发 2k/4k 证据限额，两组交付 token 数和停止结果一致。由此不能推断更小上下文对生产质量无损。额外的 [真实服务契约检查](audits/20260910-live-evidence-budget.json) 使用独立、刻意设小的 1-token 限额：read 实际返回 **109 tokens**，交付 **0 tokens**，只执行一次模型请求并以 evidence budget 停止。它只验证限制确实生效，不参与主比较。

另做了 [按正文区间并集去重的离线诊断](audits/20260910-p2-unique-context.json)。它只读取实际提交给模型的文本，合并同 revision 的重叠区间；不使用 gold 来补文本，也不宣称实际运行过压缩上下文。在线 observer 的 exact-excerpt 去重与这个区间并集口径分开保存。

## Gold-context 与 judge 诊断

4 个预选 case 分别把基线实际提交的证据、每个 gold facet 的首个可接受组合交给同一个冻结 synthesis prompt / 4B 模型，产生 8 个回答。生成只接收 query 与所给证据，不接收评分 rubric；gold 条件明确属于诊断 oracle，不是可部署检索器。

8 个回答均正常结束，所给证据对草稿 gold 的 coverage 均为 1，但这没有解决最终输出判断。例如 `pilot_040` 的 gold 已同时提供“F4 委托 policy_4”和“policy_4 的保留期限”；4B 仍要求确认 F4 是否就是链接到该政策的服务。这说明证据齐全与回答完成度必须分开评估。该例应人工检查后再决定优化生成还是标注表达。

对该 case 的两个 synthesis 输出，9B 各执行 correctness/support 两次评审，共 4 次，均通过结构与引用检查，状态仍为 provisional。其中基线上下文回答的 correctness 理由声称答案明确排除了旧的 **84 天**政策，但实际回答没有这项表述。原始证据、回答与 judge 理由全部保留；这是需要人工校准的具体问题，不能靠模型自报通过或 confidence 消除。

已导出 [44 个输出复核包](reviews/p2-budget-20260910/README.md)，当前人工完成 0 条。两个 judge 模式的校准集合均为空，校准报告保持未通过；没有虚构人工标签、误接受率或可信的 grounded success。

## 归档与检查

完整 [P2 制品](experiments/artifacts/p2-budget-20260910.tar.gz) 约 3.80 MB，[manifest](experiments/artifacts/p2-budget-20260910.tar.manifest.json) 保存哈希。包括源码与锁文件、58 篇语料、索引、模型身份、参考 tokenizer、原始请求/响应、预算记录、评分和待审输出。参考 tokenizer 在运行中作为归档补充保存，fingerprint 与全部实测 observer 一致，计数已离线重算；未改变推理输入。

[解包验证记录](audits/20260910-p2-archive-validation.json) 确认了 36 + 8 行评分和所有参考 token 计数。临时修改原始输出会被哈希检查拒绝。`--require-release` 在实验有效但质量门槛未满足时退出 **2**，避免把正常脚本退出等同于评测通过。

```sh
mkdir -p /tmp/arkb-p2-replay
tar -xzf evaluation/experiments/artifacts/p2-budget-20260910.tar.gz -C /tmp/arkb-p2-replay
.venv/bin/python evaluation/experiments/verify_bundle.py /tmp/arkb-p2-replay/p2-budget-20260910

# 当前预期退出 2：provisional / 缺独立输出复核 / 开发诊断
.venv/bin/python evaluation/experiments/verify_bundle.py \
  /tmp/arkb-p2-replay/p2-budget-20260910 --require-release
```

## 使用接口与未完成条件

```python
from arkb.agent import AgentBudget
from arkb.runtime import Runtime

with Runtime() as runtime:
    observer = runtime.agent_observer(budget=AgentBudget(
        max_query_calls=3, max_read_calls=6, max_tool_calls=9,
        max_evidence_tokens=4000, max_elapsed_ms=120000))
    result = runtime.ask('查询内容', observer=observer)
    print(result.stop_reason, result.observation['usage'])
```

每个任务必须新建 observer；重复使用会报错。API 默认不启用预算，CLI 产品路径尚未增加预算参数。P2 正式验收仍需独立 gold / 输出审阅和 judge 校准；硬取消 deadline、embedding 的逐请求 usage、生成模型精确上下文上限与基于质量/成本的正式 release 决策也尚未实现。P3 可以复用本轮接口做受控开发实验，但目前不能给出基于独立质量数据的采用/发布结论。

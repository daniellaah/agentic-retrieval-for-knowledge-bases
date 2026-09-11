# P2 输出与 judge 校准复核

状态：**pending，44 个输出包，人工已审 0 个，独立校准样本 0 个**。版本信息与两个 judge 模式的身份见 [manifest.json](manifest.json)。这是可填写的待办包，不是已完成的人审记录。

`outputs.json` 含 36 个 Agent、8 个固定生成的回答、实际提交证据、原始 query/rubric、停止状态和对应 `packet_sha256`。输出材料已经用于开发，后续不能一边根据它调 judge prompt，一边把它当独立校准集。

## 逐输出审阅

只填写每项的 `review`，不要改 `packet`。完整标注以下五项 boolean，并填写真实 reviewer、简短具体理由；完成后才将 status 改为 `human_reviewed`：

| 字段 | 核对内容 |
| --- | --- |
| `fact_correct` | 回答的事实、数值、适用条件是否正确 |
| `fact_complete` | 必需事实和限定是否遗漏；完全拒答或只说一句正确话不能算完整 |
| `claims_supported` | 只用 packet.evidence 判断候选实际论断是否得到支撑；不要用未取得的 gold 补支撑 |
| `citations_complete` | 需要来源的论断是否有充分引用；无须外部证据的任务按适用情况判断 |
| `task_fulfilled` | 是否真正完成用户任务；exact 看最终交付文件集合、read 看所问信息、缺证据看克制/澄清行为 |

未完成的判定保持 null/pending。`score_output` 仅接受绑定同一 packet hash 的完整人工判定作为语义评分输入；绑定不匹配会拒绝，错误或预算失败不会因人工认为某个片段正确而成为任务成功。

重点复核 `pilot_040` 的两种固定生成：gold 桥接证据存在时回答仍请求确认；另一种输出的 judge 理由提到了答案中未出现的旧期限。这些是诊断线索，不是已仲裁结论。

## 独立校准

`correctness-calibration.jsonl` 和 `support-calibration.jsonl` 当前为空。先完成 judge 调试，再从没有参与该调试的需求族采样，获得真实人工真值。保存原始包、review、judge 原始输出和采样规则；不要自动把 model_provisional 当成 human_pass。

每条校准记录使用以下字段：`id`、`family_id`、`split`（dev 或 calibration）、`judge_identity`（manifest 对应模式的 identity）、`human_pass`、`judge_pass`、`reviewer`。human_pass 必须为人工判定的 boolean；judge 无法判断或解析失败时 judge_pass 为 null。

Correctness 的 binary pass 需同时满足每个必需事实正确、完整和任务完成；support 的 binary pass 需满足全部应检查论断已覆盖、得到实际证据支持、引用完整。判断单位和采样规则必须先冻结。

当前统计工具为 Wilson 区间采用**每个独立 family 一个预注册校准结果**；重复 family 和 dev/calibration 泄漏会报错。默认检查点为至少 100 个独立结果、judge coverage ≥95%、条件 false-accept rate 的 95% 上界 ≤10%，并要求有人类正负例。这是工程检查点，尚未由项目风险目标校准，不能自动当成上线标准。审阅人真实性、样本代表性及一致的共同偏差无法靠 JSON 校验认证。

```sh
.venv/bin/python -m arkb.evaluation.calibration \
  evaluation/reviews/p2-budget-20260910/correctness-calibration.jsonl \
  --judge-identity '<manifest 中 correctness.identity>'
```

随包的 `*-calibration-status.json` 如实报告当前 0 样本、无可计算比率/区间和未通过状态。修改 prompt、judge model digest、评分模式或关键参数后，需要使用新的 judge identity 并重新校准。

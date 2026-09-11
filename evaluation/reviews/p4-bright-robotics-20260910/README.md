# P4 输出独立审阅包

状态：待审。当前人工完成数为 0；本目录不构成 release gold。

`review-packets.json` 按固定 seed 打乱，使用 packet_id 绑定问题、原始回答、实际提交的证据和评分侧参考。
`diagnostics.jsonl` 以相同 packet_id 关联运行成本和非互斥失败标记。审阅者先审原始 packet，再看诊断，以减少结果标签的影响。

请保留 packet 原文，只填写 review。reviewer_id 使用经项目确认的审阅人身份；status 为 pending 或 reviewed。
以下四项采用 pass / partial / fail / not_applicable，notes 必须指出具体主张、证据范围及理由：

- answer_correctness：回答是否正确解决问题；检索命中不等于回答正确。
- claim_support：回答中的事实主张是否由 submitted_evidence 支持。提交到请求不证明服务端处理成功。
- aspect_completeness：Bright 的相关方面是否完整覆盖；MuSiQue 参考支持关系审查多跳链，必要时标 not_applicable。
- abstention_appropriateness：证据不足时是否正确说明不足；超时、无输出、格式错误不能算正确拒答。

Bright qrels 的正文别名/遗漏可能影响文档代理分，遇到疑似标注问题记录证据并另建标注修订，不就地改官方 gold。
MuSiQue 的不可回答变体仍可能保留部分支持标记；部分命中不能证明上下文足够。
出现争议时由另一名独立审阅者复核并保留双方意见。任何外部公开集审阅均不能替代未见的 ARKB 发布盲测。

manifest 记录导出时各文件 hash。提交审阅结果时另存版本并保留原始 manifest，不能覆盖 packet 或伪装原 hash 未变。

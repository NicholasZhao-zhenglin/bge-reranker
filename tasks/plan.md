# Implementation Plan: 合成清洁车 Reranker 数据闭环

## Overview

构建一套明确标记为合成数据的无人驾驶清洁车领域小型数据集，完成 250 条训练/验证/测试样本、原始 BGE reranker 基线评测、错误分析和难负样本挖掘。所有数据、脚本、结果和日志均保存在仓库内。

## Architecture Decisions

- 使用合成演示文档，不将其描述为公司内部或真实生产数据。
- 以 JSONL 保存可追溯的样本；每条 query 保留来源文档、正样本和负样本标识。
- 按文档主题划分 train/validation/test，避免同一主题模板同时泄漏到训练与测试。
- 基线评测采用固定候选池和 `raw_score` 排序，指标包含 Recall@1、Recall@3、MRR。
- 难负样本只从基线模型实际误排或低间隔案例中产生，并保留原始分数与错误类型。

## Task List

### Phase 1: Data foundation

- [x] Task 1: 生成合成领域文档与 250 条带来源的 reranker 样本。
- [x] Task 2: 校验数据 schema、数量、split 与正负样本引用。

### Checkpoint: Data foundation

- [x] 250 条样本可解析，所有引用的文档存在。
- [x] 数据说明明确标记为合成演示数据。

### Phase 2: Baseline evaluation

- [x] Task 3: 批量运行原始 BGE reranker，对固定测试集打分并保存逐 query 排序结果。
- [x] Task 4: 计算指标并输出错误/低间隔案例。

### Checkpoint: Baseline

- [x] 评测可复现，日志包含模型路径、设备与运行时间。
- [x] 每个错误都有 query、gold passage、预测 passage 与分数。

### Phase 3: Hard-negative mining

- [x] Task 5: 将错误案例和低间隔案例转为带类型标签的难负样本数据集。
- [x] Task 6: 生成总结报告与下一阶段 LoRA 数据建议。

### Checkpoint: Complete

- [x] 所有数据集、脚本、结果与学习记录已保存。
- [x] 结果报告明确区分实测结果与合成数据局限。

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 合成数据与真实业务分布不同 | 高 | 文件名、说明和报告中明确标注；后续替换 raw 数据并重跑。 |
| 基线模型没有产生 Top-1 错误 | 中 | 记录低间隔风险案例，不伪造错误。 |
| 8GB 显存导致评测不稳定 | 中 | 使用 fp16、小批量、CPU 回退和逐批日志。 |

## Open Questions

- 后续真实数据的获取方式、脱敏规则和可用范围需要由用户确认。

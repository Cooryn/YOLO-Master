# A2 — STAL 小目标自适应标签分配

> 任务代号 **A2**：STAL（面积感知标签分配，面向 VisDrone 密集小目标）。
> 分阶段推进：**P0**（保底基线）→ **P1**（面积感知 on/off 对照，核心）→ **P2**（扫描 / 扩展）。

## 阶段

| 阶段 | 状态 | 内容 | 目录 |
|---|---|---|---|
| P0 | 进行中 | 真实 VisDrone2019-DET 基线 + tiny/small/medium/large 分档 AP + 每 epoch 正样本统计 | [P0/](P0/README.md) |
| P1 | 待做 | 面积感知 on/off 对照：`tal.py` 面积阈值 / αβ / topk 参数化 + `default.yaml` 注入键 | [P1/](P1/README.md) |
| P2 | 理想 | (1) 扫描面积阈值 / 放宽幅度 / warmup；(2) 第二数据集 或 seg/pose | [P2/](P2/README.md) |

## 准入检查（冒烟）

- [smoke/](smoke/README.md)：VisDrone 子集 1 epoch 冒烟 + 定位 assigner / 配置注入点（已完成）。

## 验收标准（Definition of Done）

1. STAL 模块与配置 PR —— `tal.py` 逻辑 + `default.yaml` 配置键。
2. 分档 AP on/off 对比表（tiny/small/medium/large）。
3. 每 epoch 正样本演化曲线（`fg_s8/s16/s32`）。
4. 参数敏感性说明（面积阈值 / αβ / topk / warmup）。

- 指标必须分档报，禁止只报总 mAP。
- 训练抖动 → 渐进 warmup。

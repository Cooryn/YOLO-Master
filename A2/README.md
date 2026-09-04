# A2 — STAL 小目标自适应标签分配

> 任务代号 **A2**：STAL（面积感知标签分配，面向 VisDrone 密集小目标）。
> 分阶段推进：**P0**（保底基线）→ **P1**（面积感知 on/off 对照，核心）→ **P2**（扫描 / 扩展）。

## 阶段

| 阶段 | 状态 | 内容 | 目录 |
|---|---|---|---|
| P0 | ✅ 已完成 | 真实 VisDrone2019-DET 基线 + small/medium/large 分档 AP（COCO 32²/96²）+ 每 epoch 正样本统计。主指标 APs=**0.0905** | [P0/](P0/README.md) |
| P1 | ✅ 已完成 | 面积感知 on/off 三组对照（纯 TAL / fixed / adaptive）+ `default.yaml` 注入键 + 33 项单测。APs 0.0875 / 0.0905 / **0.0925** | [P1/](P1/README.md) |
| P2 | 待做 | (1) 扫描面积阈值 / 放宽幅度 / warmup + α/β 消融；(2) 第二数据集 或 seg/pose | [P2/](P2/README.md) |

## 准入检查（冒烟）

- [smoke/](smoke/README.md)：VisDrone 子集 1 epoch 冒烟 + 定位 assigner / 配置注入点（已完成）。

## 验收标准（Definition of Done）

1. ✅ STAL 模块与配置 PR —— `tal.py` 逻辑 + `default.yaml` 配置键（P1 完成）。
2. ✅ 三组对照分档 AP on/off 对比表（纯 TAL / fixed-stride STAL / adaptive STAL；small/medium/large）。
3. ✅ 每 epoch 正样本演化曲线（`fg_s8/s16/s32` + per-tier avg_pos/zero_ratio，随 `results.csv` 写入）。
4. ⬜ 参数敏感性说明（面积阈值 / αβ / topk / warmup，属 P2 扫描）。

- 指标口径见 [P0/README.md](P0/README.md)：主指标 APs@[.50:.95]（COCO 32²/96²，maxDets=500），必须分档报，禁止只报总 mAP。
- 训练抖动 → 渐进 warmup。

# P1 — STAL 面积感知 on/off 对照（核心）

> 待做。在 `ultralytics/utils/tal.py` 改面积阈值 / αβ / topk，加 `default.yaml` 注入键，
> 同配置同 seed 对比分档 AP 与正样本分布（对照 P0 底座）。

## 关键约束

- **必须 config-driven**：面积阈值 / αβ / topk / warmup 做成 `default.yaml` 键，不能硬编码进实验脚本——否则 P2 无法扫描。
- 对照 P0 统一用 `A2/P0/tiered_eval.py` 评测，禁止只报总 mAP。

## 注入点

| 机制 | 位置 |
|---|---|
| TAL assigner 本体 | `ultralytics/utils/tal.py` `TaskAlignedAssigner` |
| 小框硬编码撑大 | `tal.py` `select_candidates_in_gts` |
| 对齐度量 | `tal.py` `get_box_metrics`（`score^α · iou^β`） |
| topk 候选数 | `tal.py` `select_topk_candidates` |
| 参数注入链 | `loss.py` → `tasks.py` `DetectionModel.init_criterion` → `default.yaml` |

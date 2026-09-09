# P2 — 扫描 + 扩展（stretch）

> 目标：(1) 扫描面积阈值 / 放宽幅度 / topk / α·β，出参数敏感性说明（验收 #4）；(2) 第二数据集 或 seg/pose。
> P1 结论「更多正样本 ≠ 成比例 AP 提升」，P2 用 30-epoch 短周期**筛参**找出甜点，再 120-epoch 复现协议**认定**。

## 前提（本轮已完成）

- 全部超参 config-driven（`default.yaml` 键），可直接扫：
  - `stal_area_small` / `stal_area_medium`（面积阈值）
  - `stal_topk_small`（small 档 top-k）、`stal_expand`（候选扩张半径）
  - **`tal_alpha` / `tal_beta`（TAL 度量 `score^α · iou^β`，本轮新增，默认 0.5 / 6.0）** ← P1 只读不动，P2 独立消融
- 入口脚本 `A2/P0/baseline_train.py` 新增 `--tal-alpha` / `--tal-beta` 覆盖，支持 `--resume` 暂停/续跑。
- 新增 `tests/test_default_config_integrity.py::test_tal_alpha_beta_type_and_value_checked`，单测 34 项通过。

## 基准（P1 adaptive 冠军）

`stal_mode=adaptive, stal_min_positive=true, stal_topk_small=13, stal_expand=1.0, tal_topk=10, tal_alpha=0.5, tal_beta=6.0`
→ **APs=0.0925**（+0.20pp vs fixed、+0.50pp vs 纯 TAL）。

## Phase 1 — 30-epoch 粗筛（全量数据，FP32，seed 0，imgsz 800，batch 6）

单变量敏感性（每次只偏离冠军一个维度），30 epoch 只作**排序**、不作认定：

| run | 变更维度 | 值 | 假说 |
|---|---|---|---|
| topk16 | `stal_topk_small` | 16 | 13 是否已达正样本上限，继续加是否还有增益 |
| topk20 | `stal_topk_small` | 20 | 正样本上限探底 |
| expand2 | `stal_expand` | 2.0 | 更大候选区域（更多低质量锚点） |
| expand05 | `stal_expand` | 0.5 | 更紧区域，减少低质量锚点 |
| beta4 | `tal_beta` | 4.0 | 降低 IoU 权重——小框 IoU 噪声大，β=6 过度压制 |
| alpha1 | `tal_alpha` | 1.0 | 提高分类权重，弱化 IoU 主导 |

## Phase 2 — 120-epoch 复现认定

Phase 1 排名前 1–2 名跑满 120 epoch（协议见 `A2/P0/tiered_eval.py` + `P0/README.md`），
只有 120-epoch 结果才能认定「APs 提升」。

## Phase 3 — 交互 / warmup（stretch，视 Phase 1 结果）

- α×β 组合消融（如 β∈{4,6} × α∈{0.5,1.0}）。
- warmup 曲线（STAL 参数随 epoch 渐进放开）——需新增机制，非纯配置可扫。
- 第二数据集 或 seg/pose（分叉决策，延后）。

## 复现命令（Phase 1）

```bash
python A2/P0/baseline_train.py --epochs 30 --imgsz 800 --batch 6 --device 0 --amp false --seed 0 \
  --stal-mode adaptive --stal-min-positive true --stal-topk-small 16 \
  --project runs/a2/p2 --name visdrone-p2-topk16-v01n

# 其余按上表替换 --stal-topk-small / --stal-expand / --tal-beta / --tal-alpha
```

## 风险 / 口径

- 30-epoch 仅筛参；close_mosaic=10 → 仅 ~20 个 post-mosaic epoch，APs 排名有噪声，Phase 2 必须复现。
- α/β 是全局 TAL 参数，改它会同时影响 medium/large 档；主指标仍盯 APs，但会一并报 APm/APl 观察溢出。
- 训练期 / 评测期面积口径不同（见 `stal-eval-protocol`），勿混用。

# P0 — VisDrone2019-DET 基线：分档指标 + 每 epoch 正样本统计

> 任务：STAL 面积感知标签分配。P0（保底）= 真实 VisDrone 基线 + 小/中/大分档 AP + 每 epoch 正样本统计，
> 作为 P1 面积感知 on/off 对照的底座。

## 交付文件

| 文件 | 作用 |
|---|---|
| `A2/P0/prepare_visdrone.py` | 下载/复用 VisDrone2019-DET → 校验 split 计数 → 写本地 `datasets/VisDrone.yaml`（绝对路径） |
| `A2/P0/baseline_train.py` | yolo11n 从零训练 + 每 epoch 正样本统计回调 |
| `A2/P0/tiered_eval.py` | 统一分档评测：tiny/small/medium/large AP（COCO 面积口径，检测按其命中 GT 的面积分档） |
| `ultralytics/utils/loss.py` | 唯一核心改动：`v8DetectionLoss` 增加 `fg_count` / `fg_count_by_stride` 计数 |

## 复现命令

```bash
# 1. 数据（幂等：已存在则跳过下载，只校验 + 写本地 yaml）
python A2/P0/prepare_visdrone.py

# 2. 链路冒烟（1 epoch，验证 loss 计数 + 回调 + 分档评测）
python A2/P0/baseline_train.py --epochs 1 --imgsz 320 --batch 8 --device 0 --name visdrone-smoke-1ep
python A2/P0/tiered_eval.py --weights runs/a2/p0/visdrone-smoke-1ep/weights/last.pt --imgsz 320 --device 0

# 3. 全量基线（100 epoch，imgsz 640）
python A2/P0/baseline_train.py --epochs 100 --imgsz 640 --batch 16 --device 0
python A2/P0/tiered_eval.py --weights runs/a2/p0/visdrone-baseline-yolo11n/weights/best.pt --imgsz 640 --device 0
```

## 数据与环境

- 数据落在用户 settings 指定的 `D:\yolo\datasets\VisDrone`（非仓库 `datasets/`）。
- Split：train **6471** / val **548** / test-dev **1610**，10 类（VisDrone2019-DET 标准）。
- 环境：RTX 5070 Ti Laptop 12GB，torch 2.13.0+cu132，ultralytics 8.4.101。

## 指标口径

- **分档阈值**（GT 框像素面积，可改 `tiered_eval.py` 的 `DEFAULT_TIERS`）：
  - `tiny` < 16²，`small` 16²–32²，`medium` 32²–96²，`large` ≥ 96²，另有 `all`。
  - 加 `tiny` 是因为 VisDrone 对象普遍偏小，纯 COCO 口径下绝大多数会落 `small`，缺区分度。
- **正样本统计**（`results.csv` 新增列，随训练自动写入）：
  - `train/fg_sum`：每 epoch 正样本（前景锚点）总数。
  - `train/fg_s8` / `train/fg_s16` / `train/fg_s32`：按 FPN 层级（stride 8/16/32）分桶的正样本数，
    即小/中/大目标的面积代理 —— 直接对齐 STAL 的「小目标正样本演化」叙事。

## 关键实现说明

- **正样本计数**：`v8DetectionLoss.get_assigned_targets_and_loss` 在 assigner 返回后累加 `fg_mask.sum()`，
  并按 `stride_tensor` 分桶；数值损失路径零改动。
- **上抛**：`model.add_callback("on_train_epoch_end", fn)` 把计数写入 `trainer.lr`（自动进 `results.csv`），
  `on_fit_epoch_end` 在 validate 之后清零（validation 复用训练 criterion，不清零会污染下一 epoch）。
- **分档评测**：standalone 脚本，COCO 语义 —— 检测按其命中 GT 的面积分档；未命中任何 GT 的检测在各档都算 FP；
  命中非本档 GT 的检测在该档忽略。总 mAP 与分档 AP 出自同一份代码，口径自洽。

## 风险与降级

- 下载受限 → `check_det_dataset` 走官方镜像，失败可改 `scripts/download_visdrone_dataset.sh hf`（hf-mirror）。
- 训练抖动 → 默认 warmup 3 epoch；不稳再调 `warmup_epochs`。
- 分档评测与训练期 `results.csv` 的 mAP 是两套 NMS 后处理，数值可能略有差异；对比实验统一用 `tiered_eval.py`。

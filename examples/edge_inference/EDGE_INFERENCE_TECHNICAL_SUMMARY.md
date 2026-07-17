# YOLO-Master-EsMoE-N 垂类模型边缘端推理加速 — 技术总结

> 对应 Issue: [Tencent/YOLO-Master#51](https://github.com/Tencent/YOLO-Master/issues/51)

## 概述

本项目完成了 YOLO-Master-EsMoE-N 在 VisDrone 垂类场景上的完整边缘端推理加速流水线：训练 → 导出 → 边缘推理 → 一致性验证 → Benchmark。

| 阶段 | 交付物 | 路径 |
|:---|:---|:---|
| 训练复现 | 可复现训练脚本 + 结果对比 | [`scripts/reproduce/`](../../scripts/reproduce/) |
| 模型导出 | ONNX + MNN + NCNN 导出脚本 | [`examples/YOLO-Master-EsMoE-VisDrone-Edge/scripts/`](../YOLO-Master-EsMoE-VisDrone-Edge/scripts/) |
| 边缘推理 | Python/C++ 多后端推理代码 | [`examples/YOLO-Master-EsMoE-VisDrone-Edge/`](../YOLO-Master-EsMoE-VisDrone-Edge/) |
| 一致性验证 | PyTorch vs 导出模型 mAP 对比 | 各示例 README |
| Benchmark | 延迟/FPS 多格式对比 | 各示例 README |
| 技术深潜 | INT8 量化、数值一致性分析 | [`examples/YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/TECHNICAL_REPORT.md`](../YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/TECHNICAL_REPORT.md) |

---

## 1. 训练复现

### 数据集

- **VisDrone2019-DET**: 密集航拍，10类，6471 训练 / 548 验证图像
- **SKU-110K**: 密集零售商品，1类，~11K 训练 / ~3K 验证图像

### 模型

| 模型 | 配置文件 | 参数量 |
|:---|:---|:---|
| YOLO-Master-v0.1-N | `v0_1/det/yolo-master-n.yaml` | 7.55M |
| YOLO-Master-EsMoE-N | `v0/det/yolo-master-n.yaml` | 2.69M |

### 运行命令

```bash
# VisDrone
python scripts/reproduce/reproduce_visdrone.py --model v0.1-N  --epochs 300 --batch 64
python scripts/reproduce/reproduce_visdrone.py --model EsMoE-N --epochs 300 --batch 64 --no-sparse-eval

# SKU-110K
python scripts/reproduce/reproduce_sku110k.py --model v0.1-N  --epochs 300 --batch 64
python scripts/reproduce/reproduce_sku110k.py --model EsMoE-N --epochs 300 --batch 64 --no-sparse-eval
```

### 训练结果摘要

| 模型 | 数据集 | mAP50 | mAP50-95 | 预训练权重 |
|:---|:---|:---|:---|:---|
| v0.1-N | VisDrone | 0.344 | 0.201 | [下载](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-v01-n-visdrone.pt) |
| EsMoE-N | VisDrone | 0.350 | 0.203 | [下载](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-esmoe-n-visdrone.pt) |
| v0.1-N | SKU-110K | 0.906 | 0.582 | [下载](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-v01-n-sku110k.pt) |
| EsMoE-N | SKU-110K | 0.904 | 0.583 | [下载](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-esmoe-n-sku110k.pt) |

> 详细训练日志、W&B 曲线、已知问题说明：[`scripts/reproduce/README.md`](../../scripts/reproduce/README.md)
>
> ⚠️ **重要：** EsMoE-N 默认使用 sparse inference，验证 mAP 会崩溃。必须使用 `--no-sparse-eval` 标志切换到 dense evaluation。

---

## 2. 模型导出

### ONNX

```python
from ultralytics import YOLO
model = YOLO("EsMoE-N_VisDrone.pt")
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
```

- opset=12 确保 ONNX Runtime / NCNN / MNN 三方兼容
- `simplify=True` 自动运行 onnxsim
- 类别名、imgsz、stride 嵌入 ONNX metadata，C++ 运行时自动读取

### NCNN (pnnx)

```bash
yolo export model=EsMoE-N_VisDrone.pt format=ncnn imgsz=640
```

### MNN

```bash
mnnconvert -f ONNX --modelFile esmoe_n_visdrone_sim.onnx --MNNModel esmoe_n_visdrone.mnn --bizCode edge
```

> 详细导出说明：[`examples/YOLO-Master-EsMoE-VisDrone-Edge/README.md`](../YOLO-Master-EsMoE-VisDrone-Edge/README.md)

---

## 3. 边缘推理

### C++ (ONNX Runtime + NCNN + MNN，单一可执行文件)

```bash
# 构建 (Linux)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DNCNN_ROOT=/path/to/ncnn
cmake --build build -j

# 推理
./build/yolomaster_edge --model esmoe_n_visdrone_sim.onnx --source image.jpg --conf 0.25 --out out
```

- 自动检测模型格式（ONNX/NCNN/MNN）和类别名
- 支持 CPU (FP32) 和 CUDA GPU 加速
- 垂类后处理：aspect-ratio-preserving letterbox + per-class multi-label NMS
- 跨平台：Linux x86_64 + Windows 10/11

### C++ VisDrone 专项推理

```bash
cmake -S examples/YOLO-Master-EsMoE-VisDrone-Edge/cpp -B build \
  -DONNXRUNTIME_ROOT_DIR=$(pwd)/third_party/onnxruntime
cmake --build build -j
./build/yolo_edge exports/best_recal.onnx image.jpg --nc 10 --bench 200
```

### Python 多后端

```python
cd examples/YOLO-Master-EsMoE-VisDrone-Edge/python
python eval_consistency.py \
    --pytorch best_recal.pt --onnx exports/best_recal.onnx --mnn exports/best_recal.mnn \
    --imgsz 640 --num-classes 10 --device cuda:0
```

- ONNX Runtime / MNN / NCNN 统一 Python 接口
- 面积自适应 NMS（小目标低 conf，大目标高 conf）
- 548 张验证集图像全量一致性验证

---

## 4. 一致性验证结果

**在 548 张 VisDrone 验证集上，多后端 mAP 与 PyTorch 原版对齐：**

| 后端 | mAP50 | mAP50-95 | ΔmAP50-95 vs PyTorch | 目标 |
|:---|:---|:---|:---|:---:|
| PyTorch (native) | 12.09% | 6.00% | — | — |
| ONNX (ONNX Runtime, CPU) | 12.08% | 5.99% | **−0.004%** | <0.5% ✅ |
| ONNX (ONNX Runtime, CUDA) | — | 0.2033¹ | −0.03% | <0.5% ✅ |
| MNN (CPU) | 12.08% | 5.99% | **−0.006%** | <0.5% ✅ |
| NCNN (CPU) | — | 0.2034¹ | −0.02% | <0.5% ✅ |
| INT8 (CPU) | — | 0.1952¹ | −0.84% | <1.0% ✅ |

> ¹ 使用完整 VisDrone 训练的模型（mAP50-95 baseline = 0.2036）

---

## 5. 边缘 Benchmark

| 后端 / 平台 | 延迟 (ms) | FPS | 备注 |
|:---|---:|---:|:---|
| C++ ONNX Runtime — Linux x86_64 | 40.0 | 25.0 | 4 threads |
| C++ ONNX Runtime — Linux x86_64 | 53.0 | 18.9 | VisDrone-Edge 示例 |
| C++ ONNX Runtime — Windows x86_64 | 83.6 | 12.0 | MSVC 19.44 |
| C++ ONNX Runtime — CUDA (H200) | 7.8 | ~128 | GPU 加速 |
| MNN (Python, CPU) | 56.1 | 17.8 | |
| NCNN (CPU) | ~80 | ~12.5 | |
| INT8 (CPU) | 137 | 7.2 | INT8 在 CPU 上无加速优势 |
| TensorRT FP16 — Jetson Orin Nano 4GB | 27.8 | 35.7 | FP16 engine |

---

## 6. 关键技术发现

### 6.1 MoE Sparse/Dense 一致性

EsMoE-N 的 `ES_MOE` 模块在 training 使用 `_dense_forward`（所有 expert），inference 默认使用 `_sparse_forward`（pruned top-k）。这种不一致导致验证 mAP 崩溃。

- **训练侧修复：** `--no-sparse-eval` flag 强制 dense evaluation（见 [`scripts/reproduce/README.md`](../../scripts/reproduce/README.md)）
- **导出侧修复：** `patch_dense_forward.py` 使 exported graph 与 eager eval 一致（见 [`examples/YOLO-Master-EsMoE-VisDrone-Edge/README.md`](../YOLO-Master-EsMoE-VisDrone-Edge/README.md)）

### 6.2 INT8 量化中的分类头坍缩

全图 INT8 量化会输出正确的 tensor shape — 但检测不到任何目标。根因：分类分支的 logits 经过 sigmoid 后，小正值落在 INT8 量化步长之下被舍入为 0。解决：混合精度（分类头保留 FP32）。

详见 [TECHNICAL_REPORT.md §3](../YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/TECHNICAL_REPORT.md)

### 6.3 NCNN MoE 限制

pnnx 无法 lower MoE routing 的 `topk` 和比较操作。解决方案：使用 dense (full-softmax) 路径导出，但会损失精度。生产环境推荐使用 ONNX 或 MNN。

---

## 7. 快速复现指南

### 从头到尾跑通 VisDrone 边缘推理

```bash
# 1. 下载预训练权重
wget https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-esmoe-n-visdrone.pt

# 2. BN 重校准（修复 sparse-eval BN drift）
python examples/YOLO-Master-EsMoE-VisDrone-Edge/scripts/recalibrate_bn.py \
    --src yolo-master-esmoe-n-visdrone.pt --dst best_recal.pt --device 0

# 3. 导出 ONNX + MNN
python examples/YOLO-Master-EsMoE-VisDrone-Edge/scripts/export_models.py \
    --model best_recal.pt --imgsz 640

# 4. 一致性验证（548 张验证图像）
cd examples/YOLO-Master-EsMoE-VisDrone-Edge/python
python eval_consistency.py --pytorch ../../best_recal.pt \
    --onnx ../exports/best_recal.onnx --mnn ../exports/best_recal.mnn \
    --imgsz 640 --num-classes 10 --device cuda:0

# 5. Benchmark
python benchmark.py --onnx ../exports/best_recal.onnx --iters 200

# 6. C++ 推理 (Linux)
cd ../cpp && bash scripts/setup_ort.sh
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=$(pwd)/third_party/onnxruntime
cmake --build build -j
./build/yolo_edge ../../exports/best_recal.onnx /path/to/image.jpg --nc 10 --bench 200
```

---

## 8. 已知限制

| 限制 | 说明 | 替代方案 |
|:---|:---|:---|
| **NCNN MoE 支持** | pnnx 无法 lower MoE routing 算子 | 使用 ONNX 或 MNN |
| **GitHub Discussion** | 仓库 Discussion 功能未启用 | 本技术报告替代 |
| **SKU-110K 边缘示例** | 边缘推理示例聚焦 VisDrone | 训练复现脚本已支持 SKU-110K |
| **ARM64 测试** | Jetson Orin 已验证；通用 ARM64 需交叉编译 | Dockerfile.arm64 提供 |

---

## 9. 相关链接

- 训练脚本 & W&B: [`scripts/reproduce/`](../../scripts/reproduce/)
- VisDrone 边缘部署: [`examples/YOLO-Master-EsMoE-VisDrone-Edge/`](../YOLO-Master-EsMoE-VisDrone-Edge/)
- 通用 C++ 多后端运行时: [`examples/YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/`](../YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/)
- 技术深潜报告: [`TECHNICAL_REPORT.md`](../YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/TECHNICAL_REPORT.md)
- 预训练权重: [Releases](https://github.com/skywalker-lt/YOLO-Master/releases)
- 边缘部署仓库: [yolo-master-edge](https://github.com/skywalker-lt/yolo-master-edge)

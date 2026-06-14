# PCB_Anomaly_EdgeAI

面向边缘部署的 PCB 少样本异常检测与缺陷定位。

本项目基于 DeepPCB 数据集，完成 PCB 缺陷分类 baseline、AutoEncoder 异常检测、异常热力图生成、推理延迟测试和 ONNX 导出，为 STM32H7 / PYNQ-Z2 / ESP32-S3 等边缘平台部署做准备。

## 项目结构

```text
PCB_Anomaly_EdgeAI/
├── data/
│   ├── raw/DeepPCB/              # 原始 DeepPCB 数据集，不纳入 Git
│   └── processed/                # 脚本生成的数据集，不纳入 Git
├── datasets/
│   └── pcb_dataset.py            # PyTorch Dataset 和 DataLoader
├── deployment/
│   └── onnx/                     # ONNX 导出结果，不纳入 Git
├── docs/
│   └── stage_plan.md             # 阶段计划
├── models/
│   ├── resnet_classifier.py      # ResNet18 分类模型
│   ├── mobilenet_classifier.py   # MobileNetV2 分类模型
│   ├── autoencoder.py            # ConvAutoEncoder 异常检测模型
│   └── tiny_autoencoder.py       # TinyAutoEncoder 轻量异常检测模型
├── paper/
│   └── paper.md                  # 论文初稿
├── results/                      # 训练、评估、图表输出，不纳入 Git
├── checkpoints/                  # 模型权重，不纳入 Git
└── scripts/
    ├── inspect_deeppcb.py        # DeepPCB 数据检查
    ├── prepare_data.py           # 阶段 2 数据整理
    ├── train_classifier.py       # 分类模型训练
    ├── evaluate_classifier.py    # 分类模型评估
    ├── train_autoencoder.py      # AutoEncoder 异常检测训练
    ├── evaluate_anomaly.py       # 异常检测评估
    ├── predict_heatmap.py        # 异常热力图生成
    ├── measure_latency.py        # 推理延迟测试
    └── export_onnx.py            # ONNX 导出
```

## 数据集

使用数据集：DeepPCB

原始数据路径：

```text
data/raw/DeepPCB/PCBData
```

已确认数据统计：

| 项目 | 数量 |
|---|---:|
| trainval | 1000 |
| test | 500 |
| `_test` 图像 | 1500 |
| `_temp` 图像 | 1501 |
| 缺陷框 | 10013 |

缺陷类别：

```text
open, short, mousebite, spur, copper, pin-hole
```

## 完整运行流程

建议在 `ai_train` 环境下运行：

```powershell
cd E:\AI_Projects\PCB_Anomaly_EdgeAI
conda activate ai_train
```

1. 检查 DeepPCB 数据：

```powershell
python scripts\inspect_deeppcb.py
```

2. 整理数据，生成分类数据集和异常检测数据集：

```powershell
python scripts\prepare_data.py --patch_size 128 --padding_ratio 0.5 --overwrite
```

3. 训练分类模型：

```powershell
python scripts\train_classifier.py --model resnet18 --epochs 3 --batch_size 16 --image_size 224 --pretrained
python scripts\train_classifier.py --model mobilenet_v2 --epochs 3 --batch_size 16 --image_size 224 --pretrained
```

4. 评估分类模型：

```powershell
python scripts\evaluate_classifier.py --model resnet18
python scripts\evaluate_classifier.py --model mobilenet_v2
```

5. 训练 AutoEncoder 异常检测模型：

```powershell
python scripts\train_autoencoder.py --model autoencoder --epochs 5 --batch_size 32 --image_size 128
python scripts\train_autoencoder.py --model tiny_ae --epochs 5 --batch_size 32 --image_size 96
```

6. 评估异常检测模型：

```powershell
python scripts\evaluate_anomaly.py --model autoencoder --image_size 128
python scripts\evaluate_anomaly.py --model tiny_ae --image_size 96
```

7. 生成异常热力图：

```powershell
python scripts\predict_heatmap.py --model autoencoder --image_size 128 --num_samples 16
```

8. 测量推理延迟：

```powershell
python scripts\measure_latency.py --task classifier --model resnet18 --image_size 224
python scripts\measure_latency.py --task anomaly --model tiny_ae --image_size 96
```

9. 导出 ONNX：

```powershell
python scripts\export_onnx.py --task anomaly --model tiny_ae --image_size 96
python scripts\export_onnx.py --task classifier --model mobilenet_v2 --image_size 224
```

如果提示缺少 ONNX：

```powershell
pip install onnx
```

## 文件作用与阶段结果占位

| 阶段 | 文件 | 作用 | 主要输出 | 当前结果 |
|---|---|---|---|---|
| 阶段 1 | `scripts/inspect_deeppcb.py` | 检查 DeepPCB 原始数据结构、split、图像和标注统计 | 控制台统计、阶段记录 | 已完成数据检查 |
| 阶段 2 | `scripts/prepare_data.py` | 从 bbox 裁剪分类 patch 和 normal/anomaly patch | `data/processed/pcb_cls`、`data/processed/pcb_anomaly`、`results/tables/stage2_prepare_summary.csv` | 已生成，缺陷框 10013 |
| 阶段 2 | `datasets/pcb_dataset.py` | 提供分类和异常检测 DataLoader | PyTorch DataLoader | 已完成 |
| 阶段 3 | `models/resnet_classifier.py` | 构建 ResNet18 分类 baseline | 分类模型结构 | 已完成 |
| 阶段 3 | `models/mobilenet_classifier.py` | 构建 MobileNetV2 分类 baseline | 分类模型结构 | 已完成 |
| 阶段 3 | `scripts/train_classifier.py` | 训练分类模型并保存日志、曲线、checkpoint | `checkpoints/classifier/*.pth`、`results/classifier/*_training_log.csv`、`results/curves/*` | TODO：正式实验结果汇总 |
| 阶段 3 | `scripts/evaluate_classifier.py` | 在 test 集评估分类模型 | classification report、混淆矩阵、样例预测图 | TODO：正式指标汇总 |
| 阶段 4 | `models/autoencoder.py` | 标准 ConvAutoEncoder 异常检测模型 | 模型结构 | 已完成 |
| 阶段 4 | `models/tiny_autoencoder.py` | 轻量 TinyAutoEncoder，面向边缘部署 | 模型结构 | 已完成 |
| 阶段 4 | `scripts/train_autoencoder.py` | 只用 normal patch 训练重建模型 | `checkpoints/anomaly/*.pth`、训练日志、重建样例 | TODO：正式训练曲线汇总 |
| 阶段 4 | `scripts/evaluate_anomaly.py` | 基于 reconstruction error 做图像级异常检测评估 | AUROC、AP、F1、ROC、score histogram | TODO：正式指标汇总 |
| 阶段 5 | `scripts/predict_heatmap.py` | 生成像素级 reconstruction error 热力图 | `results/heatmaps/*_heatmap_samples.png` | TODO：人工可视化分析 |
| 阶段 6 | `scripts/measure_latency.py` | 测量当前 PC/GPU 推理延迟 | `results/tables/latency_results.csv` | TODO：多模型延迟对比 |
| 阶段 6 | `scripts/export_onnx.py` | 导出 ONNX 模型 | `deployment/onnx/*.onnx` | TODO：安装 ONNX 后导出 |

## Git 注意事项

以下目录和大文件不纳入 Git：

```text
data/raw/
data/processed/
results/
checkpoints/
*.pth
*.onnx
*.tflite
```

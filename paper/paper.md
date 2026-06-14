# 面向边缘部署的 PCB 少样本异常检测与缺陷定位方法研究

## 摘要

印制电路板（Printed Circuit Board, PCB）缺陷检测是电子制造质量控制中的关键环节。传统监督式视觉检测方法依赖大量带标注缺陷样本，而实际工业场景中缺陷样本稀缺、类别分布不均衡，并且边缘端部署设备通常受到计算资源、存储空间和功耗限制。本文围绕 DeepPCB 数据集，研究面向边缘部署的 PCB 少样本异常检测与缺陷定位方法。

本文首先整理 DeepPCB 原始数据，构建监督式缺陷分类数据集和基于 normal/anomaly patch 的异常检测数据集；随后建立 ResNet18 与 MobileNetV2 监督分类 baseline；进一步使用 AutoEncoder 与 Tiny AutoEncoder 对正常样本进行重建学习，并利用重建误差进行图像级异常检测和像素级热力图定位；最后通过推理延迟测试和 ONNX 导出，为 STM32H7、PYNQ-Z2、ESP32-S3 等边缘平台部署做准备。

当前已确认数据集统计包括：trainval 样本 1000，test 样本 500，`_test` 图像 1500，`_temp` 图像 1501，缺陷框 10013，缺陷类别为 open、short、mousebite、spur、copper、pin-hole。分类准确率、异常检测 AUROC、推理延迟和边缘端部署结果将在后续实验完成后补充，当前以 TODO 标记。

关键词：PCB 缺陷检测；少样本异常检测；AutoEncoder；缺陷定位；边缘部署；ONNX

## 1 绪论

### 1.1 研究背景

PCB 是电子产品的重要基础部件，其制造质量直接影响设备可靠性。PCB 生产过程中可能出现 open、short、mousebite、spur、copper、pin-hole 等缺陷。若缺陷未能及时检出，可能导致后续装配、测试或现场运行阶段的故障。

近年来，深度学习方法在工业视觉检测中取得较好效果。然而，PCB 缺陷检测仍面临以下问题：

- 缺陷样本数量有限，难以覆盖所有异常形态。
- 类别标注成本较高，监督式分类模型依赖大量标注数据。
- 工业部署通常要求模型具有较低延迟和较小存储占用。
- 边缘平台算力有限，需要考虑模型轻量化和导出部署。

### 1.2 研究目标

本文目标是构建一个面向边缘部署的 PCB 少样本异常检测与缺陷定位流程，具体包括：

- 基于 DeepPCB 数据集构建分类数据集和异常检测数据集。
- 建立 ResNet18 与 MobileNetV2 监督分类 baseline。
- 建立 AutoEncoder 与 Tiny AutoEncoder 异常检测模型。
- 基于重建误差生成异常热力图，实现缺陷定位可视化。
- 测量模型推理延迟并导出 ONNX，为后续边缘部署做准备。

### 1.3 本文结构

第 2 章介绍相关技术；第 3 章说明数据集与实验设置；第 4 章给出监督式 PCB 缺陷分类 baseline；第 5 章介绍少样本异常检测与缺陷定位方法；第 6 章讨论轻量化模型与边缘部署；第 7 章总结全文并展望后续工作。

## 2 相关技术

### 2.1 PCB 缺陷检测

PCB 缺陷检测通常包括图像采集、模板对齐、缺陷定位、缺陷分类等环节。传统方法依赖图像差分、阈值分割、形态学处理和人工特征；深度学习方法则可通过卷积神经网络学习更加鲁棒的缺陷表征。

### 2.2 监督式图像分类

监督式图像分类通过带类别标签的样本训练模型。本文选择 ResNet18 和 MobileNetV2 作为 baseline。ResNet18 具有残差连接，训练稳定；MobileNetV2 使用轻量化结构，更适合边缘端部署。

当前分类实验结果：

| 模型 | Test Accuracy | Macro F1 | 备注 |
|---|---:|---:|---|
| ResNet18 | TODO | TODO | TODO |
| MobileNetV2 | TODO | TODO | TODO |

### 2.3 少样本异常检测

少样本异常检测通常只依赖正常样本建模。当模型学习到正常样本分布后，异常样本在重建误差、特征距离或密度估计上会表现出较大偏差。本文使用 AutoEncoder 进行正常图像重建学习，并将 reconstruction error 作为 anomaly score。

### 2.4 模型轻量化与边缘部署

边缘部署需要关注模型参数量、计算量、推理延迟、模型格式和硬件兼容性。本文实现 Tiny AutoEncoder，并预留 ONNX 导出流程，为后续量化和端侧推理做准备。

## 3 数据集与实验设置

### 3.1 DeepPCB 数据集

本文使用 DeepPCB 数据集。项目中的原始数据路径为：

```text
data/raw/DeepPCB/PCBData
```

数据集中每个样本包含：

- `_test.jpg`：带缺陷图像。
- `_temp.jpg`：无缺陷模板图像。
- `.txt`：缺陷框标注文件。

标注格式为：

```text
x1 y1 x2 y2 class_id
```

类别映射为：

| class_id | 类别 |
|---:|---|
| 1 | open |
| 2 | short |
| 3 | mousebite |
| 4 | spur |
| 5 | copper |
| 6 | pin-hole |

### 3.2 已确认数据统计

| 项目 | 数量 |
|---|---:|
| trainval.txt 行数 | 1000 |
| test.txt 行数 | 500 |
| `_test` 图像数量 | 1500 |
| `_temp` 图像数量 | 1501 |
| 缺陷框总数 | 10013 |

### 3.3 数据预处理

本文将 DeepPCB 数据整理为两个数据集：

监督分类数据集：

```text
data/processed/pcb_cls/
├── train/
├── val/
└── test/
```

异常检测数据集：

```text
data/processed/pcb_anomaly/
├── train/normal/
├── val/normal/
├── val/anomaly/
├── test/normal/
└── test/anomaly/
```

分类数据集从 `_test` 图像中根据 bbox 裁剪缺陷 patch。异常检测数据集中，normal patch 来自 `_temp` 图像同一 bbox 位置，anomaly patch 来自 `_test` 图像同一 bbox 位置。训练 split 只使用 normal patch。

### 3.4 实验环境

当前项目环境：

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| Python 环境 | conda `ai_train` |
| PyTorch | 2.5.1+cu121 |
| CUDA | 可用 |
| GPU | TODO |
| CPU | TODO |
| 内存 | TODO |

## 4 监督式 PCB 缺陷分类 baseline

### 4.1 模型结构

本文实现两个监督式分类 baseline：

- ResNet18：使用 torchvision ResNet18，并将最后全连接层替换为 6 类输出。
- MobileNetV2：使用 torchvision MobileNetV2，并将 classifier 输出层替换为 6 类输出。

### 4.2 训练设置

训练使用交叉熵损失和 Adam 优化器。输入图像 resize 到 224x224，并使用 ImageNet mean/std 归一化。训练阶段使用随机水平翻转、随机旋转和亮度/对比度扰动。

训练参数：

| 参数 | 值 |
|---|---|
| image_size | 224 |
| batch_size | TODO |
| epochs | TODO |
| learning rate | TODO |
| pretrained | TODO |

### 4.3 实验结果

分类模型在 test 集上的结果如下。当前不虚构实验结果，待完整训练和评估后补充。

| 模型 | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| ResNet18 | TODO | TODO | TODO | TODO |
| MobileNetV2 | TODO | TODO | TODO | TODO |

混淆矩阵与样例预测图保存于：

```text
results/classifier/
```

## 5 PCB 少样本异常检测与缺陷定位

### 5.1 AutoEncoder 异常检测方法

AutoEncoder 只使用 normal patch 训练。训练目标是最小化输入图像与重建图像之间的均方误差：

```text
MSE = mean((x - recon)^2)
```

测试时，将每张图像的平均重建误差作为 anomaly score。若 anomaly score 高于阈值，则判定为异常。

### 5.2 模型结构

本文实现两个异常检测模型：

- ConvAutoEncoder：用于 3x128x128 输入，采用卷积编码器和反卷积解码器。
- TinyAutoEncoder：轻量化模型，用于 3x96x96 或 3x64x64 输入，参数量更小，便于 ONNX 导出和边缘部署。

模型参数量：

| 模型 | 参数量 |
|---|---:|
| ConvAutoEncoder | 758115 |
| TinyAutoEncoder | 37023 |

### 5.3 异常检测指标

本文使用以下指标评估图像级异常检测性能：

- AUROC
- Average Precision
- best F1
- best threshold
- precision
- recall

当前实验结果：

| 模型 | AUROC | Average Precision | best F1 | threshold |
|---|---:|---:|---:|---:|
| ConvAutoEncoder | TODO | TODO | TODO | TODO |
| TinyAutoEncoder | TODO | TODO | TODO | TODO |

### 5.4 缺陷定位热力图

像素级误差图定义为：

```text
error_map = mean((image - recon)^2, dim=channel)
```

对 `error_map` 进行 min-max normalize 后，将其以 colormap 形式叠加到原图，得到异常热力图。热力图结果保存于：

```text
results/heatmaps/
```

## 6 轻量化模型与边缘部署

### 6.1 推理延迟测试

本文实现推理延迟测试脚本，用于在当前 PC/GPU 上测量模型 latency 和 FPS，作为后续边缘端部署前的基线。

当前延迟测试结果：

| Task | 模型 | Image Size | Batch Size | Latency ms | FPS | Model Size MB |
|---|---|---:|---:|---:|---:|---:|
| classifier | ResNet18 | 224 | 1 | TODO | TODO | TODO |
| classifier | MobileNetV2 | 224 | 1 | TODO | TODO | TODO |
| anomaly | ConvAutoEncoder | 128 | 1 | TODO | TODO | TODO |
| anomaly | TinyAutoEncoder | 96 | 1 | TODO | TODO | TODO |

### 6.2 ONNX 导出

本文实现 ONNX 导出脚本，导出文件路径为：

```text
deployment/onnx/{task}_{model}_{image_size}.onnx
```

当前 ONNX 导出状态：

| 模型 | ONNX 文件 | 状态 |
|---|---|---|
| ResNet18 | TODO | TODO |
| MobileNetV2 | TODO | TODO |
| ConvAutoEncoder | TODO | TODO |
| TinyAutoEncoder | TODO | TODO |

### 6.3 边缘平台部署计划

后续计划面向以下平台进行部署验证：

- STM32H7：关注模型压缩、量化和内存占用。
- PYNQ-Z2：关注 FPGA 加速与吞吐。
- ESP32-S3：关注超轻量模型和低功耗推理。

当前部署结果：TODO。

## 7 总结与展望

本文围绕 DeepPCB 数据集，构建了 PCB 缺陷分类、少样本异常检测、缺陷热力图定位、推理延迟测试和 ONNX 导出流程。已完成数据集检查、数据整理、分类 baseline、AutoEncoder 模型、异常检测评估脚本和边缘部署准备脚本。

后续工作包括：

- 完整训练并系统比较 ResNet18 与 MobileNetV2 分类性能。
- 完整评估 ConvAutoEncoder 与 TinyAutoEncoder 异常检测性能。
- 结合热力图进行缺陷定位质量分析。
- 进行模型量化、剪枝和端侧部署实验。
- 在 STM32H7、PYNQ-Z2、ESP32-S3 上验证实际推理延迟和资源占用。

本文当前仍处于实验开发阶段，未完成或未确认的实验数字均以 TODO 标记。

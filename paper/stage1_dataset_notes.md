\# Stage 1: DeepPCB 数据集调研与初步检查



\## 1. 数据集选择



本项目选择 DeepPCB 作为 PCB 缺陷识别与少样本异常检测研究的数据集。该数据集包含成对的 PCB 图像，包括无缺陷模板图像和带缺陷测试图像，并提供缺陷位置标注。



DeepPCB 数据集包含 6 类常见 PCB 缺陷：



\- open

\- short

\- mousebite

\- spur

\- copper

\- pin-hole



该数据集既可以用于监督式 PCB 缺陷识别 baseline，也可以用于构造少样本异常检测任务。



\## 2. 数据集目录



当前数据集存放路径：



```text

E:\\AI\_Projects\\PCB\_Anomaly\_EdgeAI\\data\\raw\\DeepPCB


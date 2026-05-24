from pathlib import Path
from collections import Counter
import random
import re


# DeepPCB 类别映射
# 1: open
# 2: short
# 3: mousebite
# 4: spur
# 5: copper / spurious copper
# 6: pin-hole
CLASS_ID_TO_NAME = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}


def find_deeppcb_root(project_root: Path) -> Path:
    """
    自动寻找 DeepPCB 数据集目录。

    常见情况：
    1. data/raw/DeepPCB/PCBData
    2. data/raw/DeepPCB-master/PCBData
    3. data/raw/DeepPCB/DeepPCB-master/PCBData
    """
    candidates = [
        project_root / "data" / "raw" / "DeepPCB",
        project_root / "data" / "raw" / "DeepPCB-master",
        project_root / "data" / "raw" / "DeepPCB" / "DeepPCB-master",
    ]

    for candidate in candidates:
        if (candidate / "PCBData").exists():
            return candidate

    raw_dir = project_root / "data" / "raw"

    if not raw_dir.exists():
        raise FileNotFoundError(
            "没有找到 data/raw 目录，请确认你当前位于 PCB_Anomaly_EdgeAI 项目目录。"
        )

    for p in raw_dir.rglob("PCBData"):
        if p.is_dir():
            return p.parent

    raise FileNotFoundError(
        "没有找到 DeepPCB 数据集。请确认存在 data/raw/DeepPCB/PCBData 目录。"
    )


def parse_annotation_file(txt_path: Path, max_warnings: int = 3):
    """
    读取一个 DeepPCB 标注文件。

    兼容两种格式：
    1. x1 y1 x2 y2 class_id
    2. x1,y1,x2,y2,class_id

    返回：
    [
        {
            "x1": ...,
            "y1": ...,
            "x2": ...,
            "y2": ...,
            "class_id": ...,
            "class_name": ...
        },
        ...
    ]
    """
    annotations = []
    warning_count = 0

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line_idx, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        # 同时支持逗号、空格、Tab 混合分隔
        parts = re.split(r"[,\s]+", line)

        if len(parts) != 5:
            if warning_count < max_warnings:
                print(f"[警告] 标注格式异常: {txt_path} 第 {line_idx} 行: {line}")
            warning_count += 1
            continue

        try:
            x1, y1, x2, y2, class_id = map(int, parts)
        except ValueError:
            if warning_count < max_warnings:
                print(f"[警告] 无法解析整数: {txt_path} 第 {line_idx} 行: {line}")
            warning_count += 1
            continue

        if class_id not in CLASS_ID_TO_NAME:
            if warning_count < max_warnings:
                print(f"[警告] 未知类别 ID: {txt_path} 第 {line_idx} 行: {line}")
            warning_count += 1
            continue

        # 简单检查 bbox 是否合理
        if x2 <= x1 or y2 <= y1:
            if warning_count < max_warnings:
                print(f"[警告] bbox 坐标异常: {txt_path} 第 {line_idx} 行: {line}")
            warning_count += 1
            continue

        class_name = CLASS_ID_TO_NAME[class_id]

        annotations.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class_id": class_id,
            "class_name": class_name,
        })

    return annotations


def read_split_file(split_path: Path):
    """
    读取 trainval.txt 或 test.txt。
    如果文件不存在，返回空列表。
    """
    if not split_path.exists():
        return []

    with split_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    return lines


def main():
    project_root = Path.cwd()

    print("=" * 80)
    print("DeepPCB 数据集检查脚本 - 修正版")
    print("=" * 80)
    print(f"当前项目目录: {project_root}")

    deeppcb_root = find_deeppcb_root(project_root)
    pcbdata_dir = deeppcb_root / "PCBData"

    print(f"\nDeepPCB 根目录: {deeppcb_root}")
    print(f"PCBData 目录: {pcbdata_dir}")

    trainval_txt = pcbdata_dir / "trainval.txt"
    test_txt = pcbdata_dir / "test.txt"

    trainval_lines = read_split_file(trainval_txt)
    test_lines = read_split_file(test_txt)

    print("\n[1] 检查划分文件")
    print(f"trainval.txt 是否存在: {trainval_txt.exists()}")
    print(f"test.txt 是否存在: {test_txt.exists()}")
    print(f"trainval.txt 行数: {len(trainval_lines)}")
    print(f"test.txt 行数: {len(test_lines)}")

    group_dirs = [p for p in pcbdata_dir.iterdir() if p.is_dir()]

    print("\n[2] group 文件夹数量")
    print(f"group 文件夹数量: {len(group_dirs)}")

    print("\n前几个 group 文件夹:")
    for p in group_dirs[:10]:
        print(f"  - {p.name}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    all_images = []
    all_txts = []

    for p in pcbdata_dir.rglob("*"):
        if p.is_file():
            suffix = p.suffix.lower()

            if suffix in image_extensions:
                all_images.append(p)

            elif suffix == ".txt":
                if p.name not in {"trainval.txt", "test.txt"}:
                    all_txts.append(p)

    print("\n[3] 文件数量统计")
    print(f"图片文件数量: {len(all_images)}")
    print(f"标注 txt 文件数量: {len(all_txts)}")

    # DeepPCB 常见命名：*_test 是缺陷图，*_temp 是模板图
    test_images = [p for p in all_images if "_test" in p.stem.lower()]
    temp_images = [p for p in all_images if "_temp" in p.stem.lower()]

    print(f"带缺陷测试图数量 _test: {len(test_images)}")
    print(f"无缺陷模板图数量 _temp: {len(temp_images)}")

    print("\n[4] 统计缺陷类别数量")
    class_counter = Counter()
    txt_box_counter = {}

    for txt_path in all_txts:
        annotations = parse_annotation_file(txt_path, max_warnings=0)
        txt_box_counter[txt_path] = len(annotations)

        for ann in annotations:
            class_counter[ann["class_name"]] += 1

    print("每类缺陷框数量:")

    for class_id, class_name in CLASS_ID_TO_NAME.items():
        count = class_counter[class_name]
        print(f"  {class_id} - {class_name}: {count}")

    total_boxes = sum(class_counter.values())
    print(f"总缺陷框数量: {total_boxes}")

    print("\n[5] 标注文件中缺陷框数量分布")
    box_count_distribution = Counter(txt_box_counter.values())

    for box_count, file_count in sorted(box_count_distribution.items()):
        print(f"  每个 txt 有 {box_count} 个框: {file_count} 个文件")

    print("\n[6] 随机查看 5 个标注文件")
    if len(all_txts) > 0:
        sample_txts = random.sample(all_txts, k=min(5, len(all_txts)))

        for txt_path in sample_txts:
            annotations = parse_annotation_file(txt_path)
            print("-" * 80)
            print(f"标注文件: {txt_path.relative_to(project_root)}")
            print(f"缺陷数量: {len(annotations)}")

            for ann in annotations[:5]:
                print(
                    f"  bbox=({ann['x1']},{ann['y1']},{ann['x2']},{ann['y2']}), "
                    f"class_id={ann['class_id']}, "
                    f"class_name={ann['class_name']}"
                )
    else:
        print("没有找到任何标注 txt 文件。")

    print("\n[7] 数据集可用性判断")

    pass_basic_check = (
        len(all_images) > 0
        and len(all_txts) > 0
        and total_boxes > 0
    )

    if pass_basic_check:
        print("结果: DeepPCB 数据集初步检查通过。")
        print("说明:")
        print("  1. 已找到图片文件。")
        print("  2. 已找到标注文件。")
        print("  3. 已成功解析缺陷框。")
        print("  4. 下一步可以整理分类数据和异常检测数据。")
    else:
        print("结果: 数据集可能不完整或路径不正确。")
        print("请检查 data/raw/DeepPCB/PCBData 目录。")

    print("=" * 80)


if __name__ == "__main__":
    main()
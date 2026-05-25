import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


CLASS_ID_TO_NAME = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class Annotation:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str


@dataclass(frozen=True)
class Sample:
    sample_id: str
    test_path: Path
    temp_path: Path
    annotation_path: Path
    annotations: Tuple[Annotation, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: prepare DeepPCB classification and anomaly-detection patch datasets."
    )
    parser.add_argument("--patch_size", type=int, default=128, help="Output patch size in pixels.")
    parser.add_argument(
        "--padding_ratio",
        type=float,
        default=0.5,
        help="Extra crop padding relative to bbox width and height.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio used inside official trainval split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clean generated processed datasets before preparing data.",
    )
    return parser.parse_args()


def find_pcbdata_dir(project_root: Path) -> Path:
    expected = project_root / "data" / "raw" / "DeepPCB" / "PCBData"
    if expected.exists():
        return expected

    raw_dir = project_root / "data" / "raw"
    if raw_dir.exists():
        for path in raw_dir.rglob("PCBData"):
            if path.is_dir():
                return path

    raise FileNotFoundError(
        "Could not find DeepPCB PCBData directory. Expected data/raw/DeepPCB/PCBData."
    )


def read_split_file(split_path: Path) -> List[Tuple[str, str]]:
    if not split_path.exists():
        return []

    pairs = []
    with split_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] Skip malformed split line {split_path}:{line_no}: {line}")
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def parse_annotation_file(txt_path: Path, max_warnings: int = 5) -> Tuple[Annotation, ...]:
    annotations: List[Annotation] = []
    warnings = 0

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = re.split(r"[,\s]+", line)
            if len(parts) != 5:
                if warnings < max_warnings:
                    print(f"[WARN] Bad annotation format {txt_path}:{line_no}: {line}")
                warnings += 1
                continue

            try:
                x1, y1, x2, y2, class_id = map(int, parts)
            except ValueError:
                if warnings < max_warnings:
                    print(f"[WARN] Bad annotation values {txt_path}:{line_no}: {line}")
                warnings += 1
                continue

            if class_id not in CLASS_ID_TO_NAME:
                if warnings < max_warnings:
                    print(f"[WARN] Unknown class id {txt_path}:{line_no}: {line}")
                warnings += 1
                continue

            if x2 <= x1 or y2 <= y1:
                if warnings < max_warnings:
                    print(f"[WARN] Invalid bbox {txt_path}:{line_no}: {line}")
                warnings += 1
                continue

            annotations.append(
                Annotation(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    class_id=class_id,
                    class_name=CLASS_ID_TO_NAME[class_id],
                )
            )

    return tuple(annotations)


def derive_image_pair(pcbdata_dir: Path, image_rel: str) -> Tuple[Path, Path]:
    image_path = pcbdata_dir / Path(image_rel)
    stem = image_path.stem
    suffix = image_path.suffix or ".jpg"
    parent = image_path.parent

    if stem.endswith("_test"):
        base_stem = stem[: -len("_test")]
    elif stem.endswith("_temp"):
        base_stem = stem[: -len("_temp")]
    else:
        base_stem = stem

    test_path = parent / f"{base_stem}_test{suffix}"
    temp_path = parent / f"{base_stem}_temp{suffix}"
    return test_path, temp_path


def load_sample_from_split_pair(
    pcbdata_dir: Path, image_rel: str, annotation_rel: str
) -> Optional[Sample]:
    test_path, temp_path = derive_image_pair(pcbdata_dir, image_rel)
    annotation_path = pcbdata_dir / Path(annotation_rel)

    if not test_path.exists() or not temp_path.exists() or not annotation_path.exists():
        return None

    annotations = parse_annotation_file(annotation_path)
    if not annotations:
        return None

    sample_id = str(Path(annotation_rel).with_suffix("")).replace("\\", "/")
    return Sample(
        sample_id=sample_id,
        test_path=test_path,
        temp_path=temp_path,
        annotation_path=annotation_path,
        annotations=annotations,
    )


def collect_official_split_samples(
    pcbdata_dir: Path, split_pairs: Sequence[Tuple[str, str]]
) -> Tuple[List[Sample], int]:
    samples = []
    failed = 0
    for image_rel, annotation_rel in split_pairs:
        sample = load_sample_from_split_pair(pcbdata_dir, image_rel, annotation_rel)
        if sample is None:
            failed += 1
            continue
        samples.append(sample)
    return samples, failed


def find_image_with_suffix(base_path: Path, suffix_name: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = base_path.with_name(f"{base_path.name}_{suffix_name}{ext}")
        if candidate.exists():
            return candidate
    return None


def collect_all_samples(pcbdata_dir: Path) -> List[Sample]:
    samples = []
    for annotation_path in sorted(pcbdata_dir.rglob("*.txt")):
        if annotation_path.name in {"trainval.txt", "test.txt"}:
            continue

        sample_stem = annotation_path.stem
        group_dir = annotation_path.parent.parent
        image_dir_name = annotation_path.parent.name.replace("_not", "")
        image_base = group_dir / image_dir_name / sample_stem
        test_path = find_image_with_suffix(image_base, "test")
        temp_path = find_image_with_suffix(image_base, "temp")

        if test_path is None or temp_path is None:
            continue

        annotations = parse_annotation_file(annotation_path, max_warnings=1)
        if not annotations:
            continue

        sample_id = str(annotation_path.relative_to(pcbdata_dir).with_suffix("")).replace("\\", "/")
        samples.append(
            Sample(
                sample_id=sample_id,
                test_path=test_path,
                temp_path=temp_path,
                annotation_path=annotation_path,
                annotations=annotations,
            )
        )

    return samples


def split_train_val(samples: Sequence[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1.")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    return train_samples, val_samples


def split_random_all(samples: Sequence[Sample], seed: int) -> Dict[str, List[Sample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * 0.70)
    val_count = int(total * 0.15)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def safe_prepare_output_dirs(project_root: Path, overwrite: bool) -> Tuple[Path, Path, Path]:
    processed_dir = project_root / "data" / "processed"
    cls_dir = processed_dir / "pcb_cls"
    anomaly_dir = processed_dir / "pcb_anomaly"
    tables_dir = project_root / "results" / "tables"

    if overwrite:
        for target in (cls_dir, anomaly_dir):
            resolved = target.resolve()
            processed_resolved = processed_dir.resolve()
            if processed_resolved not in resolved.parents:
                raise RuntimeError(f"Refuse to delete outside data/processed: {resolved}")
            if target.exists():
                print(f"[INFO] Removing generated directory: {target}")
                shutil.rmtree(target)

    for split in ("train", "val", "test"):
        for class_name in CLASS_ID_TO_NAME.values():
            (cls_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    (anomaly_dir / "train" / "normal").mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        (anomaly_dir / split / "normal").mkdir(parents=True, exist_ok=True)
        (anomaly_dir / split / "anomaly").mkdir(parents=True, exist_ok=True)

    tables_dir.mkdir(parents=True, exist_ok=True)
    return cls_dir, anomaly_dir, tables_dir


def crop_with_padding(image: Image.Image, ann: Annotation, padding_ratio: float) -> Optional[Image.Image]:
    width, height = image.size
    bbox_w = ann.x2 - ann.x1
    bbox_h = ann.y2 - ann.y1
    pad_x = int(round(bbox_w * padding_ratio))
    pad_y = int(round(bbox_h * padding_ratio))

    left = max(0, ann.x1 - pad_x)
    top = max(0, ann.y1 - pad_y)
    right = min(width, ann.x2 + pad_x)
    bottom = min(height, ann.y2 + pad_y)

    if right <= left or bottom <= top:
        return None

    return image.crop((left, top, right, bottom))


def save_patch(patch: Image.Image, output_path: Path, patch_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch = patch.convert("RGB").resize((patch_size, patch_size), Image.Resampling.BILINEAR)
    patch.save(output_path, quality=95)


def safe_file_stem(sample: Sample, ann_index: int, class_name: str) -> str:
    sample_part = sample.sample_id.replace("/", "_").replace("\\", "_")
    return f"{sample_part}_box{ann_index:02d}_{class_name}"


def process_splits(
    splits: Dict[str, Sequence[Sample]],
    cls_dir: Path,
    anomaly_dir: Path,
    patch_size: int,
    padding_ratio: float,
) -> Tuple[Counter, Counter, List[Tuple[Image.Image, Image.Image]]]:
    cls_counts: Counter = Counter()
    anomaly_counts: Counter = Counter()
    sample_pairs: List[Tuple[Image.Image, Image.Image]] = []

    for split_name, split_samples in splits.items():
        print(f"[INFO] Processing split '{split_name}' with {len(split_samples)} source samples...")
        for sample in split_samples:
            try:
                with Image.open(sample.test_path) as test_img_raw, Image.open(sample.temp_path) as temp_img_raw:
                    test_img = test_img_raw.convert("RGB")
                    temp_img = temp_img_raw.convert("RGB")

                    for ann_index, ann in enumerate(sample.annotations, start=1):
                        anomaly_patch = crop_with_padding(test_img, ann, padding_ratio)
                        normal_patch = crop_with_padding(temp_img, ann, padding_ratio)
                        if anomaly_patch is None or normal_patch is None:
                            print(f"[WARN] Skip invalid crop: {sample.annotation_path} bbox {ann_index}")
                            continue

                        file_stem = safe_file_stem(sample, ann_index, ann.class_name)

                        cls_path = cls_dir / split_name / ann.class_name / f"{file_stem}.jpg"
                        save_patch(anomaly_patch, cls_path, patch_size)
                        cls_counts[(split_name, ann.class_name)] += 1

                        normal_path = anomaly_dir / split_name / "normal" / f"{file_stem}_normal.jpg"
                        save_patch(normal_patch, normal_path, patch_size)
                        anomaly_counts[(split_name, "normal")] += 1

                        if split_name != "train":
                            anomaly_path = anomaly_dir / split_name / "anomaly" / f"{file_stem}_anomaly.jpg"
                            save_patch(anomaly_patch, anomaly_path, patch_size)
                            anomaly_counts[(split_name, "anomaly")] += 1

                        if split_name != "train" and len(sample_pairs) < 8:
                            sample_pairs.append(
                                (
                                    normal_patch.convert("RGB").resize((patch_size, patch_size)),
                                    anomaly_patch.convert("RGB").resize((patch_size, patch_size)),
                                )
                            )
            except OSError as exc:
                print(f"[WARN] Failed to open image pair for {sample.annotation_path}: {exc}")

    return cls_counts, anomaly_counts, sample_pairs


def write_summary_csv(summary_path: Path, cls_counts: Counter, anomaly_counts: Counter) -> None:
    rows = []
    for split in ("train", "val", "test"):
        for class_name in CLASS_ID_TO_NAME.values():
            rows.append(
                {
                    "dataset": "pcb_cls",
                    "split": split,
                    "class_name": class_name,
                    "count": cls_counts.get((split, class_name), 0),
                }
            )

    for split in ("train", "val", "test"):
        for label in ("normal", "anomaly"):
            rows.append(
                {
                    "dataset": "pcb_anomaly",
                    "split": split,
                    "class_name": label,
                    "count": anomaly_counts.get((split, label), 0),
                }
            )

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "class_name", "count"])
        writer.writeheader()
        writer.writerows(rows)


def make_sample_grid(sample_pairs: Sequence[Tuple[Image.Image, Image.Image]], output_path: Path, patch_size: int) -> None:
    if not sample_pairs:
        print("[WARN] No val/test patch pairs available for sample grid.")
        return

    cols = min(8, len(sample_pairs))
    label_width = 92
    row_gap = 8
    canvas_w = label_width + cols * patch_size
    canvas_h = patch_size * 2 + row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, patch_size // 2 - 8), "normal", fill=(20, 20, 20))
    draw.text((10, patch_size + row_gap + patch_size // 2 - 8), "anomaly", fill=(20, 20, 20))

    for col, (normal_patch, anomaly_patch) in enumerate(sample_pairs[:cols]):
        x = label_width + col * patch_size
        canvas.paste(normal_patch, (x, 0))
        canvas.paste(anomaly_patch, (x, patch_size + row_gap))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def print_counter_table(title: str, counter: Counter, splits: Iterable[str], labels: Iterable[str]) -> None:
    print(f"\n{title}")
    for split in splits:
        parts = [f"{label}: {counter.get((split, label), 0)}" for label in labels]
        print(f"  {split}: " + ", ".join(parts))


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()

    print("=" * 80)
    print("Stage 2 - Prepare DeepPCB Processed Datasets")
    print("=" * 80)
    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] patch_size={args.patch_size}, padding_ratio={args.padding_ratio}, val_ratio={args.val_ratio}, seed={args.seed}")

    if args.patch_size <= 0:
        raise ValueError("--patch_size must be positive.")
    if args.padding_ratio < 0:
        raise ValueError("--padding_ratio must be >= 0.")

    pcbdata_dir = find_pcbdata_dir(project_root)
    print(f"[INFO] Raw PCBData directory: {pcbdata_dir}")

    trainval_pairs = read_split_file(pcbdata_dir / "trainval.txt")
    test_pairs = read_split_file(pcbdata_dir / "test.txt")
    trainval_samples, trainval_failed = collect_official_split_samples(pcbdata_dir, trainval_pairs)
    test_samples, test_failed = collect_official_split_samples(pcbdata_dir, test_pairs)

    print("\n[1] Official split check")
    print(f"  trainval.txt lines: {len(trainval_pairs)}, matched samples: {len(trainval_samples)}, failed: {trainval_failed}")
    print(f"  test.txt lines: {len(test_pairs)}, matched samples: {len(test_samples)}, failed: {test_failed}")

    use_official = (
        len(trainval_pairs) > 0
        and len(test_pairs) > 0
        and len(trainval_samples) == len(trainval_pairs)
        and len(test_samples) == len(test_pairs)
    )

    if use_official:
        train_samples, val_samples = split_train_val(trainval_samples, args.val_ratio, args.seed)
        splits = {"train": train_samples, "val": val_samples, "test": test_samples}
        print("[INFO] Using official trainval/test split.")
    else:
        print("[WARN] Official split matching failed. Falling back to random 70/15/15 split.")
        all_samples = collect_all_samples(pcbdata_dir)
        if not all_samples:
            raise RuntimeError("No valid DeepPCB samples were collected.")
        splits = split_random_all(all_samples, args.seed)

    collected_samples = sum(len(samples) for samples in splits.values())
    collected_boxes = sum(len(sample.annotations) for samples in splits.values() for sample in samples)
    print("\n[2] Collected source samples")
    print(f"  source samples: {collected_samples}")
    print(f"  defect boxes: {collected_boxes}")

    print("\n[3] Split source sample counts")
    for split_name in ("train", "val", "test"):
        samples = splits[split_name]
        boxes = sum(len(sample.annotations) for sample in samples)
        print(f"  {split_name}: {len(samples)} samples, {boxes} boxes")

    cls_dir, anomaly_dir, tables_dir = safe_prepare_output_dirs(project_root, args.overwrite)

    print("\n[4] Generate patch datasets")
    cls_counts, anomaly_counts, sample_pairs = process_splits(
        splits=splits,
        cls_dir=cls_dir,
        anomaly_dir=anomaly_dir,
        patch_size=args.patch_size,
        padding_ratio=args.padding_ratio,
    )

    summary_csv = tables_dir / "stage2_prepare_summary.csv"
    sample_grid = tables_dir / "stage2_sample_patches.jpg"
    write_summary_csv(summary_csv, cls_counts, anomaly_counts)
    make_sample_grid(sample_pairs, sample_grid, args.patch_size)

    print_counter_table(
        "[5] Supervised classification patch counts",
        cls_counts,
        ("train", "val", "test"),
        CLASS_ID_TO_NAME.values(),
    )
    print_counter_table(
        "[6] Anomaly dataset patch counts",
        anomaly_counts,
        ("train", "val", "test"),
        ("normal", "anomaly"),
    )

    print("\n[7] Output files")
    print(f"  classification dataset: {cls_dir}")
    print(f"  anomaly dataset: {anomaly_dir}")
    print(f"  summary csv: {summary_csv}")
    print(f"  sample patches: {sample_grid}")
    print("\nStage 2 data preparation finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] Stage 2 data preparation failed: {exc}")
        raise

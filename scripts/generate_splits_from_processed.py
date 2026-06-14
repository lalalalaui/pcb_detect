import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EMPTY_SOURCE_FIELDS = (
    "source_image",
    "source_annotation",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_class",
    "bbox_index",
)

CLASSIFICATION_FIELDS = (
    "split",
    "task",
    "patch_path",
    "label",
    *EMPTY_SOURCE_FIELDS,
)

ANOMALY_FIELDS = (
    "split",
    "task",
    "patch_path",
    "label",
    "binary_label",
    *EMPTY_SOURCE_FIELDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate low-fidelity data/splits CSV files from data/processed "
            "without rerunning prepare_data.py."
        )
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing data/processed. Defaults to the repository root.",
    )
    return parser.parse_args()


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def empty_source_values() -> Dict[str, str]:
    return {field: "" for field in EMPTY_SOURCE_FIELDS}


def collect_classification_rows(project_root: Path, split: str) -> List[Dict[str, str]]:
    split_dir = project_root / "data" / "processed" / "pcb_cls" / split
    rows: List[Dict[str, str]] = []

    if not split_dir.exists():
        return rows

    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        for patch_path in list_images(class_dir):
            rows.append(
                {
                    "split": split,
                    "task": "classification",
                    "patch_path": rel_posix(patch_path, project_root),
                    "label": class_dir.name,
                    **empty_source_values(),
                }
            )

    return rows


def collect_anomaly_rows(project_root: Path, split: str) -> List[Dict[str, str]]:
    split_dir = project_root / "data" / "processed" / "pcb_anomaly" / split
    rows: List[Dict[str, str]] = []
    label_to_binary = {"normal": "0", "anomaly": "1"}

    for label, binary_label in label_to_binary.items():
        label_dir = split_dir / label
        for patch_path in list_images(label_dir):
            rows.append(
                {
                    "split": split,
                    "task": "anomaly",
                    "patch_path": rel_posix(patch_path, project_root),
                    "label": label,
                    "binary_label": binary_label,
                    **empty_source_values(),
                }
            )

    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, str]]) -> int:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)

    return len(row_list)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    splits_dir = project_root / "data" / "splits"

    print("=" * 80)
    print("Generate Split CSVs From Processed Data")
    print("=" * 80)
    print(f"[INFO] Project root: {project_root}")
    print(
        "[INFO] Low-fidelity manifest only: source and bbox fields are left empty "
        "because they cannot be reliably recovered from processed patch paths."
    )

    for split in SPLITS:
        rows = collect_classification_rows(project_root, split)
        output_path = splits_dir / f"classification_{split}.csv"
        count = write_csv(output_path, CLASSIFICATION_FIELDS, rows)
        print(f"[INFO] {output_path.relative_to(project_root).as_posix()}: {count} samples")

    for split in SPLITS:
        rows = collect_anomaly_rows(project_root, split)
        output_path = splits_dir / f"anomaly_{split}.csv"
        count = write_csv(output_path, ANOMALY_FIELDS, rows)
        print(f"[INFO] {output_path.relative_to(project_root).as_posix()}: {count} samples")

    print("\nDone.")


if __name__ == "__main__":
    main()

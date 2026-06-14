import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
GROUPS = (
    ("train", "normal"),
    ("val", "normal"),
    ("val", "anomaly"),
    ("test", "normal"),
    ("test", "anomaly"),
)


@dataclass
class ImageStats:
    split: str
    label: str
    path: Path
    mean: float
    std: float
    min_value: int
    max_value: int
    channel_delta_mean: float
    near_binary_ratio: float
    low_texture_score: float
    is_near_binary: bool
    is_low_texture: bool
    is_blank: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect processed PCB anomaly patches for binary/mask/debug-image artifacts."
    )
    parser.add_argument(
        "--data_dir",
        default="data/processed/pcb_anomaly",
        help="Processed anomaly dataset root.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/debug",
        help="Directory for grid image and CSV report.",
    )
    parser.add_argument("--samples_per_group", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thumb_size", type=int, default=96)
    return parser.parse_args()


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def low_texture_score(gray: np.ndarray) -> float:
    gray_f = gray.astype(np.float32)
    if gray_f.shape[0] < 2 or gray_f.shape[1] < 2:
        return 0.0
    gx = np.abs(np.diff(gray_f, axis=1)).mean()
    gy = np.abs(np.diff(gray_f, axis=0)).mean()
    return float((gx + gy) * 0.5)


def inspect_image(split: str, label: str, path: Path) -> ImageStats:
    rgb = load_rgb(path)
    gray = rgb.mean(axis=2)
    channel_delta = np.abs(rgb.astype(np.int16) - rgb.mean(axis=2, keepdims=True)).mean()
    near_zero_or_full = (rgb <= 8) | (rgb >= 247)
    near_binary_ratio = float(near_zero_or_full.mean())
    texture = low_texture_score(gray)
    std = float(rgb.std())

    return ImageStats(
        split=split,
        label=label,
        path=path,
        mean=float(rgb.mean()),
        std=std,
        min_value=int(rgb.min()),
        max_value=int(rgb.max()),
        channel_delta_mean=float(channel_delta),
        near_binary_ratio=near_binary_ratio,
        low_texture_score=texture,
        is_near_binary=near_binary_ratio >= 0.90,
        is_low_texture=texture < 3.0,
        is_blank=std < 3.0,
    )


def aggregate_stats(stats: Sequence[ImageStats]) -> Dict[str, float]:
    if not stats:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "near_binary_ratio_mean": float("nan"),
            "near_binary_image_fraction": float("nan"),
            "low_texture_fraction": float("nan"),
            "blank_fraction": float("nan"),
            "channel_delta_mean": float("nan"),
            "low_texture_score_mean": float("nan"),
        }

    return {
        "count": len(stats),
        "mean": float(np.mean([row.mean for row in stats])),
        "std": float(np.mean([row.std for row in stats])),
        "min": float(np.min([row.min_value for row in stats])),
        "max": float(np.max([row.max_value for row in stats])),
        "near_binary_ratio_mean": float(np.mean([row.near_binary_ratio for row in stats])),
        "near_binary_image_fraction": float(np.mean([row.is_near_binary for row in stats])),
        "low_texture_fraction": float(np.mean([row.is_low_texture for row in stats])),
        "blank_fraction": float(np.mean([row.is_blank for row in stats])),
        "channel_delta_mean": float(np.mean([row.channel_delta_mean for row in stats])),
        "low_texture_score_mean": float(np.mean([row.low_texture_score for row in stats])),
    }


def font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 11)
    except OSError:
        return ImageFont.load_default()


def draw_grid(
    selected: Dict[Tuple[str, str], Sequence[Path]],
    output_path: Path,
    thumb_size: int,
) -> None:
    cols = 8
    label_h = 34
    group_title_h = 22
    gap = 8
    cell_w = thumb_size + gap
    cell_h = thumb_size + label_h + gap
    rows_per_group = max(1, (max((len(paths) for paths in selected.values()), default=1) + cols - 1) // cols)

    width = cols * cell_w + gap
    height = len(GROUPS) * (group_title_h + rows_per_group * cell_h) + gap
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    text_font = font()

    y = gap
    for split, label in GROUPS:
        paths = list(selected.get((split, label), []))
        title = f"{split}/{label}  sampled={len(paths)}"
        draw.rectangle((0, y, width, y + group_title_h - 2), fill=(32, 40, 48))
        draw.text((gap, y + 4), title, fill=(255, 255, 255), font=text_font)
        y += group_title_h

        for row in range(rows_per_group):
            for col in range(cols):
                index = row * cols + col
                x = gap + col * cell_w
                yy = y + row * cell_h
                draw.rectangle((x - 1, yy - 1, x + thumb_size, yy + thumb_size), outline=(190, 190, 190))
                if index >= len(paths):
                    continue

                path = paths[index]
                with Image.open(path) as image:
                    thumb = image.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.NEAREST)
                canvas.paste(thumb, (x, yy))

                name = path.name
                short_name = name if len(name) <= 24 else name[:21] + "..."
                draw.text((x, yy + thumb_size + 2), f"{split}/{label}", fill=(20, 20, 20), font=text_font)
                draw.text((x, yy + thumb_size + 16), short_name, fill=(20, 20, 20), font=text_font)

        y += rows_per_group * cell_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_report(
    output_path: Path,
    all_stats: Sequence[ImageStats],
    sampled_stats: Sequence[ImageStats],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "split",
        "label",
        "count",
        "path",
        "mean",
        "std",
        "min",
        "max",
        "channel_delta_mean",
        "near_binary_ratio",
        "near_binary_image_fraction",
        "low_texture_score",
        "low_texture_fraction",
        "blank_fraction",
        "is_near_binary",
        "is_low_texture",
        "is_blank",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for split, label in GROUPS:
            rows = [row for row in all_stats if row.split == split and row.label == label]
            summary = aggregate_stats(rows)
            writer.writerow(
                {
                    "row_type": "summary_all",
                    "split": split,
                    "label": label,
                    "count": int(summary["count"]),
                    "path": "",
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "channel_delta_mean": summary["channel_delta_mean"],
                    "near_binary_ratio": summary["near_binary_ratio_mean"],
                    "near_binary_image_fraction": summary["near_binary_image_fraction"],
                    "low_texture_score": summary["low_texture_score_mean"],
                    "low_texture_fraction": summary["low_texture_fraction"],
                    "blank_fraction": summary["blank_fraction"],
                    "is_near_binary": "",
                    "is_low_texture": "",
                    "is_blank": "",
                }
            )

        for row in sampled_stats:
            writer.writerow(
                {
                    "row_type": "sample",
                    "split": row.split,
                    "label": row.label,
                    "count": "",
                    "path": row.path.as_posix(),
                    "mean": row.mean,
                    "std": row.std,
                    "min": row.min_value,
                    "max": row.max_value,
                    "channel_delta_mean": row.channel_delta_mean,
                    "near_binary_ratio": row.near_binary_ratio,
                    "near_binary_image_fraction": "",
                    "low_texture_score": row.low_texture_score,
                    "low_texture_fraction": "",
                    "blank_fraction": "",
                    "is_near_binary": int(row.is_near_binary),
                    "is_low_texture": int(row.is_low_texture),
                    "is_blank": int(row.is_blank),
                }
            )


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    rng = random.Random(args.seed)
    all_stats: List[ImageStats] = []
    sampled_stats: List[ImageStats] = []
    selected: Dict[Tuple[str, str], List[Path]] = {}

    for split, label in GROUPS:
        directory = data_dir / split / label
        paths = list_images(directory)
        sample_count = min(args.samples_per_group, len(paths))
        selected_paths = rng.sample(paths, sample_count) if sample_count else []
        selected[(split, label)] = selected_paths

        for path in paths:
            all_stats.append(inspect_image(split, label, path))
        for path in selected_paths:
            sampled_stats.append(inspect_image(split, label, path))

    grid_path = output_dir / "pcb_anomaly_dataset_grid.png"
    report_path = output_dir / "pcb_anomaly_dataset_report.csv"
    draw_grid(selected, grid_path, args.thumb_size)
    write_report(report_path, all_stats, sampled_stats)

    print("=" * 80)
    print("PCB anomaly dataset inspection")
    print("=" * 80)
    print(f"Data dir: {data_dir}")
    print(f"Grid: {grid_path}")
    print(f"CSV report: {report_path}")
    print()
    for split, label in GROUPS:
        rows = [row for row in all_stats if row.split == split and row.label == label]
        summary = aggregate_stats(rows)
        print(
            f"{split}/{label}: count={int(summary['count'])}, "
            f"mean={summary['mean']:.3f}, std={summary['std']:.3f}, "
            f"min={summary['min']:.0f}, max={summary['max']:.0f}, "
            f"near_binary_images={summary['near_binary_image_fraction']:.1%}, "
            f"low_texture={summary['low_texture_fraction']:.1%}, "
            f"blank={summary['blank_fraction']:.1%}, "
            f"channel_delta={summary['channel_delta_mean']:.3f}"
        )


if __name__ == "__main__":
    main()

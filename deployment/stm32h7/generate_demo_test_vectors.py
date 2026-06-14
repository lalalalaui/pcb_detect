import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
NORMAL_DIR = PROJECT_ROOT / "data" / "processed" / "pcb_anomaly" / "test" / "normal"
ANOMALY_DIR = PROJECT_ROOT / "data" / "processed" / "pcb_anomaly" / "test" / "anomaly"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT_C = 3
INPUT_H = 96
INPUT_W = 96
INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W
THRESHOLD_LOW = 0.000297238343
THRESHOLD_HIGH = 0.000630778263
RESULT_TO_ID = {"NORMAL": 0, "SUSPECT": 1, "ANOMALY": 2}
QUALITY_STD_MIN = 0.03
QUALITY_MEAN_MIN = 0.05
QUALITY_MEAN_MAX = 0.95
QUALITY_EDGE_DENSITY_MIN = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate STM32H7 demo test_vectors.h from processed PCB anomaly patches."
    )
    parser.add_argument("--num-normal", type=int, default=3, help="Number of normal patches.")
    parser.add_argument("--num-anomaly", type=int, default=3, help="Number of anomaly patches.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("deployment/onnx/anomaly_tiny_ae_96.onnx"),
        help="TinyAE ONNX model path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deployment/stm32h7/test_vectors.h"),
        help="Output test_vectors.h path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    parser.add_argument(
        "--selection",
        choices=["random", "representative_demo"],
        default="random",
        help="Select random patches or representative demo patches.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing image directory: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_images(directory: Path, count: int, rng: random.Random) -> List[Path]:
    if count < 0:
        raise ValueError("Sample count must be >= 0.")

    images = list_images(directory)
    if len(images) < count:
        raise RuntimeError(f"Need {count} images from {directory}, found {len(images)}.")

    return sorted(rng.sample(images, count))


def preprocess_image(image_path: Path) -> np.ndarray:
    # Deployment preprocessing matches TinyAE training: RGB, resize, ToTensor [0,1].
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((INPUT_W, INPUT_H), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return chw.astype(np.float32)


def run_onnx(session, input_name: str, input_chw: np.ndarray) -> np.ndarray:
    input_nchw = input_chw[None, ...]
    output = session.run(None, {input_name: input_nchw})[0]
    return np.asarray(output[0], dtype=np.float32)


def reconstruction_mse(input_chw: np.ndarray, output_chw: np.ndarray) -> float:
    return float(np.mean((input_chw - output_chw) ** 2))


def classify_score(score: float) -> str:
    if score < THRESHOLD_LOW:
        return "NORMAL"
    if score < THRESHOLD_HIGH:
        return "SUSPECT"
    return "ANOMALY"


def image_quality_metrics(input_chw: np.ndarray) -> Dict[str, float]:
    gray = np.mean(input_chw, axis=0)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_density = (float(np.mean(dx)) + float(np.mean(dy))) / 2.0
    return {
        "mean": float(np.mean(input_chw)),
        "std": float(np.std(input_chw)),
        "min": float(np.min(input_chw)),
        "max": float(np.max(input_chw)),
        "edge_density": edge_density,
    }


def passes_demo_quality(item: Dict[str, object]) -> bool:
    return (
        float(item["std"]) >= QUALITY_STD_MIN
        and QUALITY_MEAN_MIN <= float(item["mean"]) <= QUALITY_MEAN_MAX
        and float(item["edge_density"]) >= QUALITY_EDGE_DENSITY_MIN
    )


def score_patch(session, input_name: str, image_path: Path) -> float:
    input_chw = preprocess_image(image_path)
    output_chw = run_onnx(session, input_name, input_chw)
    return reconstruction_mse(input_chw, output_chw)


def make_row(
    index: int,
    image_path: Path,
    label: int,
    score: float,
    selection_reason: str,
    mean: float = float("nan"),
    std: float = float("nan"),
    edge_density: float = float("nan"),
    score_percentile: float = float("nan"),
    selection_warning: str = "",
) -> Dict[str, object]:
    pc_result = classify_score(score)
    return {
        "index": index,
        "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
        "label": label,
        "score": score,
        "pc_result": pc_result,
        "mean": mean,
        "std": std,
        "edge_density": edge_density,
        "score_percentile": score_percentile,
        "selection_reason": selection_reason,
        "selection_warning": selection_warning,
        "pc_result_id": RESULT_TO_ID[pc_result],
        "_path": image_path,
    }


def add_score_percentiles(candidates: List[Dict[str, object]]) -> None:
    for label in (0, 1):
        label_items = [item for item in candidates if int(item["label"]) == label]
        if len(label_items) <= 1:
            for item in label_items:
                item["score_percentile"] = 0.0
            continue

        sorted_scores = np.asarray(sorted(float(item["score"]) for item in label_items), dtype=np.float64)
        denom = max(len(sorted_scores) - 1, 1)
        for item in label_items:
            rank = int(np.searchsorted(sorted_scores, float(item["score"]), side="left"))
            item["score_percentile"] = 100.0 * rank / denom


def score_candidates(session, input_name: str) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    sources = (
        (NORMAL_DIR, 0),
        (ANOMALY_DIR, 1),
    )
    total = sum(len(list_images(directory)) for directory, _ in sources)
    processed = 0

    print(f"[INFO] Scoring all test candidates with ONNX Runtime: {total} images")
    for directory, label in sources:
        for image_path in list_images(directory):
            input_chw = preprocess_image(image_path)
            output_chw = run_onnx(session, input_name, input_chw)
            score = reconstruction_mse(input_chw, output_chw)
            quality = image_quality_metrics(input_chw)
            candidates.append(
                {
                    "image_path": image_path,
                    "label": label,
                    "score": score,
                    "pc_result": classify_score(score),
                    **quality,
                }
            )
            processed += 1
            if processed % 500 == 0 or processed == total:
                print(f"[INFO] scored {processed}/{total}")

    add_score_percentiles(candidates)
    return candidates


def pick_by_percentile_targets(
    pool: Sequence[Dict[str, object]],
    count: int,
    low_percentile: float,
    high_percentile: float,
) -> List[Dict[str, object]]:
    if count <= 0:
        return []

    candidates = [
        item
        for item in pool
        if low_percentile <= float(item["score_percentile"]) <= high_percentile
    ]
    if len(candidates) < count:
        return []

    targets = np.linspace(low_percentile, high_percentile, count + 2)[1:-1]
    selected: List[Dict[str, object]] = []
    selected_paths = set()
    for target in targets:
        available = [item for item in candidates if item["image_path"] not in selected_paths]
        if not available:
            break
        chosen = min(
            available,
            key=lambda item: (
                abs(float(item["score_percentile"]) - float(target)),
                abs(float(item["mean"]) - 0.5),
                -float(item["std"]),
                str(item["image_path"]),
            ),
        )
        selected.append(chosen)
        selected_paths.add(chosen["image_path"])

    if len(selected) < count:
        remaining = [
            item
            for item in sorted(
                candidates,
                key=lambda item: (
                    abs(float(item["score_percentile"]) - ((low_percentile + high_percentile) / 2.0)),
                    abs(float(item["mean"]) - 0.5),
                    str(item["image_path"]),
                ),
            )
            if item["image_path"] not in selected_paths
        ]
        selected.extend(remaining[: count - len(selected)])

    return selected[:count]


def rows_from_selected_items(
    selected: Sequence[Dict[str, object]],
    rng: random.Random,
) -> List[Dict[str, object]]:
    rows = [
        make_row(
            index=index,
            image_path=item["image_path"],
            label=int(item["label"]),
            score=float(item["score"]),
            mean=float(item["mean"]),
            std=float(item["std"]),
            edge_density=float(item["edge_density"]),
            score_percentile=float(item["score_percentile"]),
            selection_reason=str(item["selection_reason"]),
            selection_warning=str(item.get("selection_warning", "")),
        )
        for index, item in enumerate(selected)
    ]
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["index"] = index
    return rows


def select_representative_demo(
    candidates: Sequence[Dict[str, object]],
    num_normal: int,
    num_anomaly: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    for name, count in (("--num-normal", num_normal), ("--num-anomaly", num_anomaly)):
        if count < 0:
            raise ValueError(f"{name} must be >= 0.")

    quality_candidates = [item for item in candidates if passes_demo_quality(item)]
    normal_pool = [
        item
        for item in quality_candidates
        if int(item["label"]) == 0
    ]
    anomaly_pool = [
        item
        for item in quality_candidates
        if int(item["label"]) == 1
    ]

    normal_selected = pick_by_percentile_targets(normal_pool, num_normal, 20.0, 60.0)
    normal_reason = "normal_representative_p20_p60"
    normal_warning = ""
    if len(normal_selected) < num_normal:
        normal_selected = pick_by_percentile_targets(normal_pool, num_normal, 10.0, 80.0)
        normal_reason = "normal_representative_p10_p80"
        normal_warning = "preferred_normal_p20_p60_insufficient"
    if len(normal_selected) < num_normal:
        fallback = sorted(
            normal_pool,
            key=lambda item: (
                abs(float(item["score_percentile"]) - 50.0),
                abs(float(item["mean"]) - 0.5),
                str(item["image_path"]),
            ),
        )
        normal_selected = fallback[:num_normal]
        normal_reason = "normal_quality_fallback"
        normal_warning = "normal_percentile_range_insufficient"
    if len(normal_selected) < num_normal:
        raise RuntimeError(f"Need {num_normal} representative normal samples, found {len(normal_selected)}.")

    for item in normal_selected:
        item["selection_reason"] = normal_reason
        item["selection_warning"] = normal_warning

    anomaly_selected = pick_by_percentile_targets(anomaly_pool, num_anomaly, 60.0, 90.0)
    anomaly_reason = "anomaly_representative_p60_p90"
    anomaly_warning = ""
    if len(anomaly_selected) < num_anomaly:
        anomaly_selected = pick_by_percentile_targets(anomaly_pool, num_anomaly, 50.0, 95.0)
        anomaly_reason = "anomaly_representative_p50_p95"
        anomaly_warning = "preferred_anomaly_p60_p90_insufficient"
    if len(anomaly_selected) < num_anomaly:
        non_extreme_anomaly_pool = [item for item in anomaly_pool if float(item["score_percentile"]) <= 95.0]
        anomaly_selected = sorted(
            non_extreme_anomaly_pool,
            key=lambda item: (
                abs(float(item["score_percentile"]) - 75.0),
                abs(float(item["mean"]) - 0.5),
                str(item["image_path"]),
            ),
        )[:num_anomaly]
        anomaly_reason = "anomaly_non_extreme_fallback"
        anomaly_warning = "anomaly_percentile_range_insufficient"
    if len(anomaly_selected) < num_anomaly:
        anomaly_selected = sorted(
            [item for item in quality_candidates if int(item["label"]) == 1],
            key=lambda item: (
                -float(item["score"]),
                abs(float(item["mean"]) - 0.5),
                str(item["image_path"]),
            ),
        )[:num_anomaly]
        anomaly_reason = "anomaly_quality_fallback"
        anomaly_warning = "limited_quality_anomaly_candidates"
    if len(anomaly_selected) < num_anomaly:
        raise RuntimeError(f"Need {num_anomaly} representative anomaly samples, found {len(anomaly_selected)}.")

    for item in anomaly_selected:
        item["selection_reason"] = anomaly_reason
        item["selection_warning"] = anomaly_warning

    return rows_from_selected_items([*normal_selected, *anomaly_selected], rng)


def format_float(value: float) -> str:
    text = f"{value:.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def format_c_array(name: str, values: np.ndarray) -> List[str]:
    flat = values.reshape(-1)
    lines = [f"static const float {name}[TEST_INPUT_SIZE] = {{"]
    for start in range(0, flat.size, 8):
        chunk = ", ".join(format_float(float(value)) for value in flat[start : start + 8])
        lines.append(f"    {chunk},")
    lines.append("};")
    return lines


def generate_header(rows: Sequence[Dict[str, object]], inputs: Sequence[np.ndarray]) -> str:
    lines: List[str] = [
        "#ifndef PCB_TINY_AE_TEST_VECTORS_H",
        "#define PCB_TINY_AE_TEST_VECTORS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define TEST_VECTOR_COUNT {len(inputs)}",
        f"#define TEST_INPUT_C {INPUT_C}",
        f"#define TEST_INPUT_H {INPUT_H}",
        f"#define TEST_INPUT_W {INPUT_W}",
        f"#define TEST_INPUT_SIZE {INPUT_SIZE}",
        "#define TEST_VECTOR_HAS_EXPECTED_SCORE 1",
        "",
        "#define TEST_RESULT_NORMAL  0",
        "#define TEST_RESULT_SUSPECT 1",
        "#define TEST_RESULT_ANOMALY 2",
        "",
        f"#define TEST_THRESHOLD_LOW  {format_float(THRESHOLD_LOW)}",
        f"#define TEST_THRESHOLD_HIGH {format_float(THRESHOLD_HIGH)}",
        "",
    ]

    for index, (row, input_chw) in enumerate(zip(rows, inputs)):
        lines.append(f"/* {index}: label={row['label']} result={row['pc_result']} path={row['image_path']} */")
        lines.extend(format_c_array(f"g_test_input_{index}", input_chw))
        lines.append("")

    lines.append("static const float * const g_test_inputs[TEST_VECTOR_COUNT] = {")
    for index in range(len(inputs)):
        lines.append(f"    g_test_input_{index},")
    lines.append("};")
    lines.append("")

    lines.append("static const uint8_t g_test_label[TEST_VECTOR_COUNT] = {")
    lines.append("    " + ", ".join(str(row["label"]) for row in rows) + ",")
    lines.append("};")
    lines.append("")

    lines.append("static const float g_test_expected_score[TEST_VECTOR_COUNT] = {")
    lines.append("    " + ", ".join(format_float(float(row["score"])) for row in rows) + ",")
    lines.append("};")
    lines.append("")

    lines.append("static const uint8_t g_test_expected_result[TEST_VECTOR_COUNT] = {")
    lines.append("    " + ", ".join(str(row["pc_result_id"]) for row in rows) + ",")
    lines.append("};")
    lines.append("")

    lines.append("#endif /* PCB_TINY_AE_TEST_VECTORS_H */")
    lines.append("")
    return "\n".join(lines)


def write_manifest(output_path: Path, rows: Sequence[Dict[str, object]]) -> Path:
    manifest_path = output_path.parent / "test_vectors_manifest.csv"
    fieldnames = [
        "index",
        "image_path",
        "label",
        "score",
        "pc_result",
        "mean",
        "std",
        "edge_density",
        "score_percentile",
        "selection_reason",
        "selection_warning",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
    return manifest_path


def main() -> None:
    args = parse_args()
    onnx_path = resolve_project_path(args.onnx)
    output_path = resolve_project_path(args.output)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")

    import onnxruntime as ort

    print("=" * 80)
    print("Generate STM32H7 Demo Test Vectors")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] ONNX model: {onnx_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"[INFO] Output: {output_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"[INFO] Selection: {args.selection}, seed={args.seed}")
    print(f"[INFO] threshold_low={THRESHOLD_LOW:.12f}, threshold_high={THRESHOLD_HIGH:.12f}")
    rng = random.Random(args.seed)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    if args.selection == "representative_demo":
        candidates = score_candidates(session, input_name)
        rows = select_representative_demo(
            candidates=candidates,
            num_normal=args.num_normal,
            num_anomaly=args.num_anomaly,
            rng=rng,
        )
    else:
        selected: List[Tuple[Path, int, str, str]] = []
        selected.extend(
            (path, 0, "normal", "random_normal")
            for path in select_images(NORMAL_DIR, args.num_normal, rng)
        )
        selected.extend(
            (path, 1, "anomaly", "random_anomaly")
            for path in select_images(ANOMALY_DIR, args.num_anomaly, rng)
        )

        rows = []
        for index, (image_path, label, _label_name, selection_reason) in enumerate(selected):
            input_chw = preprocess_image(image_path)
            output_chw = run_onnx(session, input_name, input_chw)
            score = reconstruction_mse(input_chw, output_chw)
            quality = image_quality_metrics(input_chw)
            rows.append(
                make_row(
                    index=index,
                    image_path=image_path,
                    label=label,
                    score=score,
                    selection_reason=selection_reason,
                    mean=quality["mean"],
                    std=quality["std"],
                    edge_density=quality["edge_density"],
                )
            )

    inputs: List[np.ndarray] = []
    for row in rows:
        index = int(row["index"])
        image_path = row["_path"]
        input_chw = preprocess_image(image_path)
        inputs.append(input_chw)
        print(
            f"[{index}] path={row['image_path']}, label={row['label']}, "
            f"score={float(row['score']):.10f}, PC result={row['pc_result']}, "
            f"mean={float(row['mean']):.4f}, std={float(row['std']):.4f}, "
            f"edge_density={float(row['edge_density']):.6f}, "
            f"selection_reason={row['selection_reason']}"
        )
        if row["selection_warning"]:
            print(f"    selection_warning={row['selection_warning']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_header(rows, inputs), encoding="utf-8", newline="\n")
    manifest_path = write_manifest(output_path, rows)

    print("\n[Output]")
    print(f"test_vectors.h: {output_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"manifest: {manifest_path.relative_to(PROJECT_ROOT).as_posix()}")
    print("\n[Reminder]")
    print("Copy test_vectors.h to edgeai/Core/Inc/test_vectors.h, then rebuild and flash the STM32 project.")


if __name__ == "__main__":
    main()

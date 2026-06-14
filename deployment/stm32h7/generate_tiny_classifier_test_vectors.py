import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT_C = 3
INPUT_H = 96
INPUT_W = 96
INPUT_SIZE = INPUT_C * INPUT_H * INPUT_W
NUM_CLASSES = 6
CLASS_TO_IDX_JSON = PROJECT_ROOT / "results" / "tables" / "tiny_classifier_class_to_idx.json"
TRAINING_DEFAULT_MEAN = (0.485, 0.456, 0.406)
TRAINING_DEFAULT_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate STM32 fixed test vectors for classifier_tiny_classifier_96.onnx."
    )
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument(
        "--onnx",
        default="deployment/onnx/classifier_tiny_classifier_96.onnx",
        help="Tiny classifier ONNX model path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/classifier/tiny_classifier_96_best.pth",
        help="Tiny classifier checkpoint path.",
    )
    parser.add_argument(
        "--output",
        default="deployment/stm32h7/tiny_classifier_test_vectors.h",
        help="Output C header path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_module(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            f"Missing dependency: {module_name}. Run this script in the same Python environment "
            "used for training/export, or install the package first."
        )


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_checkpoint(path: Path) -> Dict[str, object]:
    if importlib.util.find_spec("torch") is None:
        print(
            "[WARN] torch is not installed, so checkpoint metadata cannot be read. "
            "Falling back to class_to_idx json or sorted data directory order, and training default mean/std."
        )
        return {}

    import torch

    if not path.exists():
        print(f"[WARN] Checkpoint not found: {path}. Falling back to class_to_idx json or data directory order.")
        return {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a dict: {path}")
    return checkpoint


def load_class_to_idx_json(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "class_to_idx" in payload:
        payload = payload["class_to_idx"]
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid class_to_idx json: {path}")

    return {str(key): int(value) for key, value in payload.items()}


def class_to_idx_from_checkpoint(checkpoint: Dict[str, object]) -> Dict[str, int]:
    raw = checkpoint.get("class_to_idx")
    if isinstance(raw, dict):
        return {str(key): int(value) for key, value in raw.items()}

    class_names = checkpoint.get("class_names")
    if isinstance(class_names, (list, tuple)):
        return {str(class_name): index for index, class_name in enumerate(class_names)}

    return {}


def resolve_metadata(checkpoint: Dict[str, object]) -> Tuple[Dict[str, int], Tuple[float, float, float], Tuple[float, float, float], int]:
    class_to_idx = class_to_idx_from_checkpoint(checkpoint)
    if not class_to_idx:
        class_to_idx = load_class_to_idx_json(CLASS_TO_IDX_JSON)
    if not class_to_idx:
        data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_cls" / "test"
        class_names = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
        class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
        print("[WARN] class_to_idx not found in checkpoint/json; using sorted data/processed/pcb_cls/test directory order.")

    mean = checkpoint.get("mean")
    std = checkpoint.get("std")
    if mean is None:
        mean = checkpoint.get("input_mean")
    if std is None:
        std = checkpoint.get("input_std")

    if mean is None or std is None:
        mean = TRAINING_DEFAULT_MEAN
        std = TRAINING_DEFAULT_STD
        print(
            "[WARN] Checkpoint does not contain mean/std; using training dataloader defaults "
            f"IMAGENET_MEAN={tuple(mean)}, IMAGENET_STD={tuple(std)}."
        )

    image_size = int(checkpoint.get("input_size", checkpoint.get("image_size", INPUT_H)))
    return class_to_idx, tuple(float(v) for v in mean), tuple(float(v) for v in std), image_size


def class_names_in_training_order(class_to_idx: Dict[str, int]) -> List[str]:
    ordered = sorted(class_to_idx.items(), key=lambda item: item[1])
    indices = [index for _, index in ordered]
    if indices != list(range(len(indices))):
        raise RuntimeError(f"class_to_idx indices must be contiguous from 0: {class_to_idx}")
    return [class_name for class_name, _ in ordered]


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing class directory: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_samples(class_names: Sequence[str], samples_per_class: int, seed: int) -> List[Dict[str, object]]:
    if samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive.")

    rng = random.Random(seed)
    data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_cls" / "test"
    rows: List[Dict[str, object]] = []

    for class_index, class_name in enumerate(class_names):
        images = list_images(data_dir / class_name)
        if len(images) < samples_per_class:
            raise RuntimeError(
                f"Need {samples_per_class} images from {data_dir / class_name}, found {len(images)}."
            )

        for image_path in sorted(rng.sample(images, samples_per_class)):
            rows.append(
                {
                    "image_path": image_path,
                    "true_label": class_index,
                    "true_class": class_name,
                }
            )

    return rows


def preprocess_image(image_path: Path, mean: Sequence[float], std: Sequence[float], image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return ((chw - mean_arr) / std_arr).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits64 = logits.astype(np.float64)
    logits64 = logits64 - np.max(logits64)
    exp = np.exp(logits64)
    return (exp / np.sum(exp)).astype(np.float32)


def run_onnx(session, input_name: str, input_chw: np.ndarray) -> np.ndarray:
    output = session.run(None, {input_name: input_chw[None, ...]})[0]
    logits = np.asarray(output, dtype=np.float32).reshape(-1)
    if logits.size != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} logits, got shape {np.asarray(output).shape}")
    return logits


def format_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def format_c_array(name: str, values: np.ndarray) -> List[str]:
    flat = values.reshape(-1)
    lines = [f"static const float {name}[TINY_CLASSIFIER_INPUT_SIZE] = {{"]
    for start in range(0, flat.size, 8):
        chunk = ", ".join(format_float(float(value)) for value in flat[start : start + 8])
        lines.append(f"    {chunk},")
    lines.append("};")
    return lines


def generate_header(rows: Sequence[Dict[str, object]]) -> str:
    lines: List[str] = [
        "#ifndef TINY_CLASSIFIER_TEST_VECTORS_H",
        "#define TINY_CLASSIFIER_TEST_VECTORS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define TINY_CLASSIFIER_TEST_VECTOR_COUNT {len(rows)}",
        f"#define TINY_CLASSIFIER_INPUT_C {INPUT_C}",
        f"#define TINY_CLASSIFIER_INPUT_H {INPUT_H}",
        f"#define TINY_CLASSIFIER_INPUT_W {INPUT_W}",
        f"#define TINY_CLASSIFIER_INPUT_SIZE {INPUT_SIZE}",
        f"#define TINY_CLASSIFIER_NUM_CLASSES {NUM_CLASSES}",
        "#define TINY_CLASSIFIER_TEST_HAS_EXPECTED_LOGITS 1",
        "",
    ]

    for index, row in enumerate(rows):
        lines.append(
            f"/* {index}: label={row['true_label']} class={row['true_class']} "
            f"top1={row['pc_top1']} path={row['image_path_rel']} */"
        )
        lines.extend(format_c_array(f"g_tiny_classifier_input_{index}", row["input_chw"]))
        lines.append("")

    lines.append("static const float *const g_tiny_classifier_inputs[TINY_CLASSIFIER_TEST_VECTOR_COUNT] = {")
    for index in range(len(rows)):
        lines.append(f"    g_tiny_classifier_input_{index},")
    lines.append("};")
    lines.append("")

    lines.append("static const uint8_t g_tiny_classifier_label[TINY_CLASSIFIER_TEST_VECTOR_COUNT] = {")
    lines.append("    " + ", ".join(str(int(row["true_label"])) for row in rows) + ",")
    lines.append("};")
    lines.append("")

    lines.append(
        "static const float g_tiny_classifier_expected_logits"
        "[TINY_CLASSIFIER_TEST_VECTOR_COUNT][TINY_CLASSIFIER_NUM_CLASSES] = {"
    )
    for row in rows:
        logits = ", ".join(format_float(float(value)) for value in row["logits"])
        lines.append(f"    {{ {logits} }},")
    lines.append("};")
    lines.append("")

    lines.append("static const uint8_t g_tiny_classifier_expected_top1[TINY_CLASSIFIER_TEST_VECTOR_COUNT] = {")
    lines.append("    " + ", ".join(str(int(row["pc_top1"])) for row in rows) + ",")
    lines.append("};")
    lines.append("")

    lines.append("#endif /* TINY_CLASSIFIER_TEST_VECTORS_H */")
    lines.append("")
    return "\n".join(lines)


def write_manifest(output_path: Path, rows: Sequence[Dict[str, object]]) -> Path:
    manifest_path = output_path.parent / "tiny_classifier_test_vectors_manifest.csv"
    fieldnames = [
        "index",
        "image_path",
        "true_label",
        "true_class",
        "pc_top1",
        "pc_top1_class",
        "pc_top1_prob",
        "logits",
        "probs",
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "index": index,
                    "image_path": row["image_path_rel"],
                    "true_label": row["true_label"],
                    "true_class": row["true_class"],
                    "pc_top1": row["pc_top1"],
                    "pc_top1_class": row["pc_top1_class"],
                    "pc_top1_prob": f"{float(row['pc_top1_prob']):.9g}",
                    "logits": " ".join(f"{float(value):.9g}" for value in row["logits"]),
                    "probs": " ".join(f"{float(value):.9g}" for value in row["probs"]),
                }
            )
    return manifest_path


def main() -> None:
    args = parse_args()
    require_module("onnxruntime")
    import onnxruntime as ort

    onnx_path = resolve_path(args.onnx)
    checkpoint_path = resolve_path(args.checkpoint)
    output_path = resolve_path(args.output)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    class_to_idx, mean, std, input_size = resolve_metadata(checkpoint)
    if input_size != 96:
        print(f"[WARN] Metadata input_size/image_size={input_size}; forcing STM32 tiny classifier vector size to 96.")
        input_size = 96

    class_names = class_names_in_training_order(class_to_idx)
    if len(class_names) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes, got {len(class_names)}: {class_names}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rows = select_samples(class_names, args.samples_per_class, args.seed)

    print("=" * 80)
    print("Generate Tiny Classifier STM32 Test Vectors")
    print("=" * 80)
    print(f"[INFO] ONNX: {onnx_path}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] class_to_idx: {class_to_idx}")
    print(f"[INFO] mean={mean}, std={std}, input_size={input_size}")
    print()

    for index, row in enumerate(rows):
        input_chw = preprocess_image(row["image_path"], mean, std, input_size)
        logits = run_onnx(session, input_name, input_chw)
        probs = softmax(logits)
        top1 = int(np.argmax(probs))

        row["input_chw"] = input_chw
        row["logits"] = logits
        row["probs"] = probs
        row["pc_top1"] = top1
        row["pc_top1_class"] = class_names[top1]
        row["pc_top1_prob"] = float(probs[top1])
        row["image_path_rel"] = row["image_path"].relative_to(PROJECT_ROOT).as_posix()

        print(
            f"{index}, {row['image_path_rel']}, true_class={row['true_class']}, "
            f"pc_top1_class={row['pc_top1_class']}, pc_top1_prob={row['pc_top1_prob']:.6f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_header(rows), encoding="utf-8", newline="\n")
    manifest_path = write_manifest(output_path, rows)

    print("\n[INFO] Generated:")
    print(f"  header: {output_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"  manifest: {manifest_path.relative_to(PROJECT_ROOT).as_posix()}")
    print("\n将 tiny_classifier_test_vectors.h 复制到 STM32 工程：")
    print("edgeai/Core/Inc/tiny_classifier_test_vectors.h")
    print("然后重新编译烧录。")


if __name__ == "__main__":
    main()

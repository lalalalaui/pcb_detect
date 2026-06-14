import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pcb_dataset import IMAGENET_MEAN, IMAGENET_STD
from models.tiny_pcb_classifier import build_tiny_pcb_classifier_96


IMAGE_SIZE = 96
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
REPORT_PATH = PROJECT_ROOT / "results" / "tables" / "tiny_classifier_onnx_pytorch_compare.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tiny_classifier_96 PyTorch and ONNX outputs.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/classifier/tiny_classifier_96_best.pth",
        help="Tiny classifier checkpoint.",
    )
    parser.add_argument(
        "--onnx",
        default="deployment/onnx/classifier_tiny_classifier_96.onnx",
        help="Tiny classifier ONNX path.",
    )
    parser.add_argument("--samples-per-class", type=int, default=2, help="Processed images per class.")
    return parser.parse_args()


def require_module(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"Missing dependency: {module_name}")


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def preprocess_image(image_path: Path) -> np.ndarray:
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return ((chw - mean) / std).astype(np.float32)[None, ...]


def list_images(directory: Path) -> List[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_samples(samples_per_class: int) -> List[Dict[str, object]]:
    data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_cls" / "test"
    class_names = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    rows = []
    for class_index, class_name in enumerate(class_names):
        for image_path in list_images(data_dir / class_name)[:samples_per_class]:
            rows.append({"sample_id": image_path.relative_to(PROJECT_ROOT).as_posix(), "label": class_index, "path": image_path})
    return rows


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a_flat, b_flat) / denom)


def compare_outputs(torch_output: np.ndarray, onnx_output: np.ndarray) -> Dict[str, object]:
    diff = np.abs(torch_output - onnx_output)
    torch_argmax = int(np.argmax(torch_output, axis=1)[0])
    onnx_argmax = int(np.argmax(onnx_output, axis=1)[0])
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine_similarity": cosine_similarity(torch_output, onnx_output),
        "torch_argmax": torch_argmax,
        "onnx_argmax": onnx_argmax,
        "argmax_match": torch_argmax == onnx_argmax,
    }


def main() -> None:
    args = parse_args()
    require_module("onnxruntime")
    import onnxruntime as ort

    checkpoint_path = resolve_path(args.checkpoint)
    onnx_path = resolve_path(args.onnx)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = int(checkpoint.get("num_classes", 6))
    model = build_tiny_pcb_classifier_96(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    test_inputs = [{"sample_id": "random_float32_seed_2026", "label": "", "array": np.random.default_rng(2026).standard_normal((1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)}]
    for sample in collect_samples(args.samples_per_class):
        sample["array"] = preprocess_image(sample["path"])
        test_inputs.append(sample)

    rows = []
    with torch.no_grad():
        for sample in test_inputs:
            input_np = sample["array"]
            torch_output = model(torch.from_numpy(input_np)).detach().cpu().numpy()
            onnx_output = np.asarray(session.run(None, {input_name: input_np})[0])
            metrics = compare_outputs(torch_output, onnx_output)
            row = {
                "sample_id": sample["sample_id"],
                "label": sample["label"],
                "torch_output_shape": list(torch_output.shape),
                "onnx_output_shape": list(onnx_output.shape),
                **metrics,
            }
            rows.append(row)
            print(
                f"{row['sample_id']} max_abs_diff={metrics['max_abs_diff']:.10f} "
                f"mean_abs_diff={metrics['mean_abs_diff']:.10f} argmax_match={metrics['argmax_match']}"
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nReport saved to: {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

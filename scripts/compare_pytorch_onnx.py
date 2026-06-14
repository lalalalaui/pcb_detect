import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_PATH = PROJECT_ROOT / "results" / "tables" / "onnx_pytorch_compare.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)
DIFF_ATOL = 1e-4
DIFF_RTOL = 1e-4


def require_module(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"Missing dependency: {module_name}")


def load_checkpoint(path: Path, torch_module: Any) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch_module.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model_state_dict": checkpoint}


def infer_num_classes_from_state_dict(state_dict: Dict[str, Any]) -> int:
    for key in ("fc.weight", "classifier.1.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return 6


def resize_to_tensor(image_path: Path, image_size: int, normalize: bool) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))
    if normalize:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.astype(np.float32)[None, ...]


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_anomaly_images(max_per_label: int = 3) -> List[Path]:
    data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_anomaly" / "val"
    paths: List[Path] = []
    for label in ("normal", "anomaly"):
        paths.extend(list_images(data_dir / label)[:max_per_label])
    return paths


def collect_classification_images(max_images: int = 6) -> List[Path]:
    data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_cls" / "val"
    paths: List[Path] = []
    if not data_dir.exists():
        return paths

    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        images = list_images(class_dir)
        if images:
            paths.append(images[0])
        if len(paths) >= max_images:
            break
    return paths


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a_flat, b_flat) / denom)


def compare_arrays(torch_output: np.ndarray, onnx_output: np.ndarray) -> Tuple[float, float, float]:
    diff = np.abs(torch_output - onnx_output)
    return float(diff.max()), float(diff.mean()), cosine_similarity(torch_output, onnx_output)


def possible_large_diff_reasons() -> str:
    return (
        "Possible causes: preprocessing mismatch; model.eval() not set; checkpoint loading mismatch; "
        "ONNX opset/export issue; batchnorm/dropout state mismatch."
    )


def row_status(max_abs_diff: float, mean_abs_diff: float) -> Tuple[str, str]:
    if max_abs_diff <= DIFF_ATOL and mean_abs_diff <= DIFF_RTOL:
        return "ok", ""
    return "warn", possible_large_diff_reasons()


def run_torch_model(model: Any, torch_module: Any, input_np: np.ndarray) -> np.ndarray:
    with torch_module.no_grad():
        input_tensor = torch_module.from_numpy(input_np).to(dtype=torch_module.float32)
        output = model(input_tensor)
    return output.detach().cpu().numpy()


def run_onnx_model(session: Any, input_np: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_np.astype(np.float32)})
    return np.asarray(outputs[0])


def make_base_row(
    model_name: str,
    sample_type: str,
    sample_id: str,
    input_np: np.ndarray,
    torch_output: np.ndarray,
    onnx_output: np.ndarray,
) -> Dict[str, Any]:
    max_abs_diff, mean_abs_diff, cosine = compare_arrays(torch_output, onnx_output)
    status, notes = row_status(max_abs_diff, mean_abs_diff)
    return {
        "model": model_name,
        "sample_type": sample_type,
        "sample_id": sample_id,
        "input_shape": list(input_np.shape),
        "torch_output_shape": list(torch_output.shape),
        "onnx_output_shape": list(onnx_output.shape),
        "max_abs_diff": f"{max_abs_diff:.10f}",
        "mean_abs_diff": f"{mean_abs_diff:.10f}",
        "cosine_similarity": f"{cosine:.10f}",
        "torch_reconstruction_error": "",
        "onnx_reconstruction_error": "",
        "reconstruction_error_abs_diff": "",
        "torch_argmax": "",
        "onnx_argmax": "",
        "argmax_match": "",
        "status": status,
        "notes": notes,
    }


def compare_tiny_autoencoder(
    torch_module: Any,
    ort_module: Any,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    from models.tiny_autoencoder import TinyAutoEncoder

    checkpoint_path = PROJECT_ROOT / "checkpoints" / "anomaly" / "tiny_ae_best.pth"
    onnx_path = PROJECT_ROOT / "deployment" / "onnx" / "anomaly_tiny_ae_96.onnx"
    checkpoint = load_checkpoint(checkpoint_path, torch_module)

    model = TinyAutoEncoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    session = ort_module.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    samples: List[Tuple[str, str, np.ndarray]] = [
        ("random", "random_float32_seed_2026", rng.random((1, 3, 96, 96), dtype=np.float32))
    ]
    for image_path in collect_anomaly_images():
        sample_id = image_path.relative_to(PROJECT_ROOT).as_posix()
        samples.append(("processed_image", sample_id, resize_to_tensor(image_path, 96, normalize=False)))

    rows: List[Dict[str, Any]] = []
    for sample_type, sample_id, input_np in samples:
        torch_output = run_torch_model(model, torch_module, input_np)
        onnx_output = run_onnx_model(session, input_np)
        row = make_base_row("TinyAutoEncoder", sample_type, sample_id, input_np, torch_output, onnx_output)

        torch_error = float(np.mean((torch_output - input_np) ** 2))
        onnx_error = float(np.mean((onnx_output - input_np) ** 2))
        row["torch_reconstruction_error"] = f"{torch_error:.10f}"
        row["onnx_reconstruction_error"] = f"{onnx_error:.10f}"
        row["reconstruction_error_abs_diff"] = f"{abs(torch_error - onnx_error):.10f}"
        rows.append(row)

    return rows


def compare_mobilenet_v2(
    torch_module: Any,
    ort_module: Any,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    from models.mobilenet_classifier import build_mobilenet_v2

    checkpoint_path = PROJECT_ROOT / "checkpoints" / "classifier" / "mobilenet_v2_best.pth"
    onnx_path = PROJECT_ROOT / "deployment" / "onnx" / "classifier_mobilenet_v2_224.onnx"
    checkpoint = load_checkpoint(checkpoint_path, torch_module)
    state_dict = checkpoint["model_state_dict"]
    num_classes = int(checkpoint.get("num_classes", infer_num_classes_from_state_dict(state_dict)))

    model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    session = ort_module.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    samples: List[Tuple[str, str, np.ndarray]] = [
        ("random", "random_float32_seed_2026", rng.standard_normal((1, 3, 224, 224)).astype(np.float32))
    ]
    for image_path in collect_classification_images(max_images=num_classes):
        sample_id = image_path.relative_to(PROJECT_ROOT).as_posix()
        samples.append(("processed_image", sample_id, resize_to_tensor(image_path, 224, normalize=True)))

    rows: List[Dict[str, Any]] = []
    for sample_type, sample_id, input_np in samples:
        torch_output = run_torch_model(model, torch_module, input_np)
        onnx_output = run_onnx_model(session, input_np)
        row = make_base_row("MobileNetV2 classifier", sample_type, sample_id, input_np, torch_output, onnx_output)

        torch_argmax = int(np.argmax(torch_output, axis=1)[0])
        onnx_argmax = int(np.argmax(onnx_output, axis=1)[0])
        row["torch_argmax"] = torch_argmax
        row["onnx_argmax"] = onnx_argmax
        row["argmax_match"] = torch_argmax == onnx_argmax
        if torch_argmax != onnx_argmax:
            row["status"] = "warn"
            row["notes"] = possible_large_diff_reasons()
        rows.append(row)

    return rows


def print_rows(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for row in rows:
        print(f"sample: {row['sample_id']}")
        print(f"  type: {row['sample_type']}")
        print(f"  max_abs_diff: {row['max_abs_diff']}")
        print(f"  mean_abs_diff: {row['mean_abs_diff']}")
        print(f"  cosine_similarity: {row['cosine_similarity']}")
        if row["torch_reconstruction_error"] != "":
            print(f"  torch_reconstruction_error: {row['torch_reconstruction_error']}")
            print(f"  onnx_reconstruction_error: {row['onnx_reconstruction_error']}")
            print(f"  reconstruction_error_abs_diff: {row['reconstruction_error_abs_diff']}")
        if row["torch_argmax"] != "":
            print(f"  torch_argmax: {row['torch_argmax']}")
            print(f"  onnx_argmax: {row['onnx_argmax']}")
            print(f"  argmax_match: {row['argmax_match']}")
        print(f"  status: {row['status']}")
        if row["notes"]:
            print(f"  notes: {row['notes']}")


def write_csv(rows: Sequence[Dict[str, Any]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "sample_type",
        "sample_id",
        "input_shape",
        "torch_output_shape",
        "onnx_output_shape",
        "max_abs_diff",
        "mean_abs_diff",
        "cosine_similarity",
        "torch_reconstruction_error",
        "onnx_reconstruction_error",
        "reconstruction_error_abs_diff",
        "torch_argmax",
        "onnx_argmax",
        "argmax_match",
        "status",
        "notes",
    ]
    with REPORT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 80)
    print("Compare PyTorch Checkpoints With ONNX Models")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print("[INFO] Device: CPU")
    print("[INFO] ONNX Runtime provider: CPUExecutionProvider")

    for module_name in ("torch", "onnxruntime"):
        require_module(module_name)

    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(2026)
    all_rows: List[Dict[str, Any]] = []

    tiny_rows = compare_tiny_autoencoder(torch, ort, rng)
    print_rows("A. TinyAutoEncoder", tiny_rows)
    all_rows.extend(tiny_rows)

    classifier_rows = compare_mobilenet_v2(torch, ort, rng)
    print_rows("B. MobileNetV2 classifier", classifier_rows)
    all_rows.extend(classifier_rows)

    write_csv(all_rows)
    warn_rows = [row for row in all_rows if row["status"] != "ok"]

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Compared rows: {len(all_rows)}")
    print(f"Warnings: {len(warn_rows)}")
    print(f"CSV saved to: {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    if warn_rows:
        print(possible_large_diff_reasons())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] PyTorch/ONNX comparison failed: {exc}")
        raise

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.autoencoder import ConvAutoEncoder
from models.mobilenet_classifier import build_mobilenet_v2
from models.resnet_classifier import build_resnet18
from models.tiny_autoencoder import TinyAutoEncoder


CLASSIFIER_MODELS = {"resnet18", "mobilenet_v2"}
ANOMALY_MODELS = {"autoencoder", "tiny_ae"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure PCB model inference latency.")
    parser.add_argument(
        "--task",
        choices=["classifier", "anomaly"],
        required=True,
        help="Model task type.",
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "mobilenet_v2", "autoencoder", "tiny_ae"],
        required=True,
        help="Model architecture.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Input image size. Defaults to 224 for classifier and 128 for anomaly.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--runs", type=int, default=100, help="Measured iterations.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, cpu, or a torch device such as cuda:0.",
    )
    return parser.parse_args()


def resolve_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def validate_task_model(task: str, model_name: str) -> None:
    if task == "classifier" and model_name not in CLASSIFIER_MODELS:
        raise ValueError("--task classifier requires --model resnet18 or mobilenet_v2.")
    if task == "anomaly" and model_name not in ANOMALY_MODELS:
        raise ValueError("--task anomaly requires --model autoencoder or tiny_ae.")


def default_image_size(task: str) -> int:
    return 224 if task == "classifier" else 128


def checkpoint_path_for(task: str, model_name: str) -> Path:
    if task == "classifier":
        return PROJECT_ROOT / "checkpoints" / "classifier" / f"{model_name}_best.pth"
    return PROJECT_ROOT / "checkpoints" / "anomaly" / f"{model_name}_best.pth"


def load_checkpoint_file(checkpoint_path: Path, device: torch.device) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please train the requested model before measuring latency."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model_state_dict": checkpoint}


def infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    for key in ("fc.weight", "classifier.1.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return 6


def build_model(task: str, model_name: str, checkpoint: Dict) -> torch.nn.Module:
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if task == "classifier":
        num_classes = int(checkpoint.get("num_classes", infer_num_classes_from_state_dict(state_dict)))
        if model_name == "resnet18":
            return build_resnet18(num_classes=num_classes, pretrained=False)
        if model_name == "mobilenet_v2":
            return build_mobilenet_v2(num_classes=num_classes, pretrained=False)

    if task == "anomaly":
        if model_name == "autoencoder":
            return ConvAutoEncoder()
        if model_name == "tiny_ae":
            return TinyAutoEncoder()

    raise ValueError(f"Unsupported task/model pair: {task}/{model_name}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    warmup: int,
    runs: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        synchronize_if_needed(device)

        start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy_input)
        synchronize_if_needed(device)
        elapsed = time.perf_counter() - start

    avg_latency_ms = elapsed * 1000.0 / runs
    fps = dummy_input.size(0) * 1000.0 / avg_latency_ms
    return avg_latency_ms, fps


def append_latency_csv(csv_path: Path, row: Dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = [
        "timestamp",
        "task",
        "model",
        "device",
        "image_size",
        "batch_size",
        "warmup",
        "runs",
        "average_latency_ms",
        "fps",
        "parameter_count",
        "model_size_mb",
        "checkpoint",
    ]

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    validate_task_model(args.task, args.model)

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")

    image_size = args.image_size if args.image_size is not None else default_image_size(args.task)
    if image_size <= 0:
        raise ValueError("--image_size must be positive.")

    device = resolve_device(args.device)
    checkpoint_path = checkpoint_path_for(args.task, args.model)

    print("=" * 80)
    print("Measure PCB Model Inference Latency")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Image size: {image_size}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Warmup: {args.warmup}")
    print(f"[INFO] Runs: {args.runs}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint_file(checkpoint_path, device)
    model = build_model(args.task, args.model, checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dummy_input = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
    param_count = count_parameters(model)
    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    avg_latency_ms, fps = measure_latency(
        model=model,
        dummy_input=dummy_input,
        warmup=args.warmup,
        runs=args.runs,
        device=device,
    )

    print("\n[RESULT] Latency baseline")
    print(f"  average latency ms: {avg_latency_ms:.4f}")
    print(f"  FPS: {fps:.2f}")
    print(f"  parameter count: {param_count}")
    print(f"  model size MB: {model_size_mb:.2f}")

    csv_path = PROJECT_ROOT / "results" / "tables" / "latency_results.csv"
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": args.task,
        "model": args.model,
        "device": str(device),
        "image_size": image_size,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "runs": args.runs,
        "average_latency_ms": f"{avg_latency_ms:.6f}",
        "fps": f"{fps:.6f}",
        "parameter_count": param_count,
        "model_size_mb": f"{model_size_mb:.6f}",
        "checkpoint": str(checkpoint_path),
    }
    append_latency_csv(csv_path, row)
    print(f"\n[INFO] Appended CSV: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] Latency measurement failed: {exc}")
        raise

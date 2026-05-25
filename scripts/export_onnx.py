import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict

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
    parser = argparse.ArgumentParser(description="Export PCB models to ONNX.")
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
    parser.add_argument("--image_size", type=int, required=True, help="Input image size.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to checkpoints/{classifier|anomaly}/{model_name}_best.pth.",
    )
    parser.add_argument(
        "--output_dir",
        default="deployment/onnx",
        help="Output directory for ONNX files.",
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def validate_task_model(task: str, model_name: str) -> None:
    if task == "classifier" and model_name not in CLASSIFIER_MODELS:
        raise ValueError("--task classifier requires --model resnet18 or mobilenet_v2.")
    if task == "anomaly" and model_name not in ANOMALY_MODELS:
        raise ValueError("--task anomaly requires --model autoencoder or tiny_ae.")


def default_checkpoint_path(task: str, model_name: str) -> Path:
    if task == "classifier":
        return PROJECT_ROOT / "checkpoints" / "classifier" / f"{model_name}_best.pth"
    return PROJECT_ROOT / "checkpoints" / "anomaly" / f"{model_name}_best.pth"


def load_checkpoint(checkpoint_path: Path) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please train the requested model before exporting ONNX."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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


def ensure_onnx_available() -> None:
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "ONNX is not installed. Please install it first:\n"
            "  pip install onnx"
        )


def main() -> None:
    args = parse_args()
    validate_task_model(args.task, args.model)

    if args.image_size <= 0:
        raise ValueError("--image_size must be positive.")
    if args.opset <= 0:
        raise ValueError("--opset must be positive.")

    print("=" * 80)
    print("Stage 6 - Export PCB Model to ONNX")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Image size: {args.image_size}")
    print(f"[INFO] Opset: {args.opset}")

    ensure_onnx_available()

    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else default_checkpoint_path(args.task, args.model)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{args.task}_{args.model}_{args.image_size}.onnx"

    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Output dir: {output_dir}")

    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model(args.task, args.model, checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print("\n[INFO] ONNX export finished")
    print(f"ONNX path: {onnx_path}")
    print(f"File size MB: {size_mb:.2f}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        message = str(exc)
        if "pip install onnx" in message:
            print(f"\n[ERROR] {message}")
        else:
            print(f"\n[ERROR] ONNX export failed: {exc}")
            raise
    except Exception as exc:
        print(f"\n[ERROR] ONNX export failed: {exc}")
        raise

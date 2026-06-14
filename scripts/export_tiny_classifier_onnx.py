import argparse
import importlib.util
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tiny_pcb_classifier import build_tiny_pcb_classifier_96, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tiny_classifier_96 to ONNX.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/classifier/tiny_classifier_96_best.pth",
        help="Tiny classifier checkpoint path.",
    )
    parser.add_argument("--image_size", type=int, default=96, help="Input image size. Keep 96.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset.")
    parser.add_argument(
        "--output",
        default="deployment/onnx/classifier_tiny_classifier_96.onnx",
        help="Output ONNX path.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_onnx_available() -> None:
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError("ONNX is not installed. Install with: pip install onnx")


def main() -> None:
    args = parse_args()
    if args.image_size != 96:
        raise ValueError("tiny_classifier_96 is intended for --image_size 96.")
    if args.opset <= 0:
        raise ValueError("--opset must be positive.")

    ensure_onnx_available()
    checkpoint_path = resolve_path(args.checkpoint)
    output_path = resolve_path(args.output)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train first with: python scripts/train_tiny_classifier.py"
        )

    print("=" * 80)
    print("Export Tiny PCB Classifier 96 to ONNX")
    print("=" * 80)
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Output: {output_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = int(checkpoint.get("num_classes", 6))
    model = build_tiny_pcb_classifier_96(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[INFO] Trainable parameters: {count_parameters(model)}")

    dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    size_kb = output_path.stat().st_size / 1024.0
    print("\n[INFO] ONNX export finished")
    print(f"ONNX path: {output_path}")
    print(f"File size KB: {size_kb:.2f}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pcb_dataset import get_anomaly_dataloaders
from models.autoencoder import ConvAutoEncoder
from models.tiny_autoencoder import TinyAutoEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PCB anomaly heatmaps with a trained AutoEncoder.")
    parser.add_argument(
        "--model",
        choices=["autoencoder", "tiny_ae"],
        default="autoencoder",
        help="AutoEncoder architecture.",
    )
    parser.add_argument("--image_size", type=int, default=128, help="Input image size.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of test samples to visualize.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to checkpoints/anomaly/{model_name}_best.pth.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, cpu, or a torch device such as cuda:0.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "autoencoder":
        return ConvAutoEncoder()
    if model_name == "tiny_ae":
        return TinyAutoEncoder()
    raise ValueError(f"Unsupported model: {model_name}")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please train an anomaly model first, for example:\n"
            "  python scripts\\train_autoencoder.py --model autoencoder --epochs 5 --batch_size 32 --image_size 128"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def normalize_error_map(error_map: torch.Tensor) -> np.ndarray:
    error = error_map.detach().cpu()
    min_value = float(error.min().item())
    max_value = float(error.max().item())
    if max_value - min_value < 1e-12:
        return torch.zeros_like(error).numpy()
    return ((error - min_value) / (max_value - min_value)).numpy()


def collect_anomaly_first_samples(test_loader, num_samples: int) -> List[Tuple[torch.Tensor, int, str]]:
    anomaly_samples: List[Tuple[torch.Tensor, int, str]] = []
    fallback_samples: List[Tuple[torch.Tensor, int, str]] = []

    for images, labels, paths in test_loader:
        for image, label, path in zip(images, labels.tolist(), paths):
            item = (image, int(label), path)
            if label == 1 and len(anomaly_samples) < num_samples:
                anomaly_samples.append(item)
            elif len(fallback_samples) < num_samples:
                fallback_samples.append(item)

        if len(anomaly_samples) >= num_samples:
            break

    if len(anomaly_samples) >= num_samples:
        return anomaly_samples[:num_samples]

    needed = num_samples - len(anomaly_samples)
    return anomaly_samples + fallback_samples[:needed]


def generate_heatmap_rows(
    model: torch.nn.Module,
    samples: Sequence[Tuple[torch.Tensor, int, str]],
    device: torch.device,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
    model.eval()
    rows = []

    with torch.no_grad():
        for image, label, _ in samples:
            image_batch = image.unsqueeze(0).to(device, non_blocking=True)
            recon = model(image_batch).squeeze(0).cpu()
            error_map = torch.mean((image - recon) ** 2, dim=0)
            score = float(error_map.mean().item())

            rows.append(
                (
                    tensor_to_image(image),
                    tensor_to_image(recon),
                    normalize_error_map(error_map),
                    label,
                    score,
                )
            )

    return rows


def save_heatmap_grid(
    output_path: Path,
    rows: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]],
) -> None:
    if not rows:
        raise RuntimeError("No samples were collected for heatmap visualization.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_rows = len(rows)
    fig, axes = plt.subplots(num_rows, 3, figsize=(9, max(2.2 * num_rows, 3)))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    label_names = {0: "normal", 1: "anomaly"}
    for row_idx, (image, recon, error_map, label, score) in enumerate(rows):
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"Original\n{label_names.get(label, label)}", fontsize=8)

        axes[row_idx, 1].imshow(recon)
        axes[row_idx, 1].set_title(f"Reconstruction\nscore={score:.6f}", fontsize=8)

        axes[row_idx, 2].imshow(image)
        axes[row_idx, 2].imshow(error_map, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[row_idx, 2].set_title("Heatmap Overlay", fontsize=8)

        for col in range(3):
            axes[row_idx, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("--image_size must be positive.")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")

    device = resolve_device(args.device)
    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint is not None
        else PROJECT_ROOT / "checkpoints" / "anomaly" / f"{args.model}_best.pth"
    )
    data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_anomaly"
    output_path = PROJECT_ROOT / "results" / "heatmaps" / f"{args.model}_heatmap_samples.png"

    print("=" * 80)
    print("Stage 5 - Generate PCB Anomaly Heatmaps")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Data directory: {data_dir}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Image size: {args.image_size}")
    print(f"[INFO] Num samples: {args.num_samples}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")

    if not data_dir.exists():
        raise FileNotFoundError(
            "Anomaly dataset was not found. Please run:\n"
            "  python scripts\\prepare_data.py --patch_size 128 --padding_ratio 0.5 --overwrite"
        )

    _, _, test_loader = get_anomaly_dataloaders(
        data_dir=str(data_dir),
        image_size=args.image_size,
        batch_size=32,
        num_workers=0,
    )
    print(f"[INFO] Test samples: {len(test_loader.dataset)}")

    model = build_model(args.model).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    checkpoint_image_size = checkpoint.get("image_size")
    if checkpoint_image_size is not None and checkpoint_image_size != args.image_size:
        print(f"[WARN] Checkpoint image_size={checkpoint_image_size}, generating with image_size={args.image_size}.")

    samples = collect_anomaly_first_samples(test_loader, args.num_samples)
    anomaly_count = sum(1 for _, label, _ in samples if label == 1)
    print(f"[INFO] Collected samples: {len(samples)} (anomaly: {anomaly_count})")

    rows = generate_heatmap_rows(model, samples, device)
    save_heatmap_grid(output_path, rows)

    print("\n[INFO] Heatmap generation finished")
    print(f"[INFO] Output file: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] Heatmap generation failed: {exc}")
        raise

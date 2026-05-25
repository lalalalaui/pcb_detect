import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pcb_dataset import get_anomaly_dataloaders
from models.autoencoder import ConvAutoEncoder, count_parameters as count_ae_parameters
from models.tiny_autoencoder import TinyAutoEncoder, count_parameters as count_tiny_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCB AutoEncoder anomaly detector on normal patches.")
    parser.add_argument(
        "--model",
        choices=["autoencoder", "tiny_ae"],
        default="autoencoder",
        help="AutoEncoder architecture.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--image_size", type=int, default=128, help="Input image size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers. Use 0 on Windows.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, cpu, or a torch device such as cuda:0.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/anomaly",
        help="Directory for AutoEncoder checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/anomaly",
        help="Directory for AutoEncoder logs and samples.",
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


def ensure_anomaly_data_exists(data_dir: Path) -> None:
    required_dirs = [
        data_dir / "train" / "normal",
        data_dir / "val" / "normal",
        data_dir / "val" / "anomaly",
    ]
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Anomaly dataset was not found. Please run:\n"
            "  python scripts\\prepare_data.py --patch_size 128 --padding_ratio 0.5 --overwrite\n"
            f"Missing directories:\n{missing_text}"
        )


def build_model(model_name: str) -> Tuple[nn.Module, int]:
    if model_name == "autoencoder":
        model = ConvAutoEncoder()
        return model, count_ae_parameters(model)
    if model_name == "tiny_ae":
        model = TinyAutoEncoder()
        return model, count_tiny_parameters(model)
    raise ValueError(f"Unsupported model: {model_name}")


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels, _ in train_loader:
        if torch.any(labels != 0):
            raise RuntimeError("Anomaly train_loader must contain normal samples only, but found non-zero labels.")

        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        recon = model(images)
        loss = criterion(recon, images)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate_reconstruction_errors(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    normal_errors: List[float] = []
    anomaly_errors: List[float] = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            recon = model(images)
            errors = torch.mean((recon - images) ** 2, dim=(1, 2, 3)).cpu()

            for error, label in zip(errors.tolist(), labels.tolist()):
                if label == 0:
                    normal_errors.append(error)
                elif label == 1:
                    anomaly_errors.append(error)

    normal_mean = float(np.mean(normal_errors)) if normal_errors else float("nan")
    anomaly_mean = float(np.mean(anomaly_errors)) if anomaly_errors else float("nan")
    return normal_mean, anomaly_mean


def save_log_csv(log_path: Path, history: Sequence[Dict[str, float]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_normal_error", "val_anomaly_error"],
        )
        writer.writeheader()
        writer.writerows(history)


def save_loss_curve(curve_path: Path, history: Sequence[Dict[str, float]], model_name: str) -> None:
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train loss")
    plt.plot(epochs, [row["val_normal_error"] for row in history], marker="o", label="val normal error")
    plt.plot(epochs, [row["val_anomaly_error"] for row in history], marker="o", label="val anomaly error")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"{model_name} AutoEncoder Reconstruction Error")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def collect_reconstruction_samples(
    model: nn.Module,
    val_loader,
    device: torch.device,
    max_per_label: int = 4,
) -> List[Tuple[torch.Tensor, torch.Tensor, int, float]]:
    model.eval()
    samples: List[Tuple[torch.Tensor, torch.Tensor, int, float]] = []
    label_counts = {0: 0, 1: 0}

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images_device = images.to(device, non_blocking=True)
            recon_device = model(images_device)
            errors = torch.mean((recon_device - images_device) ** 2, dim=(1, 2, 3)).cpu()
            recon = recon_device.cpu()

            for idx, label in enumerate(labels.tolist()):
                if label not in label_counts or label_counts[label] >= max_per_label:
                    continue
                samples.append((images[idx], recon[idx], label, float(errors[idx].item())))
                label_counts[label] += 1

            if all(count >= max_per_label for count in label_counts.values()):
                break

    return samples


def save_reconstruction_samples(
    output_path: Path,
    samples: Sequence[Tuple[torch.Tensor, torch.Tensor, int, float]],
) -> None:
    if not samples:
        print("[WARN] No reconstruction samples available to plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = len(samples)
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    label_names = {0: "normal", 1: "anomaly"}
    for col, (image, recon, label, error) in enumerate(samples):
        error_map = torch.mean(torch.abs(recon - image), dim=0)

        axes[0, col].imshow(tensor_to_image(image))
        axes[0, col].set_title(f"{label_names.get(label, label)}\nerr={error:.5f}", fontsize=8)
        axes[1, col].imshow(tensor_to_image(recon))
        axes[1, col].set_title("recon", fontsize=8)
        axes[2, col].imshow(error_map.numpy(), cmap="magma")
        axes[2, col].set_title("abs error", fontsize=8)

        for row in range(rows):
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.image_size <= 0:
        raise ValueError("--image_size must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")

    data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_anomaly"
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    output_dir = resolve_path(args.output_dir)
    curves_dir = PROJECT_ROOT / "results" / "curves"
    device = resolve_device(args.device)

    print("=" * 80)
    print("Stage 4 - Train PCB AutoEncoder Anomaly Detector")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Data directory: {data_dir}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] epochs={args.epochs}, batch_size={args.batch_size}, image_size={args.image_size}, lr={args.lr}")

    ensure_anomaly_data_exists(data_dir)
    train_loader, val_loader, _ = get_anomaly_dataloaders(
        data_dir=str(data_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Batches - train: {len(train_loader)}, val: {len(val_loader)}")
    print(f"[INFO] Train samples: {len(train_loader.dataset)}, val samples: {len(val_loader.dataset)}")

    model, param_count = build_model(args.model)
    model = model.to(device)
    print(f"[INFO] Trainable parameters: {param_count}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"{args.model}_best.pth"
    log_path = output_dir / f"{args.model}_training_log.csv"
    curve_path = curves_dir / f"{args.model}_ae_loss_curve.png"
    sample_path = output_dir / f"{args.model}_reconstruction_samples.png"

    history: List[Dict[str, float]] = []
    best_val_normal_error = float("inf")

    print("\n[INFO] Start training")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_normal_error, val_anomaly_error = evaluate_reconstruction_errors(model, val_loader, device)

        if val_normal_error < best_val_normal_error:
            best_val_normal_error = val_normal_error
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_normal_error": val_normal_error,
                    "val_anomaly_error": val_anomaly_error,
                    "param_count": param_count,
                },
                best_checkpoint_path,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_normal_error": val_normal_error,
            "val_anomaly_error": val_anomaly_error,
        }
        history.append(row)
        save_log_csv(log_path, history)

        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_normal_error={val_normal_error:.6f} "
            f"val_anomaly_error={val_anomaly_error:.6f} "
            f"best_val_normal_error={best_val_normal_error:.6f} "
            f"time={elapsed:.1f}s"
        )

    save_loss_curve(curve_path, history, args.model)

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    samples = collect_reconstruction_samples(model, val_loader, device)
    save_reconstruction_samples(sample_path, samples)

    print("\n[INFO] Training finished")
    print(f"[INFO] Best val normal error: {best_val_normal_error:.6f}")
    print(f"[INFO] Best checkpoint: {best_checkpoint_path}")
    print(f"[INFO] Training log: {log_path}")
    print(f"[INFO] Loss curve: {curve_path}")
    print(f"[INFO] Reconstruction samples: {sample_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] AutoEncoder training failed: {exc}")
        raise

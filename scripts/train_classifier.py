import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pcb_dataset import get_classification_dataloaders
from models.mobilenet_classifier import build_mobilenet_v2
from models.resnet_classifier import build_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCB defect classification baseline.")
    parser.add_argument(
        "--model",
        choices=["resnet18", "mobilenet_v2"],
        default="resnet18",
        help="Classifier architecture.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers. Use 0 on Windows.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, cpu, or a torch device such as cuda:0.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/classifier",
        help="Directory for classifier training logs.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/classifier",
        help="Directory for model checkpoints.",
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


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if model_name == "mobilenet_v2":
        return build_mobilenet_v2(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model: {model_name}")


def ensure_classification_data_exists(data_dir: Path) -> None:
    required_dirs = [data_dir / "train", data_dir / "val", data_dir / "test"]
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Classification dataset was not found. Please run:\n"
            "  python scripts\\prepare_data.py --patch_size 128 --padding_ratio 0.5 --overwrite\n"
            f"Missing directories:\n{missing_text}"
        )


def run_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def save_log_csv(log_path: Path, history: List[Dict[str, float]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def save_curve(
    output_path: Path,
    history: List[Dict[str, float]],
    train_key: str,
    val_key: str,
    ylabel: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row[train_key] for row in history], marker="o", label="train")
    plt.plot(epochs, [row[val_key] for row in history], marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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

    data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_cls"
    output_dir = resolve_path(args.output_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    curves_dir = PROJECT_ROOT / "results" / "curves"
    device = resolve_device(args.device)

    print("=" * 80)
    print("Stage 3 - Train PCB Supervised Classification Baseline")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Data directory: {data_dir}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Pretrained: {args.pretrained}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] epochs={args.epochs}, batch_size={args.batch_size}, image_size={args.image_size}, lr={args.lr}")

    ensure_classification_data_exists(data_dir)

    train_loader, val_loader, test_loader, class_names = get_classification_dataloaders(
        data_dir=str(data_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")
    print(f"[INFO] Batches - train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

    model = build_model(args.model, num_classes=num_classes, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"{args.model}_best.pth"
    log_path = output_dir / f"{args.model}_training_log.csv"
    loss_curve_path = curves_dir / f"{args.model}_loss_curve.png"
    acc_curve_path = curves_dir / f"{args.model}_accuracy_curve.png"

    history: List[Dict[str, float]] = []
    best_val_acc = -1.0

    print("\n[INFO] Start training")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_acc = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "pretrained": args.pretrained,
                },
                best_checkpoint_path,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        save_log_csv(log_path, history)

        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"best_val_acc={best_val_acc:.4f} time={elapsed:.1f}s"
        )

    save_curve(
        output_path=loss_curve_path,
        history=history,
        train_key="train_loss",
        val_key="val_loss",
        ylabel="Loss",
        title=f"{args.model} Loss",
    )
    save_curve(
        output_path=acc_curve_path,
        history=history,
        train_key="train_acc",
        val_key="val_acc",
        ylabel="Accuracy",
        title=f"{args.model} Accuracy",
    )

    print("\n[INFO] Training finished")
    print(f"[INFO] Best val acc: {best_val_acc:.4f}")
    print(f"[INFO] Best checkpoint: {best_checkpoint_path}")
    print(f"[INFO] Training log: {log_path}")
    print(f"[INFO] Loss curve: {loss_curve_path}")
    print(f"[INFO] Accuracy curve: {acc_curve_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] Classifier training failed: {exc}")
        raise

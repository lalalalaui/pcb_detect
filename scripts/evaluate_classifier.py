import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pcb_dataset import IMAGENET_MEAN, IMAGENET_STD, get_classification_dataloaders
from models.mobilenet_classifier import build_mobilenet_v2
from models.resnet_classifier import build_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PCB defect classification model.")
    parser.add_argument(
        "--model",
        choices=["resnet18", "mobilenet_v2"],
        default="resnet18",
        help="Classifier architecture.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers. Use 0 on Windows.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to checkpoints/classifier/{model_name}_best.pth.",
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


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=False)
    if model_name == "mobilenet_v2":
        return build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unsupported model: {model_name}")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please train a model first, for example:\n"
            "  python scripts\\train_classifier.py --model resnet18 --epochs 3 --batch_size 16 --image_size 224 --pretrained"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def evaluate(model: torch.nn.Module, test_loader, device: torch.device) -> Tuple[List[int], List[int], List[torch.Tensor], List[int], List[int]]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    sample_images: List[torch.Tensor] = []
    sample_true: List[int] = []
    sample_pred: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

            remaining = 16 - len(sample_images)
            if remaining > 0:
                take = min(remaining, images.size(0))
                sample_images.extend(images[:take].cpu())
                sample_true.extend(labels[:take].cpu().tolist())
                sample_pred.extend(preds[:take].cpu().tolist())

    return all_true, all_pred, sample_images, sample_true, sample_pred


def save_classification_report_csv(
    report_path: Path,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=list(range(len(class_names))),
        output_dict=True,
        zero_division=0,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_name", "precision", "recall", "f1-score", "support"],
        )
        writer.writeheader()
        for class_name in list(class_names) + ["accuracy", "macro avg", "weighted avg"]:
            values = report[class_name]
            if class_name == "accuracy":
                writer.writerow(
                    {
                        "class_name": class_name,
                        "precision": "",
                        "recall": "",
                        "f1-score": values,
                        "support": len(y_true),
                    }
                )
            else:
                writer.writerow(
                    {
                        "class_name": class_name,
                        "precision": values["precision"],
                        "recall": values["recall"],
                        "f1-score": values["f1-score"],
                        "support": values["support"],
                    }
                )


def save_confusion_matrix(
    output_path: Path,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = tensor * std + mean
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def save_sample_predictions(
    output_path: Path,
    images: Sequence[torch.Tensor],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> None:
    if not images:
        print("[WARN] No sample predictions available to plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 4
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes):
        ax.axis("off")
        if ax_idx >= len(images):
            continue

        true_name = class_names[y_true[ax_idx]]
        pred_name = class_names[y_pred[ax_idx]]
        ax.imshow(denormalize_image(images[ax_idx]))
        ax.set_title(f"P: {pred_name}\nT: {true_name}", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.image_size <= 0:
        raise ValueError("--image_size must be positive.")

    device = resolve_device(args.device)
    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint is not None
        else PROJECT_ROOT / "checkpoints" / "classifier" / f"{args.model}_best.pth"
    )
    results_dir = PROJECT_ROOT / "results" / "classifier"

    print("=" * 80)
    print("Evaluate PCB Supervised Classification Baseline")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")

    data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_cls"
    if not data_dir.exists():
        raise FileNotFoundError(
            "Classification dataset was not found. Please run:\n"
            "  python scripts\\prepare_data.py --patch_size 128 --padding_ratio 0.5 --overwrite"
        )

    _, _, test_loader, class_names = get_classification_dataloaders(
        data_dir=str(data_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Classes ({len(class_names)}): {class_names}")
    print(f"[INFO] Test batches: {len(test_loader)}")

    model = build_model(args.model, num_classes=len(class_names)).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    checkpoint_classes = checkpoint.get("class_names")
    if checkpoint_classes is not None and list(checkpoint_classes) != list(class_names):
        print(f"[WARN] Checkpoint class order differs from current dataset: {checkpoint_classes}")

    y_true, y_pred, sample_images, sample_true, sample_pred = evaluate(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    report_csv = results_dir / f"{args.model}_classification_report.csv"
    confusion_png = results_dir / f"{args.model}_confusion_matrix.png"
    samples_png = results_dir / f"{args.model}_sample_predictions.png"

    save_classification_report_csv(report_csv, y_true, y_pred, class_names)
    save_confusion_matrix(confusion_png, y_true, y_pred, class_names)
    save_sample_predictions(samples_png, sample_images, sample_true, sample_pred, class_names)

    print("\n[RESULT] Classification report")
    print(report_text)
    print(f"[RESULT] Accuracy: {acc:.4f}")

    print("\n[INFO] Output files")
    print(f"  report csv: {report_csv}")
    print(f"  confusion matrix: {confusion_png}")
    print(f"  sample predictions: {samples_png}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] Classifier evaluation failed: {exc}")
        raise

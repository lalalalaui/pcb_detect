import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pcb_dataset import get_classification_dataloaders
from models.tiny_pcb_classifier import build_tiny_pcb_classifier_96, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tiny 96x96 PCB classifier.")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--image_size", type=int, default=96, help="Input image size. Keep 96.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/classifier/tiny_classifier_96_best.pth",
        help="Tiny classifier checkpoint path.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu, etc.")
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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train first with: python scripts/train_tiny_classifier.py"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def evaluate(model: torch.nn.Module, dataloader, device: torch.device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


def save_report_csv(path: Path, y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=list(range(len(class_names))),
        output_dict=True,
        zero_division=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        for class_name in list(class_names) + ["accuracy", "macro avg", "weighted avg"]:
            values = report[class_name]
            if class_name == "accuracy":
                writer.writerow({"class_name": class_name, "precision": "", "recall": "", "f1-score": values, "support": len(y_true)})
            else:
                writer.writerow({"class_name": class_name, **values})


def save_confusion_matrix(path: Path, y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("tiny_classifier_96 Confusion Matrix")
    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.image_size != 96:
        raise ValueError("tiny_classifier_96 is intended for --image_size 96.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    device = resolve_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint)
    data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_cls"
    results_dir = PROJECT_ROOT / "results" / "classifier"

    print("=" * 80)
    print("Evaluate Tiny PCB Classifier 96")
    print("=" * 80)
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {device}")

    _, _, test_loader, class_names = get_classification_dataloaders(
        data_dir=str(data_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_tiny_pcb_classifier_96(num_classes=len(class_names)).to(device)
    print(f"[INFO] Trainable parameters: {count_parameters(model)}")
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    checkpoint_classes = checkpoint.get("class_names")
    if checkpoint_classes is not None and list(checkpoint_classes) != list(class_names):
        print(f"[WARN] Checkpoint class order differs: {checkpoint_classes}")

    y_true, y_pred = evaluate(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    report_csv = results_dir / "tiny_classifier_96_classification_report.csv"
    confusion_png = results_dir / "tiny_classifier_96_confusion_matrix.png"
    save_report_csv(report_csv, y_true, y_pred, class_names)
    save_confusion_matrix(confusion_png, y_true, y_pred, class_names)

    print("\n[RESULT] Classification report")
    print(report_text)
    print(f"[RESULT] Accuracy: {acc:.4f}")
    print(f"[INFO] report csv: {report_csv}")
    print(f"[INFO] confusion matrix: {confusion_png}")


if __name__ == "__main__":
    main()

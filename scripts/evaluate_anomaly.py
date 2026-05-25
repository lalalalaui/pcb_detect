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
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pcb_dataset import get_anomaly_dataloaders
from models.autoencoder import ConvAutoEncoder
from models.tiny_autoencoder import TinyAutoEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PCB AutoEncoder anomaly detector.")
    parser.add_argument(
        "--model",
        choices=["autoencoder", "tiny_ae"],
        default="autoencoder",
        help="AutoEncoder architecture.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--image_size", type=int, default=128, help="Input image size.")
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


def collect_scores(model: torch.nn.Module, test_loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    labels_all: List[int] = []
    scores_all: List[float] = []
    paths_all: List[str] = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device, non_blocking=True)
            recon = model(images)
            scores = torch.mean((recon - images) ** 2, dim=(1, 2, 3))

            labels_all.extend(labels.tolist())
            scores_all.extend(scores.detach().cpu().tolist())
            paths_all.extend(paths)

    return np.asarray(labels_all, dtype=np.int64), np.asarray(scores_all, dtype=np.float64), paths_all


def compute_best_f1(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return {
            "best_f1": 0.0,
            "best_threshold": float(scores.max()) if scores.size else 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    precision_for_thresholds = precision[:-1]
    recall_for_thresholds = recall[:-1]
    f1 = 2 * precision_for_thresholds * recall_for_thresholds / (
        precision_for_thresholds + recall_for_thresholds + 1e-12
    )
    best_idx = int(np.argmax(f1))
    return {
        "best_f1": float(f1[best_idx]),
        "best_threshold": float(thresholds[best_idx]),
        "precision": float(precision_for_thresholds[best_idx]),
        "recall": float(recall_for_thresholds[best_idx]),
    }


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(np.unique(labels)) < 2:
        raise RuntimeError("AUROC and Average Precision require both normal and anomaly labels in the test set.")

    metrics = {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }
    metrics.update(compute_best_f1(labels, scores))
    return metrics


def save_metrics_csv(output_path: Path, metrics: Dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in metrics.items():
            writer.writerow({"metric": metric, "value": value})


def save_scores_csv(output_path: Path, labels: np.ndarray, scores: np.ndarray, paths: Sequence[str], threshold: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "score", "pred"])
        writer.writeheader()
        for path, label, score in zip(paths, labels.tolist(), scores.tolist()):
            writer.writerow(
                {
                    "path": path,
                    "label": label,
                    "score": score,
                    "pred": int(score >= threshold),
                }
            )


def save_roc_curve(output_path: Path, labels: np.ndarray, scores: np.ndarray, auroc: float) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_score_histogram(output_path: Path, labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.hist(normal_scores, bins=50, alpha=0.65, label="normal", color="#2f6fbb")
    plt.hist(anomaly_scores, bins=50, alpha=0.65, label="anomaly", color="#c7523a")
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.6f}")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Count")
    plt.title("Anomaly Score Histogram")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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
        else PROJECT_ROOT / "checkpoints" / "anomaly" / f"{args.model}_best.pth"
    )
    data_dir = PROJECT_ROOT / "data" / "processed" / "pcb_anomaly"
    results_dir = PROJECT_ROOT / "results" / "anomaly"

    print("=" * 80)
    print("Evaluate PCB AutoEncoder Anomaly Detector")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Data directory: {data_dir}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Image size: {args.image_size}")
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
        batch_size=args.batch_size,
        num_workers=0,
    )
    print(f"[INFO] Test samples: {len(test_loader.dataset)}")
    print(f"[INFO] Test batches: {len(test_loader)}")

    model = build_model(args.model).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    checkpoint_image_size = checkpoint.get("image_size")
    if checkpoint_image_size is not None and checkpoint_image_size != args.image_size:
        print(f"[WARN] Checkpoint image_size={checkpoint_image_size}, evaluating with image_size={args.image_size}.")

    labels, scores, paths = collect_scores(model, test_loader, device)
    metrics = compute_metrics(labels, scores)

    metrics_csv = results_dir / f"{args.model}_metrics.csv"
    scores_csv = results_dir / f"{args.model}_scores.csv"
    roc_png = results_dir / f"{args.model}_roc_curve.png"
    hist_png = results_dir / f"{args.model}_score_histogram.png"

    save_metrics_csv(metrics_csv, metrics)
    save_scores_csv(scores_csv, labels, scores, paths, metrics["best_threshold"])
    save_roc_curve(roc_png, labels, scores, metrics["auroc"])
    save_score_histogram(hist_png, labels, scores, metrics["best_threshold"])

    print("\n[RESULT] Image-level anomaly detection metrics")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.6f}")

    print("\n[INFO] Output files")
    print(f"  metrics csv: {metrics_csv}")
    print(f"  scores csv: {scores_csv}")
    print(f"  roc curve: {roc_png}")
    print(f"  score histogram: {hist_png}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] Anomaly evaluation failed: {exc}")
        raise

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "deployment" / "onnx" / "anomaly_tiny_ae_96.onnx"
VAL_NORMAL_DIR = PROJECT_ROOT / "data" / "processed" / "pcb_anomaly" / "val" / "normal"
VAL_ANOMALY_DIR = PROJECT_ROOT / "data" / "processed" / "pcb_anomaly" / "val" / "anomaly"
SCORES_CSV_PATH = SCRIPT_DIR / "tiny_ae_val_scores.csv"
REPORT_CSV_PATH = SCRIPT_DIR / "tiny_ae_threshold_report.csv"
THRESHOLDS_H_PATH = SCRIPT_DIR / "thresholds.h"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT_H = 96
INPUT_W = 96


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing validation directory: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def preprocess_image(image_path: Path) -> np.ndarray:
    # TinyAE training used Resize + ToTensor, without ImageNet normalization.
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((INPUT_W, INPUT_H), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return chw.astype(np.float32)[None, ...]


def score_image(session, input_name: str, image_path: Path) -> float:
    input_tensor = preprocess_image(image_path)
    output_tensor = np.asarray(session.run(None, {input_name: input_tensor})[0], dtype=np.float32)
    return float(np.mean((input_tensor - output_tensor) ** 2))


def write_scores_csv(rows: Sequence[Dict[str, object]]) -> None:
    SCORES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCORES_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "score"])
        writer.writeheader()
        writer.writerows(rows)


def precision_recall_f1(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    preds = scores >= threshold
    positives = labels == 1
    tp = int(np.sum(preds & positives))
    fp = int(np.sum(preds & ~positives))
    fn = int(np.sum(~preds & positives))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def false_positive_rate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    preds = scores >= threshold
    negatives = labels == 0
    fp = int(np.sum(preds & negatives))
    tn = int(np.sum(~preds & negatives))
    return fp / (fp + tn) if (fp + tn) else 0.0


def search_best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    candidates = sorted(set(float(score) for score in scores))
    candidates.insert(0, float(np.nextafter(np.min(scores), -np.inf)))
    candidates.append(float(np.nextafter(np.max(scores), np.inf)))

    best_threshold = candidates[0]
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0

    for threshold in candidates:
        precision, recall, f1 = precision_recall_f1(labels, scores, threshold)
        if f1 > best_f1:
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    return best_threshold, best_f1, best_precision, best_recall


def fallback_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    wins = 0.0
    for pos_score in positives:
        wins += float(np.sum(pos_score > negatives))
        wins += 0.5 * float(np.sum(pos_score == negatives))
    return wins / float(len(positives) * len(negatives))


def fallback_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(np.sum(labels == 1))
    if positive_count == 0:
        return float("nan")

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = 0
    precision_sum = 0.0
    for index, label in enumerate(sorted_labels, start=1):
        if label == 1:
            tp += 1
            precision_sum += tp / index
    return precision_sum / positive_count


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = float(roc_auc_score(labels, scores))
        average_precision = float(average_precision_score(labels, scores))
    except Exception:
        auroc = fallback_auroc(labels, scores)
        average_precision = fallback_average_precision(labels, scores)
    return auroc, average_precision


def format_float(value: float) -> str:
    text = f"{value:.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def write_thresholds_h(
    best_f1_threshold: float,
    balanced_threshold: float,
    normal_p95_score: float,
    normal_p99_score: float,
) -> None:
    lines = [
        "#ifndef PCB_TINY_AE_THRESHOLDS_H",
        "#define PCB_TINY_AE_THRESHOLDS_H",
        "",
        "/*",
        " * TinyAE anomaly threshold calibrated from the full validation set:",
        " *   data/processed/pcb_anomaly/val/normal",
        " *   data/processed/pcb_anomaly/val/anomaly",
        " *",
        " * This threshold is not derived from the 6-sample smoke test vectors.",
        " * BEST_F1 is a high-recall threshold and may cause many false positives.",
        " * NORMAL_P99 is the low-false-positive strong alarm threshold.",
        " * BALANCED can be used for NORMAL/SUSPECT/ANOMALY three-level display.",
        " * TINY_AE_THRESHOLD defaults to NORMAL_P99 for STM32 strong alarm use.",
        " * If the model is retrained, quantized, or input preprocessing changes,",
        " * recalibrate this file before deploying.",
        " * STM32 firmware must use the same RGB 96x96 float32 [0,1] NCHW",
        " * preprocessing and mean((input - output)^2) MSE calculation.",
        " */",
        f"#define TINY_AE_THRESHOLD_BEST_F1      {format_float(best_f1_threshold)}",
        f"#define TINY_AE_THRESHOLD_BALANCED     {format_float(balanced_threshold)}",
        f"#define TINY_AE_THRESHOLD_NORMAL_P95   {format_float(normal_p95_score)}",
        f"#define TINY_AE_THRESHOLD_NORMAL_P99   {format_float(normal_p99_score)}",
        "",
        "#define TINY_AE_THRESHOLD_LOW          TINY_AE_THRESHOLD_BALANCED",
        "#define TINY_AE_THRESHOLD_HIGH         TINY_AE_THRESHOLD_NORMAL_P99",
        "#define TINY_AE_THRESHOLD              TINY_AE_THRESHOLD_HIGH",
        "",
        "#endif /* PCB_TINY_AE_THRESHOLDS_H */",
        "",
    ]
    THRESHOLDS_H_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_report_csv(metrics: Dict[str, float]) -> None:
    REPORT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing ONNX model: {MODEL_PATH}")

    import onnxruntime as ort

    normal_images = list_images(VAL_NORMAL_DIR)
    anomaly_images = list_images(VAL_ANOMALY_DIR)
    if not normal_images:
        raise RuntimeError(f"No normal validation images found in {VAL_NORMAL_DIR}")
    if not anomaly_images:
        raise RuntimeError(f"No anomaly validation images found in {VAL_ANOMALY_DIR}")

    print("=" * 80)
    print("Calibrate STM32H7 TinyAE Threshold From Full Validation Set")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] ONNX model: {MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"[INFO] normal count: {len(normal_images)}")
    print(f"[INFO] anomaly count: {len(anomaly_images)}")
    print("[INFO] Provider: CPUExecutionProvider")
    print("[INFO] Preprocess: RGB resize 96x96, ToTensor float32 NCHW, no normalization")

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rows: List[Dict[str, object]] = []
    for label, images in ((0, normal_images), (1, anomaly_images)):
        for image_path in images:
            rows.append(
                {
                    "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                    "label": label,
                    "score": score_image(session, input_name, image_path),
                }
            )

    write_scores_csv(rows)

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    best_f1_threshold, best_f1, precision, recall = search_best_f1_threshold(labels, scores)
    auroc, average_precision = classification_metrics(labels, scores)
    normal_mean_score = float(np.mean(normal_scores))
    anomaly_mean_score = float(np.mean(anomaly_scores))
    normal_p95_score = float(np.percentile(normal_scores, 95))
    normal_p99_score = float(np.percentile(normal_scores, 99))
    balanced_threshold = (normal_mean_score + anomaly_mean_score) / 2.0
    balanced_precision, balanced_recall, balanced_f1 = precision_recall_f1(
        labels,
        scores,
        balanced_threshold,
    )

    metrics: Dict[str, float] = {
        "normal_count": float(len(normal_images)),
        "anomaly_count": float(len(anomaly_images)),
        "best_f1_threshold": best_f1_threshold,
        "balanced_threshold": balanced_threshold,
        "best_f1": best_f1,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "average_precision": average_precision,
        "normal_mean_score": normal_mean_score,
        "anomaly_mean_score": anomaly_mean_score,
        "normal_p95_score": normal_p95_score,
        "normal_p99_score": normal_p99_score,
        "best_f1_false_positive_rate": false_positive_rate(labels, scores, best_f1_threshold),
        "normal_p95_false_positive_rate": false_positive_rate(labels, scores, normal_p95_score),
        "normal_p99_false_positive_rate": false_positive_rate(labels, scores, normal_p99_score),
        "balanced_precision": balanced_precision,
        "balanced_recall": balanced_recall,
        "balanced_f1": balanced_f1,
    }
    write_report_csv(metrics)
    write_thresholds_h(
        best_f1_threshold,
        balanced_threshold,
        normal_p95_score,
        normal_p99_score,
    )

    print("\n[Metrics]")
    for key, value in metrics.items():
        if key.endswith("_count"):
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.10f}")

    print("\n[Recommendation]")
    print(f"High-recall threshold: best_f1_threshold = {best_f1_threshold:.10f}")
    print(f"Balanced threshold: balanced_threshold = {balanced_threshold:.10f}")
    print(f"Strong alarm threshold: normal_p99_score = {normal_p99_score:.10f}")
    print("TINY_AE_THRESHOLD defaults to TINY_AE_THRESHOLD_HIGH, which maps to NORMAL_P99.")
    print(f"\nScores CSV: {SCORES_CSV_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Report CSV: {REPORT_CSV_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Threshold header: {THRESHOLDS_H_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

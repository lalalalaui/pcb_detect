import csv
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_PATH = PROJECT_ROOT / "deployment" / "onnx" / "classifier_mobilenet_v2_224.onnx"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "pcb_cls" / "test"
REPORT_CSV_PATH = SCRIPT_DIR / "classifier_onnx_test_report.csv"
CLASS_NAMES_PATH = SCRIPT_DIR / "classifier_class_names.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGE_SIZE = 224
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)


def require_module(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"Missing dependency: {module_name}")


def shape_from_value_info(value_info: Any) -> List[Any]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []

    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return shape


def dtype_from_value_info(value_info: Any, onnx_module: Any) -> str:
    elem_type = value_info.type.tensor_type.elem_type
    if elem_type == 0:
        return "unknown"
    return onnx_module.TensorProto.DataType.Name(elem_type)


def format_shape(shape: Iterable[Any]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def list_images(directory: Path) -> List[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_class_names() -> List[str]:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Missing classification test directory: {TEST_DIR}")
    class_names = sorted(path.name for path in TEST_DIR.iterdir() if path.is_dir())
    if not class_names:
        raise RuntimeError(f"No class directories found in {TEST_DIR}")
    return class_names


def preprocess_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    normalized = (chw - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.astype(np.float32)[None, ...]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return (exp / np.sum(exp)).astype(np.float64)


def select_test_images(class_names: Sequence[str], per_class: int = 2) -> List[Dict[str, object]]:
    selected = []
    for class_index, class_name in enumerate(class_names):
        class_dir = TEST_DIR / class_name
        images = list_images(class_dir)
        if len(images) < per_class:
            raise RuntimeError(f"Need {per_class} images for class '{class_name}', found {len(images)}.")
        for image_path in images[:per_class]:
            selected.append(
                {
                    "image_path": image_path,
                    "true_class": class_name,
                    "true_label": class_index,
                }
            )
    return selected


def inspect_transform_code() -> List[str]:
    notes = [
        "datasets/pcb_dataset.py classification eval transform:",
        f"- Resize(({IMAGE_SIZE}, {IMAGE_SIZE}))",
        "- No CenterCrop",
        "- PIL Image.open(...).convert('RGB'), so RGB input",
        "- ToTensor converts HWC RGB [0,255] to CHW float32 [0,1]",
        f"- Normalize mean={tuple(float(x) for x in IMAGENET_MEAN.reshape(-1))}",
        f"- Normalize std={tuple(float(x) for x in IMAGENET_STD.reshape(-1))}",
        "- ONNX input layout is NCHW [1, 3, 224, 224]",
        "scripts/train_classifier.py uses image_size=224 by default",
        "scripts/evaluate_classifier.py uses get_classification_dataloaders eval transform",
    ]
    return notes


def write_class_names(class_names: Sequence[str]) -> None:
    CLASS_NAMES_PATH.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def write_report(rows: Sequence[Dict[str, object]], class_count: int) -> None:
    fieldnames = [
        "image_path",
        "true_class",
        "true_label",
        "predicted_class",
        "predicted_index",
        "top1_probability",
    ]
    fieldnames.extend(f"logit_{idx}" for idx in range(class_count))
    fieldnames.extend(f"prob_{idx}" for idx in range(class_count))

    REPORT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 80)
    print("Check MobileNetV2 Classifier ONNX For STM32H743IIT6")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] ONNX path: {ONNX_PATH.relative_to(PROJECT_ROOT).as_posix()}")

    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"Missing ONNX model: {ONNX_PATH}")

    require_module("onnx")
    require_module("onnxruntime")

    import onnx
    import onnxruntime as ort

    file_size_kb = ONNX_PATH.stat().st_size / 1024.0
    print(f"[INFO] file size KB: {file_size_kb:.2f}")

    model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model)
    print("[OK] onnx.checker.check_model passed")

    graph_input = model.graph.input[0]
    graph_output = model.graph.output[0]
    print(f"[INFO] input name: {graph_input.name}")
    print(f"[INFO] input shape: {format_shape(shape_from_value_info(graph_input))}")
    print(f"[INFO] input dtype: {dtype_from_value_info(graph_input, onnx)}")
    print(f"[INFO] output name: {graph_output.name}")
    print(f"[INFO] output shape: {format_shape(shape_from_value_info(graph_output))}")
    print(f"[INFO] output dtype: {dtype_from_value_info(graph_output, onnx)}")

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    print("[OK] onnxruntime.InferenceSession loaded with CPUExecutionProvider")
    print(f"[INFO] ORT input name: {input_meta.name}")
    print(f"[INFO] ORT input shape: {format_shape(input_meta.shape)}")
    print(f"[INFO] ORT input dtype: {input_meta.type}")
    print(f"[INFO] ORT output name: {output_meta.name}")
    print(f"[INFO] ORT output shape: {format_shape(output_meta.shape)}")
    print(f"[INFO] ORT output dtype: {output_meta.type}")

    random_input = np.random.random((1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    random_output = session.run(None, {input_meta.name: random_input})[0]
    print(
        "[RANDOM] output shape={}, min={:.8f}, max={:.8f}, mean={:.8f}".format(
            list(random_output.shape),
            float(np.min(random_output)),
            float(np.max(random_output)),
            float(np.mean(random_output)),
        )
    )

    class_names = discover_class_names()
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    print("\n[Class order]")
    print(f"class_to_idx: {class_to_idx}")
    write_class_names(class_names)
    print(f"class_names saved to: {CLASS_NAMES_PATH.relative_to(PROJECT_ROOT).as_posix()}")

    print("\n[Preprocess check]")
    for note in inspect_transform_code():
        print(note)

    selected = select_test_images(class_names, per_class=2)
    rows = []
    print("\n[Sample inference]")
    for sample in selected:
        image_path = sample["image_path"]
        true_class = str(sample["true_class"])
        true_label = int(sample["true_label"])
        input_tensor = preprocess_image(image_path)
        logits = np.asarray(session.run(None, {input_meta.name: input_tensor})[0][0], dtype=np.float64)
        probs = softmax(logits)
        predicted_index = int(np.argmax(probs))
        predicted_class = class_names[predicted_index]
        top1_probability = float(probs[predicted_index])

        row: Dict[str, object] = {
            "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
            "true_class": true_class,
            "true_label": true_label,
            "predicted_class": predicted_class,
            "predicted_index": predicted_index,
            "top1_probability": top1_probability,
        }
        for idx, value in enumerate(logits.tolist()):
            row[f"logit_{idx}"] = value
        for idx, value in enumerate(probs.tolist()):
            row[f"prob_{idx}"] = value
        rows.append(row)

        print(
            f"{row['image_path']} | true={true_class}({true_label}) | "
            f"pred={predicted_class}({predicted_index}) | top1={top1_probability:.6f}"
        )
        print(f"  logits: {[round(float(v), 6) for v in logits.tolist()]}")
        print(f"  probs: {[round(float(v), 6) for v in probs.tolist()]}")

    write_report(rows, class_count=len(class_names))
    print(f"\nReport saved to: {REPORT_CSV_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

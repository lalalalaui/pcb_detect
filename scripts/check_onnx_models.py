import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (
    PROJECT_ROOT / "deployment" / "onnx" / "anomaly_tiny_ae_96.onnx",
    PROJECT_ROOT / "deployment" / "onnx" / "classifier_mobilenet_v2_224.onnx",
)


def require_module(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            f"Missing dependency: {module_name}. "
            f"Install it before running this script."
        )


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


def printable_shape(shape: Iterable[Any]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def concrete_shape(shape: Iterable[Any]) -> List[int]:
    concrete = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            concrete.append(dim)
        else:
            concrete.append(1)
    return concrete


def print_graph_io(model: Any, onnx_module: Any) -> None:
    print("\n[ONNX graph inputs]")
    for value_info in model.graph.input:
        print(f"  input name: {value_info.name}")
        print(f"  input shape: {printable_shape(shape_from_value_info(value_info))}")
        print(f"  input dtype: {dtype_from_value_info(value_info, onnx_module)}")

    print("\n[ONNX graph outputs]")
    for value_info in model.graph.output:
        print(f"  output name: {value_info.name}")
        print(f"  output shape: {printable_shape(shape_from_value_info(value_info))}")
        print(f"  output dtype: {dtype_from_value_info(value_info, onnx_module)}")


def print_session_io(session: Any) -> None:
    print("\n[ONNX Runtime inputs]")
    for input_meta in session.get_inputs():
        print(f"  input name: {input_meta.name}")
        print(f"  input shape: {printable_shape(input_meta.shape)}")
        print(f"  input dtype: {input_meta.type}")

    print("\n[ONNX Runtime outputs]")
    for output_meta in session.get_outputs():
        print(f"  output name: {output_meta.name}")
        print(f"  output shape: {printable_shape(output_meta.shape)}")
        print(f"  output dtype: {output_meta.type}")


def build_random_inputs(session: Any) -> Dict[str, np.ndarray]:
    inputs = {}
    for input_meta in session.get_inputs():
        shape = concrete_shape(input_meta.shape)
        inputs[input_meta.name] = np.random.random(shape).astype(np.float32)
    return inputs


def print_inference_outputs(outputs: Iterable[np.ndarray]) -> None:
    print("\n[Inference outputs]")
    for index, output in enumerate(outputs):
        output_array = np.asarray(output)
        print(f"  output[{index}] shape: {list(output_array.shape)}")
        print(f"  output[{index}] min: {float(output_array.min()):.8f}")
        print(f"  output[{index}] max: {float(output_array.max()):.8f}")
        print(f"  output[{index}] mean: {float(output_array.mean()):.8f}")


def check_model(model_path: Path, onnx_module: Any, ort_module: Any) -> None:
    print("\n" + "=" * 80)
    print(f"Model: {model_path.name}")
    print("=" * 80)
    print(f"[PATH] {model_path.relative_to(PROJECT_ROOT).as_posix()}")

    if not model_path.exists():
        print("[ERROR] ONNX file does not exist.")
        return

    print(f"[OK] ONNX file exists. Size: {model_path.stat().st_size} bytes")

    try:
        model = onnx_module.load(str(model_path))
        print_graph_io(model, onnx_module)
    except Exception as exc:
        print(f"[ERROR] Failed to load ONNX model for graph inspection: {exc}")
        return

    try:
        onnx_module.checker.check_model(model)
        print("\n[OK] onnx.checker.check_model passed.")
    except Exception as exc:
        print(f"\n[ERROR] onnx.checker.check_model failed: {exc}")

    try:
        session = ort_module.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        print("\n[OK] onnxruntime.InferenceSession loaded with CPUExecutionProvider.")
        print_session_io(session)
    except Exception as exc:
        print(f"\n[ERROR] Failed to create onnxruntime.InferenceSession: {exc}")
        return

    try:
        random_inputs = build_random_inputs(session)
        for name, value in random_inputs.items():
            print(
                f"\n[INPUT] {name}: random float32 shape={list(value.shape)}, "
                f"min={float(value.min()):.8f}, max={float(value.max()):.8f}"
            )
        outputs = session.run(None, random_inputs)
        print("[OK] Random float32 inference completed.")
        print_inference_outputs(outputs)
    except Exception as exc:
        print(f"\n[ERROR] Random float32 inference failed: {exc}")


def main() -> None:
    print("=" * 80)
    print("Check ONNX Deployment Models")
    print("=" * 80)
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print("[INFO] Provider: CPUExecutionProvider")

    try:
        require_module("onnx")
        require_module("onnxruntime")
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        return

    import onnx
    import onnxruntime as ort

    for model_path in MODEL_PATHS:
        check_model(model_path, onnx, ort)

    print("\nDone.")


if __name__ == "__main__":
    main()

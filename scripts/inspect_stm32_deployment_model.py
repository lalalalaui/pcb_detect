import importlib.util
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "deployment" / "onnx" / "anomaly_tiny_ae_96.onnx"
REPORT_PATH = PROJECT_ROOT / "results" / "tables" / "stm32_tiny_ae_onnx_inspection.txt"
EXPECTED_SHAPE = [1, 3, 96, 96]
FLOAT32_BYTES = 4


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


def concrete_shape(shape: Iterable[Any]) -> List[int]:
    concrete = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            concrete.append(dim)
        else:
            concrete.append(1)
    return concrete


def num_elements(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def format_shape(shape: Iterable[Any]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def add(lines: List[str], text: str = "") -> None:
    lines.append(text)


def inspect_model() -> List[str]:
    lines: List[str] = []

    add(lines, "=" * 80)
    add(lines, "STM32H743IIT6 Tiny Autoencoder ONNX Deployment Inspection")
    add(lines, "=" * 80)
    add(lines, f"ONNX file path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        add(lines, "[ERROR] ONNX file does not exist.")
        return lines

    file_size_kb = MODEL_PATH.stat().st_size / 1024.0
    add(lines, f"File size KB: {file_size_kb:.2f}")

    try:
        require_module("onnx")
        require_module("onnxruntime")
    except RuntimeError as exc:
        add(lines, f"[ERROR] {exc}")
        return lines

    import onnx
    import onnxruntime as ort

    try:
        model = onnx.load(str(MODEL_PATH))
        onnx.checker.check_model(model)
        add(lines, "onnx.checker: passed")
    except Exception as exc:
        add(lines, f"[ERROR] Failed to load/check ONNX model: {exc}")
        return lines

    if not model.graph.input:
        add(lines, "[ERROR] Model has no graph inputs.")
        return lines
    if not model.graph.output:
        add(lines, "[ERROR] Model has no graph outputs.")
        return lines

    input_info = model.graph.input[0]
    output_info = model.graph.output[0]
    input_name = input_info.name
    output_name = output_info.name
    input_shape = shape_from_value_info(input_info)
    output_shape = shape_from_value_info(output_info)
    input_dtype = dtype_from_value_info(input_info, onnx)
    output_dtype = dtype_from_value_info(output_info, onnx)

    input_buffer_bytes = num_elements(concrete_shape(input_shape)) * FLOAT32_BYTES
    output_buffer_bytes = num_elements(concrete_shape(output_shape)) * FLOAT32_BYTES
    expected_buffer_bytes = num_elements(EXPECTED_SHAPE) * FLOAT32_BYTES

    add(lines)
    add(lines, "[Tensor metadata]")
    add(lines, f"Input tensor name: {input_name}")
    add(lines, f"Input shape: {format_shape(input_shape)}")
    add(lines, f"Input dtype: {input_dtype}")
    add(lines, f"Output tensor name: {output_name}")
    add(lines, f"Output shape: {format_shape(output_shape)}")
    add(lines, f"Output dtype: {output_dtype}")

    add(lines)
    add(lines, "[Float32 buffer sizes]")
    add(lines, f"Single input float32 buffer bytes: {input_buffer_bytes}")
    add(lines, f"Single input float32 buffer KB: {input_buffer_bytes / 1024.0:.2f}")
    add(lines, f"Single output float32 buffer bytes: {output_buffer_bytes}")
    add(lines, f"Single output float32 buffer KB: {output_buffer_bytes / 1024.0:.2f}")

    add(lines)
    add(lines, "[Shape checks]")
    add(lines, f"Input is [1, 3, 96, 96]: {input_shape == EXPECTED_SHAPE}")
    add(lines, f"Output shape matches input shape: {output_shape == input_shape}")

    add(lines)
    add(lines, "[STM32H743IIT6 MCU buffer estimate]")
    add(lines, f"input_bytes = 1*3*96*96*4 = {expected_buffer_bytes}")
    add(lines, f"output_bytes = 1*3*96*96*4 = {expected_buffer_bytes}")
    add(lines, f"input KB = {expected_buffer_bytes / 1024.0:.2f}")
    add(lines, f"output KB = {expected_buffer_bytes / 1024.0:.2f}")

    try:
        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        session_input = session.get_inputs()[0]
        random_input = np.random.random(concrete_shape(session_input.shape)).astype(np.float32)
        outputs = session.run(None, {session_input.name: random_input})
        output = np.asarray(outputs[0], dtype=np.float32)
        mse = float(np.mean((output - random_input) ** 2))

        add(lines)
        add(lines, "[Random float32 inference]")
        add(lines, "Inference provider: CPUExecutionProvider")
        add(lines, f"Random input shape: {list(random_input.shape)}")
        add(lines, f"Output shape: {list(output.shape)}")
        add(lines, f"Output min: {float(output.min()):.8f}")
        add(lines, f"Output max: {float(output.max()):.8f}")
        add(lines, f"Output mean: {float(output.mean()):.8f}")
        add(lines, f"MSE reconstruction error: {mse:.8f}")
    except Exception as exc:
        add(lines)
        add(lines, f"[ERROR] Random float32 inference failed: {exc}")

    add(lines)
    add(lines, "[Deployment notes]")
    add(lines, "Current ONNX model is FP32.")
    add(lines, "If deployed directly, STM32 requires float32 input and output buffers.")
    add(lines, "Next, use STM32Cube.AI to analyze RAM, Flash, MACC, and operator support.")

    return lines


def main() -> None:
    lines = inspect_model()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved to: {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

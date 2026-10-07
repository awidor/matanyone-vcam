import argparse
from pathlib import Path

import tensorrt as trt


def build_engine(onnx_path: Path, engine_path: Path, workspace_gb: float, half_io: frozenset[str] = frozenset()) -> None:
    """Builds an FP16 engine. Tensors named in `half_io` keep FP16 at the engine boundary,
    which avoids FP32 reformat copies for large tensors passed between engines."""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Building {engine_path} from {onnx_path}")
    if not parser.parse_from_file(str(onnx_path)):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    for tensor in [network.get_input(i) for i in range(network.num_inputs)] + [
        network.get_output(i) for i in range(network.num_outputs)
    ]:
        if tensor.name in half_io:
            tensor.dtype = trt.float16

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"Wrote {engine_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", default="models/submodules")
    parser.add_argument("--engine-dir", default="engines/submodules")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)
    failures = []
    for onnx_path in sorted(onnx_dir.glob("*.onnx")):
        try:
            build_engine(onnx_path, engine_dir / f"{onnx_path.stem}_fp16.engine", args.workspace_gb)
        except Exception as exc:
            failures.append((onnx_path.name, repr(exc)))
            print(f"FAILED {onnx_path.name}: {exc!r}")

    if failures:
        print("Failures:")
        for name, exc in failures:
            print(f"- {name}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

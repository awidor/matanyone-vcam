import argparse
from pathlib import Path

import tensorrt as trt


def parse_shape(spec: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part) for part in spec.lower().replace("x", ",").split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be N,C,H,W")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="models/encoder.onnx")
    parser.add_argument("--engine", default="engines/encoder_fp16.engine")
    parser.add_argument("--min-shape", type=parse_shape, default="1,3,270,480")
    parser.add_argument("--opt-shape", type=parse_shape, default="1,3,720,1280")
    parser.add_argument("--max-shape", type=parse_shape, default="1,3,1080,1920")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    onnx_path = Path(args.onnx)
    if not parser.parse(onnx_path.read_bytes()):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * 1024**3))
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("Warning: platform_has_fast_fp16 is false; building without FP16 flag.")

    profile = builder.create_optimization_profile()
    profile.set_shape("image", args.min_shape, args.opt_shape, args.max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed.")

    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"Wrote {engine_path}")


if __name__ == "__main__":
    main()

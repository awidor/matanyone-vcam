import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


def torch_dtype(dtype):
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    return torch.float32


def bench_engine(path: Path, warmup: int, iters: int) -> tuple[float, float, list[str]]:
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {path}")
    context = engine.create_execution_context()

    tensors = {}
    descriptions = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(context.get_tensor_shape(name))
        dtype = torch_dtype(engine.get_tensor_dtype(name))
        mode = engine.get_tensor_mode(name).name
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"{path.name}:{name} has dynamic shape {shape}; set it before benchmarking")
        tensor = torch.empty(shape, device="cuda", dtype=dtype)
        if dtype.is_floating_point:
            tensor.normal_()
        else:
            tensor.zero_()
        tensors[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())
        descriptions.append(f"{mode} {name}: {shape} {dtype}")

    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(warmup):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        context.execute_async_v3(stream)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    return statistics.fmean(timings), float(np.percentile(timings, 99)), descriptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", default="engines/submodules")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for benchmarking.")

    total_mean = 0.0
    for path in sorted(Path(args.engine_dir).glob("*.engine")):
        mean, p99, descriptions = bench_engine(path, args.warmup, args.iters)
        total_mean += mean
        print(f"{path.name}: mean_ms={mean:.3f} p99_ms={p99:.3f}")
        for desc in descriptions:
            print(f"  {desc}")
    print(f"sum_mean_ms={total_mean:.3f}")


if __name__ == "__main__":
    main()

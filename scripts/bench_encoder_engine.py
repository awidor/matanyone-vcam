import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="engines/encoder_fp16.engine")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for benchmarking.")

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(args.engine).read_bytes())
    context = engine.create_execution_context()
    context.set_input_shape("image", (1, 3, args.height, args.width))

    tensors = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(context.get_tensor_shape(name))
        dtype = torch.float32 if engine.get_tensor_dtype(name) == trt.float32 else torch.float16
        tensors[name] = torch.empty(shape, device="cuda", dtype=dtype)
        context.set_tensor_address(name, tensors[name].data_ptr())

    tensors["image"].normal_()
    stream = torch.cuda.current_stream().cuda_stream

    for _ in range(args.warmup):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()

    timings = []
    for _ in range(args.iters):
        start = time.perf_counter()
        context.execute_async_v3(stream)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    mean = statistics.fmean(timings)
    p99 = float(np.percentile(timings, 99))
    print(f"mean_ms={mean:.3f}")
    print(f"p99_ms={p99:.3f}")
    for name, tensor in tensors.items():
        print(f"{name}: {tuple(tensor.shape)} {tensor.dtype}")


if __name__ == "__main__":
    main()

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


class TrtEngine:
    def __init__(self, path: Path, runtime: trt.Runtime):
        self.path = path
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {path}")
        self.context = self.engine.create_execution_context()
        self.io = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            self.io[name] = {
                "mode": self.engine.get_tensor_mode(name),
                "shape": tuple(self.context.get_tensor_shape(name)),
                "dtype": torch_dtype(self.engine.get_tensor_dtype(name)),
            }

    @property
    def inputs(self):
        return [name for name, meta in self.io.items() if meta["mode"] == trt.TensorIOMode.INPUT]

    @property
    def outputs(self):
        return [name for name, meta in self.io.items() if meta["mode"] == trt.TensorIOMode.OUTPUT]

    def bind(self, tensors: dict[str, torch.Tensor]) -> None:
        for name in self.io:
            self.context.set_tensor_address(name, tensors[name].data_ptr())

    def run(self, stream: int) -> None:
        self.context.execute_async_v3(stream)


class Pipeline:
    def __init__(self, engine_dir: Path):
        logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(logger)
        self.encode = TrtEngine(engine_dir / "encode_image_fp16.engine", self.runtime)
        self.read = TrtEngine(engine_dir / "read_memory_fp16.engine", self.runtime)
        fuse_path = engine_dir / "pixel_fusion_fp16.engine"
        self.fuse = TrtEngine(fuse_path, self.runtime) if fuse_path.exists() else None
        self.segment = TrtEngine(engine_dir / "segment_fp16.engine", self.runtime)
        self.encode_mask = TrtEngine(engine_dir / "encode_mask_fp16.engine", self.runtime)
        shallow_path = engine_dir / "encode_mask_shallow_fp16.engine"
        if not shallow_path.exists():
            fallback = engine_dir.parent / "faithful" / "encode_mask_shallow_fp16.engine"
            if fallback.exists():
                shallow_path = fallback
        self.encode_mask_shallow = TrtEngine(shallow_path, self.runtime) if shallow_path.exists() else None
        self.full_read = "input_10" in self.read.io

        self.tensors = {}
        self.aliases = {}
        self._allocate_tensors()
        self._bind_engines()

    def _new_tensor(self, name: str, shape, dtype=torch.float32):
        if name not in self.tensors:
            tensor = torch.empty(shape, device="cuda", dtype=dtype)
            if dtype.is_floating_point:
                tensor.normal_()
            else:
                tensor.zero_()
            self.tensors[name] = tensor
        return self.tensors[name]

    def _allocate_from_engine(self, engine: TrtEngine, prefix: str):
        for name, meta in engine.io.items():
            self._new_tensor(f"{prefix}:{name}", meta["shape"], meta["dtype"])

    def _allocate_tensors(self):
        # Shared semantic buffers.
        self._new_tensor("image", (1, 3, 720, 1280))
        self._new_tensor("last_mask", (1, 1, 720, 1280))
        self._new_tensor("sensory", (1, 1, 256, 45, 80))
        self._new_tensor("memory_key", (1, 64, 5, 45, 80))
        self._new_tensor("memory_shrinkage", (1, 1, 5, 45, 80))
        self._new_tensor("memory_value", (1, 1, 256, 5, 45, 80))
        self._new_tensor("last_pix_feat", (1, 256, 45, 80))
        self._new_tensor("last_msk_value", (1, 1, 256, 45, 80))

        # Encode outputs.
        for name, meta in self.encode.io.items():
            if meta["mode"] == trt.TensorIOMode.OUTPUT:
                self._new_tensor(name, meta["shape"], meta["dtype"])

        self._new_tensor("pixel_memory", (1, 1, 256, 45, 80))
        self._new_tensor("fused_pixel", (1, 1, 256, 45, 80))
        self._new_tensor("obj_memory", (1, 1, 1, 16, 257))
        self._new_tensor("new_sensory", (1, 1, 256, 45, 80))
        self._new_tensor("alpha", (1, 1, 720, 1280))
        self._new_tensor("mask_value", (1, 1, 256, 45, 80))
        self._new_tensor("object_summaries", (1, 1, 16, 257))

    def _bind_engines(self):
        self.encode.bind(
            {
                "input_0": self.tensors["image"],
                "f16": self.tensors["f16"],
                "f8": self.tensors["f8"],
                "f4": self.tensors["f4"],
                "f2": self.tensors["f2"],
                "f1": self.tensors["f1"],
                "pix_feat": self.tensors["pix_feat"],
                "key": self.tensors["key"],
                "shrinkage": self.tensors["shrinkage"],
                "selection": self.tensors["selection"],
            }
        )
        read_bindings = {
                "input_0": self.tensors["key"],
                "input_1": self.tensors["selection"],
                "input_2": self.tensors["memory_key"],
                "input_3": self.tensors["memory_shrinkage"],
                "input_4": self.tensors["memory_value"],
                "input_5": self.tensors["last_pix_feat"],
                "input_6": self.tensors["pix_feat"],
                "input_7": self.tensors["last_mask"],
                "input_8": self.tensors["last_msk_value"],
                "pixel_memory": self.tensors["pixel_memory"],
            }
        if self.full_read:
            read_bindings["input_9"] = self.tensors["sensory"]
            read_bindings["input_10"] = self.tensors["obj_memory"]
            read_bindings["memory_readout"] = self.tensors["fused_pixel"]
            del read_bindings["pixel_memory"]
        self.read.bind(read_bindings)
        if self.fuse is not None:
            self.fuse.bind(
                {
                    "input_0": self.tensors["pix_feat"],
                    "input_1": self.tensors["pixel_memory"],
                    "input_2": self.tensors["sensory"],
                    "input_3": self.tensors["last_mask"],
                    "fused_pixel": self.tensors["fused_pixel"],
                }
            )
        self.segment.bind(
            {
                "input_0": self.tensors["f16"],
                "input_1": self.tensors["f8"],
                "input_2": self.tensors["f4"],
                "input_3": self.tensors["f2"],
                "input_4": self.tensors["f1"],
                "input_5": self.tensors["fused_pixel"],
                "input_6": self.tensors["sensory"],
                "new_sensory": self.tensors["new_sensory"],
                "alpha": self.tensors["alpha"],
            }
        )
        self.encode_mask.bind(
            {
                "input_0": self.tensors["image"],
                "input_1": self.tensors["pix_feat"],
                "input_2": self.tensors["new_sensory"],
                "input_3": self.tensors["alpha"],
                "mask_value": self.tensors["mask_value"],
                "new_sensory": self.tensors["sensory"],
                "object_summaries": self.tensors["object_summaries"],
            }
        )
        if self.encode_mask_shallow is not None:
            self.encode_mask_shallow.bind(
                {
                    "input_0": self.tensors["image"],
                    "input_1": self.tensors["pix_feat"],
                    "input_3": self.tensors["alpha"],
                    "mask_value": self.tensors["last_msk_value"],
                }
            )

    def run_normal(self, stream: int):
        self.encode.run(stream)
        self.read.run(stream)
        if self.fuse is not None:
            self.fuse.run(stream)
        self.segment.run(stream)
        if self.encode_mask_shallow is not None:
            self.encode_mask_shallow.run(stream)

    def run_memory_update(self, stream: int):
        self.encode.run(stream)
        self.read.run(stream)
        if self.fuse is not None:
            self.fuse.run(stream)
        self.segment.run(stream)
        self.encode_mask.run(stream)

    def run_stage(self, name: str, stream: int):
        getattr(self, name).run(stream)


def time_path(fn, stream, warmup, iters):
    for _ in range(warmup):
        fn(stream)
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(stream)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.fmean(timings), float(np.percentile(timings, 99))


def time_stage(pipeline: Pipeline, stage: str, stream, warmup, iters):
    return time_path(lambda s: pipeline.run_stage(stage, s), stream, warmup, iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", default="engines/optimized")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required.")

    pipeline = Pipeline(Path(args.engine_dir))
    stream = torch.cuda.current_stream().cuda_stream

    if args.cuda_graph:
        for _ in range(args.warmup):
            pipeline.run_normal(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pipeline.run_normal(torch.cuda.current_stream().cuda_stream)

        def replay(_stream):
            graph.replay()

        normal_mean, normal_p99 = time_path(replay, stream, 10, args.iters)
        update_mean, update_p99 = float("nan"), float("nan")
    else:
        normal_mean, normal_p99 = time_path(pipeline.run_normal, stream, args.warmup, args.iters)
        update_mean, update_p99 = time_path(pipeline.run_memory_update, stream, args.warmup, args.iters)

    stage_results = {}
    if not args.cuda_graph:
        for stage in ["encode", "read", "fuse", "segment", "encode_mask", "encode_mask_shallow"]:
            if stage == "fuse" and pipeline.fuse is None:
                continue
            if stage == "encode_mask_shallow" and pipeline.encode_mask_shallow is None:
                continue
            stage_results[stage] = time_stage(pipeline, stage, stream, args.warmup, args.iters)

    print(f"normal_mean_ms={normal_mean:.3f}")
    print(f"normal_p99_ms={normal_p99:.3f}")
    print(f"memory_update_mean_ms={update_mean:.3f}")
    print(f"memory_update_p99_ms={update_p99:.3f}")
    if stage_results:
        for stage, (mean, p99) in stage_results.items():
            print(f"{stage}_mean_ms={mean:.3f}")
            print(f"{stage}_p99_ms={p99:.3f}")
        normal_stages = ["encode", "read", "segment"] if pipeline.fuse is None else ["encode", "read", "fuse", "segment"]
        if pipeline.encode_mask_shallow is not None:
            normal_stages.append("encode_mask_shallow")
        summed_normal = sum(stage_results[name][0] for name in normal_stages)
        update_stages = ["encode", "read", "segment", "encode_mask"] if pipeline.fuse is None else [
            "encode",
            "read",
            "fuse",
            "segment",
            "encode_mask",
        ]
        summed_update = sum(stage_results[name][0] for name in update_stages)
        print(f"summed_normal_stage_mean_ms={summed_normal:.3f}")
        print(f"summed_memory_update_stage_mean_ms={summed_update:.3f}")
        print(f"normal_chain_overhead_ms={normal_mean - summed_normal:.3f}")
        print(f"memory_update_chain_overhead_ms={update_mean - summed_update:.3f}")


if __name__ == "__main__":
    main()

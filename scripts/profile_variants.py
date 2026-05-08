import argparse
from pathlib import Path

import torch

from build_submodule_engines import build_engine
from export_submodules import export_module
from matanyone2_stateless import (
    MemoryReadout,
    MemorySimilarity,
    MemoryTopKSoftmax,
    MemoryValueReadout,
    SegmentAlphaOnly,
    load_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required.")

    device = torch.device("cuda")
    h16 = args.height // 16
    w16 = args.width // 16
    out_dir = Path("models/profile")
    engine_dir = Path("engines/profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)

    key = torch.randn(1, 64, h16, w16, device=device)
    selection = torch.rand(1, 64, h16, w16, device=device)

    for frames in [1, 2, 3, 5]:
        memory_key = torch.randn(1, 64, frames, h16, w16, device=device)
        memory_shrinkage = torch.rand(1, 1, frames, h16, w16, device=device) + 1
        memory_value = torch.randn(1, 1, 256, frames, h16, w16, device=device)
        for top_k in [10, 20, 30, None]:
            suffix = "full" if top_k is None else f"topk{top_k}"
            name = f"read_memory_t{frames}_{suffix}"
            export_module(
                name,
                MemoryReadout(top_k=top_k).eval(),
                (key, selection, memory_key, memory_shrinkage, memory_value),
                ["pixel_memory"],
                out_dir,
            )
            build_engine(out_dir / f"{name}.onnx", engine_dir / f"{name}_fp16.engine", workspace_gb=4.0)

    frames = 5
    memory_key = torch.randn(1, 64, frames, h16, w16, device=device)
    memory_shrinkage = torch.rand(1, 1, frames, h16, w16, device=device) + 1
    memory_value = torch.randn(1, 1, 256, frames, h16, w16, device=device)
    similarity = torch.randn(1, frames * h16 * w16, h16 * w16, device=device)
    affinity = torch.rand(1, frames * h16 * w16, h16 * w16, device=device)

    export_module(
        "memory_similarity_t5",
        MemorySimilarity().eval(),
        (key, selection, memory_key, memory_shrinkage),
        ["similarity"],
        out_dir,
    )
    build_engine(out_dir / "memory_similarity_t5.onnx", engine_dir / "memory_similarity_t5_fp16.engine", 4.0)

    export_module(
        "memory_topk_softmax_t5",
        MemoryTopKSoftmax(top_k=30).eval(),
        (similarity,),
        ["affinity"],
        out_dir,
    )
    build_engine(out_dir / "memory_topk_softmax_t5.onnx", engine_dir / "memory_topk_softmax_t5_fp16.engine", 4.0)

    export_module(
        "memory_value_readout_t5",
        MemoryValueReadout().eval(),
        (affinity, memory_value),
        ["pixel_memory"],
        out_dir,
    )
    build_engine(out_dir / "memory_value_readout_t5.onnx", engine_dir / "memory_value_readout_t5_fp16.engine", 4.0)

    model = load_model(device="cuda")
    f16 = torch.randn(1, 1024, h16, w16, device=device)
    f8 = torch.randn(1, 512, h16 * 2, w16 * 2, device=device)
    f4 = torch.randn(1, 256, h16 * 4, w16 * 4, device=device)
    f2 = torch.randn(1, 64, h16 * 8, w16 * 8, device=device)
    f1 = torch.randn(1, 3, args.height, args.width, device=device)
    pixel_memory = torch.randn(1, 1, 256, h16, w16, device=device)
    sensory = torch.randn(1, 1, 256, h16, w16, device=device)
    export_module(
        "segment_alpha_only",
        SegmentAlphaOnly(model).eval(),
        (f16, f8, f4, f2, f1, pixel_memory, sensory),
        ["new_sensory", "alpha"],
        out_dir,
        dynamo=True,
    )
    build_engine(out_dir / "segment_alpha_only.onnx", engine_dir / "segment_alpha_only_fp16.engine", 4.0)


if __name__ == "__main__":
    main()

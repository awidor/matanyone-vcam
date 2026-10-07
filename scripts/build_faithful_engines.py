"""Export and build only the engines the Rust app loads (engines/faithful/).

Equivalent to running export_submodules.py, build_submodule_engines.py, profile_variants.py,
profile_full_read.py, profile_mask_shallow.py and prepare_faithful_engines.py, without the
profiling variants.
"""

import argparse
from pathlib import Path

import torch

from build_submodule_engines import build_engine
from export_submodules import export_module
from matanyone2_stateless import (
    EncodeImageAndKey,
    EncodeMask,
    EncodeMaskShallow,
    FullMemoryRead,
    SegmentAlphaOnly,
    load_model,
    use_static_aux_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", default="models/faithful")
    parser.add_argument("--engine-dir", default="engines/faithful")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--only", nargs="*", help="build only these engine names")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required.")

    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model = load_model(device="cuda")
    use_static_aux_mask()
    h, w = 720, 1280
    h16, w16 = h // 16, w // 16
    t = 5

    image = torch.randn(1, 3, h, w, device=device)
    f16 = torch.randn(1, 1024, h16, w16, device=device)
    f8 = torch.randn(1, 512, h16 * 2, w16 * 2, device=device)
    f4 = torch.randn(1, 256, h16 * 4, w16 * 4, device=device)
    f2 = torch.randn(1, 64, h16 * 8, w16 * 8, device=device)
    f1 = torch.randn(1, 3, h, w, device=device)
    pix_feat = torch.randn(1, 256, h16, w16, device=device)
    key = torch.randn(1, 64, h16, w16, device=device)
    selection = torch.rand(1, 64, h16, w16, device=device)
    memory_key = torch.randn(1, 64, t, h16, w16, device=device)
    memory_shrinkage = torch.rand(1, 1, t, h16, w16, device=device) + 1
    memory_value = torch.randn(1, 1, 256, t, h16, w16, device=device)
    value = torch.randn(1, 1, 256, h16, w16, device=device)
    sensory = torch.randn(1, 1, 256, h16, w16, device=device)
    mask = torch.rand(1, 1, h, w, device=device)
    obj_memory = torch.randn(1, 1, 1, 16, 257, device=device)

    # The multi-scale features are the largest tensors passed between engines; keep them FP16.
    features = frozenset(["f16", "f8", "f4", "f2", "f1"])
    segment_features = frozenset(["input_0", "input_1", "input_2", "input_3", "input_4"])

    # (engine name, module, inputs, output names, use dynamo exporter, FP16 I/O tensors)
    exports = [
        (
            "encode_image",
            EncodeImageAndKey(model),
            (image,),
            ["f16", "f8", "f4", "f2", "f1", "pix_feat", "key", "shrinkage", "selection"],
            False,
            features,
        ),
        (
            "read_memory",
            FullMemoryRead(model, top_k=None, query_major=True),
            (key, selection, memory_key, memory_shrinkage, memory_value, pix_feat, pix_feat, mask, value, sensory, obj_memory),
            ["memory_readout"],
            False,
            frozenset(),
        ),
        (
            "segment",
            SegmentAlphaOnly(model),
            (f16, f8, f4, f2, f1, value, sensory),
            ["new_sensory", "alpha"],
            True,
            segment_features,
        ),
        (
            "encode_mask",
            EncodeMask(model),
            (image, pix_feat, sensory, mask),
            ["mask_value", "new_sensory", "object_summaries"],
            False,
            frozenset(),
        ),
        (
            "encode_mask_shallow",
            EncodeMaskShallow(model),
            (image, pix_feat, sensory, mask),
            ["mask_value"],
            False,
            frozenset(),
        ),
    ]

    for name, module, inputs, outputs, dynamo, half_io in exports:
        if args.only and name not in args.only:
            continue
        module.eval()
        with torch.inference_mode():
            export_module(name, module, inputs, outputs, onnx_dir, dynamo=dynamo)
        build_engine(onnx_dir / f"{name}.onnx", engine_dir / f"{name}_fp16.engine", args.workspace_gb, half_io)


if __name__ == "__main__":
    main()

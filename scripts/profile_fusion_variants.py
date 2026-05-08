from pathlib import Path

import torch

from build_submodule_engines import build_engine
from export_submodules import export_module
from matanyone2_stateless import ReadMemoryAndPixelFusion, SegmentAlphaOnlyNoF16, load_model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required.")

    device = torch.device("cuda")
    out_dir = Path("models/fusion_profile")
    engine_dir = Path("engines/fusion_profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)

    h16, w16 = 45, 80
    model = load_model(device="cuda")

    key = torch.randn(1, 64, h16, w16, device=device)
    selection = torch.rand(1, 64, h16, w16, device=device)
    memory_key = torch.randn(1, 64, 5, h16, w16, device=device)
    memory_shrinkage = torch.rand(1, 1, 5, h16, w16, device=device) + 1
    memory_value = torch.randn(1, 1, 256, 5, h16, w16, device=device)
    pix_feat = torch.randn(1, 256, h16, w16, device=device)
    sensory = torch.randn(1, 1, 256, h16, w16, device=device)
    last_mask = torch.rand(1, 1, 720, 1280, device=device)

    export_module(
        "read_fuse_t5_full",
        ReadMemoryAndPixelFusion(model, top_k=None).eval(),
        (key, selection, memory_key, memory_shrinkage, memory_value, pix_feat, sensory, last_mask),
        ["fused_pixel"],
        out_dir,
    )
    build_engine(out_dir / "read_fuse_t5_full.onnx", engine_dir / "read_fuse_t5_full_fp16.engine", 4.0)

    f8 = torch.randn(1, 512, 90, 160, device=device)
    f4 = torch.randn(1, 256, 180, 320, device=device)
    f2 = torch.randn(1, 64, 360, 640, device=device)
    f1 = torch.randn(1, 3, 720, 1280, device=device)
    memory_readout = torch.randn(1, 1, 256, h16, w16, device=device)
    export_module(
        "segment_alpha_no_f16",
        SegmentAlphaOnlyNoF16(model).eval(),
        (f8, f4, f2, f1, memory_readout, sensory),
        ["new_sensory", "alpha"],
        out_dir,
        dynamo=True,
    )
    build_engine(out_dir / "segment_alpha_no_f16.onnx", engine_dir / "segment_alpha_no_f16_fp16.engine", 4.0)


if __name__ == "__main__":
    main()

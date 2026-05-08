from pathlib import Path

import torch

from build_submodule_engines import build_engine
from export_submodules import export_module
from matanyone2_stateless import MemoryReadoutWithUncertainty, load_model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required.")

    device = torch.device("cuda")
    out_dir = Path("models/correctness_profile")
    engine_dir = Path("engines/correctness_profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(device="cuda")
    h16, w16 = 45, 80
    key = torch.randn(1, 64, h16, w16, device=device)
    selection = torch.rand(1, 64, h16, w16, device=device)
    memory_key = torch.randn(1, 64, 5, h16, w16, device=device)
    memory_shrinkage = torch.rand(1, 1, 5, h16, w16, device=device) + 1
    memory_value = torch.randn(1, 1, 256, 5, h16, w16, device=device)
    last_pix_feat = torch.randn(1, 256, h16, w16, device=device)
    pix_feat = torch.randn(1, 256, h16, w16, device=device)
    last_pred_mask = torch.rand(1, 1, 720, 1280, device=device)
    last_msk_value = torch.randn(1, 1, 256, h16, w16, device=device)

    export_module(
        "read_memory_uncert_t5_full",
        MemoryReadoutWithUncertainty(model, top_k=None).eval(),
        (
            key,
            selection,
            memory_key,
            memory_shrinkage,
            memory_value,
            last_pix_feat,
            pix_feat,
            last_pred_mask,
            last_msk_value,
        ),
        ["pixel_memory"],
        out_dir,
    )
    build_engine(
        out_dir / "read_memory_uncert_t5_full.onnx",
        engine_dir / "read_memory_uncert_t5_full_fp16.engine",
        4.0,
    )


if __name__ == "__main__":
    main()

from pathlib import Path

import torch

from build_submodule_engines import build_engine
from export_submodules import export_module
from matanyone2_stateless import EncodeMaskShallow, load_model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required.")

    device = torch.device("cuda")
    out_dir = Path("models/mask_profile")
    engine_dir = Path("engines/mask_profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(device="cuda")
    image = torch.randn(1, 3, 720, 1280, device=device)
    pix_feat = torch.randn(1, 256, 45, 80, device=device)
    sensory = torch.randn(1, 1, 256, 45, 80, device=device)
    mask = torch.rand(1, 1, 720, 1280, device=device)

    export_module(
        "encode_mask_shallow",
        EncodeMaskShallow(model).eval(),
        (image, pix_feat, sensory, mask),
        ["mask_value"],
        out_dir,
    )
    build_engine(out_dir / "encode_mask_shallow.onnx", engine_dir / "encode_mask_shallow_fp16.engine", 4.0)


if __name__ == "__main__":
    main()

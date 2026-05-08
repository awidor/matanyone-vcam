import argparse
from pathlib import Path

import torch

from matanyone2_stateless import (
    EncodeImageAndKey,
    EncodeMask,
    FirstFrameReadMemory,
    MemoryReadout,
    PixelFusion,
    Segment,
    load_model,
)


def export_module(name, module, inputs, output_names, out_dir, *, dynamo=False):
    path = out_dir / f"{name}.onnx"
    print(f"Exporting {name} -> {path}")
    torch.onnx.export(
        module,
        inputs,
        path,
        input_names=[f"input_{i}" for i in range(len(inputs))],
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=dynamo,
    )
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="models/submodules")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--memory-frames", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model = load_model(device="cuda")
    h16 = args.height // 16
    w16 = args.width // 16
    t = args.memory_frames

    image = torch.randn(1, 3, args.height, args.width, device=device)
    f16 = torch.randn(1, 1024, h16, w16, device=device)
    f8 = torch.randn(1, 512, h16 * 2, w16 * 2, device=device)
    f4 = torch.randn(1, 256, h16 * 4, w16 * 4, device=device)
    f2 = torch.randn(1, 64, h16 * 8, w16 * 8, device=device)
    f1 = torch.randn(1, 3, args.height, args.width, device=device)
    pix_feat = torch.randn(1, 256, h16, w16, device=device)
    key = torch.randn(1, 64, h16, w16, device=device)
    selection = torch.rand(1, 64, h16, w16, device=device)
    memory_key = torch.randn(1, 64, t, h16, w16, device=device)
    memory_shrinkage = torch.rand(1, 1, t, h16, w16, device=device) + 1
    memory_value = torch.randn(1, 1, 256, t, h16, w16, device=device)
    pixel_memory = torch.randn(1, 1, 256, h16, w16, device=device)
    sensory = torch.randn(1, 1, 256, h16, w16, device=device)
    last_mask = torch.rand(1, 1, args.height, args.width, device=device)
    obj_memory = torch.randn(1, 1, 1, 16, 257, device=device)

    exports = [
        (
            "encode_image",
            EncodeImageAndKey(model),
            (image,),
            ["f16", "f8", "f4", "f2", "f1", "pix_feat", "key", "shrinkage", "selection"],
        ),
        (
            "read_memory",
            MemoryReadout(top_k=30),
            (key, selection, memory_key, memory_shrinkage, memory_value),
            ["pixel_memory"],
        ),
        (
            "pixel_fusion",
            PixelFusion(model),
            (pix_feat, pixel_memory, sensory, last_mask),
            ["fused_pixel"],
        ),
        (
            "first_frame_read_memory",
            FirstFrameReadMemory(model),
            (pix_feat, pixel_memory, sensory, last_mask, obj_memory),
            ["memory_readout"],
        ),
        (
            "segment",
            Segment(model),
            (f16, f8, f4, f2, f1, pixel_memory, sensory),
            ["new_sensory", "logits", "prob"],
        ),
        (
            "encode_mask",
            EncodeMask(model),
            (image, pix_feat, sensory, last_mask),
            ["mask_value", "new_sensory", "object_summaries"],
        ),
    ]

    failures = []
    for name, module, inputs, outputs in exports:
        module.eval()
        try:
            with torch.inference_mode():
                export_module(name, module, inputs, outputs, out_dir, dynamo=(name == "segment"))
        except Exception as exc:
            failures.append((name, repr(exc)))
            print(f"FAILED {name}: {exc!r}")

    if failures:
        print("Failures:")
        for name, exc in failures:
            print(f"- {name}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

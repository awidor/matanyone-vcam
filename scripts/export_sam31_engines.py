#!/usr/bin/env python3
"""Export SAM 3.1 TensorRT engines for in-process Rust inference.

This is a development-time export spike. Runtime uses engines/sam31/*.engine
when present; otherwise the app falls back to sam-python feature (Python worker).

Requirements:
  - facebookresearch/sam3 installed in the active Python env
  - TensorRT 10.x + CUDA toolkit on PATH
  - SAM31_CHECKPOINT env var or models/sam3.1 checkpoint on disk

Usage:
  python scripts/export_sam31_engines.py --out engines/sam31
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="engines/sam31")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("SAM3.1 TRT export spike")
    print(f"  output: {out_dir}")
    print(f"  frame:  {args.width}x{args.height}")
    print()
    print("BLOCKER: Full SAM3.1 TRT export is not automated in this repo yet.")
    print("The sam3 model uses dynamic control flow (session state + prompt decoder)")
    print("that requires a custom ONNX export graph before TensorRT build.")
    print()
    print("Next steps:")
    print("  1. Trace sam3 image encoder for fixed 1280x720 input -> ONNX")
    print("  2. Trace prompt decoder (fg/bg points) -> ONNX or split engines")
    print("  3. trtexec --onnx=... --saveEngine=engines/sam31/sam31_decoder.engine")
    print("  4. Wire Sam31Predictor::predict_trt in crates/matanyone-core/src/sam.rs")
    print()
    print("Until export succeeds, build with sam-python (default) for click-to-mask.")
    return 2


if __name__ == "__main__":
    sys.exit(main())

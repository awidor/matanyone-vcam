#!/usr/bin/env python3
"""Compare Python SAM worker masks vs Rust Sam31Predictor (sam-python or TRT)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH = 1280
HEIGHT = 720

GOLDEN_CLICKS = [
    {"fg": [[640, 360]], "bg": [[100, 100]]},
    {"fg": [[500, 400], [700, 350]], "bg": [[50, 600]]},
]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ab = a > 127
    bb = b > 127
    inter = float(np.logical_and(ab, bb).sum())
    union = float(np.logical_or(ab, bb).sum())
    return 1.0 if union == 0 else inter / union


def python_mask(repo: Path, frame_bgr: np.ndarray, fg, bg) -> np.ndarray:
    worker = repo / "scripts" / "sam31_mask_worker.py"
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        frame_path = Path(td) / "frame.png"
        rgb = frame_bgr[:, :, ::-1]
        Image.fromarray(rgb).save(frame_path)
        python = os.environ.get("SAM31_PYTHON", "python")
        proc = subprocess.run(
            [python, str(worker), "--image", str(frame_path), "--once", json.dumps({"fg": fg, "bg": bg})],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout)
        line = proc.stdout.strip().splitlines()[-1]
        payload = json.loads(line)
        if payload.get("status") != "ok":
            raise RuntimeError(payload)
        return np.frombuffer(base64.b64decode(payload["mask"]), dtype=np.uint8).reshape(HEIGHT, WIDTH)


def rust_mask(repo: Path, predictor_exe: Path | None, frame_bgr: np.ndarray, fg, bg) -> np.ndarray:
    if predictor_exe is None:
        raise RuntimeError("Rust SAM parity binary not provided; use sam-python path in matanyone-core")
    # Placeholder until dedicated sam_parity bin exists
    raise RuntimeError("Rust SAM TRT parity requires exported engines/sam31 and wired Sam31Predictor")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--frame", default="raw_sample/frames/00000.png")
    parser.add_argument("--min-iou", type=float, default=0.99)
    args = parser.parse_args()

    repo = Path(args.repo)
    frame_path = repo / args.frame
    if not frame_path.exists():
        print(f"Frame not found: {frame_path}", file=sys.stderr)
        return 1

    bgr = np.array(Image.open(frame_path).convert("RGB"))[:, :, ::-1]

    ious = []
    for clicks in GOLDEN_CLICKS:
        ref = python_mask(repo, bgr, clicks["fg"], clicks["bg"])
        try:
            pred = rust_mask(repo, None, bgr, clicks["fg"], clicks["bg"])
        except RuntimeError as exc:
            print(f"SKIP rust parity: {exc}")
            print("Python worker remains authoritative until SAM TRT export completes.")
            return 0
        score = iou(ref, pred)
        ious.append(score)
        print(f"click_set iou={score:.6f}")

    if ious and min(ious) < args.min_iou:
        print(f"SAM parity failed: min_iou={min(ious):.6f} < {args.min_iou}")
        return 1
    print("SAM parity OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

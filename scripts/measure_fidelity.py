"""Compare Rust runner alpha against the original MatAnyone2 InferenceCore.

Both sides get identical 720p float frames (written as a raw sample for
matanyone_runner_raw), so the reported difference is the TensorRT runner's deviation
from the reference PyTorch model, not preprocessing noise.

Example:
    uv run python scripts/measure_fidelity.py --frames 120 --runner new=bin/matanyone_runner_raw.exe
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion
from matanyone2_stateless import load_model

WIDTH = 1280
HEIGHT = 720


def read_video_frames(path: Path, count: int):
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        frames = [cv2.imread(str(p)) for p in files[:count]]
    else:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    out = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        out.append(np.ascontiguousarray(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0))
    return out


def load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    return cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)


def write_sample(sample_dir: Path, frames, mask_u8: np.ndarray) -> None:
    frames_dir = sample_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.rgbf32"):
        stale.unlink()
    for index, frame in enumerate(frames):
        frame.tofile(frames_dir / f"{index:05d}.rgbf32")
    (mask_u8.astype(np.float32)[None] / 255.0).tofile(sample_dir / "mask.f32")
    (sample_dir / "count.txt").write_text(str(len(frames)), encoding="utf-8")


def run_reference(frames, mask_u8: np.ndarray):
    device = torch.device("cuda")
    model = load_model(device="cuda")
    core = InferenceCore(model, cfg=model.cfg, device=device)
    mask_t = torch.from_numpy(mask_u8).float().to(device)
    alphas = []
    with torch.inference_mode():
        for index, frame in enumerate(frames):
            image = torch.from_numpy(frame).to(device)
            prob = core.step(image, mask_t, objects=[1]) if index == 0 else core.step(image)
            alphas.append(prob[1].float().cpu().numpy())
    return alphas


def metrics(ref: np.ndarray, pred: np.ndarray):
    diff = float(np.mean(np.abs(ref - pred)))
    ref_bin = ref > 0.5
    pred_bin = pred > 0.5
    union = float(np.logical_or(ref_bin, pred_bin).sum())
    inter = float(np.logical_and(ref_bin, pred_bin).sum())
    return diff, 1.0 if union == 0 else inter / union


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="vendor/MatAnyone2/inputs/video/test-sample2.mp4")
    parser.add_argument("--mask", default="vendor/MatAnyone2/inputs/mask/test-sample2.png")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--engine-dir", default="engines/faithful")
    parser.add_argument("--sample-dir", default="raw_sample")
    parser.add_argument("--out-dir", default="output/fidelity")
    parser.add_argument(
        "--runner",
        action="append",
        default=[],
        metavar="LABEL=EXE[,ENGINE_DIR]",
        help="matanyone_runner_raw executable to evaluate, optionally with its own engine dir (repeatable)",
    )
    parser.add_argument("--fail-mean-abs", type=float, default=0.0)
    args = parser.parse_args()

    runners = [r.split("=", 1) for r in args.runner] or [["runner", "bin/matanyone_runner_raw.exe"]]
    frames = read_video_frames(Path(args.video), args.frames)
    mask_u8 = load_mask(Path(args.mask))
    sample_dir = Path(args.sample_dir)
    write_sample(sample_dir, frames, mask_u8)
    print(f"sample={sample_dir} frames={len(frames)}")

    refs = run_reference(frames, mask_u8)
    del frames

    out_dir = Path(args.out_dir)
    worst = 0.0
    for label, spec in runners:
        exe, _, engine_dir = spec.partition(",")
        run_dir = out_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run(
            [exe, str(sample_dir), engine_dir or args.engine_dir, str(run_dir), str(len(refs))],
            check=True,
            text=True,
            capture_output=True,
        )
        timing = " ".join(line for line in cp.stdout.splitlines() if line.startswith(("mean_ms", "p99_ms")))
        diffs, ious = [], []
        for index, ref in enumerate(refs):
            pred = np.fromfile(run_dir / f"{index:05d}.alphaf32", dtype=np.float32).reshape(HEIGHT, WIDTH)
            diff, iou = metrics(ref, pred)
            diffs.append(diff)
            ious.append(iou)
        tail = diffs[-max(1, len(diffs) // 4):]
        mean_abs = float(np.mean(diffs))
        worst = max(worst, mean_abs)
        print(
            f"{label}: mean_abs={mean_abs:.5f} max_abs={max(diffs):.5f} "
            f"last_quarter_abs={float(np.mean(tail)):.5f} mean_iou={float(np.mean(ious)):.5f} "
            f"min_iou={min(ious):.5f} {timing}"
        )
    if args.fail_mean_abs and worst > args.fail_mean_abs:
        raise SystemExit(f"fidelity check failed: mean_abs {worst:.5f} > {args.fail_mean_abs:.5f}")


if __name__ == "__main__":
    main()

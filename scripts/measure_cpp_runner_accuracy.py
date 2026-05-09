import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_trt_pipeline import Pipeline
from measure_trt_accuracy import seed_memory, update_deep_memory

WIDTH = 1280
HEIGHT = 720


def read_f32(path: Path, shape):
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise RuntimeError(f"{path} has {arr.size} floats, expected {expected}")
    return arr.reshape(shape)


def frame_path(sample_dir: Path, index: int):
    return sample_dir / "frames" / f"{index:05d}.rgbf32"


def run_pipeline(engine_dir: Path, sample_dir: Path, frames: int, mem_every: int):
    pipeline = Pipeline(engine_dir)
    stream = torch.cuda.current_stream().cuda_stream
    alphas = []

    image0 = torch.from_numpy(read_f32(frame_path(sample_dir, 0), (1, 3, HEIGHT, WIDTH))).cuda()
    mask = torch.from_numpy(read_f32(sample_dir / "mask.f32", (1, 1, HEIGHT, WIDTH))).cuda()
    pipeline.tensors["image"].copy_(image0)
    pipeline.encode.run(stream)
    pipeline.tensors["alpha"].copy_(mask)
    pipeline.encode_mask.run(stream)
    torch.cuda.synchronize()
    seed_memory(pipeline)
    alphas.append(pipeline.tensors["alpha"][0, 0].detach().float().cpu().numpy().copy())

    memory_slot = 1
    with torch.inference_mode():
        for index in range(1, frames):
            pipeline.tensors["image"].copy_(torch.from_numpy(read_f32(frame_path(sample_dir, index), (1, 3, HEIGHT, WIDTH))).cuda())
            if index % mem_every == 0:
                pipeline.run_memory_update(stream)
                torch.cuda.synchronize()
                update_deep_memory(pipeline, memory_slot)
                memory_slot += 1
            else:
                pipeline.run_normal(stream)
                torch.cuda.synchronize()
            pipeline.tensors["last_pix_feat"].copy_(pipeline.tensors["pix_feat"])
            pipeline.tensors["last_mask"].copy_(pipeline.tensors["alpha"])
            alphas.append(pipeline.tensors["alpha"][0, 0].detach().float().cpu().numpy().copy())
    return alphas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", default="engines/faithful")
    parser.add_argument("--sample-dir", default="raw_sample")
    parser.add_argument("--runner-exe", default="")
    parser.add_argument("--out-dir", default="output/cpp_accuracy_alpha")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--mem-every", type=int, default=5)
    parser.add_argument("--fail-mean-abs", type=float, default=0.005)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    candidates = [
        Path(args.runner_exe) if args.runner_exe else None,
        Path("build/Release/matanyone_runner_raw.exe"),
        Path("build/matanyone_runner_raw.exe"),
        Path("build/Release/matanyone_runner_raw"),
        Path("build/matanyone_runner_raw"),
    ]
    exe = next((p for p in candidates if p and p.exists()), None)
    if exe is None:
        raise SystemExit("matanyone_runner_raw executable not found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run(
        [str(exe), args.sample_dir, args.engine_dir, str(out_dir), str(args.frames)],
        check=True,
        text=True,
        capture_output=True,
    )
    print(cp.stdout, end="")

    refs = run_pipeline(Path(args.engine_dir), Path(args.sample_dir), args.frames, args.mem_every)
    diffs = []
    ious = []
    for index, ref in enumerate(refs):
        pred = read_f32(out_dir / f"{index:05d}.alphaf32", (HEIGHT, WIDTH))
        diff = float(np.mean(np.abs(ref - pred)))
        ref_bin = ref > 0.5
        pred_bin = pred > 0.5
        inter = float(np.logical_and(ref_bin, pred_bin).sum())
        union = float(np.logical_or(ref_bin, pred_bin).sum())
        iou = 1.0 if union == 0 else inter / union
        diffs.append(diff)
        ious.append(iou)
        print(f"cpp_frame={index} mean_abs_diff={diff:.6f} iou={iou:.6f}")

    mean_abs = float(np.mean(diffs))
    max_abs = float(np.max(diffs))
    mean_iou = float(np.mean(ious))
    min_iou = float(np.min(ious))
    print(f"METRIC cpp_mean_abs_diff={mean_abs:.6f}")
    print(f"METRIC cpp_max_abs_diff={max_abs:.6f}")
    print(f"METRIC cpp_mean_iou={mean_iou:.6f}")
    print(f"METRIC cpp_min_iou={min_iou:.6f}")
    if mean_abs > args.fail_mean_abs:
        raise SystemExit(f"C++ runner accuracy check failed: cpp_mean_abs_diff {mean_abs:.6f} > {args.fail_mean_abs:.6f}")


if __name__ == "__main__":
    main()

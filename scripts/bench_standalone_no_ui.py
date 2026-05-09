import argparse
import os
import subprocess
import time
from pathlib import Path

# Keep CPU helper libraries from spin-waiting on a full worker pool while
# TensorRT owns the inference path on CUDA.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import torch

from bench_trt_pipeline import Pipeline
from composite_utils import composite_bgr_on_cpu, composite_rgb_tensor_to_bgr


WIDTH = 1280
HEIGHT = 720
FRAME_BYTES = WIDTH * HEIGHT * 3
DEFAULT_FFMPEG_DIR = "C:/Tools/ffmpeg-2026-05-06-git-f2e5eff3ff-full_build"

cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def ffmpeg_bin(args, name):
    if args.ffmpeg_dir:
        return str(Path(args.ffmpeg_dir) / "bin" / name)
    return name


def start_reader(args):
    if args.synthetic:
        return None
    if args.camera:
        input_args = ["-f", "dshow", "-rtbufsize", "200M", "-i", f"video={args.camera}"]
    else:
        input_args = ["-stream_loop", "-1", "-i", args.input]
    cmd = [
        ffmpeg_bin(args, "ffmpeg.exe"),
        "-hide_banner",
        "-loglevel",
        "error",
        *input_args,
        "-vf",
        f"scale={WIDTH}:{HEIGHT}",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def read_frame(proc, synthetic_frame):
    if proc is None:
        return synthetic_frame.copy()
    raw = proc.stdout.read(FRAME_BYTES)
    if len(raw) != FRAME_BYTES:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()


def make_synthetic_frame():
    x = np.linspace(0, 255, WIDTH, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, HEIGHT, dtype=np.uint8)[:, None]
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[..., 0] = x
    frame[..., 1] = y
    frame[..., 2] = 127
    return frame


def frame_to_tensor(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().cuda().unsqueeze(0).contiguous() / 255.0


def make_center_mask():
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cx0, cx1 = int(WIDTH * 0.28), int(WIDTH * 0.72)
    cy0, cy1 = int(HEIGHT * 0.08), int(HEIGHT * 0.96)
    cv2.ellipse(mask, (WIDTH // 2, int(HEIGHT * 0.48)), ((cx1 - cx0) // 2, (cy1 - cy0) // 2), 0, 0, 360, 255, -1)
    return torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0).contiguous() / 255.0


def composite(frame_bgr, alpha, color):
    return composite_bgr_on_cpu(frame_bgr, alpha, color)


def seed_memory(pipeline):
    for slot in range(5):
        pipeline.tensors["memory_key"][:, :, slot].copy_(pipeline.tensors["key"])
        pipeline.tensors["memory_shrinkage"][:, :, slot].copy_(pipeline.tensors["shrinkage"])
        pipeline.tensors["memory_value"][:, :, :, slot].copy_(pipeline.tensors["mask_value"])
    pipeline.tensors["obj_memory"][:, :, 0].copy_(pipeline.tensors["object_summaries"])
    pipeline.tensors["last_pix_feat"].copy_(pipeline.tensors["pix_feat"])
    pipeline.tensors["last_msk_value"].copy_(pipeline.tensors["mask_value"])
    pipeline.tensors["last_mask"].copy_(pipeline.tensors["alpha"])


def update_memory_slot(pipeline, slot):
    slot = min(slot, 4)
    pipeline.tensors["memory_key"][:, :, slot].copy_(pipeline.tensors["key"])
    pipeline.tensors["memory_shrinkage"][:, :, slot].copy_(pipeline.tensors["shrinkage"])
    pipeline.tensors["memory_value"][:, :, :, slot].copy_(pipeline.tensors["mask_value"])
    pipeline.tensors["obj_memory"][:, :, 0].copy_(pipeline.tensors["object_summaries"])
    pipeline.tensors["last_msk_value"].copy_(pipeline.tensors["mask_value"])


def summarize(name, values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr)) if len(arr) else 0.0
    p99 = float(np.percentile(arr, 99)) if len(arr) else 0.0
    print(f"{name}_mean_ms={mean:.3f}")
    print(f"{name}_p99_ms={p99:.3f}")
    print(f"METRIC {name}_mean_ms={mean:.3f}")
    print(f"METRIC {name}_p99_ms={p99:.3f}")
    return mean, p99


def main():
    parser = argparse.ArgumentParser(description="Headless benchmark for the standalone app processing loop.")
    parser.add_argument("--engine-dir", default="engines/faithful")
    parser.add_argument("--input", default="vendor/MatAnyone2/inputs/video/test-sample2.mp4")
    parser.add_argument("--camera", help="DirectShow camera name. If omitted, --input is looped.")
    parser.add_argument("--synthetic", action="store_true", help="Use generated frames; avoids FFmpeg/camera read cost.")
    parser.add_argument("--ffmpeg-dir", default=DEFAULT_FFMPEG_DIR)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--mem-every", type=int, default=5)
    parser.add_argument("--bg-color", default="0,180,80", help="B,G,R")
    parser.add_argument("--no-composite", action="store_true", help="Skip GPU alpha readback + CPU compositing.")
    args = parser.parse_args()

    color = tuple(int(x) for x in args.bg_color.split(","))
    synthetic_frame = make_synthetic_frame()
    reader = start_reader(args)
    pipeline = Pipeline(Path(args.engine_dir))

    first = read_frame(reader, synthetic_frame)
    if first is None:
        raise RuntimeError("No first frame")

    pipeline.tensors["image"].copy_(frame_to_tensor(first))
    pipeline.encode.run(torch.cuda.current_stream().cuda_stream)
    pipeline.tensors["alpha"].copy_(make_center_mask())
    pipeline.encode_mask.run(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    seed_memory(pipeline)

    timings = {k: [] for k in ["read", "pre", "infer", "state", "composite", "total"]}
    memory_slot = 1
    measured = 0
    total_iters = args.warmup + args.frames

    try:
        for frame_index in range(1, total_iters + 1):
            total_start = time.perf_counter()

            t = time.perf_counter()
            frame = read_frame(reader, synthetic_frame)
            read_ms = (time.perf_counter() - t) * 1000.0
            if frame is None:
                break

            t = time.perf_counter()
            pipeline.tensors["image"].copy_(frame_to_tensor(frame))
            torch.cuda.synchronize()
            pre_ms = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            if frame_index % args.mem_every == 0:
                pipeline.run_memory_update(torch.cuda.current_stream().cuda_stream)
                update_memory_slot(pipeline, memory_slot)
                memory_slot += 1
            else:
                pipeline.run_normal(torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            pipeline.tensors["last_pix_feat"].copy_(pipeline.tensors["pix_feat"])
            pipeline.tensors["last_mask"].copy_(pipeline.tensors["alpha"])
            torch.cuda.synchronize()
            state_ms = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            if not args.no_composite:
                composite_rgb_tensor_to_bgr(pipeline.tensors["image"], pipeline.tensors["alpha"], color)
            composite_ms = (time.perf_counter() - t) * 1000.0
            total_ms = (time.perf_counter() - total_start) * 1000.0

            if frame_index > args.warmup:
                timings["read"].append(read_ms)
                timings["pre"].append(pre_ms)
                timings["infer"].append(infer_ms)
                timings["state"].append(state_ms)
                timings["composite"].append(composite_ms)
                timings["total"].append(total_ms)
                measured += 1
    finally:
        if reader is not None:
            reader.terminate()

    print(f"frames={measured}")
    for name in ["read", "pre", "infer", "state", "composite", "total"]:
        summarize(name, timings[name])


if __name__ == "__main__":
    main()

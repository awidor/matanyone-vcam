import argparse
import json
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
from PIL import Image

from bench_trt_pipeline import Pipeline
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion


WIDTH = 1280
HEIGHT = 720
FRAME_BYTES = WIDTH * HEIGHT * 3

cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def run_json(cmd):
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def ffmpeg_bin(args, name):
    if args.ffmpeg_dir:
        return str(Path(args.ffmpeg_dir) / "bin" / name)
    return name


def list_devices(args):
    cmd = [
        ffmpeg_bin(args, "ffmpeg.exe"),
        "-hide_banner",
        "-list_devices",
        "true",
        "-f",
        "dshow",
        "-i",
        "dummy",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    print(completed.stderr)


def probe_fps(args, path):
    try:
        data = run_json(
            [
                ffmpeg_bin(args, "ffprobe.exe"),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "json",
                path,
            ]
        )
        rate = data["streams"][0]["r_frame_rate"]
        num, den = rate.split("/")
        fps = float(num) / float(den)
        return fps if fps > 0 else 30.0
    except Exception:
        return 30.0


def start_reader(args):
    if args.camera:
        input_args = ["-f", "dshow", "-i", f"video={args.camera}"]
        fps = args.fps
    else:
        input_args = ["-i", args.input]
        fps = probe_fps(args, args.input)

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
    return subprocess.Popen(cmd, stdout=subprocess.PIPE), fps


def start_writer(args, path, fps):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(args, "ffmpeg.exe"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        f"{fps:.03f}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def read_frame(proc):
    raw = proc.stdout.read(FRAME_BYTES)
    if len(raw) != FRAME_BYTES:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()


def frame_to_tensor(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().cuda().unsqueeze(0).contiguous() / 255.0


def load_mask(path):
    mask = np.array(Image.open(path).convert("L"))
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0).contiguous() / 255.0


def make_auto_mask(frame_bgr, mode):
    if mode == "center":
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cx0, cx1 = int(WIDTH * 0.28), int(WIDTH * 0.72)
        cy0, cy1 = int(HEIGHT * 0.08), int(HEIGHT * 0.96)
        cv2.ellipse(mask, (WIDTH // 2, int(HEIGHT * 0.48)), ((cx1 - cx0) // 2, (cy1 - cy0) // 2), 0, 0, 360, 255, -1)
        return torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0).contiguous() / 255.0
    if mode == "green":
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        bg = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
        mask = 255 - bg
        mask = cv2.medianBlur(mask, 7)
        return torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0).contiguous() / 255.0
    raise ValueError(f"Unknown auto mask mode: {mode}")


def composite(frame_bgr, alpha, color):
    frame = frame_bgr.astype(np.float32) / 255.0
    a = alpha[0, 0].detach().clamp(0, 1).cpu().numpy()[..., None]
    bg = np.zeros_like(frame)
    bg[..., 0] = color[0] / 255.0
    bg[..., 1] = color[1] / 255.0
    bg[..., 2] = color[2] / 255.0
    return np.clip((frame * a + bg * (1.0 - a)) * 255.0, 0, 255).astype(np.uint8)


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
    if slot >= 5:
        pipeline.tensors["memory_key"][:, :, 1:4].copy_(pipeline.tensors["memory_key"][:, :, 2:5].clone())
        pipeline.tensors["memory_shrinkage"][:, :, 1:4].copy_(
            pipeline.tensors["memory_shrinkage"][:, :, 2:5].clone()
        )
        pipeline.tensors["memory_value"][:, :, :, 1:4].copy_(pipeline.tensors["memory_value"][:, :, :, 2:5].clone())
        slot = 4
    pipeline.tensors["memory_key"][:, :, slot].copy_(pipeline.tensors["key"])
    pipeline.tensors["memory_shrinkage"][:, :, slot].copy_(pipeline.tensors["shrinkage"])
    pipeline.tensors["memory_value"][:, :, :, slot].copy_(pipeline.tensors["mask_value"])
    pipeline.tensors["obj_memory"][:, :, 0].copy_(pipeline.tensors["object_summaries"])
    pipeline.tensors["last_msk_value"].copy_(pipeline.tensors["mask_value"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="vendor/MatAnyone2/inputs/video/test-sample2.mp4")
    parser.add_argument("--camera", help="DirectShow camera name, for example 'USB Video'")
    parser.add_argument("--mask", default="vendor/MatAnyone2/inputs/mask/test-sample2.png")
    parser.add_argument("--auto-mask", choices=["center", "green"])
    parser.add_argument("--output", default="output/standalone_ffmpeg.mp4")
    parser.add_argument("--ffmpeg-dir", default="C:/Tools/ffmpeg-2026-05-06-git-f2e5eff3ff-full_build")
    parser.add_argument("--engine-dir", default="engines/faithful")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--mem-every", type=int, default=5)
    parser.add_argument("--bg-color", default="0,180,80", help="B,G,R")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices(args)
        return

    color = tuple(int(x) for x in args.bg_color.split(","))
    pipeline = Pipeline(Path(args.engine_dir))
    reader, fps = start_reader(args)
    writer = start_writer(args, args.output, fps)

    first = read_frame(reader)
    if first is None:
        raise RuntimeError("No first frame from FFmpeg")

    pipeline.tensors["image"].copy_(frame_to_tensor(first))
    pipeline.encode.run(torch.cuda.current_stream().cuda_stream)
    if args.auto_mask:
        pipeline.tensors["alpha"].copy_(make_auto_mask(first, args.auto_mask))
    else:
        pipeline.tensors["alpha"].copy_(load_mask(args.mask))
    pipeline.encode_mask.run(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    seed_memory(pipeline)

    timings = []
    frame_index = 0
    memory_slot = 1
    frame = first
    try:
        while frame is not None:
            if frame_index > 0:
                pipeline.tensors["image"].copy_(frame_to_tensor(frame))
                is_mem_frame = frame_index % args.mem_every == 0
                start = time.perf_counter()
                if is_mem_frame:
                    pipeline.run_memory_update(torch.cuda.current_stream().cuda_stream)
                    update_memory_slot(pipeline, memory_slot)
                    memory_slot += 1
                else:
                    pipeline.run_normal(torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)
                pipeline.tensors["last_pix_feat"].copy_(pipeline.tensors["pix_feat"])
                pipeline.tensors["last_mask"].copy_(pipeline.tensors["alpha"])

            out = composite(frame, pipeline.tensors["alpha"], color)
            writer.stdin.write(out.tobytes())
            if args.preview:
                cv2.imshow("MatAnyone Standalone", out)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            frame_index += 1
            if args.frames and frame_index >= args.frames:
                break
            frame = read_frame(reader)
    finally:
        if writer.stdin:
            writer.stdin.close()
        writer.wait()
        reader.terminate()
        if args.preview:
            cv2.destroyAllWindows()

    if timings:
        print(f"frames={frame_index}")
        print(f"mean_ms={float(np.mean(timings)):.3f}")
        print(f"p99_ms={float(np.percentile(timings, 99)):.3f}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()

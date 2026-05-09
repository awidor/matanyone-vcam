import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_VENDOR = REPO_ROOT / "vendor" / "MatAnyone2"
sys.path.insert(0, str(REPO_VENDOR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_trt_pipeline import Pipeline
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion
from matanyone2_stateless import load_model


WIDTH = 1280
HEIGHT = 720


def read_frames(video_path: Path, count: int):
    cap = cv2.VideoCapture(str(video_path))
    frames_bgr = []
    frames_rgb_t = []
    while len(frames_bgr) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        frames_bgr.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb_t.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    cap.release()
    if not frames_bgr:
        raise RuntimeError(f"No frames read from {video_path}")
    return frames_bgr, frames_rgb_t


def frame_to_tensor(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().cuda().unsqueeze(0).contiguous() / 255.0


def load_mask(path: Path):
    mask = np.array(Image.open(path).convert("L"))
    mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    return mask


def seed_memory(pipeline: Pipeline):
    for slot in range(5):
        pipeline.tensors["memory_key"][:, :, slot].copy_(pipeline.tensors["key"])
        pipeline.tensors["memory_shrinkage"][:, :, slot].copy_(pipeline.tensors["shrinkage"])
        pipeline.tensors["memory_value"][:, :, :, slot].copy_(pipeline.tensors["mask_value"])
    pipeline.tensors["obj_memory"][:, :, 0].copy_(pipeline.tensors["object_summaries"])
    pipeline.tensors["last_pix_feat"].copy_(pipeline.tensors["pix_feat"])
    pipeline.tensors["last_msk_value"].copy_(pipeline.tensors["mask_value"])
    pipeline.tensors["last_mask"].copy_(pipeline.tensors["alpha"])


def update_deep_memory(pipeline: Pipeline, memory_slot: int):
    # Keep slot 0 as the initial mask and retain the four most recent update frames.
    if memory_slot >= 5:
        pipeline.tensors["memory_key"][:, :, 1:4].copy_(pipeline.tensors["memory_key"][:, :, 2:5].clone())
        pipeline.tensors["memory_shrinkage"][:, :, 1:4].copy_(pipeline.tensors["memory_shrinkage"][:, :, 2:5].clone())
        pipeline.tensors["memory_value"][:, :, :, 1:4].copy_(pipeline.tensors["memory_value"][:, :, :, 2:5].clone())
        slot = 4
    else:
        slot = memory_slot
    pipeline.tensors["memory_key"][:, :, slot].copy_(pipeline.tensors["key"])
    pipeline.tensors["memory_shrinkage"][:, :, slot].copy_(pipeline.tensors["shrinkage"])
    pipeline.tensors["memory_value"][:, :, :, slot].copy_(pipeline.tensors["mask_value"])
    pipeline.tensors["obj_memory"][:, :, 0].copy_(pipeline.tensors["object_summaries"])
    pipeline.tensors["last_msk_value"].copy_(pipeline.tensors["mask_value"])


def run_reference(frames_rgb_t, mask_np):
    device = torch.device("cuda")
    model = load_model(device="cuda")
    original = InferenceCore(model, cfg=model.cfg, device=device)
    mask_t = torch.from_numpy(mask_np).float().to(device)
    refs = []
    with torch.inference_mode():
        for index, frame in enumerate(frames_rgb_t):
            frame = frame.to(device)
            if index == 0:
                prob = original.step(frame, mask_t, objects=[1])
            else:
                prob = original.step(frame)
            refs.append(prob[1].detach().float().clone())
    return refs


def run_trt(engine_dir: Path, frames_bgr, mask_np, mem_every: int):
    pipeline = Pipeline(engine_dir)
    stream = torch.cuda.current_stream().cuda_stream
    mask_t = torch.from_numpy(mask_np).float().cuda().unsqueeze(0).unsqueeze(0).contiguous() / 255.0
    alphas = []

    pipeline.tensors["image"].copy_(frame_to_tensor(frames_bgr[0]))
    pipeline.encode.run(stream)
    pipeline.tensors["alpha"].copy_(mask_t)
    pipeline.encode_mask.run(stream)
    torch.cuda.synchronize()
    seed_memory(pipeline)
    alphas.append(pipeline.tensors["alpha"][0, 0].detach().float().clone())

    memory_slot = 1
    with torch.inference_mode():
        for index, frame in enumerate(frames_bgr[1:], start=1):
            pipeline.tensors["image"].copy_(frame_to_tensor(frame))
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
            alphas.append(pipeline.tensors["alpha"][0, 0].detach().float().clone())
    return alphas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", default="engines/faithful")
    parser.add_argument("--video", default="vendor/MatAnyone2/inputs/video/test-sample2.mp4")
    parser.add_argument("--mask", default="vendor/MatAnyone2/inputs/mask/test-sample2.png")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--mem-every", type=int, default=5)
    parser.add_argument("--fail-mean-abs", type=float, default=0.03)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)

    frames_bgr, frames_rgb_t = read_frames(Path(args.video), args.frames)
    mask_np = load_mask(Path(args.mask))
    refs = run_reference(frames_rgb_t, mask_np)
    got = run_trt(Path(args.engine_dir), frames_bgr, mask_np, args.mem_every)

    diffs = []
    ious = []
    for index, (ref, pred) in enumerate(zip(refs, got)):
        ref = ref.cuda()
        pred = pred.cuda()
        diff = (ref - pred).abs().mean().item()
        ref_bin = ref > 0.5
        pred_bin = pred > 0.5
        inter = (ref_bin & pred_bin).sum().item()
        union = (ref_bin | pred_bin).sum().item()
        iou = 1.0 if union == 0 else inter / union
        diffs.append(diff)
        ious.append(iou)
        print(f"frame={index} mean_abs_diff={diff:.6f} iou={iou:.6f}")

    mean_abs = float(np.mean(diffs))
    max_abs = float(np.max(diffs))
    mean_iou = float(np.mean(ious))
    min_iou = float(np.min(ious))
    print(f"METRIC mean_abs_diff={mean_abs:.6f}")
    print(f"METRIC max_abs_diff={max_abs:.6f}")
    print(f"METRIC mean_iou={mean_iou:.6f}")
    print(f"METRIC min_iou={min_iou:.6f}")
    if mean_abs > args.fail_mean_abs:
        raise SystemExit(f"Accuracy check failed: mean_abs_diff {mean_abs:.6f} > {args.fail_mean_abs:.6f}")


if __name__ == "__main__":
    main()

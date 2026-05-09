import argparse
import os
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
import torch.nn.functional as F
from PIL import Image

from bench_trt_pipeline import Pipeline
from composite_utils import composite_bgr_on_cpu, composite_rgb_tensor_to_bgr
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def to_tensor(frame_bgr):
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().cuda() / 255.0
    return tensor.unsqueeze(0).contiguous()


def load_mask(path):
    mask = np.array(Image.open(path).convert("L"))
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
    tensor = torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.contiguous()


def composite(frame_bgr, alpha, color):
    frame = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
    return composite_bgr_on_cpu(frame, alpha, color)


def open_capture(args):
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open input source")
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="vendor/MatAnyone2/inputs/video/test-sample2.mp4")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--mask", default="vendor/MatAnyone2/inputs/mask/test-sample2.png")
    parser.add_argument("--engine-dir", default="engines/faithful")
    parser.add_argument("--output", default="output/standalone_preview.mp4")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="0 means until source ends")
    parser.add_argument("--mem-every", type=int, default=5)
    parser.add_argument("--bg-color", default="0,180,80", help="B,G,R")
    args = parser.parse_args()

    color = tuple(int(x) for x in args.bg_color.split(","))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(Path(args.engine_dir))
    stream = torch.cuda.current_stream().cuda_stream
    cap = open_capture(args)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("No first frame")

    pipeline.tensors["image"].copy_(to_tensor(frame))
    pipeline.encode.run(stream)
    pipeline.tensors["alpha"].copy_(load_mask(args.mask))
    pipeline.encode_mask.run(stream)
    torch.cuda.synchronize()

    for slot in range(5):
        pipeline.tensors["memory_key"][:, :, slot].copy_(pipeline.tensors["key"])
        pipeline.tensors["memory_shrinkage"][:, :, slot].copy_(pipeline.tensors["shrinkage"])
        pipeline.tensors["memory_value"][:, :, :, slot].copy_(pipeline.tensors["mask_value"])
    pipeline.tensors["obj_memory"][:, :, 0].copy_(pipeline.tensors["object_summaries"])
    pipeline.tensors["last_pix_feat"].copy_(pipeline.tensors["pix_feat"])
    pipeline.tensors["last_msk_value"].copy_(pipeline.tensors["mask_value"])
    pipeline.tensors["last_mask"].copy_(pipeline.tensors["alpha"])

    frame_index = 0
    timings = []
    while True:
        if frame_index > 0:
            ok, frame = cap.read()
            if not ok:
                break
            pipeline.tensors["image"].copy_(to_tensor(frame))
            is_mem_frame = frame_index % args.mem_every == 0
            start = time.perf_counter()
            if is_mem_frame:
                pipeline.run_memory_update(stream)
                slot = min(frame_index // args.mem_every, 4)
                pipeline.tensors["memory_key"][:, :, slot].copy_(pipeline.tensors["key"])
                pipeline.tensors["memory_shrinkage"][:, :, slot].copy_(pipeline.tensors["shrinkage"])
                pipeline.tensors["memory_value"][:, :, :, slot].copy_(pipeline.tensors["mask_value"])
                pipeline.tensors["obj_memory"][:, :, 0].copy_(pipeline.tensors["object_summaries"])
                pipeline.tensors["last_msk_value"].copy_(pipeline.tensors["mask_value"])
            else:
                pipeline.run_normal(stream)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
            pipeline.tensors["last_pix_feat"].copy_(pipeline.tensors["pix_feat"])
            pipeline.tensors["last_mask"].copy_(pipeline.tensors["alpha"])

        out = composite_rgb_tensor_to_bgr(pipeline.tensors["image"], pipeline.tensors["alpha"], color)
        writer.write(out)
        if args.preview:
            cv2.imshow("MatAnyone Standalone", out)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_index += 1
        if args.frames and frame_index >= args.frames:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    if timings:
        print(f"frames={frame_index}")
        print(f"mean_ms={float(np.mean(timings)):.3f}")
        print(f"p99_ms={float(np.percentile(timings, 99)):.3f}")
    print(f"wrote={output}")


if __name__ == "__main__":
    main()

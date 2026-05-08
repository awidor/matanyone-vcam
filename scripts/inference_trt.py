import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bench_trt_pipeline import Pipeline
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion


def frame_to_tensor(frame_bgr):
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().cuda() / 255.0
    tensor = F.interpolate(tensor.unsqueeze(0), size=(720, 1280), mode="bilinear", align_corners=False)
    return tensor.contiguous()


def mask_to_tensor(path):
    mask = np.array(Image.open(path).convert("L"))
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    tensor = torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = F.interpolate(tensor, size=(720, 1280), mode="nearest")
    return tensor.contiguous()


def write_alpha(writer, alpha):
    img = (alpha[0, 0].detach().clamp(0, 1).mul(255).byte().cpu().numpy())
    writer.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="vendor/MatAnyone2/inputs/video/test-sample2.mp4")
    parser.add_argument("--mask", default="vendor/MatAnyone2/inputs/mask/test-sample2.png")
    parser.add_argument("--engine-dir", default="engines/faithful")
    parser.add_argument("--output", default="output/trt_alpha.mp4")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--mem-every", type=int, default=5)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(Path(args.engine_dir))
    stream = torch.cuda.current_stream().cuda_stream

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 720),
    )

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {args.video}")

    pipeline.tensors["image"].copy_(frame_to_tensor(frame))
    pipeline.encode.run(stream)
    pipeline.tensors["alpha"].copy_(mask_to_tensor(args.mask))
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
    write_alpha(writer, pipeline.tensors["alpha"])

    timings = []
    mem_slot = 1
    frame_index = 1
    while frame_index < args.frames:
        ok, frame = cap.read()
        if not ok:
            break

        pipeline.tensors["image"].copy_(frame_to_tensor(frame))
        is_mem_frame = frame_index % args.mem_every == 0
        start = time.perf_counter()
        if is_mem_frame:
            pipeline.run_memory_update(stream)
        else:
            pipeline.run_normal(stream)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

        if is_mem_frame:
            slot = min(mem_slot, 4)
            if mem_slot >= 5:
                pipeline.tensors["memory_key"][:, :, 1:4].copy_(pipeline.tensors["memory_key"][:, :, 2:5].clone())
                pipeline.tensors["memory_shrinkage"][:, :, 1:4].copy_(
                    pipeline.tensors["memory_shrinkage"][:, :, 2:5].clone()
                )
                pipeline.tensors["memory_value"][:, :, :, 1:4].copy_(
                    pipeline.tensors["memory_value"][:, :, :, 2:5].clone()
                )
                slot = 4
            pipeline.tensors["memory_key"][:, :, slot].copy_(pipeline.tensors["key"])
            pipeline.tensors["memory_shrinkage"][:, :, slot].copy_(pipeline.tensors["shrinkage"])
            pipeline.tensors["memory_value"][:, :, :, slot].copy_(pipeline.tensors["mask_value"])
            pipeline.tensors["obj_memory"][:, :, 0].add_(pipeline.tensors["object_summaries"])
            pipeline.tensors["last_msk_value"].copy_(pipeline.tensors["mask_value"])
            mem_slot += 1

        pipeline.tensors["last_pix_feat"].copy_(pipeline.tensors["pix_feat"])
        pipeline.tensors["last_mask"].copy_(pipeline.tensors["alpha"])
        write_alpha(writer, pipeline.tensors["alpha"])
        frame_index += 1

    cap.release()
    writer.release()
    if timings:
        print(f"frames={len(timings) + 1}")
        print(f"mean_ms={float(np.mean(timings)):.3f}")
        print(f"p99_ms={float(np.percentile(timings, 99)):.3f}")
        steady = timings[min(5, len(timings)) :]
        if steady:
            print(f"steady_mean_ms={float(np.mean(steady)):.3f}")
            print(f"steady_p99_ms={float(np.percentile(steady, 99)):.3f}")
    print(f"wrote={output}")


if __name__ == "__main__":
    main()

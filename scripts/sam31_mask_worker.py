#!/usr/bin/env python3
import argparse
import base64
import contextlib
import json
import os
import sys
import tempfile
import traceback
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image


WIDTH = 1280
HEIGHT = 720


def _emit(payload):
    print(json.dumps(payload), flush=True)


def _debug_write_mask(mask):
    debug_path = os.environ.get("SAM31_DEBUG_MASK")
    if not debug_path:
        return
    try:
        from PIL import Image

        Image.fromarray(mask).save(debug_path)
    except Exception as exc:
        print(f"Could not write SAM3.1 debug mask: {exc}", file=sys.stderr)


def _load_sam31():
    repo = os.environ.get("SAM31_REPO")
    if repo:
        sys.path.insert(0, repo)
    try:
        from sam3.model_builder import build_sam3_predictor
    except Exception as exc:
        raise RuntimeError(
            "Could not import SAM3.1. Install facebookresearch/sam3 in the Python "
            "interpreter used by SAM31_PYTHON, or set SAM31_REPO to a local sam3 clone."
        ) from exc

    checkpoint = os.environ.get("SAM31_CHECKPOINT")
    return build_sam3_predictor(
        version="sam3.1",
        checkpoint_path=checkpoint,
        compile=os.environ.get("SAM31_COMPILE", "0") == "1",
        warm_up=os.environ.get("SAM31_WARM_UP", "0") == "1",
        async_loading_frames=False,
        use_fa3=os.environ.get("SAM31_USE_FA3", "0") == "1",
    )


def _prepare_session_image(image_path, work_dir):
    image = Image.open(image_path).convert("RGB").resize((WIDTH, HEIGHT))
    frame_dir = Path(work_dir) / "frames"
    frame_dir.mkdir()
    frame_path = frame_dir / "00000.jpg"
    image.save(frame_path, quality=95)
    return frame_dir


def _mask_from_outputs(outputs):
    masks = outputs.get("out_binary_masks")
    if masks is None:
        masks = outputs.get("pred_masks")
    if masks is None:
        raise RuntimeError(f"SAM3.1 output did not include masks: {list(outputs.keys())}")

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.size == 0:
        return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    masks = np.squeeze(masks)
    if masks.ndim == 3:
        masks = masks[0]
    if masks.shape != (HEIGHT, WIDTH):
        raise RuntimeError(f"Unexpected SAM3.1 mask shape: {masks.shape}")
    return (masks > 0).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3.1 requires a CUDA GPU in the official implementation.")
        with contextlib.redirect_stdout(sys.stderr):
            predictor = _load_sam31()
        with tempfile.TemporaryDirectory(prefix="sam31_session_") as work_dir:
            frame_dir = _prepare_session_image(args.image, work_dir)
            with contextlib.redirect_stdout(sys.stderr):
                inference_state = predictor.model.init_state(
                    resource_path=str(frame_dir),
                    offload_video_to_cpu=False,
                    async_loading_frames=False,
                )
            _emit({"status": "ready"})

            try:
                for line in sys.stdin:
                    request = json.loads(line)
                    if request.get("quit"):
                        break

                    fg_pts = request.get("fg", [])
                    bg_pts = request.get("bg", [])
                    if not fg_pts and not bg_pts:
                        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                    else:
                        points = [
                            [float(x) / float(WIDTH), float(y) / float(HEIGHT)]
                            for x, y in (fg_pts + bg_pts)
                        ]
                        labels = [1] * len(fg_pts) + [0] * len(bg_pts)
                        with contextlib.redirect_stdout(sys.stderr):
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                _, outputs = predictor.model.add_prompt(
                                    inference_state=inference_state,
                                    frame_idx=0,
                                    points=torch.tensor(points, dtype=torch.float32),
                                    point_labels=torch.tensor(labels, dtype=torch.int32),
                                    obj_id=1,
                                    rel_coordinates=True,
                                    clear_old_points=True,
                                )
                        mask = _mask_from_outputs(outputs)

                    _debug_write_mask(mask)
                    _emit(
                        {
                            "status": "ok",
                            "mask": base64.b64encode(mask.tobytes()).decode("ascii"),
                        }
                    )
            finally:
                del inference_state
                gc.collect()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"status": "error", "error": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_W, _H = 1280, 720
_DEFAULT_CHECKPOINT = _PROJECT_ROOT / "models" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _build_predictor(frame_bgr, checkpoint, model_cfg):
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2.1 checkpoint not found: {checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(model_cfg, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return predictor


def _rerun(predictor, fg_pts, bg_pts, mask):
    all_pts = fg_pts + bg_pts
    all_lbls = [1] * len(fg_pts) + [0] * len(bg_pts)
    if not all_pts:
        mask[:] = 0
        return

    coords = np.array(all_pts, dtype=np.float32)
    labels = np.array(all_lbls, dtype=np.int32)
    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
        else:
            masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)

    if masks is None or len(masks) == 0:
        mask[:] = 0
        return

    m = masks[int(np.argmax(scores))]
    if m.shape != (_H, _W):
        m = cv2.resize(m.astype(np.uint8), (_W, _H), cv2.INTER_NEAREST)
    mask[:] = (m > 0).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="output/initial_mask.png")
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--model-cfg", default=_DEFAULT_CONFIG)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Could not read {args.image}")
    img = cv2.resize(img, (_W, _H), interpolation=cv2.INTER_LINEAR)

    predictor = _build_predictor(img, args.checkpoint, args.model_cfg)
    fg_pts, bg_pts, history = [], [], []
    mask = np.zeros((_H, _W), dtype=np.uint8)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            fg_pts.append([x, y])
            history.append("fg")
            _rerun(predictor, fg_pts, bg_pts, mask)
        elif event == cv2.EVENT_RBUTTONDOWN:
            bg_pts.append([x, y])
            history.append("bg")
            _rerun(predictor, fg_pts, bg_pts, mask)

    cv2.namedWindow("SAM2.1 Mask")
    cv2.setMouseCallback("SAM2.1 Mask", on_mouse)

    while True:
        preview = img.copy()
        overlay = preview.copy()
        overlay[mask > 0] = (0, 255, 0)
        preview = cv2.addWeighted(preview, 0.65, overlay, 0.35, 0)
        for x, y in fg_pts:
            cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)
        for x, y in bg_pts:
            cv2.circle(preview, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("SAM2.1 Mask", preview)

        key = cv2.waitKey(16) & 0xFF
        if key == ord("s"):
            break
        if key == ord("c"):
            fg_pts.clear()
            bg_pts.clear()
            history.clear()
            mask[:] = 0
        elif key == ord("z") and history:
            if history.pop() == "fg":
                fg_pts.pop()
            else:
                bg_pts.pop()
            _rerun(predictor, fg_pts, bg_pts, mask)
        elif key in (27, ord("q")):
            cv2.destroyAllWindows()
            sys.exit(1)

    cv2.destroyAllWindows()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), mask)
    print(f"wrote={output}")


if __name__ == "__main__":
    main()

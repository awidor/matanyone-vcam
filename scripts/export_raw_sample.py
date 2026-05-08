import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from matanyone2.utils.inference_utils import gen_dilate, gen_erosion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="vendor/MatAnyone2/inputs/video/test-sample2.mp4")
    parser.add_argument("--mask", default="vendor/MatAnyone2/inputs/mask/test-sample2.png")
    parser.add_argument("--out-dir", default="raw_sample")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    count = 0
    while count < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        tensor = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor.tofile(frames_dir / f"{count:05d}.rgbf32")
        count += 1
    cap.release()

    mask = np.array(Image.open(args.mask).convert("L"))
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
    (mask.astype(np.float32)[None] / 255.0).tofile(out_dir / "mask.f32")
    (out_dir / "count.txt").write_text(str(count), encoding="utf-8")
    print(f"wrote={out_dir} frames={count}")


if __name__ == "__main__":
    main()

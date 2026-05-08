import argparse
from pathlib import Path

import cv2
import numpy as np


points = []
drawing = False


def on_mouse(event, x, y, flags, _):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="output/initial_mask.png")
    parser.add_argument("--brush", type=int, default=24)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not read {args.image}")
    image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    preview = image.copy()

    cv2.namedWindow("Draw target mask")
    cv2.setMouseCallback("Draw target mask", on_mouse)

    while True:
        preview[:] = image
        for p0, p1 in zip(points, points[1:]):
            cv2.line(mask, p0, p1, 255, args.brush)
        overlay = preview.copy()
        overlay[mask > 0] = (0, 255, 0)
        preview = cv2.addWeighted(preview, 0.65, overlay, 0.35, 0)
        cv2.imshow("Draw target mask", preview)
        key = cv2.waitKey(16) & 0xFF
        if key == ord("s"):
            break
        if key == ord("c"):
            points.clear()
            mask[:] = 0
        if key == 27 or key == ord("q"):
            raise SystemExit(1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), mask)
    cv2.destroyAllWindows()
    print(f"wrote={output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "scripts" / "sam31_mask_worker.py"
WIDTH = 1280
HEIGHT = 720


class Sam31Client:
    def __init__(self, frame_bgr):
        self._tmp = tempfile.TemporaryDirectory(prefix="matanyone_sam31_")
        self._frame_path = Path(self._tmp.name) / "frame.png"
        cv2.imwrite(str(self._frame_path), frame_bgr)
        python = os.environ.get("SAM31_PYTHON", sys.executable)
        self._proc = subprocess.Popen(
            [python, str(WORKER), "--image", str(self._frame_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        ready = self._read_response()
        if ready.get("status") != "ready":
            raise RuntimeError(ready.get("error", "SAM3.1 worker failed to start"))

    def _drain_stderr(self):
        for line in self._proc.stderr:
            self._stderr.append(line.rstrip())

    def _read_response(self):
        while True:
            line = self._proc.stdout.readline()
            if not line:
                detail = "\n".join(self._stderr[-8:])
                raise RuntimeError(f"SAM3.1 worker exited:\n{detail}")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self._stderr.append(line.rstrip())
                continue
            if isinstance(response, dict) and "status" in response:
                return response

    def predict(self, fg_pts, bg_pts):
        self._proc.stdin.write(json.dumps({"fg": fg_pts, "bg": bg_pts}) + "\n")
        self._proc.stdin.flush()
        response = self._read_response()
        if response.get("status") != "ok":
            raise RuntimeError(response.get("error", "SAM3.1 prompt failed"))
        raw = base64.b64decode(response["mask"])
        return np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH)).copy()

    def close(self):
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write(json.dumps({"quit": True}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5.0)
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
            self._tmp.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="output/initial_mask.png")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Could not read {args.image}")
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    client = Sam31Client(img)
    fg_pts, bg_pts, history = [], [], []
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    def rerun():
        nonlocal mask
        mask = client.predict(fg_pts, bg_pts)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            fg_pts.append([x, y])
            history.append("fg")
            rerun()
        elif event == cv2.EVENT_RBUTTONDOWN:
            bg_pts.append([x, y])
            history.append("bg")
            rerun()

    cv2.namedWindow("SAM3.1 Mask")
    cv2.setMouseCallback("SAM3.1 Mask", on_mouse)

    try:
        while True:
            preview = img.copy()
            overlay = preview.copy()
            overlay[mask > 0] = (0, 255, 0)
            preview = cv2.addWeighted(preview, 0.65, overlay, 0.35, 0)
            for x, y in fg_pts:
                cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)
            for x, y in bg_pts:
                cv2.circle(preview, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("SAM3.1 Mask", preview)

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
                rerun()
            elif key in (27, ord("q")):
                cv2.destroyAllWindows()
                return 1
    finally:
        client.close()

    cv2.destroyAllWindows()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), mask)
    print(f"wrote={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

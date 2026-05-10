import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
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
import pyvirtualcam
import torch
from PySide6 import QtCore, QtGui, QtWidgets

from bench_trt_pipeline import Pipeline
from composite_utils import composite_bgr_on_cpu, composite_rgb_tensor_to_bgr


WIDTH = 1280
HEIGHT = 720
FRAME_BYTES = WIDTH * HEIGHT * 3
FPS = 30.0
BG_COLOR = (0, 180, 80)
FFMPEG_DIR = Path("C:/Tools/ffmpeg-2026-05-06-git-f2e5eff3ff-full_build")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAM31_PYTHON = os.environ.get("SAM31_PYTHON", sys.executable)
SAM31_WORKER = PROJECT_ROOT / "scripts" / "sam31_mask_worker.py"

cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def ffmpeg_exe(name):
    return str(FFMPEG_DIR / "bin" / name)


def list_dshow_devices():
    cmd = [ffmpeg_exe("ffmpeg.exe"), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    devices = []
    for line in completed.stderr.splitlines():
        if ' "' in line and '" (' in line and "(video)" in line:
            devices.append(line.split('"')[1])
    return devices


class FFmpegReader:
    def __init__(self, source):
        input_args = ["-f", "dshow", "-rtbufsize", "200M", "-i", f"video={source}"]
        cmd = [
            ffmpeg_exe("ffmpeg.exe"),
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
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self._cond = threading.Condition()
        self._latest = None
        self._frame_count = 0
        self._last_seen = 0
        self._eof = False
        self._stop = False
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self):
        try:
            while not self._stop:
                raw = self.proc.stdout.read(FRAME_BYTES)
                if len(raw) != FRAME_BYTES:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()
                with self._cond:
                    self._latest = frame
                    self._frame_count += 1
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._eof = True
                self._cond.notify_all()

    def read(self):
        with self._cond:
            while not self._eof and self._frame_count == self._last_seen:
                self._cond.wait(timeout=5.0)
            if self._frame_count == self._last_seen:
                return None
            self._last_seen = self._frame_count
            return self._latest

    def close(self):
        self._stop = True
        if self.proc.poll() is None:
            self.proc.terminate()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class FramePacer:
    def __init__(self, fps):
        self.interval = 1.0 / fps
        self.next_frame_time = time.perf_counter()

    def sleep_until_next_frame(self):
        self.next_frame_time += self.interval
        now = time.perf_counter()
        delay = self.next_frame_time - now
        if delay <= 0:
            self.next_frame_time = now
            return 0.0
        time.sleep(delay)
        return delay * 1000.0


def frame_to_tensor(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().cuda().unsqueeze(0).contiguous() / 255.0


def mask_to_tensor(mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0).contiguous() / 255.0


class Sam31Client:
    def __init__(self, frame_bgr):
        if not SAM31_WORKER.exists():
            raise FileNotFoundError(f"SAM3.1 worker not found: {SAM31_WORKER}")

        self._tmp = tempfile.TemporaryDirectory(prefix="matanyone_sam31_")
        self._frame_path = Path(self._tmp.name) / "frame.png"
        cv2.imwrite(str(self._frame_path), frame_bgr)
        cmd = [SAM31_PYTHON, str(SAM31_WORKER), "--image", str(self._frame_path)]
        self._proc = subprocess.Popen(
            cmd,
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
        if self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            self._stderr.append(line.rstrip())

    def _read_response(self):
        if self._proc.stdout is None:
            raise RuntimeError("SAM3.1 worker stdout is unavailable")
        while True:
            line = self._proc.stdout.readline()
            if not line:
                detail = "\n".join(self._stderr[-8:])
                if detail:
                    raise RuntimeError(f"SAM3.1 worker exited:\n{detail}")
                raise RuntimeError("SAM3.1 worker exited")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self._stderr.append(line.rstrip())
                continue
            if isinstance(response, dict) and "status" in response:
                return response

    def predict(self, fg_pts, bg_pts):
        if self._proc.poll() is not None:
            detail = "\n".join(self._stderr[-8:])
            raise RuntimeError(f"SAM3.1 worker is not running:\n{detail}")
        payload = {"fg": fg_pts, "bg": bg_pts}
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        response = self._read_response()
        if response.get("status") != "ok":
            raise RuntimeError(response.get("error", "SAM3.1 prompt failed"))
        raw = base64.b64decode(response["mask"])
        return np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH)).copy()

    def close(self):
        try:
            if self._proc.poll() is None and self._proc.stdin is not None:
                self._proc.stdin.write(json.dumps({"quit": True}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5.0)
        except Exception:
            if self._proc.poll() is None:
                self._proc.terminate()
        finally:
            self._tmp.cleanup()


def composite(frame_bgr, alpha, color):
    return composite_bgr_on_cpu(frame_bgr, alpha, color)


def open_virtual_camera():
    """Open the virtual camera output used by conferencing apps.

    pyvirtualcam's OBS backend is itself the producer for "OBS Virtual Camera".
    It cannot attach to an OBS virtual-camera output that OBS Studio has already
    started, so force the OBS backend and provide an actionable error instead of
    falling through to the unrelated UnityCapture backend.
    """
    try:
        return pyvirtualcam.Camera(
            width=WIDTH,
            height=HEIGHT,
            fps=int(FPS),
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend="obs",
        )
    except RuntimeError as exc:
        message = str(exc)
        if "already started" in message.lower():
            raise RuntimeError(
                "OBS Virtual Camera is already started by OBS Studio. Stop it in OBS first; "
                "MatAnyone must own/start the OBS Virtual Camera output so it can publish frames."
            ) from exc
        raise RuntimeError(f"Could not open OBS Virtual Camera output: {message}") from exc


class Sam31MaskDialog(QtWidgets.QDialog):
    def __init__(self, frame_bgr):
        super().__init__()
        self.setWindowTitle("SAM3.1 Target Mask")
        self.frame = frame_bgr.copy()
        self.mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self.fg_pts = []
        self.bg_pts = []
        self.history = []
        self.predictor = Sam31Client(self.frame)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(960, 540)
        self.image_label.mousePressEvent = self.mouse_press
        self.status = QtWidgets.QLabel("Left-click subject. Right-click background to refine.")
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self.clear)
        undo = QtWidgets.QPushButton("Undo")
        undo.clicked.connect(self.undo)
        ok = QtWidgets.QPushButton("Use Mask")
        ok.clicked.connect(self.accept)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(clear)
        row.addWidget(undo)
        row.addWidget(ok)
        layout.addLayout(row)
        self.refresh()

    def map_pos(self, event):
        x = int(event.position().x() * WIDTH / self.image_label.width())
        y = int(event.position().y() * HEIGHT / self.image_label.height())
        return max(0, min(WIDTH - 1, x)), max(0, min(HEIGHT - 1, y))

    def mouse_press(self, event):
        x, y = self.map_pos(event)
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.bg_pts.append([x, y])
            self.history.append("bg")
        else:
            self.fg_pts.append([x, y])
            self.history.append("fg")
        self.run_prompt()

    def run_prompt(self):
        self.status.setText("Running SAM3.1...")
        QtWidgets.QApplication.processEvents()
        try:
            self.mask[:] = self.predictor.predict(self.fg_pts, self.bg_pts)
            self.status.setText(f"Foreground clicks: {len(self.fg_pts)}  Background clicks: {len(self.bg_pts)}")
        except Exception as exc:
            self.status.setText(f"SAM3.1 error: {exc}")
        self.refresh()

    def clear(self):
        self.fg_pts.clear()
        self.bg_pts.clear()
        self.history.clear()
        self.mask[:] = 0
        self.refresh()

    def undo(self):
        if not self.history:
            return
        if self.history.pop() == "fg":
            self.fg_pts.pop()
        else:
            self.bg_pts.pop()
        self.run_prompt()

    def refresh(self):
        overlay = self.frame.copy()
        overlay[self.mask > 0] = (0, 255, 0)
        image = cv2.addWeighted(self.frame, 0.65, overlay, 0.35, 0)
        for x, y in self.fg_pts:
            cv2.circle(image, (x, y), 6, (0, 0, 255), -1)
        for x, y in self.bg_pts:
            cv2.circle(image, (x, y), 6, (255, 0, 0), -1)
        rgb = cv2.cvtColor(cv2.resize(image, (960, 540)), cv2.COLOR_BGR2RGB)
        qimage = QtGui.QImage(rgb.data, 960, 540, 960 * 3, QtGui.QImage.Format_RGB888).copy()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimage))

    def cleanup(self):
        if self.predictor is not None:
            self.predictor.close()
            self.predictor = None

    def accept(self):
        if not np.any(self.mask):
            self.status.setText("Click the subject before using the mask.")
            return
        self.cleanup()
        super().accept()

    def reject(self):
        self.cleanup()
        super().reject()


class Worker(QtCore.QThread):
    frame_ready = QtCore.Signal(object, object)
    status = QtCore.Signal(str)
    finished_cleanly = QtCore.Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

    def seed_memory(self, pipeline):
        for slot in range(5):
            pipeline.tensors["memory_key"][:, :, slot].copy_(pipeline.tensors["key"])
            pipeline.tensors["memory_shrinkage"][:, :, slot].copy_(pipeline.tensors["shrinkage"])
            pipeline.tensors["memory_value"][:, :, :, slot].copy_(pipeline.tensors["mask_value"])
        pipeline.tensors["obj_memory"][:, :, 0].copy_(pipeline.tensors["object_summaries"])
        pipeline.tensors["last_pix_feat"].copy_(pipeline.tensors["pix_feat"])
        pipeline.tensors["last_msk_value"].copy_(pipeline.tensors["mask_value"])
        pipeline.tensors["last_mask"].copy_(pipeline.tensors["alpha"])

    def update_memory_slot(self, pipeline, slot):
        slot = min(slot, 4)
        pipeline.tensors["memory_key"][:, :, slot].copy_(pipeline.tensors["key"])
        pipeline.tensors["memory_shrinkage"][:, :, slot].copy_(pipeline.tensors["shrinkage"])
        pipeline.tensors["memory_value"][:, :, :, slot].copy_(pipeline.tensors["mask_value"])
        pipeline.tensors["obj_memory"][:, :, 0].copy_(pipeline.tensors["object_summaries"])
        pipeline.tensors["last_msk_value"].copy_(pipeline.tensors["mask_value"])

    def run(self):
        reader = None
        vcam = None
        try:
            reader = FFmpegReader(self.config["source"])
            first = reader.read()
            if first is None:
                raise RuntimeError("No first frame")

            pipeline = Pipeline(Path(self.config["engine_dir"]))
            pipeline.tensors["image"].copy_(frame_to_tensor(first))
            pipeline.encode.run(torch.cuda.current_stream().cuda_stream)
            pipeline.tensors["alpha"].copy_(mask_to_tensor(self.config["mask"]))
            pipeline.encode_mask.run(torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            self.seed_memory(pipeline)

            if self.config.get("output_virtual_camera", False):
                vcam = open_virtual_camera()

            pacer = FramePacer(FPS)
            frame_index = 0
            memory_slot = 1
            frame = first
            while frame is not None and not self.stop_requested:
                start = time.perf_counter()
                pre_ms = infer_ms = state_ms = composite_ms = vcam_send_ms = pacing_wait_ms = 0.0
                if frame_index > 0:
                    t = time.perf_counter()
                    pipeline.tensors["image"].copy_(frame_to_tensor(frame))
                    torch.cuda.synchronize()
                    pre_ms = (time.perf_counter() - t) * 1000.0

                    t = time.perf_counter()
                    if frame_index % self.config["mem_every"] == 0:
                        pipeline.run_memory_update(torch.cuda.current_stream().cuda_stream)
                        self.update_memory_slot(pipeline, memory_slot)
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
                out = composite_rgb_tensor_to_bgr(pipeline.tensors["image"], pipeline.tensors["alpha"], self.config["bg_color"])
                composite_ms = (time.perf_counter() - t) * 1000.0
                pacing_wait_ms = pacer.sleep_until_next_frame()
                if vcam is not None:
                    t = time.perf_counter()
                    vcam.send(out)
                    vcam_send_ms = (time.perf_counter() - t) * 1000.0
                elapsed = (time.perf_counter() - start) * 1000.0
                profile = {
                    "total": elapsed,
                    "pre": pre_ms,
                    "infer": infer_ms,
                    "state": state_ms,
                    "composite": composite_ms,
                    "vcam_send": vcam_send_ms,
                    "pacing_wait": pacing_wait_ms,
                    "compute_total": elapsed - pacing_wait_ms,
                }
                self.frame_ready.emit(out, profile)
                frame_index += 1
                frame = reader.read()
            self.finished_cleanly.emit()
        except Exception as exc:
            self.status.emit(f"Error: {exc}")
        finally:
            if vcam:
                vcam.close()
            if reader:
                reader.close()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MatAnyone Standalone")
        self.worker = None
        self.initial_mask = None
        self.first_frame = None

        self.camera = QtWidgets.QComboBox()
        self.camera.addItems(list_dshow_devices())
        refresh = QtWidgets.QPushButton("Refresh Cameras")
        refresh.clicked.connect(self.refresh_cameras)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.output_vcam = QtWidgets.QCheckBox("Publish processed output to virtual camera")
        self.output_vcam.setToolTip(
            "Leave this off when using OBS Virtual Camera as the input. "
            "OBS Virtual Camera cannot be both this app's input and output."
        )
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.preview = QtWidgets.QLabel()
        self.preview.setMinimumSize(960, 540)
        self.preview.setStyleSheet("background:#111")
        self.status_bar = QtWidgets.QLabel("Idle")

        form = QtWidgets.QGridLayout()
        form.addWidget(QtWidgets.QLabel("Input Camera"), 0, 0)
        form.addWidget(self.camera, 0, 1)
        form.addWidget(refresh, 0, 2)
        form.addWidget(self.output_vcam, 1, 1, 1, 2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(row)
        layout.addWidget(self.preview)
        layout.addWidget(self.status_bar)
        root = QtWidgets.QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

    def refresh_cameras(self):
        self.camera.clear()
        self.camera.addItems(list_dshow_devices())

    def grab_first_frame(self):
        reader = FFmpegReader(self.camera.currentText())
        frame = reader.read()
        reader.close()
        if frame is None:
            raise RuntimeError("Could not read first frame")
        return frame

    def build_mask(self, first):
        dialog = Sam31MaskDialog(first)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            raise RuntimeError("SAM3.1 mask selection canceled")
        return dialog.mask

    def start(self):
        try:
            first = self.grab_first_frame()
            mask = self.build_mask(first)
            source = self.camera.currentText()
            output_vcam = self.output_vcam.isChecked()
            if output_vcam and "obs virtual camera" in source.lower():
                raise RuntimeError(
                    "OBS Virtual Camera is already the input. Choose a different input camera, "
                    "or turn off virtual-camera output and use the on-screen preview."
                )
            config = {
                "source": source,
                "mask": mask,
                "engine_dir": "engines/faithful",
                "mem_every": 5,
                "bg_color": BG_COLOR,
                "output_virtual_camera": output_vcam,
            }
            self.worker = Worker(config)
            self.worker.frame_ready.connect(self.on_frame)
            self.worker.status.connect(self.status_bar.setText)
            self.worker.finished_cleanly.connect(self.on_finished)
            self.worker.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_bar.setText("Running")
        except Exception as exc:
            self.status_bar.setText(f"Error: {exc}")

    def stop(self):
        if self.worker:
            self.worker.stop()

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_bar.setText("Stopped")

    def on_frame(self, frame_bgr, profile):
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (960, 540)), cv2.COLOR_BGR2RGB)
        qimage = QtGui.QImage(rgb.data, 960, 540, 960 * 3, QtGui.QImage.Format_RGB888).copy()
        self.preview.setPixmap(QtGui.QPixmap.fromImage(qimage))
        if profile is None:
            return
        if isinstance(profile, dict):
            self.status_bar.setText(
                f"total {profile['total']:.1f} ms | compute {profile['compute_total']:.1f} "
                f"(pre {profile['pre']:.1f}, infer {profile['infer']:.1f}, state {profile['state']:.1f}, "
                f"comp {profile['composite']:.1f}, send {profile['vcam_send']:.1f}) | pace {profile['pacing_wait']:.1f}"
            )
        else:
            self.status_bar.setText(f"{profile:.1f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1040, 820)
    window.show()
    if args.smoke_test:
        QtCore.QTimer.singleShot(500, app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

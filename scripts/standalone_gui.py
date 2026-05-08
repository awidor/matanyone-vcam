import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyvirtualcam
import torch
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from bench_trt_pipeline import Pipeline
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion


WIDTH = 1280
HEIGHT = 720
FRAME_BYTES = WIDTH * HEIGHT * 3
FFMPEG_DIR = Path("C:/Tools/ffmpeg-2026-05-06-git-f2e5eff3ff-full_build")


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
    def __init__(self, source, is_camera=False):
        if is_camera:
            input_args = ["-f", "dshow", "-rtbufsize", "200M", "-i", f"video={source}"]
        else:
            input_args = ["-i", source]
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
        self.is_camera = is_camera
        if is_camera:
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
        if self.is_camera:
            with self._cond:
                while not self._eof and self._frame_count == self._last_seen:
                    self._cond.wait(timeout=5.0)
                if self._frame_count == self._last_seen:
                    return None
                self._last_seen = self._frame_count
                return self._latest
        raw = self.proc.stdout.read(FRAME_BYTES)
        if len(raw) != FRAME_BYTES:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

    def close(self):
        if self.is_camera:
            self._stop = True
        if self.proc.poll() is None:
            self.proc.terminate()
        if self.is_camera and self._thread.is_alive():
            self._thread.join(timeout=1.0)


class FFmpegWriter:
    def __init__(self, path, fps=30.0):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg_exe("ffmpeg.exe"),
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
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr):
        self.proc.stdin.write(frame_bgr.tobytes())

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()


def frame_to_tensor(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().cuda().unsqueeze(0).contiguous() / 255.0


def mask_to_tensor(mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0).contiguous() / 255.0


def load_mask(path):
    mask = np.array(Image.open(path).convert("L"))
    mask = gen_dilate(mask, 10, 10)
    mask = gen_erosion(mask, 10, 10)
    return mask


def auto_mask(frame_bgr, mode):
    if mode == "center":
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.ellipse(mask, (WIDTH // 2, int(HEIGHT * 0.48)), (int(WIDTH * 0.22), int(HEIGHT * 0.44)), 0, 0, 360, 255, -1)
        return mask
    if mode == "green":
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        bg = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
        return cv2.medianBlur(255 - bg, 7)
    raise ValueError(mode)


def composite(frame_bgr, alpha, color):
    frame = frame_bgr.astype(np.float32) / 255.0
    a = alpha[0, 0].detach().clamp(0, 1).cpu().numpy()[..., None]
    bg = np.zeros_like(frame)
    bg[..., 0] = color[0] / 255.0
    bg[..., 1] = color[1] / 255.0
    bg[..., 2] = color[2] / 255.0
    return np.clip((frame * a + bg * (1.0 - a)) * 255.0, 0, 255).astype(np.uint8)


class MaskDialog(QtWidgets.QDialog):
    def __init__(self, frame_bgr):
        super().__init__()
        self.setWindowTitle("Draw Target Mask")
        self.frame = frame_bgr.copy()
        self.mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self.brush = 28
        self.drawing = False
        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(960, 540)
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self.clear)
        ok = QtWidgets.QPushButton("Use Mask")
        ok.clicked.connect(self.accept)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(clear)
        row.addWidget(ok)
        layout.addLayout(row)
        self.refresh()

    def map_pos(self, event):
        x = int(event.position().x() * WIDTH / self.image_label.width())
        y = int(event.position().y() * HEIGHT / self.image_label.height())
        return max(0, min(WIDTH - 1, x)), max(0, min(HEIGHT - 1, y))

    def paint_at(self, event):
        cv2.circle(self.mask, self.map_pos(event), self.brush, 255, -1)
        self.refresh()

    def mouse_press(self, event):
        self.drawing = True
        self.paint_at(event)

    def mouse_move(self, event):
        if self.drawing:
            self.paint_at(event)

    def mouse_release(self, event):
        self.drawing = False
        self.paint_at(event)

    def clear(self):
        self.mask[:] = 0
        self.refresh()

    def refresh(self):
        overlay = self.frame.copy()
        overlay[self.mask > 0] = (0, 255, 0)
        image = cv2.addWeighted(self.frame, 0.65, overlay, 0.35, 0)
        rgb = cv2.cvtColor(cv2.resize(image, (960, 540)), cv2.COLOR_BGR2RGB)
        qimage = QtGui.QImage(rgb.data, 960, 540, 960 * 3, QtGui.QImage.Format_RGB888).copy()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimage))


class Worker(QtCore.QThread):
    frame_ready = QtCore.Signal(object, float)
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
        writer = None
        vcam = None
        try:
            reader = FFmpegReader(self.config["source"], self.config["is_camera"])
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

            if self.config["record"]:
                writer = FFmpegWriter(self.config["output"], self.config["fps"])
            if self.config["virtual_camera"]:
                vcam = pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=int(self.config["fps"]), fmt=pyvirtualcam.PixelFormat.BGR)

            frame_index = 0
            memory_slot = 1
            frame = first
            while frame is not None and not self.stop_requested:
                start = time.perf_counter()
                if frame_index > 0:
                    pipeline.tensors["image"].copy_(frame_to_tensor(frame))
                    if frame_index % self.config["mem_every"] == 0:
                        pipeline.run_memory_update(torch.cuda.current_stream().cuda_stream)
                        self.update_memory_slot(pipeline, memory_slot)
                        memory_slot += 1
                    else:
                        pipeline.run_normal(torch.cuda.current_stream().cuda_stream)
                    torch.cuda.synchronize()
                    pipeline.tensors["last_pix_feat"].copy_(pipeline.tensors["pix_feat"])
                    pipeline.tensors["last_mask"].copy_(pipeline.tensors["alpha"])

                out = composite(frame, pipeline.tensors["alpha"], self.config["bg_color"])
                if writer:
                    writer.write(out)
                if vcam:
                    vcam.send(out)
                    vcam.sleep_until_next_frame()
                elapsed = (time.perf_counter() - start) * 1000.0
                self.frame_ready.emit(out, elapsed)
                frame_index += 1
                frame = reader.read()
            self.finished_cleanly.emit()
        except Exception as exc:
            self.status.emit(f"Error: {exc}")
        finally:
            if writer:
                writer.close()
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

        self.source_type = QtWidgets.QComboBox()
        self.source_type.addItems(["Video File", "Camera"])
        self.source = QtWidgets.QLineEdit("vendor/MatAnyone2/inputs/video/test-sample2.mp4")
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.browse_video)
        self.camera = QtWidgets.QComboBox()
        self.camera.addItems(list_dshow_devices())
        refresh = QtWidgets.QPushButton("Refresh Cameras")
        refresh.clicked.connect(self.refresh_cameras)
        self.mask_mode = QtWidgets.QComboBox()
        self.mask_mode.addItems(["Draw", "Mask File", "Auto Center", "Green Screen"])
        self.mask_path = QtWidgets.QLineEdit("vendor/MatAnyone2/inputs/mask/test-sample2.png")
        mask_browse = QtWidgets.QPushButton("Browse Mask")
        mask_browse.clicked.connect(self.browse_mask)
        self.output = QtWidgets.QLineEdit("output/standalone_gui.mp4")
        self.record = QtWidgets.QCheckBox("Record MP4")
        self.record.setChecked(True)
        self.vcam = QtWidgets.QCheckBox("Virtual Camera")
        self.bg = QtWidgets.QPushButton("Background Color")
        self.bg_color = (0, 180, 80)
        self.bg.clicked.connect(self.pick_color)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.preview = QtWidgets.QLabel()
        self.preview.setMinimumSize(960, 540)
        self.preview.setStyleSheet("background:#111")
        self.status_bar = QtWidgets.QLabel("Idle")

        form = QtWidgets.QGridLayout()
        form.addWidget(QtWidgets.QLabel("Source"), 0, 0)
        form.addWidget(self.source_type, 0, 1)
        form.addWidget(self.source, 1, 1)
        form.addWidget(browse, 1, 2)
        form.addWidget(self.camera, 2, 1)
        form.addWidget(refresh, 2, 2)
        form.addWidget(QtWidgets.QLabel("Mask"), 3, 0)
        form.addWidget(self.mask_mode, 3, 1)
        form.addWidget(self.mask_path, 4, 1)
        form.addWidget(mask_browse, 4, 2)
        form.addWidget(self.record, 5, 1)
        form.addWidget(self.output, 6, 1)
        form.addWidget(self.vcam, 7, 1)
        form.addWidget(self.bg, 8, 1)
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

    def browse_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Video")
        if path:
            self.source.setText(path)

    def browse_mask(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Mask", filter="Images (*.png *.jpg *.bmp)")
        if path:
            self.mask_path.setText(path)

    def refresh_cameras(self):
        self.camera.clear()
        self.camera.addItems(list_dshow_devices())

    def pick_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.bg_color = (color.blue(), color.green(), color.red())

    def grab_first_frame(self):
        is_camera = self.source_type.currentText() == "Camera"
        source = self.camera.currentText() if is_camera else self.source.text()
        reader = FFmpegReader(source, is_camera)
        frame = reader.read()
        reader.close()
        if frame is None:
            raise RuntimeError("Could not read first frame")
        return frame

    def build_mask(self, first):
        mode = self.mask_mode.currentText()
        if mode == "Mask File":
            return load_mask(self.mask_path.text())
        if mode == "Auto Center":
            return auto_mask(first, "center")
        if mode == "Green Screen":
            return auto_mask(first, "green")
        dialog = MaskDialog(first)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            raise RuntimeError("Mask drawing canceled")
        return dialog.mask

    def start(self):
        try:
            first = self.grab_first_frame()
            mask = self.build_mask(first)
            is_camera = self.source_type.currentText() == "Camera"
            config = {
                "source": self.camera.currentText() if is_camera else self.source.text(),
                "is_camera": is_camera,
                "mask": mask,
                "engine_dir": "engines/faithful",
                "record": self.record.isChecked(),
                "output": self.output.text(),
                "virtual_camera": self.vcam.isChecked(),
                "fps": 30.0,
                "mem_every": 5,
                "bg_color": self.bg_color,
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

    def on_frame(self, frame_bgr, ms):
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (960, 540)), cv2.COLOR_BGR2RGB)
        qimage = QtGui.QImage(rgb.data, 960, 540, 960 * 3, QtGui.QImage.Format_RGB888).copy()
        self.preview.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.status_bar.setText(f"{ms:.1f} ms")


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

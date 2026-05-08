# Standalone App Architecture

Date: 2026-05-08

## Goal

Pivot from OBS plugin to a standalone real-time matting app:

1. Capture webcam/video.
2. Initialize target from a first-frame mask.
3. Run MatAnyone TensorRT pipeline.
4. Preview composited output.
5. Later expose output as a virtual camera.

## Recommended Path

### Stage 1: Standalone Processor

Use the existing TensorRT pipeline and FFmpeg pipes for video I/O, with Python/OpenCV only for preview display and mask preparation.

Pros:

- Works with FFmpeg as the standard media boundary.
- Fastest path to validate usability.
- Avoids OBS SDK and virtual camera SDK issues.

### Stage 2: Live Preview

Use OpenCV webcam capture and a preview window.

Target initialization options:

- Load first-frame mask from file.
- Draw/edit mask in UI.
- Add MediaPipe/SAM helper later.

### Stage 3: Virtual Camera

Options:

- OBS Virtual Camera as a consumer target: easiest if OBS is installed, but still needs a frame injection route.
- UnityCapture/DirectShow-style virtual camera: requires a virtual camera driver/filter.
- Spout/Syphon-like Windows route: useful for OBS intake but not a general virtual camera.
- Native Windows Media Foundation virtual camera: most correct but highest implementation cost.

Recommended virtual camera path: use an existing virtual camera/sink package first, then only build a native driver/filter if needed.

## Current Local Constraints

- OBS runtime is installed, but OBS development headers/import libs are not.
- FFmpeg is assumed to exist on PATH for the standalone app path.
- Python OpenCV is available through `uv`.
- TensorRT C++ SDK is available.
- C++ runner and Python TRT pipeline both work.

## Immediate Implementation Target

Build a standalone Python app around the faithful TensorRT pipeline and FFmpeg:

```powershell
uv run python scripts\standalone_ffmpeg.py --input vendor\MatAnyone2\inputs\video\test-sample2.mp4 --mask vendor\MatAnyone2\inputs\mask\test-sample2.png --preview
```

Launch the desktop app:

```powershell
.\run_standalone_gui.ps1
```

The app supports:

- Video file input through FFmpeg.
- DirectShow camera input through FFmpeg.
- Drawn target mask on the first frame.
- Mask file input.
- Auto-center and green-screen placeholder masks.
- Live preview.
- MP4 recording.
- OBS Virtual Camera output through `pyvirtualcam` when enabled.

Then add webcam mode:

```powershell
uv run python scripts\standalone_ffmpeg.py --camera "Camera Device Name" --mask path\to\mask.png --preview
```

List DirectShow devices:

```powershell
uv run python scripts\standalone_ffmpeg.py --list-devices
```

Generate a first-frame image and draw a mask:

```powershell
uv run python scripts\extract_first_frame_ffmpeg.py --input input.mp4 --output output\first_frame.png
uv run python scripts\make_initial_mask.py --image output\first_frame.png --output output\initial_mask.png
```

In the mask UI: draw with the left mouse button, `s` saves, `c` clears, `q`/Esc exits.

Automatic placeholder masks are available for quick testing:

```powershell
uv run python scripts\standalone_ffmpeg.py --input input.mp4 --auto-mask center --output output\auto_center.mp4
uv run python scripts\standalone_ffmpeg.py --input input.mp4 --auto-mask green --output output\green_screen.mp4
```

FFmpeg path can be supplied explicitly:

```powershell
uv run python scripts\standalone_ffmpeg.py --ffmpeg-dir C:\Tools\ffmpeg-2026-05-06-git-f2e5eff3ff-full_build --input vendor\MatAnyone2\inputs\video\test-sample2.mp4 --mask vendor\MatAnyone2\inputs\mask\test-sample2.png --output output\standalone_ffmpeg.mp4
```

Sample run:

```text
frames=30
mean_ms=22.912
p99_ms=46.134
wrote=output\standalone_ffmpeg_30.mp4
```

GUI smoke test:

```powershell
uv run python scripts\standalone_gui.py --smoke-test
```

Virtual camera backend check:

```text
OBS Virtual Camera
```

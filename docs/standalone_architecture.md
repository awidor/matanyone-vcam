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
- SAM3.1 click target mask on the first frame.
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
uv run python scripts\make_initial_mask_sam31.py --image output\first_frame.png --output output\initial_mask.png
```

In the standalone GUI, left-click the subject once on the first frame. Right-click background areas only if the mask needs refinement.

In the helper mask UI: left-click = foreground point, right-click = background point, `s` saves, `c` clears all points, `z` undoes last click, `q`/Esc exits.

### First-time SAM3.1 setup

SAM3.1 runs in a worker process but uses the same `.venv` as the MatAnyone app, so it shares the existing CUDA PyTorch install instead of downloading another large torch wheel. Install the official SAM3 repository and the Windows Triton package:

```powershell
git clone https://github.com/facebookresearch/sam3.git vendor\sam3
uv pip install --python .venv\Scripts\python.exe -e vendor\sam3 --no-deps
uv pip install --python .venv\Scripts\python.exe timm>=1.0.17 ftfy==6.1.1 regex triton-windows
hf auth login
$env:SAM31_REPO = "$PWD\vendor\sam3"
```

Optional: set `SAM31_CHECKPOINT` to a local `.pt` checkpoint path to avoid Hugging Face downloads at runtime. The tool fails immediately if SAM3.1 cannot be imported, CUDA is unavailable, or the checkpoint cannot be loaded.

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

## Headless Latency Benchmark

The desktop GUI status can report about `60 ms/frame`, while TensorRT-only benchmark scripts report roughly `22-30 ms/frame`. These are measuring different scopes:

- TensorRT benchmark: model pipeline execution on CUDA.
- GUI/app loop: frame ingest, CPU→GPU conversion, TensorRT inference, state copies, GPU→CPU alpha readback, CPU compositing, preview/virtual-camera work, and virtual-camera pacing wait.

Use the headless app-loop benchmark to measure this programmatically without UI:

```powershell
uv run python scripts\bench_standalone_no_ui.py --synthetic --frames 300 --warmup 30
```

Useful variants:

```powershell
# Isolate processing without FFmpeg/camera input and without compositing.
uv run python scripts\bench_standalone_no_ui.py --synthetic --frames 300 --warmup 30 --no-composite

# Loop a real video input through FFmpeg, but still avoid UI.
uv run python scripts\bench_standalone_no_ui.py --input vendor\MatAnyone2\inputs\video\test-sample2.mp4 --frames 300 --warmup 30

# Benchmark a DirectShow camera input.
uv run python scripts\bench_standalone_no_ui.py --camera "Camera Device Name" --frames 300 --warmup 30
```

The script emits both human-readable values and `METRIC` lines for automated optimization:

```text
METRIC read_mean_ms=...
METRIC pre_mean_ms=...
METRIC infer_mean_ms=...
METRIC state_mean_ms=...
METRIC composite_mean_ms=...
METRIC total_mean_ms=...
```

Metric meanings:

- `read`: reading/decoding or generating the next frame.
- `pre`: BGR→RGB, NumPy→Torch, CPU→GPU copy, float normalization.
- `infer`: TensorRT MatAnyone pipeline plus memory-update path when scheduled.
- `state`: persistent tensor copies (`last_pix_feat`, `last_mask`, etc.).
- `composite`: current CPU composite path, including `alpha.cpu().numpy()` GPU readback.
- `total`: full headless processing loop excluding UI preview and virtual-camera pacing.

Initial synthetic-frame measurements showed:

```text
--no-composite total_mean_ms ≈ 27.8
with composite total_mean_ms ≈ 62.7
composite_mean_ms ≈ 33.7
infer_mean_ms ≈ 24.4
```

Conclusion: the observed ~60 ms app latency is mostly compositing/readback overhead, not TensorRT inference. The next optimization target should be replacing `composite()` with a GPU path or otherwise avoiding per-frame `alpha.cpu().numpy()` and float NumPy blending.

The GUI status bar also reports a breakdown:

```text
total X ms | compute Y (pre A, infer B, state C, comp D, send E) | wait Z
```

`wait` is virtual-camera frame pacing (`sleep_until_next_frame`) and should not be compared to model benchmark latency.

# MatAnyone VCam

Real-time background matting for a webcam, published as a virtual camera that any video app can use. A Rust app on Windows runs [MatAnyone2](https://github.com/pq-yang/MatAnyone2) through TensorRT at 720p.

- **Pick the subject once.** Click the first frame with SAM 3.1 (left click: subject, right click: background), load a mask PNG, or start from a centre mask. MatAnyone2 tracks it from there.
- **Use it like a webcam.** Output goes to a virtual camera (Unity Video Capture) that other apps can select. OBS isn't required.
- **Preview it live.** An egui window shows the composite while it runs.

There are no prebuilt downloads. TensorRT engines are specific to a GPU and TensorRT version, so you export them on the machine that runs the app.

## Performance

RTX 3080, 1280×720, FP16 engines ([`docs/benchmark_baselines.json`](docs/benchmark_baselines.json)):

| Measurement | Mean | p99 |
| :--- | ---: | ---: |
| Frame latency at 30 fps, BGR in to composited BGR out | 15.1 ms | 17.2 ms |
| Mixed frames, Rust runner | 16.3 ms | 17.8 ms |
| Normal frame, engines only | 17.6 ms | 18.8 ms |
| Memory-update frame, engines only | 18.1 ms | 19.7 ms |

That leaves about half of the 33 ms frame budget of a 30 fps camera. MatAnyone2 runs as five TensorRT engines, replayed as CUDA graphs. The memory read was the slowest part until it was rewritten as plain attention (one similarity GEMM and a softmax along the contiguous axis), which took it from 10.7 ms to 4.6 ms. Alpha differs from the original PyTorch model by a mean of 0.0002 on the sample clips (`scripts/measure_fidelity.py`). See [`docs/profile_optimization.md`](docs/profile_optimization.md).

## Requirements

- Windows with an NVIDIA RTX 30 or 40 series GPU. The bundled CUDA kernels are compiled for `sm_86`, which runs on both.
- Rust, Visual Studio C++ build tools, and LLVM (libclang, for `bindgen`).
- CUDA 13.x and TensorRT 10.16.1.11 for CUDA 13 (`TensorRT-10.16.1.11.Windows.amd64.cuda-13.2.zip`). Engines only load in the TensorRT version that built them, so this must match the `tensorrt` Python package exactly. Set `CUDA_ROOT` and `TENSORRT_ROOT` if they aren't at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1` and `C:\Tools\TensorRT-10.16.1.11`.
- [uv](https://docs.astral.sh/uv/) for exporting the engines (Python 3.10, PyTorch with CUDA 12.8).
- [Unity Video Capture](https://github.com/schellingb/UnityCapture), registered as a camera device.

## Build

Export the engines. This downloads the MatAnyone2 checkpoint and builds the five engines the app loads into `engines/faithful/` (about 10 minutes).

```powershell
uv sync
uv run python scripts/build_faithful_engines.py
```

Build the app. This copies `matanyone-vcam.exe` and the TensorRT and CUDA runtime DLLs to the repository root.

```powershell
.\build.ps1
.\build.ps1 -Bundle -Zip   # optional portable folder in dist/
```

## Run

```powershell
.\matanyone-vcam.exe
```

Choose the input camera, pick the subject, and press **Start**. In your video app, select **Unity Video Capture** as the camera.

SAM 3.1 click selection runs in a Python worker. Install [facebookresearch/sam3](https://github.com/facebookresearch/sam3) into a Python environment and point `SAM31_PYTHON` at its interpreter. `SAM31_CHECKPOINT` selects a local checkpoint. Without SAM, use a mask PNG or the centre mask.

## Development

```powershell
.\bin\matanyone_core_smoke.exe --synthetic --frames=200 --interval-ms=33 engines\faithful   # frame latency at 30 fps, no camera
.\bin\matanyone_runner_bench.exe engines\faithful
.\scripts\check_bench_parity.ps1                               # latency within ±2% of the baselines
uv run python scripts/measure_fidelity.py --runner runner=bin\matanyone_runner_raw.exe   # alpha vs the original model
```

After editing `crates/matanyone-core/kernels/matanyone_kernels.cu`, regenerate the embedded kernels with `.\scripts\build_kernels.ps1`.

| Path | Contents |
| :--- | :--- |
| `app/` | Capture, preview, SAM clicks, and virtual camera output (egui, nokhwa, virtualcam) |
| `crates/matanyone-core/` | TensorRT runner, CUDA kernels, and the `Session` API |
| `scripts/` | Engine export, benchmarks, accuracy checks, and the Python reference app |
| `docs/` | Design notes, profiling, and benchmark baselines |

This started as an OBS filter plugin. That code was removed when the project became a standalone app.

## License

The code in this repository is [GPL-3.0](LICENSE).

MatAnyone2 and its weights use the [NTU S-Lab License 1.0](https://github.com/pq-yang/MatAnyone2/blob/main/LICENSE.txt), which allows non-commercial use only. Engines exported from those weights fall under the same terms, so don't use them commercially, including for paid streaming, without the authors' permission. See [`docs/license_audit.md`](docs/license_audit.md). SAM 3.1 has its own license from Meta.

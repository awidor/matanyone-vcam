# MatAnyone VCam

Real-time background matting for a webcam, published as a virtual camera that any video app can use. A Rust app on Windows runs [MatAnyone2](https://github.com/pq-yang/MatAnyone2) through TensorRT at 720p.

- **Pick the subject once.** Click the first frame with SAM 3.1 (left click: subject, right click: background), load a mask PNG, or start from a centre mask. MatAnyone2 tracks it from there.
- **Use it like a webcam.** Output goes to a virtual camera (Unity Video Capture) that other apps can select. OBS isn't required.
- **Preview it live.** An egui window shows the composite while it runs.

There are no prebuilt downloads. TensorRT engines are specific to a GPU and TensorRT version, so you export them on the machine that runs the app.

## Performance

RTX 3080, 1280×720, FP16 engines ([`docs/benchmark_baselines.json`](docs/benchmark_baselines.json)):

| Frame | Mean | p99 |
| :--- | ---: | ---: |
| Mixed frames, Rust runner | 21.1 ms | 23.2 ms |
| Normal frame | 21.3 ms | 24.7 ms |
| Memory-update frame | 23.3 ms | 27.0 ms |

Inference fits inside the 33 ms frame budget of a 30 fps camera. MatAnyone2 is split into six TensorRT engines. The biggest win came from replacing top-k memory attention with full softmax: memory reads went from 10.2 ms to 5.3 ms, and alpha changed by a mean of 0.00013. See [`docs/profile_optimization.md`](docs/profile_optimization.md).

## Requirements

- Windows with an NVIDIA RTX 30 or 40 series GPU. The bundled CUDA kernels are compiled for `sm_86`, which runs on both.
- Rust, Visual Studio C++ build tools, and LLVM (libclang, for `bindgen`).
- CUDA 13.1 and TensorRT 10.16. Set `CUDA_ROOT` and `TENSORRT_ROOT` if they aren't at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1` and `C:\Tools\TensorRT-10.16.1.11`.
- [uv](https://docs.astral.sh/uv/) for exporting the engines (Python 3.10, PyTorch with CUDA 12.8).
- [Unity Video Capture](https://github.com/schellingb/UnityCapture), registered as a camera device.

## Build

Export the engines. The first step downloads the MatAnyone2 checkpoint.

```powershell
uv sync
uv run python scripts/export_submodules.py
uv run python scripts/build_submodule_engines.py
uv run python scripts/profile_variants.py
uv run python scripts/profile_full_read.py
uv run python scripts/profile_mask_shallow.py
uv run python scripts/prepare_faithful_engines.py   # -> engines/faithful/
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
.\bin\matanyone_core_smoke.exe --synthetic engines\faithful   # core test, no camera
.\bin\matanyone_runner_bench.exe engines\faithful
.\scripts\check_bench_parity.ps1                               # latency within ±2% of the baselines
```

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

# Standalone App Architecture

Date: 2026-05-21

## Goal

Standalone real-time matting app with virtual camera output (no OBS dependency):

1. Capture webcam via Media Foundation (Rust/nokhwa).
2. Initialize target from mask PNG, center placeholder, or SAM3.1 clicks.
3. Run MatAnyone TensorRT pipeline via C++ core (`core/`).
4. Preview composited output in egui UI.
5. Publish output through UnityCapture / virtualcam (non-OBS virtual camera).

## Architecture

```text
Webcam (nokhwa/MSMF)
  -> Rust app worker thread
  -> matanyone_core (C API / TensorRT + CUDA kernels)
  -> composited BGR frame
  -> preview + virtualcam output
```

Key paths:

| Component | Path |
|-----------|------|
| C++ core + C API | `core/` |
| Rust app | `app/` |
| Engine export scripts | `scripts/` |
| TensorRT engines | `engines/faithful/` |

## Build

```powershell
.\build.ps1
```

Portable bundle (executable + runtime DLLs + engines):

```powershell
.\build.ps1 -Bundle
# optional zip: .\build.ps1 -Bundle -Zip
```

Output folder: `dist/matanyone-vcam-win64/` — can be copied to another machine. The app loads `engines/faithful` relative to `matanyone-vcam.exe`; runtime DLLs sit in the same folder.

Manual steps:

```powershell
cmake -S core -B build/core -G "Visual Studio 17 2022" -A x64
cmake --build build/core --config Release
cargo build --release --manifest-path app/Cargo.toml
.\bundle.ps1 -SkipBuild
```

Run (development):

```powershell
.\run_vcam.ps1
```

Run (bundled folder):

```powershell
.\dist\matanyone-vcam-win64\matanyone-vcam.exe
```

Core smoke test (no camera):

```powershell
.\build\core\Release\matanyone_core_smoke.exe --synthetic engines\faithful
```

## Virtual Camera

The Rust app uses the [`virtualcam`](https://crates.io/crates/virtualcam) crate with the UnityCapture backend when available. This does **not** require OBS Studio.

Unity Video Capture must be registered on the system (typically via the UnityCapture installer). If UnityCapture is unavailable, the crate falls back to its default backend.

## SAM3.1 First-Frame Mask

Enable **SAM3.1 click mode** in the app settings, click the subject in the preview, then press **Start**. The app shells out to `scripts/sam31_mask_worker.py` using the project Python environment.

Setup is unchanged from the prior standalone GUI path — see `docs/standalone_architecture.md` history in git for SAM install commands.

## Python Tooling

The Python/PySide6 standalone GUI (`scripts/standalone_gui.py`) remains available for development and benchmarking, but the shipped product is the Rust app.

## Performance Notes

- TensorRT inference: ~24 ms/frame (C++ core)
- App loop adds capture, BGRA conversion, readback, preview, and virtual-camera pacing
- Use `scripts/bench_standalone_no_ui.py` for Python pipeline benchmarks during engine development

# MatAnyone Virtual Camera — Project Plan

**Goal**: Real-time AI video matting (subject extraction) as a standalone Windows app with virtual camera output, running on the local RTX 3080.

**Primary deliverable**: Rust desktop app (`app/matanyone-vcam`) backed by C++ TensorRT core (`core/`).

**Non-goals**: OBS plugin, cross-GPU support, 4K, multi-person tracking, Linux/Mac.

**Target**: MatAnyone2, 720×1280, 30fps, ~30ms/frame end-to-end.

---

## Architecture

```text
Webcam (MSMF / nokhwa)
  -> Rust worker thread
  -> matanyone_core C API
       BGR -> CUDA preprocess -> MatAnyoneRunner (TensorRT)
       -> alpha composite -> BGR readback
  -> egui preview + virtualcam (UnityCapture)
```

## Build & bundle

```powershell
.\build.ps1              # dev build
.\build.ps1 -Bundle      # portable folder in dist/matanyone-vcam-win64/
.\build.ps1 -Bundle -Zip # same + zip archive
```

The bundle includes `matanyone-vcam.exe`, TensorRT/CUDA runtime DLLs, and `engines/faithful/`. Paths are resolved relative to the executable.

---

## Components

| Layer | Path | Notes |
|-------|------|-------|
| Rust UI + I/O | `app/` | egui, nokhwa, virtualcam, SAM subprocess |
| C++ inference | `core/` | TensorRT runner, CUDA cubin kernels, C API |
| Engine tooling | `scripts/` | Export, bench, Python reference GUI |
| Engines | `engines/faithful/` | FP16 TensorRT engines |

---

## Status

- [x] C++ core extracted from OBS plugin with stable C API
- [x] Core smoke test (`matanyone_core_smoke`)
- [x] Rust app with capture, inference, preview, virtual camera
- [x] SAM3.1 click-to-mask via Python worker subprocess
- [x] OBS plugin code removed

---

## Historical Note

This repo previously targeted an OBS Studio filter plugin. That path has been retired in favor of the standalone virtual camera app above. Python standalone GUI scripts remain for development.

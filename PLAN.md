# MatAnyone Virtual Camera — Project Plan

**Goal**: Real-time AI video matting (subject extraction) as a standalone Windows app with virtual camera output, running on the local RTX 3080.

**Primary deliverable**: Rust workspace — `matanyone-vcam` app + `matanyone-core` crate (TensorRT/CUDA inference).

**Non-goals**: OBS plugin, cross-GPU support, 4K, multi-person tracking, Linux/Mac.

**Target**: MatAnyone2, 720×1280, 30fps, ~30ms/frame end-to-end.

---

## Architecture

```text
Webcam (MSMF / nokhwa)
  -> Rust worker thread
  -> matanyone_core::Session
       BGR -> CUDA cubin preprocess -> MatAnyoneRunner (TensorRT)
       -> alpha composite -> BGR readback
  -> egui preview + virtualcam (UnityCapture)
```

## Build & bundle

```powershell
.\build.ps1              # cargo build --release; copies exe + TensorRT/CUDA DLLs to repo root
.\build.ps1 -Bundle      # portable folder in dist/matanyone-vcam-win64/
.\build.ps1 -Bundle -Zip # same + zip archive
```

After build, run from the repo root (no PATH setup needed — runtime DLLs sit beside the exe):

```powershell
.\matanyone-vcam.exe
# or
.\run_vcam.ps1           # optional; sets PATH if you prefer
```

Dev/bench binaries land in `bin/` with the same runtime DLLs (also under `target/release/`). The bundle includes `matanyone-vcam.exe`, TensorRT/CUDA runtime DLLs, and `engines/faithful/`. Paths are resolved relative to the executable.

Benchmark parity (RTX 3080 baselines in `docs/benchmark_baselines.json`):

```powershell
.\scripts\check_bench_parity.ps1
```

---

## Components

| Layer | Path | Notes |
|-------|------|-------|
| User-facing exe | `./matanyone-vcam.exe` | Copied to repo root by `build.ps1` |
| Rust UI + I/O | `app/` | egui, nokhwa, virtualcam |
| Rust inference | `crates/matanyone-core/` | TensorRT runner, CUDA cubin, Session API |
| Dev/bench tools | `bin/` | smoke + bench exes (copied from `target/release/`) |
| Engine tooling | `scripts/` | Export, bench, Python reference GUI |
| Engines | `engines/faithful/` | FP16 TensorRT engines |

---

## Status

- [x] Rust `matanyone-core` with TensorRT runner + CUDA kernels (C++ core removed)
- [x] Core smoke test (`.\bin\matanyone_core_smoke.exe --synthetic engines\faithful`)
- [x] Rust app with capture, inference, preview, virtual camera
- [x] SAM3.1 click-to-mask via `sam-python` feature (default); TRT export spike pending
- [x] OBS plugin code removed

---

## Historical Note

This repo previously targeted an OBS Studio filter plugin and a C++ `core/` static library. That path has been retired in favor of the Rust workspace above. Python standalone GUI scripts remain for development.

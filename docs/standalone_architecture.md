# Standalone App Architecture



Date: 2026-05-21



## Goal



Standalone real-time matting app with virtual camera output (no OBS dependency):



1. Capture webcam via Media Foundation (Rust/nokhwa).

2. Initialize target from mask PNG, center placeholder, or SAM3.1 clicks.

3. Run MatAnyone TensorRT pipeline via `matanyone-core` (Rust).

4. Preview composited output in egui UI.

5. Publish output through UnityCapture / virtualcam (non-OBS virtual camera).



## Layout



```text

matanyone-vcam/

  matanyone-vcam.exe      # main app (copied here by build.ps1)
  nvinfer_*.dll, cudart64_*.dll  # TensorRT/CUDA runtime (copied by build.ps1)

  bin/                    # dev tools + runtime DLLs for bench exes

  app/                    # Rust UI source

  crates/matanyone-core/  # inference crate source

  engines/faithful/       # TensorRT engines (loaded relative to exe)

  scripts/                # export + Python reference GUI

  docs/

  build.ps1, bundle.ps1, run_vcam.ps1

  target/                 # cargo output (gitignored)

```



Key paths:



| Component | Path |

|-----------|------|

| User-facing exe | `./matanyone-vcam.exe` |

| Inference crate | `crates/matanyone-core/` |

| Rust app | `app/` |

| Dev/bench tools | `bin/` |

| Engine export scripts | `scripts/` |

| TensorRT engines | `engines/faithful/` |

| Benchmark baselines | `docs/benchmark_baselines.json` |



## Build



```powershell

.\build.ps1

```



Copies `matanyone-vcam.exe` and TensorRT/CUDA runtime DLLs to the repo root; bench/smoke exes and the same DLLs go to `bin/`. Double-click or `.\matanyone-vcam.exe` works without `run_vcam.ps1`.



Portable bundle (executable + runtime DLLs + engines):



```powershell

.\build.ps1 -Bundle

# optional zip: .\build.ps1 -Bundle -Zip

```



Output folder: `dist/matanyone-vcam-win64/` — can be copied to another machine. The app loads `engines/faithful` relative to `matanyone-vcam.exe`; runtime DLLs sit in the same folder.



Benchmark harness (optional, in `bin/` after build):



```powershell

.\bin\matanyone_runner_bench.exe engines\faithful

.\scripts\check_bench_parity.ps1

```



Run (development, from repo root):



```powershell

.\matanyone-vcam.exe

# or

.\run_vcam.ps1

```



Run (bundled folder):



```powershell

.\dist\matanyone-vcam-win64\matanyone-vcam.exe

```



Core smoke test (no camera):



```powershell

.\bin\matanyone_core_smoke.exe --synthetic engines\faithful

```



## Virtual Camera



The Rust app uses the [`virtualcam`](https://crates.io/crates/virtualcam) crate with the UnityCapture backend when available. This does **not** require OBS Studio.



Unity Video Capture must be registered on the system (typically via the UnityCapture installer). If UnityCapture is unavailable, the crate falls back to its default backend.



## SAM3.1



Click-to-mask uses `Sam31Predictor` in `matanyone-core`. Default build enables `sam-python` (Python worker subprocess). In-process TRT inference requires exported engines under `engines/sam31/` via `scripts/export_sam31_engines.py` (export spike; see script for blockers).



## Accuracy gates



- Latency: `scripts/check_bench_parity.ps1` (±2% vs `docs/benchmark_baselines.json`)

- Alpha: `scripts/measure_fidelity.py` (runner alpha vs the original PyTorch `InferenceCore` on the same frames; 0.00019 mean abs diff on the 96-frame sample)

- SAM: `scripts/measure_sam_parity.py` (when TRT SAM export completes)


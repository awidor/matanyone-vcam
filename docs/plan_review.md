# PLAN.md Review

Date: 2026-05-08

## Verdict

The plan is directionally sound as an R&D plan, but it should be treated as gated research rather than a guaranteed implementation path.

The largest risks are:

- License: MatAnyone2 is non-commercial only unless written permission is obtained.
- Exportability: the memory manager and attention/consolidation logic are likely the central engineering risk.
- Environment: this machine is not ready for Phase 1 yet.
- Performance: 720p 30fps may be feasible on an RTX 3080, but the plan's latency budget is unproven until the encoder and full split graph are benchmarked.

## Phase 0 Result

Completed. See `docs/license_audit.md`.

The project can proceed for local, non-commercial prototyping. It should pivot before commercial or income-generating use.

## Local Environment Check

Observed on 2026-05-08:

- GPU: NVIDIA GeForce RTX 3080, 10 GiB VRAM.
- Driver: 596.36.
- `nvidia-smi` reports CUDA Version 13.2. This is the maximum CUDA runtime supported by the installed driver, not the installed toolkit version.
- Installed CUDA toolkit found on PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe`.
- Repo-local uv environment: `.venv`, CPython 3.10.19.
- PyTorch: 2.11.0+cu128, CUDA available, device is NVIDIA GeForce RTX 3080.
- ONNX: 1.21.0.
- ONNX Runtime GPU: 1.23.2, with TensorRT, CUDA, and CPU providers available.
- TensorRT Python: 10.16.1.11.
- Polygraphy: 0.49.26.
- MatAnyone2 source clone: `vendor/MatAnyone2` at commit `e3370127319c63a6dc8a49c69de2d41d90137f91`.
- `trtexec`: not found on PATH. It is not shipped by the TensorRT pip wheel.
- System `python`: 3.13.12. Use `.venv\Scripts\python.exe` for this project.
- `superpowers`: not found on PATH.

## Required Fixes Before Phase 1

1. Install full TensorRT SDK for Windows, or otherwise provide `trtexec.exe`, and put it on PATH.
2. Decide whether to keep the now-installed CUDA 12.8/13.x-compatible stack, or force the original CUDA 12.4 stack.
3. If keeping this stack, update the plan's software table to match the verified environment.

## Plan Corrections

- Reword "CUDA driver 13.2" to "NVIDIA driver 596.36 reports CUDA runtime compatibility up to 13.2." CUDA 13.2 is not itself the driver version.
- Add an explicit commercial-use decision before Phase 1. If the goal includes monetized streaming or paid distribution, pivot now.
- Add a first deliverable before export: clone upstream MatAnyone2 and freeze the exact commit hash. The architecture/export plan should be tied to that commit.
- Add a small baseline run of upstream Python inference before the encoder export. This confirms the checkpoint, dependencies, sample input, and GPU work before surgery starts.
- Add an environment lock file after Phase 1 succeeds so later C++/TensorRT work is reproducible.

## Execution Status

Phase 1 encoder spike has been executed with TensorRT Python instead of `trtexec`. See `docs/phase1_encoder_spike.md`.

The standalone `trtexec.exe` CLI is still not installed, but it is no longer blocking the Phase 1 result.

Phase 2 submodule export has been executed for the single-object, no-long-term-memory path. See `docs/phase2_submodule_export.md`.

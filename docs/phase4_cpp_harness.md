# Phase 4 C++ TensorRT Harness

Date: 2026-05-08

## Result

The native C++ TensorRT benchmark harness builds and runs.

Added:

- `CMakeLists.txt`
- `cpp/matanyone_trt_bench.cpp`

Configure:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
```

Build:

```powershell
cmake --build build --config Release
```

Run:

```powershell
$env:PATH='C:\Tools\TensorRT-10.16.1.11\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;' + $env:PATH
build\Release\matanyone_trt_bench.exe engines/faithful
```

Output:

```text
normal_mean_ms=21.2783
normal_p99_ms=24.6561
memory_update_mean_ms=23.2759
memory_update_p99_ms=27.0284
```

## Scope

This is a native engine execution harness, not yet a complete C++ `MatAnyoneRunner`.

It validates:

- TensorRT C++ SDK linkage
- CUDA runtime linkage
- Engine deserialization
- Static device buffer allocation
- Sequential engine execution timing

Still needed for the complete Phase 4 runner:

- Real input frame upload/interop
- Alpha output retrieval or GPU blend target
- Parity check against the Python TRT harness

## MatAnyoneRunner

Added a native runner skeleton with shared semantic buffers and memory bank update logic:

- `cpp/trt_common.h`
- `cpp/trt_common.cpp`
- `cpp/matanyone_runner.h`
- `cpp/matanyone_runner.cpp`
- `cpp/matanyone_runner_bench.cpp`

Build:

```powershell
cmake --build build --config Release
```

Run:

```powershell
$env:PATH='C:\Tools\TensorRT-10.16.1.11\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;' + $env:PATH
build\Release\matanyone_runner_bench.exe engines/faithful
```

Output:

```text
mixed_mean_ms=21.0462
mixed_p99_ms=23.22
```

This runner owns shared CUDA buffers, binds them across engines, runs the encode/read/segment/mask sequence, and updates a fixed 5-frame memory bank on device. It is still using synthetic input buffers; OBS/DX11 interop comes later.

## Raw Frame Harness

Added a dependency-free real-frame path using raw float files:

- `scripts/export_raw_sample.py`
- `cpp/matanyone_runner_raw.cpp`

Prepare sample frames:

```powershell
uv run python scripts\export_raw_sample.py --frames 60
```

Run native raw-frame harness:

```powershell
$env:PATH='C:\Tools\TensorRT-10.16.1.11\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;' + $env:PATH
build\Release\matanyone_runner_raw.exe raw_sample engines/faithful
```

Output:

```text
frames=60
mean_ms=24.7373
p99_ms=37.9331
wrote=output/raw_alpha
```

This exercises real host RGB frame input, first-frame mask initialization, native runner inference, alpha readback, and raw alpha output files without adding an OpenCV C++ dependency.

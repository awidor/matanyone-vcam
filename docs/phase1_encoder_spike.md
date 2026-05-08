# Phase 1 Encoder Export Spike

Date: 2026-05-08

## Result

Phase 1 encoder spike passed.

- MatAnyone2 commit: `e3370127319c63a6dc8a49c69de2d41d90137f91`
- Checkpoint: `models/matanyone2.pth`
- ONNX: `models/encoder.onnx`
- TensorRT engine: `engines/encoder_fp16.engine`
- ONNX opset: 17
- TensorRT: 10.16.1.11 via Python builder
- CUDA PyTorch: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 3080

## Benchmark

Command:

```powershell
uv run python scripts\bench_encoder_engine.py
```

Result at `1x3x720x1280`, 200 warmup iterations, 500 timed iterations:

- Mean: `3.707 ms`
- p99: `5.109 ms`

This is below the Phase 1 success threshold of 10 ms and far below the 15 ms kill threshold.

## Export Commands

```powershell
uv run python scripts\export_encoder.py
uv run python scripts\build_encoder_engine.py
uv run python scripts\bench_encoder_engine.py
```

## Notes

The plan originally referenced `trtexec`. This workspace does not have `trtexec.exe`, and the TensorRT pip wheel does not ship it. The engine was built with the TensorRT Python API instead.

The dynamic profile used by `scripts/build_encoder_engine.py` is:

- min: `1x3x270x480`
- opt: `1x3x720x1280`
- max: `1x3x1080x1920`

The plan's min shape was written as `1x3x480x270`, but TensorRT uses NCHW and the opt/max entries are landscape. The script uses the landscape 16:9 interpretation.

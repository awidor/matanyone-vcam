# Phase 2 Submodule Export

Date: 2026-05-08

## Result

The stateless split is viable for the single-object, no-long-term-memory path.

Completed:

- `scripts/matanyone2_stateless.py`
- `scripts/export_submodules.py`
- `scripts/build_submodule_engines.py`
- `scripts/bench_submodule_engines.py`
- `scripts/validate_stateless_parity.py`

Exported ONNX:

- `models/submodules/encode_image.onnx`
- `models/submodules/read_memory.onnx`
- `models/submodules/pixel_fusion.onnx`
- `models/submodules/first_frame_read_memory.onnx`
- `models/submodules/segment.onnx`
- `models/submodules/encode_mask.onnx`

Built TensorRT FP16 engines:

- `engines/submodules/encode_image_fp16.engine`
- `engines/submodules/read_memory_fp16.engine`
- `engines/submodules/pixel_fusion_fp16.engine`
- `engines/submodules/first_frame_read_memory_fp16.engine`
- `engines/submodules/segment_fp16.engine`
- `engines/submodules/encode_mask_fp16.engine`

## Parity

Short parity passed exactly on the bundled sample:

```powershell
uv run python scripts\validate_stateless_parity.py --frames 8
```

Output:

```text
frame=0 mean_abs_diff=0.000000
frame=1 mean_abs_diff=0.000000
frame=2 mean_abs_diff=0.000000
frame=3 mean_abs_diff=0.000000
frame=4 mean_abs_diff=0.000000
frame=5 mean_abs_diff=0.000000
frame=6 mean_abs_diff=0.000000
frame=7 mean_abs_diff=0.000000
overall_mean_abs_diff=0.000000
```

The 100-frame PyTorch parity run is intentionally deferred because it runs both the original and stateless PyTorch paths and takes too long for interactive work.

## Fast TensorRT Benchmark

Command:

```powershell
uv run python scripts\bench_submodule_engines.py --warmup 20 --iters 50
```

Results:

| Engine | Mean ms | p99 ms |
|--------|---------|--------|
| encode_image | 4.138 | 5.077 |
| encode_mask | 2.284 | 3.202 |
| first_frame_read_memory | 2.965 | 3.828 |
| pixel_fusion | 0.449 | 0.793 |
| read_memory | 12.744 | 14.158 |
| segment | 8.410 | 10.259 |

Naive summed mean for all engines is `30.991 ms`. Normal non-memory frames should skip `encode_mask`, but still need memory read/fusion/segment and therefore are near the 30 fps boundary.

## Main Bottleneck

`read_memory_fp16.engine` is the current bottleneck at `12.744 ms` mean. It includes top-k attention over 5 memory frames at 45x80 tokens.

See `docs/profile_optimization.md` for the optimized profile. The key finding is that TensorRT full softmax is faster than TensorRT top-k for the current 5-frame 720p memory shape, with negligible short-sample output difference.

Next optimization targets:

- Fuse `read_memory` and `pixel_fusion`.
- Use alpha-only segment outputs.
- Benchmark full end-to-end TensorRT execution with reused buffers.
- Consider custom CUDA for attention only if full softmax quality fails on harder clips.

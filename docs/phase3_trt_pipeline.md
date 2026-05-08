# Phase 3 TensorRT Pipeline Harness

Date: 2026-05-08

## Result

Added a fixed-shape, fixed-buffer TensorRT pipeline benchmark:

```powershell
uv run python scripts\bench_trt_pipeline.py --warmup 50 --iters 200
```

The harness reuses GPU tensors and runs the optimized engines in sequence on one CUDA stream:

1. `encode_image`
2. `read_memory` with full softmax, 5 memory frames
3. `pixel_fusion`
4. `segment` alpha-only
5. optional `encode_mask` on memory-update frames

## Pipeline Timing

Representative chained timing:

```text
normal_mean_ms=17.614
normal_p99_ms=20.804
memory_update_mean_ms=21.628
memory_update_p99_ms=23.860
```

Stage-timed run:

```text
normal_mean_ms=19.079
normal_p99_ms=21.411
memory_update_mean_ms=21.502
memory_update_p99_ms=23.415
encode_mean_ms=4.234
read_mean_ms=6.234
fuse_mean_ms=0.455
segment_mean_ms=8.294
encode_mask_mean_ms=2.300
```

The chain overhead is effectively zero; the cost is in the engines themselves.

## CUDA Graph

Normal-frame CUDA graph replay:

```powershell
uv run python scripts\bench_trt_pipeline.py --warmup 30 --iters 200 --cuda-graph
```

Result:

```text
normal_mean_ms=17.040
normal_p99_ms=18.364
```

CUDA graph capture improves jitter and modestly improves mean time. This is relevant for the eventual C++ runner because all shapes and buffers are static.

## Current Budget

At 720p, the optimized path is viable for 30 fps before OBS integration:

- Normal frame: roughly `17-19 ms`
- Memory-update frame: roughly `21-24 ms`

Remaining OBS work must fit in about `9-12 ms` for normal frames and `9 ms` for update frames to stay under `33.3 ms`.

## Next Optimization Target

Do not lower resolution yet.

The next useful optimization is fusing `read_memory + pixel_fusion` or reducing segment cost. Launch overhead is not the limiting factor.

## Faithful Pipeline Correction

The initial optimized pipeline benchmark omitted the object transformer after pixel fusion. That made it a useful speed probe but not a faithful inference path.

The corrected full-read engine includes:

1. memory similarity/readout
2. uncertainty prediction and blend
3. pixel fusion
4. object transformer readout

Corrected synthetic timing:

```text
normal_mean_ms=21.790
normal_p99_ms=23.797
memory_update_mean_ms=22.551
memory_update_p99_ms=24.409
```

The normal frame includes shallow `encode_mask` to update `last_msk_value`, matching the original inference logic.

## Video Harness

Added:

```powershell
uv run python scripts\inference_trt.py --frames 60 --output output\trt_alpha_60.mp4
```

Result:

```text
frames=60
mean_ms=23.104
p99_ms=39.825
steady_mean_ms=22.354
steady_p99_ms=23.907
wrote=output\trt_alpha_60.mp4
```

The full TRT video harness processes the bundled sample and writes an alpha video. The high total-run p99 comes from startup/cold frames; steady-state timing is within the 30 fps budget.

Native C++ engine execution is now covered in `docs/phase4_cpp_harness.md`.

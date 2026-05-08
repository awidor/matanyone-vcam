# Profiling And Optimization Notes

Date: 2026-05-08

## Constraint

Resolution stays at 720p. Resolution reduction is a last resort.

## Baseline

Initial static TensorRT submodule benchmark:

| Component | Mean ms | p99 ms |
|-----------|---------|--------|
| encode_image | 4.138 | 5.077 |
| read_memory, top-k 30, 5 frames | 12.744 | 14.158 |
| pixel_fusion | 0.449 | 0.793 |
| segment, full outputs | 8.410 | 10.259 |
| encode_mask | 2.284 | 3.202 |

Normal non-memory frame estimate:

```text
encode_image + read_memory + pixel_fusion + segment = 25.741 ms
```

Memory-update frame estimate:

```text
normal frame + encode_mask = 28.025 ms
```

## Read Memory Profiling

At 720p, memory features are `45x80`, so one memory frame is 3600 tokens.

The original `top_k=30` path is slower in TensorRT than full softmax at 5 memory frames.

| Variant | Mean ms | p99 ms |
|---------|---------|--------|
| T=5, top-k 30 | 10.204 | 11.228 |
| T=5, top-k 20 | 10.686 | 11.906 |
| T=5, top-k 10 | 9.276 | 10.324 |
| T=5, full softmax | 5.341 | 6.259 |

The decomposed T=5 profile:

| Stage | Mean ms | p99 ms |
|-------|---------|--------|
| similarity | 1.446 | 1.956 |
| top-k softmax | 4.098 | 5.050 |
| value readout | 1.016 | 1.432 |

Conclusion: TensorRT top-k/scatter is the bad fit. Full softmax is faster and much simpler.

## Quality Check

Full softmax was compared against the original top-k 30 path using the bundled sample:

```powershell
uv run python scripts\validate_stateless_parity.py --frames 20 --stateless-full-softmax --tolerance 0.005
```

Result:

```text
overall_mean_abs_diff=0.000132
```

This is far below the existing parity tolerance of `0.005`.

## Segment Profiling

The original segment export returned both foreground/background probability channels. OBS only needs alpha.

| Variant | Mean ms | p99 ms |
|---------|---------|--------|
| segment, full outputs | 8.410 | 10.259 |
| segment, alpha only | 6.989 | 7.712 |

Conclusion: alpha-only segment output is a clean win.

## Optimized Engine Set

Created by:

```powershell
uv run python scripts\prepare_optimized_engines.py
```

Benchmarked by:

```powershell
uv run python scripts\bench_submodule_engines.py --engine-dir engines\optimized --warmup 20 --iters 50
```

Optimized results:

| Component | Mean ms | p99 ms |
|-----------|---------|--------|
| encode_image | 3.659 | 4.406 |
| read_memory, full softmax, 5 frames | 5.307 | 5.790 |
| pixel_fusion | 0.399 | 0.631 |
| segment, alpha only | 6.989 | 7.712 |
| encode_mask | 2.008 | 2.413 |
| first_frame_read_memory | 2.591 | 3.148 |

Normal non-memory frame estimate:

```text
encode_image + read_memory + pixel_fusion + segment = 16.354 ms
```

Memory-update frame estimate:

```text
normal frame + encode_mask = 18.362 ms
```

These numbers leave enough room for DX11/CUDA interop, resize, and blend work while staying inside 30 fps.

## Next Optimizations

Do before lowering resolution:

- Fuse `read_memory` and `pixel_fusion` into one engine or CUDA path to avoid an extra launch and intermediate tensor.
- Export a segment variant that does not take unused `f16`/`f1` inputs if the ONNX exporter keeps them as dead inputs.
- Benchmark end-to-end TensorRT execution with reused buffers and one CUDA stream.
- Consider custom CUDA for memory attention only if full softmax quality fails on harder clips.

End-to-end TensorRT pipeline profiling is now in `docs/phase3_trt_pipeline.md`.

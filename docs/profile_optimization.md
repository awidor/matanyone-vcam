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

## 2026-10-07: Runner fidelity, memory read layout, CUDA graphs

RTX 3080, TensorRT 10.16.1.11, CUDA 13.2. Latency comparisons are interleaved runs on the same machine.

### Runner fidelity

The Rust runner, like the C++ runner before it, refreshed per-frame state only on memory frames. `InferenceCore.step` does this on every frame:

- `sensory` takes the segment decoder's update (`stagger_updates` 5 >= `mem_every` 5, so every frame).
- `last_mask` and `last_pix_feat` take the current frame.
- `last_msk_value` comes from the shallow mask encoder on non-memory frames.

The object memory is a running sum (`MemoryManager.add_memory`), not the latest summary. Sensory also starts at zero for a new target; the runner never cleared it.

`scripts/measure_fidelity.py` compares runner alpha against the original PyTorch `InferenceCore` on identical frames:

| Clip | Runner | Mean abs diff | Worst frame | Min IoU |
|------|--------|---------------|-------------|---------|
| test-sample2, 96 frames | before | 0.00128 | 0.01940 | 0.876 |
| test-sample2, 96 frames | after | 0.00019 | 0.00068 | 0.996 |
| test-sample1, 30 frames | before | 0.00053 | 0.00175 | 0.994 |
| test-sample1, 30 frames | after | 0.00019 | 0.00050 | 0.998 |

The shallow mask encoder adds 1.5 ms on non-memory frames, but it runs after the alpha is read back (see CUDA graphs), so it does not add latency.

### Memory read

A detailed TensorRT profile of the 10.0 ms read engine:

| Kernel | Mean ms |
|--------|---------|
| `__myl_NegMulAddSubReshMulMulMaxrSubExpSumDivMulMove` (similarity epilogue + softmax) | 5.28 |
| Three similarity/readout MatMuls | 1.00 |
| `/object_transformer/NonZero[size][DevicetoShapeHostCopy]` and other NonZero | ~0.3 |

The softmax reduced along the memory axis of an `N x HW` (18000 x 3600) matrix, which is the strided axis. The `NonZero` ops come from `aux_mask[torch.where(...)] = False` in `QueryTransformer._get_aux_mask`. They force a device-to-host shape copy on every enqueue and make the engine impossible to capture in a CUDA graph.

`MemoryReadoutQueryMajor` (in `matanyone2_stateless.py`) computes the same thing as plain attention. `[qe, 2 qk qe, -sum(qe qk^2)]` dotted with `[-mk^2, mk, 1] * ms / sqrt(CK)` gives the whole anisotropic L2 similarity in one GEMM (K = 129, padded to 136), followed by a softmax over the contiguous memory axis and `value @ affinity^T`. `use_static_aux_mask()` swaps in `aux_mask & ~aux_mask.all(-1, keepdim=True)`, which has the same semantics without `NonZero`. In FP32 PyTorch the readout differs by at most 7e-4 (values up to 13.5) and alpha by 3e-7 on average.

| Read engine | Mean ms |
|-------------|---------|
| Original | 10.70 |
| Query-major, static mask | 4.55 |
| Query-major, static mask, CUDA graph | 3.38 |

### CUDA graphs and the session

- Each frame is two captured graphs. The head (encode, read, segment) produces the alpha; the tail (sensory and last-frame copies, shallow or deep mask encoder, object memory sum) prepares the next frame. `Session` composites and reads back after the head and returns while the tail runs.
- Frames upload as packed BGR through a pinned buffer, and the preprocess kernel writes straight into the encoder input. The composite reads the runner's alpha and writes BGR. This removes the CPU BGR/BGRA conversions, two device copies and the per-frame allocations.
- `f16`..`f1` cross the encode/segment boundary as FP16. Interleaved trtexec medians: encode 3.82 -> 3.46 ms, segment 7.60 -> 7.36 ms.
- Tried without a reliable gain: builder optimization level 5 and `TilingOptimizationLevel.FULL` on the segment engine.

### Results

| Measurement | Before mean / p99 | After mean / p99 |
|-------------|-------------------|------------------|
| `matanyone_runner_bench`, mixed frames | 22.1-22.6 / 25.2-26.6 ms | 15.9-16.1 / 17.4-17.8 ms |
| `Session::process_bgr`, back to back | 26.2-26.9 / 30.1-30.6 ms | 16.2-16.4 / 17.9-18.3 ms |
| `Session::process_bgr`, paced at 30 fps | 26.2-27.1 / 30.6-31.6 ms | 15.1-15.2 / 16.4-17.2 ms |

The "after" runner also does the per-frame work listed under Runner fidelity.

### Next

- Segment (7.4 ms) is now the largest stage. About 2.6 ms of it is high-resolution pointwise work that TensorRT leaves unfused, mostly in `up_2_1` and `up_4_2`: pre-activation ResBlocks where the upsample-plus-skip sum feeds both a ReLU and the 1x1 shortcut conv. Moving the shortcut conv before the bilinear upsample (they commute) and folding it into the skip projection is exact, would let the add and ReLU fuse, and would save an estimated 0.8 ms.
- The bank seeds every slot with the first frame and keeps the first frame plus 4 recent memory frames. The original grows from 1 frame and keeps the first plus 5. Matching it needs a masked T=6 read (about +0.6 ms), which is not worth it at the current 0.0002 mean difference.

# MatAnyone OBS Plugin — MVP Plan

**Goal**: Real-time AI video matting (subject extraction) as an OBS filter plugin, running on the local RTX 3080.  
**Non-goals**: Distribution, cross-GPU support, 4K, multi-person tracking, Linux/Mac.  
**Target**: MatAnyone2 (identical inference architecture to v1, better quality), 720p, 30fps, <25ms/frame.

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | RTX 3080 10GB (Ampere SM 8.6) |
| VRAM available | ~8.5GB (1.7GB used by desktop) |
| CPU | Ryzen 7 5700X (8c/16t) |
| RAM | 32GB |
| CUDA driver | 13.2 (supports CUDA runtime up to 12.x) |

---

## Architecture

```
OBS Frame (DX11 texture)
    ↓ DX11/CUDA interop — stays on GPU, no CPU bounce
CUDA RGB tensor (720p)
    ↓
[MediaPipe selfie segmenter]  ← runs on first frame + re-init triggers
    → initial mask (720p binary)
    ↓
[MatAnyone Core — TensorRT FP16 engines]
    encode_image()     → pixel features + keys
    read_memory()      → memory readout      ← memory bank tensors (C++ managed)
    pixel_fusion()     → fused features
    segment()          → alpha matte (0–1)
    [if mem_frame] encode_mask() → memory bank update
    ↓
Alpha matte (720p)
    ↓ CUDA blend shader (subject over virtual background)
Composited DX11 texture
    ↓
OBS output
```

Memory bank state lives in C++ as raw CUDA tensors between frames — no Python, no dynamic Python logic.

---

## Model Architecture (confirmed from source)

```yaml
PixelEncoder:  ResNet-50  (ms_dims [1024,512,256,64,3])  ~25.6M params
MaskEncoder:   ResNet-18  (final_dim 256)                 ~11.7M params
Transformer:   3 blocks, embed_dim 256, ff_dim 2048,
               8 heads, 16 queries                        ~5-6M params
Decoder:       Conv upsampler, up_dims [256,128,128,64,16] ~3-4M params
Total:         ~46-48M params, ~190MB FP32 / ~95MB FP16
```

MatAnyone1 and MatAnyone2 have identical inference architectures. v2 is better quality via training data improvements. Use v2.

---

## Phases

### Phase 0 — License Audit (1 day) `GATE`

**Deliverable**: Written decision on whether the project is shippable.

MatAnyone is licensed under NTU S-Lab License 1.0. Read the actual license file. Determine:
- Is streaming use (even privately) permitted?
- Is distributing a compiled binary that embeds the model weights permitted?
- Is commercial use (streaming for income) permitted?

**Kill criterion**: If the license prohibits the intended use, pivot to RVM (Robust Video Matting, Apache 2.0) or RMBG. The OBS plugin architecture is identical — only the model changes.

---

### Phase 1 — Encoder Export Spike (3–5 days) `GATE`

**Goal**: Get just the PixelEncoder (ResNet-50) through TensorRT FP16. Measure real latency on this machine.

**Steps**:
1. Install: PyTorch 2.x + CUDA 12.x, TensorRT 10.x, `trtexec`, onnx, onnxruntime-gpu
2. Clone MatAnyone2 repo, install dependencies
3. Write `scripts/export_encoder.py`:
   - Load checkpoint, extract `model.pixel_encoder`
   - Run one forward pass at 720p (1280×720), print all intermediate tensor shapes — **save this output as `docs/tensor_shapes.md`, it's the spec for the C++ memory bank**
   - Export encoder to ONNX: `torch.onnx.export()` with dynamic batch+spatial dims
   - Compile to TensorRT engine via `trtexec --onnx=encoder.onnx --fp16 --minShapes=... --optShapes=... --maxShapes=...`
4. Write `scripts/bench_encoder.py`: run 200 warmup + 500 timed forward passes, report mean/p99 latency

**Dynamic shape ranges** (decide at build time, not runtime):
- min: `1×3×480×270`
- opt: `1×3×720×1280`  (our target)
- max: `1×3×1080×1920`

**Success metric**: ≤10ms per encoder forward pass at 720p with TensorRT FP16.  
**Kill criterion**: >15ms on encoder alone, or ONNX export fails with unfixable ops. Investigate before killing — may just need opset version bump or custom op registration.

---

### Phase 2 — Model Surgery (1–2 weeks) `HIGH RISK`

**Goal**: Refactor MatAnyone2 so every submodule takes memory tensors as explicit inputs/outputs. No hidden Python state.

This is the hardest phase. The memory bank (`kv_memory_store.py`, `memory_manager.py`, `inference_core.py`) is deeply integrated Python logic. It must be surgically separated from the neural net compute kernels.

**Steps**:
1. Study `inference_core.py` → `step()` method. Map every tensor that flows between steps.
2. Create `matanyone2_stateless.py` — a refactored inference class where each neural op is a pure function:
   ```python
   # Input: frame + explicit state tensors
   # Output: alpha + updated state tensors
   # No hidden self.* state between calls
   def step(rgb, mem_keys, mem_values, sensory, last_mask, last_pix_feat):
       ...
       return alpha, new_mem_keys, new_mem_values, new_sensory, new_last_mask, new_last_pix_feat
   ```
3. Write `scripts/validate_parity.py`: run original and refactored versions on the same 100-frame test video, assert alpha outputs match within tolerance (e.g., mean absolute difference < 0.005).
4. Export each submodule to ONNX individually:
   - `encode_image.onnx` (ResNet-50 + key projection)
   - `read_memory.onnx` (attention readout — takes memory tensors as inputs)
   - `pixel_fusion.onnx`
   - `segment.onnx` (decoder)
   - `encode_mask.onnx` (ResNet-18 — runs only on memory-update frames)
5. Compile each to a TensorRT FP16 `.engine` file.

**Note on top-k and consolidation**: The top-k attention selection and memory consolidation logic lives in Python and involves Python control flow (conditionals, sorting). This does NOT get exported. It gets reimplemented in C++ in Phase 4. The ONNX submodules only export the pure tensor compute — the attention dot-products, not the selection logic.

**Success metric**: `validate_parity.py` passes. All 5 submodule engines built.  
**Kill criterion**: Submodule export reveals ops that TensorRT 10 can't handle after fallback attempts. In that case, try exporting to ONNX with opset 17+ or consider LibTorch fallback.

---

### Phase 3 — End-to-End Python Harness (3–5 days)

**Goal**: Full inference pipeline in Python using the TensorRT engines. Proves the split architecture works before touching C++.

**Steps**:
1. Write `scripts/inference_trt.py`:
   - Load all 5 `.engine` files via TensorRT Python runtime
   - Implement memory bank management in Python (same logic as original, but feeding tensors to/from TRT engines explicitly)
   - Process a test video end-to-end
2. Benchmark: FPS at 720p, peak VRAM usage
3. Compare output quality vs original MatAnyone Python inference

**Success metric**: ≥30fps at 720p. Visual quality matches original.

---

### Phase 4 — C++ Inference Harness (1 week)

**Goal**: Standalone C++ executable that processes a video file using TensorRT engines + C++ memory bank.

**Steps**:
1. Set up CMake project with TensorRT C++ runtime, CUDA, OpenCV (for video I/O in tests)
2. Implement `MatAnyoneRunner` class:
   - Load `.engine` files at startup
   - Allocate CUDA memory for memory bank (fixed max size: `max_mem_frames × spatial_tokens × (key_dim + value_dim)`)
   - Implement memory consolidation + top-k attention in CUDA/C++
   - Implement sensory state as a fixed-size CUDA tensor
   - `ProcessFrame(CUdeviceptr rgb_in, CUdeviceptr alpha_out)` — the only public method
3. Test on same video as Phase 3. Diff alpha outputs vs Python harness.

**Key C++ memory bank state** (from `memory_manager.py` analysis):
- `work_mem_keys`: `[max_mem_frames, 1, key_dim, H*W]` CUDA tensor
- `work_mem_values`: `[max_mem_frames, 1, num_objects, value_dim, H*W]` CUDA tensor  
- `work_mem_shrinkage`: `[max_mem_frames, 1, key_dim, H*W]` CUDA tensor
- `sensory`: `[1, sensory_dim, H, W]` CUDA tensor
- `last_mask`, `last_pix_feat`, `last_msk_value`: single-frame CUDA tensors
- `mem_frame_count`: integer

**Success metric**: C++ harness matches Python harness output. ≥30fps.

---

### Phase 5 — First-Frame Segmenter (3–5 days)

**Goal**: Automatic subject detection on stream start — no manual mask required.

**Choice**: MediaPipe selfie segmentation
- Model size: ~3MB
- Latency: ~2ms on GPU
- Output: binary person mask
- Limitation: person-class only (fine for streaming use case)

**Steps**:
1. Integrate MediaPipe C++ library or export MediaPipe model to TensorRT
2. Run on first frame only (or on a slow background timer every N seconds for re-init)
3. Implement re-init policy:
   - Scene cut detected (histogram difference > threshold): re-run segmenter, reset memory bank, run 10 warmup frames (hide these behind a short fade)
   - Subject leaves frame: maintain last valid mask, degrade gracefully
4. Handle warmup: the model needs ~10 frames to establish temporal coherence. During warmup, output the background-only or pass through original frame (decide in UI).

---

### Phase 6 — OBS Plugin Scaffolding (3–5 days)

**Goal**: A working passthrough OBS filter plugin (no AI yet).

**Steps**:
1. Clone [obs-backgroundremoval](https://github.com/royshil/obs-backgroundremoval) as reference
2. Create new CMake project using OBS plugin template
3. Implement `filter_video` callback that reads a DX11 texture and returns it unchanged
4. Test: plugin loads in OBS, appears in filter list, doesn't crash
5. **Critical**: implement DX11/CUDA interop using `cudaGraphicsD3D11RegisterResource` — study how obs-backgroundremoval does this. This keeps frames on GPU and avoids CPU bounce (~5–15ms penalty).

---

### Phase 7 — Integration (1 week)

**Goal**: Wire the C++ inference harness into the OBS filter plugin.

**Steps**:
1. Instantiate `MatAnyoneRunner` when filter is created, destroy when removed
2. In `filter_video`:
   - Get DX11 texture → CUDA via interop (no CPU copy)
   - Resize to 720p via CUDA if needed
   - Run MediaPipe segmenter if first frame or re-init triggered
   - Call `ProcessFrame()`
   - Blend alpha matte with virtual background via CUDA shader
   - Write result back to DX11 texture via interop
3. Add minimal settings panel: background image/color selector, resolution selector (480p/720p/1080p)

---

### Phase 8 — Polish (ongoing)

- Scene cut detection and graceful re-init
- Warmup frame hiding (fade or hold last frame)
- Performance profiling — identify any remaining bottlenecks
- Memory leak testing (24hr+ stream simulation)
- Quality tuning: `max_mem_frames`, `top_k` values

---

## Software Stack

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.5.x + CUDA 12.4 |
| TensorRT | 10.x |
| ONNX opset | 17 |
| OBS Studio | 31.x |
| MSVC | 2022 (17.x) |
| CMake | 3.28+ |
| CUDA Toolkit | 12.4 |
| MediaPipe | 0.10.x |

---

## VRAM Budget (720p, FP16)

| Item | Size |
|------|------|
| TRT engine workspace | ~300MB |
| Model weights FP16 | ~95MB |
| Memory bank (5 frames) | ~50MB |
| Feature maps in flight | ~150MB |
| MediaPipe model | ~5MB |
| Desktop + apps (existing) | ~1.8GB |
| **Total** | **~2.4GB** |
| **Remaining** | **~7.6GB free** |

---

## Open Questions

- [ ] Does the NTU S-Lab license permit this use? (Phase 0 — must answer before proceeding)
- [ ] What is the actual per-frame latency after TRT FP16 compilation? (Phase 1 gate)
- [ ] Do all 5 ONNX submodule exports succeed cleanly, or are custom ops needed? (Phase 2 gate)
- [ ] How to handle warmup frames in OBS — freeze frame, fade, or transparent? (UX decision)
- [ ] At what resolution does quality become unacceptably soft? (needs visual test)
- [ ] Can MediaPipe C++ run on the same CUDA stream as MatAnyone to avoid sync overhead?

---

## Kill Criteria Summary

| Gate | Kill if... | Pivot to |
|------|-----------|---------|
| Phase 0 | License prohibits use | RVM (Apache 2.0) |
| Phase 1 | Encoder >15ms at 720p | Lower target resolution, or RVM |
| Phase 2 | TRT export hits unfixable wall | LibTorch + TorchScript instead of TRT |
| Phase 3 | Quality unacceptably degraded | Investigate FP32 fallback for sensitive submodules |

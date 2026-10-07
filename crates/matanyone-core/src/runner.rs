use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::cuda::{
    memcpy_2d_async, memcpy_async, memset_async, CudaGraph, CudaStream, DevicePtr, MEMCPY_D2D, MEMCPY_D2H,
    MEMCPY_H2D,
};
use crate::kernels::{init_cuda_kernels, launch_accumulate_f32};
use crate::trt::{TrtEngine, TrtRuntime, DTYPE_FLOAT};

/// Engine I/O tensor -> runner buffer. Tensors an engine lacks are skipped, so older
/// engine sets (separate pixel_fusion, no shallow mask encoder) still bind.
type Bindings = &'static [(&'static str, &'static str)];

const ENCODE: Bindings = &[
    ("input_0", "image"),
    ("f16", "f16"),
    ("f8", "f8"),
    ("f4", "f4"),
    ("f2", "f2"),
    ("f1", "f1"),
    ("pix_feat", "pix_feat"),
    ("key", "key"),
    ("shrinkage", "shrinkage"),
    ("selection", "selection"),
];
const READ: Bindings = &[
    ("input_0", "key"),
    ("input_1", "selection"),
    ("input_2", "memory_key"),
    ("input_3", "memory_shrinkage"),
    ("input_4", "memory_value"),
    ("input_5", "last_pix_feat"),
    ("input_6", "pix_feat"),
    ("input_7", "last_mask"),
    ("input_8", "last_msk_value"),
    ("input_9", "sensory"),
    ("input_10", "obj_memory"),
    ("pixel_memory", "pixel_memory"),
    ("memory_readout", "memory_readout"),
];
const PIXEL_FUSION: Bindings = &[
    ("input_0", "pix_feat"),
    ("input_1", "pixel_memory"),
    ("input_2", "sensory"),
    ("input_3", "last_mask"),
    ("fused_pixel", "memory_readout"),
];
const SEGMENT: Bindings = &[
    ("input_0", "f16"),
    ("input_1", "f8"),
    ("input_2", "f4"),
    ("input_3", "f2"),
    ("input_4", "f1"),
    ("input_5", "memory_readout"),
    ("input_6", "sensory"),
    ("new_sensory", "new_sensory"),
    ("alpha", "alpha"),
];
/// Deep update on memory frames: takes the segment's sensory and writes the updated one back.
const ENCODE_MASK: Bindings = &[
    ("input_0", "image"),
    ("input_1", "pix_feat"),
    ("input_2", "new_sensory"),
    ("input_3", "alpha"),
    ("mask_value", "last_msk_value"),
    ("new_sensory", "sensory"),
    ("object_summaries", "object_summaries"),
];
/// Every other frame refreshes only `last_msk_value` (InferenceCore.step, non-memory branch).
const ENCODE_MASK_SHALLOW: Bindings = &[
    ("input_0", "image"),
    ("input_1", "pix_feat"),
    ("input_2", "sensory"),
    ("input_3", "alpha"),
    ("mask_value", "last_msk_value"),
];

struct Stage {
    engine: TrtEngine,
    bindings: Bindings,
}

impl Stage {
    fn load(runtime: &TrtRuntime, path: &Path, bindings: Bindings) -> Result<Self> {
        Ok(Self { engine: TrtEngine::load(runtime, path)?, bindings })
    }

    fn load_optional(runtime: &TrtRuntime, path: &Path, bindings: Bindings) -> Result<Option<Self>> {
        path.exists().then(|| Self::load(runtime, path, bindings)).transpose()
    }

    fn bound(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.bindings.iter().copied().filter(|(tensor, _)| self.engine.has_tensor(tensor))
    }
}

/// Memory bank geometry, derived from the engines' tensor sizes.
/// Slot 0 holds the initial frame permanently; slots 1.. are a FIFO of memory frames.
struct Bank {
    slots: usize,
    /// Bytes of one channel of one frame (H*W elements).
    plane_bytes: usize,
    key_channels: usize,
    value_channels: usize,
}

/// A frame is split so callers can use the alpha before the memory work for the next
/// frame finishes: `head` produces the alpha, a tail updates the state the next frame reads.
struct FrameGraphs {
    head: CudaGraph,
    tail_normal: CudaGraph,
    tail_update: CudaGraph,
}

pub struct MatAnyoneRunner {
    // Field order is drop order: graphs and engines go before the buffers they reference,
    // and the runtime goes last because engines must not outlive it.
    graphs: Option<FrameGraphs>,
    encode: Stage,
    read: Stage,
    pixel_fusion: Option<Stage>,
    segment: Stage,
    encode_mask: Stage,
    encode_mask_shallow: Option<Stage>,
    buffers: HashMap<&'static str, DevicePtr>,
    bank: Bank,
    next_slot: usize,
    stream: CudaStream,
    _runtime: TrtRuntime,
}

impl MatAnyoneRunner {
    pub fn new(engine_dir: &Path) -> Result<Self> {
        init_cuda_kernels()?;
        let runtime = TrtRuntime::new()?;
        let path = |name: &str| engine_dir.join(name);
        let encode = Stage::load(&runtime, &path("encode_image_fp16.engine"), ENCODE)?;
        let read = Stage::load(&runtime, &path("read_memory_fp16.engine"), READ)?;
        let pixel_fusion = Stage::load_optional(&runtime, &path("pixel_fusion_fp16.engine"), PIXEL_FUSION)?;
        let segment = Stage::load(&runtime, &path("segment_fp16.engine"), SEGMENT)?;
        let encode_mask = Stage::load(&runtime, &path("encode_mask_fp16.engine"), ENCODE_MASK)?;
        let encode_mask_shallow =
            Stage::load_optional(&runtime, &path("encode_mask_shallow_fp16.engine"), ENCODE_MASK_SHALLOW)?;

        let mut runner = Self {
            graphs: None,
            encode,
            read,
            pixel_fusion,
            segment,
            encode_mask,
            encode_mask_shallow,
            buffers: HashMap::new(),
            bank: Bank { slots: 0, plane_bytes: 0, key_channels: 0, value_channels: 0 },
            next_slot: 1,
            stream: CudaStream::new()?,
            _runtime: runtime,
        };
        runner.allocate_and_bind()?;
        runner.bank = runner.bank_layout()?;

        // TensorRT finishes lazy initialisation on the first enqueue, which must not
        // happen inside a capture. Run both paths once on the zeroed buffers, then capture.
        runner.enqueue_head()?;
        runner.enqueue_tail(false)?;
        runner.enqueue_tail(true)?;
        runner.stream.synchronize()?;
        runner.graphs = match runner.capture_graphs() {
            Ok(graphs) => Some(graphs),
            Err(err) => {
                eprintln!("CUDA graph capture failed, enqueueing engines directly: {err:#}");
                None
            }
        };
        Ok(runner)
    }

    fn stages(&self) -> impl Iterator<Item = &Stage> {
        [Some(&self.encode), Some(&self.read), self.pixel_fusion.as_ref(), Some(&self.segment)]
            .into_iter()
            .chain([Some(&self.encode_mask), self.encode_mask_shallow.as_ref()])
            .flatten()
    }

    /// Sizes every buffer from the engines that use it and binds all engine tensors.
    fn allocate_and_bind(&mut self) -> Result<()> {
        let mut specs: HashMap<&'static str, (usize, i32)> = HashMap::new();
        for stage in self.stages() {
            for (tensor, buffer) in stage.bound() {
                let bytes = stage.engine.tensor_bytes(tensor)?;
                let dtype = stage.engine.tensor_dtype(tensor)?;
                let spec = specs.entry(buffer).or_insert((bytes, dtype));
                if spec.1 != dtype {
                    bail!("{buffer}: {tensor} in {} has dtype {dtype}, other engines use {}", stage.engine.path(), spec.1);
                }
                spec.0 = spec.0.max(bytes);
            }
        }
        for name in ["image", "alpha", "obj_memory", "object_summaries"] {
            if specs.get(name).is_some_and(|spec| spec.1 != DTYPE_FLOAT) {
                bail!("{name} must be FP32");
            }
        }
        for (name, (bytes, _)) in specs {
            let buffer = DevicePtr::alloc(bytes, name)?;
            memset_async(buffer.as_ptr(), 0, bytes, self.stream.raw(), name)?;
            self.buffers.insert(name, buffer);
        }
        for stage in self.stages() {
            for (tensor, buffer) in stage.bound() {
                stage.engine.bind(tensor, self.buffers[buffer].as_ptr())?;
            }
        }
        self.stream.synchronize()
    }

    fn bank_layout(&self) -> Result<Bank> {
        let plane_bytes = self.bytes("shrinkage")?;
        let bank = Bank {
            slots: self.bytes("memory_shrinkage")? / plane_bytes,
            plane_bytes,
            key_channels: self.bytes("key")? / plane_bytes,
            value_channels: self.bytes("last_msk_value")? / plane_bytes,
        };
        if bank.slots < 2
            || self.bytes("memory_key")? != bank.slots * bank.key_channels * plane_bytes
            || self.bytes("memory_value")? != bank.slots * bank.value_channels * plane_bytes
        {
            bail!("memory bank tensors do not match key/shrinkage/value sizes");
        }
        Ok(bank)
    }

    fn ptr(&self, name: &str) -> Result<*mut c_void> {
        Ok(self.buffers.get(name).with_context(|| format!("missing buffer {name}"))?.as_ptr())
    }

    fn bytes(&self, name: &str) -> Result<usize> {
        Ok(self.buffers.get(name).with_context(|| format!("missing buffer {name}"))?.bytes)
    }

    fn copy(&self, dst: &str, src: &str) -> Result<()> {
        memcpy_async(self.ptr(dst)?, self.ptr(src)?, self.bytes(src)?, MEMCPY_D2D, self.stream.raw(), dst)
    }

    /// Image -> alpha.
    fn enqueue_head(&self) -> Result<()> {
        let stream = &self.stream;
        self.encode.engine.enqueue(stream)?;
        self.read.engine.enqueue(stream)?;
        if let Some(pf) = &self.pixel_fusion {
            pf.engine.enqueue(stream)?;
        }
        self.segment.engine.enqueue(stream)
    }

    /// State the next frame reads, except the memory bank write, which depends on the slot.
    fn enqueue_tail(&self, memory_update: bool) -> Result<()> {
        let stream = &self.stream;
        if memory_update {
            self.encode_mask.engine.enqueue(stream)?;
            if self.buffers.contains_key("obj_memory") {
                let n = (self.bytes("object_summaries")? / 4) as i32;
                launch_accumulate_f32(
                    self.ptr("obj_memory")? as *mut f32,
                    self.ptr("object_summaries")? as *const f32,
                    n,
                    stream,
                )?;
            }
        } else {
            self.copy("sensory", "new_sensory")?;
            if let Some(shallow) = &self.encode_mask_shallow {
                shallow.engine.enqueue(stream)?;
            }
        }
        self.copy("last_pix_feat", "pix_feat")?;
        self.copy("last_mask", "alpha")
    }

    fn capture_graphs(&self) -> Result<FrameGraphs> {
        Ok(FrameGraphs {
            head: CudaGraph::capture(&self.stream, || self.enqueue_head())?,
            tail_normal: CudaGraph::capture(&self.stream, || self.enqueue_tail(false))?,
            tail_update: CudaGraph::capture(&self.stream, || self.enqueue_tail(true))?,
        })
    }

    /// Copies the current key, shrinkage and mask value into memory slot `slot`.
    fn write_bank_slot(&self, slot: usize) -> Result<()> {
        let Bank { slots, plane_bytes, key_channels, value_channels } = self.bank;
        let stream = self.stream.raw();
        let plane = |name: &str| -> Result<*mut c_void> {
            Ok(unsafe { (self.ptr(name)? as *mut u8).add(slot * plane_bytes) as *mut c_void })
        };
        memcpy_2d_async(
            plane("memory_key")?,
            slots * plane_bytes,
            self.ptr("key")?,
            plane_bytes,
            plane_bytes,
            key_channels,
            MEMCPY_D2D,
            stream,
            "memory_key slot",
        )?;
        memcpy_async(plane("memory_shrinkage")?, self.ptr("shrinkage")?, plane_bytes, MEMCPY_D2D, stream, "memory_shrinkage slot")?;
        memcpy_2d_async(
            plane("memory_value")?,
            slots * plane_bytes,
            self.ptr("last_msk_value")?,
            plane_bytes,
            plane_bytes,
            value_channels,
            MEMCPY_D2D,
            stream,
            "memory_value slot",
        )
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Planar RGB f32 model input; write the frame here before `initialize`/`process_frame`.
    pub fn image_ptr(&self) -> *mut c_void {
        self.buffers["image"].as_ptr()
    }

    /// Alpha f32 of the latest frame, valid until the next `process_frame`.
    pub fn alpha_ptr(&self) -> *mut c_void {
        self.buffers["alpha"].as_ptr()
    }

    pub fn upload_image(&self, rgb_nchw: &[f32]) -> Result<()> {
        let bytes = self.bytes("image")?;
        if rgb_nchw.len() * 4 != bytes {
            bail!("image has {} floats, expected {}", rgb_nchw.len(), bytes / 4);
        }
        memcpy_async(self.image_ptr(), rgb_nchw.as_ptr() as *const _, bytes, MEMCPY_H2D, self.stream.raw(), "upload image")
    }

    /// Starts tracking from the frame in the image buffer and its alpha mask.
    pub fn initialize(&mut self, alpha_mask: &[f32]) -> Result<()> {
        let alpha_bytes = self.bytes("alpha")?;
        if alpha_mask.len() * 4 != alpha_bytes {
            bail!("mask has {} floats, expected {}", alpha_mask.len(), alpha_bytes / 4);
        }
        let stream = self.stream.raw();
        memcpy_async(self.alpha_ptr(), alpha_mask.as_ptr() as *const _, alpha_bytes, MEMCPY_H2D, stream, "upload mask")?;
        // Sensory memory starts at zero for a new object (initialize_sensory_if_needed).
        for name in ["sensory", "new_sensory"] {
            memset_async(self.ptr(name)?, 0, self.bytes(name)?, stream, name)?;
        }
        self.encode.engine.enqueue(&self.stream)?;
        self.encode_mask.engine.enqueue(&self.stream)?;
        for slot in 0..self.bank.slots {
            self.write_bank_slot(slot)?;
        }
        if self.buffers.contains_key("obj_memory") {
            self.copy("obj_memory", "object_summaries")?;
        }
        self.copy("last_pix_feat", "pix_feat")?;
        self.copy("last_mask", "alpha")?;
        self.next_slot = 1;
        self.stream.synchronize()
    }

    /// Segments the frame in the image buffer. `memory_update` also encodes it into memory.
    pub fn process_frame(&mut self, memory_update: bool) -> Result<()> {
        self.segment_frame()?;
        self.finish_frame(memory_update)
    }

    /// Enqueues the alpha computation for the frame in the image buffer. Work that reads
    /// the alpha can be enqueued before `finish_frame`, which must follow before the next frame.
    pub fn segment_frame(&mut self) -> Result<()> {
        match &self.graphs {
            Some(graphs) => graphs.head.launch(&self.stream),
            None => self.enqueue_head(),
        }
    }

    /// Enqueues the state updates the next frame depends on.
    pub fn finish_frame(&mut self, memory_update: bool) -> Result<()> {
        match &self.graphs {
            Some(graphs) if memory_update => graphs.tail_update.launch(&self.stream)?,
            Some(graphs) => graphs.tail_normal.launch(&self.stream)?,
            None => self.enqueue_tail(memory_update)?,
        }
        if memory_update {
            self.write_bank_slot(self.next_slot)?;
            self.next_slot = self.next_slot % (self.bank.slots - 1) + 1;
        }
        Ok(())
    }

    pub fn copy_alpha_to_host(&self, alpha: &mut [f32]) -> Result<()> {
        let bytes = self.bytes("alpha")?;
        if alpha.len() * 4 != bytes {
            bail!("alpha buffer has {} floats, expected {}", alpha.len(), bytes / 4);
        }
        memcpy_async(alpha.as_mut_ptr() as *mut _, self.alpha_ptr(), bytes, MEMCPY_D2H, self.stream.raw(), "download alpha")?;
        self.stream.synchronize()
    }
}

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::cuda::{memcpy_2d_async, memcpy_async, memset_async, CudaStream, MEMCPY_D2D, MEMCPY_D2H, MEMCPY_H2D};
use crate::trt::{TrtEngine, TrtRuntime};

const HW_BYTES: usize = 45 * 80 * std::mem::size_of::<f32>();
const SLOT_PITCH: usize = 5 * HW_BYTES;

pub struct MatAnyoneRunner {
    runtime: TrtRuntime,
    encode: TrtEngine,
    read: TrtEngine,
    pixel_fusion: Option<TrtEngine>,
    segment: TrtEngine,
    encode_mask: TrtEngine,
    stream: CudaStream,
    buffers: HashMap<String, Buffer>,
    mem_slot: i32,
}

struct Buffer {
    device: crate::cuda::DevicePtr,
}

impl MatAnyoneRunner {
    pub fn new(engine_dir: &Path) -> Result<Self> {
        let runtime = TrtRuntime::new()?;
        let encode = TrtEngine::load(&runtime, &engine_dir.join("encode_image_fp16.engine"))?;
        let read = TrtEngine::load(&runtime, &engine_dir.join("read_memory_fp16.engine"))?;
        let pixel_fusion_path = engine_dir.join("pixel_fusion_fp16.engine");
        let pixel_fusion = if pixel_fusion_path.exists() {
            Some(TrtEngine::load(&runtime, &pixel_fusion_path)?)
        } else {
            None
        };
        let segment = TrtEngine::load(&runtime, &engine_dir.join("segment_fp16.engine"))?;
        let encode_mask = TrtEngine::load(&runtime, &engine_dir.join("encode_mask_fp16.engine"))?;
        let stream = CudaStream::new()?;

        let mut runner = Self {
            runtime,
            encode,
            read,
            pixel_fusion,
            segment,
            encode_mask,
            stream,
            buffers: HashMap::new(),
            mem_slot: 1,
        };
        runner.allocate_static_buffers()?;
        runner.bind_engines()?;

        memset_async(
            runner.buffer_ptr("memory_key")?,
            0,
            runner.buffer_bytes("memory_key")?,
            runner.stream.raw(),
            "memset memory_key",
        )?;
        memset_async(
            runner.buffer_ptr("memory_shrinkage")?,
            0,
            runner.buffer_bytes("memory_shrinkage")?,
            runner.stream.raw(),
            "memset memory_shrinkage",
        )?;
        memset_async(
            runner.buffer_ptr("memory_value")?,
            0,
            runner.buffer_bytes("memory_value")?,
            runner.stream.raw(),
            "memset memory_value",
        )?;
        memset_async(
            runner.buffer_ptr("obj_memory")?,
            0,
            runner.buffer_bytes("obj_memory")?,
            runner.stream.raw(),
            "memset obj_memory",
        )?;
        runner.stream.synchronize()?;
        Ok(runner)
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    fn allocate_buffer(&mut self, name: &str, dims: &[i64]) -> Result<()> {
        if self.buffers.contains_key(name) {
            return Ok(());
        }
        let vol: i64 = dims.iter().product();
        let bytes = vol as usize * std::mem::size_of::<f32>();
        let device = crate::cuda::DevicePtr::alloc(bytes, name)?;
        self.buffers.insert(name.to_string(), Buffer { device });
        Ok(())
    }

    fn allocate_static_buffers(&mut self) -> Result<()> {
        self.allocate_buffer("image", &[1, 3, 720, 1280])?;
        self.allocate_buffer("last_mask", &[1, 1, 720, 1280])?;
        self.allocate_buffer("alpha", &[1, 1, 720, 1280])?;
        self.allocate_buffer("f16", &[1, 1024, 45, 80])?;
        self.allocate_buffer("f8", &[1, 512, 90, 160])?;
        self.allocate_buffer("f4", &[1, 256, 180, 320])?;
        self.allocate_buffer("f2", &[1, 64, 360, 640])?;
        self.allocate_buffer("f1", &[1, 3, 720, 1280])?;
        self.allocate_buffer("pix_feat", &[1, 256, 45, 80])?;
        self.allocate_buffer("last_pix_feat", &[1, 256, 45, 80])?;
        self.allocate_buffer("key", &[1, 64, 45, 80])?;
        self.allocate_buffer("shrinkage", &[1, 1, 45, 80])?;
        self.allocate_buffer("selection", &[1, 64, 45, 80])?;
        self.allocate_buffer("memory_key", &[1, 64, 5, 45, 80])?;
        self.allocate_buffer("memory_shrinkage", &[1, 1, 5, 45, 80])?;
        self.allocate_buffer("memory_value", &[1, 1, 256, 5, 45, 80])?;
        self.allocate_buffer("last_msk_value", &[1, 1, 256, 45, 80])?;
        self.allocate_buffer("sensory", &[1, 1, 256, 45, 80])?;
        self.allocate_buffer("new_sensory", &[1, 1, 256, 45, 80])?;
        self.allocate_buffer("obj_memory", &[1, 1, 1, 16, 257])?;
        self.allocate_buffer("pixel_memory", &[1, 1, 256, 45, 80])?;
        self.allocate_buffer("memory_readout", &[1, 1, 256, 45, 80])?;
        self.allocate_buffer("mask_value", &[1, 1, 256, 45, 80])?;
        self.allocate_buffer("object_summaries", &[1, 1, 16, 257])?;
        Ok(())
    }

    fn buffer_ptr(&self, name: &str) -> Result<*mut std::ffi::c_void> {
        Ok(self
            .buffers
            .get(name)
            .with_context(|| format!("missing buffer {name}"))?
            .device
            .as_ptr())
    }

    fn buffer_bytes(&self, name: &str) -> Result<usize> {
        Ok(self
            .buffers
            .get(name)
            .with_context(|| format!("missing buffer {name}"))?
            .device
            .bytes)
    }

    fn bind_engines(&self) -> Result<()> {
        self.encode.bind("input_0", self.buffer_ptr("image")?)?;
        self.encode.bind("f16", self.buffer_ptr("f16")?)?;
        self.encode.bind("f8", self.buffer_ptr("f8")?)?;
        self.encode.bind("f4", self.buffer_ptr("f4")?)?;
        self.encode.bind("f2", self.buffer_ptr("f2")?)?;
        self.encode.bind("f1", self.buffer_ptr("f1")?)?;
        self.encode.bind("pix_feat", self.buffer_ptr("pix_feat")?)?;
        self.encode.bind("key", self.buffer_ptr("key")?)?;
        self.encode.bind("shrinkage", self.buffer_ptr("shrinkage")?)?;
        self.encode.bind("selection", self.buffer_ptr("selection")?)?;

        self.read.bind("input_0", self.buffer_ptr("key")?)?;
        self.read.bind("input_1", self.buffer_ptr("selection")?)?;
        self.read.bind("input_2", self.buffer_ptr("memory_key")?)?;
        self.read.bind("input_3", self.buffer_ptr("memory_shrinkage")?)?;
        self.read.bind("input_4", self.buffer_ptr("memory_value")?)?;
        self.read.bind("input_5", self.buffer_ptr("last_pix_feat")?)?;
        self.read.bind("input_6", self.buffer_ptr("pix_feat")?)?;
        self.read.bind("input_7", self.buffer_ptr("last_mask")?)?;
        self.read.bind("input_8", self.buffer_ptr("last_msk_value")?)?;
        self.read.bind("input_9", self.buffer_ptr("sensory")?)?;
        self.read.bind("input_10", self.buffer_ptr("obj_memory")?)?;
        self.read.bind("pixel_memory", self.buffer_ptr("pixel_memory")?)?;
        self.read.bind("memory_readout", self.buffer_ptr("memory_readout")?)?;

        if let Some(ref pf) = self.pixel_fusion {
            pf.bind("input_0", self.buffer_ptr("pix_feat")?)?;
            pf.bind("input_1", self.buffer_ptr("pixel_memory")?)?;
            pf.bind("input_2", self.buffer_ptr("sensory")?)?;
            pf.bind("input_3", self.buffer_ptr("last_mask")?)?;
            pf.bind("fused_pixel", self.buffer_ptr("memory_readout")?)?;
        }

        self.segment.bind("input_0", self.buffer_ptr("f16")?)?;
        self.segment.bind("input_1", self.buffer_ptr("f8")?)?;
        self.segment.bind("input_2", self.buffer_ptr("f4")?)?;
        self.segment.bind("input_3", self.buffer_ptr("f2")?)?;
        self.segment.bind("input_4", self.buffer_ptr("f1")?)?;
        self.segment.bind("input_5", self.buffer_ptr("memory_readout")?)?;
        self.segment.bind("input_6", self.buffer_ptr("sensory")?)?;
        self.segment.bind("new_sensory", self.buffer_ptr("new_sensory")?)?;
        self.segment.bind("alpha", self.buffer_ptr("alpha")?)?;

        self.encode_mask.bind("input_0", self.buffer_ptr("image")?)?;
        self.encode_mask.bind("input_1", self.buffer_ptr("pix_feat")?)?;
        self.encode_mask.bind("input_2", self.buffer_ptr("new_sensory")?)?;
        self.encode_mask.bind("input_3", self.buffer_ptr("alpha")?)?;
        self.encode_mask.bind("mask_value", self.buffer_ptr("mask_value")?)?;
        self.encode_mask.bind("new_sensory", self.buffer_ptr("sensory")?)?;
        self.encode_mask
            .bind("object_summaries", self.buffer_ptr("object_summaries")?)?;
        Ok(())
    }

    fn copy_time_slot_from_contiguous(
        &self,
        dst: *mut u8,
        slot: i32,
        src: *const std::ffi::c_void,
        channels: i32,
        what: &str,
    ) -> Result<()> {
        memcpy_2d_async(
            unsafe { dst.add(slot as usize * HW_BYTES) as *mut _ },
            SLOT_PITCH,
            src,
            HW_BYTES,
            HW_BYTES,
            channels as usize,
            MEMCPY_D2D,
            self.stream.raw(),
            what,
        )
    }

    fn copy_time_slot_to_slot(
        &self,
        base: *mut u8,
        dst_slot: i32,
        src_slot: i32,
        channels: i32,
        what: &str,
    ) -> Result<()> {
        memcpy_2d_async(
            unsafe { base.add(dst_slot as usize * HW_BYTES) as *mut _ },
            SLOT_PITCH,
            unsafe { base.add(src_slot as usize * HW_BYTES) as *const _ },
            SLOT_PITCH,
            HW_BYTES,
            channels as usize,
            MEMCPY_D2D,
            self.stream.raw(),
            what,
        )
    }

    fn advance_memory_bank(&mut self, deep_update: bool) -> Result<()> {
        let shrink_frame_bytes = 45 * 80 * std::mem::size_of::<f32>();
        let value_frame_bytes = 256 * 45 * 80 * std::mem::size_of::<f32>();
        let slot = if self.mem_slot < 5 { self.mem_slot } else { 4 };

        let memory_key = self.buffer_ptr("memory_key")? as *mut u8;
        let memory_shrink = self.buffer_ptr("memory_shrinkage")? as *mut u8;
        let memory_value = self.buffer_ptr("memory_value")? as *mut u8;

        if self.mem_slot >= 5 {
            for dst in 1..4 {
                let src = dst + 1;
                self.copy_time_slot_to_slot(memory_key, dst, src, 64, "shift memory_key slot")?;
                memcpy_async(
                    unsafe { memory_shrink.add(dst as usize * shrink_frame_bytes) as *mut _ },
                    unsafe { memory_shrink.add(src as usize * shrink_frame_bytes) as *const _ },
                    shrink_frame_bytes,
                    MEMCPY_D2D,
                    self.stream.raw(),
                    "shift memory_shrinkage slot",
                )?;
                self.copy_time_slot_to_slot(memory_value, dst, src, 256, "shift memory_value slot")?;
            }
        }

        self.copy_time_slot_from_contiguous(
            memory_key,
            slot,
            self.buffer_ptr("key")?,
            64,
            "copy memory_key slot",
        )?;
        memcpy_async(
            unsafe { memory_shrink.add(slot as usize * shrink_frame_bytes) as *mut _ },
            self.buffer_ptr("shrinkage")?,
            shrink_frame_bytes,
            MEMCPY_D2D,
            self.stream.raw(),
            "copy memory_shrinkage slot",
        )?;
        self.copy_time_slot_from_contiguous(
            memory_value,
            slot,
            self.buffer_ptr("mask_value")?,
            256,
            "copy memory_value slot",
        )?;
        memcpy_async(
            self.buffer_ptr("last_msk_value")?,
            self.buffer_ptr("mask_value")?,
            value_frame_bytes,
            MEMCPY_D2D,
            self.stream.raw(),
            "copy last_msk_value",
        )?;

        if deep_update {
            memcpy_async(
                self.buffer_ptr("obj_memory")?,
                self.buffer_ptr("object_summaries")?,
                1 * 1 * 16 * 257 * std::mem::size_of::<f32>(),
                MEMCPY_D2D,
                self.stream.raw(),
                "copy obj_memory",
            )?;
            memcpy_async(
                self.buffer_ptr("last_pix_feat")?,
                self.buffer_ptr("pix_feat")?,
                value_frame_bytes,
                MEMCPY_D2D,
                self.stream.raw(),
                "copy last_pix_feat",
            )?;
            memcpy_async(
                self.buffer_ptr("last_mask")?,
                self.buffer_ptr("alpha")?,
                720 * 1280 * std::mem::size_of::<f32>(),
                MEMCPY_D2D,
                self.stream.raw(),
                "copy last_mask",
            )?;
        }
        self.mem_slot += 1;
        Ok(())
    }

    fn seed_memory_bank(&mut self) -> Result<()> {
        let shrink_frame_bytes = 45 * 80 * std::mem::size_of::<f32>();
        let value_frame_bytes = 256 * 45 * 80 * std::mem::size_of::<f32>();
        let memory_key = self.buffer_ptr("memory_key")? as *mut u8;
        let memory_shrink = self.buffer_ptr("memory_shrinkage")? as *mut u8;
        let memory_value = self.buffer_ptr("memory_value")? as *mut u8;

        for slot in 0..5 {
            self.copy_time_slot_from_contiguous(
                memory_key,
                slot,
                self.buffer_ptr("key")?,
                64,
                "seed memory_key",
            )?;
            memcpy_async(
                unsafe { memory_shrink.add(slot as usize * shrink_frame_bytes) as *mut _ },
                self.buffer_ptr("shrinkage")?,
                shrink_frame_bytes,
                MEMCPY_D2D,
                self.stream.raw(),
                "seed memory_shrinkage",
            )?;
            self.copy_time_slot_from_contiguous(
                memory_value,
                slot,
                self.buffer_ptr("mask_value")?,
                256,
                "seed memory_value",
            )?;
        }
        memcpy_async(
            self.buffer_ptr("obj_memory")?,
            self.buffer_ptr("object_summaries")?,
            1 * 1 * 16 * 257 * std::mem::size_of::<f32>(),
            MEMCPY_D2D,
            self.stream.raw(),
            "seed obj_memory",
        )?;
        memcpy_async(
            self.buffer_ptr("last_pix_feat")?,
            self.buffer_ptr("pix_feat")?,
            value_frame_bytes,
            MEMCPY_D2D,
            self.stream.raw(),
            "seed last_pix_feat",
        )?;
        memcpy_async(
            self.buffer_ptr("last_msk_value")?,
            self.buffer_ptr("mask_value")?,
            value_frame_bytes,
            MEMCPY_D2D,
            self.stream.raw(),
            "seed last_msk_value",
        )?;
        memcpy_async(
            self.buffer_ptr("last_mask")?,
            self.buffer_ptr("alpha")?,
            720 * 1280 * std::mem::size_of::<f32>(),
            MEMCPY_D2D,
            self.stream.raw(),
            "seed last_mask",
        )?;
        self.mem_slot = 1;
        Ok(())
    }

    pub fn initialize(&mut self, rgb_nchw: &[f32], alpha_mask_nchw: &[f32]) -> Result<()> {
        memcpy_async(
            self.buffer_ptr("image")?,
            rgb_nchw.as_ptr() as *const _,
            3 * 720 * 1280 * std::mem::size_of::<f32>(),
            MEMCPY_H2D,
            self.stream.raw(),
            "copy initial image",
        )?;
        memcpy_async(
            self.buffer_ptr("alpha")?,
            alpha_mask_nchw.as_ptr() as *const _,
            720 * 1280 * std::mem::size_of::<f32>(),
            MEMCPY_H2D,
            self.stream.raw(),
            "copy initial alpha",
        )?;
        self.encode.enqueue(&self.stream)?;
        self.encode_mask.enqueue(&self.stream)?;
        self.seed_memory_bank()?;
        self.stream.synchronize()
    }

    pub fn initialize_device(
        &mut self,
        rgb_nchw_device: *const std::ffi::c_void,
        alpha_mask_nchw_device: *const std::ffi::c_void,
    ) -> Result<()> {
        memcpy_async(
            self.buffer_ptr("image")?,
            rgb_nchw_device,
            3 * 720 * 1280 * std::mem::size_of::<f32>(),
            MEMCPY_D2D,
            self.stream.raw(),
            "copy initial image from device",
        )?;
        memcpy_async(
            self.buffer_ptr("alpha")?,
            alpha_mask_nchw_device,
            720 * 1280 * std::mem::size_of::<f32>(),
            MEMCPY_D2D,
            self.stream.raw(),
            "copy initial alpha from device",
        )?;
        self.encode.enqueue(&self.stream)?;
        self.encode_mask.enqueue(&self.stream)?;
        self.seed_memory_bank()?;
        self.stream.synchronize()
    }

    pub fn process_frame_rgb(&mut self, rgb_nchw: &[f32], memory_update: bool) -> Result<()> {
        memcpy_async(
            self.buffer_ptr("image")?,
            rgb_nchw.as_ptr() as *const _,
            3 * 720 * 1280 * std::mem::size_of::<f32>(),
            MEMCPY_H2D,
            self.stream.raw(),
            "copy frame image",
        )?;
        self.process_frame(memory_update)
    }

    pub fn process_frame_rgb_device(
        &mut self,
        rgb_nchw_device: *const std::ffi::c_void,
        memory_update: bool,
    ) -> Result<()> {
        memcpy_async(
            self.buffer_ptr("image")?,
            rgb_nchw_device,
            3 * 720 * 1280 * std::mem::size_of::<f32>(),
            MEMCPY_D2D,
            self.stream.raw(),
            "copy frame image from device",
        )?;
        self.process_frame(memory_update)
    }

    pub fn copy_alpha_to_host(&self, alpha_nchw: &mut [f32]) -> Result<()> {
        memcpy_async(
            alpha_nchw.as_mut_ptr() as *mut _,
            self.buffer_ptr("alpha")?,
            720 * 1280 * std::mem::size_of::<f32>(),
            MEMCPY_D2H,
            self.stream.raw(),
            "copy alpha to host",
        )?;
        self.stream.synchronize()
    }

    pub fn copy_alpha_to_device(&self, alpha_device: *mut std::ffi::c_void) -> Result<()> {
        memcpy_async(
            alpha_device,
            self.buffer_ptr("alpha")?,
            720 * 1280 * std::mem::size_of::<f32>(),
            MEMCPY_D2D,
            self.stream.raw(),
            "copy alpha to device",
        )
    }

    pub fn process_frame(&mut self, memory_update: bool) -> Result<()> {
        self.encode.enqueue(&self.stream)?;
        self.read.enqueue(&self.stream)?;
        if let Some(ref pf) = self.pixel_fusion {
            pf.enqueue(&self.stream)?;
        }
        self.segment.enqueue(&self.stream)?;
        if memory_update {
            self.encode_mask.enqueue(&self.stream)?;
            self.advance_memory_bank(true)?;
        }
        Ok(())
    }

    pub fn alpha_device_ptr(&self) -> Result<*mut std::ffi::c_void> {
        self.buffer_ptr("alpha")
    }
}

// Silence unused runtime field warning — runtime must outlive engines
impl Drop for MatAnyoneRunner {
    fn drop(&mut self) {
        let _ = &self.runtime;
    }
}

pub fn default_engine_dir() -> PathBuf {
    PathBuf::from("engines/faithful")
}

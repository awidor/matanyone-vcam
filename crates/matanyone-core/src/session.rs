use std::path::Path;

use anyhow::{Context, Result, bail};
use image::GenericImageView;

use crate::cuda::{memcpy_async, CudaEvent, DevicePtr, HostBuffer, MEMCPY_D2H, MEMCPY_H2D};
use crate::kernels::{launch_bgr_to_rgb_nchw, launch_composite_bgr};
use crate::runner::MatAnyoneRunner;

pub const MODEL_W: usize = 1280;
pub const MODEL_H: usize = 720;
pub const FRAME_BYTES: usize = MODEL_W * MODEL_H * 3;

/// Staging for an incoming packed-BGR frame: pinned host copy plus its device mirror.
struct Upload {
    host: HostBuffer,
    device: DevicePtr,
}

pub struct Session {
    runner: MatAnyoneRunner,
    upload: Option<Upload>,
    d_bgr_out: DevicePtr,
    h_bgr_out: HostBuffer,
    readback_done: CudaEvent,
    initialized: bool,
    frame_index: u64,
}

impl Session {
    pub fn new(engine_dir: &Path) -> Result<Self> {
        Ok(Self {
            runner: MatAnyoneRunner::new(engine_dir)?,
            upload: None,
            d_bgr_out: DevicePtr::alloc(FRAME_BYTES, "d_bgr_out")?,
            h_bgr_out: HostBuffer::alloc(FRAME_BYTES, "h_bgr_out")?,
            readback_done: CudaEvent::new()?,
            initialized: false,
            frame_index: 0,
        })
    }

    pub fn init_from_mask_file(
        &mut self,
        bgr: &[u8],
        src_w: u32,
        src_h: u32,
        mask_png: &Path,
    ) -> Result<()> {
        let mask = load_mask_png(mask_png, MODEL_W, MODEL_H)?;
        self.init_from_mask(bgr, src_w, src_h, &mask, MODEL_W, MODEL_H)
    }

    pub fn init_from_mask(
        &mut self,
        bgr: &[u8],
        src_w: u32,
        src_h: u32,
        mask_hw: &[f32],
        mask_w: usize,
        mask_h: usize,
    ) -> Result<()> {
        self.upload_bgr(bgr, src_w, src_h)?;
        let mask_model = if mask_w == MODEL_W && mask_h == MODEL_H {
            mask_hw.to_vec()
        } else {
            resize_mask_nearest(mask_hw, mask_w, mask_h, MODEL_W, MODEL_H)
        };
        self.runner.initialize(&mask_model)?;
        self.initialized = true;
        self.frame_index = 1;
        Ok(())
    }

    pub fn process_bgr(
        &mut self,
        bgr_in: &[u8],
        src_w: u32,
        src_h: u32,
        bgr_out: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        bg_rgb: (f32, f32, f32),
        memory_update: bool,
    ) -> Result<()> {
        if !self.initialized {
            bail!("session not initialized");
        }
        if bgr_out.len() != dst_w as usize * dst_h as usize * 3 {
            bail!("output buffer has {} bytes, expected {dst_w}x{dst_h}x3", bgr_out.len());
        }
        self.upload_bgr(bgr_in, src_w, src_h)?;
        self.runner.segment_frame()?;
        self.frame_index += 1;

        let stream = self.runner.stream();
        launch_composite_bgr(
            self.runner.image_ptr() as *const f32,
            self.runner.alpha_ptr() as *const f32,
            bg_rgb,
            self.d_bgr_out.as_ptr() as *mut u8,
            MODEL_W as i32,
            MODEL_H as i32,
            stream,
        )?;
        memcpy_async(
            self.h_bgr_out.as_mut_slice().as_mut_ptr() as *mut _,
            self.d_bgr_out.as_ptr(),
            FRAME_BYTES,
            MEMCPY_D2H,
            stream.raw(),
            "readback bgr",
        )?;
        self.readback_done.record(stream)?;
        // The memory update for the next frame runs on the GPU while the caller uses this one.
        self.runner.finish_frame(memory_update)?;
        self.readback_done.synchronize()?;

        let model_bgr = self.h_bgr_out.as_slice();
        if dst_w == MODEL_W as u32 && dst_h == MODEL_H as u32 {
            bgr_out.copy_from_slice(model_bgr);
            return Ok(());
        }
        for y in 0..dst_h {
            let sy = ((y * MODEL_H as u32) / dst_h).min(MODEL_H as u32 - 1);
            for x in 0..dst_w {
                let sx = ((x * MODEL_W as u32) / dst_w).min(MODEL_W as u32 - 1);
                let src_idx = (sy as usize * MODEL_W + sx as usize) * 3;
                let dst_idx = (y as usize * dst_w as usize + x as usize) * 3;
                bgr_out[dst_idx..dst_idx + 3].copy_from_slice(&model_bgr[src_idx..src_idx + 3]);
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.initialized = false;
        self.frame_index = 0;
        Ok(())
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Uploads a packed BGR frame of any size into the runner's model input (resized on GPU).
    fn upload_bgr(&mut self, bgr: &[u8], src_w: u32, src_h: u32) -> Result<()> {
        let bytes = src_w as usize * src_h as usize * 3;
        if bgr.len() < bytes {
            bail!("frame has {} bytes, expected {src_w}x{src_h}x3", bgr.len());
        }
        if self.upload.as_ref().is_none_or(|u| u.host.len() < bytes) {
            self.upload = Some(Upload {
                host: HostBuffer::alloc(bytes, "bgr upload")?,
                device: DevicePtr::alloc(bytes, "d_bgr_in")?,
            });
        }
        let upload = self.upload.as_mut().unwrap();
        upload.host.as_mut_slice()[..bytes].copy_from_slice(&bgr[..bytes]);
        let stream = self.runner.stream();
        memcpy_async(
            upload.device.as_ptr(),
            upload.host.as_slice().as_ptr() as *const _,
            bytes,
            MEMCPY_H2D,
            stream.raw(),
            "upload bgr",
        )?;
        launch_bgr_to_rgb_nchw(
            upload.device.as_ptr() as *const u8,
            src_w as i32,
            src_h as i32,
            (src_w * 3) as i32,
            self.runner.image_ptr() as *mut f32,
            MODEL_W as i32,
            MODEL_H as i32,
            stream,
        )
    }
}

fn resize_mask_nearest(src: &[f32], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_w * dst_h];
    for y in 0..dst_h {
        let sy = (y * src_h / dst_h).min(src_h.saturating_sub(1));
        for x in 0..dst_w {
            let sx = (x * src_w / dst_w).min(src_w.saturating_sub(1));
            dst[y * dst_w + x] = src[sy * src_w + sx];
        }
    }
    dst
}

fn load_mask_png(path: &Path, dst_w: usize, dst_h: usize) -> Result<Vec<f32>> {
    let img = image::open(path).with_context(|| format!("failed to load mask image: {}", path.display()))?;
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8();
    // The matte is the alpha channel when the PNG has a real one; otherwise the gray level
    // (max of RGB), as in grayscale masks from SAM or the MatAnyone2 samples.
    let use_alpha = img.color().has_alpha() && rgba.pixels().any(|px| px[3] < 255);
    let alpha: Vec<f32> = rgba
        .pixels()
        .map(|px| {
            let value = if use_alpha { px[3] } else { px[0].max(px[1]).max(px[2]) };
            value as f32 / 255.0
        })
        .collect();
    if w as usize == dst_w && h as usize == dst_h {
        return Ok(alpha);
    }
    Ok(resize_mask_nearest(&alpha, w as usize, h as usize, dst_w, dst_h))
}

pub fn center_mask(w: usize, h: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; w * h];
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let rx = w as f32 * 0.22;
    let ry = h as f32 * 0.35;
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / rx;
            let dy = (y as f32 - cy) / ry;
            if dx * dx + dy * dy <= 1.0 {
                mask[y * w + x] = 1.0;
            }
        }
    }
    mask
}

pub fn make_synthetic_bgr(w: usize, h: usize, frame: i32) -> Vec<u8> {
    let mut bgr = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            bgr[idx] = ((x + frame as usize * 3) % 256) as u8;
            bgr[idx + 1] = ((y + frame as usize * 2) % 256) as u8;
            bgr[idx + 2] = ((x + y + frame as usize) % 256) as u8;
        }
    }
    bgr
}

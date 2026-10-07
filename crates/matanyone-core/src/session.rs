use std::path::Path;

use anyhow::{Context, Result, bail};
use image::GenericImageView;

use crate::cuda::{memcpy_async, CudaEvent, DevicePtr, HostBuffer, MEMCPY_D2H, MEMCPY_H2D};
use crate::kernels::{launch_bgr_to_rgb_nchw, launch_composite_bgr, launch_nv12_to_rgb_nchw};
use crate::runner::MatAnyoneRunner;

pub const MODEL_W: usize = 1280;
pub const MODEL_H: usize = 720;
pub const FRAME_BYTES: usize = MODEL_W * MODEL_H * 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// Packed 8-bit BGR.
    Bgr,
    /// 8-bit Y plane followed by interleaved UV at half resolution, BT.709 limited range.
    Nv12,
}

/// An input frame at any size; the session converts and resizes it on the GPU.
#[derive(Clone, Copy)]
pub struct Frame<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
}

impl<'a> Frame<'a> {
    pub fn bgr(data: &'a [u8], width: u32, height: u32) -> Self {
        Self { data, width, height, format: PixelFormat::Bgr }
    }

    pub fn nv12(data: &'a [u8], width: u32, height: u32) -> Self {
        Self { data, width, height, format: PixelFormat::Nv12 }
    }

    pub fn byte_len(&self) -> usize {
        let pixels = self.width as usize * self.height as usize;
        match self.format {
            PixelFormat::Bgr => pixels * 3,
            PixelFormat::Nv12 => pixels * 3 / 2,
        }
    }

    /// Packed BGR at `dst_w`x`dst_h` (nearest neighbour) on the CPU, for one-off uses
    /// such as SAM prompts and the first preview.
    pub fn to_bgr(&self, dst_w: usize, dst_h: usize) -> Vec<u8> {
        let (w, h) = (self.width as usize, self.height as usize);
        let mut out = vec![0u8; dst_w * dst_h * 3];
        for y in 0..dst_h {
            let sy = (y * h / dst_h).min(h - 1);
            for x in 0..dst_w {
                let sx = (x * w / dst_w).min(w - 1);
                let px = &mut out[(y * dst_w + x) * 3..][..3];
                match self.format {
                    PixelFormat::Bgr => px.copy_from_slice(&self.data[(sy * w + sx) * 3..][..3]),
                    PixelFormat::Nv12 => {
                        let luma = self.data[sy * w + sx] as f32;
                        let uv = &self.data[w * h + (sy / 2) * w + (sx / 2) * 2..][..2];
                        let yn = (luma - 16.0) / 219.0;
                        let un = (uv[0] as f32 - 128.0) / 224.0;
                        let vn = (uv[1] as f32 - 128.0) / 224.0;
                        let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                        px[0] = to_u8(yn + 1.8556 * un);
                        px[1] = to_u8(yn - 0.1873 * un - 0.4681 * vn);
                        px[2] = to_u8(yn + 1.5748 * vn);
                    }
                }
            }
        }
        out
    }
}

/// Staging for an incoming frame: pinned host copy plus its device mirror.
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

    pub fn init_from_mask_file(&mut self, frame: Frame, mask_png: &Path) -> Result<()> {
        let mask = load_mask_png(mask_png, MODEL_W, MODEL_H)?;
        self.init_from_mask(frame, &mask, MODEL_W, MODEL_H)
    }

    pub fn init_from_mask(&mut self, frame: Frame, mask_hw: &[f32], mask_w: usize, mask_h: usize) -> Result<()> {
        self.upload(frame)?;
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

    /// Mattes `frame` and writes the composite as packed BGR at `dst_w`x`dst_h`.
    pub fn process(
        &mut self,
        frame: Frame,
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
        self.upload(frame)?;
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

    /// Uploads a frame of any size into the runner's model input (converted and resized on GPU).
    fn upload(&mut self, frame: Frame) -> Result<()> {
        let bytes = frame.byte_len();
        if frame.data.len() < bytes {
            bail!("{:?} frame has {} bytes, expected {bytes} for {}x{}", frame.format, frame.data.len(), frame.width, frame.height);
        }
        if frame.format == PixelFormat::Nv12 && (frame.width % 2 != 0 || frame.height % 2 != 0) {
            bail!("NV12 frame size {}x{} is not even", frame.width, frame.height);
        }
        if self.upload.as_ref().is_none_or(|u| u.host.len() < bytes) {
            self.upload = Some(Upload {
                host: HostBuffer::alloc(bytes, "frame upload")?,
                device: DevicePtr::alloc(bytes, "d_frame_in")?,
            });
        }
        let upload = self.upload.as_mut().unwrap();
        upload.host.as_mut_slice()[..bytes].copy_from_slice(&frame.data[..bytes]);
        let stream = self.runner.stream();
        memcpy_async(
            upload.device.as_ptr(),
            upload.host.as_slice().as_ptr() as *const _,
            bytes,
            MEMCPY_H2D,
            stream.raw(),
            "upload frame",
        )?;
        let (src, w, h) = (upload.device.as_ptr() as *const u8, frame.width as i32, frame.height as i32);
        let rgb = self.runner.image_ptr() as *mut f32;
        match frame.format {
            PixelFormat::Bgr => launch_bgr_to_rgb_nchw(src, w, h, w * 3, rgb, MODEL_W as i32, MODEL_H as i32, stream),
            PixelFormat::Nv12 => launch_nv12_to_rgb_nchw(src, w, h, rgb, MODEL_W as i32, MODEL_H as i32, stream),
        }
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

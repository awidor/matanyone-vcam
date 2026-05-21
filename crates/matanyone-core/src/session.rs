use std::path::Path;

use anyhow::{Context, Result, bail};
use image::GenericImageView;

use crate::cuda::{memcpy_async, DevicePtr, CudaStream, MEMCPY_D2H, MEMCPY_H2D};
use crate::ffi;
use crate::kernels::{init_cuda_kernels, launch_alpha_composite, launch_bgra_to_rgb_nchw};
use crate::runner::MatAnyoneRunner;

pub const MODEL_W: usize = 1280;
pub const MODEL_H: usize = 720;
pub const FRAME_BYTES: usize = MODEL_W * MODEL_H * 3;

pub struct Session {
    runner: MatAnyoneRunner,
    d_rgb_nchw: DevicePtr,
    d_alpha: DevicePtr,
    d_bgra_in: Option<DevicePtr>,
    d_bgra_out: DevicePtr,
    initialized: bool,
    frame_index: u64,
}

impl Session {
    pub fn new(engine_dir: &Path) -> Result<Self> {
        init_cuda_kernels().context("failed to initialize CUDA kernels")?;
        let runner = MatAnyoneRunner::new(engine_dir)?;
        let rgb_bytes = 3 * MODEL_H * MODEL_W * std::mem::size_of::<f32>();
        let alpha_bytes = MODEL_H * MODEL_W * std::mem::size_of::<f32>();
        let bgra_bytes = MODEL_W * MODEL_H * 4;
        Ok(Self {
            runner,
            d_rgb_nchw: DevicePtr::alloc(rgb_bytes, "d_rgb_nchw")?,
            d_alpha: DevicePtr::alloc(alpha_bytes, "d_alpha")?,
            d_bgra_in: None,
            d_bgra_out: DevicePtr::alloc(bgra_bytes, "d_bgra_out")?,
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
        let stream_raw = self.runner.stream().raw();
        self.upload_bgr_to_rgb(bgr, src_w, src_h, stream_raw)?;

        let mask_model = if mask_w == MODEL_W && mask_h == MODEL_H {
            mask_hw.to_vec()
        } else {
            resize_mask_nearest(mask_hw, mask_w, mask_h, MODEL_W, MODEL_H)
        };

        let d_init_alpha = DevicePtr::alloc(MODEL_W * MODEL_H * 4, "init alpha")?;
        memcpy_async(
            d_init_alpha.as_ptr(),
            mask_model.as_ptr() as *const _,
            MODEL_W * MODEL_H * std::mem::size_of::<f32>(),
            MEMCPY_H2D,
            stream_raw,
            "upload init alpha",
        )?;
        self.runner
            .initialize_device(self.d_rgb_nchw.as_ptr(), d_init_alpha.as_ptr())?;
        self.runner
            .copy_alpha_to_device(self.d_alpha.as_ptr())?;
        self.initialized = true;
        self.frame_index = 1;
        self.runner.stream().synchronize()
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
        let stream_raw = self.runner.stream().raw();
        self.upload_bgr_to_rgb(bgr_in, src_w, src_h, stream_raw)?;
        self.runner
            .process_frame_rgb_device(self.d_rgb_nchw.as_ptr(), memory_update)?;
        self.runner
            .copy_alpha_to_device(self.d_alpha.as_ptr())?;
        self.frame_index += 1;

        let stream = self.runner.stream();
        launch_alpha_composite(
            self.d_rgb_nchw.as_ptr() as *const f32,
            self.d_alpha.as_ptr() as *const f32,
            bg_rgb.0,
            bg_rgb.1,
            bg_rgb.2,
            self.d_bgra_out.as_ptr() as *mut u8,
            MODEL_H as i32,
            MODEL_W as i32,
            stream,
        )?;
        self.readback_bgr(bgr_out, dst_w, dst_h, stream)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.initialized = false;
        self.frame_index = 0;
        Ok(())
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    fn upload_bgr_to_rgb(
        &mut self,
        bgr: &[u8],
        src_w: u32,
        src_h: u32,
        stream_raw: ffi::cudaStream_t,
    ) -> Result<()> {
        let bgra_host = bgr_to_bgra(bgr, src_w as usize, src_h as usize);
        let bytes = bgra_host.len();
        if self.d_bgra_in.as_ref().map(|b| b.bytes).unwrap_or(0) < bytes {
            self.d_bgra_in = Some(DevicePtr::alloc(bytes, "d_bgra_in")?);
        }
        let d_bgra = self.d_bgra_in.as_ref().unwrap();
        memcpy_async(
            d_bgra.as_ptr(),
            bgra_host.as_ptr() as *const _,
            bytes,
            MEMCPY_H2D,
            stream_raw,
            "upload bgra",
        )?;
        launch_bgra_to_rgb_nchw(
            d_bgra.as_ptr() as *const u8,
            src_w as i32,
            src_h as i32,
            (src_w * 4) as i32,
            self.d_rgb_nchw.as_ptr() as *mut f32,
            MODEL_W as i32,
            MODEL_H as i32,
            self.runner.stream(),
        )
    }

    fn readback_bgr(&self, bgr_out: &mut [u8], dst_w: u32, dst_h: u32, stream: &CudaStream) -> Result<()> {
        let mut host_bgra = vec![0u8; MODEL_W * MODEL_H * 4];
        memcpy_async(
            host_bgra.as_mut_ptr() as *mut _,
            self.d_bgra_out.as_ptr(),
            host_bgra.len(),
            MEMCPY_D2H,
            stream.raw(),
            "readback bgra",
        )?;
        stream.synchronize()?;

        if dst_w == MODEL_W as u32 && dst_h == MODEL_H as u32 {
            bgra_to_bgr(&host_bgra, bgr_out);
            return Ok(());
        }

        let mut model_bgr = vec![0u8; MODEL_W * MODEL_H * 3];
        bgra_to_bgr(&host_bgra, &mut model_bgr);
        for y in 0..dst_h {
            let sy = ((y * MODEL_H as u32) / dst_h).min(MODEL_H as u32 - 1);
            for x in 0..dst_w {
                let sx = ((x * MODEL_W as u32) / dst_w).min(MODEL_W as u32 - 1);
                let src_idx = ((sy as usize * MODEL_W + sx as usize) * 3) as usize;
                let dst_idx = ((y as usize * dst_w as usize + x as usize) * 3) as usize;
                bgr_out[dst_idx..dst_idx + 3].copy_from_slice(&model_bgr[src_idx..src_idx + 3]);
            }
        }
        Ok(())
    }
}

fn bgr_to_bgra(bgr: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut bgra = vec![0u8; w * h * 4];
    for i in 0..w * h {
        bgra[i * 4] = bgr[i * 3];
        bgra[i * 4 + 1] = bgr[i * 3 + 1];
        bgra[i * 4 + 2] = bgr[i * 3 + 2];
        bgra[i * 4 + 3] = 255;
    }
    bgra
}

fn bgra_to_bgr(bgra: &[u8], bgr: &mut [u8]) {
    for i in 0..bgr.len() / 3 {
        bgr[i * 3] = bgra[i * 4];
        bgr[i * 3 + 1] = bgra[i * 4 + 1];
        bgr[i * 3 + 2] = bgra[i * 4 + 2];
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
    let mut alpha = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let px = rgba.get_pixel(x, y);
            let a = if px[3] > 0 {
                px[3] as f32 / 255.0
            } else {
                px[0].max(px[1]).max(px[2]) as f32 / 255.0
            };
            alpha[(y * w + x) as usize] = a;
        }
    }
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

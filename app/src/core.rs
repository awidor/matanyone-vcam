use std::path::Path;

use anyhow::Result;

pub use matanyone_core::{Frame, MODEL_H, MODEL_W, FRAME_BYTES, center_mask};

pub struct Session {
    inner: matanyone_core::Session,
    frame_index: u64,
}

impl Session {
    pub fn new(engine_dir: &Path) -> Result<Self> {
        Ok(Self {
            inner: matanyone_core::Session::new(engine_dir)?,
            frame_index: 0,
        })
    }

    pub fn init_from_mask_file(&mut self, frame: Frame, mask_path: &Path) -> Result<()> {
        self.inner.init_from_mask_file(frame, mask_path)?;
        self.frame_index = 1;
        Ok(())
    }

    pub fn init_center_mask(&mut self, frame: Frame) -> Result<()> {
        let mask = center_mask(MODEL_W, MODEL_H);
        self.inner.init_from_mask(frame, &mask, MODEL_W, MODEL_H)?;
        self.frame_index = 1;
        Ok(())
    }

    pub fn process(&mut self, frame: Frame, bgr_out: &mut [u8], bg: (f32, f32, f32)) -> Result<()> {
        let memory_update = self.frame_index > 0 && self.frame_index % 5 == 0;
        self.inner
            .process(frame, bgr_out, MODEL_W as u32, MODEL_H as u32, bg, memory_update)?;
        self.frame_index += 1;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.inner.reset()?;
        self.frame_index = 0;
        Ok(())
    }
}

pub fn bgr_to_rgb(bgr: &[u8], rgb: &mut [u8]) {
    for (src, dst) in bgr.chunks_exact(3).zip(rgb.chunks_exact_mut(3)) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
    }
}

pub fn resize_bgr_nearest(src: &[u8], src_w: u32, src_h: u32, dst: &mut [u8], dst_w: u32, dst_h: u32) {
    for y in 0..dst_h {
        let sy = (y * src_h / dst_h).min(src_h.saturating_sub(1));
        for x in 0..dst_w {
            let sx = (x * src_w / dst_w).min(src_w.saturating_sub(1));
            let src_idx = ((sy * src_w + sx) * 3) as usize;
            let dst_idx = ((y * dst_w + x) * 3) as usize;
            dst[dst_idx..dst_idx + 3].copy_from_slice(&src[src_idx..src_idx + 3]);
        }
    }
}

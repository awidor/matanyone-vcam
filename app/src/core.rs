use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uchar};
use std::path::Path;

use anyhow::{Context, Result, bail};

pub const MODEL_W: usize = 1280;
pub const MODEL_H: usize = 720;
pub const FRAME_BYTES: usize = MODEL_W * MODEL_H * 3;

#[repr(C)]
pub struct MatAnyoneSession {
    _private: [u8; 0],
}

extern "C" {
    fn matanyone_create(engine_dir: *const c_char) -> *mut MatAnyoneSession;
    fn matanyone_destroy(session: *mut MatAnyoneSession);
    fn matanyone_last_error() -> *const c_char;
    fn matanyone_internal_width() -> c_int;
    fn matanyone_internal_height() -> c_int;
    fn matanyone_init_from_mask_file(
        session: *mut MatAnyoneSession,
        bgr: *const c_uchar,
        src_w: c_int,
        src_h: c_int,
        mask_png_path: *const c_char,
    ) -> c_int;
    fn matanyone_init_from_mask(
        session: *mut MatAnyoneSession,
        bgr: *const c_uchar,
        src_w: c_int,
        src_h: c_int,
        mask_hw: *const c_float,
        mask_w: c_int,
        mask_h: c_int,
    ) -> c_int;
    fn matanyone_process_bgr(
        session: *mut MatAnyoneSession,
        bgr_in: *const c_uchar,
        src_w: c_int,
        src_h: c_int,
        bgr_out: *mut c_uchar,
        dst_w: c_int,
        dst_h: c_int,
        bg_r: c_float,
        bg_g: c_float,
        bg_b: c_float,
        memory_update: c_int,
    ) -> c_int;
    fn matanyone_reset(session: *mut MatAnyoneSession) -> c_int;
}

fn last_error() -> String {
    unsafe {
        let ptr = matanyone_last_error();
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

pub struct Session {
    ptr: *mut MatAnyoneSession,
    frame_index: u64,
}

impl Session {
    pub fn new(engine_dir: &Path) -> Result<Self> {
        let engine = CString::new(engine_dir.to_string_lossy().as_bytes())
            .context("invalid engine directory")?;
        let ptr = unsafe { matanyone_create(engine.as_ptr()) };
        if ptr.is_null() {
            bail!("matanyone_create failed: {}", last_error());
        }
        Ok(Self {
            ptr,
            frame_index: 0,
        })
    }

    pub fn init_from_mask_file(&mut self, bgr: &[u8], src_w: u32, src_h: u32, mask_path: &Path) -> Result<()> {
        let mask = CString::new(mask_path.to_string_lossy().as_bytes()).context("invalid mask path")?;
        let rc = unsafe {
            matanyone_init_from_mask_file(
                self.ptr,
                bgr.as_ptr(),
                src_w as c_int,
                src_h as c_int,
                mask.as_ptr(),
            )
        };
        if rc != 0 {
            bail!("matanyone_init_from_mask_file failed: {}", last_error());
        }
        self.frame_index = 1;
        Ok(())
    }

    pub fn init_center_mask(&mut self, bgr: &[u8], src_w: u32, src_h: u32) -> Result<()> {
        let mask = center_mask(MODEL_W, MODEL_H);
        let rc = unsafe {
            matanyone_init_from_mask(
                self.ptr,
                bgr.as_ptr(),
                src_w as c_int,
                src_h as c_int,
                mask.as_ptr(),
                MODEL_W as c_int,
                MODEL_H as c_int,
            )
        };
        if rc != 0 {
            bail!("matanyone_init_from_mask failed: {}", last_error());
        }
        self.frame_index = 1;
        Ok(())
    }

    pub fn process_bgr(
        &mut self,
        bgr_in: &[u8],
        src_w: u32,
        src_h: u32,
        bgr_out: &mut [u8],
        bg: (f32, f32, f32),
    ) -> Result<()> {
        let memory_update = if self.frame_index > 0 && self.frame_index % 5 == 0 {
            1
        } else {
            0
        };
        let rc = unsafe {
            matanyone_process_bgr(
                self.ptr,
                bgr_in.as_ptr(),
                src_w as c_int,
                src_h as c_int,
                bgr_out.as_mut_ptr(),
                MODEL_W as c_int,
                MODEL_H as c_int,
                bg.0,
                bg.1,
                bg.2,
                memory_update,
            )
        };
        if rc != 0 {
            bail!("matanyone_process_bgr failed: {}", last_error());
        }
        self.frame_index += 1;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        let rc = unsafe { matanyone_reset(self.ptr) };
        if rc != 0 {
            bail!("matanyone_reset failed: {}", last_error());
        }
        self.frame_index = 0;
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            matanyone_destroy(self.ptr);
        }
    }
}

unsafe impl Send for Session {}

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

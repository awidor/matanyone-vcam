use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::ImageBuffer;
use matanyone_core::Sam31Predictor;

use crate::core::{MODEL_H, MODEL_W};

pub fn mask_from_sam_clicks(
    image_bgr: &[u8],
    fg_points: &[[i32; 2]],
    bg_points: &[[i32; 2]],
    repo_root: &Path,
) -> Result<PathBuf> {
    let engine_dir = repo_root.join("engines/sam31");
    let predictor = Sam31Predictor::from_engine_dir(&engine_dir)?;
    let mask = predictor.predict(fg_points, bg_points, image_bgr)?;
    if mask.len() != MODEL_W * MODEL_H {
        bail!("unexpected SAM mask size {}", mask.len());
    }
    let out = std::env::temp_dir().join(format!("matanyone_sam_mask_{}.png", std::process::id()));
    let image: image::GrayImage =
        ImageBuffer::from_raw(MODEL_W as u32, MODEL_H as u32, mask)
        .context("failed to build mask image")?;
    image.save(&out).context("failed to write SAM mask png")?;
    Ok(out)
}

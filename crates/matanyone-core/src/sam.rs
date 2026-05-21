//! SAM 3.1 click-to-mask predictor.
//!
//! When TensorRT engines exist under `engines/sam31/`, uses in-process inference.
//! Otherwise falls back to the Python worker when built with `sam-python` feature.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::session::{MODEL_H, MODEL_W};

pub struct Sam31Predictor {
    engine_dir: PathBuf,
    #[cfg(feature = "sam-python")]
    python_fallback: bool,
}

impl Sam31Predictor {
    pub fn from_engine_dir(path: &Path) -> Result<Self> {
        let engine_dir = path.to_path_buf();
        let has_engines = engine_dir.join("sam31_image_encoder.engine").exists()
            || engine_dir.join("sam31_decoder.engine").exists();

        if !has_engines {
            #[cfg(feature = "sam-python")]
            {
                return Ok(Self {
                    engine_dir,
                    python_fallback: true,
                });
            }
            #[cfg(not(feature = "sam-python"))]
            {
                bail!(
                    "SAM31 engines not found in {}. Run scripts/export_sam31_engines.py or build with --features sam-python",
                    engine_dir.display()
                );
            }
        }

        Ok(Self {
            engine_dir,
            #[cfg(feature = "sam-python")]
            python_fallback: false,
        })
    }

    pub fn predict(&self, fg_points: &[[i32; 2]], bg_points: &[[i32; 2]], image_bgr: &[u8]) -> Result<Vec<u8>> {
        let has_trt = self.engine_dir.join("sam31_decoder.engine").exists();
        if has_trt {
            return self.predict_trt(fg_points, bg_points, image_bgr);
        }

        #[cfg(feature = "sam-python")]
        if self.python_fallback {
            return predict_python(image_bgr, fg_points, bg_points);
        }

        bail!("SAM31 inference unavailable")
    }

    fn predict_trt(
        &self,
        _fg_points: &[[i32; 2]],
        _bg_points: &[[i32; 2]],
        _image_bgr: &[u8],
    ) -> Result<Vec<u8>> {
        // TRT SAM engines require export via scripts/export_sam31_engines.py.
        // Placeholder until engines are exported and decoder bindings are wired.
        bail!(
            "SAM31 TRT engines found in {} but Rust decoder is not yet wired. Export with scripts/export_sam31_engines.py and complete Sam31Predictor TRT path.",
            self.engine_dir.display()
        )
    }
}

#[cfg(feature = "sam-python")]
fn predict_python(image_bgr: &[u8], fg_points: &[[i32; 2]], bg_points: &[[i32; 2]]) -> Result<Vec<u8>> {
    sam_python::mask_from_clicks(image_bgr, fg_points, bg_points)
}

#[cfg(feature = "sam-python")]
mod sam_python {
    use std::io::{BufRead, BufReader, Write};
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use anyhow::{Context, Result, bail};
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD as BASE64;
    use image::ImageBuffer;
    use serde_json::json;

    use super::{MODEL_H, MODEL_W};

    pub fn mask_from_clicks(
        image_bgr: &[u8],
        fg_points: &[[i32; 2]],
        bg_points: &[[i32; 2]],
    ) -> Result<Vec<u8>> {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let worker = repo_root.join("scripts/sam31_mask_worker.py");
        let temp_dir = tempfile::tempdir().context("SAM temp dir")?;
        let frame_path = temp_dir.path().join("frame.png");
        save_bgr_png(image_bgr, &frame_path)?;

        let python = std::env::var("SAM31_PYTHON").unwrap_or_else(|_| "python".to_string());
        let mut child = Command::new(&python)
            .arg(&worker)
            .arg("--image")
            .arg(&frame_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(&repo_root)
            .spawn()
            .with_context(|| format!("start SAM worker via {python}"))?;

        let mut stdin = child.stdin.take().context("SAM stdin")?;
        let stdout = child.stdout.take().context("SAM stdout")?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        let read = || -> Result<serde_json::Value> {
            let line = rx
                .recv_timeout(Duration::from_secs(180))
                .context("SAM timeout")?;
            serde_json::from_str(&line).context("SAM JSON")
        };

        let ready = read()?;
        if ready.get("status").and_then(|v| v.as_str()) != Some("ready") {
            bail!("SAM worker failed to start");
        }

        writeln!(stdin, "{}", json!({ "fg": fg_points, "bg": bg_points }))?;
        stdin.flush()?;
        let response = read()?;
        if response.get("status").and_then(|v| v.as_str()) != Some("ok") {
            bail!("SAM prompt failed");
        }
        let encoded = response
            .get("mask")
            .and_then(|v| v.as_str())
            .context("SAM mask missing")?;
        BASE64.decode(encoded).context("SAM mask decode")
    }

    fn save_bgr_png(bgr: &[u8], path: &std::path::Path) -> Result<()> {
        let mut rgb = vec![0u8; MODEL_W * MODEL_H * 3];
        for (src, dst) in bgr.chunks_exact(3).zip(rgb.chunks_exact_mut(3)) {
            dst[0] = src[2];
            dst[1] = src[1];
            dst[2] = src[0];
        }
        let image: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(MODEL_W as u32, MODEL_H as u32, rgb)
                .context("invalid frame")?;
        image.save(path).context("write frame png")?;
        Ok(())
    }
}

#[cfg(feature = "sam-python")]
use tempfile;

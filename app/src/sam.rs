use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::ImageBuffer;
use serde_json::json;

use crate::core::{MODEL_H, MODEL_W};

pub struct Sam31Client {
    child: Child,
    stdin: std::process::ChildStdin,
    rx: Receiver<String>,
    _temp_dir: tempfile::TempDir,
}

impl Sam31Client {
    pub fn start(image_bgr: &[u8], repo_root: &Path) -> Result<Self> {
        let worker = repo_root.join("scripts").join("sam31_mask_worker.py");
        if !worker.exists() {
            bail!("SAM worker not found at {}", worker.display());
        }

        let temp_dir = tempfile::Builder::new()
            .prefix("matanyone_sam31_")
            .tempdir()
            .context("failed to create SAM temp dir")?;
        let frame_path = temp_dir.path().join("frame.png");
        save_bgr_png(image_bgr, MODEL_W, MODEL_H, &frame_path)?;

        let python = std::env::var("SAM31_PYTHON").unwrap_or_else(|_| "python".to_string());
        let mut child = Command::new(&python)
            .arg(&worker)
            .arg("--image")
            .arg(&frame_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(repo_root)
            .spawn()
            .with_context(|| format!("failed to start SAM worker via {python}"))?;

        let mut stdin = child.stdin.take().context("SAM worker stdin unavailable")?;
        let stdout = child.stdout.take().context("SAM worker stdout unavailable")?;
        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        let mut client = Self {
            child,
            stdin,
            rx,
            _temp_dir: temp_dir,
        };
        let ready = client.read_response()?;
        if ready.get("status").and_then(|v| v.as_str()) != Some("ready") {
            bail!(
                "SAM worker failed to start: {}",
                ready.get("error").and_then(|v| v.as_str()).unwrap_or("unknown")
            );
        }
        Ok(client)
    }

    fn read_response(&self) -> Result<serde_json::Value> {
        let line = self
            .rx
            .recv_timeout(Duration::from_secs(180))
            .context("timed out waiting for SAM worker")?;
        serde_json::from_str(&line).context("invalid SAM JSON")
    }

    pub fn predict(&mut self, fg_points: &[[i32; 2]], bg_points: &[[i32; 2]]) -> Result<Vec<u8>> {
        let payload = json!({
            "fg": fg_points,
            "bg": bg_points,
        });
        writeln!(self.stdin, "{}", payload).context("SAM prompt write failed")?;
        self.stdin.flush().ok();
        let response = self.read_response()?;
        if response.get("status").and_then(|v| v.as_str()) != Some("ok") {
            bail!(
                "SAM prompt failed: {}",
                response.get("error").and_then(|v| v.as_str()).unwrap_or("unknown")
            );
        }
        let encoded = response
            .get("mask")
            .and_then(|v| v.as_str())
            .context("SAM response missing mask")?;
        BASE64
            .decode(encoded)
            .context("SAM mask decode failed")
    }
}

impl Drop for Sam31Client {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "{}", json!({"quit": true}));
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

fn save_bgr_png(bgr: &[u8], width: usize, height: usize, path: &Path) -> Result<()> {
    let mut rgb = vec![0u8; width * height * 3];
    for (src, dst) in bgr.chunks_exact(3).zip(rgb.chunks_exact_mut(3)) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
    }
    let image: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width as u32, height as u32, rgb).context("invalid frame dimensions")?;
    image.save(path).context("failed to write SAM frame png")?;
    Ok(())
}

pub fn mask_from_sam_clicks(
    image_bgr: &[u8],
    fg_points: &[[i32; 2]],
    bg_points: &[[i32; 2]],
    repo_root: &Path,
) -> Result<PathBuf> {
    let mut client = Sam31Client::start(image_bgr, repo_root)?;
    let mask = client.predict(fg_points, bg_points)?;
    if mask.len() != MODEL_W * MODEL_H {
        bail!("unexpected SAM mask size {}", mask.len());
    }
    let out = std::env::temp_dir().join(format!("matanyone_sam_mask_{}.png", std::process::id()));
    let image = image::GrayImage::from_raw(MODEL_W as u32, MODEL_H as u32, mask)
        .context("failed to build mask image")?;
    image.save(&out).context("failed to write SAM mask png")?;
    Ok(out)
}

use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result, bail};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;

use crate::core::{MODEL_H, MODEL_W, resize_bgr_nearest};
use crate::paths;

#[derive(Clone)]
pub struct CameraEntry {
    pub name: String,
    source: CameraSource,
}

#[derive(Clone)]
enum CameraSource {
    MediaFoundation { index: u32 },
    DirectShow { name: String },
}

enum CameraBackend {
    Native(NativeCapture),
    FfmpegDshow(FfmpegDshowCapture),
}

pub struct CameraCapture {
    backend: CameraBackend,
}

struct NativeCapture {
    camera: Camera,
    scratch: Vec<u8>,
    resized: Vec<u8>,
}

struct FfmpegDshowCapture {
    child: Child,
    frame: Vec<u8>,
}

impl CameraEntry {
    pub fn is_obs_virtual_camera(&self) -> bool {
        self.name.to_ascii_lowercase().contains("obs virtual camera")
    }
}

impl CameraCapture {
    pub fn list_devices() -> Result<Vec<CameraEntry>> {
        let mut entries = list_media_foundation_devices()?;
        if let Some(ffmpeg) = paths::find_ffmpeg() {
            for name in list_dshow_devices(&ffmpeg)? {
                if entries.iter().any(|entry| device_names_match(&entry.name, &name)) {
                    continue;
                }
                entries.push(CameraEntry {
                    name: name.clone(),
                    source: CameraSource::DirectShow { name },
                });
            }
        }
        Ok(entries)
    }

    pub fn open(entry: &CameraEntry) -> Result<Self> {
        let backend = match &entry.source {
            CameraSource::MediaFoundation { index } => {
                CameraBackend::Native(NativeCapture::open(*index)?)
            }
            CameraSource::DirectShow { name } => {
                let ffmpeg = paths::find_ffmpeg().context(
                    "ffmpeg is required for DirectShow cameras such as OBS Virtual Camera; \
                     set FFMPEG_DIR or install ffmpeg on PATH",
                )?;
                CameraBackend::FfmpegDshow(FfmpegDshowCapture::open(&ffmpeg, name)?)
            }
        };
        Ok(Self { backend })
    }

    pub fn capture_bgr(&mut self) -> Result<(&[u8], u32, u32)> {
        match &mut self.backend {
            CameraBackend::Native(native) => native.capture_bgr(),
            CameraBackend::FfmpegDshow(dshow) => dshow.capture_bgr(),
        }
    }
}

impl Drop for CameraCapture {
    fn drop(&mut self) {
        if let CameraBackend::Native(native) = &mut self.backend {
            let _ = native.camera.stop_stream();
        }
        if let CameraBackend::FfmpegDshow(dshow) = &mut self.backend {
            let _ = dshow.child.kill();
            let _ = dshow.child.wait();
        }
    }
}

fn list_media_foundation_devices() -> Result<Vec<CameraEntry>> {
    let backend = ApiBackend::MediaFoundation;
    let devices = nokhwa::query(backend).context("failed to query Media Foundation cameras")?;
    Ok(devices
        .into_iter()
        .enumerate()
        .map(|(index, info)| CameraEntry {
            name: info.human_name().to_string(),
            source: CameraSource::MediaFoundation {
                index: index as u32,
            },
        })
        .collect())
}

fn list_dshow_devices(ffmpeg: &Path) -> Result<Vec<String>> {
    let output = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            "dshow",
            "-i",
            "dummy",
        ])
        .output()
        .with_context(|| format!("failed to run {}", ffmpeg.display()))?;

    let mut devices = Vec::new();
    for line in String::from_utf8_lossy(&output.stderr).lines() {
        if line.contains('"') && line.contains("(video)") {
            if let Some(name) = line.split('"').nth(1) {
                devices.push(name.to_string());
            }
        }
    }
    Ok(devices)
}

fn device_names_match(left: &str, right: &str) -> bool {
    let left = normalize_device_name(left);
    let right = normalize_device_name(right);
    left == right || left.contains(&right) || right.contains(&left)
}

fn normalize_device_name(name: &str) -> String {
    name.to_ascii_lowercase()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect()
}

impl NativeCapture {
    fn open(device_index: u32) -> Result<Self> {
        let index = CameraIndex::Index(device_index);
        let format = CameraFormat::new(
            nokhwa::utils::Resolution::new(MODEL_W as u32, MODEL_H as u32),
            FrameFormat::MJPEG,
            30,
        );
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(format));
        let mut camera = Camera::new(index, requested).context("failed to open camera")?;
        camera.open_stream().context("failed to start camera stream")?;
        Ok(Self {
            camera,
            scratch: Vec::new(),
            resized: vec![0; MODEL_W * MODEL_H * 3],
        })
    }

    fn capture_bgr(&mut self) -> Result<(&[u8], u32, u32)> {
        let frame = self.camera.frame().context("camera frame failed")?;
        let decoded = frame.decode_image::<RgbFormat>().context("decode frame failed")?;
        let width = decoded.width();
        let height = decoded.height();
        let rgb = decoded.into_raw();
        self.scratch.resize(rgb.len(), 0);
        for (src, dst) in rgb.chunks_exact(3).zip(self.scratch.chunks_exact_mut(3)) {
            dst[0] = src[2];
            dst[1] = src[1];
            dst[2] = src[0];
        }
        if width as usize == MODEL_W && height as usize == MODEL_H {
            Ok((&self.scratch, width, height))
        } else {
            resize_bgr_nearest(
                &self.scratch,
                width,
                height,
                &mut self.resized,
                MODEL_W as u32,
                MODEL_H as u32,
            );
            Ok((&self.resized, MODEL_W as u32, MODEL_H as u32))
        }
    }
}

impl FfmpegDshowCapture {
    fn open(ffmpeg: &Path, device_name: &str) -> Result<Self> {
        let frame_bytes = MODEL_W * MODEL_H * 3;
        let mut child = Command::new(ffmpeg)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "dshow",
                "-rtbufsize",
                "200M",
                "-i",
                &format!("video={device_name}"),
                "-vf",
                &format!("scale={MODEL_W}:{MODEL_H}"),
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to start ffmpeg for DirectShow device '{device_name}'"))?;

        let mut frame = vec![0u8; frame_bytes];
        let stdout = child
            .stdout
            .as_mut()
            .context("ffmpeg stdout unavailable")?;
        match stdout.read_exact(&mut frame) {
            Ok(()) => Ok(Self { child, frame }),
            Err(err) => {
                let stderr = child
                    .stderr
                    .as_mut()
                    .and_then(|pipe| {
                        let mut buf = String::new();
                        pipe.read_to_string(&mut buf).ok()?;
                        Some(buf)
                    })
                    .unwrap_or_default();
                let _ = child.kill();
                let _ = child.wait();
                if stderr.trim().is_empty() {
                    bail!("failed to read first frame from '{device_name}': {err}");
                }
                bail!("failed to open DirectShow device '{device_name}': {stderr}");
            }
        }
    }

    fn capture_bgr(&mut self) -> Result<(&[u8], u32, u32)> {
        let stdout = self
            .child
            .stdout
            .as_mut()
            .context("ffmpeg stdout unavailable")?;
        stdout
            .read_exact(&mut self.frame)
            .context("ffmpeg frame read failed")?;
        Ok((&self.frame, MODEL_W as u32, MODEL_H as u32))
    }
}

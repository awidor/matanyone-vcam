use anyhow::{Context, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;

use crate::core::{MODEL_H, MODEL_W, resize_bgr_nearest};

pub struct CameraCapture {
    camera: Camera,
    scratch: Vec<u8>,
    resized: Vec<u8>,
}

impl CameraCapture {
    pub fn list_devices() -> Result<Vec<String>> {
        let backend = ApiBackend::MediaFoundation;
        let devices = nokhwa::query(backend).context("failed to query cameras")?;
        Ok(devices.into_iter().map(|info| info.human_name().to_string()).collect())
    }

    pub fn open(device_index: usize) -> Result<Self> {
        let index = CameraIndex::Index(device_index as u32);
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

    pub fn capture_bgr(&mut self) -> Result<(&[u8], u32, u32)> {
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
            resize_bgr_nearest(&self.scratch, width, height, &mut self.resized, MODEL_W as u32, MODEL_H as u32);
            Ok((&self.resized, MODEL_W as u32, MODEL_H as u32))
        }
    }
}

impl Drop for CameraCapture {
    fn drop(&mut self) {
        let _ = self.camera.stop_stream();
    }
}

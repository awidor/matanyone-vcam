use anyhow::{Context, Result};
use virtualcam::pixel_format::PixelFormat;
use virtualcam::Camera;

use crate::core::{MODEL_H, MODEL_W};

pub struct VirtualCamera {
    camera: Camera,
}

impl VirtualCamera {
    pub fn open() -> Result<Self> {
        let camera = Camera::builder(MODEL_W as u32, MODEL_H as u32, 30.0)
            .format(PixelFormat::BGR)
            .backend("unity")
            .build()
            .or_else(|_| {
                Camera::builder(MODEL_W as u32, MODEL_H as u32, 30.0)
                    .format(PixelFormat::BGR)
                    .build()
            })
            .context("failed to open virtual camera (UnityCapture or default backend)")?;
        Ok(Self { camera })
    }

    pub fn device_name(&self) -> &str {
        self.camera.device()
    }

    pub fn send_bgr(&mut self, frame: &[u8]) -> Result<()> {
        self.camera
            .send(frame)
            .context("virtual camera send failed")?;
        self.camera.sleep_until_next_frame();
        Ok(())
    }
}

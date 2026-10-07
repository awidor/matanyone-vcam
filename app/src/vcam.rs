use anyhow::{Context, Result};
use virtualcam::{PixelFormat, VirtualCamError};

use crate::capture::device_names_match;
use crate::core::{MODEL_H, MODEL_W};
use crate::{obs_vcam, unity_capture};

/// Where the composited frames go.
pub enum VirtualCamera {
    Unity(unity_capture::UnityCapture),
    Obs(virtualcam::Camera),
}

impl VirtualCamera {
    /// Picks an output that is not the input: Unity Video Capture when installed, otherwise
    /// OBS Virtual Camera when OBS is not publishing to it itself. `Err` explains why there is
    /// no output, and the app runs preview-only.
    pub fn open_for_input(input_name: &str) -> std::result::Result<Self, String> {
        let unity_installed = unity_capture::is_installed();
        if unity_installed && !device_names_match(input_name, unity_capture::DEVICE_NAME) {
            return Ok(Self::Unity(unity_capture::UnityCapture::new()));
        }
        if device_names_match(input_name, obs_vcam::DEVICE_NAME) {
            return Err("none (OBS Virtual Camera is the input; install Unity Video Capture to publish)".into());
        }
        match virtualcam::Camera::builder(MODEL_W as u32, MODEL_H as u32, 30.0)
            .format(PixelFormat::BGR)
            .backend("obs")
            .build()
        {
            Ok(camera) => Ok(Self::Obs(camera)),
            Err(VirtualCamError::DeviceInUse(_)) => Err(
                "none (OBS is using its virtual camera; stop it in OBS or install Unity Video Capture)".into(),
            ),
            Err(VirtualCamError::DeviceNotFound(_)) => {
                Err("none (install Unity Video Capture or OBS Studio)".into())
            }
            Err(err) => Err(format!("none ({err})")),
        }
    }

    pub fn device_name(&self) -> &str {
        match self {
            Self::Unity(_) => unity_capture::DEVICE_NAME,
            Self::Obs(_) => obs_vcam::DEVICE_NAME,
        }
    }

    pub fn send_bgr(&mut self, frame: &[u8]) -> Result<()> {
        match self {
            Self::Unity(unity) => unity.send_bgr(frame, MODEL_W as u32, MODEL_H as u32),
            Self::Obs(camera) => camera.send(frame).context("virtual camera send failed"),
        }
    }
}

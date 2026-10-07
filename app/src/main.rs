mod capture;
mod core;
mod paths;
mod sam;
mod vcam;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use eframe::egui;
use parking_lot::Mutex;
use tray_icon::{Icon, TrayIconBuilder, menu::{Menu, MenuItem}};

use crate::capture::{CameraCapture, CameraEntry};
use crate::core::{FRAME_BYTES, MODEL_H, MODEL_W, Session, bgr_to_rgb};
use crate::vcam::VirtualCamera;

#[derive(Parser, Debug)]
#[command(name = "matanyone-vcam", about = "MatAnyone standalone virtual camera")]
struct Args {
    /// TensorRT engine directory (default: `engines/faithful` next to the executable).
    #[arg(long, value_name = "DIR")]
    engine_dir: Option<PathBuf>,

    #[arg(long, value_name = "PNG")]
    mask: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    camera: usize,

    #[arg(long)]
    no_vcam: bool,
}

struct PreviewState {
    texture: Option<egui::TextureHandle>,
    /// Newest worker frame (RGB, model size) not yet uploaded to `texture`.
    pending: Option<Vec<u8>>,
    status: String,
    fps: f32,
    ms_per_frame: f32,
}

impl PreviewState {
    fn new() -> Self {
        Self {
            texture: None,
            pending: None,
            status: "Idle".to_string(),
            fps: 0.0,
            ms_per_frame: 0.0,
        }
    }

    /// Uploads the pending frame, if any; the texture is reused across repaints otherwise.
    fn upload_pending(&mut self, ctx: &egui::Context) {
        let Some(rgb) = self.pending.take() else {
            return;
        };
        let image = egui::ColorImage::from_rgb([MODEL_W, MODEL_H], &rgb);
        match self.texture.as_mut() {
            Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture("preview", image, egui::TextureOptions::LINEAR)),
        }
    }
}

struct WorkerControl {
    stop: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
}

struct FramePoll {
    frame_slot: Arc<Mutex<Option<Vec<u8>>>>,
    status_slot: Arc<Mutex<String>>,
    metrics_slot: Arc<Mutex<(f32, f32)>>,
    vcam_name_slot: Arc<Mutex<String>>,
}

struct AppSettings {
    engine_dir: String,
    mask_path: Option<PathBuf>,
    camera_index: usize,
    enable_vcam: bool,
    bg_color: [u8; 3],
    use_center_mask: bool,
    sam_fg_points: Vec<[i32; 2]>,
    sam_bg_points: Vec<[i32; 2]>,
    sam_mode: bool,
}

struct MatAnyoneApp {
    repo_root: PathBuf,
    settings: AppSettings,
    preview: PreviewState,
    worker: Option<WorkerControl>,
    frame_poll: Option<FramePoll>,
    cameras: Vec<CameraEntry>,
    vcam_device: String,
    tray: Option<tray_icon::TrayIcon>,
    show_window: bool,
}

impl MatAnyoneApp {
    fn new(args: Args, _install_root: PathBuf, repo_root: PathBuf, engine_dir: PathBuf) -> Result<Self> {
        let cameras = CameraCapture::list_devices().unwrap_or_default();
        let menu = Menu::with_items(&[
            &MenuItem::with_id("show", "Show MatAnyone", true, None),
            &MenuItem::with_id("quit", "Quit", true, None),
        ])?;
        let icon = Icon::from_rgba(vec![0, 180, 80, 255], 1, 1)?;
        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("MatAnyone Virtual Camera")
            .with_icon(icon)
            .build()
            .ok();

        let use_center_mask = args.mask.is_none();
        Ok(Self {
            settings: AppSettings {
                engine_dir: engine_dir.to_string_lossy().into_owned(),
                mask_path: args.mask,
                camera_index: args.camera,
                enable_vcam: !args.no_vcam,
                bg_color: [0, 180, 80],
                use_center_mask,
                sam_fg_points: Vec::new(),
                sam_bg_points: Vec::new(),
                sam_mode: false,
            },
            preview: PreviewState::new(),
            worker: None,
            frame_poll: None,
            cameras,
            vcam_device: String::new(),
            tray,
            repo_root,
            show_window: true,
        })
    }

    fn start_worker(&mut self) {
        if self.worker.as_ref().is_some_and(|w| w.running.load(Ordering::SeqCst)) {
            return;
        }

        let Some(camera_entry) = self.cameras.get(self.settings.camera_index).cloned() else {
            self.preview.status = "Select a camera first.".to_string();
            return;
        };
        let enable_vcam = self.settings.enable_vcam;
        if enable_vcam && camera_entry.is_obs_virtual_camera() {
            self.preview.status = "OBS Virtual Camera cannot be both input and output. \
                Disable virtual-camera output or choose a different input."
                .to_string();
            return;
        }

        let stop = Arc::new(AtomicBool::new(false));
        let running = Arc::new(AtomicBool::new(true));
        self.worker = Some(WorkerControl {
            stop: Arc::clone(&stop),
            running: Arc::clone(&running),
        });

        let engine_dir = PathBuf::from(self.settings.engine_dir.clone());
        let mask_path = self.settings.mask_path.clone();
        let camera_name = camera_entry.name.clone();
        let bg = self.settings.bg_color;
        let use_center_mask = self.settings.use_center_mask || mask_path.is_none();
        let sam_fg = self.settings.sam_fg_points.clone();
        let sam_bg = self.settings.sam_bg_points.clone();
        let repo_root = self.repo_root.clone();

        let frame_slot = Arc::new(Mutex::new(None::<Vec<u8>>));
        let status_slot = Arc::new(Mutex::new("Starting...".to_string()));
        let metrics_slot = Arc::new(Mutex::new((0.0f32, 0.0f32)));
        let vcam_name_slot = Arc::new(Mutex::new(String::new()));

        self.frame_poll = Some(FramePoll {
            frame_slot: Arc::clone(&frame_slot),
            status_slot: Arc::clone(&status_slot),
            metrics_slot: Arc::clone(&metrics_slot),
            vcam_name_slot: Arc::clone(&vcam_name_slot),
        });

        thread::spawn(move || {
            let worker_result = (|| -> Result<()> {
                let mut session = Session::new(&engine_dir)?;
                let mut camera = CameraCapture::open(&camera_entry)?;
                let mut out = vec![0u8; FRAME_BYTES];
                let mut rgb_preview = vec![0u8; FRAME_BYTES];
                let mut initialized = false;
                let mut frame_index = 0u64;
                let mut vcam = if enable_vcam {
                    Some(VirtualCamera::open()?)
                } else {
                    None
                };

                if let Some(vcam) = vcam.as_ref() {
                    *vcam_name_slot.lock() = vcam.device_name().to_string();
                }

                let bg_norm = (
                    bg[2] as f32 / 255.0,
                    bg[1] as f32 / 255.0,
                    bg[0] as f32 / 255.0,
                );

                while !stop.load(Ordering::SeqCst) {
                    let start = Instant::now();
                    let (bgr, w, h) = camera.capture_bgr()?;

                    if !initialized {
                        if !sam_fg.is_empty() {
                            let mask_file = sam::mask_from_sam_clicks(bgr, &sam_fg, &sam_bg, &repo_root)?;
                            session.init_from_mask_file(bgr, w, h, &mask_file)?;
                        } else if let Some(mask) = mask_path.as_ref() {
                            session.init_from_mask_file(bgr, w, h, mask)?;
                        } else if use_center_mask {
                            session.init_center_mask(bgr, w, h)?;
                        } else {
                            anyhow::bail!("no mask configured");
                        }
                        initialized = true;
                        frame_index = 1;
                        bgr_to_rgb(bgr, &mut rgb_preview);
                        *frame_slot.lock() = Some(rgb_preview.clone());
                    } else {
                        session.process_bgr(bgr, w, h, &mut out, bg_norm)?;
                        bgr_to_rgb(&out, &mut rgb_preview);
                        *frame_slot.lock() = Some(rgb_preview.clone());
                        if let Some(vcam) = vcam.as_mut() {
                            vcam.send_bgr(&out)?;
                        }
                        frame_index += 1;
                    }

                    let elapsed = start.elapsed();
                    let ms = elapsed.as_secs_f32() * 1000.0;
                    let fps = if ms > 0.0 { 1000.0 / ms } else { 0.0 };
                    *metrics_slot.lock() = (fps, ms);
                    *status_slot.lock() = format!(
                        "Running frame={frame_index} camera={camera_name} vcam={}",
                        if enable_vcam { "on" } else { "off" }
                    );

                    let target = Duration::from_secs_f64(1.0 / 30.0);
                    if elapsed < target {
                        thread::sleep(target - elapsed);
                    }
                }
                Ok(())
            })();

            if let Err(err) = worker_result {
                *status_slot.lock() = format!("Error: {err:#}");
            } else {
                *status_slot.lock() = "Stopped".to_string();
            }
            running.store(false, Ordering::SeqCst);
        });

        self.preview.status = "Starting...".to_string();
    }

    fn stop_worker(&mut self) {
        if let Some(worker) = &self.worker {
            worker.stop.store(true, Ordering::SeqCst);
        }
        self.preview.status = "Stopping...".to_string();
    }

    fn poll_worker(&mut self) {
        let Some(poll) = &self.frame_poll else {
            return;
        };
        if let Some(frame) = poll.frame_slot.lock().take() {
            self.preview.pending = Some(frame);
        }
        self.preview.status = poll.status_slot.lock().clone();
        let (fps, ms) = *poll.metrics_slot.lock();
        self.preview.fps = fps;
        self.preview.ms_per_frame = ms;
        let vcam_name = poll.vcam_name_slot.lock().clone();
        if !vcam_name.is_empty() {
            self.vcam_device = vcam_name;
        }
        if self.worker.as_ref().is_some_and(|w| !w.running.load(Ordering::SeqCst)) {
            self.frame_poll = None;
            self.worker = None;
        }
    }
}

impl eframe::App for MatAnyoneApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();

        if !self.show_window {
            ctx.request_repaint_after(Duration::from_millis(200));
            return;
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("MatAnyone Virtual Camera");
                ui.separator();
                if ui.button("Start").clicked() {
                    self.start_worker();
                }
                if ui.button("Stop").clicked() {
                    self.stop_worker();
                }
                if ui.button("Hide to tray").clicked() {
                    self.show_window = false;
                }
            });
            ui.label(&self.preview.status);
            if !self.vcam_device.is_empty() {
                ui.label(format!("Virtual camera device: {}", self.vcam_device));
            }
            ui.label(format!(
                "Timing: {:.1} ms/frame ({:.1} fps)",
                self.preview.ms_per_frame, self.preview.fps
            ));
        });

        egui::SidePanel::left("settings").min_width(280.0).show(ctx, |ui| {
            ui.heading("Settings");
            ui.label("Engine directory");
            ui.text_edit_singleline(&mut self.settings.engine_dir);
            ui.label("Initial mask PNG (optional)");
            let mut mask_text = self
                .settings
                .mask_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            if ui.text_edit_singleline(&mut mask_text).changed() {
                self.settings.mask_path = if mask_text.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(mask_text))
                };
            }
            ui.checkbox(&mut self.settings.use_center_mask, "Use center placeholder mask");
            ui.label("Background color (BGR)");
            ui.color_edit_button_srgb(&mut self.settings.bg_color);

            ui.separator();
            ui.label("Input camera");
            egui::ComboBox::from_id_salt("camera")
                .selected_text(
                    self.cameras
                        .get(self.settings.camera_index)
                        .map(|entry| entry.name.as_str())
                        .unwrap_or("Select a camera"),
                )
                .show_ui(ui, |ui| {
                    for (idx, entry) in self.cameras.iter().enumerate() {
                        ui.selectable_value(&mut self.settings.camera_index, idx, &entry.name);
                    }
                });
            if ui.button("Refresh cameras").clicked() {
                self.cameras = CameraCapture::list_devices().unwrap_or_default();
                if self.settings.camera_index >= self.cameras.len() && !self.cameras.is_empty() {
                    self.settings.camera_index = 0;
                }
            }
            if self.cameras.is_empty() {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "No cameras found. Start OBS Virtual Camera or connect a webcam, then refresh.",
                );
            }
            ui.label("DirectShow devices (e.g. OBS Virtual Camera) require ffmpeg on PATH or FFMPEG_DIR.");
            ui.checkbox(
                &mut self.settings.enable_vcam,
                "Publish to virtual camera (turn off when using OBS Virtual Camera as input)",
            );

            ui.separator();
            ui.checkbox(&mut self.settings.sam_mode, "SAM3.1 click mode (first frame)");
            if self.settings.sam_mode {
                ui.label("Left-click preview: foreground. Right-click: background.");
                if ui.button("Clear SAM points").clicked() {
                    self.settings.sam_fg_points.clear();
                    self.settings.sam_bg_points.clear();
                }
                ui.label(format!(
                    "FG points: {}  BG points: {}",
                    self.settings.sam_fg_points.len(),
                    self.settings.sam_bg_points.len()
                ));
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.preview.upload_pending(ctx);
            let Some(texture) = self.preview.texture.as_ref() else {
                ui.centered_and_justified(|ui| {
                    ui.label("Preview will appear after processing starts.");
                });
                return;
            };
            let response = ui.add(
                egui::Image::from_texture(texture)
                    .fit_to_exact_size(egui::vec2(960.0, 540.0))
                    .sense(egui::Sense::click()),
            );

            // `clicked()` is the primary button only; right clicks add background points.
            if self.settings.sam_mode && (response.clicked() || response.secondary_clicked()) {
                if let Some(pos) = response.interact_pointer_pos() {
                    let rect = response.rect;
                    let x = ((pos.x - rect.min.x) / rect.width() * MODEL_W as f32) as i32;
                    let y = ((pos.y - rect.min.y) / rect.height() * MODEL_H as f32) as i32;
                    let point = [
                        x.clamp(0, MODEL_W as i32 - 1),
                        y.clamp(0, MODEL_H as i32 - 1),
                    ];
                    if response.secondary_clicked() {
                        self.settings.sam_bg_points.push(point);
                    } else {
                        self.settings.sam_fg_points.push(point);
                    }
                }
            }
        });

        ctx.request_repaint_after(Duration::from_millis(33));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop_worker();
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let install_root = paths::install_root();
    let engine_dir = args
        .engine_dir
        .map(|path| paths::resolve_against(&install_root, path))
        .unwrap_or_else(|| paths::default_engine_dir(&install_root));
    let repo_root = paths::bundle_root(&install_root);
    let mask = args.mask.map(|path| paths::resolve_against(&install_root, path));
    let app_args = Args {
        engine_dir: None,
        mask,
        camera: args.camera,
        no_vcam: args.no_vcam,
    };

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "MatAnyone Virtual Camera",
        native_options,
        Box::new(move |_cc| {
            let app =
                MatAnyoneApp::new(app_args, install_root, repo_root, engine_dir).context("failed to initialize app")?;
            Ok(Box::new(app) as Box<dyn eframe::App>)
        }),
    )
    .map_err(|err| anyhow::anyhow!("eframe exited with error: {err}"))?;

    Ok(())
}

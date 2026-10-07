//! End-to-end `Session` smoke test without a camera.
//!
//! Usage:
//!   matanyone_core_smoke --synthetic [--frames=N] [--interval-ms=MS] [engine_dir]
//!
//! `--interval-ms` paces frames like a camera (e.g. 33), so the timing shows per-frame latency
//! rather than back-to-back throughput.
//!   matanyone_core_smoke [engine_dir] [frames_dir] [mask_png] [output_png] [max_frames]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use matanyone_core::{center_mask, make_synthetic_bgr, mean, percentile, Frame, Session, FRAME_BYTES, MODEL_H, MODEL_W};

const BG: (f32, f32, f32) = (0.0, 180.0 / 255.0, 80.0 / 255.0);

fn read_bgr(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let rgb = image::open(path)
        .with_context(|| format!("failed to load image: {}", path.display()))?
        .to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut bgr = rgb.into_raw();
    for px in bgr.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
    Ok((bgr, w, h))
}

fn write_png(path: &Path, bgr: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut rgb = bgr.to_vec();
    for px in rgb.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
    image::RgbImage::from_raw(MODEL_W as u32, MODEL_H as u32, rgb)
        .context("failed to build output image")?
        .save(path)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn frame_path(dir: &Path, index: usize) -> Option<PathBuf> {
    ["png", "jpg"]
        .iter()
        .map(|ext| dir.join(format!("{index:05}.{ext}")))
        .find(|p| p.exists())
}

fn run_synthetic(session: &mut Session, output: &Path, max_frames: usize, interval: Duration) -> Result<()> {
    let (w, h) = (MODEL_W as u32, MODEL_H as u32);
    let first = make_synthetic_bgr(MODEL_W, MODEL_H, 0);
    session.init_from_mask(Frame::bgr(&first, w, h), &center_mask(MODEL_W, MODEL_H), MODEL_W, MODEL_H)?;

    // A few distinct frames, reused, so frame generation stays out of the timing.
    let inputs: Vec<Vec<u8>> = (1..=8).map(|i| make_synthetic_bgr(MODEL_W, MODEL_H, i)).collect();
    let mut out = vec![0u8; FRAME_BYTES];
    let mut timings = Vec::with_capacity(max_frames);
    let mut next_frame = Instant::now();
    for i in 1..max_frames {
        std::thread::sleep(next_frame.saturating_duration_since(Instant::now()));
        next_frame += interval;
        let start = Instant::now();
        session
            .process(Frame::bgr(&inputs[i % inputs.len()], w, h), &mut out, w, h, BG, i % 5 == 0)
            .with_context(|| format!("process failed frame {i}"))?;
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    write_png(output, &out)?;
    println!("wrote={} mode=synthetic frames={max_frames}", output.display());
    // Skip warm-up frames when there are enough to time.
    let steady = if timings.len() > 20 { &timings[10..] } else { &timings[..] };
    println!("session_mean_ms={}", mean(steady));
    println!("session_p99_ms={}", percentile(steady.to_vec(), 99.0));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let synthetic = args.iter().any(|a| a == "--synthetic");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    let flag = |name: &str| -> Result<Option<u64>> {
        Ok(args.iter().find_map(|a| a.strip_prefix(name)).map(|n| n.parse()).transpose()?)
    };
    let synthetic_frames = flag("--frames=")?.map(|n| n as usize);
    let interval = Duration::from_millis(flag("--interval-ms=")?.unwrap_or(0));

    let engine_dir = PathBuf::from(positional.first().map_or("engines/faithful", |s| s.as_str()));
    let input_dir = PathBuf::from(positional.get(1).map_or("raw_sample/frames", |s| s.as_str()));
    let mask_path = PathBuf::from(positional.get(2).map_or("raw_sample/mask.png", |s| s.as_str()));
    let output = PathBuf::from(positional.get(3).map_or("output/core_smoke.png", |s| s.as_str()));
    let max_frames: usize = positional.get(4).map(|s| s.parse()).transpose()?.unwrap_or(5);

    let mut session = Session::new(&engine_dir)?;
    if synthetic || !input_dir.exists() {
        return run_synthetic(&mut session, &output, synthetic_frames.unwrap_or(max_frames), interval);
    }

    let first_path = frame_path(&input_dir, 0).context("missing first frame 00000.png/.jpg")?;
    let (first, w, h) = read_bgr(&first_path)?;
    session.init_from_mask_file(Frame::bgr(&first, w, h), &mask_path)?;

    let mut out = vec![0u8; FRAME_BYTES];
    for i in 0..max_frames {
        let Some(path) = frame_path(&input_dir, i) else {
            break;
        };
        let (bgr, w, h) = read_bgr(&path)?;
        session
            .process(Frame::bgr(&bgr, w, h), &mut out, MODEL_W as u32, MODEL_H as u32, BG, i > 0 && i % 5 == 0)
            .with_context(|| format!("process failed frame {i}"))?;
    }
    write_png(&output, &out)?;
    println!("wrote={}", output.display());
    Ok(())
}

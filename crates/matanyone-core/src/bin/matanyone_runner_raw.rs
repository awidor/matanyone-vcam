//! Runs `MatAnyoneRunner` over a raw float sample (`scripts/export_raw_sample.py`) and
//! writes one `.alphaf32` per frame for `scripts/measure_runner_accuracy.py`.
//!
//! Usage: matanyone_runner_raw [sample_dir] [engine_dir] [out_dir] [max_frames]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use matanyone_core::{mean, percentile, MatAnyoneRunner, MODEL_H, MODEL_W};

fn read_floats(path: &Path, count: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() < count * 4 {
        bail!("{} has {} bytes, expected {}", path.display(), bytes.len(), count * 4);
    }
    Ok(bytes[..count * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn write_floats(path: &Path, data: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let sample_dir = PathBuf::from(args.next().unwrap_or_else(|| "raw_sample".into()));
    let engine_dir = PathBuf::from(args.next().unwrap_or_else(|| "engines/faithful".into()));
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "output/raw_alpha".into()));
    let max_frames: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(0);
    fs::create_dir_all(&out_dir)?;

    let image_count = 3 * MODEL_H * MODEL_W;
    let alpha_count = MODEL_H * MODEL_W;
    let mask = read_floats(&sample_dir.join("mask.f32"), alpha_count)?;
    let first = read_floats(&sample_dir.join("frames").join("00000.rgbf32"), image_count)?;

    let mut runner = MatAnyoneRunner::new(&engine_dir)?;
    runner.upload_image(&first)?;
    runner.initialize(&mask)?;

    let mut alpha = vec![0.0f32; alpha_count];
    runner.copy_alpha_to_host(&mut alpha)?;
    write_floats(&out_dir.join("00000.alphaf32"), &alpha)?;

    let mut timings = Vec::new();
    let mut frame_index = 1usize;
    loop {
        if max_frames > 0 && frame_index >= max_frames {
            break;
        }
        let frame_path = sample_dir.join("frames").join(format!("{frame_index:05}.rgbf32"));
        if !frame_path.exists() {
            break;
        }
        let frame = read_floats(&frame_path, image_count)?;
        let start = Instant::now();
        runner.upload_image(&frame)?;
        runner.process_frame(frame_index % 5 == 0)?;
        runner.stream().synchronize()?;
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
        runner.copy_alpha_to_host(&mut alpha)?;
        write_floats(&out_dir.join(format!("{frame_index:05}.alphaf32")), &alpha)?;
        frame_index += 1;
    }

    println!("frames={frame_index}");
    if !timings.is_empty() {
        println!("mean_ms={}", mean(&timings));
        println!("p99_ms={}", percentile(timings, 99.0));
    }
    println!("wrote={}", out_dir.display());
    Ok(())
}

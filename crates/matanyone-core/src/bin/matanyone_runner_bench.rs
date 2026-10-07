//! Steady-state `MatAnyoneRunner` latency with a memory update every fifth frame.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use matanyone_core::{mean, percentile, MatAnyoneRunner};

const WARMUP: usize = 30;
const ITERS: usize = 100;

fn main() -> Result<()> {
    let engine_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("engines/faithful"));

    let mut runner = MatAnyoneRunner::new(&engine_dir)?;
    for i in 0..WARMUP {
        runner.process_frame(i % 5 == 0)?;
    }
    runner.stream().synchronize()?;

    let mut timings = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        let start = Instant::now();
        runner.process_frame(i % 5 == 0)?;
        runner.stream().synchronize()?;
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    println!("mixed_mean_ms={}", mean(&timings));
    println!("mixed_p99_ms={}", percentile(timings, 99.0));
    Ok(())
}

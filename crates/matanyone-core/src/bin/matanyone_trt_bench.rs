//! Raw engine-chain latency: each engine runs on its own scratch buffers, so this
//! measures TensorRT execution only (no memory-bank copies or data dependencies).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use matanyone_core::{mean, percentile, BenchEngine, CudaStream, TrtRuntime};

const WARMUP: usize = 30;
const ITERS: usize = 100;

fn time_chain(engines: &[&BenchEngine], stream: &CudaStream) -> Result<Vec<f64>> {
    for _ in 0..WARMUP {
        for engine in engines {
            engine.enqueue(stream)?;
        }
    }
    stream.synchronize()?;

    let mut timings = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        for engine in engines {
            engine.enqueue(stream)?;
        }
        stream.synchronize()?;
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(timings)
}

fn main() -> Result<()> {
    let engine_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("engines/faithful"));

    let runtime = TrtRuntime::new()?;
    let load = |name: &str| BenchEngine::load(&runtime, &engine_dir.join(name));
    let encode = load("encode_image_fp16.engine")?;
    let read = load("read_memory_fp16.engine")?;
    let segment = load("segment_fp16.engine")?;
    let encode_mask = load("encode_mask_fp16.engine")?;
    let encode_mask_shallow = load("encode_mask_shallow_fp16.engine")?;
    let stream = CudaStream::new()?;

    let normal = time_chain(&[&encode, &read, &segment, &encode_mask_shallow], &stream)?;
    let update = time_chain(&[&encode, &read, &segment, &encode_mask], &stream)?;

    println!("normal_mean_ms={}", mean(&normal));
    println!("normal_p99_ms={}", percentile(normal, 99.0));
    println!("memory_update_mean_ms={}", mean(&update));
    println!("memory_update_p99_ms={}", percentile(update, 99.0));

    for (name, engine) in [
        ("encode", &encode),
        ("read", &read),
        ("segment", &segment),
        ("encode_mask", &encode_mask),
        ("encode_mask_shallow", &encode_mask_shallow),
    ] {
        println!("stage_{name}_mean_ms={}", mean(&time_chain(&[engine], &stream)?));
    }
    Ok(())
}

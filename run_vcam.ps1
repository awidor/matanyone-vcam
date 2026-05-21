$env:PATH = "C:\Tools\ffmpeg-2026-05-06-git-f2e5eff3ff-full_build\bin;C:\Tools\TensorRT-10.16.1.11\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;" + $env:PATH
cargo run --release --manifest-path app\Cargo.toml -- --engine-dir engines\faithful

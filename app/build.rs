fn main() {
    let tensorrt_root = std::env::var("TENSORRT_ROOT")
        .unwrap_or_else(|_| "C:/Tools/TensorRT-10.16.1.11".to_string());
    let cuda_root = std::env::var("CUDA_ROOT").unwrap_or_else(|_| {
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1".to_string()
    });

    println!(
        "cargo:rustc-link-search=native={}",
        std::path::PathBuf::from(&tensorrt_root).join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        std::path::PathBuf::from(&cuda_root).join("lib/x64").display()
    );
    println!("cargo:rustc-link-lib=dylib=nvinfer_10");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
}

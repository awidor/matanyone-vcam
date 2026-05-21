use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.join("..");
    let core_dir = repo_root.join("core");

    if !repo_root.join("build/core/Release/matanyone_core.lib").exists()
        && !repo_root.join("build/core/Debug/matanyone_core.lib").exists()
    {
        let mut config = cmake::Config::new(&core_dir);
        config.generator("Visual Studio 17 2022");
        config.profile("Release");
        config.define("CMAKE_GENERATOR_PLATFORM", "x64");
        config.build();
    }

    let profile = if env::var("PROFILE").unwrap() == "release" {
        "Release"
    } else {
        "Debug"
    };

    let lib_dir = repo_root.join("build/core").join(profile);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=matanyone_core");

    let tensorrt_root = env::var("TENSORRT_ROOT")
        .unwrap_or_else(|_| "C:/Tools/TensorRT-10.16.1.11".to_string());
    let cuda_root = env::var("CUDA_ROOT").unwrap_or_else(|_| {
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1".to_string()
    });

    println!(
        "cargo:rustc-link-search=native={}",
        PathBuf::from(&tensorrt_root).join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        PathBuf::from(&cuda_root).join("lib/x64").display()
    );

    println!("cargo:rustc-link-lib=dylib=nvinfer_10");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");

    println!("cargo:rerun-if-changed=../core/include/matanyone_core_api.h");
    println!("cargo:rerun-if-changed=../core/CMakeLists.txt");
}

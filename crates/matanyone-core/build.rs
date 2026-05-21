use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn tensorrt_root() -> PathBuf {
    PathBuf::from(
        env::var("TENSORRT_ROOT").unwrap_or_else(|_| "C:/Tools/TensorRT-10.16.1.11".to_string()),
    )
}

fn cuda_root() -> PathBuf {
    PathBuf::from(env::var("CUDA_ROOT").unwrap_or_else(|_| {
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1".to_string()
    }))
}

fn extract_cubin(out_dir: &Path, header: &Path) {
    let text = fs::read_to_string(header).expect("read matanyone_cubin.h");
    let mut bytes = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with("0x") {
            continue;
        }
        for part in line.split(',') {
            let part = part.trim().trim_end_matches(',');
            if part.starts_with("0x") {
                bytes.push(u8::from_str_radix(&part[2..], 16).expect("cubin byte"));
            }
        }
    }
    let out = out_dir.join("matanyone.cubin");
    fs::write(&out, &bytes).expect("write cubin");
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.join("../..");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cubin_header = manifest_dir.join("kernels/matanyone_cubin.h");
    println!("cargo:rerun-if-changed={}", cubin_header.display());
    extract_cubin(&out_dir, &cubin_header);

    let trt_root = tensorrt_root();
    let cuda = cuda_root();

    let mut cpp = cc::Build::new();
    cpp.cpp(true)
        .std("c++17")
        .file(manifest_dir.join("ffi/trt_wrapper.cpp"))
        .include(manifest_dir.join("ffi"))
        .include(trt_root.join("include"))
        .include(cuda.join("include"))
        .define("NOMINMAX", None);
    if cfg!(target_os = "windows") {
        cpp.flag("/EHsc");
    }
    let profile = env::var("PROFILE").unwrap_or_default();
    if profile == "release" {
        if cfg!(target_os = "windows") {
            cpp.flag("/O2");
        } else {
            cpp.flag("-O3");
        }
    }
    cpp.compile("matanyone_trt_ffi");

    let bindings = bindgen::Builder::default()
        .header(manifest_dir.join("ffi/trt_wrapper.h").to_string_lossy())
        .header(cuda.join("include/cuda_runtime_api.h").to_string_lossy())
        .header(cuda.join("include/cuda.h").to_string_lossy())
        .clang_arg("-xc++")
        .clang_arg("-std=c++17")
        .clang_arg(format!("-I{}", trt_root.join("include").display()))
        .clang_arg(format!("-I{}", cuda.join("include").display()))
        .allowlist_function("matanyone_trt_.*")
        .allowlist_type("TrtRuntime")
        .allowlist_type("TrtEngineHandle")
        .allowlist_type("TrtBenchEngine")
        .allowlist_type("cudaError_t")
        .allowlist_type("CUresult")
        .allowlist_type("CUmodule")
        .allowlist_type("CUfunction")
        .allowlist_var("CUDA_SUCCESS")
        .allowlist_function("cuda.*")
        .allowlist_function("cu.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write bindings");

    println!(
        "cargo:rustc-link-search=native={}",
        trt_root.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cuda.join("lib/x64").display()
    );
    println!("cargo:rustc-link-lib=dylib=nvinfer_10");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");

    println!("cargo:rerun-if-changed=ffi/trt_wrapper.cpp");
    println!("cargo:rerun-if-changed=ffi/trt_wrapper.h");
}

use std::env;
use std::path::{Path, PathBuf};

/// Directory containing the running executable (portable install root).
pub fn install_root() -> PathBuf {
    env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.to_path_buf()))
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Resolve `path` against `base` when relative; leave absolute paths unchanged.
pub fn resolve_against(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

/// Default TensorRT engine directory next to the executable.
pub fn default_engine_dir(install_root: &Path) -> PathBuf {
    install_root.join("engines").join("faithful")
}

/// Repository / bundle root for auxiliary assets (`scripts/`, etc.).
pub fn bundle_root(install_root: &Path) -> PathBuf {
    if install_root.join("scripts").join("sam31_mask_worker.py").exists() {
        return install_root.to_path_buf();
    }

    // Developer layout: exe in target/release (cargo run) — repo root is ../..
    let dev_root = install_root.join("..").join("..");
    if dev_root.join("scripts").join("sam31_mask_worker.py").exists() {
        if let Ok(canonical) = dev_root.canonicalize() {
            return canonical;
        }
    }

    install_root.to_path_buf()
}

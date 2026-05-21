use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

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

/// FFmpeg executable used for DirectShow capture (OBS Virtual Camera, etc.).
pub fn find_ffmpeg() -> Option<PathBuf> {
    if let Ok(dir) = env::var("FFMPEG_DIR") {
        let root = PathBuf::from(&dir);
        let candidate = root.join("bin").join("ffmpeg.exe");
        if candidate.is_file() {
            return Some(candidate);
        }
        let candidate = root.join("ffmpeg.exe");
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let default = PathBuf::from(r"C:\Tools\ffmpeg-2026-05-06-git-f2e5eff3ff-full_build\bin\ffmpeg.exe");
    if default.is_file() {
        return Some(default);
    }

    let output = Command::new("where").arg("ffmpeg").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let path = stdout.lines().next()?.trim();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

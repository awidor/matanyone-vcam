use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;

use anyhow::{Result, anyhow};

use crate::cuda::{check_cu, CudaStream};
use crate::ffi;

struct Kernels {
    bgr_to_rgb_nchw: ffi::CUfunction,
    composite_bgr: ffi::CUfunction,
    accumulate_f32: ffi::CUfunction,
}

// CUfunction handles are process-global driver objects, valid from any thread.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

static KERNELS: OnceLock<Result<Kernels, String>> = OnceLock::new();

fn cubin_bytes() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/matanyone.cubin"))
}

pub fn init_cuda_kernels() -> Result<()> {
    kernels().map(|_| ())
}

fn kernels() -> Result<&'static Kernels> {
    KERNELS
        .get_or_init(|| load().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("failed to initialize CUDA kernels: {e}"))
}

fn load() -> Result<Kernels> {
    unsafe {
        crate::cuda::check_cuda(ffi::cudaSetDevice(0), "cudaSetDevice")?;
        let _ = ffi::cudaFree(ptr::null_mut());
        check_cu(ffi::cuInit(0), "cuInit")?;

        let mut module = ptr::null_mut();
        check_cu(
            ffi::cuModuleLoadData(&mut module, cubin_bytes().as_ptr() as *const _),
            "cuModuleLoadData",
        )?;
        let function = |name: &[u8]| -> Result<ffi::CUfunction> {
            let mut func = ptr::null_mut();
            check_cu(
                ffi::cuModuleGetFunction(&mut func, module, name.as_ptr() as *const _),
                "cuModuleGetFunction",
            )?;
            Ok(func)
        };
        Ok(Kernels {
            bgr_to_rgb_nchw: function(b"bgr_to_rgb_nchw\0")?,
            composite_bgr: function(b"composite_bgr\0")?,
            accumulate_f32: function(b"accumulate_f32\0")?,
        })
    }
}

fn launch(
    func: ffi::CUfunction,
    grid: (u32, u32),
    block: (u32, u32),
    args: &mut [*mut c_void],
    stream: &CudaStream,
    what: &str,
) -> Result<()> {
    check_cu(
        unsafe {
            ffi::cuLaunchKernel(
                func,
                grid.0,
                grid.1,
                1,
                block.0,
                block.1,
                1,
                0,
                stream.raw(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            )
        },
        what,
    )
}

fn arg<T>(value: &T) -> *mut c_void {
    value as *const T as *mut c_void
}

fn grid_2d(w: i32, h: i32) -> (u32, u32) {
    (((w + 31) / 32) as u32, ((h + 7) / 8) as u32)
}

/// Packed BGR8 at any size -> planar RGB f32 at `dst_w`x`dst_h` (bilinear resize).
pub fn launch_bgr_to_rgb_nchw(
    bgr: *const u8,
    src_w: i32,
    src_h: i32,
    src_pitch: i32,
    rgb_nchw: *mut f32,
    dst_w: i32,
    dst_h: i32,
    stream: &CudaStream,
) -> Result<()> {
    let k = kernels()?;
    launch(
        k.bgr_to_rgb_nchw,
        grid_2d(dst_w, dst_h),
        (32, 8),
        &mut [arg(&bgr), arg(&src_w), arg(&src_h), arg(&src_pitch), arg(&rgb_nchw), arg(&dst_w), arg(&dst_h)],
        stream,
        "launch bgr_to_rgb_nchw",
    )
}

/// Planar RGB f32 + alpha -> packed BGR8 over a solid background (RGB in [0, 1]).
pub fn launch_composite_bgr(
    rgb_nchw: *const f32,
    alpha: *const f32,
    bg_rgb: (f32, f32, f32),
    out_bgr: *mut u8,
    w: i32,
    h: i32,
    stream: &CudaStream,
) -> Result<()> {
    let k = kernels()?;
    let (r, g, b) = bg_rgb;
    launch(
        k.composite_bgr,
        grid_2d(w, h),
        (32, 8),
        &mut [arg(&rgb_nchw), arg(&alpha), arg(&r), arg(&g), arg(&b), arg(&out_bgr), arg(&w), arg(&h)],
        stream,
        "launch composite_bgr",
    )
}

/// `dst += src` over `n` floats.
pub fn launch_accumulate_f32(dst: *mut f32, src: *const f32, n: i32, stream: &CudaStream) -> Result<()> {
    let k = kernels()?;
    launch(
        k.accumulate_f32,
        (((n + 255) / 256) as u32, 1),
        (256, 1),
        &mut [arg(&dst), arg(&src), arg(&n)],
        stream,
        "launch accumulate_f32",
    )
}

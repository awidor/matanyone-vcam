use std::ptr;
use std::sync::Once;

use anyhow::{Result, bail};

use crate::cuda::{check_cu, CudaStream, MEMCPY_D2D, MEMCPY_D2H, MEMCPY_H2D, CUDA_SUCCESS};
use crate::ffi;

static INIT: Once = Once::new();
static mut G_MODULE: ffi::CUmodule = ptr::null_mut();
static mut G_BGRA_TO_RGB: ffi::CUfunction = ptr::null_mut();
static mut G_ALPHA_COMPOSITE: ffi::CUfunction = ptr::null_mut();
static mut G_ALPHA_OUTPUT: ffi::CUfunction = ptr::null_mut();
static mut G_COPY_BGRA: ffi::CUfunction = ptr::null_mut();

fn cubin_bytes() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/matanyone.cubin"))
}

pub fn init_cuda_kernels() -> Result<()> {
    let mut err = Ok(());
    INIT.call_once(|| {
        if let Err(e) = init_cuda_kernels_inner() {
            err = Err(e);
        }
    });
    err
}

fn init_cuda_kernels_inner() -> Result<()> {
    unsafe {
        if !G_MODULE.is_null() {
            return Ok(());
        }
        crate::cuda::check_cuda(ffi::cudaSetDevice(0), "cudaSetDevice")?;
        let _ = ffi::cudaFree(ptr::null_mut());
        check_cu(ffi::cuInit(0), "cuInit")?;

        let data = cubin_bytes();
        check_cu(
            ffi::cuModuleLoadData(&mut G_MODULE, data.as_ptr() as *const _),
            "cuModuleLoadData",
        )?;

        let names = [
            (
                &mut G_BGRA_TO_RGB,
                b"_Z23bgra_to_rgb_nchw_kernelPKhiiiPfii\0".as_ptr() as *const i8,
            ),
            (
                &mut G_ALPHA_COMPOSITE,
                b"_Z22alpha_composite_kernelPKfS0_fffPhii\0".as_ptr() as *const i8,
            ),
            (
                &mut G_ALPHA_OUTPUT,
                b"_Z19alpha_output_kernelPKfPhii\0".as_ptr() as *const i8,
            ),
            (
                &mut G_COPY_BGRA,
                b"_Z16copy_bgra_kernelPKhiPhiii\0".as_ptr() as *const i8,
            ),
        ];
        for (func, name) in names {
            check_cu(ffi::cuModuleGetFunction(func, G_MODULE, name), "cuModuleGetFunction")?;
        }
    }
    Ok(())
}

pub fn launch_bgra_to_rgb_nchw(
    bgra: *const u8,
    src_w: i32,
    src_h: i32,
    src_pitch: i32,
    rgb_nchw: *mut f32,
    dst_w: i32,
    dst_h: i32,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        if G_BGRA_TO_RGB.is_null() {
            return Ok(());
        }
        let mut args = [
            &bgra as *const _ as *mut _,
            &src_w as *const _ as *mut _,
            &src_h as *const _ as *mut _,
            &src_pitch as *const _ as *mut _,
            &rgb_nchw as *const _ as *mut _,
            &dst_w as *const _ as *mut _,
            &dst_h as *const _ as *mut _,
        ];
        let grid_x = ((dst_w + 31) / 32) as u32;
        let grid_y = ((dst_h + 15) / 16) as u32;
        check_cu(
            ffi::cuLaunchKernel(
                G_BGRA_TO_RGB,
                grid_x,
                grid_y,
                1,
                32,
                16,
                1,
                0,
                stream.raw(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "launch_bgra_to_rgb_nchw",
        )
    }
}

pub fn launch_alpha_composite(
    src_rgb: *const f32,
    alpha: *const f32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    out_bgra: *mut u8,
    h: i32,
    w: i32,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        if G_ALPHA_COMPOSITE.is_null() {
            bail!("alpha composite kernel unavailable");
        }
        let mut args = [
            &src_rgb as *const _ as *mut _,
            &alpha as *const _ as *mut _,
            &bg_r as *const _ as *mut _,
            &bg_g as *const _ as *mut _,
            &bg_b as *const _ as *mut _,
            &out_bgra as *const _ as *mut _,
            &h as *const _ as *mut _,
            &w as *const _ as *mut _,
        ];
        let grid_x = ((w + 31) / 32) as u32;
        let grid_y = ((h + 15) / 16) as u32;
        check_cu(
            ffi::cuLaunchKernel(
                G_ALPHA_COMPOSITE,
                grid_x,
                grid_y,
                1,
                32,
                16,
                1,
                0,
                stream.raw(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "launch_alpha_composite",
        )
    }
}

pub fn launch_alpha_output(
    alpha: *const f32,
    out_alpha: *mut u8,
    h: i32,
    w: i32,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        if G_ALPHA_OUTPUT.is_null() {
            return Ok(());
        }
        let mut args = [
            &alpha as *const _ as *mut _,
            &out_alpha as *const _ as *mut _,
            &h as *const _ as *mut _,
            &w as *const _ as *mut _,
        ];
        let grid_x = ((w + 31) / 32) as u32;
        let grid_y = ((h + 15) / 16) as u32;
        check_cu(
            ffi::cuLaunchKernel(
                G_ALPHA_OUTPUT,
                grid_x,
                grid_y,
                1,
                32,
                16,
                1,
                0,
                stream.raw(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "launch_alpha_output",
        )
    }
}

pub fn launch_copy_bgra(
    src: *const u8,
    src_pitch: i32,
    dst: *mut u8,
    dst_pitch: i32,
    h: i32,
    w: i32,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        if G_COPY_BGRA.is_null() {
            return Ok(());
        }
        let mut args = [
            &src as *const _ as *mut _,
            &src_pitch as *const _ as *mut _,
            &dst as *const _ as *mut _,
            &dst_pitch as *const _ as *mut _,
            &h as *const _ as *mut _,
            &w as *const _ as *mut _,
        ];
        let grid_x = ((w + 31) / 32) as u32;
        let grid_y = ((h + 15) / 16) as u32;
        check_cu(
            ffi::cuLaunchKernel(
                G_COPY_BGRA,
                grid_x,
                grid_y,
                1,
                32,
                16,
                1,
                0,
                stream.raw(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ),
            "launch_copy_bgra",
        )
    }
}

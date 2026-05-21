use std::ffi::{CStr, CString};
use std::ptr;

use anyhow::{Context, Result, bail};

use crate::ffi;

pub const CUDA_SUCCESS: ffi::CUresult = 0;
pub const CUDA_ERROR_SUCCESS: ffi::cudaError_t = 0;
pub const MEMCPY_H2D: ffi::cudaMemcpyKind = 1;
pub const MEMCPY_D2H: ffi::cudaMemcpyKind = 2;
pub const MEMCPY_D2D: ffi::cudaMemcpyKind = 3;

pub fn check_cuda(status: ffi::cudaError_t, what: &str) -> Result<()> {
    if status != CUDA_ERROR_SUCCESS {
        unsafe {
            let msg = ffi::cudaGetErrorString(status);
            let text = if msg.is_null() {
                "unknown cuda error".to_string()
            } else {
                CStr::from_ptr(msg).to_string_lossy().into_owned()
            };
            bail!("{what}: {text}");
        }
    }
    Ok(())
}

pub fn check_cu(result: ffi::CUresult, what: &str) -> Result<()> {
    if result != CUDA_SUCCESS {
        unsafe {
            let mut msg = ptr::null();
            ffi::cuGetErrorString(result, &mut msg);
            let text = if msg.is_null() {
                "unknown driver error".to_string()
            } else {
                CStr::from_ptr(msg).to_string_lossy().into_owned()
            };
            bail!("{what}: {text}");
        }
    }
    Ok(())
}

pub struct DevicePtr {
    ptr: *mut std::ffi::c_void,
    pub bytes: usize,
}

impl DevicePtr {
    pub fn alloc(bytes: usize, name: &str) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        check_cuda(
            unsafe { ffi::cudaMalloc(&mut ptr, bytes) },
            &format!("cudaMalloc {name}"),
        )?;
        Ok(Self { ptr, bytes })
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = ffi::cudaFree(self.ptr);
            }
        }
    }
}

pub struct CudaStream(pub ffi::cudaStream_t);

impl CudaStream {
    pub fn new() -> Result<Self> {
        let mut stream = ptr::null_mut();
        check_cuda(unsafe { ffi::cudaStreamCreate(&mut stream) }, "cudaStreamCreate")?;
        Ok(Self(stream))
    }

    pub fn raw(&self) -> ffi::cudaStream_t {
        self.0
    }

    pub fn synchronize(&self) -> Result<()> {
        check_cuda(unsafe { ffi::cudaStreamSynchronize(self.0) }, "cudaStreamSynchronize")
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = ffi::cudaStreamDestroy(self.0);
            }
        }
    }
}

pub fn memcpy_async(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    bytes: usize,
    kind: ffi::cudaMemcpyKind,
    stream: ffi::cudaStream_t,
    what: &str,
) -> Result<()> {
    check_cuda(
        unsafe { ffi::cudaMemcpyAsync(dst, src, bytes, kind, stream) },
        what,
    )
}

pub fn memcpy_2d_async(
    dst: *mut std::ffi::c_void,
    dst_pitch: usize,
    src: *const std::ffi::c_void,
    src_pitch: usize,
    width: usize,
    height: usize,
    kind: ffi::cudaMemcpyKind,
    stream: ffi::cudaStream_t,
    what: &str,
) -> Result<()> {
    check_cuda(
        unsafe {
            ffi::cudaMemcpy2DAsync(
                dst,
                dst_pitch,
                src,
                src_pitch,
                width,
                height,
                kind,
                stream,
            )
        },
        what,
    )
}

pub fn memset_async(
    dst: *mut std::ffi::c_void,
    value: i32,
    bytes: usize,
    stream: ffi::cudaStream_t,
    what: &str,
) -> Result<()> {
    check_cuda(
        unsafe { ffi::cudaMemsetAsync(dst, value, bytes, stream) },
        what,
    )
}

pub fn path_to_c(path: &std::path::Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes()).context("invalid path")
}

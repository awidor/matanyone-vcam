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

/// Page-locked host memory, so async copies to and from the device skip the driver's staging copy.
pub struct HostBuffer {
    ptr: *mut u8,
    len: usize,
}

impl HostBuffer {
    pub fn alloc(len: usize, name: &str) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        check_cuda(
            unsafe { ffi::cudaHostAlloc(&mut ptr, len.max(1), 0) },
            &format!("cudaHostAlloc {name}"),
        )?;
        Ok(Self { ptr: ptr as *mut u8, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cudaFreeHost(self.ptr as *mut _);
        }
    }
}

/// A completion marker in a stream. Waiting on it blocks the thread instead of spinning.
pub struct CudaEvent(ffi::cudaEvent_t);

impl CudaEvent {
    pub fn new() -> Result<Self> {
        const BLOCKING_SYNC: u32 = 0x01;
        const DISABLE_TIMING: u32 = 0x02;
        let mut event = ptr::null_mut();
        check_cuda(
            unsafe { ffi::cudaEventCreateWithFlags(&mut event, BLOCKING_SYNC | DISABLE_TIMING) },
            "cudaEventCreateWithFlags",
        )?;
        Ok(Self(event))
    }

    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        check_cuda(unsafe { ffi::cudaEventRecord(self.0, stream.raw()) }, "cudaEventRecord")
    }

    pub fn synchronize(&self) -> Result<()> {
        check_cuda(unsafe { ffi::cudaEventSynchronize(self.0) }, "cudaEventSynchronize")
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cudaEventDestroy(self.0);
        }
    }
}

/// An instantiated CUDA graph captured from a stream.
pub struct CudaGraph(ffi::cudaGraphExec_t);

impl CudaGraph {
    /// Captures everything `record` enqueues on `stream` without executing it.
    pub fn capture(stream: &CudaStream, record: impl FnOnce() -> Result<()>) -> Result<Self> {
        check_cuda(
            unsafe {
                ffi::cudaStreamBeginCapture(
                    stream.raw(),
                    ffi::cudaStreamCaptureMode_cudaStreamCaptureModeThreadLocal,
                )
            },
            "cudaStreamBeginCapture",
        )?;
        let recorded = record();
        let mut graph = ptr::null_mut();
        let ended = check_cuda(
            unsafe { ffi::cudaStreamEndCapture(stream.raw(), &mut graph) },
            "cudaStreamEndCapture",
        );
        recorded?;
        ended?;
        let mut exec = ptr::null_mut();
        let instantiated = check_cuda(
            unsafe { ffi::cudaGraphInstantiate(&mut exec, graph, 0) },
            "cudaGraphInstantiate",
        );
        unsafe {
            let _ = ffi::cudaGraphDestroy(graph);
        }
        instantiated?;
        Ok(Self(exec))
    }

    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        check_cuda(unsafe { ffi::cudaGraphLaunch(self.0, stream.raw()) }, "cudaGraphLaunch")
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cudaGraphExecDestroy(self.0);
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

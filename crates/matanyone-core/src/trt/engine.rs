use std::ffi::CString;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::cuda::{path_to_c, CudaStream};
use crate::ffi;

pub const DTYPE_FLOAT: i32 = 0;

fn dtype_size(dtype: i32) -> Result<usize> {
    Ok(match dtype {
        0 | 3 => 4,     // kFLOAT, kINT32
        1 | 7 => 2,     // kHALF, kBF16
        2 | 4 | 5 => 1, // kINT8, kBOOL, kUINT8
        8 => 8,         // kINT64
        _ => bail!("unsupported TensorRT data type {dtype}"),
    })
}

pub struct TrtRuntime {
    ptr: *mut ffi::TrtRuntime,
}

impl TrtRuntime {
    pub fn new() -> Result<Self> {
        let ptr = unsafe { ffi::matanyone_trt_create_runtime() };
        if ptr.is_null() {
            bail!("createInferRuntime failed");
        }
        Ok(Self { ptr })
    }

    pub fn as_ptr(&self) -> *mut ffi::TrtRuntime {
        self.ptr
    }
}

impl Drop for TrtRuntime {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::matanyone_trt_destroy_runtime(self.ptr);
            }
        }
    }
}

pub struct TrtEngine {
    ptr: *mut ffi::TrtEngineHandle,
    path: String,
}

impl TrtEngine {
    pub fn load(runtime: &TrtRuntime, path: &Path) -> Result<Self> {
        let cpath = path_to_c(path)?;
        let ptr = unsafe { ffi::matanyone_trt_load_engine(runtime.as_ptr(), cpath.as_ptr()) };
        if ptr.is_null() {
            bail!("Failed to deserialize {}", path.display());
        }
        Ok(Self {
            ptr,
            path: path.display().to_string(),
        })
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        let cname = CString::new(name).unwrap();
        unsafe { ffi::matanyone_trt_has_tensor(self.ptr, cname.as_ptr()) != 0 }
    }

    /// TensorRT `DataType` code of an I/O tensor (0 = FP32, 1 = FP16, ...).
    pub fn tensor_dtype(&self, name: &str) -> Result<i32> {
        let cname = CString::new(name).context("tensor name")?;
        match unsafe { ffi::matanyone_trt_tensor_dtype(self.ptr, cname.as_ptr()) } {
            -1 => bail!("{name} is not an I/O tensor of {}", self.path),
            dtype => Ok(dtype),
        }
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<usize> {
        let cname = CString::new(name).context("tensor name")?;
        let volume = unsafe { ffi::matanyone_trt_tensor_volume(self.ptr, cname.as_ptr()) };
        if volume <= 0 {
            bail!("{name} has no static shape in {}", self.path);
        }
        Ok(volume as usize * dtype_size(self.tensor_dtype(name)?)?)
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn bind(&self, name: &str, ptr: *mut std::ffi::c_void) -> Result<()> {
        let cname = CString::new(name).context("tensor name")?;
        if !self.has_tensor(name) {
            return Ok(());
        }
        if unsafe { ffi::matanyone_trt_bind_tensor(self.ptr, cname.as_ptr(), ptr) } == 0 {
            bail!("setTensorAddress failed for {name} in {}", self.path);
        }
        Ok(())
    }

    pub fn enqueue(&self, stream: &CudaStream) -> Result<()> {
        if unsafe { ffi::matanyone_trt_enqueue(self.ptr, stream.raw()) } == 0 {
            bail!("enqueueV3 failed for {}", self.path);
        }
        Ok(())
    }
}

impl Drop for TrtEngine {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::matanyone_trt_destroy_engine(self.ptr);
            }
        }
    }
}

pub struct BenchEngine {
    ptr: *mut ffi::TrtBenchEngine,
    path: String,
}

impl BenchEngine {
    pub fn load(runtime: &TrtRuntime, path: &Path) -> Result<Self> {
        let cpath = path_to_c(path)?;
        let ptr =
            unsafe { ffi::matanyone_trt_load_bench_engine(runtime.as_ptr(), cpath.as_ptr()) };
        if ptr.is_null() {
            bail!("Failed to load bench engine {}", path.display());
        }
        Ok(Self {
            ptr,
            path: path.display().to_string(),
        })
    }

    pub fn enqueue(&self, stream: &CudaStream) -> Result<()> {
        if unsafe { ffi::matanyone_trt_bench_enqueue(self.ptr, stream.raw()) } == 0 {
            bail!("enqueueV3 failed for {}", self.path);
        }
        Ok(())
    }
}

impl Drop for BenchEngine {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::matanyone_trt_destroy_bench_engine(self.ptr);
            }
        }
    }
}

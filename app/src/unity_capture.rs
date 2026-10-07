//! Sends frames to Unity Video Capture using its shared-memory protocol
//! (schellingb/UnityCapture `Source/shared.inl`). The app reading the camera creates the
//! mutex, the "sent" event and the data mapping; the sender opens them and creates "want".

use std::mem::size_of;
use std::ptr;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use windows::core::w;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_ABANDONED, WAIT_OBJECT_0};
use windows::Win32::System::Memory::{MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS};
use windows::Win32::System::Registry::{RegCloseKey, RegOpenKeyExW, HKEY, HKEY_CLASSES_ROOT, KEY_READ};
use windows::Win32::System::Threading::{
    CreateEventW, OpenEventW, OpenMutexW, ReleaseMutex, SetEvent, WaitForSingleObject, EVENT_MODIFY_STATE,
    SYNCHRONIZATION_SYNCHRONIZE,
};

pub const DEVICE_NAME: &str = "Unity Video Capture";

const FORMAT_UINT8: i32 = 0;
const RESIZEMODE_LINEAR: i32 = 1;
/// How long the filter keeps showing the last frame before reporting that sending stopped.
const TIMEOUT_MS: i32 = 1000;
const RECONNECT_INTERVAL: Duration = Duration::from_millis(250);
/// The reader signals "want" before every frame; without it for this long, nobody is watching.
const IDLE_DISCONNECT: Duration = Duration::from_secs(3);

/// `SharedMemHeader` in shared.inl; pixel data follows it.
#[repr(C)]
struct SharedHeader {
    max_size: u32,
    width: i32,
    height: i32,
    stride: i32,
    format: i32,
    resize_mode: i32,
    mirror_mode: i32,
    timeout: i32,
}

/// Whether the first Unity Video Capture device is registered (64- or 32-bit filter).
pub fn is_installed() -> bool {
    [
        w!("CLSID\\{5C2CD55C-92AD-4999-8666-912BD3E70010}"),
        w!("CLSID\\{5C2CD55C-92AD-4999-8666-912BD3E70020}"),
    ]
    .into_iter()
    .any(|path| {
        let mut key = HKEY::default();
        let opened = unsafe { RegOpenKeyExW(HKEY_CLASSES_ROOT, path, 0, KEY_READ, &mut key) }.is_ok();
        if opened {
            unsafe {
                let _ = RegCloseKey(key);
            }
        }
        opened
    })
}

struct Handle(HANDLE);

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

struct View(MEMORY_MAPPED_VIEW_ADDRESS);

impl Drop for View {
    fn drop(&mut self) {
        unsafe {
            let _ = UnmapViewOfFile(self.0);
        }
    }
}

struct Connection {
    // Unmap before closing the mapping handle.
    view: View,
    _mapping: Handle,
    sent: Handle,
    want: Handle,
    mutex: Handle,
}

/// Locks the shared mutex; `None` if the reader holds it for longer than `timeout_ms`.
fn lock(mutex: HANDLE, timeout_ms: u32) -> Option<MutexGuard> {
    let result = unsafe { WaitForSingleObject(mutex, timeout_ms) };
    (result == WAIT_OBJECT_0 || result == WAIT_ABANDONED).then_some(MutexGuard(mutex))
}

struct MutexGuard(HANDLE);

impl Drop for MutexGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = ReleaseMutex(self.0);
        }
    }
}

impl Connection {
    /// Opens the reader's objects. Fails while no app has the camera open.
    fn open() -> Option<Self> {
        let mutex = Handle(unsafe { OpenMutexW(SYNCHRONIZATION_SYNCHRONIZE, false, w!("UnityCapture_Mutx")) }.ok()?);
        let guard = lock(mutex.0, 1000)?;
        let want = Handle(unsafe { CreateEventW(None, false, false, w!("UnityCapture_Want")) }.ok()?);
        let sent = Handle(unsafe { OpenEventW(EVENT_MODIFY_STATE, false, w!("UnityCapture_Sent")) }.ok()?);
        let mapping = Handle(unsafe { OpenFileMappingW(FILE_MAP_WRITE.0, false, w!("UnityCapture_Data")) }.ok()?);
        let view = unsafe { MapViewOfFile(mapping.0, FILE_MAP_WRITE, 0, 0, 0) };
        if view.Value.is_null() {
            return None;
        }
        drop(guard);
        Some(Self { view: View(view), _mapping: mapping, sent, want, mutex })
    }

    fn header(&self) -> *mut SharedHeader {
        self.view.0.Value as *mut SharedHeader
    }
}

pub struct UnityCapture {
    connection: Option<Connection>,
    last_attempt: Option<Instant>,
    last_wanted: Instant,
    rgba: Vec<u8>,
}

impl UnityCapture {
    pub fn new() -> Self {
        Self { connection: None, last_attempt: None, last_wanted: Instant::now(), rgba: Vec::new() }
    }

    /// Publishes a packed BGR frame. Does nothing while no app has the camera open.
    pub fn send_bgr(&mut self, bgr: &[u8], width: u32, height: u32) -> Result<()> {
        if self.connection.is_none() {
            if self.last_attempt.is_some_and(|t| t.elapsed() < RECONNECT_INTERVAL) {
                return Ok(());
            }
            self.last_attempt = Some(Instant::now());
            self.connection = Connection::open();
            self.last_wanted = Instant::now();
        }
        let Some(connection) = &self.connection else {
            return Ok(());
        };

        let (w, h) = (width as usize, height as usize);
        if bgr.len() != w * h * 3 {
            bail!("frame has {} bytes, expected {width}x{height}x3", bgr.len());
        }
        // The filter copies rows straight into a bottom-up DIB, so send the bottom row first.
        self.rgba.resize(w * h * 4, 0);
        for (src_row, dst_row) in bgr.chunks_exact(w * 3).rev().zip(self.rgba.chunks_exact_mut(w * 4)) {
            for (src, dst) in src_row.chunks_exact(3).zip(dst_row.chunks_exact_mut(4)) {
                dst.copy_from_slice(&[src[2], src[1], src[0], 255]);
            }
        }

        let header = connection.header();
        if (unsafe { ptr::read_volatile(ptr::addr_of!((*header).max_size)) } as usize) < self.rgba.len() {
            bail!("Unity Video Capture buffer is smaller than a {width}x{height} frame");
        }
        let Some(guard) = lock(connection.mutex.0, 100) else {
            return Ok(()); // reader busy; skip this frame
        };
        unsafe {
            (*header).width = width as i32;
            (*header).height = height as i32;
            (*header).stride = width as i32;
            (*header).format = FORMAT_UINT8;
            (*header).resize_mode = RESIZEMODE_LINEAR;
            (*header).mirror_mode = 0;
            (*header).timeout = TIMEOUT_MS;
            let data = (header as *mut u8).add(size_of::<SharedHeader>());
            ptr::copy_nonoverlapping(self.rgba.as_ptr(), data, self.rgba.len());
        }
        drop(guard);
        unsafe {
            let _ = SetEvent(connection.sent.0);
        }

        if unsafe { WaitForSingleObject(connection.want.0, 0) } == WAIT_OBJECT_0 {
            self.last_wanted = Instant::now();
        } else if self.last_wanted.elapsed() > IDLE_DISCONNECT {
            // Release our handles so the objects disappear once the last reader closes.
            self.connection = None;
        }
        Ok(())
    }
}

//! Reads OBS Virtual Camera frames straight from OBS's shared-memory queue
//! (obs-studio `shared/obs-shared-memory-queue`), so neither DirectShow nor ffmpeg is needed.

use std::mem::size_of;
use std::ptr;
use std::time::{Duration, Instant};

use matanyone_core::Frame;
use windows::core::w;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Memory::{
    MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, VirtualQuery, FILE_MAP_READ, MEMORY_BASIC_INFORMATION,
    MEMORY_MAPPED_VIEW_ADDRESS,
};
use windows::Win32::System::Registry::{RegCloseKey, RegOpenKeyExW, HKEY, HKEY_CLASSES_ROOT, KEY_READ};

pub const DEVICE_NAME: &str = "OBS Virtual Camera";

const STATE_READY: u32 = 2;
const STATE_STOPPING: u32 = 3;
const FRAME_HEADER_SIZE: usize = 32;
/// OBS releases the queue with STOPPING; if it dies instead, frames just stop. Let go of
/// the queue then too, because OBS cannot restart its virtual camera while we hold it.
const STALL_TIMEOUT: Duration = Duration::from_secs(2);
const REOPEN_INTERVAL: Duration = Duration::from_millis(100);

/// `struct queue_header` in shared-memory-queue.c.
#[repr(C)]
struct QueueHeader {
    write_idx: u32,
    read_idx: u32,
    state: u32,
    offsets: [u32; 3],
    queue_type: u32,
    cx: u32,
    cy: u32,
    interval: u64,
    reserved: [u32; 8],
}

/// Whether the OBS Virtual Camera DirectShow filter is registered.
pub fn is_installed() -> bool {
    let mut key = HKEY::default();
    let opened = unsafe {
        RegOpenKeyExW(
            HKEY_CLASSES_ROOT,
            w!("CLSID\\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"),
            0,
            KEY_READ,
            &mut key,
        )
    };
    if opened.is_err() {
        return false;
    }
    unsafe {
        let _ = RegCloseKey(key);
    }
    true
}

struct Queue {
    handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl Queue {
    fn open() -> Option<Self> {
        let handle = unsafe { OpenFileMappingW(FILE_MAP_READ.0, false, w!("OBSVirtualCamVideo")) }.ok()?;
        let view = unsafe { MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0) };
        let mut info = MEMORY_BASIC_INFORMATION::default();
        let queried = unsafe { VirtualQuery(Some(view.Value), &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
        let queue = Self { handle, view, len: info.RegionSize };
        (!view.Value.is_null() && queried != 0 && queue.len >= size_of::<QueueHeader>()).then_some(queue)
    }

    fn header(&self) -> *const QueueHeader {
        self.view.Value as *const QueueHeader
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe {
            if !self.view.Value.is_null() {
                let _ = UnmapViewOfFile(self.view);
            }
            let _ = CloseHandle(self.handle);
        }
    }
}

pub struct ObsVirtualCamReader {
    queue: Option<Queue>,
    last_open_attempt: Option<Instant>,
    last_idx: Option<u32>,
    last_frame_at: Instant,
    frame: Vec<u8>,
    width: u32,
    height: u32,
}

impl ObsVirtualCamReader {
    pub fn new() -> Self {
        Self {
            queue: None,
            last_open_attempt: None,
            last_idx: None,
            last_frame_at: Instant::now(),
            frame: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Waits up to `timeout` for a new frame; `None` while OBS is not publishing one.
    pub fn next_frame(&mut self, timeout: Duration) -> Option<Frame<'_>> {
        let deadline = Instant::now() + timeout;
        while !self.poll() {
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        Some(Frame::nv12(&self.frame, self.width, self.height))
    }

    /// Copies the newest frame if there is one we have not returned yet.
    fn poll(&mut self) -> bool {
        if self.queue.is_none() {
            if self.last_open_attempt.is_some_and(|t| t.elapsed() < REOPEN_INTERVAL) {
                return false;
            }
            self.last_open_attempt = Some(Instant::now());
            self.queue = Queue::open();
            self.last_idx = None;
            self.last_frame_at = Instant::now();
        }
        let Some(queue) = &self.queue else {
            return false;
        };

        let header = queue.header();
        // OBS updates these fields from another process; read each one once.
        let state = unsafe { ptr::read_volatile(ptr::addr_of!((*header).state)) };
        if state == STATE_STOPPING || self.last_frame_at.elapsed() > STALL_TIMEOUT {
            self.queue = None;
            return false;
        }
        if state != STATE_READY {
            return false;
        }
        let idx = unsafe { ptr::read_volatile(ptr::addr_of!((*header).read_idx)) };
        if self.last_idx == Some(idx) {
            return false;
        }
        let (cx, cy, offset) = unsafe {
            (
                ptr::read_volatile(ptr::addr_of!((*header).cx)),
                ptr::read_volatile(ptr::addr_of!((*header).cy)),
                ptr::read_volatile(ptr::addr_of!((*header).offsets[(idx % 3) as usize])) as usize,
            )
        };
        let len = cx as usize * cy as usize * 3 / 2;
        let start = offset + FRAME_HEADER_SIZE;
        if cx == 0 || cy == 0 || cx % 2 != 0 || cy % 2 != 0 || start + len > queue.len {
            self.queue = None;
            return false;
        }
        self.frame.resize(len, 0);
        unsafe {
            ptr::copy_nonoverlapping((queue.view.Value as *const u8).add(start), self.frame.as_mut_ptr(), len);
        }
        self.width = cx;
        self.height = cy;
        self.last_idx = Some(idx);
        self.last_frame_at = Instant::now();
        true
    }
}

//! Async readback infrastructure for CMax-SLAM contrast values

use bevy::prelude::*;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

/// Contrast values from GPU reduction (3 IWE slices)
#[derive(Clone, Debug, Default)]
pub struct ContrastValues {
    pub center: f32,
    pub plus: f32,
    pub minus: f32,
}

/// Main world resource - receives contrast values from render world
#[derive(Resource)]
pub struct ContrastReceiver {
    pub rx: Mutex<Receiver<ContrastValues>>,
}

/// Render world resource - sends contrast values to main world
#[derive(Resource, Clone)]
pub struct ContrastSender {
    pub tx: Arc<Mutex<Sender<ContrastValues>>>,
}

/// GPU-side contrast result buffer layout (matches WGSL struct)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuContrastResult {
    pub sum_sq_center: u32,
    pub sum_sq_plus: u32,
    pub sum_sq_minus: u32,
    pub pixel_count: u32,
}

/// Create channel and return both ends
pub fn create_contrast_channel() -> (ContrastSender, ContrastReceiver) {
    let (tx, rx) = channel();
    (
        ContrastSender {
            tx: Arc::new(Mutex::new(tx)),
        },
        ContrastReceiver {
            rx: Mutex::new(rx),
        },
    )
}

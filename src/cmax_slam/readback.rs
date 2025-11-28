//! Async readback infrastructure for CMax-SLAM contrast values

use bevy::prelude::*;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

/// Contrast values from GPU reduction (7 IWE slices)
#[derive(Clone, Debug, Default)]
pub struct ContrastValues {
    pub center: f32,
    pub omega_plus: f32,
    pub omega_minus: f32,
    pub cx_plus: f32,
    pub cx_minus: f32,
    pub cy_plus: f32,
    pub cy_minus: f32,
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
/// 7 contrast values + pixel_count = 8 × u32 = 32 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuContrastResult {
    pub sum_sq_center: u32,
    pub sum_sq_omega_plus: u32,
    pub sum_sq_omega_minus: u32,
    pub sum_sq_cx_plus: u32,
    pub sum_sq_cx_minus: u32,
    pub sum_sq_cy_plus: u32,
    pub sum_sq_cy_minus: u32,
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

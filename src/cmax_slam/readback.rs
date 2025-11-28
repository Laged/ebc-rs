//! Async readback infrastructure for CMax-SLAM contrast values

use bevy::prelude::*;
use bevy::render::{
    renderer::RenderDevice,
    render_resource::{Buffer, MapMode, PollType},
};
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

/// Trigger async readback of contrast values
pub fn trigger_contrast_readback(
    staging_buffer: &Buffer,
    _sender: &ContrastSender,
    render_device: &RenderDevice,
) {
    let slice = staging_buffer.slice(..);

    slice.map_async(MapMode::Read, move |result: Result<(), _>| {
        if result.is_ok() {
            // Note: We can't access the buffer data in this callback directly
            // The actual read happens in a polling system
        }
    });

    // Poll to drive the async operation
    let _ = render_device.poll(PollType::Poll);
}

/// Check if readback is ready and send values
/// Returns true if readback completed successfully
pub fn poll_contrast_readback(
    staging_buffer: &Buffer,
    sender: &ContrastSender,
) -> bool {
    let slice = staging_buffer.slice(..);

    // Try to get mapped range - this will panic if buffer isn't mapped yet
    // In production, the render node should call this only after confirming the map is ready
    let data = slice.get_mapped_range();

    if data.len() >= 16 {
        let result: &GpuContrastResult = bytemuck::from_bytes(&data[..16]);

        // Convert to float contrast values
        let values = ContrastValues {
            center: result.sum_sq_center as f32,
            plus: result.sum_sq_plus as f32,
            minus: result.sum_sq_minus as f32,
        };

        // Send to main world
        if let Ok(tx) = sender.tx.lock() {
            let _ = tx.send(values);
        }

        drop(data);
        staging_buffer.unmap();
        return true;
    }

    false
}

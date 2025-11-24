use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::*,
    },
};
use bytemuck::{Pod, Zeroable};

// GPU event representation
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuEvent {
    pub timestamp: u32,
    pub x: u32,
    pub y: u32,
    pub polarity: u32,
}

// Main world event storage
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct EventData {
    pub events: Vec<GpuEvent>,
}

// Handle to surface texture (Layer 0 output)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct SurfaceImage {
    pub handle: Handle<Image>,
}

// Handle to gradient texture (Layer 1 output)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct GradientImage {
    pub handle: Handle<Image>,
}

// Render world GPU buffers
#[derive(Resource, Default)]
pub struct GpuEventBuffer {
    pub buffer: Option<Buffer>,
    pub count: u32,
    pub surface_buffer: Option<Buffer>,
    pub gradient_buffer: Option<Buffer>,
    pub dimensions: UVec2,
    pub dim_buffer: Option<Buffer>,
    pub uploaded: bool,
    pub bind_group_ready: bool,
}

// Edge detection parameters
#[derive(Resource, ExtractResource, Clone)]
pub struct EdgeParams {
    pub threshold: f32,
    pub show_gradient: bool,
}

impl Default for EdgeParams {
    fn default() -> Self {
        Self {
            threshold: 1000.0,
            show_gradient: true,
        }
    }
}

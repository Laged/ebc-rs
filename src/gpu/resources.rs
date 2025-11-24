use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::Buffer;
use super::types::GpuEvent;

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
#[derive(Resource, Clone)]
pub struct EdgeParams {
    pub threshold: f32,
    pub show_gradient: bool,
    pub show_raw: bool,
    // Filter toggles (keyboard 1/2/3/4)
    pub filter_dead_pixels: bool,     // Filter 1: Dead pixel check
    pub filter_density: bool,          // Filter 2: Event density check
    pub filter_bidirectional: bool,    // Filter 3: Bidirectional gradient
    pub filter_temporal: bool,         // Filter 4: Temporal variance
}

impl ExtractResource for EdgeParams {
    type Source = EdgeParams;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

impl Default for EdgeParams {
    fn default() -> Self {
        Self {
            threshold: 1000.0,
            show_gradient: true,
            show_raw: false,
            // Only dead pixels filter ON by default - others can be toggled with 2/3/4 keys
            filter_dead_pixels: true,
            filter_density: false,
            filter_bidirectional: false,
            filter_temporal: false,
        }
    }
}

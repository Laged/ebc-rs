use bytemuck::{Pod, Zeroable};

// GPU-compatible EdgeParams struct that matches WGSL layout
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuEdgeParams {
    pub threshold: f32,
    pub filter_dead_pixels: u32,
    pub filter_density: u32,
    pub filter_bidirectional: u32,
    pub filter_temporal: u32,
}

// GPU event representation
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuEvent {
    pub timestamp: u32,
    pub x: u32,
    pub y: u32,
    pub polarity: u32,
}

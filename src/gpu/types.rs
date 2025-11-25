use bytemuck::{Pod, Zeroable};
use bevy::render::render_resource::ShaderType;

// GPU-compatible params struct that matches WGSL layout
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct GpuParams {
    // Pre-processing
    pub filter_dead_pixels: u32,
    pub filter_density: u32,
    pub filter_temporal: u32,
    pub min_density_count: u32,
    pub min_temporal_spread: f32,

    // Sobel
    pub sobel_threshold: f32,

    // Canny
    pub canny_low_threshold: f32,
    pub canny_high_threshold: f32,

    // LoG
    pub log_threshold: f32,

    // Post-processing
    pub filter_bidirectional: u32,
    pub bidirectional_ratio: f32,

    // Padding for 16-byte alignment
    pub _padding: f32,
}

// Keep old name as alias for backwards compatibility
pub type GpuEdgeParams = GpuParams;

// GPU event representation
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuEvent {
    pub timestamp: u32,
    pub x: u32,
    pub y: u32,
    pub polarity: u32,
}

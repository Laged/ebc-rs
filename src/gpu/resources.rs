use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::{Buffer, BufferAsyncError};
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

// Handle to sobel texture (Layer 1 output)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct SobelImage {
    pub handle: Handle<Image>,
}

// Handle to canny texture (Layer 2 output)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct CannyImage {
    pub handle: Handle<Image>,
}

// Handle to log texture (Layer 3 output)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct LogImage {
    pub handle: Handle<Image>,
}

// Render world GPU buffers
#[derive(Resource, Default)]
pub struct GpuEventBuffer {
    pub buffer: Option<Buffer>,
    pub count: u32,
    pub surface_buffer: Option<Buffer>,
    pub sobel_buffer: Option<Buffer>,
    pub dimensions: UVec2,
    pub dim_buffer: Option<Buffer>,
    pub uploaded: bool,
    pub bind_group_ready: bool,
}

/// Buffer for reading edge texture data back to CPU
#[derive(Resource, Default)]
pub struct EdgeReadbackBuffer {
    /// Staging buffer for Sobel texture readback
    pub sobel_staging: Option<Buffer>,
    /// Staging buffer for Canny texture readback
    pub canny_staging: Option<Buffer>,
    /// Staging buffer for LoG texture readback
    pub log_staging: Option<Buffer>,
    /// Texture dimensions
    pub dimensions: UVec2,
    /// CPU-side edge data (Sobel)
    pub sobel_data: Vec<f32>,
    /// CPU-side edge data (Canny)
    pub canny_data: Vec<f32>,
    /// CPU-side edge data (LoG)
    pub log_data: Vec<f32>,
    /// Whether data is ready for CPU consumption
    pub ready: bool,
    /// Which detector to read back (to avoid reading all three every frame)
    pub active_detector: ActiveDetector,
    /// Channel receiver for async map completion
    pub map_receiver: Option<std::sync::Mutex<std::sync::mpsc::Receiver<Result<(), BufferAsyncError>>>>,
    /// Whether a map operation is in flight
    pub mapping_in_progress: bool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ActiveDetector {
    #[default]
    Sobel,
    Canny,
    Log,
}

// Edge detection parameters
#[derive(Resource, Clone)]
pub struct EdgeParams {
    // Pre-processing
    pub filter_dead_pixels: bool,
    pub filter_density: bool,
    pub filter_temporal: bool,
    pub min_density_count: u32,
    pub min_temporal_spread_us: f32,

    // Detector toggles (UI only)
    pub show_sobel: bool,
    pub show_canny: bool,
    pub show_log: bool,
    pub show_raw: bool,

    // Sobel
    pub sobel_threshold: f32,
    // Keep old `threshold` field for backwards compatibility
    pub threshold: f32,

    // Canny
    pub canny_low_threshold: f32,
    pub canny_high_threshold: f32,

    // LoG
    pub log_threshold: f32,

    // Post-processing
    pub filter_bidirectional: bool,
    pub bidirectional_ratio: f32,
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
            filter_dead_pixels: true,
            filter_density: false,
            filter_temporal: false,
            min_density_count: 5,
            min_temporal_spread_us: 500.0,
            show_sobel: true,
            show_canny: false,
            show_log: false,
            show_raw: false,
            sobel_threshold: 1000.0,
            threshold: 1000.0, // Keep for backwards compatibility
            canny_low_threshold: 50.0,
            canny_high_threshold: 150.0,
            log_threshold: 10.0,
            filter_bidirectional: false,
            bidirectional_ratio: 0.3,
        }
    }
}

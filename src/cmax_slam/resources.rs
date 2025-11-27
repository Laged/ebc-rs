//! Resources for CMax-SLAM motion-compensated edge detection

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

/// CMax-SLAM parameters (CPU-side control)
#[derive(Resource, Clone)]
pub struct CmaxSlamParams {
    /// Enable/disable the pipeline
    pub enabled: bool,
    /// Learning rate for gradient descent
    pub learning_rate: f32,
    /// Maximum iterations per frame
    pub max_iterations: u32,
    /// Convergence threshold (gradient magnitude)
    pub convergence_threshold: f32,
    /// Smoothing factor for temporal filtering of omega
    pub smoothing_alpha: f32,
    /// Weight for edge correlation term
    pub edge_weight: f32,
}

impl Default for CmaxSlamParams {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_rate: 0.001,
            max_iterations: 10,
            convergence_threshold: 1e-6,
            smoothing_alpha: 0.3,
            edge_weight: 5.0,
        }
    }
}

/// CMax-SLAM state (persisted across frames)
#[derive(Resource, Default, Clone)]
pub struct CmaxSlamState {
    /// Current estimated angular velocity (rad/us)
    pub omega: f32,
    /// Current estimated rotation center
    pub centroid: Vec2,
    /// Contrast value at current omega
    pub contrast: f32,
    /// Gradient of contrast w.r.t. omega
    pub gradient: f32,
    /// Iteration count for current optimization
    pub iterations: u32,
    /// Whether optimization has converged
    pub converged: bool,
}

/// GPU-compatible CMax-SLAM parameters
/// Layout: 9 useful fields (36 bytes) + 3 padding (12 bytes) = 48 bytes total
/// This matches WGSL struct exactly (no vec3 alignment issues)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuCmaxSlamParams {
    /// Center of rotation X
    pub centroid_x: f32,      // 0-3
    /// Center of rotation Y
    pub centroid_y: f32,      // 4-7
    /// Reference time for warping
    pub t_ref: f32,           // 8-11
    /// Current angular velocity estimate
    pub omega: f32,           // 12-15
    /// Small delta for numerical gradient
    pub delta_omega: f32,     // 16-19
    /// Weight for edge correlation
    pub edge_weight: f32,     // 20-23
    /// Window start timestamp
    pub window_start: u32,    // 24-27
    /// Window end timestamp
    pub window_end: u32,      // 28-31
    /// Total event count
    pub event_count: u32,     // 32-35
    /// Padding to 48 bytes (multiple of 16 for GPU alignment)
    pub _pad: [u32; 3],       // 36-47
}

/// GPU result buffer layout
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct GpuCmaxSlamResult {
    /// Contrast at omega
    pub contrast_center: f32,
    /// Contrast at omega + delta
    pub contrast_plus: f32,
    /// Contrast at omega - delta
    pub contrast_minus: f32,
    /// Padding
    pub _padding: f32,
}

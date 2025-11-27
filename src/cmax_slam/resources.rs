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
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuCmaxSlamParams {
    /// Center of rotation X
    pub centroid_x: f32,
    /// Center of rotation Y
    pub centroid_y: f32,
    /// Reference time for warping
    pub t_ref: f32,
    /// Current angular velocity estimate
    pub omega: f32,
    /// Small delta for numerical gradient
    pub delta_omega: f32,
    /// Weight for edge correlation
    pub edge_weight: f32,
    /// Window start timestamp
    pub window_start: u32,
    /// Window end timestamp
    pub window_end: u32,
    /// Total event count
    pub event_count: u32,
    /// Padding for alignment
    pub _padding: [u32; 3],
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

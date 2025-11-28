//! Resources for CMax-SLAM motion-compensated edge detection

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};
use std::collections::VecDeque;

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
    /// EMA smoothing factor (0=frozen, 1=no smoothing)
    pub ema_alpha: f32,
    /// Max step as fraction of omega
    pub max_step_fraction: f32,
}

impl Default for CmaxSlamParams {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_rate: 0.5,
            max_iterations: 10,
            convergence_threshold: 1e-6,
            smoothing_alpha: 0.3,
            edge_weight: 5.0,
            ema_alpha: 0.2,
            max_step_fraction: 0.02,
        }
    }
}

/// CMax-SLAM state (persisted across frames)
#[derive(Resource, Clone)]
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

    // Phase 1: Optimizer fields
    /// Perturbation size for numerical gradient (rad/μs)
    pub delta_omega: f32,
    /// Recent omega values for convergence detection
    pub omega_history: VecDeque<f32>,
    /// Whether cold start initialization has occurred
    pub initialized: bool,

    // EMA smoothing
    /// EMA smoothing factor (0=frozen, 1=no smoothing)
    pub ema_alpha: f32,

    // Step clamping
    /// Max step as fraction of omega (default 0.02)
    pub max_step_fraction: f32,
    /// For debugging/UI display
    pub last_raw_step: f32,
    /// Flag for UI
    pub step_was_clamped: bool,

    // Phase 2: Centroid tracking fields
    /// Perturbation size for centroid position (pixels)
    pub delta_pos: f32,
    /// Recent centroid positions for convergence detection
    pub centroid_history: VecDeque<Vec2>,
    /// Learning rate for omega updates
    pub lr_omega: f32,
    /// Learning rate for centroid updates
    pub lr_centroid: f32,
}

impl Default for CmaxSlamState {
    fn default() -> Self {
        Self {
            omega: 0.0,
            centroid: Vec2::new(640.0, 360.0),
            contrast: 0.0,
            gradient: 0.0,
            iterations: 0,
            converged: false,
            delta_omega: 1e-6,
            omega_history: VecDeque::with_capacity(16),
            initialized: false,
            ema_alpha: 0.2,
            max_step_fraction: 0.02,
            last_raw_step: 0.0,
            step_was_clamped: false,
            delta_pos: 3.0,
            centroid_history: VecDeque::with_capacity(16),
            lr_omega: 0.5,
            lr_centroid: 0.1,
        }
    }
}

/// GPU-compatible CMax-SLAM parameters
/// Layout: 10 useful fields (40 bytes) + 2 padding (8 bytes) = 48 bytes total
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
    /// Small delta for centroid position
    pub delta_pos: f32,       // 20-23
    /// Weight for edge correlation
    pub edge_weight: f32,     // 24-27
    /// Window start timestamp
    pub window_start: u32,    // 28-31
    /// Window end timestamp
    pub window_end: u32,      // 32-35
    /// Total event count
    pub event_count: u32,     // 36-39
    /// Padding to 48 bytes (multiple of 16 for GPU alignment)
    pub _pad: [u32; 2],       // 40-47
}

/// GPU result buffer layout
/// 7 contrast values + pixel_count = 8 × f32 = 32 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct GpuCmaxSlamResult {
    /// Contrast at center (omega, cx, cy)
    pub contrast_center: f32,
    /// Contrast at omega + delta_omega
    pub contrast_omega_plus: f32,
    /// Contrast at omega - delta_omega
    pub contrast_omega_minus: f32,
    /// Contrast at cx + delta_pos
    pub contrast_cx_plus: f32,
    /// Contrast at cx - delta_pos
    pub contrast_cx_minus: f32,
    /// Contrast at cy + delta_pos
    pub contrast_cy_plus: f32,
    /// Contrast at cy - delta_pos
    pub contrast_cy_minus: f32,
    /// Pixel count (for normalization)
    pub pixel_count: f32,
}

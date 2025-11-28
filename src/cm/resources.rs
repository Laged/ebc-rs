// src/cm/resources.rs
//! Resources for Contrast Maximization RPM estimation

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

/// CM parameters for CPU-side control
#[derive(Resource, Clone)]
pub struct CmParams {
    pub n_omega: u32,
    pub enabled: bool,
}

impl Default for CmParams {
    fn default() -> Self {
        Self {
            n_omega: 64,
            enabled: true,
        }
    }
}

/// GPU-compatible CM parameters
/// Note: WGSL alignment rules require vec3<u32> to be 16-byte aligned
/// Layout: 9 x f32/u32 = 36 bytes, then padding to 48 for vec3 alignment,
/// then vec3 = 12 bytes + 4 padding = 64 total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuCmParams {
    pub centroid_x: f32,      // 0-3
    pub centroid_y: f32,      // 4-7
    pub t_ref: f32,           // 8-11
    pub omega_min: f32,       // 12-15
    pub omega_step: f32,      // 16-19
    pub n_omega: u32,         // 20-23
    pub window_start: u32,    // 24-27
    pub window_end: u32,      // 28-31
    pub event_count: u32,     // 32-35
    pub _pad1: [u32; 3],      // 36-47 (padding to align vec3 at offset 48)
    pub _padding: [u32; 3],   // 48-59 (the actual vec3<u32> data)
    pub _pad2: u32,           // 60-63 (trailing alignment for vec3)
}

/// CM results read back from GPU
#[derive(Resource, Default, Clone, Debug)]
pub struct CmResult {
    pub best_omega: f32,
    pub best_contrast: f32,
    pub rpm: f32,
    pub confidence: f32,
}

impl CmResult {
    /// Update with temporal smoothing
    pub fn update_smoothed(&mut self, new_omega: f32, new_contrast: f32, alpha: f32) {
        if self.best_omega > 0.0 && new_omega > 0.0 {
            self.best_omega = alpha * new_omega + (1.0 - alpha) * self.best_omega;
        } else if new_omega > 0.0 {
            self.best_omega = new_omega;
        }
        self.best_contrast = new_contrast;
        // Convert rad/μs to RPM: ω * (60s/min) / (2π rad/rev) * (1e6 μs/s)
        self.rpm = self.best_omega.abs() * 60.0 / std::f32::consts::TAU * 1e6;
        self.confidence = (new_contrast / 1000.0).min(1.0);
    }
}

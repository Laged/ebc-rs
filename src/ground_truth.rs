//! Ground truth configuration for synthetic fan data validation.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use serde::Deserialize;
use std::f32::consts::PI;
use std::path::Path;

/// Ground truth fan parameters loaded from JSON sidecar file.
#[derive(Resource, ExtractResource, Deserialize, Debug, Clone, Default)]
pub struct GroundTruthConfig {
    /// Whether ground truth rendering is enabled
    #[serde(default)]
    pub enabled: bool,
    /// Fan center X coordinate (pixels)
    #[serde(default = "default_center_x")]
    pub center_x: f32,
    /// Fan center Y coordinate (pixels)
    #[serde(default = "default_center_y")]
    pub center_y: f32,
    /// Minimum radius (blade root, pixels)
    #[serde(default = "default_radius_min")]
    pub radius_min: f32,
    /// Maximum radius (blade tip, pixels)
    #[serde(default = "default_radius_max")]
    pub radius_max: f32,
    /// Number of blades
    #[serde(default = "default_blade_count")]
    pub blade_count: u32,
    /// Rotations per minute
    #[serde(default = "default_rpm")]
    pub rpm: f32,
    /// Logarithmic spiral curvature parameter
    #[serde(default = "default_sweep_k")]
    pub sweep_k: f32,
    /// Blade angular width at root (radians)
    #[serde(default = "default_width_root")]
    pub width_root_rad: f32,
    /// Blade angular width at tip (radians)
    #[serde(default = "default_width_tip")]
    pub width_tip_rad: f32,
    /// Edge detection thickness (pixels)
    #[serde(default = "default_edge_thickness")]
    pub edge_thickness_px: f32,
}

fn default_center_x() -> f32 { 640.0 }
fn default_center_y() -> f32 { 360.0 }
fn default_radius_min() -> f32 { 50.0 }
fn default_radius_max() -> f32 { 200.0 }
fn default_blade_count() -> u32 { 3 }
fn default_rpm() -> f32 { 1200.0 }
fn default_sweep_k() -> f32 { 0.5 }
fn default_width_root() -> f32 { 0.5 }
fn default_width_tip() -> f32 { 0.3 }
fn default_edge_thickness() -> f32 { 2.0 }

impl GroundTruthConfig {
    /// Angular velocity in radians per second
    pub fn angular_velocity(&self) -> f32 {
        self.rpm * 2.0 * PI / 60.0
    }

    /// Try to load ground truth config from JSON sidecar file.
    /// Returns None if file doesn't exist or isn't valid ground truth JSON.
    pub fn load_from_sidecar(dat_path: &Path) -> Option<Self> {
        // Try _truth.json suffix first (e.g., fan_test.dat -> fan_test_truth.json)
        let truth_path = dat_path.with_extension("").to_string_lossy().to_string() + "_truth.json";
        let truth_path = Path::new(&truth_path);

        if !truth_path.exists() {
            return None;
        }

        let contents = std::fs::read_to_string(truth_path).ok()?;

        // Parse JSON - expect {"params": {...}, "frames": [...]}
        let json: serde_json::Value = serde_json::from_str(&contents).ok()?;
        let params = json.get("params")?;

        let mut config: GroundTruthConfig = serde_json::from_value(params.clone()).ok()?;
        config.enabled = true;

        Some(config)
    }

    /// Generate ground truth edge mask for a given timestamp.
    /// Returns a boolean mask where true = edge pixel.
    ///
    /// This mirrors the GPU shader logic but runs on CPU for metric computation.
    pub fn generate_edge_mask(&self, time_us: f32, width: u32, height: u32) -> Vec<bool> {
        let mut mask = vec![false; (width * height) as usize];

        let time_secs = time_us / 1_000_000.0;
        let base_angle = self.angular_velocity() * time_secs;
        let blade_spacing = 2.0 * PI / self.blade_count as f32;

        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - self.center_x;
                let dy = y as f32 - self.center_y;
                let r = (dx * dx + dy * dy).sqrt();

                // Skip pixels outside blade radius range
                if r < self.radius_min || r > self.radius_max {
                    continue;
                }

                let theta = dy.atan2(dx);

                // Logarithmic spiral offset for blade curvature
                let sweep_angle = self.sweep_k * (r / self.radius_min).ln();

                // Blade width interpolation (wider at root, narrower at tip)
                let r_norm = (r - self.radius_min) / (self.radius_max - self.radius_min);
                let blade_width = self.width_root_rad + (self.width_tip_rad - self.width_root_rad) * r_norm;
                let half_width = blade_width * 0.5;

                // Edge thickness in angular terms at this radius
                let edge_thickness_rad = self.edge_thickness_px / r;

                // Check each blade
                for blade_idx in 0..self.blade_count {
                    let blade_root_angle = base_angle + (blade_idx as f32 * blade_spacing);
                    let blade_center = blade_root_angle + sweep_angle;

                    // Normalize angle difference to [-PI, PI]
                    let mut angle_diff = theta - blade_center;
                    while angle_diff > PI { angle_diff -= 2.0 * PI; }
                    while angle_diff < -PI { angle_diff += 2.0 * PI; }

                    // Leading edge check (front of blade)
                    let leading_edge_angle = half_width;
                    let leading_dist = (angle_diff - leading_edge_angle).abs();

                    // Trailing edge check (back of blade)
                    let trailing_edge_angle = -half_width;
                    let trailing_dist = (angle_diff - trailing_edge_angle).abs();

                    // Pixel is an edge if close to either blade boundary
                    if leading_dist < edge_thickness_rad || trailing_dist < edge_thickness_rad {
                        let idx = (y * width + x) as usize;
                        mask[idx] = true;
                        break; // No need to check other blades
                    }
                }
            }
        }

        mask
    }
}

/// Metrics for comparing detected edges against ground truth
#[derive(Debug, Clone, Default)]
pub struct GroundTruthMetrics {
    pub true_positives: u32,
    pub false_positives: u32,
    pub false_negatives: u32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub iou: f32,
}

impl GroundTruthMetrics {
    /// Compute metrics comparing detected edges to ground truth.
    ///
    /// # Arguments
    /// * `detected` - Detected edge values (> 0 means edge)
    /// * `ground_truth` - Ground truth edge mask (true = edge)
    pub fn compute(detected: &[f32], ground_truth: &[bool]) -> Self {
        assert_eq!(detected.len(), ground_truth.len(), "Size mismatch");

        let mut tp = 0u32;
        let mut fp = 0u32;
        let mut fn_ = 0u32;

        for (d, g) in detected.iter().zip(ground_truth.iter()) {
            let is_detected = *d > 0.0;
            let is_ground_truth = *g;

            match (is_detected, is_ground_truth) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {} // true negative, not counted
            }
        }

        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };

        let recall = if tp + fn_ > 0 {
            tp as f32 / (tp + fn_) as f32
        } else {
            0.0
        };

        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let iou = if tp + fp + fn_ > 0 {
            tp as f32 / (tp + fp + fn_) as f32
        } else {
            0.0
        };

        Self {
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_,
            precision,
            recall,
            f1_score,
            iou,
        }
    }
}


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
    // Exact match metrics
    pub true_positives: u32,
    pub false_positives: u32,
    pub false_negatives: u32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub iou: f32,

    // Distance-tolerant metrics
    pub tolerance_precision: f32,  // % of detected edges within tolerance of ground truth
    pub tolerance_recall: f32,     // % of ground truth edges within tolerance of detected
    pub tolerance_f1: f32,         // F1 using tolerance-based matching
    pub avg_distance: f32,         // Average distance from GT edge to nearest detected edge
    pub median_distance: f32,      // Median distance from GT edge to nearest detected edge
}

impl GroundTruthMetrics {
    /// Compute metrics comparing detected edges to ground truth.
    /// Uses both exact matching and distance-tolerant matching.
    ///
    /// # Arguments
    /// * `detected` - Detected edge values (> 0 means edge)
    /// * `ground_truth` - Ground truth edge mask (true = edge)
    /// * `width` - Image width for coordinate computation
    /// * `tolerance` - Distance tolerance in pixels for "close enough" matching
    pub fn compute(detected: &[f32], ground_truth: &[bool], width: u32, tolerance: f32) -> Self {
        assert_eq!(detected.len(), ground_truth.len(), "Size mismatch");

        // Exact match metrics
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
                (false, false) => {}
            }
        }

        let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f32 / (tp + fn_) as f32 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else { 0.0 };
        let iou = if tp + fp + fn_ > 0 { tp as f32 / (tp + fp + fn_) as f32 } else { 0.0 };

        // Extract pixel coordinates
        let height = detected.len() as u32 / width;
        let mut detected_pixels: Vec<(i32, i32)> = Vec::new();
        let mut gt_pixels: Vec<(i32, i32)> = Vec::new();

        for (i, (&d, &g)) in detected.iter().zip(ground_truth.iter()).enumerate() {
            let x = (i as u32 % width) as i32;
            let y = (i as u32 / width) as i32;
            if d > 0.0 {
                detected_pixels.push((x, y));
            }
            if g {
                gt_pixels.push((x, y));
            }
        }

        // Distance-tolerant metrics
        let tolerance_sq = tolerance * tolerance;

        // Build a simple spatial grid for faster lookup (divide into cells)
        let cell_size = (tolerance * 2.0).max(10.0) as i32;

        // Index detected pixels into grid
        let mut detected_grid: std::collections::HashMap<(i32, i32), Vec<(i32, i32)>> =
            std::collections::HashMap::new();
        for &(x, y) in &detected_pixels {
            let cell = (x / cell_size, y / cell_size);
            detected_grid.entry(cell).or_default().push((x, y));
        }

        // For each GT pixel, find distance to nearest detected pixel
        let mut distances: Vec<f32> = Vec::with_capacity(gt_pixels.len());
        let mut gt_within_tolerance = 0u32;

        for &(gx, gy) in &gt_pixels {
            let cell_x = gx / cell_size;
            let cell_y = gy / cell_size;

            let mut min_dist_sq = f32::MAX;

            // Check neighboring cells
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let check_cell = (cell_x + dx, cell_y + dy);
                    if let Some(pixels) = detected_grid.get(&check_cell) {
                        for &(px, py) in pixels {
                            let dist_sq = ((px - gx) * (px - gx) + (py - gy) * (py - gy)) as f32;
                            min_dist_sq = min_dist_sq.min(dist_sq);
                        }
                    }
                }
            }

            // Only record distance if we found at least one detected pixel
            if min_dist_sq < f32::MAX {
                let dist = min_dist_sq.sqrt();
                distances.push(dist);

                if min_dist_sq <= tolerance_sq {
                    gt_within_tolerance += 1;
                }
            }
        }

        // Index GT pixels into grid for detected->GT lookup
        let mut gt_grid: std::collections::HashMap<(i32, i32), Vec<(i32, i32)>> =
            std::collections::HashMap::new();
        for &(x, y) in &gt_pixels {
            let cell = (x / cell_size, y / cell_size);
            gt_grid.entry(cell).or_default().push((x, y));
        }

        // For each detected pixel, check if within tolerance of any GT pixel
        let mut detected_within_tolerance = 0u32;

        for &(dx, dy) in &detected_pixels {
            let cell_x = dx / cell_size;
            let cell_y = dy / cell_size;

            let mut found = false;
            'outer: for dcx in -1..=1 {
                for dcy in -1..=1 {
                    let check_cell = (cell_x + dcx, cell_y + dcy);
                    if let Some(pixels) = gt_grid.get(&check_cell) {
                        for &(gx, gy) in pixels {
                            let dist_sq = ((dx - gx) * (dx - gx) + (dy - gy) * (dy - gy)) as f32;
                            if dist_sq <= tolerance_sq {
                                found = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
            if found {
                detected_within_tolerance += 1;
            }
        }

        // Compute tolerance-based precision/recall
        let tolerance_precision = if !detected_pixels.is_empty() {
            detected_within_tolerance as f32 / detected_pixels.len() as f32
        } else { 0.0 };

        let tolerance_recall = if !gt_pixels.is_empty() {
            gt_within_tolerance as f32 / gt_pixels.len() as f32
        } else { 0.0 };

        let tolerance_f1 = if tolerance_precision + tolerance_recall > 0.0 {
            2.0 * tolerance_precision * tolerance_recall / (tolerance_precision + tolerance_recall)
        } else { 0.0 };

        // Compute distance statistics
        let avg_distance = if !distances.is_empty() {
            distances.iter().sum::<f32>() / distances.len() as f32
        } else { 0.0 };

        let median_distance = if !distances.is_empty() {
            let mut sorted = distances.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        } else { 0.0 };

        Self {
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_,
            precision,
            recall,
            f1_score,
            iou,
            tolerance_precision,
            tolerance_recall,
            tolerance_f1,
            avg_distance,
            median_distance,
        }
    }
}


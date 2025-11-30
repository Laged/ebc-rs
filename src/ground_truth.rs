//! Ground truth configuration for synthetic fan data validation.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::Path;

/// Centroid motion configuration for synthetic data
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CentroidMotion {
    #[serde(rename = "static")]
    Static,
    #[serde(rename = "linear_drift")]
    LinearDrift { velocity_x: f32, velocity_y: f32 },
    #[serde(rename = "oscillation")]
    Oscillation {
        amplitude_x: f32,
        amplitude_y: f32,
        frequency_hz: f32,
        phase_offset: f32
    },
}

impl CentroidMotion {
    /// Compute centroid position at given time
    pub fn centroid_at_time(&self, t_us: f32, base_x: f32, base_y: f32) -> Vec2 {
        let t_secs = t_us / 1_000_000.0;
        match self {
            CentroidMotion::Static => Vec2::new(base_x, base_y),
            CentroidMotion::LinearDrift { velocity_x, velocity_y } => {
                Vec2::new(base_x + velocity_x * t_secs, base_y + velocity_y * t_secs)
            }
            CentroidMotion::Oscillation { amplitude_x, amplitude_y, frequency_hz, phase_offset } => {
                let phase = 2.0 * PI * frequency_hz * t_secs + phase_offset;
                Vec2::new(
                    base_x + amplitude_x * phase.cos(),
                    base_y + amplitude_y * phase.sin()
                )
            }
        }
    }
}

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
    /// Centroid motion configuration
    #[serde(default)]
    pub motion: Option<CentroidMotion>,
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

    /// Compute ground truth centroid position at a given timestamp
    pub fn centroid_at_time(&self, t_us: f32) -> Vec2 {
        match &self.motion {
            Some(motion) => motion.centroid_at_time(t_us, self.center_x, self.center_y),
            None => Vec2::new(self.center_x, self.center_y),
        }
    }

    /// Check if a point at a specific time is on an edge.
    /// Returns the distance in pixels to the nearest edge, or None if outside radius range.
    pub fn distance_to_edge(&self, x: f32, y: f32, time_us: f32) -> Option<f32> {
        let dx = x - self.center_x;
        let dy = y - self.center_y;
        let r = (dx * dx + dy * dy).sqrt();

        // Skip points outside blade radius range
        if r < self.radius_min || r > self.radius_max {
            return None;
        }

        let theta = dy.atan2(dx);
        let time_secs = time_us / 1_000_000.0;
        let base_angle = self.angular_velocity() * time_secs;
        let blade_spacing = 2.0 * PI / self.blade_count as f32;

        // Logarithmic spiral offset
        let sweep_angle = self.sweep_k * (r / self.radius_min).ln();

        // Blade width at this radius
        let r_norm = (r - self.radius_min) / (self.radius_max - self.radius_min);
        let blade_width = self.width_root_rad + (self.width_tip_rad - self.width_root_rad) * r_norm;
        let half_width = blade_width * 0.5;

        // Find minimum distance to any blade edge
        let mut min_dist_px = f32::MAX;

        for blade in 0..self.blade_count {
            let blade_center = base_angle + (blade as f32 * blade_spacing) + sweep_angle;

            // Normalize angle difference to [-PI, PI]
            let mut angle_diff = theta - blade_center;
            while angle_diff > PI { angle_diff -= 2.0 * PI; }
            while angle_diff < -PI { angle_diff += 2.0 * PI; }

            // Distance to leading and trailing edges in pixels
            let dist_to_leading = (angle_diff - half_width).abs() * r;
            let dist_to_trailing = (angle_diff + half_width).abs() * r;

            min_dist_px = min_dist_px.min(dist_to_leading.min(dist_to_trailing));
        }

        Some(min_dist_px)
    }

    /// Check if a point at a specific time is on an edge (within threshold).
    pub fn is_edge(&self, x: f32, y: f32, time_us: f32) -> bool {
        self.distance_to_edge(x, y, time_us)
            .map(|d| d <= self.edge_thickness_px)
            .unwrap_or(false)
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

    /// Generate ground truth edge mask averaged over a time window.
    /// Returns a boolean mask where true = edge pixel at any point in the window.
    ///
    /// For rotating objects, this accounts for motion blur during the window.
    pub fn generate_edge_mask_window(
        &self,
        window_start_us: f32,
        window_end_us: f32,
        width: u32,
        height: u32,
        num_samples: u32,
    ) -> Vec<bool> {
        let mut mask = vec![false; (width * height) as usize];

        // Sample multiple time points across the window
        for sample in 0..num_samples {
            let t = window_start_us
                + (window_end_us - window_start_us) * (sample as f32 / num_samples as f32);

            let sample_mask = self.generate_edge_mask(t, width, height);

            // OR together all samples
            for (i, &is_edge) in sample_mask.iter().enumerate() {
                if is_edge {
                    mask[i] = true;
                }
            }
        }

        mask
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

                    // Check distance to leading edge (at +half_width)
                    let dist_to_leading = (angle_diff - half_width).abs();
                    // Check distance to trailing edge (at -half_width)
                    let dist_to_trailing = (angle_diff + half_width).abs();

                    // Pixel is an edge if within thickness of either blade boundary
                    // This matches the synthesis which generates events AT the blade edges
                    if dist_to_leading < edge_thickness_rad || dist_to_trailing < edge_thickness_rad {
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
        let _height = detected.len() as u32 / width;
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

#[cfg(test)]
mod task9_tests {
    use super::{GroundTruthConfig, CentroidMotion};
    use std::path::PathBuf;

    #[test]
    fn test_static_centroid() {
        let config = GroundTruthConfig {
            enabled: true,
            center_x: 640.0,
            center_y: 360.0,
            motion: None,
            ..Default::default()
        };
        
        let c0 = config.centroid_at_time(0.0);
        let c1 = config.centroid_at_time(1_000_000.0);
        
        assert_eq!(c0.x, 640.0);
        assert_eq!(c0.y, 360.0);
        assert_eq!(c1.x, 640.0);
        assert_eq!(c1.y, 360.0);
    }

    #[test]
    fn test_linear_drift() {
        let config = GroundTruthConfig {
            enabled: true,
            center_x: 640.0,
            center_y: 360.0,
            motion: Some(CentroidMotion::LinearDrift {
                velocity_x: 50.0,
                velocity_y: 20.0,
            }),
            ..Default::default()
        };
        
        let c0 = config.centroid_at_time(0.0);
        let c1 = config.centroid_at_time(1_000_000.0); // 1 second
        let c2 = config.centroid_at_time(2_000_000.0); // 2 seconds
        
        assert_eq!(c0.x, 640.0);
        assert_eq!(c0.y, 360.0);
        assert!((c1.x - 690.0).abs() < 0.01);
        assert!((c1.y - 380.0).abs() < 0.01);
        assert!((c2.x - 740.0).abs() < 0.01);
        assert!((c2.y - 400.0).abs() < 0.01);
    }

    #[test]
    fn test_backward_compatibility_no_motion() {
        // fan_test.dat should not have motion field
        let path = PathBuf::from("data/synthetic/fan_test.dat");
        if let Some(config) = GroundTruthConfig::load_from_sidecar(&path) {
            assert!(config.motion.is_none());
            let c = config.centroid_at_time(0.0);
            assert_eq!(c.x, 640.0);
            assert_eq!(c.y, 360.0);
        }
    }

    #[test]
    fn test_motion_field_present() {
        // fan_drift.dat should have motion field
        let path = PathBuf::from("data/synthetic/fan_drift.dat");
        if let Some(config) = GroundTruthConfig::load_from_sidecar(&path) {
            assert!(config.motion.is_some());
            match config.motion {
                Some(CentroidMotion::LinearDrift { velocity_x, velocity_y }) => {
                    assert_eq!(velocity_x, 50.0);
                    assert_eq!(velocity_y, 20.0);
                }
                _ => panic!("Expected LinearDrift motion"),
            }
        }
    }
}

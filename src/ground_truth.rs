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
}


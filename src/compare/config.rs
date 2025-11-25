//! Configuration for compare_live binary.

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Per-detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_window_size")]
    pub window_size_us: f32,
    #[serde(default = "default_true")]
    pub filter_dead_pixels: bool,
}

fn default_threshold() -> f32 { 50.0 }
fn default_window_size() -> f32 { 100000.0 }
fn default_true() -> bool { true }

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            threshold: default_threshold(),
            window_size_us: default_window_size(),
            filter_dead_pixels: default_true(),
        }
    }
}

/// Canny-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CannyConfig {
    #[serde(default = "default_canny_low")]
    pub low_threshold: f32,
    #[serde(default = "default_canny_high")]
    pub high_threshold: f32,
    #[serde(default = "default_window_size")]
    pub window_size_us: f32,
    #[serde(default = "default_true")]
    pub filter_dead_pixels: bool,
}

fn default_canny_low() -> f32 { 50.0 }
fn default_canny_high() -> f32 { 150.0 }

impl Default for CannyConfig {
    fn default() -> Self {
        Self {
            low_threshold: default_canny_low(),
            high_threshold: default_canny_high(),
            window_size_us: default_window_size(),
            filter_dead_pixels: default_true(),
        }
    }
}

/// Display settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    #[serde(default = "default_true")]
    pub show_ground_truth: bool,
    #[serde(default = "default_metrics_hz")]
    pub metrics_update_hz: u32,
}

fn default_metrics_hz() -> u32 { 10 }

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            show_ground_truth: true,
            metrics_update_hz: default_metrics_hz(),
        }
    }
}

/// Complete configuration for compare_live
#[derive(Debug, Clone, Serialize, Deserialize, Default, Resource)]
pub struct CompareConfig {
    #[serde(default)]
    pub sobel: DetectorConfig,
    #[serde(default)]
    pub canny: CannyConfig,
    #[serde(default)]
    pub log: DetectorConfig,
    #[serde(default)]
    pub display: DisplayConfig,
}

impl CompareConfig {
    /// Load from TOML file
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: CompareConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Load with fallback chain: path -> config/detectors.toml -> results/best_*.json -> defaults
    pub fn load_with_fallback(path: Option<&Path>) -> Self {
        // Try explicit path first
        if let Some(p) = path {
            if let Ok(config) = Self::load(p) {
                return config;
            }
        }

        // Try default TOML location
        let default_toml = Path::new("config/detectors.toml");
        if default_toml.exists() {
            if let Ok(config) = Self::load(default_toml) {
                return config;
            }
        }

        // Try loading from hypersearch JSON files
        if let Some(config) = Self::from_hypersearch_results() {
            return config;
        }

        // Fall back to defaults
        Self::default()
    }

    /// Load from results/best_*.json files
    fn from_hypersearch_results() -> Option<Self> {
        use crate::HyperConfig;

        let mut config = CompareConfig::default();
        let mut found_any = false;

        // Load Sobel config
        if let Ok(contents) = std::fs::read_to_string("results/best_sobel.json") {
            if let Ok(hyper) = serde_json::from_str::<HyperConfig>(&contents) {
                config.sobel.threshold = hyper.threshold;
                config.sobel.window_size_us = hyper.window_size_us;
                config.sobel.filter_dead_pixels = hyper.filter_dead_pixels;
                found_any = true;
            }
        }

        // Load Canny config
        if let Ok(contents) = std::fs::read_to_string("results/best_canny.json") {
            if let Ok(hyper) = serde_json::from_str::<HyperConfig>(&contents) {
                config.canny.low_threshold = hyper.canny_low;
                config.canny.high_threshold = hyper.canny_high;
                config.canny.window_size_us = hyper.window_size_us;
                config.canny.filter_dead_pixels = hyper.filter_dead_pixels;
                found_any = true;
            }
        }

        // Load LoG config
        if let Ok(contents) = std::fs::read_to_string("results/best_log.json") {
            if let Ok(hyper) = serde_json::from_str::<HyperConfig>(&contents) {
                config.log.threshold = hyper.threshold;
                config.log.window_size_us = hyper.window_size_us;
                config.log.filter_dead_pixels = hyper.filter_dead_pixels;
                found_any = true;
            }
        }

        if found_any { Some(config) } else { None }
    }

    /// Save to TOML file
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let contents = toml::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, contents)?;
        Ok(())
    }
}

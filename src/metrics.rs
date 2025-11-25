//! Edge quality metrics computation
//!
//! Quantifies edge detection quality without visual inspection.

use bevy::prelude::*;

/// Comprehensive edge detection quality metrics
#[derive(Debug, Clone, Default, Resource)]
pub struct EdgeMetrics {
    // Basic counts
    pub edge_pixel_count: u32,
    pub total_pixels: u32,
    pub edge_density: f32,

    // Spatial distribution
    pub centroid: Vec2,
    pub std_dev: Vec2,
    pub bounding_box: (Vec2, Vec2), // (min, max)

    // Circular fit (for fan detection)
    pub circle_center: Vec2,
    pub circle_radius: f32,
    pub circle_fit_error: f32,
    pub circle_inlier_ratio: f32,

    // Angular distribution
    pub angular_peaks: Vec<f32>,
    pub detected_blade_count: u32,

    // Temporal stability
    pub frame_to_frame_iou: f32,
}

impl EdgeMetrics {
    /// Compute basic metrics from edge data
    pub fn compute_basic(pixels: &[f32], width: u32, height: u32) -> Self {
        let total_pixels = pixels.len() as u32;
        let mut edge_count = 0u32;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut min_x = width as f32;
        let mut max_x = 0.0f32;
        let mut min_y = height as f32;
        let mut max_y = 0.0f32;

        for (i, &value) in pixels.iter().enumerate() {
            if value > 0.0 {
                edge_count += 1;
                let x = (i % width as usize) as f32;
                let y = (i / width as usize) as f32;
                sum_x += x as f64;
                sum_y += y as f64;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }

        let centroid = if edge_count > 0 {
            Vec2::new(
                (sum_x / edge_count as f64) as f32,
                (sum_y / edge_count as f64) as f32,
            )
        } else {
            Vec2::new(width as f32 / 2.0, height as f32 / 2.0)
        };

        // Compute standard deviation
        let mut var_x = 0.0f64;
        let mut var_y = 0.0f64;
        if edge_count > 1 {
            for (i, &value) in pixels.iter().enumerate() {
                if value > 0.0 {
                    let x = (i % width as usize) as f32;
                    let y = (i / width as usize) as f32;
                    var_x += ((x - centroid.x) as f64).powi(2);
                    var_y += ((y - centroid.y) as f64).powi(2);
                }
            }
            var_x /= (edge_count - 1) as f64;
            var_y /= (edge_count - 1) as f64;
        }

        Self {
            edge_pixel_count: edge_count,
            total_pixels,
            edge_density: edge_count as f32 / total_pixels as f32,
            centroid,
            std_dev: Vec2::new(var_x.sqrt() as f32, var_y.sqrt() as f32),
            bounding_box: (Vec2::new(min_x, min_y), Vec2::new(max_x, max_y)),
            ..Default::default()
        }
    }
}

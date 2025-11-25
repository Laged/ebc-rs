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

/// Extract edge pixel coordinates from flat array
pub fn extract_edge_pixels(pixels: &[f32], width: u32) -> Vec<Vec2> {
    pixels
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            if v > 0.0 {
                let x = (i % width as usize) as f32;
                let y = (i / width as usize) as f32;
                Some(Vec2::new(x, y))
            } else {
                None
            }
        })
        .collect()
}

/// Fit circle through 3 points
/// Returns (center, radius) or None if points are collinear
fn fit_circle_3_points(p1: Vec2, p2: Vec2, p3: Vec2) -> Option<(Vec2, f32)> {
    let ax = p1.x;
    let ay = p1.y;
    let bx = p2.x;
    let by = p2.y;
    let cx = p3.x;
    let cy = p3.y;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-10 {
        return None; // Collinear points
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;

    let center = Vec2::new(ux, uy);
    let radius = center.distance(p1);

    Some((center, radius))
}

/// RANSAC circle fitting
/// Returns (center, radius, fit_error, inlier_ratio)
pub fn fit_circle_ransac(
    edge_pixels: &[Vec2],
    iterations: u32,
    inlier_threshold: f32,
) -> Option<(Vec2, f32, f32, f32)> {
    if edge_pixels.len() < 3 {
        return None;
    }

    let mut best_center = Vec2::ZERO;
    let mut best_radius = 0.0f32;
    let mut best_inliers = 0usize;
    let mut rng_state = 12345u32; // Simple LCG

    let rand_idx = |state: &mut u32, max: usize| -> usize {
        *state = state.wrapping_mul(1103515245).wrapping_add(12345);
        (*state as usize) % max
    };

    for _ in 0..iterations {
        // Sample 3 random points
        let i1 = rand_idx(&mut rng_state, edge_pixels.len());
        let i2 = rand_idx(&mut rng_state, edge_pixels.len());
        let i3 = rand_idx(&mut rng_state, edge_pixels.len());

        if i1 == i2 || i2 == i3 || i1 == i3 {
            continue;
        }

        let Some((center, radius)) =
            fit_circle_3_points(edge_pixels[i1], edge_pixels[i2], edge_pixels[i3])
        else {
            continue;
        };

        // Skip unreasonable circles
        if radius < 10.0 || radius > 500.0 {
            continue;
        }

        // Count inliers
        let inliers: usize = edge_pixels
            .iter()
            .filter(|p| (p.distance(center) - radius).abs() < inlier_threshold)
            .count();

        if inliers > best_inliers {
            best_inliers = inliers;
            best_center = center;
            best_radius = radius;
        }
    }

    if best_inliers < 10 {
        return None;
    }

    // Compute fit error as average distance from circle
    let total_error: f32 = edge_pixels
        .iter()
        .filter(|p| (p.distance(best_center) - best_radius).abs() < inlier_threshold)
        .map(|p| (p.distance(best_center) - best_radius).abs())
        .sum();

    let fit_error = total_error / best_inliers as f32;
    let inlier_ratio = best_inliers as f32 / edge_pixels.len() as f32;

    Some((best_center, best_radius, fit_error, inlier_ratio))
}

/// Build histogram of edge pixel angles relative to center
pub fn angular_histogram(edge_pixels: &[Vec2], center: Vec2, num_bins: usize) -> Vec<u32> {
    let mut histogram = vec![0u32; num_bins];

    for p in edge_pixels {
        let dx = p.x - center.x;
        let dy = p.y - center.y;
        let angle = dy.atan2(dx); // -PI to PI
        let normalized = (angle + std::f32::consts::PI) / std::f32::consts::TAU; // 0 to 1
        let bin = ((normalized * num_bins as f32) as usize).min(num_bins - 1);
        histogram[bin] += 1;
    }

    histogram
}

/// Find peaks in angular histogram (blade positions)
pub fn find_angular_peaks(histogram: &[u32], min_prominence: u32) -> Vec<f32> {
    let num_bins = histogram.len();
    let bin_size = std::f32::consts::TAU / num_bins as f32;
    let mut peaks = Vec::new();

    for i in 0..num_bins {
        let prev = histogram[(i + num_bins - 1) % num_bins];
        let curr = histogram[i];
        let next = histogram[(i + 1) % num_bins];

        // Local maximum with sufficient prominence
        if curr > prev && curr > next && curr >= min_prominence {
            let angle = (i as f32 + 0.5) * bin_size - std::f32::consts::PI;
            peaks.push(angle);
        }
    }

    peaks
}

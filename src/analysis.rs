//! Fan motion analysis from edge detection data
//!
//! This module receives GPU readback data and computes metrics
//! for fan geometry detection and RPM calculation.

use bevy::prelude::*;
use crate::metrics::{EdgeMetrics, extract_edge_pixels, fit_circle_ransac, angular_histogram, find_angular_peaks};

/// Edge data received from GPU readback
#[derive(Resource, Default, Clone)]
pub struct EdgeData {
    /// Edge pixel values from active detector
    pub pixels: Vec<f32>,
    /// Image dimensions
    pub width: u32,
    pub height: u32,
    /// Which detector this data came from
    pub detector: String,
    /// Whether new data is available
    pub updated: bool,
}

/// Channel receiver for edge data from render world
#[derive(Resource)]
pub struct EdgeDataReceiver(pub std::sync::Mutex<std::sync::mpsc::Receiver<EdgeData>>);

/// Channel sender for edge data (lives in render world)
#[derive(Resource)]
pub struct EdgeDataSender(pub std::sync::mpsc::Sender<EdgeData>);

/// Plugin for fan motion analysis
pub struct AnalysisPlugin;

impl Plugin for AnalysisPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EdgeData>()
            .init_resource::<EdgeMetrics>()
            .add_systems(Update, (receive_edge_data, compute_metrics).chain());
    }
}

fn receive_edge_data(
    receiver: Option<Res<EdgeDataReceiver>>,
    mut edge_data: ResMut<EdgeData>,
) {
    if let Some(receiver) = receiver {
        if let Ok(rx) = receiver.0.try_lock() {
            // Get the latest data (drain any backlog)
            while let Ok(data) = rx.try_recv() {
                *edge_data = data;
                edge_data.updated = true;
            }
        }
    }
}

fn compute_metrics(
    edge_data: Res<EdgeData>,
    mut metrics: ResMut<EdgeMetrics>,
) {
    if !edge_data.updated || edge_data.pixels.is_empty() {
        return;
    }

    // Compute basic metrics
    *metrics = EdgeMetrics::compute_basic(&edge_data.pixels, edge_data.width, edge_data.height);

    // Extract edge pixels for advanced analysis
    let edge_pixels = extract_edge_pixels(&edge_data.pixels, edge_data.width);

    if edge_pixels.len() < 100 {
        return; // Not enough edges for reliable analysis
    }

    // RANSAC circle fitting
    if let Some((center, radius, error, inlier_ratio)) =
        fit_circle_ransac(&edge_pixels, 200, 5.0)
    {
        metrics.circle_center = center;
        metrics.circle_radius = radius;
        metrics.circle_fit_error = error;
        metrics.circle_inlier_ratio = inlier_ratio;

        // Angular histogram from detected center
        let histogram = angular_histogram(&edge_pixels, center, 360);
        let peaks = find_angular_peaks(&histogram, 50);

        metrics.angular_peaks = peaks.clone();
        metrics.detected_blade_count = peaks.len() as u32;
    }
}

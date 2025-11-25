//! Multi-detector readback for metrics computation.

use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderDevice;
use std::sync::mpsc::{Sender, Receiver};
use std::sync::Mutex;

/// Metrics for a single detector
#[derive(Debug, Clone, Default)]
pub struct DetectorMetrics {
    pub edge_count: u32,
    pub tolerance_precision: f32,
    pub tolerance_recall: f32,
    pub tolerance_f1: f32,
    pub avg_distance: f32,
}

/// Combined metrics for all detectors
#[derive(Resource, Debug, Clone, Default)]
pub struct AllDetectorMetrics {
    pub raw: DetectorMetrics,
    pub sobel: DetectorMetrics,
    pub canny: DetectorMetrics,
    pub log: DetectorMetrics,
    pub frame_time_ms: f32,
    pub last_update: f64,
}

/// Channel to send metrics from render world to main world
#[derive(Resource)]
pub struct MetricsSender(pub Sender<AllDetectorMetrics>);

#[derive(Resource)]
pub struct MetricsReceiver(pub Mutex<Receiver<AllDetectorMetrics>>);

/// Per-detector edge data for metrics computation
#[derive(Debug, Clone, Default)]
pub struct DetectorEdgeData {
    pub pixels: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Combined edge data from all detectors
#[derive(Resource, Default)]
pub struct AllEdgeData {
    pub raw: DetectorEdgeData,
    pub sobel: DetectorEdgeData,
    pub canny: DetectorEdgeData,
    pub log: DetectorEdgeData,
}

/// System to receive metrics in main world
pub fn receive_metrics(
    receiver: Res<MetricsReceiver>,
    mut metrics: ResMut<AllDetectorMetrics>,
    time: Res<Time>,
) {
    if let Ok(rx) = receiver.0.try_lock() {
        while let Ok(new_metrics) = rx.try_recv() {
            *metrics = new_metrics;
            metrics.last_update = time.elapsed_secs_f64();
        }
    }
}

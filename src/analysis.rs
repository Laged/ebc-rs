//! Fan motion analysis from edge detection data
//!
//! This module receives GPU readback data and computes metrics
//! for fan geometry detection and RPM calculation.

use bevy::prelude::*;

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
            .add_systems(Update, receive_edge_data);
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

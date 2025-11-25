use bevy::prelude::*;

pub mod analysis;
pub mod compare;
pub mod edge_detection;
pub mod event_renderer;
pub mod gpu;
pub mod ground_truth;
pub mod hyperparams;
pub mod loader;
pub mod metrics;
pub mod playback;
pub mod synthesis;

#[derive(Resource, Clone)]
pub struct EventFilePath(pub String);

impl Default for EventFilePath {
    fn default() -> Self {
        Self("data/fan/fan_const_rpm.dat".to_string())
    }
}

pub use analysis::{AnalysisPlugin, EdgeData};
pub use ground_truth::{GroundTruthConfig, GroundTruthMetrics};
pub use hyperparams::{HyperConfig, HyperResult};
pub use loader::DatLoader;
pub use metrics::EdgeMetrics;

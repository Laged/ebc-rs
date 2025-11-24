use bevy::prelude::*;

pub mod analysis;
pub mod gizmos;
pub mod gpu;
pub mod loader;
pub mod mvp;
pub mod plugins;
pub mod render;
pub mod synthesis;

// New resource to hold the event file path
#[derive(Resource, Clone)]
pub struct EventFilePath(pub String);

impl Default for EventFilePath {
    fn default() -> Self {
        Self("data/fan/fan_const_rpm.dat".to_string()) // Default hardcoded path
    }
}

// Re-export commonly used items
pub use analysis::{AnalysisPlugin, FanAnalysis};
pub use gpu::{EventData, PlaybackState};
pub use loader::DatLoader;
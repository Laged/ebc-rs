use bevy::prelude::*;

pub mod analysis;
pub mod gizmos;
pub mod gpu;
pub mod loader;
pub mod plugins;
pub mod render;
pub mod synthesis; // Added this

// Re-export commonly used items

pub use analysis::{AnalysisPlugin, FanAnalysis};

pub use gpu::{EventData, PlaybackState};

pub use loader::DatLoader;

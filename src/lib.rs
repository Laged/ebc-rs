use bevy::prelude::*;

pub mod edge_detection;
pub mod event_renderer;
pub mod gpu;
pub mod loader;
pub mod playback;
pub mod synthesis;

#[derive(Resource, Clone)]
pub struct EventFilePath(pub String);

impl Default for EventFilePath {
    fn default() -> Self {
        Self("data/fan/fan_const_rpm.dat".to_string())
    }
}

pub use loader::DatLoader;

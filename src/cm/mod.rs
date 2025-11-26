// src/cm/mod.rs
//! Contrast Maximization for RPM estimation
//!
//! Replaces Canny edge detector with CM-based motion compensation.

mod resources;
mod pipeline;

pub use resources::*;
pub use pipeline::*;

use bevy::prelude::*;

/// Plugin for Contrast Maximization RPM estimation
pub struct CmPlugin;

impl Plugin for CmPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CmParams>()
           .init_resource::<CmResult>();
    }
}

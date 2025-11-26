// src/cm/mod.rs
//! Contrast Maximization for RPM estimation
//!
//! Replaces Canny edge detector with CM-based motion compensation.

mod resources;
mod pipeline;
mod systems;

pub use resources::*;
pub use pipeline::*;
pub use systems::*;

use bevy::prelude::*;
use bevy::render::{
    RenderApp, Render, RenderSystems,
    render_graph::RenderGraph,
    ExtractSchedule,
};

use crate::gpu::PreprocessLabel;
use crate::compare::CompositeLabel;

/// Plugin for Contrast Maximization RPM estimation
pub struct CmPlugin;

impl Plugin for CmPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CmParams>()
           .init_resource::<CmResult>();
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<CmPipeline>()
            .init_resource::<CmBindGroups>()
            .add_systems(ExtractSchedule, extract_cm_params)
            .add_systems(Render, prepare_cm.in_set(RenderSystems::Prepare));

        // Add to render graph
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(CmLabel, CmNode::default());
        graph.add_node_edge(PreprocessLabel, CmLabel);
        graph.add_node_edge(CmLabel, CompositeLabel);
    }
}

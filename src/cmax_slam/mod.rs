//! CMax-SLAM: Motion-compensated edge detection using Contrast Maximization
//!
//! Replaces the SOBEL quadrant with gradient-optimized motion compensation.

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

/// Plugin for CMax-SLAM motion-compensated edge detection
pub struct CmaxSlamPlugin;

impl Plugin for CmaxSlamPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CmaxSlamParams>()
           .init_resource::<CmaxSlamState>();
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<CmaxSlamPipeline>()
            .init_resource::<CmaxSlamBindGroups>()
            .add_systems(ExtractSchedule, extract_cmax_slam_params)
            .add_systems(Render, prepare_cmax_slam.in_set(RenderSystems::Prepare));

        // Add to render graph after Preprocess
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(CmaxSlamLabel, CmaxSlamNode::default());
        graph.add_node_edge(PreprocessLabel, CmaxSlamLabel);
    }
}

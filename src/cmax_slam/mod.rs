//! CMax-SLAM: Motion-compensated edge detection using Contrast Maximization
//!
//! Replaces the SOBEL quadrant with gradient-optimized motion compensation.

mod resources;
mod pipeline;
mod systems;
mod readback;

pub use resources::*;
pub use pipeline::*;
pub use systems::*;
pub use readback::{
    ContrastValues, ContrastReceiver, ContrastSender,
    GpuContrastResult, create_contrast_channel,
};

use bevy::prelude::*;
use bevy::render::{
    RenderApp, Render, RenderSystems,
    render_graph::RenderGraph,
    ExtractSchedule,
};

use crate::gpu::SobelLabel;

/// Plugin for CMax-SLAM motion-compensated edge detection
pub struct CmaxSlamPlugin;

impl Plugin for CmaxSlamPlugin {
    fn build(&self, app: &mut App) {
        // Create channel for contrast values
        let (sender, receiver) = create_contrast_channel();

        app.insert_resource(receiver)
           .init_resource::<CmaxSlamParams>()
           .init_resource::<CmaxSlamState>()
           .add_systems(Update, receive_contrast_results);

        // Add sender to render app
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(sender);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<CmaxSlamPipeline>()
            .init_resource::<CmaxSlamBindGroups>()
            .add_systems(ExtractSchedule, extract_cmax_slam_params)
            .add_systems(Render, prepare_cmax_slam.in_set(RenderSystems::Prepare));

        // Add to render graph AFTER Sobel - CMax-SLAM overwrites SobelImage
        // This ensures motion-compensated edges replace raw Sobel output
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(CmaxSlamLabel, CmaxSlamNode::default());
        graph.add_node_edge(SobelLabel, CmaxSlamLabel);
    }
}

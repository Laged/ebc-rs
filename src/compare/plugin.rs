//! Plugin that integrates composite rendering into the render graph.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_graph::RenderGraph;
use bevy::render::{Render, RenderApp, RenderSystems, ExtractSchedule, Extract};

use super::{
    CompositeBindGroup, CompositeImage, CompositeLabel, CompositeNode,
    CompositePipeline, prepare_composite,
    MetricsSender, PendingMetricsSender, MultiReadbackLabel, MultiReadbackBuffers,
    MultiReadbackNode, prepare_multi_readback, read_multi_readback,
    setup_metrics_channel, receive_metrics, AllDetectorMetrics,
};
use crate::gpu::{LogLabel, EdgeParams};
use crate::cm::CmLabel;

/// Plugin that adds composite rendering to the render graph
pub struct CompositeRenderPlugin;

impl Plugin for CompositeRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CompositeImage>()
            .add_plugins(ExtractResourcePlugin::<CompositeImage>::default());

        // Set up metrics channel (main world receiver, render world sender)
        setup_metrics_channel(app);

        // Add system to receive metrics in main world
        app.init_resource::<AllDetectorMetrics>()
            .add_systems(Update, receive_metrics);
    }

    fn finish(&self, app: &mut App) {
        // Custom extraction system for EdgeParams
        fn extract_edge_params_for_composite(
            mut commands: Commands,
            edge_params: Extract<Res<EdgeParams>>,
        ) {
            commands.insert_resource(edge_params.clone());
        }

        // Move sender to render world
        let sender = app.world_mut().remove_resource::<PendingMetricsSender>()
            .expect("PendingMetricsSender should exist");

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .insert_resource(MetricsSender(sender.0))
            .init_resource::<CompositePipeline>()
            .init_resource::<CompositeBindGroup>()
            .init_resource::<MultiReadbackBuffers>()
            .add_systems(ExtractSchedule, extract_edge_params_for_composite)
            .add_systems(Render, prepare_composite.in_set(RenderSystems::Queue))
            .add_systems(Render, prepare_multi_readback.in_set(RenderSystems::Queue))
            .add_systems(Render, read_multi_readback.in_set(RenderSystems::Cleanup));

        // Add composite node to render graph after LoG and CM
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(CompositeLabel, CompositeNode::default());
        render_graph.add_node_edge(LogLabel, CompositeLabel);
        render_graph.add_node_edge(CmLabel, CompositeLabel);

        // Add multi-readback node after composite
        render_graph.add_node(MultiReadbackLabel, MultiReadbackNode::default());
        render_graph.add_node_edge(CompositeLabel, MultiReadbackLabel);
        render_graph.add_node_edge(MultiReadbackLabel, bevy::render::graph::CameraDriverLabel);
    }
}

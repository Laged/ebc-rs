//! Plugin that integrates composite rendering into the render graph.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_graph::RenderGraph;
use bevy::render::{Render, RenderApp, RenderSystems};

use super::{
    CompositeBindGroup, CompositeImage, CompositeLabel, CompositeNode,
    CompositePipeline, prepare_composite,
};
use crate::gpu::LogLabel;

/// Plugin that adds composite rendering to the render graph
pub struct CompositeRenderPlugin;

impl Plugin for CompositeRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CompositeImage>()
            .add_plugins(ExtractResourcePlugin::<CompositeImage>::default());
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<CompositePipeline>()
            .init_resource::<CompositeBindGroup>()
            .add_systems(Render, prepare_composite.in_set(RenderSystems::Queue));

        // Add composite node to render graph after LoG
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(CompositeLabel, CompositeNode::default());
        render_graph.add_node_edge(LogLabel, CompositeLabel);
        render_graph.add_node_edge(CompositeLabel, bevy::render::graph::CameraDriverLabel);
    }
}

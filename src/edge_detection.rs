use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::{RenderApp, Render, RenderSystems, ExtractSchedule, Extract};
use bevy::render::render_graph::RenderGraph;
use crate::gpu::*;
use crate::event_renderer::EventRendererPlugin;

pub struct EdgeDetectionPlugin;

impl Plugin for EdgeDetectionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SurfaceImage>()
            .init_resource::<SobelImage>()
            .init_resource::<CannyImage>()
            .init_resource::<LogImage>()
            .init_resource::<crate::playback::PlaybackState>()
            .init_resource::<EdgeParams>()
            .add_plugins(ExtractResourcePlugin::<EventData>::default())
            .add_plugins(ExtractResourcePlugin::<SurfaceImage>::default())
            .add_plugins(ExtractResourcePlugin::<SobelImage>::default())
            .add_plugins(ExtractResourcePlugin::<CannyImage>::default())
            .add_plugins(ExtractResourcePlugin::<LogImage>::default())
            .add_plugins(ExtractResourcePlugin::<crate::playback::PlaybackState>::default())
            .add_plugins(EventRendererPlugin);
    }

    fn finish(&self, app: &mut App) {
        // Custom extraction system for EdgeParams
        fn extract_edge_params(
            mut commands: Commands,
            edge_params: Extract<Res<EdgeParams>>,
        ) {
            commands.insert_resource(edge_params.clone());
        }

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<EventComputePipeline>()
            .init_resource::<SobelPipeline>()
            .init_resource::<CannyPipeline>()
            .init_resource::<LogPipeline>()
            .init_resource::<GpuEventBuffer>()
            .add_systems(ExtractSchedule, extract_edge_params)
            .add_systems(Render, prepare_events.in_set(RenderSystems::Prepare))
            .add_systems(Render, queue_bind_group.in_set(RenderSystems::Queue))
            .add_systems(Render, prepare_sobel.in_set(RenderSystems::Queue))
            .add_systems(Render, prepare_canny.in_set(RenderSystems::Queue))
            .add_systems(Render, prepare_log.in_set(RenderSystems::Queue));

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(EventLabel, EventAccumulationNode::default());
        render_graph.add_node(SobelLabel, SobelNode::default());
        render_graph.add_node(CannyLabel, CannyNode::default());
        render_graph.add_node(LogLabel, LogNode::default());
        // Render graph: Event → Sobel → Canny → LoG → Camera
        render_graph.add_node_edge(EventLabel, SobelLabel);
        render_graph.add_node_edge(SobelLabel, CannyLabel);
        render_graph.add_node_edge(CannyLabel, LogLabel);
        render_graph.add_node_edge(LogLabel, bevy::render::graph::CameraDriverLabel);
    }
}

use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue, RenderContext},
    render_graph::{Node, RenderLabel},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use super::types::GpuEdgeParams;
use super::resources::{EdgeParams, SurfaceImage, GradientImage};

#[derive(Resource)]
pub struct GradientPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for GradientPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Gradient Pipeline Layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Uint,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/spatial_gradient.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Spatial Gradient Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        GradientPipeline { layout, pipeline }
    }
}

#[derive(Resource)]
pub struct GradientBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct EdgeParamsBuffer(pub Buffer);

pub fn prepare_gradient(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<GradientPipeline>,
    edge_params: Res<EdgeParams>,
    surface_image: Res<SurfaceImage>,
    gradient_image: Res<GradientImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    edge_buffer: Option<Res<EdgeParamsBuffer>>,
) {
    // Pack all edge params with correct types matching WGSL struct
    let gpu_params = GpuEdgeParams {
        threshold: edge_params.threshold,
        filter_dead_pixels: if edge_params.filter_dead_pixels { 1 } else { 0 },
        filter_density: if edge_params.filter_density { 1 } else { 0 },
        filter_bidirectional: if edge_params.filter_bidirectional { 1 } else { 0 },
        filter_temporal: if edge_params.filter_temporal { 1 } else { 0 },
    };

    // Create or update edge params buffer
    let buffer = if let Some(existing) = edge_buffer {
        render_queue.write_buffer(&existing.0, 0, bytemuck::bytes_of(&gpu_params));
        existing.0.clone()
    } else {
        let new_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Edge Params Buffer"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        commands.insert_resource(EdgeParamsBuffer(new_buffer.clone()));
        new_buffer
    };

    // Create bind group if textures are ready
    if let (Some(surface_gpu), Some(gradient_gpu)) = (
        gpu_images.get(&surface_image.handle),
        gpu_images.get(&gradient_image.handle),
    ) {
        let bind_group = render_device.create_bind_group(
            Some("Gradient Bind Group"),
            &pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&surface_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gradient_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        );
        commands.insert_resource(GradientBindGroup(bind_group));
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct GradientLabel;

#[derive(Default)]
pub struct GradientNode;

impl Node for GradientNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GradientPipeline>();
        let Some(bind_group) = world.get_resource::<GradientBindGroup>() else {
            return Ok(());
        };

        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Spatial Gradient"),
                    timestamp_writes: None,
                });

            pass.set_pipeline(compute_pipeline);
            pass.set_bind_group(0, &bind_group.0, &[]);

            // Dispatch for 1280x720 with 8x8 workgroups
            let workgroups_x = (1280 + 7) / 8;
            let workgroups_y = (720 + 7) / 8;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        Ok(())
    }
}

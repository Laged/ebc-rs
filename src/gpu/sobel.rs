use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue, RenderContext},
    render_graph::{Node, RenderLabel},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use super::types::GpuParams;
use super::resources::{EdgeParams, FilteredSurfaceImage, SobelImage};

#[derive(Resource)]
pub struct SobelPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for SobelPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Sobel Pipeline Layout"),
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
            .load("shaders/sobel.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Sobel Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        SobelPipeline { layout, pipeline }
    }
}

#[derive(Resource)]
pub struct SobelBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct EdgeParamsBuffer(pub Buffer);

pub fn prepare_sobel(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<SobelPipeline>,
    edge_params: Res<EdgeParams>,
    filtered_image: Res<FilteredSurfaceImage>,
    sobel_image: Res<SobelImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    edge_buffer: Option<Res<EdgeParamsBuffer>>,
) {
    // Pack all edge params with correct types matching WGSL struct
    let gpu_params = GpuParams {
        filter_dead_pixels: if edge_params.filter_dead_pixels { 1 } else { 0 },
        filter_density: if edge_params.filter_density { 1 } else { 0 },
        filter_temporal: if edge_params.filter_temporal { 1 } else { 0 },
        min_density_count: edge_params.min_density_count,
        min_temporal_spread: edge_params.min_temporal_spread_us,
        sobel_threshold: edge_params.sobel_threshold,
        canny_low_threshold: edge_params.canny_low_threshold,
        canny_high_threshold: edge_params.canny_high_threshold,
        log_threshold: edge_params.log_threshold,
        filter_bidirectional: if edge_params.filter_bidirectional { 1 } else { 0 },
        bidirectional_ratio: edge_params.bidirectional_ratio,
        _padding: 0.0,
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
    // Now uses FilteredSurfaceImage instead of SurfaceImage (pre-filtered by preprocess stage)
    if let (Some(filtered_gpu), Some(sobel_gpu)) = (
        gpu_images.get(&filtered_image.handle),
        gpu_images.get(&sobel_image.handle),
    ) {
        let bind_group = render_device.create_bind_group(
            Some("Sobel Bind Group"),
            &pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&filtered_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&sobel_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        );
        commands.insert_resource(SobelBindGroup(bind_group));
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct SobelLabel;

#[derive(Default)]
pub struct SobelNode;

impl Node for SobelNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<SobelPipeline>();
        let Some(bind_group) = world.get_resource::<SobelBindGroup>() else {
            return Ok(());
        };

        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Sobel"),
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

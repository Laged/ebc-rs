use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue, RenderContext},
    render_graph::{Node, RenderLabel},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use bytemuck::{Pod, Zeroable};
use super::resources::{EdgeParams, SurfaceImage, LogImage};

// GPU-compatible LogParams struct that matches WGSL layout
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuLogParams {
    pub sigma: f32,
    pub threshold: f32,
}

#[derive(Resource)]
pub struct LogPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for LogPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("LoG Pipeline Layout"),
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
            .load("shaders/log.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("LoG Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        LogPipeline { layout, pipeline }
    }
}

#[derive(Resource)]
pub struct LogBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct LogParamsBuffer(pub Buffer);

pub fn prepare_log(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<LogPipeline>,
    edge_params: Res<EdgeParams>,
    surface_image: Res<SurfaceImage>,
    log_image: Res<LogImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    log_buffer: Option<Res<LogParamsBuffer>>,
) {
    // Pack LoG params with correct types matching WGSL struct
    let gpu_params = GpuLogParams {
        sigma: edge_params.log_sigma,
        threshold: edge_params.log_threshold,
    };

    // Create or update LoG params buffer
    let buffer = if let Some(existing) = log_buffer {
        render_queue.write_buffer(&existing.0, 0, bytemuck::bytes_of(&gpu_params));
        existing.0.clone()
    } else {
        let new_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("LoG Params Buffer"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        commands.insert_resource(LogParamsBuffer(new_buffer.clone()));
        new_buffer
    };

    // Create bind group if textures are ready
    if let (Some(surface_gpu), Some(log_gpu)) = (
        gpu_images.get(&surface_image.handle),
        gpu_images.get(&log_image.handle),
    ) {
        let bind_group = render_device.create_bind_group(
            Some("LoG Bind Group"),
            &pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&surface_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&log_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        );
        commands.insert_resource(LogBindGroup(bind_group));
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct LogLabel;

#[derive(Default)]
pub struct LogNode;

impl Node for LogNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<LogPipeline>();
        let Some(bind_group) = world.get_resource::<LogBindGroup>() else {
            return Ok(());
        };

        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("LoG"),
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

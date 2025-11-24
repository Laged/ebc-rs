use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue, RenderContext},
    render_graph::{Node, RenderLabel},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use bytemuck::{Pod, Zeroable};
use super::resources::{EdgeParams, SurfaceImage, CannyImage};

// GPU-compatible CannyParams struct that matches WGSL layout
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuCannyParams {
    pub sigma: f32,
    pub low_threshold: f32,
    pub high_threshold: f32,
    pub _padding: f32,
}

#[derive(Resource)]
pub struct CannyPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for CannyPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Canny Pipeline Layout"),
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
            .load("shaders/canny.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Canny Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        CannyPipeline { layout, pipeline }
    }
}

#[derive(Resource)]
pub struct CannyBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct CannyParamsBuffer(pub Buffer);

pub fn prepare_canny(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<CannyPipeline>,
    edge_params: Res<EdgeParams>,
    surface_image: Res<SurfaceImage>,
    canny_image: Res<CannyImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    canny_buffer: Option<Res<CannyParamsBuffer>>,
) {
    // Pack Canny params with correct types matching WGSL struct
    let gpu_params = GpuCannyParams {
        sigma: edge_params.canny_sigma,
        low_threshold: edge_params.canny_low_threshold,
        high_threshold: edge_params.canny_high_threshold,
        _padding: 0.0,
    };

    // Create or update Canny params buffer
    let buffer = if let Some(existing) = canny_buffer {
        render_queue.write_buffer(&existing.0, 0, bytemuck::bytes_of(&gpu_params));
        existing.0.clone()
    } else {
        let new_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Canny Params Buffer"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        commands.insert_resource(CannyParamsBuffer(new_buffer.clone()));
        new_buffer
    };

    // Create bind group if textures are ready
    if let (Some(surface_gpu), Some(canny_gpu)) = (
        gpu_images.get(&surface_image.handle),
        gpu_images.get(&canny_image.handle),
    ) {
        let bind_group = render_device.create_bind_group(
            Some("Canny Bind Group"),
            &pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&surface_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&canny_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        );
        commands.insert_resource(CannyBindGroup(bind_group));
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct CannyLabel;

#[derive(Default)]
pub struct CannyNode;

impl Node for CannyNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<CannyPipeline>();
        let Some(bind_group) = world.get_resource::<CannyBindGroup>() else {
            return Ok(());
        };

        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Canny"),
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

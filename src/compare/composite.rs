//! GPU composite pipeline for 2x2 grid rendering.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;

use crate::gpu::{FilteredSurfaceImage, SobelImage, CannyImage, LogImage, EdgeParams};

/// Composite shader parameters (matching WGSL layout)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompositeParams {
    pub show_raw: u32,
    pub show_sobel: u32,
    pub show_canny: u32,
    pub show_log: u32,
}

/// Output composite image (2560x1440)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct CompositeImage {
    pub handle: Handle<Image>,
}

/// Label for composite render node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct CompositeLabel;

/// Composite compute pipeline
#[derive(Resource)]
pub struct CompositePipeline {
    pub pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
}

impl FromWorld for CompositePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "composite_bind_group_layout",
            &[
                // Raw/filtered surface (input) - R32Uint
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
                // Sobel (input)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Canny (input)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // LoG (input)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output composite (output)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // CompositeParams uniform buffer
                BindGroupLayoutEntry {
                    binding: 5,
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

        let shader = world.load_asset("shaders/composite.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("composite_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

/// Composite bind group
#[derive(Resource, Default)]
pub struct CompositeBindGroup {
    pub bind_group: Option<BindGroup>,
}

/// Composite parameters buffer
#[derive(Resource)]
pub struct CompositeParamsBuffer(pub Buffer);

/// System to prepare composite bind group
pub fn prepare_composite(
    mut commands: Commands,
    mut bind_group: ResMut<CompositeBindGroup>,
    pipeline: Res<CompositePipeline>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    filtered_image: Res<FilteredSurfaceImage>,
    sobel_image: Res<SobelImage>,
    canny_image: Res<CannyImage>,
    log_image: Res<LogImage>,
    composite_image: Res<CompositeImage>,
    edge_params: Res<EdgeParams>,
    params_buffer: Option<Res<CompositeParamsBuffer>>,
) {
    let (Some(filtered), Some(sobel), Some(canny), Some(log), Some(composite)) = (
        gpu_images.get(&filtered_image.handle),
        gpu_images.get(&sobel_image.handle),
        gpu_images.get(&canny_image.handle),
        gpu_images.get(&log_image.handle),
        gpu_images.get(&composite_image.handle),
    ) else {
        return;
    };

    // Create CompositeParams from EdgeParams
    let params = CompositeParams {
        show_raw: if edge_params.show_raw { 1 } else { 0 },
        show_sobel: if edge_params.show_sobel { 1 } else { 0 },
        show_canny: if edge_params.show_canny { 1 } else { 0 },
        show_log: if edge_params.show_log { 1 } else { 0 },
    };

    // Create or reuse params buffer
    if params_buffer.is_none() {
        let new_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("composite_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        commands.insert_resource(CompositeParamsBuffer(new_buffer));
        return; // Wait for next frame to have the buffer resource
    }

    let params_buffer = params_buffer.unwrap();
    render_queue.write_buffer(&params_buffer.0, 0, bytemuck::bytes_of(&params));

    bind_group.bind_group = Some(render_device.create_bind_group(
        "composite_bind_group",
        &pipeline.bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&filtered.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&sobel.texture_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&canny.texture_view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&log.texture_view),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&composite.texture_view),
            },
            BindGroupEntry {
                binding: 5,
                resource: params_buffer.0.as_entire_binding(),
            },
        ],
    ));
}

/// Composite render node
#[derive(Default)]
pub struct CompositeNode;

impl Node for CompositeNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_res = world.resource::<CompositePipeline>();
        let bind_group = world.resource::<CompositeBindGroup>();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.pipeline) else {
            return Ok(());
        };

        let Some(ref bind_group) = bind_group.bind_group else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("composite_pass"),
                timestamp_writes: None,
            });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        // Dispatch for 2560x1440 output with 8x8 workgroups
        let workgroups_x = (2560 + 7) / 8;
        let workgroups_y = (1440 + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);

        Ok(())
    }
}

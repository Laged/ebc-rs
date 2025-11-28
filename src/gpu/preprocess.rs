use bevy::prelude::*;
use bevy::render::{
    render_graph::{Node, RenderLabel},
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use super::resources::{FilteredSurfaceImage, SurfaceImage, ShortWindowSurfaceImage, EdgeParams};
use super::types::GpuParams;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct PreprocessLabel;

#[derive(Resource)]
pub struct PreprocessPipeline {
    pub pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
}

impl FromWorld for PreprocessPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            Some("Preprocess Bind Group Layout"),
            &[
                // Surface texture input
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
                // Filtered output
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Uint,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Params uniform
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuParams::min_size()),
                    },
                    count: None,
                },
            ],
        );

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/preprocess.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Preprocess Pipeline".into()),
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

#[derive(Resource, Default)]
pub struct PreprocessBindGroup {
    pub bind_group: Option<BindGroup>,
    pub params_buffer: Option<Buffer>,
}

pub fn prepare_preprocess(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<PreprocessPipeline>,
    mut bind_group_res: ResMut<PreprocessBindGroup>,
    surface_image: Res<ShortWindowSurfaceImage>,
    filtered_image: Res<FilteredSurfaceImage>,
    edge_params: Res<EdgeParams>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    let Some(surface_gpu) = gpu_images.get(&surface_image.handle) else {
        return;
    };
    let Some(filtered_gpu) = gpu_images.get(&filtered_image.handle) else {
        return;
    };

    // Create/update params buffer
    let gpu_params = GpuParams {
        filter_dead_pixels: edge_params.filter_dead_pixels as u32,
        filter_density: edge_params.filter_density as u32,
        filter_temporal: edge_params.filter_temporal as u32,
        min_density_count: edge_params.min_density_count,
        min_temporal_spread: edge_params.min_temporal_spread_us,
        sobel_threshold: edge_params.sobel_threshold,
        canny_low_threshold: edge_params.canny_low_threshold,
        canny_high_threshold: edge_params.canny_high_threshold,
        log_threshold: edge_params.log_threshold,
        filter_bidirectional: edge_params.filter_bidirectional as u32,
        bidirectional_ratio: edge_params.bidirectional_ratio,
        _padding: 0.0,
    };

    let params_buffer = bind_group_res.params_buffer.get_or_insert_with(|| {
        render_device.create_buffer(&BufferDescriptor {
            label: Some("Preprocess Params Buffer"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });

    render_queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&gpu_params));

    let bind_group = render_device.create_bind_group(
        Some("Preprocess Bind Group"),
        &pipeline.bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&surface_gpu.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&filtered_gpu.texture_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    );

    bind_group_res.bind_group = Some(bind_group);
}

#[derive(Default)]
pub struct PreprocessNode;

impl Node for PreprocessNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_res = world.resource::<PreprocessPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let bind_group_res = world.resource::<PreprocessBindGroup>();

        let Some(bind_group) = &bind_group_res.bind_group else {
            return Ok(());
        };

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.pipeline) else {
            return Ok(());
        };

        let filtered_image = world.resource::<FilteredSurfaceImage>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let Some(filtered_gpu) = gpu_images.get(&filtered_image.handle) else {
            return Ok(());
        };

        let width = filtered_gpu.texture.width();
        let height = filtered_gpu.texture.height();

        let mut pass = render_context.command_encoder().begin_compute_pass(
            &ComputePassDescriptor {
                label: Some("Preprocess Pass"),
                timestamp_writes: None,
            },
        );

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);

        Ok(())
    }
}

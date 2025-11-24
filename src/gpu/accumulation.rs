use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue, RenderContext},
    render_graph::{Node, RenderLabel},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use super::resources::{EventData, SurfaceImage, GpuEventBuffer};

#[derive(Resource)]
pub struct EventComputePipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for EventComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Event Accumulation Layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
            .load("shaders/accumulation.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Event Accumulation Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        EventComputePipeline { layout, pipeline }
    }
}

#[derive(Resource)]
pub struct EventBindGroup(pub BindGroup);

pub fn prepare_events(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gpu_buffer: ResMut<GpuEventBuffer>,
    event_data: Res<EventData>,
    playback_state: Res<crate::playback::PlaybackState>,
) {
    if event_data.events.is_empty() {
        return;
    }

    let width = 1280;
    let height = 720;

    // Calculate time window
    let window_end = playback_state.current_time as u32;
    let window_start = if window_end > playback_state.window_size as u32 {
        window_end - playback_state.window_size as u32
    } else {
        0
    };

    // Update dimension buffer every frame
    if let Some(dim_buffer) = &gpu_buffer.dim_buffer {
        let dimensions = [width, height, window_start, window_end];
        render_queue.write_buffer(dim_buffer, 0, bytemuck::cast_slice(&dimensions));
    }

    if gpu_buffer.uploaded {
        return;
    }

    // One-time upload
    info!("Uploading {} events to GPU", event_data.events.len());

    let byte_data: &[u8] = bytemuck::cast_slice(&event_data.events);
    gpu_buffer.buffer = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Event Buffer"),
            contents: byte_data,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        }),
    );
    gpu_buffer.count = event_data.events.len() as u32;
    gpu_buffer.dimensions = UVec2::new(width, height);

    let dimensions = [width, height, 0u32, 20000u32];
    gpu_buffer.dim_buffer = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dimensions),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }),
    );

    let size = width * height * 4;
    gpu_buffer.surface_buffer = Some(render_device.create_buffer(&BufferDescriptor {
        label: Some("Surface Buffer"),
        size: size as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    render_queue.write_buffer(
        gpu_buffer.surface_buffer.as_ref().unwrap(),
        0,
        &vec![0u8; size as usize],
    );

    gpu_buffer.uploaded = true;
}

pub fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<EventComputePipeline>,
    render_device: Res<RenderDevice>,
    mut gpu_buffer: ResMut<GpuEventBuffer>,
) {
    if gpu_buffer.bind_group_ready {
        return;
    }

    if let (Some(events), Some(surface), Some(dim_buffer)) = (
        &gpu_buffer.buffer,
        &gpu_buffer.surface_buffer,
        &gpu_buffer.dim_buffer,
    ) {
        let bind_group = render_device.create_bind_group(
            Some("Event Bind Group"),
            &pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: events.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: surface.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dim_buffer.as_entire_binding(),
                },
            ],
        );
        commands.insert_resource(EventBindGroup(bind_group));
        gpu_buffer.bind_group_ready = true;
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct EventLabel;

#[derive(Default)]
pub struct EventAccumulationNode;

impl Node for EventAccumulationNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<EventComputePipeline>();
        let Some(bind_group) = world.get_resource::<EventBindGroup>() else {
            return Ok(());
        };
        let gpu_buffer = world.resource::<GpuEventBuffer>();
        let surface_image = world.resource::<SurfaceImage>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();

        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            // Clear surface buffer
            if let Some(surface_buffer) = &gpu_buffer.surface_buffer {
                render_context
                    .command_encoder()
                    .clear_buffer(surface_buffer, 0, None);
            }

            // Run compute pass
            {
                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Event Accumulation"),
                        timestamp_writes: None,
                    });

                pass.set_pipeline(compute_pipeline);
                pass.set_bind_group(0, &bind_group.0, &[]);

                let workgroup_size = 64;
                let count = gpu_buffer.count;
                if count > 0 {
                    let total_workgroups = (count + workgroup_size - 1) / workgroup_size;
                    let max_workgroups_per_dim = 65535;

                    let x_workgroups = total_workgroups.min(max_workgroups_per_dim);
                    let y_workgroups =
                        (total_workgroups + max_workgroups_per_dim - 1) / max_workgroups_per_dim;

                    pass.dispatch_workgroups(x_workgroups, y_workgroups, 1);
                }
            }

            // Copy buffer to texture
            if let Some(surface_buffer) = &gpu_buffer.surface_buffer {
                if let Some(gpu_image) = gpu_images.get(&surface_image.handle) {
                    render_context.command_encoder().copy_buffer_to_texture(
                        TexelCopyBufferInfo {
                            buffer: surface_buffer,
                            layout: TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(gpu_buffer.dimensions.x * 4),
                                rows_per_image: Some(gpu_buffer.dimensions.y),
                            },
                        },
                        TexelCopyTextureInfo {
                            texture: &gpu_image.texture,
                            mip_level: 0,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        Extent3d {
                            width: gpu_buffer.dimensions.x,
                            height: gpu_buffer.dimensions.y,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
        }
        Ok(())
    }
}

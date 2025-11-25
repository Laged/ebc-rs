//! GPU pipeline for ground truth blade rendering.

use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue, RenderContext},
    render_graph::{Node, RenderLabel},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use crate::ground_truth::GroundTruthConfig;
use crate::playback::PlaybackState;
use super::resources::GroundTruthImage;

/// GPU uniform buffer for ground truth parameters
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGroundTruthParams {
    pub center_x: f32,
    pub center_y: f32,
    pub r_min: f32,
    pub r_max: f32,
    pub blade_count: u32,
    pub angular_velocity: f32,
    pub current_time: f32,
    pub sweep_k: f32,
    pub width_root: f32,
    pub width_tip: f32,
    pub edge_thickness: f32,
    pub _padding: f32,
}

#[derive(Resource)]
pub struct GroundTruthPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for GroundTruthPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Ground Truth Pipeline Layout"),
            &[
                // Output texture (RGBA8)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Params uniform buffer
                BindGroupLayoutEntry {
                    binding: 1,
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
            .load("shaders/ground_truth.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Ground Truth Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        GroundTruthPipeline { layout, pipeline }
    }
}

#[derive(Resource, Default)]
pub struct GroundTruthBindGroup {
    pub bind_group: Option<BindGroup>,
    pub params_buffer: Option<Buffer>,
}

pub fn prepare_ground_truth(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<GroundTruthPipeline>,
    config: Res<GroundTruthConfig>,
    playback: Res<PlaybackState>,
    gt_image: Res<GroundTruthImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    bind_group_res: Option<Res<GroundTruthBindGroup>>,
) {
    // Skip if ground truth is disabled
    if !config.enabled {
        return;
    }

    // Pack params for GPU
    let gpu_params = GpuGroundTruthParams {
        center_x: config.center_x,
        center_y: config.center_y,
        r_min: config.radius_min,
        r_max: config.radius_max,
        blade_count: config.blade_count,
        angular_velocity: config.angular_velocity(),
        current_time: playback.current_time / 1_000_000.0, // Convert us to seconds
        sweep_k: config.sweep_k,
        width_root: config.width_root_rad,
        width_tip: config.width_tip_rad,
        edge_thickness: config.edge_thickness_px,
        _padding: 0.0,
    };

    // Create or update params buffer
    let buffer = if let Some(ref existing_res) = bind_group_res {
        if let Some(ref existing) = existing_res.params_buffer {
            render_queue.write_buffer(existing, 0, bytemuck::bytes_of(&gpu_params));
            existing.clone()
        } else {
            let new_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("Ground Truth Params Buffer"),
                contents: bytemuck::bytes_of(&gpu_params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            new_buffer
        }
    } else {
        let new_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Ground Truth Params Buffer"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        new_buffer
    };

    // Create bind group if texture is ready
    if let Some(gt_gpu) = gpu_images.get(&gt_image.handle) {
        let bind_group = render_device.create_bind_group(
            Some("Ground Truth Bind Group"),
            &pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&gt_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        );
        commands.insert_resource(GroundTruthBindGroup {
            bind_group: Some(bind_group),
            params_buffer: Some(buffer),
        });
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct GroundTruthLabel;

#[derive(Default)]
pub struct GroundTruthNode;

impl Node for GroundTruthNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        // Check if ground truth is enabled
        let config = world.resource::<GroundTruthConfig>();
        if !config.enabled {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GroundTruthPipeline>();
        let Some(bind_group_res) = world.get_resource::<GroundTruthBindGroup>() else {
            return Ok(());
        };

        let Some(ref bind_group) = bind_group_res.bind_group else {
            return Ok(());
        };

        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Ground Truth"),
                    timestamp_writes: None,
                });

            pass.set_pipeline(compute_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            // Dispatch for 1280x720 with 8x8 workgroups
            let workgroups_x = (1280 + 7) / 8;
            let workgroups_y = (720 + 7) / 8;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        Ok(())
    }
}

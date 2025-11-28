//! GPU pipeline for CMax-SLAM gradient-based optimization

use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderContext},
    render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
};

/// Render graph label for CMax-SLAM node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct CmaxSlamLabel;

/// CMax-SLAM Pipeline resources
#[derive(Resource)]
pub struct CmaxSlamPipeline {
    /// Warp + contrast compute pipeline
    pub warp_contrast_pipeline: CachedComputePipelineId,
    pub warp_contrast_layout: BindGroupLayout,
    /// Output copy pipeline
    pub output_pipeline: CachedComputePipelineId,
    pub output_layout: BindGroupLayout,
    /// Reduction pipeline
    pub reduce_pipeline: CachedComputePipelineId,
    pub reduce_layout: BindGroupLayout,
}

impl FromWorld for CmaxSlamPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();

        // Warp + Contrast layout: events, params, IWE buffer, contrast results
        let warp_contrast_layout = render_device.create_bind_group_layout(
            "cmax_slam_warp_contrast_layout",
            &[
                // Events buffer
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
                // Params uniform
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
                // IWE buffer (3 slices: center, plus, minus)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Contrast results buffer
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        // Output layout: IWE buffer, params, output texture
        let output_layout = render_device.create_bind_group_layout(
            "cmax_slam_output_layout",
            &[
                // IWE buffer (read center slice)
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
                // Params uniform
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
                // Output texture (Sobel slot)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        );

        let pipeline_cache = world.resource::<PipelineCache>();

        let warp_contrast_shader = asset_server.load("shaders/cmax_slam_warp.wgsl");
        let warp_contrast_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cmax_slam_warp_contrast_pipeline".into()),
            layout: vec![warp_contrast_layout.clone()],
            shader: warp_contrast_shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let output_shader = asset_server.load("shaders/cmax_slam_output.wgsl");
        let output_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cmax_slam_output_pipeline".into()),
            layout: vec![output_layout.clone()],
            shader: output_shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        // Reduction shader pipeline
        let reduce_shader = asset_server.load("shaders/cmax_slam_reduce.wgsl");

        let reduce_layout = render_device.create_bind_group_layout(
            "cmax_slam_reduce_layout",
            &[
                // IWE buffer (read)
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
                // Contrast result (read_write)
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
            ],
        );

        let reduce_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cmax_slam_reduce_pipeline".into()),
            layout: vec![reduce_layout.clone()],
            shader: reduce_shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        Self {
            warp_contrast_pipeline,
            warp_contrast_layout,
            output_pipeline,
            output_layout,
            reduce_pipeline,
            reduce_layout,
        }
    }
}

/// Bind groups for CMax-SLAM pipeline
#[derive(Resource, Default)]
pub struct CmaxSlamBindGroups {
    pub warp_contrast: Option<BindGroup>,
    pub output: Option<BindGroup>,
    pub reduce: Option<BindGroup>,
}

/// GPU buffers for CMax-SLAM
#[derive(Resource, Clone)]
pub struct CmaxSlamBuffers {
    pub params: Buffer,
    pub iwe: Buffer,              // 3 slices: center, +delta, -delta
    pub contrast: Buffer,         // GpuCmaxSlamResult
    pub contrast_result: Buffer,  // Reduction output (STORAGE | COPY_SRC)
    pub contrast_staging: Buffer, // Async readback (MAP_READ | COPY_DST)
}

/// CMax-SLAM render node
#[derive(Default)]
pub struct CmaxSlamNode;

impl Node for CmaxSlamNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<CmaxSlamPipeline>();
        let Some(bind_groups) = world.get_resource::<CmaxSlamBindGroups>() else {
            return Ok(());
        };

        let (Some(warp_bg), Some(output_bg), Some(reduce_bg)) = (&bind_groups.warp_contrast, &bind_groups.output, &bind_groups.reduce) else {
            return Ok(());
        };

        // Get pipelines
        let Some(warp_pl) = pipeline_cache.get_compute_pipeline(pipeline.warp_contrast_pipeline) else {
            return Ok(());
        };
        let Some(output_pl) = pipeline_cache.get_compute_pipeline(pipeline.output_pipeline) else {
            return Ok(());
        };
        let Some(reduce_pl) = pipeline_cache.get_compute_pipeline(pipeline.reduce_pipeline) else {
            return Ok(());
        };

        // Check if enabled
        let extracted = world.get_resource::<super::ExtractedCmaxSlamParams>();
        if extracted.map(|e| !e.enabled).unwrap_or(true) {
            return Ok(());
        }

        let Some(buffers) = world.get_resource::<CmaxSlamBuffers>() else {
            return Ok(());
        };

        let encoder = render_context.command_encoder();

        // Clear IWE buffer
        encoder.clear_buffer(&buffers.iwe, 0, None);

        // Pass 1: Warp events and compute contrast for 3 omega values
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cmax_slam_warp_contrast"),
                timestamp_writes: None,
            });
            pass.set_pipeline(warp_pl);
            pass.set_bind_group(0, warp_bg, &[]);
            // Dispatch for event count
            pass.dispatch_workgroups(4096, 1, 1);
        }

        // Clear contrast result buffer before reduction
        encoder.clear_buffer(&buffers.contrast_result, 0, None);

        // Pass 2: Reduction - compute sum of squares for each IWE slice
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cmax_slam_reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(reduce_pl);
            pass.set_bind_group(0, reduce_bg, &[]);

            // Dispatch enough workgroups to cover all pixels
            // SLICE_SIZE = 1280 * 720 = 921600
            // Workgroups needed = ceil(921600 / 256) = 3600
            let workgroups = (921600 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy result to staging buffer for async readback
        encoder.copy_buffer_to_buffer(
            &buffers.contrast_result,
            0,
            &buffers.contrast_staging,
            0,
            std::mem::size_of::<super::GpuContrastResult>() as u64,
        );

        // Pass 3: Copy best IWE to output texture
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cmax_slam_output"),
                timestamp_writes: None,
            });
            pass.set_pipeline(output_pl);
            pass.set_bind_group(0, output_bg, &[]);
            let wg_x = 1280_u32.div_ceil(8);
            let wg_y = 720_u32.div_ceil(8);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        Ok(())
    }
}

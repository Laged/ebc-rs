// src/cm/pipeline.rs
//! GPU pipeline for Contrast Maximization

use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderContext},
    render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
};
use bytemuck::{Pod, Zeroable};

/// Render graph label for CM node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct CmLabel;

/// Contrast params for GPU
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuContrastParams {
    pub n_omega: u32,
    pub width: u32,
    pub height: u32,
    pub _padding: u32,
}

/// CM Pipeline resources
#[derive(Resource)]
pub struct CmPipeline {
    pub warp_pipeline: CachedComputePipelineId,
    pub warp_layout: BindGroupLayout,
    pub contrast_pipeline: CachedComputePipelineId,
    pub contrast_layout: BindGroupLayout,
    pub select_pipeline: CachedComputePipelineId,
    pub copy_pipeline: CachedComputePipelineId,
    pub select_layout: BindGroupLayout,
}

impl FromWorld for CmPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();

        // Warp pipeline layout
        let warp_layout = render_device.create_bind_group_layout(
            "cm_warp_layout",
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
                // CM params
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
                // IWE buffer (read_write)
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
            ],
        );

        // Contrast pipeline layout
        let contrast_layout = render_device.create_bind_group_layout(
            "cm_contrast_layout",
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
                // Contrast params
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
                // Contrast output
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
                // Edge texture (for edge-informed CM)
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
            ],
        );

        // Select pipeline layout
        let select_layout = render_device.create_bind_group_layout(
            "cm_select_layout",
            &[
                // Contrast buffer
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
                // IWE buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Select params
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
                // Output texture
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Result buffer
                BindGroupLayoutEntry {
                    binding: 4,
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

        let pipeline_cache = world.resource::<PipelineCache>();

        let warp_shader = asset_server.load("shaders/cm_warp.wgsl");
        let warp_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cm_warp_pipeline".into()),
            layout: vec![warp_layout.clone()],
            shader: warp_shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let contrast_shader = asset_server.load("shaders/cm_contrast.wgsl");
        let contrast_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cm_contrast_pipeline".into()),
            layout: vec![contrast_layout.clone()],
            shader: contrast_shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let select_shader = asset_server.load("shaders/cm_select.wgsl");
        let select_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cm_select_pipeline".into()),
            layout: vec![select_layout.clone()],
            shader: select_shader.clone(),
            shader_defs: vec![],
            entry_point: Some("find_best".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let copy_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cm_copy_pipeline".into()),
            layout: vec![select_layout.clone()],
            shader: select_shader,
            shader_defs: vec![],
            entry_point: Some("copy_best".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        Self {
            warp_pipeline,
            warp_layout,
            contrast_pipeline,
            contrast_layout,
            select_pipeline,
            copy_pipeline,
            select_layout,
        }
    }
}

/// Bind groups for CM pipeline
#[derive(Resource, Default)]
pub struct CmBindGroups {
    pub warp: Option<BindGroup>,
    pub contrast: Option<BindGroup>,
    pub select: Option<BindGroup>,
}

/// GPU buffers for CM
#[derive(Resource, Clone)]
pub struct CmBuffers {
    pub params: Buffer,
    pub iwe: Buffer,
    pub contrast: Buffer,
    pub contrast_params: Buffer,
    pub select_params: Buffer,
    pub result: Buffer,
}

/// CM render node
#[derive(Default)]
pub struct CmNode;

impl Node for CmNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<CmPipeline>();
        let Some(bind_groups) = world.get_resource::<CmBindGroups>() else {
            return Ok(());
        };

        let (Some(warp_bg), Some(contrast_bg), Some(select_bg)) =
            (&bind_groups.warp, &bind_groups.contrast, &bind_groups.select) else {
            return Ok(());
        };

        // Get pipelines
        let Some(warp_pl) = pipeline_cache.get_compute_pipeline(pipeline.warp_pipeline) else {
            return Ok(());
        };
        let Some(contrast_pl) = pipeline_cache.get_compute_pipeline(pipeline.contrast_pipeline) else {
            return Ok(());
        };
        let Some(select_pl) = pipeline_cache.get_compute_pipeline(pipeline.select_pipeline) else {
            return Ok(());
        };
        let Some(copy_pl) = pipeline_cache.get_compute_pipeline(pipeline.copy_pipeline) else {
            return Ok(());
        };

        // Check if CM is enabled
        let extracted = world.get_resource::<super::ExtractedCmParams>();
        if extracted.map(|e| !e.enabled).unwrap_or(true) {
            return Ok(());
        }

        // Get buffers for clearing
        let Some(buffers) = world.get_resource::<CmBuffers>() else {
            return Ok(());
        };

        let encoder = render_context.command_encoder();

        // Clear buffers
        encoder.clear_buffer(&buffers.iwe, 0, None);
        encoder.clear_buffer(&buffers.contrast, 0, None);

        // Pass 1: Warp events to build IWE
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cm_warp"),
                timestamp_writes: None,
            });
            pass.set_pipeline(warp_pl);
            pass.set_bind_group(0, warp_bg, &[]);
            // Dispatch for event count (will be set in prepare)
            pass.dispatch_workgroups(4096, 1, 1); // ~1M events max
        }

        // Pass 2: Compute contrast for each omega
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cm_contrast"),
                timestamp_writes: None,
            });
            pass.set_pipeline(contrast_pl);
            pass.set_bind_group(0, contrast_bg, &[]);
            // Dispatch for image size * n_omega
            let wg_x = 1280_u32.div_ceil(8);
            let wg_y = 720_u32.div_ceil(8);
            pass.dispatch_workgroups(wg_x, wg_y, 64); // n_omega slices
        }

        // Pass 3: Find best omega
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cm_select"),
                timestamp_writes: None,
            });
            pass.set_pipeline(select_pl);
            pass.set_bind_group(0, select_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Pass 4: Copy best IWE to output
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cm_copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(copy_pl);
            pass.set_bind_group(0, select_bg, &[]);
            let wg_x = 1280_u32.div_ceil(8);
            let wg_y = 720_u32.div_ceil(8);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        Ok(())
    }
}

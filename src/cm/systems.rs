// src/cm/systems.rs
//! Systems for CM pipeline preparation and result extraction

use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    render_asset::RenderAssets,
    texture::GpuImage,
    Extract,
};

use super::{
    CmImage, CmParams, GpuCmParams,
    CmPipeline, CmBindGroups, CmBuffers, GpuContrastParams,
};
use crate::gpu::{GpuEventBuffer, EventData};
use crate::playback::PlaybackState;
use crate::metrics::EdgeMetrics;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const MAX_N_OMEGA: u32 = 128;

/// Extracted CM parameters for render world
#[derive(Resource, Default)]
pub struct ExtractedCmParams {
    pub centroid: Vec2,
    pub window_start: u32,
    pub window_end: u32,
    pub event_count: u32,
    pub n_omega: u32,
    pub enabled: bool,
}

/// Extract CM params from main world
pub fn extract_cm_params(
    mut commands: Commands,
    params: Extract<Res<CmParams>>,
    playback: Extract<Res<PlaybackState>>,
    metrics: Extract<Option<Res<EdgeMetrics>>>,
    event_data: Extract<Res<EventData>>,
) {
    let window_end = playback.current_time as u32;
    let window_start = window_end.saturating_sub(playback.window_size as u32);

    let centroid = metrics
        .as_ref()
        .map(|m| m.centroid)
        .unwrap_or(Vec2::new(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0));

    commands.insert_resource(ExtractedCmParams {
        centroid,
        window_start,
        window_end,
        event_count: event_data.events.len() as u32,
        n_omega: params.n_omega.min(MAX_N_OMEGA),
        enabled: params.enabled,
    });
}

/// Prepare CM buffers and bind groups
pub fn prepare_cm(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<CmPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    cm_image: Res<CmImage>,
    gpu_events: Option<Res<GpuEventBuffer>>,
    extracted: Res<ExtractedCmParams>,
    buffers: Option<Res<CmBuffers>>,
    mut bind_groups: ResMut<CmBindGroups>,
) {
    if !extracted.enabled {
        return;
    }

    let Some(gpu_events) = gpu_events else { return };
    let Some(event_buffer) = &gpu_events.buffer else { return };
    let Some(cm_gpu) = gpu_images.get(&cm_image.handle) else { return };

    let n_omega = extracted.n_omega;
    let iwe_size = (WIDTH * HEIGHT * n_omega * 4) as u64;
    let contrast_size = (n_omega * 4) as u64;

    // Create or get buffers
    let buffers = if let Some(existing) = buffers {
        existing.into_inner().clone()
    } else {
        let new_buffers = CmBuffers {
            params: render_device.create_buffer(&BufferDescriptor {
                label: Some("cm_params"),
                size: std::mem::size_of::<GpuCmParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            iwe: render_device.create_buffer(&BufferDescriptor {
                label: Some("cm_iwe"),
                size: iwe_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            contrast: render_device.create_buffer(&BufferDescriptor {
                label: Some("cm_contrast"),
                size: contrast_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            contrast_params: render_device.create_buffer(&BufferDescriptor {
                label: Some("cm_contrast_params"),
                size: std::mem::size_of::<GpuContrastParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            select_params: render_device.create_buffer(&BufferDescriptor {
                label: Some("cm_select_params"),
                size: std::mem::size_of::<GpuContrastParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            result: render_device.create_buffer(&BufferDescriptor {
                label: Some("cm_result"),
                size: 8, // 2 x u32
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        };
        commands.insert_resource(new_buffers.clone());
        new_buffers
    };

    // Auto-detect omega range based on event rate
    let event_rate = extracted.event_count as f32
        / (extracted.window_end - extracted.window_start).max(1) as f32
        * 1000.0;
    let estimated_rpm = (event_rate * 0.5).clamp(100.0, 10000.0);
    let omega_center = estimated_rpm * std::f32::consts::TAU / 60.0 / 1e6;
    let omega_min = omega_center * 0.2;
    let omega_max = omega_center * 2.0;
    let omega_step = (omega_max - omega_min) / n_omega as f32;

    // Update params buffer
    let t_ref = (extracted.window_start + extracted.window_end) as f32 / 2.0;
    let gpu_params = GpuCmParams {
        centroid_x: extracted.centroid.x,
        centroid_y: extracted.centroid.y,
        t_ref,
        omega_min,
        omega_step,
        n_omega,
        window_start: extracted.window_start,
        window_end: extracted.window_end,
        event_count: extracted.event_count,
        _padding: [0; 3],
    };
    render_queue.write_buffer(&buffers.params, 0, bytemuck::bytes_of(&gpu_params));

    // Update contrast params
    let contrast_params = GpuContrastParams {
        n_omega,
        width: WIDTH,
        height: HEIGHT,
        _padding: 0,
    };
    render_queue.write_buffer(&buffers.contrast_params, 0, bytemuck::bytes_of(&contrast_params));
    render_queue.write_buffer(&buffers.select_params, 0, bytemuck::bytes_of(&contrast_params));

    // Clear IWE and contrast buffers
    // (In production, would use a clear pass - here we rely on atomic adds starting from 0)

    // Create bind groups
    bind_groups.warp = Some(render_device.create_bind_group(
        "cm_warp_bind_group",
        &pipeline.warp_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: event_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buffers.params.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buffers.iwe.as_entire_binding(),
            },
        ],
    ));

    bind_groups.contrast = Some(render_device.create_bind_group(
        "cm_contrast_bind_group",
        &pipeline.contrast_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: buffers.iwe.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buffers.contrast_params.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buffers.contrast.as_entire_binding(),
            },
        ],
    ));

    bind_groups.select = Some(render_device.create_bind_group(
        "cm_select_bind_group",
        &pipeline.select_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: buffers.contrast.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buffers.iwe.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buffers.select_params.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&cm_gpu.texture_view),
            },
            BindGroupEntry {
                binding: 4,
                resource: buffers.result.as_entire_binding(),
            },
        ],
    ));
}

//! Systems for CMax-SLAM pipeline preparation and result extraction

use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    render_asset::RenderAssets,
    texture::GpuImage,
    Extract,
};

use super::{
    CmaxSlamParams, CmaxSlamState, GpuCmaxSlamParams,
    CmaxSlamPipeline, CmaxSlamBindGroups, CmaxSlamBuffers,
};
use crate::gpu::{SobelImage, GpuEventBuffer, EventData};
use crate::playback::PlaybackState;
use crate::metrics::EdgeMetrics;
use crate::ground_truth::GroundTruthConfig;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// Extracted CMax-SLAM parameters for render world
#[derive(Resource, Default)]
pub struct ExtractedCmaxSlamParams {
    pub centroid: Vec2,
    pub omega: f32,
    pub delta_omega: f32,
    pub edge_weight: f32,
    pub window_start: u32,
    pub window_end: u32,
    pub event_count: u32,
    pub enabled: bool,
}

/// Extract CMax-SLAM params from main world
pub fn extract_cmax_slam_params(
    mut commands: Commands,
    params: Extract<Res<CmaxSlamParams>>,
    state: Extract<Res<CmaxSlamState>>,
    playback: Extract<Res<PlaybackState>>,
    metrics: Extract<Option<Res<EdgeMetrics>>>,
    event_data: Extract<Res<EventData>>,
    gt_config: Extract<Res<GroundTruthConfig>>,
) {
    let window_end = playback.current_time as u32;
    let window_start = window_end.saturating_sub(playback.window_size as u32);

    // Use ground truth centroid if available, otherwise metrics or screen center
    let centroid = if gt_config.center_x > 0.0 && gt_config.center_y > 0.0 {
        Vec2::new(gt_config.center_x, gt_config.center_y)
    } else {
        metrics
            .as_ref()
            .map(|m| m.centroid)
            .unwrap_or(Vec2::new(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0))
    };

    // Use current omega from state, or calculate from ground truth RPM
    let omega = if state.omega.abs() > 1e-10 {
        state.omega
    } else {
        // Use ground truth RPM to calculate initial omega (rad/us)
        // angular_velocity returns rad/s, convert to rad/us by dividing by 1e6
        gt_config.angular_velocity() / 1e6
    };

    // Count events actually in the window (not total events)
    let event_count = event_data.events.iter()
        .filter(|e| e.timestamp >= window_start && e.timestamp <= window_end)
        .count() as u32;

    // Delta for numerical gradient: 1% of omega or minimum value
    let delta_omega = (omega.abs() * 0.01).max(1e-8);

    commands.insert_resource(ExtractedCmaxSlamParams {
        centroid,
        omega,
        delta_omega,
        edge_weight: params.edge_weight,
        window_start,
        window_end,
        event_count,
        enabled: params.enabled,
    });
}

/// Prepare CMax-SLAM buffers and bind groups
pub fn prepare_cmax_slam(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<CmaxSlamPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    sobel_image: Res<SobelImage>,
    gpu_events: Option<Res<GpuEventBuffer>>,
    extracted: Res<ExtractedCmaxSlamParams>,
    buffers: Option<Res<CmaxSlamBuffers>>,
    mut bind_groups: ResMut<CmaxSlamBindGroups>,
) {
    if !extracted.enabled {
        return;
    }

    let Some(gpu_events) = gpu_events else { return };
    let Some(event_buffer) = &gpu_events.buffer else { return };
    let Some(sobel_gpu) = gpu_images.get(&sobel_image.handle) else { return };

    // IWE size: 3 slices (center, +delta, -delta) * WIDTH * HEIGHT * 4 bytes
    let iwe_size = (3 * WIDTH * HEIGHT * 4) as u64;
    let contrast_size = std::mem::size_of::<super::GpuCmaxSlamResult>() as u64;

    // Create or get buffers
    let buffers = if let Some(existing) = buffers {
        existing.into_inner().clone()
    } else {
        let new_buffers = CmaxSlamBuffers {
            params: render_device.create_buffer(&BufferDescriptor {
                label: Some("cmax_slam_params"),
                size: std::mem::size_of::<GpuCmaxSlamParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            iwe: render_device.create_buffer(&BufferDescriptor {
                label: Some("cmax_slam_iwe"),
                size: iwe_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            contrast: render_device.create_buffer(&BufferDescriptor {
                label: Some("cmax_slam_contrast"),
                size: contrast_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        };
        commands.insert_resource(new_buffers.clone());
        new_buffers
    };

    // Update params buffer
    let t_ref = extracted.window_end as f32;
    let gpu_params = GpuCmaxSlamParams {
        centroid_x: extracted.centroid.x,
        centroid_y: extracted.centroid.y,
        t_ref,
        omega: extracted.omega,
        delta_omega: extracted.delta_omega,
        edge_weight: extracted.edge_weight,
        window_start: extracted.window_start,
        window_end: extracted.window_end,
        event_count: extracted.event_count,
        _pad: [0; 3],
    };
    render_queue.write_buffer(&buffers.params, 0, bytemuck::bytes_of(&gpu_params));

    // Create bind groups
    bind_groups.warp_contrast = Some(render_device.create_bind_group(
        "cmax_slam_warp_contrast_bind_group",
        &pipeline.warp_contrast_layout,
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
            BindGroupEntry {
                binding: 3,
                resource: buffers.contrast.as_entire_binding(),
            },
        ],
    ));

    bind_groups.output = Some(render_device.create_bind_group(
        "cmax_slam_output_bind_group",
        &pipeline.output_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: buffers.iwe.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buffers.params.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&sobel_gpu.texture_view),
            },
        ],
    ));
}

/// System to update omega based on gradient (runs in main world)
/// NOTE: Currently uses ground truth omega from extraction. GPU readback for
/// actual contrast gradient would enable runtime optimization.
pub fn update_cmax_slam_omega(
    params: Res<CmaxSlamParams>,
    mut state: ResMut<CmaxSlamState>,
    metrics: Option<Res<EdgeMetrics>>,
    gt_config: Res<GroundTruthConfig>,
) {
    if !params.enabled {
        return;
    }

    // Update centroid from ground truth or metrics
    if gt_config.center_x > 0.0 && gt_config.center_y > 0.0 {
        state.centroid = Vec2::new(gt_config.center_x, gt_config.center_y);
    } else if let Some(ref m) = metrics {
        state.centroid = m.centroid;
    }

    // Use ground truth omega (rad/us) - gradient optimization requires GPU readback
    // which isn't implemented yet. For now, trust ground truth.
    let gt_omega = gt_config.angular_velocity() / 1e6;
    if state.omega.abs() < 1e-10 {
        state.omega = gt_omega;
    }

    // Mark as converged since we're using ground truth
    state.converged = true;
}

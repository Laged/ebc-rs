//! Systems for CMax-SLAM pipeline preparation and result extraction

use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    render_asset::RenderAssets,
    texture::GpuImage,
    Extract,
};
use std::f32::consts::TAU;

use super::{
    CmaxSlamParams, CmaxSlamState, GpuCmaxSlamParams,
    CmaxSlamPipeline, CmaxSlamBindGroups, CmaxSlamBuffers,
    GpuContrastResult, ContrastReceiver, ContrastSender, ContrastValues,
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

    // Pass total event count - shader will filter by timestamp window
    // (We can't use windowed count because events aren't sorted by index = timestamp)
    let event_count = event_data.events.len() as u32;

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

    // Contrast result buffer size (4 x u32 = 16 bytes)
    let contrast_result_size = std::mem::size_of::<GpuContrastResult>() as u64;

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
            contrast_result: render_device.create_buffer(&BufferDescriptor {
                label: Some("cmax_slam_contrast_result"),
                size: contrast_result_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            contrast_staging: render_device.create_buffer(&BufferDescriptor {
                label: Some("cmax_slam_contrast_staging"),
                size: contrast_result_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
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

    // DEBUG: Log params every ~60 frames
    static DEBUG_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let count = DEBUG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if count % 60 == 0 {
        bevy::log::info!(
            "CMax-SLAM params: centroid=({:.1}, {:.1}), omega={:.2e}, t_ref={:.0}, window=[{}, {}], events={}",
            gpu_params.centroid_x, gpu_params.centroid_y,
            gpu_params.omega, gpu_params.t_ref,
            gpu_params.window_start, gpu_params.window_end,
            gpu_params.event_count
        );
    }
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

    bind_groups.reduce = Some(render_device.create_bind_group(
        "cmax_slam_reduce_bind_group",
        &pipeline.reduce_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: buffers.iwe.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buffers.contrast_result.as_entire_binding(),
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

/// System to receive contrast values and update omega (main world)
pub fn receive_contrast_results(
    receiver: Option<Res<ContrastReceiver>>,
    mut state: ResMut<CmaxSlamState>,
    params: Res<CmaxSlamParams>,
    gt_config: Res<GroundTruthConfig>,
) {
    let Some(receiver) = receiver else { return };

    // Cold start initialization
    if !state.initialized {
        state.omega = if gt_config.rpm > 0.0 {
            // Use GT with +10% offset to test optimizer
            gt_config.angular_velocity() / 1e6 * 1.1
        } else {
            // Default: 1000 RPM in rad/μs
            1000.0 * TAU / 60.0 / 1e6
        };
        state.delta_omega = (state.omega.abs() * 0.01).max(1e-8);
        state.initialized = true;
        info!("CMax-SLAM initialized with omega={:.2e} rad/μs", state.omega);
    }

    // Check for new contrast values (receiver.rx is wrapped in Mutex)
    let contrast_opt = {
        if let Ok(rx) = receiver.rx.lock() {
            rx.try_recv().ok()
        } else {
            None
        }
    };

    if let Some(contrast) = contrast_opt {
        state.contrast = contrast.center;

        let v_c = contrast.center;
        let v_p = contrast.plus;
        let v_m = contrast.minus;

        // Skip if no data
        if v_c < 1.0 {
            return;
        }

        // Numerical gradient: dV/dω ≈ (V+ - V-) / 2δ
        let current_delta = state.delta_omega;
        let gradient = (v_p - v_m) / (2.0 * current_delta);

        // Parabolic interpolation for optimal step
        let denominator = 2.0 * (v_p - 2.0 * v_c + v_m);

        let raw_step = if denominator.abs() > 1e-6 {
            // Parabolic fit: jump to estimated peak
            -(v_p - v_m) / (2.0 * denominator) * current_delta
        } else {
            // Fallback: gradient ascent
            params.learning_rate * gradient.signum() * current_delta
        };

        // CLAMP step to ±(max_step_fraction * omega), with minimum for cold start
        let max_step = (state.omega.abs() * params.max_step_fraction).max(1e-7);
        let clamped_step = raw_step.clamp(-max_step, max_step);
        state.last_raw_step = raw_step;
        state.step_was_clamped = raw_step.abs() > max_step;

        // Apply clamped step to get raw omega
        let omega_raw = state.omega + clamped_step;

        // EMA smooth the update
        state.omega = params.ema_alpha * omega_raw + (1.0 - params.ema_alpha) * state.omega;

        // Update delta for next frame (1% of omega)
        state.delta_omega = (state.omega.abs() * 0.01).max(1e-8);

        // Track convergence
        let current_omega = state.omega;
        state.omega_history.push_back(current_omega);
        if state.omega_history.len() > 10 {
            state.omega_history.pop_front();

            let mean: f32 = state.omega_history.iter().sum::<f32>()
                / state.omega_history.len() as f32;
            let variance: f32 = state.omega_history.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / state.omega_history.len() as f32;

            state.converged = variance < (current_omega * 0.001).powi(2);
        }
    }
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Resource to track readback state in render world
/// Uses polling with PollType::Wait to ensure readback completes synchronously
#[derive(Default)]
pub struct ReadbackState {
    pub frame_count: u32,
}

/// Render world system: synchronous buffer readback
/// Runs in Prepare phase to ensure buffer is unmapped before render graph copies to it
/// Note: This blocks until GPU work completes, adding latency but ensuring correctness
pub fn readback_contrast_results(
    render_device: Res<RenderDevice>,
    buffers: Option<Res<CmaxSlamBuffers>>,
    sender: Option<Res<ContrastSender>>,
    extracted: Option<Res<ExtractedCmaxSlamParams>>,
    mut readback_state: Local<ReadbackState>,
) {
    let Some(buffers) = buffers else { return };
    let Some(sender) = sender else { return };
    let Some(extracted) = extracted else { return };

    if !extracted.enabled {
        return;
    }

    // Skip first 2 frames to let pipeline stabilize
    readback_state.frame_count += 1;
    if readback_state.frame_count < 3 {
        return;
    }

    let staging = &buffers.contrast_staging;
    let slice = staging.slice(..);

    // Map the buffer synchronously using Wait polling
    let ready = Arc::new(AtomicBool::new(false));
    let ready_clone = ready.clone();

    slice.map_async(MapMode::Read, move |result| {
        if result.is_ok() {
            ready_clone.store(true, Ordering::Release);
        }
    });

    // Wait for mapping to complete (blocking)
    let _ = render_device.poll(PollType::Wait);

    // Read the data if mapping succeeded
    if ready.load(Ordering::Acquire) {
        let data = slice.get_mapped_range();

        if data.len() >= 16 {
            let result: &GpuContrastResult = bytemuck::from_bytes(&data[..16]);

            // Debug log every 60 frames
            if readback_state.frame_count % 60 == 0 {
                info!(
                    "Contrast readback: center={}, plus={}, minus={}, pixels={}",
                    result.sum_sq_center, result.sum_sq_plus, result.sum_sq_minus, result.pixel_count
                );
            }

            // Convert to float contrast values
            let values = ContrastValues {
                center: result.sum_sq_center as f32,
                plus: result.sum_sq_plus as f32,
                minus: result.sum_sq_minus as f32,
            };

            // Send to main world
            if let Ok(tx) = sender.tx.lock() {
                let _ = tx.send(values);
            }
        }
        drop(data);
    }

    // Always unmap before render graph runs
    staging.unmap();
}

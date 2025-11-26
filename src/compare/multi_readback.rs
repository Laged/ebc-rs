//! Multi-detector readback for metrics computation.

use bevy::prelude::*;
use bevy::render::{
    render_graph::{Node, RenderLabel},
    render_resource::*,
    renderer::{RenderContext, RenderDevice},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::sync::Mutex;

use crate::gpu::{FilteredSurfaceImage, SobelImage, CannyImage, LogImage};
use crate::cm::CmResult;

/// Metrics for a single detector
#[derive(Debug, Clone, Default)]
pub struct DetectorMetrics {
    pub edge_count: u32,
    pub tolerance_precision: f32,
    pub tolerance_recall: f32,
    pub tolerance_f1: f32,
    pub avg_distance: f32,
}

/// Combined metrics for all detectors
#[derive(Resource, Debug, Clone, Default)]
pub struct AllDetectorMetrics {
    pub raw: DetectorMetrics,
    pub sobel: DetectorMetrics,
    pub cm: CmResult,  // Changed from canny: DetectorMetrics
    pub log: DetectorMetrics,
    pub frame_time_ms: f32,
    pub last_update: f64,
}

/// Channel to send metrics from render world to main world
#[derive(Resource)]
pub struct MetricsSender(pub Sender<AllDetectorMetrics>);

#[derive(Resource)]
pub struct MetricsReceiver(pub Mutex<Receiver<AllDetectorMetrics>>);

/// Per-detector edge data for metrics computation
#[derive(Debug, Clone, Default)]
pub struct DetectorEdgeData {
    pub pixels: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Combined edge data from all detectors
#[derive(Resource, Default)]
pub struct AllEdgeData {
    pub raw: DetectorEdgeData,
    pub sobel: DetectorEdgeData,
    pub canny: DetectorEdgeData,
    pub log: DetectorEdgeData,
}

/// System to receive metrics in main world
pub fn receive_metrics(
    receiver: Option<Res<MetricsReceiver>>,
    mut metrics: ResMut<AllDetectorMetrics>,
    time: Res<Time>,
) {
    let Some(ref receiver) = receiver else { return };

    if let Ok(rx) = receiver.0.try_lock() {
        // Get the most recent metrics, draining any backlog
        while let Ok(new_metrics) = rx.try_recv() {
            *metrics = new_metrics;
            metrics.last_update = time.elapsed_secs_f64();
        }
    };
}

/// Create metrics channel and add resources
pub fn setup_metrics_channel(app: &mut App) {
    let (tx, rx) = channel();
    app.insert_resource(MetricsReceiver(Mutex::new(rx)));

    // Sender goes to render world, created in finish()
    app.insert_resource(PendingMetricsSender(tx));
}

/// Temporary storage for sender until render app is available
#[derive(Resource)]
pub struct PendingMetricsSender(pub Sender<AllDetectorMetrics>);

// ============================================================================
// Multi-detector readback system (render world)
// ============================================================================

/// Render graph label for multi-detector readback
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct MultiReadbackLabel;

/// Staging buffers for all detectors
#[derive(Resource, Default)]
pub struct MultiReadbackBuffers {
    pub raw_staging: Option<Buffer>,
    pub sobel_staging: Option<Buffer>,
    pub canny_staging: Option<Buffer>,
    pub log_staging: Option<Buffer>,
    pub dimensions: UVec2,
    /// Which detector we're currently mapping (round-robin)
    pub current_detector: u8,
    /// Map receiver for async completion
    pub map_receiver: Option<Mutex<Receiver<Result<(), BufferAsyncError>>>>,
    pub mapping_in_progress: bool,
    /// Cached edge counts (updated after each successful read)
    pub raw_count: u32,
    pub sobel_count: u32,
    pub canny_count: u32,
    pub log_count: u32,
}

/// Render graph node that copies all detector textures to staging buffers
#[derive(Default)]
pub struct MultiReadbackNode;

impl Node for MultiReadbackNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let buffers = world.resource::<MultiReadbackBuffers>();

        if buffers.dimensions.x == 0 || buffers.dimensions.y == 0 {
            return Ok(());
        }

        if buffers.mapping_in_progress {
            return Ok(());
        }

        let gpu_images = world.resource::<RenderAssets<GpuImage>>();

        let bytes_per_row = buffers.dimensions.x * 4;
        let padded_bytes_per_row = (bytes_per_row + 255) & !255;
        let dimensions = buffers.dimensions;

        // Copy based on which detector we're reading this frame
        match buffers.current_detector {
            0 => {
                // Raw (FilteredSurface is R32Uint)
                let raw = world.resource::<FilteredSurfaceImage>();
                if let (Some(gpu), Some(buf)) = (gpu_images.get(&raw.handle), &buffers.raw_staging) {
                    render_context.command_encoder().copy_texture_to_buffer(
                        gpu.texture.as_image_copy(),
                        TexelCopyBufferInfo {
                            buffer: buf,
                            layout: TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(padded_bytes_per_row),
                                rows_per_image: Some(dimensions.y),
                            },
                        },
                        Extent3d {
                            width: dimensions.x,
                            height: dimensions.y,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            1 => {
                let sobel = world.resource::<SobelImage>();
                if let (Some(gpu), Some(buf)) = (gpu_images.get(&sobel.handle), &buffers.sobel_staging) {
                    render_context.command_encoder().copy_texture_to_buffer(
                        gpu.texture.as_image_copy(),
                        TexelCopyBufferInfo {
                            buffer: buf,
                            layout: TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(padded_bytes_per_row),
                                rows_per_image: Some(dimensions.y),
                            },
                        },
                        Extent3d {
                            width: dimensions.x,
                            height: dimensions.y,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            2 => {
                let canny = world.resource::<CannyImage>();
                if let (Some(gpu), Some(buf)) = (gpu_images.get(&canny.handle), &buffers.canny_staging) {
                    render_context.command_encoder().copy_texture_to_buffer(
                        gpu.texture.as_image_copy(),
                        TexelCopyBufferInfo {
                            buffer: buf,
                            layout: TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(padded_bytes_per_row),
                                rows_per_image: Some(dimensions.y),
                            },
                        },
                        Extent3d {
                            width: dimensions.x,
                            height: dimensions.y,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            3 => {
                let log = world.resource::<LogImage>();
                if let (Some(gpu), Some(buf)) = (gpu_images.get(&log.handle), &buffers.log_staging) {
                    render_context.command_encoder().copy_texture_to_buffer(
                        gpu.texture.as_image_copy(),
                        TexelCopyBufferInfo {
                            buffer: buf,
                            layout: TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(padded_bytes_per_row),
                                rows_per_image: Some(dimensions.y),
                            },
                        },
                        Extent3d {
                            width: dimensions.x,
                            height: dimensions.y,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            _ => {}
        }

        Ok(())
    }
}

/// System to prepare multi-readback staging buffers
pub fn prepare_multi_readback(
    render_device: Res<RenderDevice>,
    mut buffers: ResMut<MultiReadbackBuffers>,
    sobel_image: Res<SobelImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    let Some(gpu_image) = gpu_images.get(&sobel_image.handle) else {
        return;
    };

    let width = gpu_image.texture.width();
    let height = gpu_image.texture.height();

    if buffers.dimensions.x != width || buffers.dimensions.y != height {
        buffers.dimensions = UVec2::new(width, height);

        let bytes_per_row = width * 4;
        let padded_bytes_per_row = (bytes_per_row + 255) & !255;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        // Create staging buffer for each detector
        buffers.raw_staging = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("Raw Multi-Readback Staging"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        buffers.sobel_staging = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("Sobel Multi-Readback Staging"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        buffers.canny_staging = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("Canny Multi-Readback Staging"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        buffers.log_staging = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("LoG Multi-Readback Staging"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        info!("Created multi-readback buffers: {}x{}", width, height);
    }
}

/// System to read back edge counts from staging buffers
pub fn read_multi_readback(
    mut buffers: ResMut<MultiReadbackBuffers>,
    sender: Option<Res<MetricsSender>>,
    render_device: Res<RenderDevice>,
) {
    // Poll device for async callbacks
    let _ = render_device.poll(PollType::Poll);

    let buffers = buffers.as_mut();

    // Get current detector's staging buffer
    let staging = match buffers.current_detector {
        0 => &buffers.raw_staging,
        1 => &buffers.sobel_staging,
        2 => &buffers.canny_staging,
        3 => &buffers.log_staging,
        _ => return,
    };

    let Some(staging_buffer) = staging else {
        return;
    };

    // Check if we have a pending map operation
    if let Some(receiver_mutex) = buffers.map_receiver.take() {
        enum Action {
            ProcessData,
            MapError,
            StillWaiting,
            Disconnected,
            CouldntLock,
        }

        let action = {
            let lock_result = receiver_mutex.try_lock();
            match lock_result {
                Ok(receiver) => {
                    match receiver.try_recv() {
                        Ok(Ok(())) => Action::ProcessData,
                        Ok(Err(_)) => Action::MapError,
                        Err(std::sync::mpsc::TryRecvError::Empty) => Action::StillWaiting,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => Action::Disconnected,
                    }
                }
                Err(_) => Action::CouldntLock,
            }
        };

        match action {
            Action::ProcessData => {
                let slice = staging_buffer.slice(..);
                {
                    let data = slice.get_mapped_range();

                    let width = buffers.dimensions.x as usize;
                    let height = buffers.dimensions.y as usize;
                    let bytes_per_row = width * 4;
                    let padded_bytes_per_row = (bytes_per_row + 255) & !255;

                    // Count non-zero pixels
                    let mut count = 0u32;

                    if buffers.current_detector == 0 {
                        // Raw is R32Uint
                        for y in 0..height {
                            let src_offset = y * padded_bytes_per_row;
                            let row_bytes = &data[src_offset..src_offset + bytes_per_row];
                            let row_uints: &[u32] = bytemuck::cast_slice(row_bytes);
                            for &val in row_uints {
                                if val > 0 {
                                    count += 1;
                                }
                            }
                        }
                    } else {
                        // Sobel/Canny/LoG are R32Float
                        for y in 0..height {
                            let src_offset = y * padded_bytes_per_row;
                            let row_bytes = &data[src_offset..src_offset + bytes_per_row];
                            let row_floats: &[f32] = bytemuck::cast_slice(row_bytes);
                            for &val in row_floats {
                                if val > 0.5 {
                                    count += 1;
                                }
                            }
                        }
                    }

                    // Store count for this detector
                    match buffers.current_detector {
                        0 => buffers.raw_count = count,
                        1 => buffers.sobel_count = count,
                        2 => buffers.canny_count = count,
                        3 => buffers.log_count = count,
                        _ => {}
                    }
                }
                staging_buffer.unmap();
                buffers.mapping_in_progress = false;

                // After reading LoG (detector 3), send all metrics
                if buffers.current_detector == 3 {
                    if let Some(sender) = &sender {
                        let metrics = AllDetectorMetrics {
                            raw: DetectorMetrics {
                                edge_count: buffers.raw_count,
                                ..Default::default()
                            },
                            sobel: DetectorMetrics {
                                edge_count: buffers.sobel_count,
                                ..Default::default()
                            },
                            cm: CmResult::default(),  // CM metrics updated separately
                            log: DetectorMetrics {
                                edge_count: buffers.log_count,
                                ..Default::default()
                            },
                            frame_time_ms: 0.0, // Updated in main world
                            last_update: 0.0,
                        };
                        let _ = sender.0.send(metrics);
                    }
                }

                // Move to next detector
                buffers.current_detector = (buffers.current_detector + 1) % 4;
            }
            Action::MapError => {
                error!("Multi-readback buffer map failed");
                buffers.mapping_in_progress = false;
                buffers.current_detector = (buffers.current_detector + 1) % 4;
            }
            Action::StillWaiting | Action::CouldntLock => {
                buffers.map_receiver = Some(receiver_mutex);
            }
            Action::Disconnected => {
                error!("Multi-readback map channel disconnected");
                buffers.mapping_in_progress = false;
                buffers.current_detector = (buffers.current_detector + 1) % 4;
            }
        }
    } else if !buffers.mapping_in_progress && buffers.dimensions.x > 0 {
        // Start a new map operation
        let slice = staging_buffer.slice(..);
        let (sender, receiver) = channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        buffers.map_receiver = Some(Mutex::new(receiver));
        buffers.mapping_in_progress = true;
    }
}

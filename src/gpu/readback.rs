//! GPU texture readback for edge detection results
//!
//! This module copies edge detection textures from GPU to CPU-accessible
//! staging buffers, then maps them for metric computation.

use bevy::prelude::*;
use bevy::render::{
    render_graph::{Node, RenderLabel},
    render_resource::*,
    renderer::{RenderContext, RenderDevice},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use super::resources::{EdgeReadbackBuffer, ActiveDetector};
use super::{SobelImage, CannyImage, LogImage, GroundTruthImage};

/// Render graph label for the readback node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct ReadbackLabel;

/// Render graph node that copies edge textures to staging buffers
#[derive(Default)]
pub struct ReadbackNode;

impl Node for ReadbackNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let readback = world.resource::<EdgeReadbackBuffer>();

        // Only copy if staging buffers exist and we have valid dimensions
        if readback.dimensions.x == 0 || readback.dimensions.y == 0 {
            return Ok(());
        }

        // Skip copy if buffer is currently being mapped (from previous frame)
        // wgpu doesn't allow writing to a buffer with pending map operation
        if readback.mapping_in_progress {
            return Ok(());
        }

        let gpu_images = world.resource::<RenderAssets<GpuImage>>();

        // Get the active detector's texture and staging buffer
        let (image_res, staging_buffer) = match readback.active_detector {
            ActiveDetector::Sobel => {
                let sobel = world.resource::<SobelImage>();
                (gpu_images.get(&sobel.handle), &readback.sobel_staging)
            }
            ActiveDetector::Canny => {
                let canny = world.resource::<CannyImage>();
                (gpu_images.get(&canny.handle), &readback.canny_staging)
            }
            ActiveDetector::Log => {
                let log = world.resource::<LogImage>();
                (gpu_images.get(&log.handle), &readback.log_staging)
            }
        };

        let Some(gpu_image) = image_res else {
            return Ok(());
        };
        let Some(staging) = staging_buffer else {
            return Ok(());
        };

        // Copy texture to staging buffer
        let bytes_per_row = readback.dimensions.x * 4; // R32Float = 4 bytes
        // wgpu requires rows aligned to 256 bytes
        let padded_bytes_per_row = (bytes_per_row + 255) & !255;

        render_context.command_encoder().copy_texture_to_buffer(
            gpu_image.texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: staging,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(readback.dimensions.y),
                },
            },
            Extent3d {
                width: readback.dimensions.x,
                height: readback.dimensions.y,
                depth_or_array_layers: 1,
            },
        );

        // Also copy ground truth texture if available (for metric computation)
        if let Some(gt_staging) = &readback.ground_truth_staging {
            let gt_image = world.resource::<GroundTruthImage>();
            if let Some(gt_gpu) = gpu_images.get(&gt_image.handle) {
                // Ground truth is RGBA8, so 4 bytes per pixel
                let gt_bytes_per_row = readback.dimensions.x * 4;
                let gt_padded_bytes_per_row = (gt_bytes_per_row + 255) & !255;

                render_context.command_encoder().copy_texture_to_buffer(
                    gt_gpu.texture.as_image_copy(),
                    TexelCopyBufferInfo {
                        buffer: gt_staging,
                        layout: TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(gt_padded_bytes_per_row),
                            rows_per_image: Some(readback.dimensions.y),
                        },
                    },
                    Extent3d {
                        width: readback.dimensions.x,
                        height: readback.dimensions.y,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        Ok(())
    }
}

/// System to create/update staging buffers for readback
pub fn prepare_readback(
    render_device: Res<RenderDevice>,
    mut readback: ResMut<EdgeReadbackBuffer>,
    sobel_image: Res<SobelImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    // Get dimensions from Sobel texture (all edge textures are same size)
    let Some(gpu_image) = gpu_images.get(&sobel_image.handle) else {
        return;
    };

    let width = gpu_image.texture.width();
    let height = gpu_image.texture.height();

    // Update dimensions if changed
    if readback.dimensions.x != width || readback.dimensions.y != height {
        readback.dimensions = UVec2::new(width, height);

        // Calculate buffer size with row padding
        let bytes_per_row = width * 4; // R32Float = 4 bytes
        let padded_bytes_per_row = (bytes_per_row + 255) & !255;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        // Create staging buffers for each detector
        readback.sobel_staging = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("Sobel Readback Staging"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        readback.canny_staging = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("Canny Readback Staging"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        readback.log_staging = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("LoG Readback Staging"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        // Create ground truth staging buffer (RGBA8, so 4 bytes per pixel instead of R32Float)
        let gt_bytes_per_row = width * 4; // RGBA8 = 4 bytes per pixel
        let gt_padded_bytes_per_row = (gt_bytes_per_row + 255) & !255;
        let gt_buffer_size = (gt_padded_bytes_per_row * height) as u64;

        readback.ground_truth_staging = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("Ground Truth Readback Staging"),
            size: gt_buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        // Allocate CPU-side vectors
        let pixel_count = (width * height) as usize;
        readback.sobel_data = vec![0.0; pixel_count];
        readback.canny_data = vec![0.0; pixel_count];
        readback.log_data = vec![0.0; pixel_count];
        readback.ground_truth_data = vec![0.0; pixel_count];

        info!("Created readback buffers: {}x{} ({} pixels)", width, height, pixel_count);
    }
}

/// System to map staging buffer and copy data to CPU vectors
/// Runs in RenderSystems::Cleanup after all GPU work is submitted
pub fn read_readback_result(
    readback: ResMut<EdgeReadbackBuffer>,
    sender: Option<Res<crate::analysis::EdgeDataSender>>,
    render_device: Res<RenderDevice>,
) {
    // Poll the device to process pending callbacks (including map_async)
    let _ = render_device.poll(PollType::Poll);
    let readback = readback.into_inner();

    // Get the active staging buffer
    let staging = match readback.active_detector {
        ActiveDetector::Sobel => &readback.sobel_staging,
        ActiveDetector::Canny => &readback.canny_staging,
        ActiveDetector::Log => &readback.log_staging,
    };
    let Some(staging_buffer) = staging else {
        return;
    };

    // Check if we have a pending map operation
    if let Some(receiver_mutex) = readback.map_receiver.take() {
        // Try to lock and check the result, determining what action to take
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
        }; // lock_result is dropped here

        match action {
            Action::ProcessData => {
                // Map succeeded - read the data
                let slice = staging_buffer.slice(..);
                {
                    let data = slice.get_mapped_range();

                    // Copy to appropriate vector, handling row padding
                    let width = readback.dimensions.x as usize;
                    let height = readback.dimensions.y as usize;
                    let bytes_per_row = width * 4;
                    let padded_bytes_per_row = (bytes_per_row + 255) & !255;

                    let target = match readback.active_detector {
                        ActiveDetector::Sobel => &mut readback.sobel_data,
                        ActiveDetector::Canny => &mut readback.canny_data,
                        ActiveDetector::Log => &mut readback.log_data,
                    };

                    // Copy row by row to handle padding
                    for y in 0..height {
                        let src_offset = y * padded_bytes_per_row;
                        let dst_offset = y * width;
                        let row_bytes = &data[src_offset..src_offset + bytes_per_row];
                        let row_floats: &[f32] = bytemuck::cast_slice(row_bytes);
                        target[dst_offset..dst_offset + width].copy_from_slice(row_floats);
                    }
                }
                staging_buffer.unmap();
                readback.mapping_in_progress = false;
                readback.ready = true;

                // Send data to main world via channel
                if let Some(sender) = &sender {
                    let (detector_name, pixels) = match readback.active_detector {
                        ActiveDetector::Sobel => ("sobel", readback.sobel_data.clone()),
                        ActiveDetector::Canny => ("canny", readback.canny_data.clone()),
                        ActiveDetector::Log => ("log", readback.log_data.clone()),
                    };
                    let _ = sender.0.send(crate::analysis::EdgeData {
                        pixels,
                        width: readback.dimensions.x,
                        height: readback.dimensions.y,
                        detector: detector_name.to_string(),
                        updated: true,
                    });
                }
            }
            Action::MapError => {
                error!("Buffer map failed");
                readback.mapping_in_progress = false;
            }
            Action::StillWaiting | Action::CouldntLock => {
                // Put receiver back
                readback.map_receiver = Some(receiver_mutex);
            }
            Action::Disconnected => {
                error!("Buffer map channel disconnected");
                readback.mapping_in_progress = false;
            }
        }
    } else if !readback.mapping_in_progress && readback.dimensions.x > 0 {
        // Start a new map operation
        let slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        readback.map_receiver = Some(std::sync::Mutex::new(receiver));
        readback.mapping_in_progress = true;
        readback.ready = false;
    }
}

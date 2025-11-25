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
use super::{SobelImage, CannyImage, LogImage};

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

        // Allocate CPU-side vectors
        let pixel_count = (width * height) as usize;
        readback.sobel_data = vec![0.0; pixel_count];
        readback.canny_data = vec![0.0; pixel_count];
        readback.log_data = vec![0.0; pixel_count];

        info!("Created readback buffers: {}x{} ({} pixels)", width, height, pixel_count);
    }
}

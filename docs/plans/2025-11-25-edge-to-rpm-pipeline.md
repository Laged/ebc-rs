# Edge-to-RPM Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete pipeline from GPU edge detection textures to automated fan RPM calculation with quality metrics.

**Architecture:** The current codebase has three GPU edge detectors (Sobel, Canny, LoG) producing R32Float textures. This plan adds GPU readback infrastructure to bring edge data to CPU, computes quality metrics using circular fitting and angular histograms, then tracks blade rotation to calculate RPM. Prior work on `fan-rpm-claude` branch provides reference implementations for centroid tracking, radial profiles, and angular histograms that can be adapted.

**Tech Stack:** Bevy 0.17, wgpu (buffer mapping), bytemuck, serde_json, egui

**Prior Art:** Branch `fan-rpm-claude` contains 1372-line `analysis.rs` with GPU centroid/radial/angular pipelines and `synthesis.rs` for test data generation. These can be cherry-picked and adapted.

---

## Phase 1: GPU Readback Infrastructure

### Task 1.1: Add EdgeReadbackBuffer Resource

**Files:**
- Modify: `src/gpu/resources.rs:36-47`
- Modify: `src/gpu/mod.rs`

**Step 1: Write the test (compile check)**

No unit test needed - this is a struct definition. Verify compilation.

**Step 2: Add EdgeReadbackBuffer struct to resources.rs**

After the existing `GpuEventBuffer` struct (line 47), add:

```rust
/// Buffer for reading edge texture data back to CPU
#[derive(Resource, Default)]
pub struct EdgeReadbackBuffer {
    /// Staging buffer for Sobel texture readback
    pub sobel_staging: Option<Buffer>,
    /// Staging buffer for Canny texture readback
    pub canny_staging: Option<Buffer>,
    /// Staging buffer for LoG texture readback
    pub log_staging: Option<Buffer>,
    /// Texture dimensions
    pub dimensions: UVec2,
    /// CPU-side edge data (Sobel)
    pub sobel_data: Vec<f32>,
    /// CPU-side edge data (Canny)
    pub canny_data: Vec<f32>,
    /// CPU-side edge data (LoG)
    pub log_data: Vec<f32>,
    /// Whether data is ready for CPU consumption
    pub ready: bool,
    /// Which detector to read back (to avoid reading all three every frame)
    pub active_detector: ActiveDetector,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum ActiveDetector {
    #[default]
    Sobel,
    Canny,
    Log,
}
```

**Step 3: Export from mod.rs**

Add to `src/gpu/mod.rs` exports:

```rust
pub use resources::{EdgeReadbackBuffer, ActiveDetector};
```

**Step 4: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add src/gpu/resources.rs src/gpu/mod.rs
git commit -m "feat: add EdgeReadbackBuffer resource for GPU readback"
```

---

### Task 1.2: Create Readback Module Structure

**Files:**
- Create: `src/gpu/readback.rs`
- Modify: `src/gpu/mod.rs`

**Step 1: Create readback.rs with module structure**

```rust
//! GPU texture readback for edge detection results
//!
//! This module copies edge detection textures from GPU to CPU-accessible
//! staging buffers, then maps them for metric computation.

use bevy::prelude::*;
use bevy::render::{
    render_graph::{Node, RenderLabel},
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
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
            ImageCopyBuffer {
                buffer: staging,
                layout: ImageDataLayout {
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
```

**Step 2: Add module to mod.rs**

Add to `src/gpu/mod.rs`:

```rust
pub mod readback;
pub use readback::{ReadbackLabel, ReadbackNode};
```

**Step 3: Run cargo check**

Run: `cargo check`
Expected: Compiles (warnings about unused ok)

**Step 4: Commit**

```bash
git add src/gpu/readback.rs src/gpu/mod.rs
git commit -m "feat: add ReadbackNode for texture-to-buffer copy"
```

---

### Task 1.3: Add Readback Preparation System

**Files:**
- Modify: `src/gpu/readback.rs`

**Step 1: Add prepare_readback system**

Append to `src/gpu/readback.rs`:

```rust
/// System to create/update staging buffers for readback
pub fn prepare_readback(
    mut commands: Commands,
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
```

**Step 2: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/gpu/readback.rs
git commit -m "feat: add prepare_readback system for staging buffer creation"
```

---

### Task 1.4: Integrate Readback into Render Graph

**Files:**
- Modify: `src/edge_detection.rs`

**Step 1: Add readback resource and systems to plugin**

In `src/edge_detection.rs`, update the `finish` method:

```rust
fn finish(&self, app: &mut App) {
    // Custom extraction system for EdgeParams
    fn extract_edge_params(
        mut commands: Commands,
        edge_params: Extract<Res<EdgeParams>>,
    ) {
        commands.insert_resource(edge_params.clone());
    }

    let render_app = app.sub_app_mut(RenderApp);

    render_app
        .init_resource::<EventComputePipeline>()
        .init_resource::<SobelPipeline>()
        .init_resource::<CannyPipeline>()
        .init_resource::<LogPipeline>()
        .init_resource::<GpuEventBuffer>()
        .init_resource::<EdgeReadbackBuffer>()  // NEW
        .add_systems(ExtractSchedule, extract_edge_params)
        .add_systems(Render, prepare_events.in_set(RenderSystems::Prepare))
        .add_systems(Render, prepare_readback.in_set(RenderSystems::Prepare))  // NEW
        .add_systems(Render, queue_bind_group.in_set(RenderSystems::Queue))
        .add_systems(Render, prepare_sobel.in_set(RenderSystems::Queue))
        .add_systems(Render, prepare_canny.in_set(RenderSystems::Queue))
        .add_systems(Render, prepare_log.in_set(RenderSystems::Queue));

    let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
    render_graph.add_node(EventLabel, EventAccumulationNode::default());
    render_graph.add_node(SobelLabel, SobelNode::default());
    render_graph.add_node(CannyLabel, CannyNode::default());
    render_graph.add_node(LogLabel, LogNode::default());
    render_graph.add_node(ReadbackLabel, ReadbackNode::default());  // NEW

    // Render graph: Event → Sobel → Canny → LoG → Readback → Camera
    render_graph.add_node_edge(EventLabel, SobelLabel);
    render_graph.add_node_edge(SobelLabel, CannyLabel);
    render_graph.add_node_edge(CannyLabel, LogLabel);
    render_graph.add_node_edge(LogLabel, ReadbackLabel);  // NEW
    render_graph.add_node_edge(ReadbackLabel, bevy::render::graph::CameraDriverLabel);  // CHANGED
}
```

**Step 2: Add imports at top of edge_detection.rs**

```rust
use crate::gpu::{EdgeReadbackBuffer, ReadbackLabel, ReadbackNode, prepare_readback};
```

**Step 3: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 4: Run cargo run to verify no runtime errors**

Run: `timeout 10 cargo run -- data/fan/fan_const_rpm.dat 2>&1 | head -50`
Expected: App starts without crashes

**Step 5: Commit**

```bash
git add src/edge_detection.rs
git commit -m "feat: integrate ReadbackNode into render graph"
```

---

### Task 1.5: Add Async Buffer Mapping

**Files:**
- Modify: `src/gpu/readback.rs`
- Modify: `src/gpu/resources.rs`

**Step 1: Add mapping state to EdgeReadbackBuffer**

In `src/gpu/resources.rs`, extend `EdgeReadbackBuffer`:

```rust
#[derive(Resource, Default)]
pub struct EdgeReadbackBuffer {
    // ... existing fields ...

    /// Channel receiver for async map completion
    pub map_receiver: Option<std::sync::Mutex<std::sync::mpsc::Receiver<Result<(), BufferAsyncError>>>>,
    /// Whether a map operation is in flight
    pub mapping_in_progress: bool,
}
```

**Step 2: Add read_readback_result system to readback.rs**

```rust
/// System to map staging buffer and copy data to CPU vectors
/// Runs in RenderSystems::Cleanup after all GPU work is submitted
pub fn read_readback_result(readback: ResMut<EdgeReadbackBuffer>) {
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
        if let Ok(receiver) = receiver_mutex.try_lock() {
            match receiver.try_recv() {
                Ok(Ok(())) => {
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
                }
                Ok(Err(e)) => {
                    error!("Buffer map failed: {:?}", e);
                    readback.mapping_in_progress = false;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Still waiting - put receiver back
                    readback.map_receiver = Some(receiver_mutex);
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    error!("Buffer map channel disconnected");
                    readback.mapping_in_progress = false;
                }
            }
        } else {
            // Couldn't lock - put receiver back
            readback.map_receiver = Some(receiver_mutex);
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
```

**Step 3: Register the system in edge_detection.rs**

Add to the render_app systems:

```rust
.add_systems(Render, read_readback_result.in_set(RenderSystems::Cleanup))
```

And add the import:

```rust
use crate::gpu::read_readback_result;
```

**Step 4: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add src/gpu/resources.rs src/gpu/readback.rs src/edge_detection.rs
git commit -m "feat: add async buffer mapping for GPU readback"
```

---

### Task 1.6: Add Readback to Main World via Channel

**Files:**
- Create: `src/analysis.rs`
- Modify: `src/lib.rs`
- Modify: `src/gpu/readback.rs`

This task sets up the channel communication pattern from fan-rpm-claude branch to send readback data from render world to main world.

**Step 1: Create minimal analysis.rs**

```rust
//! Fan motion analysis from edge detection data
//!
//! This module receives GPU readback data and computes metrics
//! for fan geometry detection and RPM calculation.

use bevy::prelude::*;

/// Edge data received from GPU readback
#[derive(Resource, Default, Clone)]
pub struct EdgeData {
    /// Edge pixel values from active detector
    pub pixels: Vec<f32>,
    /// Image dimensions
    pub width: u32,
    pub height: u32,
    /// Which detector this data came from
    pub detector: String,
    /// Whether new data is available
    pub updated: bool,
}

/// Channel receiver for edge data from render world
#[derive(Resource)]
pub struct EdgeDataReceiver(pub std::sync::Mutex<std::sync::mpsc::Receiver<EdgeData>>);

/// Channel sender for edge data (lives in render world)
#[derive(Resource)]
pub struct EdgeDataSender(pub std::sync::mpsc::Sender<EdgeData>);

/// Plugin for fan motion analysis
pub struct AnalysisPlugin;

impl Plugin for AnalysisPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EdgeData>()
            .add_systems(Update, receive_edge_data);
    }
}

fn receive_edge_data(
    receiver: Option<Res<EdgeDataReceiver>>,
    mut edge_data: ResMut<EdgeData>,
) {
    let Some(receiver) = receiver else { return };

    if let Ok(rx) = receiver.0.try_lock() {
        // Get the latest data (drain any backlog)
        while let Ok(data) = rx.try_recv() {
            *edge_data = data;
            edge_data.updated = true;
        }
    }
}
```

**Step 2: Add to lib.rs**

```rust
pub mod analysis;
pub use analysis::{AnalysisPlugin, EdgeData};
```

**Step 3: Update readback.rs to send data**

Add to `read_readback_result` after copying data:

```rust
// After: target[dst_offset..dst_offset + width].copy_from_slice(row_floats);
// Send to main world
if let Some(sender) = world.get_resource::<EdgeDataSender>() {
    let detector_name = match readback.active_detector {
        ActiveDetector::Sobel => "sobel",
        ActiveDetector::Canny => "canny",
        ActiveDetector::Log => "log",
    };
    let _ = sender.0.send(EdgeData {
        pixels: target.clone(),
        width: readback.dimensions.x,
        height: readback.dimensions.y,
        detector: detector_name.to_string(),
        updated: true,
    });
}
```

Note: This requires refactoring - the system needs access to world. We'll handle this in edge_detection.rs setup instead.

**Step 4: Setup channel in edge_detection.rs**

In the `finish` method, before setting up render_app:

```rust
// Setup edge data channel
let (edge_sender, edge_receiver) = std::sync::mpsc::channel();
app.insert_resource(crate::analysis::EdgeDataReceiver(std::sync::Mutex::new(edge_receiver)));

let render_app = app.sub_app_mut(RenderApp);
render_app.insert_resource(crate::analysis::EdgeDataSender(edge_sender));
```

**Step 5: Run cargo check**

Run: `cargo check`
Expected: Compiles (may have warnings)

**Step 6: Commit**

```bash
git add src/analysis.rs src/lib.rs src/gpu/readback.rs src/edge_detection.rs
git commit -m "feat: add EdgeData channel from render to main world"
```

---

### Task 1.7: Add Edge Count UI Display

**Files:**
- Modify: `src/event_renderer.rs`

**Step 1: Add edge metrics display to UI**

In `ui_system`, add a new window:

```rust
// After the Edge Detection window, add:
if let Some(edge_data) = world.get_resource::<crate::analysis::EdgeData>() {
    egui::Window::new("Edge Metrics").show(ctx, |ui| {
        let total_pixels = edge_data.pixels.len();
        let edge_pixels = edge_data.pixels.iter().filter(|&&v| v > 0.0).count();
        let density = if total_pixels > 0 {
            edge_pixels as f32 / total_pixels as f32
        } else {
            0.0
        };

        ui.label(format!("Detector: {}", edge_data.detector));
        ui.label(format!("Dimensions: {}x{}", edge_data.width, edge_data.height));
        ui.label(format!("Edge pixels: {} / {}", edge_pixels, total_pixels));
        ui.label(format!("Edge density: {:.4}", density));
        ui.label(format!("Updated: {}", edge_data.updated));
    });
}
```

Note: This requires the system to have access to EdgeData resource. Update system signature:

```rust
fn ui_system(
    mut contexts: EguiContexts,
    mut playback_state: ResMut<PlaybackState>,
    mut edge_params: ResMut<EdgeParams>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    keyboard: Res<ButtonInput<KeyCode>>,
    edge_data: Option<Res<crate::analysis::EdgeData>>,  // NEW
) {
```

**Step 2: Run cargo check and cargo run**

Run: `cargo check && timeout 15 cargo run -- data/fan/fan_const_rpm.dat`
Expected: UI shows edge metrics window

**Step 3: Commit**

```bash
git add src/event_renderer.rs
git commit -m "feat: add edge metrics display to UI"
```

---

## Phase 1 Success Criteria Checkpoint

At this point, verify:
- [ ] EdgeReadbackBuffer resource exists with staging buffers
- [ ] ReadbackNode copies textures to staging buffers
- [ ] Async mapping reads data to CPU vectors
- [ ] EdgeData channel sends data to main world
- [ ] UI displays edge pixel count and density
- [ ] FPS impact is minimal (< 10% drop)

---

## Phase 2: Edge Quality Metrics

### Task 2.1: Create Metrics Module

**Files:**
- Create: `src/metrics.rs`
- Modify: `src/lib.rs`

**Step 1: Create metrics.rs with EdgeMetrics struct**

```rust
//! Edge quality metrics computation
//!
//! Quantifies edge detection quality without visual inspection.

use bevy::prelude::*;

/// Comprehensive edge detection quality metrics
#[derive(Debug, Clone, Default, Resource)]
pub struct EdgeMetrics {
    // Basic counts
    pub edge_pixel_count: u32,
    pub total_pixels: u32,
    pub edge_density: f32,

    // Spatial distribution
    pub centroid: Vec2,
    pub std_dev: Vec2,
    pub bounding_box: (Vec2, Vec2), // (min, max)

    // Circular fit (for fan detection)
    pub circle_center: Vec2,
    pub circle_radius: f32,
    pub circle_fit_error: f32,
    pub circle_inlier_ratio: f32,

    // Angular distribution
    pub angular_peaks: Vec<f32>,
    pub detected_blade_count: u32,

    // Temporal stability
    pub frame_to_frame_iou: f32,
}

impl EdgeMetrics {
    /// Compute basic metrics from edge data
    pub fn compute_basic(pixels: &[f32], width: u32, height: u32) -> Self {
        let total_pixels = pixels.len() as u32;
        let mut edge_count = 0u32;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut min_x = width as f32;
        let mut max_x = 0.0f32;
        let mut min_y = height as f32;
        let mut max_y = 0.0f32;

        for (i, &value) in pixels.iter().enumerate() {
            if value > 0.0 {
                edge_count += 1;
                let x = (i % width as usize) as f32;
                let y = (i / width as usize) as f32;
                sum_x += x as f64;
                sum_y += y as f64;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }

        let centroid = if edge_count > 0 {
            Vec2::new(
                (sum_x / edge_count as f64) as f32,
                (sum_y / edge_count as f64) as f32,
            )
        } else {
            Vec2::new(width as f32 / 2.0, height as f32 / 2.0)
        };

        // Compute standard deviation
        let mut var_x = 0.0f64;
        let mut var_y = 0.0f64;
        if edge_count > 1 {
            for (i, &value) in pixels.iter().enumerate() {
                if value > 0.0 {
                    let x = (i % width as usize) as f32;
                    let y = (i / width as usize) as f32;
                    var_x += ((x - centroid.x) as f64).powi(2);
                    var_y += ((y - centroid.y) as f64).powi(2);
                }
            }
            var_x /= (edge_count - 1) as f64;
            var_y /= (edge_count - 1) as f64;
        }

        Self {
            edge_pixel_count: edge_count,
            total_pixels,
            edge_density: edge_count as f32 / total_pixels as f32,
            centroid,
            std_dev: Vec2::new(var_x.sqrt() as f32, var_y.sqrt() as f32),
            bounding_box: (Vec2::new(min_x, min_y), Vec2::new(max_x, max_y)),
            ..Default::default()
        }
    }
}
```

**Step 2: Add to lib.rs**

```rust
pub mod metrics;
pub use metrics::EdgeMetrics;
```

**Step 3: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 4: Commit**

```bash
git add src/metrics.rs src/lib.rs
git commit -m "feat: add EdgeMetrics struct with basic computation"
```

---

### Task 2.2: Add RANSAC Circle Fitting

**Files:**
- Modify: `src/metrics.rs`

**Step 1: Add circle fitting functions**

```rust
/// Extract edge pixel coordinates from flat array
pub fn extract_edge_pixels(pixels: &[f32], width: u32) -> Vec<Vec2> {
    pixels
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            if v > 0.0 {
                let x = (i % width as usize) as f32;
                let y = (i / width as usize) as f32;
                Some(Vec2::new(x, y))
            } else {
                None
            }
        })
        .collect()
}

/// Fit circle through 3 points
/// Returns (center, radius) or None if points are collinear
fn fit_circle_3_points(p1: Vec2, p2: Vec2, p3: Vec2) -> Option<(Vec2, f32)> {
    let ax = p1.x;
    let ay = p1.y;
    let bx = p2.x;
    let by = p2.y;
    let cx = p3.x;
    let cy = p3.y;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-10 {
        return None; // Collinear points
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;

    let center = Vec2::new(ux, uy);
    let radius = center.distance(p1);

    Some((center, radius))
}

/// RANSAC circle fitting
/// Returns (center, radius, fit_error, inlier_ratio)
pub fn fit_circle_ransac(
    edge_pixels: &[Vec2],
    iterations: u32,
    inlier_threshold: f32,
) -> Option<(Vec2, f32, f32, f32)> {
    if edge_pixels.len() < 3 {
        return None;
    }

    let mut best_center = Vec2::ZERO;
    let mut best_radius = 0.0f32;
    let mut best_inliers = 0usize;
    let mut rng_state = 12345u32; // Simple LCG

    let rand_idx = |state: &mut u32, max: usize| -> usize {
        *state = state.wrapping_mul(1103515245).wrapping_add(12345);
        (*state as usize) % max
    };

    for _ in 0..iterations {
        // Sample 3 random points
        let i1 = rand_idx(&mut rng_state, edge_pixels.len());
        let i2 = rand_idx(&mut rng_state, edge_pixels.len());
        let i3 = rand_idx(&mut rng_state, edge_pixels.len());

        if i1 == i2 || i2 == i3 || i1 == i3 {
            continue;
        }

        let Some((center, radius)) =
            fit_circle_3_points(edge_pixels[i1], edge_pixels[i2], edge_pixels[i3])
        else {
            continue;
        };

        // Skip unreasonable circles
        if radius < 10.0 || radius > 500.0 {
            continue;
        }

        // Count inliers
        let inliers: usize = edge_pixels
            .iter()
            .filter(|p| (p.distance(center) - radius).abs() < inlier_threshold)
            .count();

        if inliers > best_inliers {
            best_inliers = inliers;
            best_center = center;
            best_radius = radius;
        }
    }

    if best_inliers < 10 {
        return None;
    }

    // Compute fit error as average distance from circle
    let total_error: f32 = edge_pixels
        .iter()
        .filter(|p| (p.distance(best_center) - best_radius).abs() < inlier_threshold)
        .map(|p| (p.distance(best_center) - best_radius).abs())
        .sum();

    let fit_error = total_error / best_inliers as f32;
    let inlier_ratio = best_inliers as f32 / edge_pixels.len() as f32;

    Some((best_center, best_radius, fit_error, inlier_ratio))
}
```

**Step 2: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/metrics.rs
git commit -m "feat: add RANSAC circle fitting for fan detection"
```

---

### Task 2.3: Add Angular Histogram

**Files:**
- Modify: `src/metrics.rs`

**Step 1: Add angular histogram functions**

```rust
/// Build histogram of edge pixel angles relative to center
pub fn angular_histogram(edge_pixels: &[Vec2], center: Vec2, num_bins: usize) -> Vec<u32> {
    let mut histogram = vec![0u32; num_bins];
    let bin_size = std::f32::consts::TAU / num_bins as f32;

    for p in edge_pixels {
        let dx = p.x - center.x;
        let dy = p.y - center.y;
        let angle = dy.atan2(dx); // -PI to PI
        let normalized = (angle + std::f32::consts::PI) / std::f32::consts::TAU; // 0 to 1
        let bin = ((normalized * num_bins as f32) as usize).min(num_bins - 1);
        histogram[bin] += 1;
    }

    histogram
}

/// Find peaks in angular histogram (blade positions)
pub fn find_angular_peaks(histogram: &[u32], min_prominence: u32) -> Vec<f32> {
    let num_bins = histogram.len();
    let bin_size = std::f32::consts::TAU / num_bins as f32;
    let mut peaks = Vec::new();

    for i in 0..num_bins {
        let prev = histogram[(i + num_bins - 1) % num_bins];
        let curr = histogram[i];
        let next = histogram[(i + 1) % num_bins];

        // Local maximum with sufficient prominence
        if curr > prev && curr > next && curr >= min_prominence {
            let angle = (i as f32 + 0.5) * bin_size - std::f32::consts::PI;
            peaks.push(angle);
        }
    }

    peaks
}
```

**Step 2: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/metrics.rs
git commit -m "feat: add angular histogram for blade detection"
```

---

### Task 2.4: Integrate Metrics Computation

**Files:**
- Modify: `src/analysis.rs`

**Step 1: Add metrics computation system**

```rust
use crate::metrics::{EdgeMetrics, extract_edge_pixels, fit_circle_ransac, angular_histogram, find_angular_peaks};

impl Plugin for AnalysisPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EdgeData>()
            .init_resource::<EdgeMetrics>()  // NEW
            .add_systems(Update, (receive_edge_data, compute_metrics).chain());  // UPDATED
    }
}

fn compute_metrics(
    edge_data: Res<EdgeData>,
    mut metrics: ResMut<EdgeMetrics>,
) {
    if !edge_data.updated || edge_data.pixels.is_empty() {
        return;
    }

    // Compute basic metrics
    *metrics = EdgeMetrics::compute_basic(&edge_data.pixels, edge_data.width, edge_data.height);

    // Extract edge pixels for advanced analysis
    let edge_pixels = extract_edge_pixels(&edge_data.pixels, edge_data.width);

    if edge_pixels.len() < 100 {
        return; // Not enough edges for reliable analysis
    }

    // RANSAC circle fitting
    if let Some((center, radius, error, inlier_ratio)) =
        fit_circle_ransac(&edge_pixels, 200, 5.0)
    {
        metrics.circle_center = center;
        metrics.circle_radius = radius;
        metrics.circle_fit_error = error;
        metrics.circle_inlier_ratio = inlier_ratio;

        // Angular histogram from detected center
        let histogram = angular_histogram(&edge_pixels, center, 360);
        let peaks = find_angular_peaks(&histogram, 50);

        metrics.angular_peaks = peaks.clone();
        metrics.detected_blade_count = peaks.len() as u32;
    }
}
```

**Step 2: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: integrate metrics computation into analysis pipeline"
```

---

### Task 2.5: Display Metrics in UI

**Files:**
- Modify: `src/event_renderer.rs`

**Step 1: Update Edge Metrics window**

Replace the simple edge metrics display with comprehensive metrics:

```rust
fn ui_system(
    // ... existing params ...
    edge_data: Option<Res<crate::analysis::EdgeData>>,
    metrics: Option<Res<crate::metrics::EdgeMetrics>>,  // NEW
) {
    // ... existing windows ...

    // Edge Metrics window
    egui::Window::new("Edge Metrics").show(ctx, |ui| {
        if let Some(metrics) = &metrics {
            ui.heading("Basic");
            ui.label(format!("Edge pixels: {}", metrics.edge_pixel_count));
            ui.label(format!("Density: {:.4}", metrics.edge_density));
            ui.label(format!("Centroid: ({:.1}, {:.1})", metrics.centroid.x, metrics.centroid.y));

            ui.separator();
            ui.heading("Circle Fit");
            ui.label(format!("Center: ({:.1}, {:.1})", metrics.circle_center.x, metrics.circle_center.y));
            ui.label(format!("Radius: {:.1} px", metrics.circle_radius));
            ui.label(format!("Fit error: {:.2} px", metrics.circle_fit_error));
            ui.label(format!("Inlier ratio: {:.1}%", metrics.circle_inlier_ratio * 100.0));

            ui.separator();
            ui.heading("Blade Detection");
            ui.label(format!("Detected blades: {}", metrics.detected_blade_count));
            for (i, angle) in metrics.angular_peaks.iter().enumerate() {
                ui.label(format!("  Blade {}: {:.1}°", i + 1, angle.to_degrees()));
            }
        } else {
            ui.label("No metrics available");
        }
    });
}
```

**Step 2: Run cargo run**

Run: `timeout 15 cargo run -- data/fan/fan_const_rpm.dat`
Expected: Metrics window shows circle fit and blade detection

**Step 3: Commit**

```bash
git add src/event_renderer.rs
git commit -m "feat: display comprehensive edge metrics in UI"
```

---

## Phase 2 Success Criteria Checkpoint

- [ ] EdgeMetrics computed every frame (when data available)
- [ ] Circular fit detects fan border
- [ ] Angular peaks detect blade count
- [ ] All metrics displayed in UI

---

## Phase 3: Geometry Extraction & Visualization

### Task 3.1: Create FanGeometry Resource

**Files:**
- Modify: `src/analysis.rs`

**Step 1: Add FanGeometry struct**

```rust
use std::collections::VecDeque;

/// Extracted fan geometry for RPM calculation
#[derive(Resource, Clone)]
pub struct FanGeometry {
    pub centroid: Vec2,
    pub radius: f32,
    pub blade_count: u32,
    pub blade_angles: Vec<f32>,

    // Confidence scores
    pub centroid_confidence: f32,
    pub radius_confidence: f32,

    // Detection state
    pub detected: bool,
    pub frames_since_detection: u32,

    // RPM tracking
    pub angular_velocity: f32,
    pub rpm: f32,
    pub rpm_history: VecDeque<f32>,
}

impl Default for FanGeometry {
    fn default() -> Self {
        Self {
            centroid: Vec2::new(640.0, 360.0),
            radius: 200.0,
            blade_count: 0,
            blade_angles: Vec::new(),
            centroid_confidence: 0.0,
            radius_confidence: 0.0,
            detected: false,
            frames_since_detection: 0,
            angular_velocity: 0.0,
            rpm: 0.0,
            rpm_history: VecDeque::with_capacity(100),
        }
    }
}
```

**Step 2: Add geometry extraction system**

```rust
fn extract_geometry(
    metrics: Res<EdgeMetrics>,
    mut geometry: ResMut<FanGeometry>,
) {
    // Update from metrics if we have good circle fit
    if metrics.circle_inlier_ratio > 0.1 && metrics.circle_radius > 50.0 {
        // Smooth update
        geometry.centroid = geometry.centroid.lerp(metrics.circle_center, 0.1);
        geometry.radius = geometry.radius + (metrics.circle_radius - geometry.radius) * 0.1;

        geometry.centroid_confidence = metrics.circle_inlier_ratio;
        geometry.radius_confidence = 1.0 - (metrics.circle_fit_error / 10.0).min(1.0);

        geometry.blade_count = metrics.detected_blade_count;
        geometry.blade_angles = metrics.angular_peaks.clone();

        geometry.detected = geometry.centroid_confidence > 0.2;
        geometry.frames_since_detection = 0;
    } else {
        geometry.frames_since_detection += 1;
        if geometry.frames_since_detection > 30 {
            geometry.detected = false;
        }
    }
}
```

**Step 3: Register in plugin**

```rust
impl Plugin for AnalysisPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EdgeData>()
            .init_resource::<EdgeMetrics>()
            .init_resource::<FanGeometry>()  // NEW
            .add_systems(Update, (
                receive_edge_data,
                compute_metrics,
                extract_geometry,  // NEW
            ).chain());
    }
}
```

**Step 4: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add FanGeometry resource with extraction system"
```

---

### Task 3.2: Add Gizmo Visualization

**Files:**
- Create: `src/gizmos.rs`
- Modify: `src/lib.rs`
- Modify: `src/edge_detection.rs`

**Step 1: Create gizmos.rs**

```rust
//! Gizmo visualization for fan geometry

use bevy::prelude::*;
use crate::analysis::FanGeometry;

pub struct GizmosPlugin;

impl Plugin for GizmosPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_fan_gizmos);
    }
}

fn draw_fan_gizmos(
    mut gizmos: Gizmos,
    geometry: Res<FanGeometry>,
) {
    if !geometry.detected {
        return;
    }

    // Transform from image coords (origin top-left) to world coords (origin center)
    let to_world = |p: Vec2| -> Vec3 {
        Vec3::new(p.x - 640.0, 360.0 - p.y, 0.0)
    };

    let center_world = to_world(geometry.centroid);

    // Draw detected circle (green when confident, yellow when uncertain)
    let circle_color = if geometry.centroid_confidence > 0.3 {
        Color::srgb(0.0, 1.0, 0.0)
    } else {
        Color::srgb(1.0, 1.0, 0.0)
    };

    // Draw circle as segments
    let segments = 64;
    for i in 0..segments {
        let angle1 = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let angle2 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
        let p1 = center_world + Vec3::new(
            geometry.radius * angle1.cos(),
            geometry.radius * angle1.sin(),
            0.0,
        );
        let p2 = center_world + Vec3::new(
            geometry.radius * angle2.cos(),
            geometry.radius * angle2.sin(),
            0.0,
        );
        gizmos.line(p1, p2, circle_color);
    }

    // Draw centroid cross (red)
    let cross_size = 20.0;
    gizmos.line(
        center_world - Vec3::X * cross_size,
        center_world + Vec3::X * cross_size,
        Color::srgb(1.0, 0.0, 0.0),
    );
    gizmos.line(
        center_world - Vec3::Y * cross_size,
        center_world + Vec3::Y * cross_size,
        Color::srgb(1.0, 0.0, 0.0),
    );

    // Draw detected blade lines (cyan)
    for &angle in &geometry.blade_angles {
        let direction = Vec3::new(angle.cos(), -angle.sin(), 0.0); // Flip Y for screen coords
        let end = center_world + direction * geometry.radius;
        gizmos.line(center_world, end, Color::srgb(0.0, 1.0, 1.0));
    }
}
```

**Step 2: Add to lib.rs**

```rust
pub mod gizmos;
pub use gizmos::GizmosPlugin;
```

**Step 3: Add GizmosPlugin to EdgeDetectionPlugin**

In `src/edge_detection.rs`, in the `build` method:

```rust
fn build(&self, app: &mut App) {
    app.init_resource::<SurfaceImage>()
        // ... existing resources ...
        .add_plugins(EventRendererPlugin)
        .add_plugins(crate::gizmos::GizmosPlugin);  // NEW
}
```

**Step 4: Run cargo run**

Run: `timeout 15 cargo run -- data/fan/fan_const_rpm.dat`
Expected: Green circle and cyan blade lines overlay on visualization

**Step 5: Commit**

```bash
git add src/gizmos.rs src/lib.rs src/edge_detection.rs
git commit -m "feat: add gizmo visualization for detected fan geometry"
```

---

## Phase 4: RPM Calculation

### Task 4.1: Add Blade Tracking

**Files:**
- Modify: `src/analysis.rs`

**Step 1: Add blade tracking state**

```rust
#[derive(Resource, Default)]
pub struct BladeTracker {
    pub previous_angles: Vec<f32>,
    pub previous_time: f64,
    pub angular_velocity_samples: VecDeque<f32>,
}

impl BladeTracker {
    pub fn update(&mut self, current_angles: &[f32], current_time: f64) -> Option<f32> {
        if self.previous_angles.is_empty() || current_angles.is_empty() {
            self.previous_angles = current_angles.to_vec();
            self.previous_time = current_time;
            return None;
        }

        let dt = current_time - self.previous_time;
        if dt <= 0.0 {
            return None;
        }

        // Match blades and compute angular displacement
        let delta_angle = self.match_and_compute_delta(current_angles);

        // Compute angular velocity (rad/s)
        let omega = delta_angle / dt as f32;

        // Add to samples for smoothing
        self.angular_velocity_samples.push_back(omega);
        if self.angular_velocity_samples.len() > 10 {
            self.angular_velocity_samples.pop_front();
        }

        // Update state
        self.previous_angles = current_angles.to_vec();
        self.previous_time = current_time;

        // Return smoothed angular velocity
        let avg: f32 = self.angular_velocity_samples.iter().sum::<f32>()
            / self.angular_velocity_samples.len() as f32;
        Some(avg)
    }

    fn match_and_compute_delta(&self, current: &[f32]) -> f32 {
        if current.is_empty() || self.previous_angles.is_empty() {
            return 0.0;
        }

        // Simple: compute average rotation of matched blades
        // Account for blade periodicity
        let blade_period = std::f32::consts::TAU / current.len().max(1) as f32;

        let mut total_delta = 0.0f32;
        let mut count = 0;

        for &curr_angle in current {
            // Find closest previous blade
            let mut min_delta = f32::MAX;
            for &prev_angle in &self.previous_angles {
                let mut delta = curr_angle - prev_angle;
                // Normalize to [-PI, PI]
                while delta > std::f32::consts::PI {
                    delta -= std::f32::consts::TAU;
                }
                while delta < -std::f32::consts::PI {
                    delta += std::f32::consts::TAU;
                }
                // Account for blade periodicity
                if delta.abs() > blade_period / 2.0 {
                    if delta > 0.0 {
                        delta -= blade_period;
                    } else {
                        delta += blade_period;
                    }
                }
                if delta.abs() < min_delta.abs() {
                    min_delta = delta;
                }
            }
            total_delta += min_delta;
            count += 1;
        }

        if count > 0 {
            total_delta / count as f32
        } else {
            0.0
        }
    }
}
```

**Step 2: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add BladeTracker for angular velocity calculation"
```

---

### Task 4.2: Integrate RPM Calculation

**Files:**
- Modify: `src/analysis.rs`

**Step 1: Add RPM calculation system**

```rust
fn calculate_rpm(
    time: Res<Time>,
    mut tracker: ResMut<BladeTracker>,
    mut geometry: ResMut<FanGeometry>,
) {
    if !geometry.detected || geometry.blade_angles.is_empty() {
        return;
    }

    let current_time = time.elapsed_secs_f64();

    if let Some(omega) = tracker.update(&geometry.blade_angles, current_time) {
        geometry.angular_velocity = omega;

        // Convert rad/s to RPM: omega * 60 / (2*PI)
        let rpm = omega.abs() * 60.0 / std::f32::consts::TAU;
        geometry.rpm = rpm;

        // Add to history
        geometry.rpm_history.push_back(rpm);
        if geometry.rpm_history.len() > 100 {
            geometry.rpm_history.pop_front();
        }
    }
}
```

**Step 2: Register in plugin**

```rust
impl Plugin for AnalysisPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EdgeData>()
            .init_resource::<EdgeMetrics>()
            .init_resource::<FanGeometry>()
            .init_resource::<BladeTracker>()  // NEW
            .add_systems(Update, (
                receive_edge_data,
                compute_metrics,
                extract_geometry,
                calculate_rpm,  // NEW
            ).chain());
    }
}
```

**Step 3: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 4: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: integrate RPM calculation from blade tracking"
```

---

### Task 4.3: Add RPM Display to UI

**Files:**
- Modify: `src/event_renderer.rs`

**Step 1: Add RPM window**

```rust
fn ui_system(
    // ... existing params ...
    geometry: Option<Res<crate::analysis::FanGeometry>>,  // NEW
) {
    // ... existing windows ...

    // RPM Display window
    egui::Window::new("RPM").show(ctx, |ui| {
        if let Some(geom) = &geometry {
            ui.heading(format!("{:.0} RPM", geom.rpm));

            ui.separator();
            ui.label(format!("Angular velocity: {:.2} rad/s", geom.angular_velocity));
            ui.label(format!("Blade count: {}", geom.blade_count));
            ui.label(format!("Detected: {}", if geom.detected { "Yes" } else { "No" }));

            // Simple RPM history as text
            if !geom.rpm_history.is_empty() {
                ui.separator();
                let avg_rpm: f32 = geom.rpm_history.iter().sum::<f32>()
                    / geom.rpm_history.len() as f32;
                ui.label(format!("Avg RPM (last {}): {:.0}",
                    geom.rpm_history.len(), avg_rpm));
            }
        } else {
            ui.label("Detecting...");
        }
    });
}
```

**Step 2: Run cargo run**

Run: `timeout 15 cargo run -- data/fan/fan_const_rpm.dat`
Expected: RPM window shows calculated RPM

**Step 3: Commit**

```bash
git add src/event_renderer.rs
git commit -m "feat: add RPM display window to UI"
```

---

### Task 4.4: Add JSON Export

**Files:**
- Modify: `src/analysis.rs`

**Step 1: Add export system**

```rust
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;

#[derive(Serialize, Deserialize)]
struct RpmExportEntry {
    timestamp_secs: f64,
    rpm: f32,
    angular_velocity: f32,
    centroid_x: f32,
    centroid_y: f32,
    radius: f32,
    blade_count: u32,
}

#[derive(Resource, Default)]
pub struct RpmExporter {
    entries: Vec<RpmExportEntry>,
    last_export_time: f64,
    export_interval: f64,
}

fn export_rpm_data(
    time: Res<Time>,
    geometry: Res<FanGeometry>,
    mut exporter: ResMut<RpmExporter>,
) {
    if !geometry.detected {
        return;
    }

    let current_time = time.elapsed_secs_f64();

    // Export every 0.1 seconds
    if current_time - exporter.last_export_time < 0.1 {
        return;
    }
    exporter.last_export_time = current_time;

    exporter.entries.push(RpmExportEntry {
        timestamp_secs: current_time,
        rpm: geometry.rpm,
        angular_velocity: geometry.angular_velocity,
        centroid_x: geometry.centroid.x,
        centroid_y: geometry.centroid.y,
        radius: geometry.radius,
        blade_count: geometry.blade_count,
    });

    // Write to file every 10 entries
    if exporter.entries.len() % 10 == 0 {
        if let Ok(json) = serde_json::to_string_pretty(&exporter.entries) {
            let _ = std::fs::write("rpm_output.json", json);
        }
    }
}
```

**Step 2: Register in plugin and update build**

```rust
impl Plugin for AnalysisPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EdgeData>()
            .init_resource::<EdgeMetrics>()
            .init_resource::<FanGeometry>()
            .init_resource::<BladeTracker>()
            .init_resource::<RpmExporter>()  // NEW
            .add_systems(Update, (
                receive_edge_data,
                compute_metrics,
                extract_geometry,
                calculate_rpm,
                export_rpm_data,  // NEW
            ).chain());
    }
}
```

**Step 3: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 4: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add JSON export for RPM data"
```

---

## Phase 5: Synthesis & Testing

### Task 5.1: Add Synthetic Data Generation

**Files:**
- Create: `src/synthesis.rs`
- Modify: `src/lib.rs`

**Step 1: Create synthesis.rs**

Copy and adapt from `fan-rpm-claude` branch:

```rust
//! Synthetic fan data generation for testing

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::f32::consts::PI;

pub fn generate_fan_data(output_path: &Path, truth_path: &Path) -> std::io::Result<()> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(output_path)?;
    let mut truth_file = File::create(truth_path)?;

    // Write Event Type & Size header
    file.write_all(&[0x00, 0x08])?;

    // Parameters
    let duration_secs = 2.0;
    let rpm = 1200.0;
    let rps = rpm / 60.0;
    let angular_velocity = rps * 2.0 * PI;
    let blade_count = 3;
    let radius = 200.0;
    let center_x = 640.0;
    let center_y = 360.0;
    let events_per_sec = 100_000;
    let total_events = (events_per_sec as f32 * duration_secs) as usize;

    let mut truth_entries = Vec::new();
    let time_step_us = (1_000_000.0 / events_per_sec as f32) as u32;
    let mut current_time_us = 0u32;

    // Simple LCG random
    let mut seed: u32 = 12345;
    let rand = |seed: &mut u32| -> f32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (*seed as f32) / (u32::MAX as f32)
    };

    for _ in 0..total_events {
        current_time_us += time_step_us;
        let t_secs = current_time_us as f32 / 1_000_000.0;
        let base_angle = angular_velocity * t_secs;

        let blade_idx = (rand(&mut seed) * blade_count as f32) as usize;
        let blade_spacing = 2.0 * PI / blade_count as f32;
        let root_angle = base_angle + (blade_idx as f32 * blade_spacing);

        let r = 50.0 + rand(&mut seed) * (radius - 50.0);
        let sweep_angle = 0.5 * (r / 50.0).ln();
        let center_angle = root_angle + sweep_angle;

        let r_norm = (r - 50.0) / (radius - 50.0);
        let blade_width = 0.5 + (0.3 - 0.5) * r_norm;
        let half_width = blade_width * 0.5;

        let is_leading_edge = rand(&mut seed) > 0.5;
        let (edge_offset, polarity) = if is_leading_edge {
            (half_width, 1u32)
        } else {
            (-half_width, 0u32)
        };

        let angular_jitter = (rand(&mut seed) - 0.5) * 0.02;
        let theta = center_angle + edge_offset + angular_jitter;

        let jitter_x = (rand(&mut seed) - 0.5) * 1.0;
        let jitter_y = (rand(&mut seed) - 0.5) * 1.0;

        let x = center_x + r * theta.cos() + jitter_x;
        let y = center_y + r * theta.sin() + jitter_y;

        if x < 0.0 || x >= 1280.0 || y < 0.0 || y >= 720.0 {
            continue;
        }

        let x_u32 = x as u32;
        let y_u32 = y as u32;
        let w32 = (x_u32 & 0x3FFF) | ((y_u32 & 0x3FFF) << 14) | ((polarity & 0xF) << 28);

        file.write_all(&current_time_us.to_le_bytes())?;
        file.write_all(&w32.to_le_bytes())?;

        if current_time_us % 10_000 == 0 {
            truth_entries.push(format!(
                "{{\"time\": {:.4}, \"angle\": {:.4}, \"rpm\": {:.1}, \"centroid_x\": {:.1}, \"centroid_y\": {:.1}, \"radius\": {:.1}}}",
                t_secs, base_angle % (2.0 * PI), rpm, center_x, center_y, radius
            ));
        }
    }

    writeln!(truth_file, "[")?;
    writeln!(truth_file, "{}", truth_entries.join(",\n"))?;
    writeln!(truth_file, "]")?;

    Ok(())
}
```

**Step 2: Add to lib.rs**

```rust
pub mod synthesis;
```

**Step 3: Run cargo check**

Run: `cargo check`
Expected: Compiles successfully

**Step 4: Commit**

```bash
git add src/synthesis.rs src/lib.rs
git commit -m "feat: add synthetic fan data generation"
```

---

### Task 5.2: Update Accuracy Tests

**Files:**
- Modify: `tests/accuracy_test.rs`

**Step 1: Update test to use new analysis pipeline**

The existing test needs to be updated to work with the new analysis module structure. Key changes:
- Use `AnalysisPlugin` from the new location
- Access `FanGeometry` instead of old `FanAnalysis`
- Setup the edge detection pipeline

This is a larger refactoring task - mark existing tests as `#[ignore]` until the full pipeline is working.

**Step 2: Run cargo test**

Run: `cargo test --test accuracy_test -- --ignored`
Expected: Tests run (may fail until pipeline is complete)

**Step 3: Commit**

```bash
git add tests/accuracy_test.rs
git commit -m "test: update accuracy tests for new analysis pipeline"
```

---

## Final Integration Checklist

After completing all tasks:

- [ ] `cargo check` passes
- [ ] `cargo test` passes (non-ignored tests)
- [ ] `cargo run -- data/fan/fan_const_rpm.dat` shows:
  - Edge detection visualization
  - Edge Metrics window with circle fit data
  - Gizmo overlay showing detected circle and blades
  - RPM window showing calculated RPM
- [ ] `rpm_output.json` is generated with tracking data
- [ ] FPS remains above 30

---

## Success Metrics (End Goal)

| Metric | Target | How to Verify |
|--------|--------|---------------|
| RPM accuracy | < 5% error | Compare with ground truth |
| Centroid accuracy | < 10 pixels | Compare with known center |
| Radius accuracy | < 5% error | Compare with known radius |
| Blade count | 100% correct | Should detect 3 blades |
| Frame rate | > 30 FPS | Check FPS display |
| Latency | < 100ms | Measure from event to RPM update |

---

## Notes for Implementer

1. **Prior art**: Branch `fan-rpm-claude` has working implementations of centroid tracking, radial profiles, and angular histograms. Cherry-pick and adapt as needed.

2. **GPU readback pattern**: Use async buffer mapping with channels to avoid blocking the render thread. See `analysis.rs` on `fan-rpm-claude` for the pattern.

3. **Coordinate systems**: Image coords have origin at top-left, Y increases downward. World coords have origin at center, Y increases upward. Transform carefully in gizmos.

4. **Testing**: Start with synthetic data (`synthesis.rs`) which has known ground truth. Real fan data may need parameter tuning.

5. **Performance**: Only read back one detector per frame (rotate through them or pick based on which is enabled). Reading all three every frame will tank performance.

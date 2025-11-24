# MVP Edge Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a clean MVP with raw event visualization (Layer 0) and Spatial Timestamp Gradient edge detection (Layer 1) in a new `src/mvp/` module.

**Architecture:** Create fresh `src/mvp/` module with EventAccumulationNode (Layer 0: red/blue raw events) and GradientNode (Layer 1: yellow edge overlay using Sobel operator). GPU compute pipeline processes timestamp gradients, fragment shader composites with 50% alpha blend. UI provides playback controls, window size, and edge threshold tuning.

**Tech Stack:** Bevy 0.15, WGPU compute shaders, bevy_egui for UI

---

## Task 1: Create MVP Module Structure

**Files:**
- Create: `src/mvp/mod.rs`
- Create: `src/mvp/gpu.rs`
- Create: `src/mvp/render.rs`
- Create: `src/mvp/playback.rs`
- Modify: `src/lib.rs`

**Step 1: Create mvp module declaration**

Create `src/mvp/mod.rs`:

```rust
pub mod gpu;
pub mod playback;
pub mod render;

use bevy::prelude::*;

pub struct MvpPlugin;

impl Plugin for MvpPlugin {
    fn build(&self, app: &mut App) {
        info!("MvpPlugin: Building MVP visualization pipeline");
    }
}
```

**Step 2: Export mvp module from lib**

Add to `src/lib.rs` after existing module declarations:

```rust
pub mod mvp;
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: SUCCESS (module compiles but does nothing yet)

**Step 4: Commit**

```bash
git add src/mvp/mod.rs src/mvp/gpu.rs src/mvp/render.rs src/mvp/playback.rs src/lib.rs
git commit -m "feat(mvp): create clean MVP module structure"
```

---

## Task 2: Implement PlaybackState Resource

**Files:**
- Modify: `src/mvp/playback.rs`

**Step 1: Define PlaybackState resource**

Write `src/mvp/playback.rs`:

```rust
use bevy::{
    prelude::*,
    render::extract_resource::ExtractResource,
};

#[derive(Resource, ExtractResource, Clone)]
pub struct PlaybackState {
    pub is_playing: bool,
    pub current_time: f32,   // Microseconds
    pub window_size: f32,    // Microseconds
    pub playback_speed: f32, // Real-time multiplier
    pub looping: bool,
    pub max_timestamp: u32,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            is_playing: true,
            current_time: 20000.0,
            window_size: 1_000_000.0,  // 1 second
            playback_speed: 0.1,        // 0.1x speed
            looping: true,
            max_timestamp: 1_000_000,
        }
    }
}

pub fn playback_system(time: Res<Time>, mut playback_state: ResMut<PlaybackState>) {
    if playback_state.is_playing {
        let delta_us = time.delta_secs() * 1_000_000.0 * playback_state.playback_speed;
        playback_state.current_time += delta_us;

        if playback_state.current_time > playback_state.max_timestamp as f32 {
            if playback_state.looping {
                playback_state.current_time = 0.0;
            } else {
                playback_state.current_time = playback_state.max_timestamp as f32;
                playback_state.is_playing = false;
            }
        }
    }
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add src/mvp/playback.rs
git commit -m "feat(mvp): implement playback state and system"
```

---

## Task 3: Implement GPU Data Structures

**Files:**
- Modify: `src/mvp/gpu.rs`

**Step 1: Define core GPU data structures**

Write `src/mvp/gpu.rs`:

```rust
use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::*,
    },
};
use bytemuck::{Pod, Zeroable};

// GPU event representation
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuEvent {
    pub timestamp: u32,
    pub x: u32,
    pub y: u32,
    pub polarity: u32,
}

// Main world event storage
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct EventData {
    pub events: Vec<GpuEvent>,
}

// Handle to surface texture (Layer 0 output)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct SurfaceImage {
    pub handle: Handle<Image>,
}

// Handle to gradient texture (Layer 1 output)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct GradientImage {
    pub handle: Handle<Image>,
}

// Render world GPU buffers
#[derive(Resource, Default)]
pub struct GpuEventBuffer {
    pub buffer: Option<Buffer>,
    pub count: u32,
    pub surface_buffer: Option<Buffer>,
    pub gradient_buffer: Option<Buffer>,
    pub dimensions: UVec2,
    pub dim_buffer: Option<Buffer>,
    pub uploaded: bool,
    pub bind_group_ready: bool,
}

// Edge detection parameters
#[derive(Resource, ExtractResource, Clone)]
pub struct EdgeParams {
    pub threshold: f32,
    pub show_gradient: bool,
}

impl Default for EdgeParams {
    fn default() -> Self {
        Self {
            threshold: 1000.0,
            show_gradient: true,
        }
    }
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add src/mvp/gpu.rs
git commit -m "feat(mvp): define GPU data structures and resources"
```

---

## Task 4: Implement Event Accumulation Pipeline (Layer 0)

**Files:**
- Modify: `src/mvp/gpu.rs`

**Step 1: Add EventComputePipeline resource**

Append to `src/mvp/gpu.rs`:

```rust
use bevy::render::{
    renderer::{RenderDevice, RenderQueue, RenderContext},
    render_graph::{Node, RenderLabel},
    render_asset::RenderAssets,
    texture::GpuImage,
};

#[derive(Resource)]
pub struct EventComputePipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for EventComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Event Accumulation Layout"),
            &[
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
            ],
        );

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/accumulation.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Event Accumulation Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        EventComputePipeline { layout, pipeline }
    }
}

#[derive(Resource)]
pub struct EventBindGroup(pub BindGroup);
```

**Step 2: Add prepare_events system**

Append to `src/mvp/gpu.rs`:

```rust
pub fn prepare_events(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gpu_buffer: ResMut<GpuEventBuffer>,
    event_data: Res<EventData>,
    playback_state: Res<crate::mvp::playback::PlaybackState>,
) {
    if event_data.events.is_empty() {
        return;
    }

    let width = 1280;
    let height = 720;

    // Calculate time window
    let window_end = playback_state.current_time as u32;
    let window_start = if window_end > playback_state.window_size as u32 {
        window_end - playback_state.window_size as u32
    } else {
        0
    };

    // Update dimension buffer every frame
    if let Some(dim_buffer) = &gpu_buffer.dim_buffer {
        let dimensions = [width, height, window_start, window_end];
        render_queue.write_buffer(dim_buffer, 0, bytemuck::cast_slice(&dimensions));
    }

    if gpu_buffer.uploaded {
        return;
    }

    // One-time upload
    info!("Uploading {} events to GPU", event_data.events.len());

    let byte_data: &[u8] = bytemuck::cast_slice(&event_data.events);
    gpu_buffer.buffer = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Event Buffer"),
            contents: byte_data,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        }),
    );
    gpu_buffer.count = event_data.events.len() as u32;
    gpu_buffer.dimensions = UVec2::new(width, height);

    let dimensions = [width, height, 0u32, 20000u32];
    gpu_buffer.dim_buffer = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dimensions),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }),
    );

    let size = width * height * 4;
    gpu_buffer.surface_buffer = Some(render_device.create_buffer(&BufferDescriptor {
        label: Some("Surface Buffer"),
        size: size as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    render_queue.write_buffer(
        gpu_buffer.surface_buffer.as_ref().unwrap(),
        0,
        &vec![0u8; size as usize],
    );

    gpu_buffer.uploaded = true;
}
```

**Step 3: Add queue_bind_group system**

Append to `src/mvp/gpu.rs`:

```rust
pub fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<EventComputePipeline>,
    render_device: Res<RenderDevice>,
    mut gpu_buffer: ResMut<GpuEventBuffer>,
) {
    if gpu_buffer.bind_group_ready {
        return;
    }

    if let (Some(events), Some(surface), Some(dim_buffer)) = (
        &gpu_buffer.buffer,
        &gpu_buffer.surface_buffer,
        &gpu_buffer.dim_buffer,
    ) {
        let bind_group = render_device.create_bind_group(
            Some("Event Bind Group"),
            &pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: events.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: surface.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dim_buffer.as_entire_binding(),
                },
            ],
        );
        commands.insert_resource(EventBindGroup(bind_group));
        gpu_buffer.bind_group_ready = true;
    }
}
```

**Step 4: Add EventAccumulationNode**

Append to `src/mvp/gpu.rs`:

```rust
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct EventLabel;

#[derive(Default)]
pub struct EventAccumulationNode;

impl Node for EventAccumulationNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<EventComputePipeline>();
        let Some(bind_group) = world.get_resource::<EventBindGroup>() else {
            return Ok(());
        };
        let gpu_buffer = world.resource::<GpuEventBuffer>();
        let surface_image = world.resource::<SurfaceImage>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();

        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            // Clear surface buffer
            if let Some(surface_buffer) = &gpu_buffer.surface_buffer {
                render_context
                    .command_encoder()
                    .clear_buffer(surface_buffer, 0, None);
            }

            // Run compute pass
            {
                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Event Accumulation"),
                        timestamp_writes: None,
                    });

                pass.set_pipeline(compute_pipeline);
                pass.set_bind_group(0, &bind_group.0, &[]);

                let workgroup_size = 64;
                let count = gpu_buffer.count;
                if count > 0 {
                    let total_workgroups = (count + workgroup_size - 1) / workgroup_size;
                    let max_workgroups_per_dim = 65535;

                    let x_workgroups = total_workgroups.min(max_workgroups_per_dim);
                    let y_workgroups =
                        (total_workgroups + max_workgroups_per_dim - 1) / max_workgroups_per_dim;

                    pass.dispatch_workgroups(x_workgroups, y_workgroups, 1);
                }
            }

            // Copy buffer to texture
            if let Some(surface_buffer) = &gpu_buffer.surface_buffer {
                if let Some(gpu_image) = gpu_images.get(&surface_image.handle) {
                    render_context.command_encoder().copy_buffer_to_texture(
                        TexelCopyBufferInfo {
                            buffer: surface_buffer,
                            layout: TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(gpu_buffer.dimensions.x * 4),
                                rows_per_image: Some(gpu_buffer.dimensions.y),
                            },
                        },
                        TexelCopyTextureInfo {
                            texture: &gpu_image.texture,
                            mip_level: 0,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        Extent3d {
                            width: gpu_buffer.dimensions.x,
                            height: gpu_buffer.dimensions.y,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
        }
        Ok(())
    }
}
```

**Step 5: Verify compilation**

Run: `cargo check`
Expected: SUCCESS

**Step 6: Commit**

```bash
git add src/mvp/gpu.rs
git commit -m "feat(mvp): implement event accumulation pipeline (Layer 0)"
```

---

## Task 5: Create Spatial Gradient Shader (Layer 1)

**Files:**
- Create: `assets/shaders/spatial_gradient.wgsl`

**Step 1: Write Sobel edge detection compute shader**

Create `assets/shaders/spatial_gradient.wgsl`:

```wgsl
@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var gradient_output: texture_storage_2d<r8unorm, write>;

struct EdgeParams {
    threshold: f32,
}

@group(0) @binding(2) var<uniform> params: EdgeParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(surface_texture);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Skip border pixels
    if (coords.x < 1 || coords.y < 1 || coords.x >= i32(dims.x) - 1 || coords.y >= i32(dims.y) - 1) {
        textureStore(gradient_output, coords, vec4<f32>(0.0));
        return;
    }

    // Load 3x3 neighborhood and extract timestamps
    var timestamps: array<f32, 9>;
    var idx = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let pos = coords + vec2<i32>(dx, dy);
            let packed = textureLoad(surface_texture, pos, 0).r;
            // Extract timestamp (ignore polarity bit)
            timestamps[idx] = f32(packed & 0x7FFFFFFFu);
            idx++;
        }
    }

    // Sobel kernels
    // Gx: [-1  0  1]    Gy: [-1 -2 -1]
    //     [-2  0  2]        [ 0  0  0]
    //     [-1  0  1]        [ 1  2  1]

    let gx = -timestamps[0] + timestamps[2]
             - 2.0 * timestamps[3] + 2.0 * timestamps[5]
             - timestamps[6] + timestamps[8];

    let gy = -timestamps[0] - 2.0 * timestamps[1] - timestamps[2]
             + timestamps[6] + 2.0 * timestamps[7] + timestamps[8];

    let magnitude = sqrt(gx * gx + gy * gy);

    // Threshold and write
    let edge_value = select(0.0, 1.0, magnitude > params.threshold);
    textureStore(gradient_output, coords, vec4<f32>(edge_value));
}
```

**Step 2: Verify shader compiles with project**

Run: `cargo check`
Expected: SUCCESS (shader will be validated at runtime)

**Step 3: Commit**

```bash
git add assets/shaders/spatial_gradient.wgsl
git commit -m "feat(mvp): add spatial gradient edge detection shader"
```

---

## Task 6: Implement Gradient Pipeline (Layer 1)

**Files:**
- Modify: `src/mvp/gpu.rs`

**Step 1: Add GradientPipeline resource**

Append to `src/mvp/gpu.rs`:

```rust
#[derive(Resource)]
pub struct GradientPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for GradientPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Gradient Pipeline Layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Uint,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
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
            ],
        );

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/spatial_gradient.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Spatial Gradient Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        GradientPipeline { layout, pipeline }
    }
}

#[derive(Resource)]
pub struct GradientBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct EdgeParamsBuffer(pub Buffer);
```

**Step 2: Add prepare_gradient system**

Append to `src/mvp/gpu.rs`:

```rust
pub fn prepare_gradient(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<GradientPipeline>,
    edge_params: Res<EdgeParams>,
    surface_image: Res<SurfaceImage>,
    gradient_image: Res<GradientImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    edge_buffer: Option<Res<EdgeParamsBuffer>>,
) {
    // Create or update edge params buffer
    let buffer = if let Some(existing) = edge_buffer {
        render_queue.write_buffer(&existing.0, 0, bytemuck::cast_slice(&[edge_params.threshold]));
        existing.0.clone()
    } else {
        let new_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Edge Params Buffer"),
            contents: bytemuck::cast_slice(&[edge_params.threshold]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        commands.insert_resource(EdgeParamsBuffer(new_buffer.clone()));
        new_buffer
    };

    // Create bind group if textures are ready
    if let (Some(surface_gpu), Some(gradient_gpu)) = (
        gpu_images.get(&surface_image.handle),
        gpu_images.get(&gradient_image.handle),
    ) {
        let bind_group = render_device.create_bind_group(
            Some("Gradient Bind Group"),
            &pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&surface_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&gradient_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        );
        commands.insert_resource(GradientBindGroup(bind_group));
    }
}
```

**Step 3: Add GradientNode**

Append to `src/mvp/gpu.rs`:

```rust
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct GradientLabel;

#[derive(Default)]
pub struct GradientNode;

impl Node for GradientNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GradientPipeline>();
        let Some(bind_group) = world.get_resource::<GradientBindGroup>() else {
            return Ok(());
        };

        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Spatial Gradient"),
                    timestamp_writes: None,
                });

            pass.set_pipeline(compute_pipeline);
            pass.set_bind_group(0, &bind_group.0, &[]);

            // Dispatch for 1280x720 with 8x8 workgroups
            let workgroups_x = (1280 + 7) / 8;
            let workgroups_y = (720 + 7) / 8;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        Ok(())
    }
}
```

**Step 4: Verify compilation**

Run: `cargo check`
Expected: SUCCESS

**Step 5: Commit**

```bash
git add src/mvp/gpu.rs
git commit -m "feat(mvp): implement gradient compute pipeline (Layer 1)"
```

---

## Task 7: Update Visualizer Shader for Compositing

**Files:**
- Modify: `assets/shaders/visualizer.wgsl`

**Step 1: Read current visualizer shader**

Run: Read the file to understand current structure

**Step 2: Add gradient texture binding**

Add after existing texture bindings:

```wgsl
@group(2) @binding(2) var gradient_texture: texture_2d<f32>;
@group(2) @binding(3) var gradient_sampler: sampler;
```

**Step 3: Add show_gradient uniform field**

Modify the params struct to add:

```wgsl
struct EventParams {
    width: f32,
    height: f32,
    time: f32,
    decay_tau: f32,
    show_gradient: u32,
}
```

**Step 4: Update fragment function for compositing**

Modify the fragment shader to composite layers:

```wgsl
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel_coord = vec2<i32>(in.uv * vec2<f32>(params.width, params.height));

    // Layer 0: Raw events (red/blue)
    let packed_val = textureLoad(surface_texture, pixel_coord, 0).r;
    let timestamp = f32(packed_val & 0x7FFFFFFFu);
    let polarity = (packed_val >> 31u) & 1u;

    let age = params.time - timestamp;
    let alpha = exp(-age / params.decay_tau);

    var raw_color: vec3<f32>;
    if (polarity == 1u) {
        raw_color = vec3<f32>(1.0, 0.0, 0.0) * alpha;  // Red
    } else {
        raw_color = vec3<f32>(0.0, 0.0, 1.0) * alpha;  // Blue
    }

    // Layer 1: Gradient edges (yellow)
    if (params.show_gradient == 1u) {
        let edge_val = textureSample(gradient_texture, gradient_sampler, in.uv).r;
        if (edge_val > 0.0) {
            let yellow = vec3<f32>(1.0, 1.0, 0.0);
            raw_color = mix(raw_color, yellow, 0.5);  // 50% alpha blend
        }
    }

    return vec4<f32>(raw_color, 1.0);
}
```

**Step 5: Verify shader will compile**

Run: `cargo check`
Expected: SUCCESS (runtime validation will occur when app runs)

**Step 6: Commit**

```bash
git add assets/shaders/visualizer.wgsl
git commit -m "feat(mvp): update visualizer shader for gradient compositing"
```

---

## Task 8: Implement Render Plugin and Material

**Files:**
- Modify: `src/mvp/render.rs`

**Step 1: Define EventMaterial**

Write `src/mvp/render.rs`:

```rust
use crate::mvp::gpu::{EventData, SurfaceImage, GradientImage, EdgeParams};
use crate::mvp::playback::PlaybackState;
use crate::loader::DatLoader;
use crate::EventFilePath;
use bevy::asset::RenderAssetUsages;
use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};

#[derive(ShaderType, Debug, Clone, Copy)]
struct EventParams {
    width: f32,
    height: f32,
    time: f32,
    decay_tau: f32,
    show_gradient: u32,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct EventMaterial {
    #[uniform(0)]
    params: EventParams,
    #[texture(1, sample_type = "u_int")]
    surface_texture: Handle<Image>,
    #[texture(2)]
    #[sampler(3)]
    gradient_texture: Handle<Image>,
}

impl Material for EventMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/visualizer.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

#[derive(Resource)]
struct CurrentMaterialHandle(Handle<EventMaterial>);
```

**Step 2: Add load_data system**

Append to `src/mvp/render.rs`:

```rust
fn load_data(
    mut commands: Commands,
    mut playback_state: ResMut<PlaybackState>,
    event_file_path: Res<EventFilePath>,
) {
    let path = &event_file_path.0;
    match DatLoader::load(path) {
        Ok(events) => {
            info!("MVP: Loaded {} events from {}", events.len(), path);
            if let Some(last) = events.last() {
                playback_state.max_timestamp = last.timestamp;
                playback_state.current_time = last.timestamp as f32;
                info!("MVP: Timestamp range: 0 to {}", last.timestamp);
            }
            commands.insert_resource(EventData { events });
        }
        Err(e) => {
            error!("MVP: Failed to load data from {}: {:?}", path, e);
            commands.insert_resource(EventData { events: Vec::new() });
            playback_state.max_timestamp = 0;
            playback_state.current_time = 0.0;
        }
    }
}
```

**Step 3: Add setup_scene system**

Append to `src/mvp/render.rs`:

```rust
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<EventMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut surface_image_res: ResMut<SurfaceImage>,
    mut gradient_image_res: ResMut<GradientImage>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 1000.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let width = 1280;
    let height = 720;
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    // Surface texture (R32Uint for timestamps)
    let mut surface_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    surface_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    let surface_handle = images.add(surface_image);
    surface_image_res.handle = surface_handle.clone();

    // Gradient texture (R8Unorm for edge magnitude)
    let mut gradient_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0],
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    gradient_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let gradient_handle = images.add(gradient_image);
    gradient_image_res.handle = gradient_handle.clone();

    // Material
    let material_handle = materials.add(EventMaterial {
        surface_texture: surface_handle,
        gradient_texture: gradient_handle,
        params: EventParams {
            width: 1280.0,
            height: 720.0,
            time: 20000.0,
            decay_tau: 50000.0,
            show_gradient: 1,
        },
    });
    commands.insert_resource(CurrentMaterialHandle(material_handle.clone()));

    // Quad
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(1280.0, 720.0))),
        MeshMaterial3d(material_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}
```

**Step 4: Add update_material_params system**

Append to `src/mvp/render.rs`:

```rust
fn update_material_params(
    playback_state: Res<PlaybackState>,
    edge_params: Res<EdgeParams>,
    mut materials: ResMut<Assets<EventMaterial>>,
    current_material: Res<CurrentMaterialHandle>,
) {
    if let Some(material) = materials.get_mut(&current_material.0) {
        material.params.time = playback_state.current_time;
        material.params.show_gradient = if edge_params.show_gradient { 1 } else { 0 };
    }
}
```

**Step 5: Verify compilation**

Run: `cargo check`
Expected: May have some errors about missing UI system - that's next

**Step 6: Commit**

```bash
git add src/mvp/render.rs
git commit -m "feat(mvp): implement render material and scene setup"
```

---

## Task 9: Implement UI System

**Files:**
- Modify: `src/mvp/render.rs`

**Step 1: Add ui_system**

Append to `src/mvp/render.rs`:

```rust
fn ui_system(
    mut contexts: EguiContexts,
    mut playback_state: ResMut<PlaybackState>,
    mut edge_params: ResMut<EdgeParams>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
) {
    // Playback Controls
    egui::Window::new("Playback Controls").show(contexts.ctx_mut(), |ui| {
        ui.horizontal(|ui| {
            if ui.button(if playback_state.is_playing { "Pause" } else { "Play" }).clicked() {
                playback_state.is_playing = !playback_state.is_playing;
            }
            ui.checkbox(&mut playback_state.looping, "Loop");
        });

        let max_time = playback_state.max_timestamp as f32;
        ui.add(
            egui::Slider::new(&mut playback_state.current_time, 0.0..=max_time)
                .text("Time (μs)"),
        );

        ui.add(
            egui::Slider::new(&mut playback_state.window_size, 1.0..=100_000.0)
                .text("Window (μs)")
                .logarithmic(true),
        );

        ui.add(
            egui::Slider::new(&mut playback_state.playback_speed, 0.01..=100.0)
                .text("Speed (×)")
                .logarithmic(true),
        );

        ui.label(format!("Time: {:.2} ms", playback_state.current_time / 1000.0));
        ui.label(format!("Window: {:.2} ms", playback_state.window_size / 1000.0));

        if let Some(fps) = diagnostics.get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                ui.label(format!("FPS: {:.1}", value));
            }
        }
    });

    // Edge Detection Controls
    egui::Window::new("Edge Detection").show(contexts.ctx_mut(), |ui| {
        ui.checkbox(&mut edge_params.show_gradient, "Show Edge Detection (Yellow)");

        ui.add(
            egui::Slider::new(&mut edge_params.threshold, 0.0..=10_000.0)
                .text("Edge Threshold"),
        );

        ui.label("Layer 0: Red/Blue raw events");
        ui.label("Layer 1: Yellow edge detection (Sobel STG)");
    });
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add src/mvp/render.rs
git commit -m "feat(mvp): implement UI system with playback and edge controls"
```

---

## Task 10: Wire Up MvpPlugin

**Files:**
- Modify: `src/mvp/mod.rs`
- Modify: `src/mvp/render.rs`

**Step 1: Complete MvpPlugin in mod.rs**

Replace `src/mvp/mod.rs` with:

```rust
pub mod gpu;
pub mod playback;
pub mod render;

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::render::render_graph::RenderGraph;

pub use gpu::{EventData, SurfaceImage, GradientImage, EdgeParams};
pub use playback::PlaybackState;

pub struct MvpPlugin;

impl Plugin for MvpPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<EventData>::default())
            .add_plugins(ExtractResourcePlugin::<SurfaceImage>::default())
            .add_plugins(ExtractResourcePlugin::<GradientImage>::default())
            .add_plugins(ExtractResourcePlugin::<PlaybackState>::default())
            .add_plugins(ExtractResourcePlugin::<EdgeParams>::default())
            .add_plugins(render::EventRenderPlugin)
            .init_resource::<SurfaceImage>()
            .init_resource::<GradientImage>()
            .init_resource::<PlaybackState>()
            .init_resource::<EdgeParams>();
    }
}
```

**Step 2: Add EventRenderPlugin to render.rs**

Append to `src/mvp/render.rs`:

```rust
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy_egui::EguiPrimaryContextPass;

pub struct EventRenderPlugin;

impl Plugin for EventRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<EventMaterial>::default())
            .add_plugins(EguiPlugin)
            .add_plugins(FrameTimeDiagnosticsPlugin)
            .add_systems(Startup, (load_data, setup_scene).chain())
            .add_systems(Update, (
                crate::mvp::playback::playback_system,
                update_material_params,
            ).chain())
            .add_systems(EguiPrimaryContextPass, ui_system);
    }

    fn finish(&self, app: &mut App) {
        use crate::mvp::gpu::*;
        use bevy::render::render_graph::RenderGraph;

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<EventComputePipeline>()
            .init_resource::<GradientPipeline>()
            .init_resource::<GpuEventBuffer>()
            .add_systems(Render, prepare_events.in_set(RenderSystems::Prepare))
            .add_systems(Render, queue_bind_group.in_set(RenderSystems::Queue))
            .add_systems(Render, prepare_gradient.in_set(RenderSystems::Queue));

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(EventLabel, EventAccumulationNode::default());
        render_graph.add_node(GradientLabel, GradientNode::default());
        render_graph.add_node_edge(EventLabel, GradientLabel);
        render_graph.add_node_edge(GradientLabel, bevy::render::graph::CameraDriverLabel);
    }
}
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add src/mvp/mod.rs src/mvp/render.rs
git commit -m "feat(mvp): wire up MvpPlugin with render graph"
```

---

## Task 11: Switch main.rs to MvpPlugin

**Files:**
- Modify: `src/main.rs`

**Step 1: Read current main.rs**

Understand current plugin structure

**Step 2: Replace CorePlugins with MVP**

Comment out old plugins and use MvpPlugin:

```rust
use bevy::prelude::*;
use ebc_rs::mvp::MvpPlugin;
use ebc_rs::EventFilePath;

fn main() {
    let data_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/fan/fan_const_rpm.dat".to_string());

    App::new()
        .insert_resource(EventFilePath(data_path))
        .add_plugins(DefaultPlugins)
        .add_plugins(MvpPlugin)
        .run();
}
```

**Step 3: Build and check for errors**

Run: `cargo build`
Expected: May have compilation errors - fix them iteratively

**Step 4: Fix any missing imports or type errors**

Iterate until `cargo build` succeeds

**Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat(mvp): switch main.rs to use MvpPlugin"
```

---

## Task 12: Test MVP - Layer 0 Only

**Files:**
- None (testing only)

**Step 1: Run the application**

Run: `cargo run -- --data-path data/fan/fan_const_rpm.dat`
Expected: Window opens, shows red/blue events

**Step 2: Test playback controls**

- Click play/pause - playback should start/stop
- Move time slider - visualization should update
- Adjust window size - event density should change
- Adjust speed - playback rate should change

**Step 3: Verify FPS**

Check FPS display in UI
Expected: 60+ FPS

**Step 4: Test edge detection toggle**

- Uncheck "Show Edge Detection" in UI
- Should only see raw red/blue events
- Check it again

**Step 5: Document results**

If issues found, note them for fixes. If working, proceed.

---

## Task 13: Test MVP - Layer 1 Edge Detection

**Files:**
- None (testing only)

**Step 1: Enable edge detection**

Check "Show Edge Detection (Yellow)" in UI
Expected: Yellow edges appear on fan blades

**Step 2: Adjust threshold**

Move "Edge Threshold" slider from 0 to 10,000
Expected: Edge detection sensitivity changes

**Step 3: Find optimal threshold**

Find value where edges are clear but not noisy
Expected: Around 1000-2000 should work well

**Step 4: Verify compositing**

Yellow should blend 50% with raw red/blue
Expected: See both layers together

**Step 5: Visual validation**

- Yellow edges align with fan blade boundaries
- No jitter or artifacts
- Smooth playback at 60 FPS

---

## Task 14: Final Cleanup and Documentation

**Files:**
- Create: `docs/plans/mvp-completion-notes.md`
- Modify: `README.md` (if needed)

**Step 1: Write completion notes**

Document:
- What works
- Known limitations
- Optimal threshold values found
- Performance metrics

**Step 2: Clean up debug logging**

Remove or reduce any excessive logging

**Step 3: Final build test**

Run: `cargo build --release`
Expected: Clean build with zero warnings

**Step 4: Commit cleanup**

```bash
git add .
git commit -m "docs(mvp): add completion notes and cleanup"
```

**Step 5: Celebrate!**

MVP is complete. Ready to add Layer 2 (CMAX) or improve Layer 1.

---

## Verification Checklist

- [ ] Layer 0 (raw events) renders correctly
- [ ] Layer 1 (edge detection) shows yellow edges
- [ ] Playback controls work (play/pause/speed/time)
- [ ] Window size adjustment works
- [ ] Edge threshold slider updates edges in real-time
- [ ] Toggle edge detection on/off works
- [ ] Performance is 60+ FPS
- [ ] No crashes or errors
- [ ] Clean build with no warnings

---

## Next Steps After MVP

1. **Improve Layer 1:**
   - Adaptive thresholding
   - Temporal edge filtering
   - Multi-scale Sobel

2. **Add Layer 2 (CMAX):**
   - Implement motion compensation
   - Contrast maximization
   - RPM estimation from angular velocity

3. **Deprecate old code:**
   - Delete `src/analysis.rs`, `src/gizmos.rs`
   - Delete old shader files
   - Remove unused dependencies

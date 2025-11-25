# Hyperparameter Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Separate pre-processing filters into dedicated shader, enable systematic hyperparameter optimization across all edge detectors.

**Architecture:** New PreprocessNode in render graph applies shared filters (dead pixel, density, temporal) to surface texture before any detector runs. Detectors read from FilteredSurfaceImage instead of raw SurfaceImage. Subprocess-based test runner enables parallel hyperparameter grid search.

**Tech Stack:** Bevy 0.17, wgpu compute shaders (WGSL), clap for CLI, serde/serde_json for config serialization, rayon for parallel subprocess orchestration.

---

## Task 1: Update GpuParams Structure

**Files:**
- Modify: `src/gpu/types.rs`
- Modify: `src/gpu/resources.rs`

**Step 1: Update GpuParams in types.rs**

Replace `GpuEdgeParams` with expanded `GpuParams`:

```rust
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuParams {
    // Pre-processing
    pub filter_dead_pixels: u32,
    pub filter_density: u32,
    pub filter_temporal: u32,
    pub min_density_count: u32,
    pub min_temporal_spread: f32,

    // Sobel
    pub sobel_threshold: f32,

    // Canny
    pub canny_low_threshold: f32,
    pub canny_high_threshold: f32,

    // LoG
    pub log_threshold: f32,

    // Post-processing
    pub filter_bidirectional: u32,
    pub bidirectional_ratio: f32,

    // Padding for 16-byte alignment
    pub _padding: f32,
}

// Keep GpuEvent unchanged
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuEvent {
    pub timestamp: u32,
    pub x: u32,
    pub y: u32,
    pub polarity: u32,
}
```

**Step 2: Update EdgeParams in resources.rs**

Update the `EdgeParams` struct with new fields:

```rust
#[derive(Resource, Clone)]
pub struct EdgeParams {
    // Pre-processing
    pub filter_dead_pixels: bool,
    pub filter_density: bool,
    pub filter_temporal: bool,
    pub min_density_count: u32,
    pub min_temporal_spread_us: f32,

    // Detector toggles (UI only)
    pub show_sobel: bool,
    pub show_canny: bool,
    pub show_log: bool,
    pub show_raw: bool,

    // Sobel
    pub sobel_threshold: f32,

    // Canny
    pub canny_low_threshold: f32,
    pub canny_high_threshold: f32,

    // LoG
    pub log_threshold: f32,

    // Post-processing
    pub filter_bidirectional: bool,
    pub bidirectional_ratio: f32,
}

impl Default for EdgeParams {
    fn default() -> Self {
        Self {
            filter_dead_pixels: true,
            filter_density: false,
            filter_temporal: false,
            min_density_count: 5,
            min_temporal_spread_us: 500.0,
            show_sobel: true,
            show_canny: false,
            show_log: false,
            show_raw: false,
            sobel_threshold: 1000.0,
            canny_low_threshold: 50.0,
            canny_high_threshold: 150.0,
            log_threshold: 10.0,
            filter_bidirectional: false,
            bidirectional_ratio: 0.3,
        }
    }
}
```

**Step 3: Build to verify no compile errors**

Run: `cargo build`
Expected: Compiles with warnings about unused fields (OK for now)

**Step 4: Commit**

```bash
git add src/gpu/types.rs src/gpu/resources.rs
git commit -m "refactor: expand GpuParams with configurable filter thresholds"
```

---

## Task 2: Add FilteredSurfaceImage Resource

**Files:**
- Modify: `src/gpu/resources.rs`
- Modify: `src/gpu/mod.rs`

**Step 1: Add FilteredSurfaceImage resource**

In `src/gpu/resources.rs`, add after `SurfaceImage`:

```rust
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct FilteredSurfaceImage {
    pub handle: Handle<Image>,
}
```

**Step 2: Export in mod.rs**

In `src/gpu/mod.rs`, add to exports:

```rust
pub use resources::{
    ActiveDetector, CannyImage, EdgeParams, EdgeReadbackBuffer,
    EventData, FilteredSurfaceImage, GpuEventBuffer, LogImage,
    SobelImage, SurfaceImage
};
```

**Step 3: Build to verify**

Run: `cargo build`
Expected: Compiles successfully

**Step 4: Commit**

```bash
git add src/gpu/resources.rs src/gpu/mod.rs
git commit -m "feat: add FilteredSurfaceImage resource for preprocess output"
```

---

## Task 3: Create Preprocess Shader

**Files:**
- Create: `assets/shaders/preprocess.wgsl`

**Step 1: Write preprocess shader**

```wgsl
@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var filtered_output: texture_storage_2d<r32uint, write>;

struct GpuParams {
    filter_dead_pixels: u32,
    filter_density: u32,
    filter_temporal: u32,
    min_density_count: u32,
    min_temporal_spread: f32,
    sobel_threshold: f32,
    canny_low_threshold: f32,
    canny_high_threshold: f32,
    log_threshold: f32,
    filter_bidirectional: u32,
    bidirectional_ratio: f32,
    _padding: f32,
}

@group(0) @binding(2) var<uniform> params: GpuParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(surface_texture);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Load center pixel
    let center_packed = textureLoad(surface_texture, coords, 0).r;
    let center_timestamp = center_packed >> 1u;

    // Filter 1: Dead pixel check
    if (params.filter_dead_pixels == 1u && center_timestamp < 1u) {
        textureStore(filtered_output, coords, vec4<u32>(0u));
        return;
    }

    // Skip border for neighborhood filters
    if (coords.x < 1 || coords.y < 1 || coords.x >= i32(dims.x) - 1 || coords.y >= i32(dims.y) - 1) {
        textureStore(filtered_output, coords, vec4<u32>(center_packed));
        return;
    }

    // Load 3x3 neighborhood timestamps
    var timestamps: array<f32, 9>;
    var idx = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let pos = coords + vec2<i32>(dx, dy);
            let packed = textureLoad(surface_texture, pos, 0).r;
            timestamps[idx] = f32(packed >> 1u);
            idx++;
        }
    }

    // Filter 2: Event density check
    if (params.filter_density == 1u) {
        var active_count = 0u;
        for (var i = 0u; i < 9u; i++) {
            if (timestamps[i] > 1.0) {
                active_count++;
            }
        }
        if (active_count < params.min_density_count) {
            textureStore(filtered_output, coords, vec4<u32>(0u));
            return;
        }
    }

    // Filter 3: Temporal variance check
    if (params.filter_temporal == 1u) {
        var min_ts = timestamps[0];
        var max_ts = timestamps[0];
        for (var i = 1u; i < 9u; i++) {
            if (timestamps[i] > 0.0) {
                min_ts = min(min_ts, timestamps[i]);
                max_ts = max(max_ts, timestamps[i]);
            }
        }
        let ts_range = max_ts - min_ts;
        if (ts_range < params.min_temporal_spread) {
            textureStore(filtered_output, coords, vec4<u32>(0u));
            return;
        }
    }

    // Passed all filters - copy original value
    textureStore(filtered_output, coords, vec4<u32>(center_packed));
}
```

**Step 2: Verify shader syntax**

Run: `cargo build` (shader loaded at runtime, but good to check project still builds)
Expected: Compiles

**Step 3: Commit**

```bash
git add assets/shaders/preprocess.wgsl
git commit -m "feat: add preprocess.wgsl shader with shared filters"
```

---

## Task 4: Create PreprocessPipeline and PreprocessNode

**Files:**
- Create: `src/gpu/preprocess.rs`
- Modify: `src/gpu/mod.rs`

**Step 1: Create preprocess.rs**

```rust
use bevy::prelude::*;
use bevy::render::{
    render_graph::{Node, RenderLabel},
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use super::resources::{FilteredSurfaceImage, SurfaceImage, EdgeParams};
use super::types::GpuParams;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct PreprocessLabel;

#[derive(Resource)]
pub struct PreprocessPipeline {
    pub pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
}

impl FromWorld for PreprocessPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            Some("Preprocess Bind Group Layout"),
            &[
                // Surface texture input
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
                // Filtered output
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Uint,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Params uniform
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuParams::min_size()),
                    },
                    count: None,
                },
            ],
        );

        let shader = world.load_asset("shaders/preprocess.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Preprocess Pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

#[derive(Resource, Default)]
pub struct PreprocessBindGroup {
    pub bind_group: Option<BindGroup>,
    pub params_buffer: Option<Buffer>,
}

pub fn prepare_preprocess(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<PreprocessPipeline>,
    mut bind_group_res: ResMut<PreprocessBindGroup>,
    surface_image: Res<SurfaceImage>,
    filtered_image: Res<FilteredSurfaceImage>,
    edge_params: Res<EdgeParams>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    let Some(surface_gpu) = gpu_images.get(&surface_image.handle) else {
        return;
    };
    let Some(filtered_gpu) = gpu_images.get(&filtered_image.handle) else {
        return;
    };

    // Create/update params buffer
    let gpu_params = GpuParams {
        filter_dead_pixels: edge_params.filter_dead_pixels as u32,
        filter_density: edge_params.filter_density as u32,
        filter_temporal: edge_params.filter_temporal as u32,
        min_density_count: edge_params.min_density_count,
        min_temporal_spread: edge_params.min_temporal_spread_us,
        sobel_threshold: edge_params.sobel_threshold,
        canny_low_threshold: edge_params.canny_low_threshold,
        canny_high_threshold: edge_params.canny_high_threshold,
        log_threshold: edge_params.log_threshold,
        filter_bidirectional: edge_params.filter_bidirectional as u32,
        bidirectional_ratio: edge_params.bidirectional_ratio,
        _padding: 0.0,
    };

    let params_buffer = bind_group_res.params_buffer.get_or_insert_with(|| {
        render_device.create_buffer(&BufferDescriptor {
            label: Some("Preprocess Params Buffer"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });

    render_queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&gpu_params));

    let bind_group = render_device.create_bind_group(
        Some("Preprocess Bind Group"),
        &pipeline.bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&surface_gpu.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&filtered_gpu.texture_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    );

    bind_group_res.bind_group = Some(bind_group);
}

#[derive(Default)]
pub struct PreprocessNode;

impl Node for PreprocessNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_res = world.resource::<PreprocessPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let bind_group_res = world.resource::<PreprocessBindGroup>();

        let Some(bind_group) = &bind_group_res.bind_group else {
            return Ok(());
        };

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.pipeline) else {
            return Ok(());
        };

        let filtered_image = world.resource::<FilteredSurfaceImage>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let Some(filtered_gpu) = gpu_images.get(&filtered_image.handle) else {
            return Ok(());
        };

        let width = filtered_gpu.texture.width();
        let height = filtered_gpu.texture.height();

        let mut pass = render_context.command_encoder().begin_compute_pass(
            &ComputePassDescriptor {
                label: Some("Preprocess Pass"),
                timestamp_writes: None,
            },
        );

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);

        Ok(())
    }
}
```

**Step 2: Add to mod.rs**

In `src/gpu/mod.rs`:

```rust
pub mod preprocess;

pub use preprocess::{
    PreprocessLabel, PreprocessNode, PreprocessPipeline, PreprocessBindGroup,
    prepare_preprocess
};
```

**Step 3: Build to verify**

Run: `cargo build`
Expected: May have errors about GpuParams::min_size() - fix by adding ShaderType derive

**Step 4: Fix GpuParams if needed**

If min_size() error, update `src/gpu/types.rs`:

```rust
use bevy::render::render_resource::ShaderType;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct GpuParams {
    // ... fields unchanged
}
```

Also add to Cargo.toml if needed: `bevy = { version = "0.17", features = ["shader_format_glsl"] }`

**Step 5: Build again**

Run: `cargo build`
Expected: Compiles successfully

**Step 6: Commit**

```bash
git add src/gpu/preprocess.rs src/gpu/mod.rs src/gpu/types.rs
git commit -m "feat: add PreprocessPipeline and PreprocessNode"
```

---

## Task 5: Integrate Preprocess into Render Graph

**Files:**
- Modify: `src/edge_detection.rs`
- Modify: `src/event_renderer.rs`

**Step 1: Update edge_detection.rs to register preprocess**

Add preprocess to the render graph in `src/edge_detection.rs`:

```rust
use crate::gpu::*;

// In EdgeDetectionPlugin::build(), add:
.add_plugins(ExtractResourcePlugin::<FilteredSurfaceImage>::default())
.init_resource::<FilteredSurfaceImage>()

// In EdgeDetectionPlugin::finish(), add to render_app:
.init_resource::<PreprocessPipeline>()
.init_resource::<PreprocessBindGroup>()
.add_systems(Render, prepare_preprocess.in_set(RenderSystems::Queue))

// Update render graph edges:
render_graph.add_node(PreprocessLabel, PreprocessNode::default());
render_graph.add_node_edge(EventLabel, PreprocessLabel);
render_graph.add_node_edge(PreprocessLabel, SobelLabel);
// Remove: render_graph.add_node_edge(EventLabel, SobelLabel);
```

**Step 2: Create FilteredSurface texture in event_renderer.rs**

In `setup_edge_textures` function, after creating sobel_image, add:

```rust
// Filtered surface texture (same format as surface)
let mut filtered_image = Image::new_fill(
    size,
    TextureDimension::D2,
    &[0, 0, 0, 0],
    TextureFormat::R32Uint,
    RenderAssetUsages::RENDER_WORLD,
);
filtered_image.texture_descriptor.usage =
    TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
filtered_surface.handle = images.add(filtered_image);
```

Update function signature to include `mut filtered_surface: ResMut<FilteredSurfaceImage>`.

**Step 3: Build and test**

Run: `cargo build && cargo run -- data/fan/fan_const_rpm.dat`
Expected: App runs, preprocess node executes (may need debugging)

**Step 4: Commit**

```bash
git add src/edge_detection.rs src/event_renderer.rs
git commit -m "feat: integrate PreprocessNode into render graph"
```

---

## Task 6: Update Sobel Shader to Use Filtered Input

**Files:**
- Modify: `assets/shaders/sobel.wgsl`
- Modify: `src/gpu/sobel.rs`

**Step 1: Simplify sobel.wgsl**

Remove all filter logic, read from filtered texture:

```wgsl
@group(0) @binding(0) var filtered_texture: texture_2d<u32>;
@group(0) @binding(1) var gradient_output: texture_storage_2d<r32float, write>;

struct GpuParams {
    filter_dead_pixels: u32,
    filter_density: u32,
    filter_temporal: u32,
    min_density_count: u32,
    min_temporal_spread: f32,
    sobel_threshold: f32,
    canny_low_threshold: f32,
    canny_high_threshold: f32,
    log_threshold: f32,
    filter_bidirectional: u32,
    bidirectional_ratio: f32,
    _padding: f32,
}

@group(0) @binding(2) var<uniform> params: GpuParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(filtered_texture);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
        return;
    }

    // Skip border pixels
    if (coords.x < 1 || coords.y < 1 || coords.x >= i32(dims.x) - 1 || coords.y >= i32(dims.y) - 1) {
        textureStore(gradient_output, coords, vec4<f32>(0.0));
        return;
    }

    // Load 3x3 neighborhood timestamps from FILTERED texture
    var timestamps: array<f32, 9>;
    var idx = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let pos = coords + vec2<i32>(dx, dy);
            let packed = textureLoad(filtered_texture, pos, 0).r;
            timestamps[idx] = f32(packed >> 1u);
            idx++;
        }
    }

    // Sobel kernels
    let gx = -timestamps[0] + timestamps[2]
             - 2.0 * timestamps[3] + 2.0 * timestamps[5]
             - timestamps[6] + timestamps[8];

    let gy = -timestamps[0] - 2.0 * timestamps[1] - timestamps[2]
             + timestamps[6] + 2.0 * timestamps[7] + timestamps[8];

    // Optional: Bidirectional gradient check (post-filter)
    if (params.filter_bidirectional == 1u) {
        let gx_abs = abs(gx);
        let gy_abs = abs(gy);
        let min_directional = params.sobel_threshold * params.bidirectional_ratio;
        if (gx_abs < min_directional || gy_abs < min_directional) {
            textureStore(gradient_output, coords, vec4<f32>(0.0));
            return;
        }
    }

    let magnitude = sqrt(gx * gx + gy * gy);
    let edge_value = select(0.0, 1.0, magnitude > params.sobel_threshold);
    textureStore(gradient_output, coords, vec4<f32>(edge_value));
}
```

**Step 2: Update sobel.rs to use FilteredSurfaceImage**

Change the bind group to use `FilteredSurfaceImage` instead of `SurfaceImage`:

```rust
// In prepare_sobel, change:
// surface_image: Res<SurfaceImage>,
// to:
filtered_image: Res<FilteredSurfaceImage>,

// And update the texture binding accordingly
```

**Step 3: Build and test**

Run: `cargo build && cargo run -- data/fan/fan_const_rpm.dat`
Expected: Sobel still works, now using filtered input

**Step 4: Commit**

```bash
git add assets/shaders/sobel.wgsl src/gpu/sobel.rs
git commit -m "refactor: sobel uses filtered texture, remove inline filters"
```

---

## Task 7: Update Canny and LoG Shaders

**Files:**
- Modify: `assets/shaders/canny.wgsl`
- Modify: `assets/shaders/log.wgsl`
- Modify: `src/gpu/canny.rs`
- Modify: `src/gpu/log.rs`

**Step 1: Update canny.wgsl input**

Change first line from `surface_texture` to `filtered_texture`:

```wgsl
@group(0) @binding(0) var filtered_texture: texture_2d<u32>;
```

Update all `textureLoad(surface_texture, ...)` to `textureLoad(filtered_texture, ...)`.

**Step 2: Update log.wgsl input**

Same change - `surface_texture` to `filtered_texture`.

**Step 3: Update canny.rs and log.rs**

Change bind group setup to use `FilteredSurfaceImage` instead of `SurfaceImage`.

**Step 4: Build and test**

Run: `cargo build && cargo run -- data/fan/fan_const_rpm.dat`
Expected: All three detectors work with filtered input

**Step 5: Run detector tests**

Run: `cargo test test_sobel_real -- --nocapture`
Run: `cargo test test_canny_real -- --nocapture`
Run: `cargo test test_log_real -- --nocapture`
Expected: All pass with edge detection working

**Step 6: Commit**

```bash
git add assets/shaders/canny.wgsl assets/shaders/log.wgsl src/gpu/canny.rs src/gpu/log.rs
git commit -m "refactor: canny and log use filtered texture input"
```

---

## Task 8: Update Detector Comparison Tests

**Files:**
- Modify: `tests/detector_comparison.rs`

**Step 1: Update test plugin to include FilteredSurfaceImage**

In `DetectorTestPlugin::build()`, add:

```rust
.add_plugins(ExtractResourcePlugin::<FilteredSurfaceImage>::default())
.init_resource::<FilteredSurfaceImage>()
```

In `DetectorTestPlugin::finish()`, add:

```rust
.init_resource::<PreprocessPipeline>()
.init_resource::<PreprocessBindGroup>()
.add_systems(Render, prepare_preprocess.in_set(RenderSystems::Queue))
```

Update render graph:
```rust
render_graph.add_node(PreprocessLabel, PreprocessNode::default());
render_graph.add_node_edge(EventLabel, PreprocessLabel);
render_graph.add_node_edge(PreprocessLabel, SobelLabel);
```

In `setup_test_textures`, add filtered surface texture creation.

**Step 2: Run tests**

Run: `cargo test test_sobel_real -- --nocapture`
Expected: PASS with similar or better results (filters now shared)

**Step 3: Commit**

```bash
git add tests/detector_comparison.rs
git commit -m "test: update detector tests for preprocess pipeline"
```

---

## Task 9: Create Hypertest CLI Binary

**Files:**
- Create: `src/bin/hypertest.rs`
- Modify: `Cargo.toml`

**Step 1: Add clap dependency**

In `Cargo.toml`:

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**Step 2: Create hypertest.rs**

```rust
//! Single-configuration hyperparameter test runner
//!
//! Usage: cargo run --release --bin hypertest -- --data fan.dat --detector sobel

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "hypertest")]
#[command(about = "Run edge detection with specific hyperparameters")]
struct Args {
    #[arg(long)]
    data: PathBuf,

    #[arg(long, default_value = "sobel")]
    detector: String,

    #[arg(long, default_value = "100.0")]
    window_size: f32,

    #[arg(long, default_value = "1000.0")]
    threshold: f32,

    #[arg(long, default_value = "50.0")]
    canny_low: f32,

    #[arg(long, default_value = "150.0")]
    canny_high: f32,

    #[arg(long)]
    filter_dead_pixels: bool,

    #[arg(long)]
    filter_density: bool,

    #[arg(long)]
    filter_temporal: bool,

    #[arg(long, default_value = "5")]
    min_density: u32,

    #[arg(long, default_value = "500.0")]
    min_temporal: f32,

    #[arg(long)]
    filter_bidirectional: bool,

    #[arg(long, default_value = "0.3")]
    bidirectional_ratio: f32,

    #[arg(long, default_value = "50")]
    frames: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HyperConfig {
    pub detector: String,
    pub window_size_us: f32,
    pub threshold: f32,
    pub canny_low: f32,
    pub canny_high: f32,
    pub filter_dead_pixels: bool,
    pub filter_density: bool,
    pub filter_temporal: bool,
    pub min_density_count: u32,
    pub min_temporal_spread_us: f32,
    pub filter_bidirectional: bool,
    pub bidirectional_ratio: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HyperResult {
    pub config: HyperConfig,
    pub avg_edge_count: f32,
    pub edge_density: f32,
    pub centroid_stability: f32,
    pub radius_stability: f32,
    pub circle_fit_error: f32,
    pub inlier_ratio: f32,
    pub detected_blade_count: f32,
    pub frames_processed: usize,
}

fn main() {
    let args = Args::parse();

    let config = HyperConfig {
        detector: args.detector.clone(),
        window_size_us: args.window_size,
        threshold: args.threshold,
        canny_low: args.canny_low,
        canny_high: args.canny_high,
        filter_dead_pixels: args.filter_dead_pixels,
        filter_density: args.filter_density,
        filter_temporal: args.filter_temporal,
        min_density_count: args.min_density,
        min_temporal_spread_us: args.min_temporal,
        filter_bidirectional: args.filter_bidirectional,
        bidirectional_ratio: args.bidirectional_ratio,
    };

    // For now, output placeholder - will integrate with test framework
    let result = HyperResult {
        config,
        avg_edge_count: 0.0,
        edge_density: 0.0,
        centroid_stability: 0.0,
        radius_stability: 0.0,
        circle_fit_error: 0.0,
        inlier_ratio: 0.0,
        detected_blade_count: 0.0,
        frames_processed: 0,
    };

    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
```

**Step 3: Build**

Run: `cargo build --bin hypertest`
Expected: Compiles

**Step 4: Test CLI**

Run: `cargo run --bin hypertest -- --data data/fan/fan_const_rpm.dat --detector sobel --filter-dead-pixels`
Expected: Outputs JSON with config

**Step 5: Commit**

```bash
git add src/bin/hypertest.rs Cargo.toml
git commit -m "feat: add hypertest CLI binary for single-config testing"
```

---

## Task 10: Integrate Hypertest with Bevy Test Runner

**Files:**
- Modify: `src/bin/hypertest.rs`
- Create: `src/hyperparams.rs`

**Step 1: Create shared hyperparams module**

Create `src/hyperparams.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HyperConfig {
    pub detector: String,
    pub window_size_us: f32,
    pub threshold: f32,
    pub canny_low: f32,
    pub canny_high: f32,
    pub filter_dead_pixels: bool,
    pub filter_density: bool,
    pub filter_temporal: bool,
    pub min_density_count: u32,
    pub min_temporal_spread_us: f32,
    pub filter_bidirectional: bool,
    pub bidirectional_ratio: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HyperResult {
    pub config: HyperConfig,
    pub avg_edge_count: f32,
    pub edge_density: f32,
    pub centroid_stability: f32,
    pub radius_stability: f32,
    pub circle_fit_error: f32,
    pub inlier_ratio: f32,
    pub detected_blade_count: f32,
    pub frames_processed: usize,
}

impl HyperConfig {
    pub fn to_edge_params(&self) -> crate::gpu::EdgeParams {
        crate::gpu::EdgeParams {
            filter_dead_pixels: self.filter_dead_pixels,
            filter_density: self.filter_density,
            filter_temporal: self.filter_temporal,
            min_density_count: self.min_density_count,
            min_temporal_spread_us: self.min_temporal_spread_us,
            sobel_threshold: self.threshold,
            canny_low_threshold: self.canny_low,
            canny_high_threshold: self.canny_high,
            log_threshold: self.threshold,
            filter_bidirectional: self.filter_bidirectional,
            bidirectional_ratio: self.bidirectional_ratio,
            ..Default::default()
        }
    }
}
```

**Step 2: Export in lib.rs**

Add to `src/lib.rs`:

```rust
pub mod hyperparams;
pub use hyperparams::{HyperConfig, HyperResult};
```

**Step 3: Update hypertest.rs to run actual tests**

The hypertest binary should run a headless Bevy app similar to detector_comparison tests, using the config to set EdgeParams. (Implementation mirrors detector_comparison.rs but parameterized by CLI args.)

**Step 4: Build and test**

Run: `cargo run --release --bin hypertest -- --data data/fan/fan_const_rpm.dat --detector sobel --filter-dead-pixels --frames 30`
Expected: Outputs JSON with actual metrics

**Step 5: Commit**

```bash
git add src/hyperparams.rs src/lib.rs src/bin/hypertest.rs
git commit -m "feat: hypertest runs actual Bevy test with config"
```

---

## Task 11: Create Hypersearch Grid Search Binary

**Files:**
- Create: `src/bin/hypersearch.rs`

**Step 1: Create hypersearch.rs**

```rust
//! Grid search orchestrator - spawns hypertest subprocesses in parallel

use clap::Parser;
use rayon::prelude::*;
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(name = "hypersearch")]
struct Args {
    #[arg(long)]
    data: PathBuf,

    #[arg(long)]
    output: PathBuf,

    #[arg(long, value_delimiter = ',')]
    window_sizes: Vec<f32>,

    #[arg(long, value_delimiter = ',')]
    thresholds: Vec<f32>,

    #[arg(long, default_value = "sobel,canny,log")]
    detectors: String,

    #[arg(long, default_value = "30")]
    frames: usize,

    #[arg(long)]
    all_filter_combos: bool,
}

fn main() {
    let args = Args::parse();

    let detectors: Vec<&str> = args.detectors.split(',').collect();
    let filter_combos = if args.all_filter_combos {
        vec![
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (true, false, true),
            (true, true, true),
        ]
    } else {
        vec![(true, false, false)] // Default: only dead_pixels
    };

    let mut configs = Vec::new();
    for detector in &detectors {
        for &window in &args.window_sizes {
            for &threshold in &args.thresholds {
                for &(dead, density, temporal) in &filter_combos {
                    configs.push((
                        detector.to_string(),
                        window,
                        threshold,
                        dead,
                        density,
                        temporal,
                    ));
                }
            }
        }
    }

    println!("Running {} configurations...", configs.len());

    let results: Vec<String> = configs
        .par_iter()
        .map(|(detector, window, threshold, dead, density, temporal)| {
            let mut cmd = Command::new("cargo");
            cmd.args([
                "run", "--release", "--bin", "hypertest", "--",
                "--data", args.data.to_str().unwrap(),
                "--detector", detector,
                "--window-size", &window.to_string(),
                "--threshold", &threshold.to_string(),
                "--frames", &args.frames.to_string(),
            ]);

            if *dead { cmd.arg("--filter-dead-pixels"); }
            if *density { cmd.arg("--filter-density"); }
            if *temporal { cmd.arg("--filter-temporal"); }

            let output = cmd.output().expect("Failed to run hypertest");
            String::from_utf8_lossy(&output.stdout).to_string()
        })
        .collect();

    // Parse results and write CSV
    let mut csv_lines = vec![
        "detector,window_size,threshold,dead_pixels,density,temporal,edge_count,centroid_stability,inlier_ratio".to_string()
    ];

    for json_str in &results {
        if let Ok(result) = serde_json::from_str::<ebc_rs::HyperResult>(json_str.trim()) {
            csv_lines.push(format!(
                "{},{},{},{},{},{},{},{},{}",
                result.config.detector,
                result.config.window_size_us,
                result.config.threshold,
                result.config.filter_dead_pixels,
                result.config.filter_density,
                result.config.filter_temporal,
                result.avg_edge_count,
                result.centroid_stability,
                result.inlier_ratio,
            ));
        }
    }

    std::fs::write(&args.output, csv_lines.join("\n")).expect("Failed to write CSV");
    println!("Results written to {}", args.output.display());
}
```

**Step 2: Add rayon dependency**

In `Cargo.toml`:

```toml
rayon = "1.10"
```

**Step 3: Build**

Run: `cargo build --release --bin hypersearch`
Expected: Compiles

**Step 4: Test with small grid**

Run: `cargo run --release --bin hypersearch -- --data data/fan/fan_const_rpm.dat --output results/test.csv --window-sizes 100,500 --thresholds 500,1000 --detectors sobel --frames 10`
Expected: Creates CSV with 4 rows (2 windows × 2 thresholds)

**Step 5: Commit**

```bash
git add src/bin/hypersearch.rs Cargo.toml
git commit -m "feat: add hypersearch grid search orchestrator"
```

---

## Task 12: Final Integration Test

**Files:**
- None (testing only)

**Step 1: Run full coarse grid search**

```bash
mkdir -p results
cargo run --release --bin hypersearch -- \
    --data data/fan/fan_const_rpm.dat \
    --output results/coarse_search.csv \
    --window-sizes 50,100,200,500,1000 \
    --thresholds 500,1000,2000 \
    --detectors sobel,canny,log \
    --frames 30 \
    --all-filter-combos
```

Expected: Creates CSV with results for all configurations

**Step 2: Analyze results**

Check which configuration has best centroid_stability with reasonable edge_count.

**Step 3: Run fine search on best region**

Based on coarse results, run fine search around best parameters.

**Step 4: Document best configurations**

Create `results/best_configs.md` documenting optimal parameters per detector.

**Step 5: Final commit**

```bash
git add results/
git commit -m "feat: complete hyperparameter optimization infrastructure"
```

---

## Summary

| Task | Description | Estimated Time |
|------|-------------|----------------|
| 1 | Update GpuParams structure | 10 min |
| 2 | Add FilteredSurfaceImage | 5 min |
| 3 | Create preprocess shader | 15 min |
| 4 | Create PreprocessPipeline/Node | 20 min |
| 5 | Integrate into render graph | 15 min |
| 6 | Update Sobel shader | 10 min |
| 7 | Update Canny/LoG shaders | 10 min |
| 8 | Update detector tests | 10 min |
| 9 | Create hypertest CLI | 15 min |
| 10 | Integrate with test runner | 20 min |
| 11 | Create hypersearch binary | 15 min |
| 12 | Final integration test | 15 min |

**Total: ~2.5 hours**

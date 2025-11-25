# Ground Truth Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a GPU-based ground truth rendering pipeline that analytically computes exact blade positions per frame, enabling pixel-perfect comparison with edge detectors.

**Architecture:** A new compute shader (`ground_truth.wgsl`) runs in parallel with existing edge detectors, outputting a two-channel texture (R=edge, G=interior). Parameters come from a JSON sidecar file loaded alongside synthetic .dat files.

**Tech Stack:** Bevy 0.17, WGSL compute shaders, serde_json for config parsing

---

## Task 1: Update Synthesis to Output Extended JSON

**Files:**
- Modify: `src/synthesis.rs`

**Step 1: Update the JSON output format**

Change the JSON output from an array of frame entries to an object with `params` and `frames`:

```rust
// In generate_fan_data(), replace the ground truth writing section (lines 146-163)
// with this new format:

    // Blade geometry constants used in generation
    let r_min = 50.0_f32;
    let sweep_k = 0.5_f32;
    let width_root_rad = 0.5_f32;
    let width_tip_rad = 0.3_f32;

    // Write ground truth JSON with params header
    writeln!(truth_file, "{{")?;
    writeln!(truth_file, "  \"params\": {{")?;
    writeln!(truth_file, "    \"center_x\": {:.1},", center_x)?;
    writeln!(truth_file, "    \"center_y\": {:.1},", center_y)?;
    writeln!(truth_file, "    \"radius_min\": {:.1},", r_min)?;
    writeln!(truth_file, "    \"radius_max\": {:.1},", radius)?;
    writeln!(truth_file, "    \"blade_count\": {},", blade_count)?;
    writeln!(truth_file, "    \"rpm\": {:.1},", rpm)?;
    writeln!(truth_file, "    \"sweep_k\": {:.2},", sweep_k)?;
    writeln!(truth_file, "    \"width_root_rad\": {:.2},", width_root_rad)?;
    writeln!(truth_file, "    \"width_tip_rad\": {:.2},", width_tip_rad)?;
    writeln!(truth_file, "    \"edge_thickness_px\": 2.0")?;
    writeln!(truth_file, "  }},")?;
    writeln!(truth_file, "  \"frames\": [")?;
    writeln!(truth_file, "{}", truth_entries.join(",\n"))?;
    writeln!(truth_file, "  ]")?;
    writeln!(truth_file, "}}")?;
```

**Step 2: Regenerate synthetic data**

Run: `cargo run --bin generate_synthetic`

**Step 3: Verify new JSON format**

Run: `head -20 data/synthetic/fan_test_truth.json`

Expected output:
```json
{
  "params": {
    "center_x": 640.0,
    "center_y": 360.0,
    "radius_min": 50.0,
    ...
  },
  "frames": [
    ...
  ]
}
```

**Step 4: Commit**

```bash
git add src/synthesis.rs
git commit -m "feat(synthesis): output extended JSON with blade geometry params"
```

---

## Task 2: Create GroundTruthConfig Resource and JSON Loader

**Files:**
- Create: `src/ground_truth.rs`
- Modify: `src/lib.rs`

**Step 1: Create the ground truth module**

```rust
// src/ground_truth.rs
//! Ground truth configuration for synthetic fan data validation.

use bevy::prelude::*;
use serde::Deserialize;
use std::f32::consts::PI;
use std::path::Path;

/// Ground truth fan parameters loaded from JSON sidecar file.
#[derive(Resource, Deserialize, Debug, Clone, Default)]
pub struct GroundTruthConfig {
    /// Whether ground truth rendering is enabled
    #[serde(default)]
    pub enabled: bool,
    /// Fan center X coordinate (pixels)
    #[serde(default = "default_center_x")]
    pub center_x: f32,
    /// Fan center Y coordinate (pixels)
    #[serde(default = "default_center_y")]
    pub center_y: f32,
    /// Minimum radius (blade root, pixels)
    #[serde(default = "default_radius_min")]
    pub radius_min: f32,
    /// Maximum radius (blade tip, pixels)
    #[serde(default = "default_radius_max")]
    pub radius_max: f32,
    /// Number of blades
    #[serde(default = "default_blade_count")]
    pub blade_count: u32,
    /// Rotations per minute
    #[serde(default = "default_rpm")]
    pub rpm: f32,
    /// Logarithmic spiral curvature parameter
    #[serde(default = "default_sweep_k")]
    pub sweep_k: f32,
    /// Blade angular width at root (radians)
    #[serde(default = "default_width_root")]
    pub width_root_rad: f32,
    /// Blade angular width at tip (radians)
    #[serde(default = "default_width_tip")]
    pub width_tip_rad: f32,
    /// Edge detection thickness (pixels)
    #[serde(default = "default_edge_thickness")]
    pub edge_thickness_px: f32,
}

fn default_center_x() -> f32 { 640.0 }
fn default_center_y() -> f32 { 360.0 }
fn default_radius_min() -> f32 { 50.0 }
fn default_radius_max() -> f32 { 200.0 }
fn default_blade_count() -> u32 { 3 }
fn default_rpm() -> f32 { 1200.0 }
fn default_sweep_k() -> f32 { 0.5 }
fn default_width_root() -> f32 { 0.5 }
fn default_width_tip() -> f32 { 0.3 }
fn default_edge_thickness() -> f32 { 2.0 }

impl GroundTruthConfig {
    /// Angular velocity in radians per second
    pub fn angular_velocity(&self) -> f32 {
        self.rpm * 2.0 * PI / 60.0
    }

    /// Try to load ground truth config from JSON sidecar file.
    /// Returns None if file doesn't exist or isn't valid ground truth JSON.
    pub fn load_from_sidecar(dat_path: &Path) -> Option<Self> {
        // Try _truth.json suffix first (e.g., fan_test.dat -> fan_test_truth.json)
        let truth_path = dat_path.with_extension("").to_string_lossy().to_string() + "_truth.json";
        let truth_path = Path::new(&truth_path);

        if !truth_path.exists() {
            return None;
        }

        let contents = std::fs::read_to_string(truth_path).ok()?;

        // Parse JSON - expect {"params": {...}, "frames": [...]}
        let json: serde_json::Value = serde_json::from_str(&contents).ok()?;
        let params = json.get("params")?;

        let mut config: GroundTruthConfig = serde_json::from_value(params.clone()).ok()?;
        config.enabled = true;

        Some(config)
    }
}

/// JSON structure for the sidecar file
#[derive(Deserialize)]
struct GroundTruthJson {
    params: GroundTruthConfig,
    #[allow(dead_code)]
    frames: Vec<serde_json::Value>,
}
```

**Step 2: Export from lib.rs**

Add to `src/lib.rs`:

```rust
pub mod ground_truth;
pub use ground_truth::GroundTruthConfig;
```

**Step 3: Build and verify**

Run: `cargo build`

Expected: Compiles without errors

**Step 4: Commit**

```bash
git add src/ground_truth.rs src/lib.rs
git commit -m "feat: add GroundTruthConfig resource with JSON sidecar loader"
```

---

## Task 3: Add GroundTruthImage Resource

**Files:**
- Modify: `src/gpu/resources.rs`
- Modify: `src/gpu/mod.rs`

**Step 1: Add GroundTruthImage resource**

Add to `src/gpu/resources.rs` after LogImage:

```rust
// Handle to ground truth texture (validation output)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct GroundTruthImage {
    pub handle: Handle<Image>,
}
```

**Step 2: Export from mod.rs**

Update the resources export line in `src/gpu/mod.rs`:

```rust
pub use resources::{ActiveDetector, CannyImage, EdgeParams, EdgeReadbackBuffer, EventData, FilteredSurfaceImage, GpuEventBuffer, GroundTruthImage, LogImage, SobelImage, SurfaceImage};
```

**Step 3: Build and verify**

Run: `cargo build`

**Step 4: Commit**

```bash
git add src/gpu/resources.rs src/gpu/mod.rs
git commit -m "feat(gpu): add GroundTruthImage resource"
```

---

## Task 4: Create Ground Truth Compute Shader

**Files:**
- Create: `assets/shaders/ground_truth.wgsl`

**Step 1: Write the compute shader**

```wgsl
// Ground truth blade geometry shader
// Analytically computes exact blade positions for synthetic data validation

struct Params {
    center_x: f32,
    center_y: f32,
    r_min: f32,
    r_max: f32,
    blade_count: u32,
    angular_velocity: f32,
    current_time: f32,
    sweep_k: f32,
    width_root: f32,
    width_tip: f32,
    edge_thickness: f32,
    _padding: f32,
}

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// Wrap angle to [-PI, PI]
fn wrap_angle(angle: f32) -> f32 {
    var a = angle;
    while (a > PI) { a -= TWO_PI; }
    while (a < -PI) { a += TWO_PI; }
    return a;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    // Bounds check (1280x720)
    if (x >= 1280 || y >= 720) {
        return;
    }

    let coords = vec2<i32>(x, y);

    // Convert to polar coordinates relative to fan center
    let dx = f32(x) - params.center_x;
    let dy = f32(y) - params.center_y;
    let r = sqrt(dx * dx + dy * dy);
    let theta = atan2(dy, dx);

    // Default: background (black)
    var output = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    // Skip pixels outside fan radius
    if (r < params.r_min || r > params.r_max) {
        textureStore(output_texture, coords, output);
        return;
    }

    // Calculate current base rotation angle
    let base_angle = params.angular_velocity * params.current_time;

    // Blade width varies with radius (wider at root, narrower at tip)
    let r_norm = (r - params.r_min) / (params.r_max - params.r_min);
    let half_width = (params.width_root + (params.width_tip - params.width_root) * r_norm) * 0.5;

    // Edge thickness in angular units at this radius
    let edge_angular_thickness = params.edge_thickness / r;

    // Check each blade
    for (var blade = 0u; blade < params.blade_count; blade++) {
        // Calculate this blade's center angle
        let blade_spacing = TWO_PI / f32(params.blade_count);
        let blade_base_angle = base_angle + f32(blade) * blade_spacing;

        // Logarithmic spiral: blade curves as radius increases
        let sweep_angle = params.sweep_k * log(r / params.r_min);
        let blade_center = blade_base_angle + sweep_angle;

        // Check if pixel angle is within blade
        let angle_diff = wrap_angle(theta - blade_center);

        if (abs(angle_diff) < half_width) {
            // Interior pixel - G channel
            output.g = 1.0;

            // Edge pixel - within threshold of boundary
            // Check both leading and trailing edges
            let dist_to_edge = abs(abs(angle_diff) - half_width);
            if (dist_to_edge < edge_angular_thickness) {
                // Edge pixel - R channel
                output.r = 1.0;
            }
        }
    }

    textureStore(output_texture, coords, output);
}
```

**Step 2: Verify shader syntax**

Run: `cargo build` (Bevy validates shaders at compile time)

**Step 3: Commit**

```bash
git add assets/shaders/ground_truth.wgsl
git commit -m "feat(shaders): add ground_truth.wgsl compute shader"
```

---

## Task 5: Create Ground Truth GPU Pipeline

**Files:**
- Create: `src/gpu/ground_truth.rs`
- Modify: `src/gpu/mod.rs`

**Step 1: Create the pipeline module**

```rust
// src/gpu/ground_truth.rs
//! GPU pipeline for ground truth blade rendering.

use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue, RenderContext},
    render_graph::{Node, RenderLabel},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use crate::ground_truth::GroundTruthConfig;
use crate::playback::PlaybackState;
use super::resources::GroundTruthImage;

/// GPU uniform buffer for ground truth parameters
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGroundTruthParams {
    pub center_x: f32,
    pub center_y: f32,
    pub r_min: f32,
    pub r_max: f32,
    pub blade_count: u32,
    pub angular_velocity: f32,
    pub current_time: f32,
    pub sweep_k: f32,
    pub width_root: f32,
    pub width_tip: f32,
    pub edge_thickness: f32,
    pub _padding: f32,
}

#[derive(Resource)]
pub struct GroundTruthPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for GroundTruthPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Ground Truth Pipeline Layout"),
            &[
                // Output texture (RGBA8)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Params uniform buffer
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
            ],
        );

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/ground_truth.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Ground Truth Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        GroundTruthPipeline { layout, pipeline }
    }
}

#[derive(Resource, Default)]
pub struct GroundTruthBindGroup {
    pub bind_group: Option<BindGroup>,
    pub params_buffer: Option<Buffer>,
}

pub fn prepare_ground_truth(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<GroundTruthPipeline>,
    config: Res<GroundTruthConfig>,
    playback: Res<PlaybackState>,
    gt_image: Res<GroundTruthImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut bind_group_res: ResMut<GroundTruthBindGroup>,
) {
    // Skip if ground truth is disabled
    if !config.enabled {
        return;
    }

    // Pack params for GPU
    let gpu_params = GpuGroundTruthParams {
        center_x: config.center_x,
        center_y: config.center_y,
        r_min: config.radius_min,
        r_max: config.radius_max,
        blade_count: config.blade_count,
        angular_velocity: config.angular_velocity(),
        current_time: playback.current_time / 1_000_000.0, // Convert us to seconds
        sweep_k: config.sweep_k,
        width_root: config.width_root_rad,
        width_tip: config.width_tip_rad,
        edge_thickness: config.edge_thickness_px,
        _padding: 0.0,
    };

    // Create or update params buffer
    let buffer = if let Some(ref existing) = bind_group_res.params_buffer {
        render_queue.write_buffer(existing, 0, bytemuck::bytes_of(&gpu_params));
        existing.clone()
    } else {
        let new_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Ground Truth Params Buffer"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        bind_group_res.params_buffer = Some(new_buffer.clone());
        new_buffer
    };

    // Create bind group if texture is ready
    if let Some(gt_gpu) = gpu_images.get(&gt_image.handle) {
        let bind_group = render_device.create_bind_group(
            Some("Ground Truth Bind Group"),
            &pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&gt_gpu.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        );
        bind_group_res.bind_group = Some(bind_group);
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct GroundTruthLabel;

#[derive(Default)]
pub struct GroundTruthNode;

impl Node for GroundTruthNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        // Check if ground truth is enabled
        let config = world.resource::<GroundTruthConfig>();
        if !config.enabled {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GroundTruthPipeline>();
        let bind_group_res = world.resource::<GroundTruthBindGroup>();

        let Some(ref bind_group) = bind_group_res.bind_group else {
            return Ok(());
        };

        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Ground Truth"),
                    timestamp_writes: None,
                });

            pass.set_pipeline(compute_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            // Dispatch for 1280x720 with 8x8 workgroups
            let workgroups_x = (1280 + 7) / 8;
            let workgroups_y = (720 + 7) / 8;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        Ok(())
    }
}
```

**Step 2: Export from mod.rs**

Add to `src/gpu/mod.rs`:

```rust
pub mod ground_truth;

// In the re-exports section:
pub use ground_truth::{
    GroundTruthBindGroup, GroundTruthLabel, GroundTruthNode, GroundTruthPipeline,
    prepare_ground_truth,
};
```

**Step 3: Build and verify**

Run: `cargo build`

**Step 4: Commit**

```bash
git add src/gpu/ground_truth.rs src/gpu/mod.rs
git commit -m "feat(gpu): add ground truth compute pipeline"
```

---

## Task 6: Integrate Ground Truth into Render Graph

**Files:**
- Modify: `src/edge_detection.rs`

**Step 1: Read the current edge_detection.rs to find integration points**

Run: Read `src/edge_detection.rs` to find:
- Where pipelines are initialized
- Where prepare systems are registered
- Where render graph nodes are added

**Step 2: Add ground truth pipeline initialization**

Add imports at top:
```rust
use crate::gpu::{
    GroundTruthBindGroup, GroundTruthLabel, GroundTruthNode, GroundTruthPipeline,
    prepare_ground_truth,
};
use crate::gpu::GroundTruthImage;
use crate::ground_truth::GroundTruthConfig;
```

In `build()` for render app:
```rust
// Add after other pipeline inits
.init_resource::<GroundTruthPipeline>()
.init_resource::<GroundTruthBindGroup>()
```

Add prepare system:
```rust
.add_systems(Render, prepare_ground_truth.in_set(RenderSystems::Prepare))
```

Add render graph node:
```rust
render_graph.add_node(GroundTruthLabel, GroundTruthNode);
render_graph.add_node_edge(PreprocessLabel, GroundTruthLabel);
```

**Step 3: Add ExtractResource for GroundTruthConfig and GroundTruthImage**

```rust
.add_plugins(ExtractResourcePlugin::<GroundTruthConfig>::default())
.add_plugins(ExtractResourcePlugin::<GroundTruthImage>::default())
```

**Step 4: Build and verify**

Run: `cargo build`

**Step 5: Commit**

```bash
git add src/edge_detection.rs
git commit -m "feat: integrate ground truth pipeline into render graph"
```

---

## Task 7: Create Ground Truth Texture in Setup

**Files:**
- Modify: `src/event_renderer.rs`

**Step 1: Add GroundTruthImage to setup_scene**

Add to function signature:
```rust
mut ground_truth_image_res: ResMut<GroundTruthImage>,
```

Add texture creation (after log_image creation):
```rust
// Create ground truth texture (RGBA8 for R=edge, G=interior)
let mut ground_truth_image = Image::new_fill(
    Extent3d {
        width: 1280,
        height: 720,
        depth_or_array_layers: 1,
    },
    TextureDimension::D2,
    &[0, 0, 0, 255],
    TextureFormat::Rgba8Unorm,
    RenderAssetUsages::RENDER_WORLD,
);
ground_truth_image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
    | TextureUsages::TEXTURE_BINDING
    | TextureUsages::COPY_SRC;
let ground_truth_handle = images.add(ground_truth_image);
ground_truth_image_res.handle = ground_truth_handle.clone();
```

**Step 2: Build and verify**

Run: `cargo build`

**Step 3: Commit**

```bash
git add src/event_renderer.rs
git commit -m "feat: create ground truth texture in setup_scene"
```

---

## Task 8: Load Ground Truth Config on Data Load

**Files:**
- Modify: `src/event_renderer.rs`

**Step 1: Add ground truth loading to load_data**

Add import:
```rust
use crate::ground_truth::GroundTruthConfig;
```

Add to load_data function after loading events:
```rust
// Try to load ground truth config from sidecar JSON
let gt_config = GroundTruthConfig::load_from_sidecar(std::path::Path::new(path))
    .unwrap_or_default();

if gt_config.enabled {
    info!(
        "Loaded ground truth config: {} blades, {} RPM, center ({}, {})",
        gt_config.blade_count, gt_config.rpm, gt_config.center_x, gt_config.center_y
    );
}
commands.insert_resource(gt_config);
```

**Step 2: Initialize default GroundTruthConfig in main app**

In the app setup (likely main.rs or lib.rs), add:
```rust
.init_resource::<GroundTruthConfig>()
```

**Step 3: Build and test with synthetic data**

Run: `cargo run -- data/synthetic/fan_test.dat`

Expected: Log message showing ground truth config loaded

**Step 4: Commit**

```bash
git add src/event_renderer.rs src/main.rs
git commit -m "feat: load ground truth config from JSON sidecar"
```

---

## Task 9: Add Ground Truth to Visualizer

**Files:**
- Modify: `assets/shaders/visualizer.wgsl`
- Modify: `src/event_renderer.rs`

**Step 1: Update visualizer shader**

Add to Params struct:
```wgsl
    show_ground_truth: u32,
```

Add bindings (after log):
```wgsl
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var ground_truth_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var ground_truth_sampler: sampler;
```

Add rendering layer (after LoG):
```wgsl
    // Layer 4: Ground truth edges (green) with 50% alpha blend
    if (params.show_ground_truth == 1u) {
        let gt_val = textureSample(ground_truth_texture, ground_truth_sampler, in.uv);
        // R channel = edge pixels
        if (gt_val.r > 0.0) {
            let green = vec3<f32>(0.0, 1.0, 0.0);
            output_color = mix(output_color, green, 0.5);
        }
    }
```

**Step 2: Update EventParams struct in event_renderer.rs**

```rust
#[derive(ShaderType, Debug, Clone, Copy)]
struct EventParams {
    width: f32,
    height: f32,
    time: f32,
    decay_tau: f32,
    show_sobel: u32,
    show_raw: u32,
    show_canny: u32,
    show_log: u32,
    show_ground_truth: u32,
}
```

**Step 3: Update EventMaterial struct**

```rust
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct EventMaterial {
    #[uniform(0)]
    params: EventParams,
    #[texture(1, sample_type = "u_int")]
    surface_texture: Handle<Image>,
    #[texture(2)]
    #[sampler(3)]
    sobel_texture: Handle<Image>,
    #[texture(4)]
    #[sampler(5)]
    canny_texture: Handle<Image>,
    #[texture(6)]
    #[sampler(7)]
    log_texture: Handle<Image>,
    #[texture(8)]
    #[sampler(9)]
    ground_truth_texture: Handle<Image>,
}
```

**Step 4: Update material creation**

In setup_scene, add ground_truth_texture to material:
```rust
ground_truth_texture: ground_truth_handle,
```

**Step 5: Add show_ground_truth to EdgeParams**

In `src/gpu/resources.rs`:
```rust
pub show_ground_truth: bool,
```

**Step 6: Build and verify**

Run: `cargo build`

**Step 7: Commit**

```bash
git add assets/shaders/visualizer.wgsl src/event_renderer.rs src/gpu/resources.rs
git commit -m "feat: add ground truth visualization (green overlay)"
```

---

## Task 10: Add UI Toggle for Ground Truth

**Files:**
- Modify: `src/event_renderer.rs`

**Step 1: Add checkbox in UI system**

Find the edge params UI section and add:
```rust
ui.checkbox(&mut edge_params.show_ground_truth, "Show Ground Truth (Green)");
```

**Step 2: Update material update system**

Add to the system that syncs EdgeParams to material:
```rust
material.params.show_ground_truth = if edge_params.show_ground_truth { 1 } else { 0 };
```

**Step 3: Test visually**

Run: `cargo run -- data/synthetic/fan_test.dat`

Expected: Checkbox appears, green overlay shows blade edges when enabled

**Step 4: Commit**

```bash
git add src/event_renderer.rs
git commit -m "feat: add UI toggle for ground truth visualization"
```

---

## Task 11: Add Precision/Recall Metrics (Optional Enhancement)

**Files:**
- Modify: `src/hyperparams.rs`
- Modify: `src/bin/hypertest.rs`

**Step 1: Add metrics fields to HyperResult**

```rust
pub precision: f32,
pub recall: f32,
pub f1_score: f32,
pub iou: f32,
```

**Step 2: Compute metrics when ground truth is available**

In hypertest, after readback:
```rust
// Compare detector edges to ground truth if available
if config.enabled {
    let (tp, fp, fn_) = compare_edges(&detector_data, &ground_truth_data);
    let precision = tp as f32 / (tp + fp).max(1) as f32;
    let recall = tp as f32 / (tp + fn_).max(1) as f32;
    let f1 = 2.0 * precision * recall / (precision + recall).max(0.001);
    let iou = tp as f32 / (tp + fp + fn_).max(1) as f32;
    // Store in result
}
```

**Step 3: Update CSV export**

Add columns: `precision,recall,f1,iou`

**Step 4: Commit**

```bash
git add src/hyperparams.rs src/bin/hypertest.rs
git commit -m "feat: add precision/recall/F1/IoU metrics for ground truth comparison"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Update synthesis JSON output | `src/synthesis.rs` |
| 2 | Create GroundTruthConfig + loader | `src/ground_truth.rs`, `src/lib.rs` |
| 3 | Add GroundTruthImage resource | `src/gpu/resources.rs`, `src/gpu/mod.rs` |
| 4 | Create ground_truth.wgsl shader | `assets/shaders/ground_truth.wgsl` |
| 5 | Create GPU pipeline | `src/gpu/ground_truth.rs`, `src/gpu/mod.rs` |
| 6 | Integrate into render graph | `src/edge_detection.rs` |
| 7 | Create texture in setup | `src/event_renderer.rs` |
| 8 | Load config on data load | `src/event_renderer.rs` |
| 9 | Add to visualizer | `visualizer.wgsl`, `event_renderer.rs` |
| 10 | Add UI toggle | `src/event_renderer.rs` |
| 11 | Add metrics (optional) | `hyperparams.rs`, `hypertest.rs` |

Each task has a commit checkpoint. Test with: `cargo run -- data/synthetic/fan_test.dat`

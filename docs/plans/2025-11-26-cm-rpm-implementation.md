# Contrast Maximization RPM Estimation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Canny edge detector in compare_live Q3 with Event Contrast Maximization (CM) for direct RPM estimation from event data.

**Architecture:** Single-pass GPU pipeline that warps events by candidate angular velocities, computes gradient-based contrast for each, and selects the best match. Uses existing centroid tracking for rotation center and auto-detects RPM range from event statistics.

**Tech Stack:** Rust, Bevy 0.15, WGPU compute shaders (WGSL), bytemuck for GPU structs

---

### Task 1: Create CM Module Structure

**Files:**
- Create: `src/cm/mod.rs`
- Create: `src/cm/resources.rs`
- Modify: `src/lib.rs`

**Step 1: Create resources file with type definitions**

```rust
// src/cm/resources.rs
//! Resources for Contrast Maximization RPM estimation

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bytemuck::{Pod, Zeroable};

/// CM output image (replaces Canny in Q3)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct CmImage {
    pub handle: Handle<Image>,
}

/// CM parameters for CPU-side control
#[derive(Resource, Clone)]
pub struct CmParams {
    pub n_omega: u32,
    pub enabled: bool,
}

impl Default for CmParams {
    fn default() -> Self {
        Self {
            n_omega: 64,
            enabled: true,
        }
    }
}

/// GPU-compatible CM parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuCmParams {
    pub centroid_x: f32,
    pub centroid_y: f32,
    pub t_ref: f32,
    pub omega_min: f32,
    pub omega_step: f32,
    pub n_omega: u32,
    pub window_start: u32,
    pub window_end: u32,
    pub event_count: u32,
    pub _padding: [u32; 3],
}

/// CM results read back from GPU
#[derive(Resource, Default, Clone, Debug)]
pub struct CmResult {
    pub best_omega: f32,
    pub best_contrast: f32,
    pub rpm: f32,
    pub confidence: f32,
}

impl CmResult {
    /// Update with temporal smoothing
    pub fn update_smoothed(&mut self, new_omega: f32, new_contrast: f32, alpha: f32) {
        if self.best_omega > 0.0 && new_omega > 0.0 {
            self.best_omega = alpha * new_omega + (1.0 - alpha) * self.best_omega;
        } else if new_omega > 0.0 {
            self.best_omega = new_omega;
        }
        self.best_contrast = new_contrast;
        // Convert rad/μs to RPM: ω * (60s/min) / (2π rad/rev) * (1e6 μs/s)
        self.rpm = self.best_omega.abs() * 60.0 / std::f32::consts::TAU * 1e6;
        self.confidence = (new_contrast / 1000.0).min(1.0);
    }
}
```

**Step 2: Create module file**

```rust
// src/cm/mod.rs
//! Contrast Maximization for RPM estimation
//!
//! Replaces Canny edge detector with CM-based motion compensation.

mod resources;

pub use resources::*;

use bevy::prelude::*;

/// Plugin for Contrast Maximization RPM estimation
pub struct CmPlugin;

impl Plugin for CmPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CmParams>()
           .init_resource::<CmResult>();
    }
}
```

**Step 3: Export module from lib.rs**

Add to `src/lib.rs`:
```rust
pub mod cm;
```

**Step 4: Verify compilation**

Run: `cargo check`
Expected: Compiles with no errors

**Step 5: Commit**

```bash
git add src/cm/mod.rs src/cm/resources.rs src/lib.rs
git commit -m "feat(cm): add module structure and resource types"
```

---

### Task 2: Create CM Warp Shader

**Files:**
- Create: `assets/shaders/cm_warp.wgsl`

**Step 1: Write the warp shader**

```wgsl
// Contrast Maximization: Warp events by candidate angular velocities
// Builds Image of Warped Events (IWE) for each omega candidate

struct GpuEvent {
    timestamp: u32,
    data: u32,  // packed x[13:0], y[27:14], polarity[31:28]
}

struct CmParams {
    centroid_x: f32,
    centroid_y: f32,
    t_ref: f32,
    omega_min: f32,
    omega_step: f32,
    n_omega: u32,
    window_start: u32,
    window_end: u32,
    event_count: u32,
    _padding: vec3<u32>,
}

@group(0) @binding(0) var<storage, read> events: array<GpuEvent>;
@group(0) @binding(1) var<uniform> params: CmParams;
@group(0) @binding(2) var<storage, read_write> iwe_buffer: array<atomic<u32>>;

// IWE dimensions
const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;

fn unpack_x(data: u32) -> u32 {
    return data & 0x3FFFu;  // bits 0-13
}

fn unpack_y(data: u32) -> u32 {
    return (data >> 14u) & 0x3FFFu;  // bits 14-27
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let event_idx = gid.x;
    if (event_idx >= params.event_count) {
        return;
    }

    let event = events[event_idx];

    // Filter by time window
    if (event.timestamp < params.window_start || event.timestamp > params.window_end) {
        return;
    }

    // Unpack event coordinates
    let ex = f32(unpack_x(event.data));
    let ey = f32(unpack_y(event.data));

    // Convert to polar around centroid
    let dx = ex - params.centroid_x;
    let dy = ey - params.centroid_y;
    let r = sqrt(dx * dx + dy * dy);

    // Skip events at center (undefined angle)
    if (r < 1.0) {
        return;
    }

    let theta = atan2(dy, dx);
    let dt = f32(event.timestamp) - params.t_ref;

    // For each omega candidate, warp and accumulate
    for (var i = 0u; i < params.n_omega; i++) {
        let omega = params.omega_min + f32(i) * params.omega_step;
        let theta_warped = theta - omega * dt;

        // Convert back to Cartesian
        let x_warped = params.centroid_x + r * cos(theta_warped);
        let y_warped = params.centroid_y + r * sin(theta_warped);

        let ix = u32(x_warped);
        let iy = u32(y_warped);

        if (ix < WIDTH && iy < HEIGHT) {
            // Calculate buffer index: slice * (WIDTH * HEIGHT) + y * WIDTH + x
            let buffer_idx = i * (WIDTH * HEIGHT) + iy * WIDTH + ix;
            atomicAdd(&iwe_buffer[buffer_idx], 1u);
        }
    }
}
```

**Step 2: Verify shader syntax**

Run: `cargo build`
Expected: Compiles (shader loaded at runtime, syntax checked then)

**Step 3: Commit**

```bash
git add assets/shaders/cm_warp.wgsl
git commit -m "feat(cm): add event warp shader for IWE construction"
```

---

### Task 3: Create CM Contrast Shader

**Files:**
- Create: `assets/shaders/cm_contrast.wgsl`

**Step 1: Write the contrast calculation shader**

```wgsl
// Contrast Maximization: Compute gradient-based contrast for each IWE slice
// Uses sum of squared Sobel gradient magnitudes

struct ContrastParams {
    n_omega: u32,
    width: u32,
    height: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> iwe_buffer: array<u32>;
@group(0) @binding(1) var<uniform> params: ContrastParams;
@group(0) @binding(2) var<storage, read_write> contrast: array<atomic<u32>>;

const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;

fn get_iwe(omega_idx: u32, x: i32, y: i32) -> f32 {
    if (x < 0 || x >= i32(WIDTH) || y < 0 || y >= i32(HEIGHT)) {
        return 0.0;
    }
    let idx = omega_idx * (WIDTH * HEIGHT) + u32(y) * WIDTH + u32(x);
    return f32(iwe_buffer[idx]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let omega_idx = gid.z;

    if (gid.x >= WIDTH || gid.y >= HEIGHT || omega_idx >= params.n_omega) {
        return;
    }

    // Skip border pixels for Sobel
    if (x < 1 || x >= i32(WIDTH) - 1 || y < 1 || y >= i32(HEIGHT) - 1) {
        return;
    }

    // Load 3x3 neighborhood
    let p00 = get_iwe(omega_idx, x - 1, y - 1);
    let p01 = get_iwe(omega_idx, x, y - 1);
    let p02 = get_iwe(omega_idx, x + 1, y - 1);
    let p10 = get_iwe(omega_idx, x - 1, y);
    let p12 = get_iwe(omega_idx, x + 1, y);
    let p20 = get_iwe(omega_idx, x - 1, y + 1);
    let p21 = get_iwe(omega_idx, x, y + 1);
    let p22 = get_iwe(omega_idx, x + 1, y + 1);

    // Sobel kernels
    let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
    let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

    // Squared magnitude (avoid sqrt for performance)
    let mag_sq = gx * gx + gy * gy;

    // Atomic add to contrast sum (scaled to avoid precision loss)
    let scaled = u32(mag_sq * 100.0);
    if (scaled > 0u) {
        atomicAdd(&contrast[omega_idx], scaled);
    }
}
```

**Step 2: Commit**

```bash
git add assets/shaders/cm_contrast.wgsl
git commit -m "feat(cm): add gradient-based contrast calculation shader"
```

---

### Task 4: Create CM Select Shader

**Files:**
- Create: `assets/shaders/cm_select.wgsl`

**Step 1: Write the selection shader**

```wgsl
// Contrast Maximization: Find best omega and copy its IWE to output
// Single workgroup performs parallel reduction

struct SelectParams {
    n_omega: u32,
    width: u32,
    height: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> contrast: array<u32>;
@group(0) @binding(1) var<storage, read> iwe_buffer: array<u32>;
@group(0) @binding(2) var<uniform> params: SelectParams;
@group(0) @binding(3) var output_texture: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;  // [best_idx, best_contrast]

const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;

var<workgroup> shared_idx: array<u32, 128>;
var<workgroup> shared_val: array<u32, 128>;

@compute @workgroup_size(128)
fn find_best(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Each thread handles multiple omega values
    var local_best_idx = 0u;
    var local_best_val = 0u;

    var i = tid;
    while (i < params.n_omega) {
        let val = contrast[i];
        if (val > local_best_val) {
            local_best_val = val;
            local_best_idx = i;
        }
        i += 128u;
    }

    shared_idx[tid] = local_best_idx;
    shared_val[tid] = local_best_val;
    workgroupBarrier();

    // Parallel reduction
    var stride = 64u;
    while (stride > 0u) {
        if (tid < stride && tid + stride < 128u) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 writes result
    if (tid == 0u) {
        result[0] = shared_idx[0];
        result[1] = shared_val[0];
    }
}

@compute @workgroup_size(8, 8)
fn copy_best(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let x = gid.x;
    let y = gid.y;

    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }

    let best_idx = result[0];
    let iwe_idx = best_idx * (WIDTH * HEIGHT) + y * WIDTH + x;
    let value = f32(iwe_buffer[iwe_idx]);

    // Normalize for visualization (log scale for better contrast)
    let normalized = log2(value + 1.0) / 10.0;

    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(normalized, 0.0, 0.0, 1.0));
}
```

**Step 2: Commit**

```bash
git add assets/shaders/cm_select.wgsl
git commit -m "feat(cm): add best omega selection and IWE copy shader"
```

---

### Task 5: Implement CM Pipeline

**Files:**
- Create: `src/cm/pipeline.rs`
- Modify: `src/cm/mod.rs`

**Step 1: Write pipeline implementation**

```rust
// src/cm/pipeline.rs
//! GPU pipeline for Contrast Maximization

use bevy::prelude::*;
use bevy::render::{
    render_resource::*,
    renderer::{RenderDevice, RenderQueue, RenderContext},
    render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
    render_asset::RenderAssets,
    texture::GpuImage,
};
use bytemuck::{Pod, Zeroable};

use super::{CmImage, GpuCmParams};
use crate::gpu::{GpuEventBuffer, EventData};
use crate::playback::PlaybackState;
use crate::metrics::EdgeMetrics;

/// Render graph label for CM node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct CmLabel;

/// Contrast params for GPU
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuContrastParams {
    pub n_omega: u32,
    pub width: u32,
    pub height: u32,
    pub _padding: u32,
}

/// CM Pipeline resources
#[derive(Resource)]
pub struct CmPipeline {
    pub warp_pipeline: CachedComputePipelineId,
    pub warp_layout: BindGroupLayout,
    pub contrast_pipeline: CachedComputePipelineId,
    pub contrast_layout: BindGroupLayout,
    pub select_pipeline: CachedComputePipelineId,
    pub copy_pipeline: CachedComputePipelineId,
    pub select_layout: BindGroupLayout,
}

impl FromWorld for CmPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();

        // Warp pipeline layout
        let warp_layout = render_device.create_bind_group_layout(
            "cm_warp_layout",
            &[
                // Events buffer
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
                // CM params
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
                // IWE buffer (read_write)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        // Contrast pipeline layout
        let contrast_layout = render_device.create_bind_group_layout(
            "cm_contrast_layout",
            &[
                // IWE buffer (read)
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
                // Contrast params
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
                // Contrast output
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        // Select pipeline layout
        let select_layout = render_device.create_bind_group_layout(
            "cm_select_layout",
            &[
                // Contrast buffer
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
                // IWE buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Select params
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
                // Output texture
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Result buffer
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let pipeline_cache = world.resource::<PipelineCache>();

        let warp_shader = asset_server.load("shaders/cm_warp.wgsl");
        let warp_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cm_warp_pipeline".into()),
            layout: vec![warp_layout.clone()],
            shader: warp_shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let contrast_shader = asset_server.load("shaders/cm_contrast.wgsl");
        let contrast_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cm_contrast_pipeline".into()),
            layout: vec![contrast_layout.clone()],
            shader: contrast_shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let select_shader = asset_server.load("shaders/cm_select.wgsl");
        let select_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cm_select_pipeline".into()),
            layout: vec![select_layout.clone()],
            shader: select_shader.clone(),
            shader_defs: vec![],
            entry_point: Some("find_best".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let copy_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cm_copy_pipeline".into()),
            layout: vec![select_layout.clone()],
            shader: select_shader,
            shader_defs: vec![],
            entry_point: Some("copy_best".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        Self {
            warp_pipeline,
            warp_layout,
            contrast_pipeline,
            contrast_layout,
            select_pipeline,
            copy_pipeline,
            select_layout,
        }
    }
}

/// Bind groups for CM pipeline
#[derive(Resource, Default)]
pub struct CmBindGroups {
    pub warp: Option<BindGroup>,
    pub contrast: Option<BindGroup>,
    pub select: Option<BindGroup>,
}

/// GPU buffers for CM
#[derive(Resource)]
pub struct CmBuffers {
    pub params: Buffer,
    pub iwe: Buffer,
    pub contrast: Buffer,
    pub contrast_params: Buffer,
    pub select_params: Buffer,
    pub result: Buffer,
}

/// CM render node
#[derive(Default)]
pub struct CmNode;

impl Node for CmNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<CmPipeline>();
        let Some(bind_groups) = world.get_resource::<CmBindGroups>() else {
            return Ok(());
        };

        let (Some(warp_bg), Some(contrast_bg), Some(select_bg)) =
            (&bind_groups.warp, &bind_groups.contrast, &bind_groups.select) else {
            return Ok(());
        };

        // Get pipelines
        let Some(warp_pl) = pipeline_cache.get_compute_pipeline(pipeline.warp_pipeline) else {
            return Ok(());
        };
        let Some(contrast_pl) = pipeline_cache.get_compute_pipeline(pipeline.contrast_pipeline) else {
            return Ok(());
        };
        let Some(select_pl) = pipeline_cache.get_compute_pipeline(pipeline.select_pipeline) else {
            return Ok(());
        };
        let Some(copy_pl) = pipeline_cache.get_compute_pipeline(pipeline.copy_pipeline) else {
            return Ok(());
        };

        let encoder = render_context.command_encoder();

        // Pass 1: Warp events to build IWE
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cm_warp"),
                timestamp_writes: None,
            });
            pass.set_pipeline(warp_pl);
            pass.set_bind_group(0, warp_bg, &[]);
            // Dispatch for event count (will be set in prepare)
            pass.dispatch_workgroups(4096, 1, 1); // ~1M events max
        }

        // Pass 2: Compute contrast for each omega
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cm_contrast"),
                timestamp_writes: None,
            });
            pass.set_pipeline(contrast_pl);
            pass.set_bind_group(0, contrast_bg, &[]);
            // Dispatch for image size * n_omega
            let wg_x = (1280 + 7) / 8;
            let wg_y = (720 + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 64); // n_omega slices
        }

        // Pass 3: Find best omega
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cm_select"),
                timestamp_writes: None,
            });
            pass.set_pipeline(select_pl);
            pass.set_bind_group(0, select_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Pass 4: Copy best IWE to output
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cm_copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(copy_pl);
            pass.set_bind_group(0, select_bg, &[]);
            let wg_x = (1280 + 7) / 8;
            let wg_y = (720 + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        Ok(())
    }
}
```

**Step 2: Update mod.rs to export pipeline**

```rust
// src/cm/mod.rs - add after resources
mod pipeline;

pub use pipeline::*;
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: Compiles (may have warnings about unused)

**Step 4: Commit**

```bash
git add src/cm/pipeline.rs src/cm/mod.rs
git commit -m "feat(cm): implement GPU pipeline with warp, contrast, select passes"
```

---

### Task 6: Implement CM Systems

**Files:**
- Create: `src/cm/systems.rs`
- Modify: `src/cm/mod.rs`

**Step 1: Write systems for buffer preparation**

```rust
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
    CmImage, CmParams, CmResult, GpuCmParams,
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
```

**Step 2: Update mod.rs with systems export and full plugin**

```rust
// src/cm/mod.rs - replace entire file
//! Contrast Maximization for RPM estimation
//!
//! Replaces Canny edge detector with CM-based motion compensation.

mod resources;
mod pipeline;
mod systems;

pub use resources::*;
pub use pipeline::*;
pub use systems::*;

use bevy::prelude::*;
use bevy::render::{
    RenderApp, Render, RenderSet,
    render_graph::RenderGraph,
    ExtractSchedule,
};

use crate::gpu::PreprocessLabel;
use crate::compare::CompositeLabel;

/// Plugin for Contrast Maximization RPM estimation
pub struct CmPlugin;

impl Plugin for CmPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CmParams>()
           .init_resource::<CmResult>();
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<CmPipeline>()
            .init_resource::<CmBindGroups>()
            .add_systems(ExtractSchedule, extract_cm_params)
            .add_systems(Render, prepare_cm.in_set(RenderSet::Prepare));

        // Add to render graph
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(CmLabel, CmNode::default());
        graph.add_node_edge(PreprocessLabel, CmLabel);
        graph.add_node_edge(CmLabel, CompositeLabel);
    }
}
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: May have import errors - fix in next step

**Step 4: Commit**

```bash
git add src/cm/systems.rs src/cm/mod.rs
git commit -m "feat(cm): implement extraction and preparation systems"
```

---

### Task 7: Integrate CM into compare_live

**Files:**
- Modify: `src/bin/compare_live.rs`
- Modify: `src/gpu/resources.rs`

**Step 1: Add CmImage resource initialization**

In `src/gpu/resources.rs`, add after `LogImage`:
```rust
/// CM output image (replaces Canny in compare mode)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct CmImage {
    pub handle: Handle<Image>,
}
```

**Step 2: Update compare_live.rs to use CmPlugin**

Find the plugin setup section and:
1. Add `use ebc_rs::cm::CmPlugin;`
2. Replace `.add_plugins(CannyPipeline)` with `.add_plugins(CmPlugin)`
3. Initialize `CmImage` resource similar to other images

**Step 3: Verify compilation**

Run: `cargo check --bin compare_live`

**Step 4: Commit**

```bash
git add src/bin/compare_live.rs src/gpu/resources.rs
git commit -m "feat(cm): integrate CmPlugin into compare_live binary"
```

---

### Task 8: Update Composite Shader

**Files:**
- Modify: `assets/shaders/composite.wgsl`
- Modify: `src/compare/composite.rs`

**Step 1: Update composite.wgsl binding 2 comment**

Change `// Canny (input)` to `// CM (input)` - the texture format is the same (R32Float).

**Step 2: Update composite.rs to use CmImage**

Replace `CannyImage` import and binding with `CmImage`.

**Step 3: Verify compilation**

Run: `cargo check`

**Step 4: Commit**

```bash
git add assets/shaders/composite.wgsl src/compare/composite.rs
git commit -m "feat(cm): update composite to use CM output instead of Canny"
```

---

### Task 9: Update UI for CM Display

**Files:**
- Modify: `src/compare/ui.rs`
- Modify: `src/compare/mod.rs`

**Step 1: Update AllDetectorMetrics**

In `src/compare/mod.rs`:
```rust
use crate::cm::CmResult;

#[derive(Resource, Default)]
pub struct AllDetectorMetrics {
    pub raw: DetectorMetrics,
    pub sobel: DetectorMetrics,
    pub cm: CmResult,  // Changed from canny
    pub log: DetectorMetrics,
    pub frame_time_ms: f32,
}
```

**Step 2: Update UI panel**

In `src/compare/ui.rs`, replace CANNY panel with CM panel showing RPM.

**Step 3: Verify compilation**

Run: `cargo check`

**Step 4: Commit**

```bash
git add src/compare/ui.rs src/compare/mod.rs
git commit -m "feat(cm): update UI to display RPM from CM results"
```

---

### Task 10: Test with Synthetic Data

**Files:**
- None (testing only)

**Step 1: Generate synthetic data**

Run: `cargo run --release --bin generate_synthetic`

**Step 2: Run compare_live**

Run: `cargo run --release --bin compare_live -- data/synthetic/fan_test.dat`

**Step 3: Verify Q3 shows CM output**

Expected: Q3 panel shows deblurred IWE and RPM value

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix(cm): address issues found during testing"
```

---

### Task 11: Test with Real Fan Data

**Files:**
- None (testing only)

**Step 1: Run with constant RPM data**

Run: `cargo run --release --bin compare_live -- data/fan/fan_const_rpm.dat`

**Step 2: Verify RPM stability**

Expected: RPM value should be relatively stable for constant-speed fan

**Step 3: Run with varying RPM data**

Run: `cargo run --release --bin compare_live -- data/fan/fan_varying_rpm.dat`

**Step 4: Verify RPM tracking**

Expected: RPM value should change as fan speed varies

**Step 5: Final commit**

```bash
git add -A
git commit -m "test(cm): verify CM RPM estimation with real fan data"
```

---

### Task 12: Documentation and Cleanup

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2025-11-26-cm-rpm-estimation.md`

**Step 1: Update README with CM information**

Add section about CM-based RPM estimation to README.

**Step 2: Mark design doc as implemented**

Update status in design doc header.

**Step 3: Final commit**

```bash
git add README.md docs/plans/2025-11-26-cm-rpm-estimation.md
git commit -m "docs: update README and mark CM design as implemented"
```

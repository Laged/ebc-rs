# Phase 1: Auto-RPM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the CMax-SLAM optimization loop to estimate RPM autonomously without ground truth.

**Architecture:** GPU computes 3 IWE slices (ω, ω+δ, ω-δ), reduces to contrast values via workgroup reduction, async readback to CPU, gradient optimizer updates ω for next frame.

**Tech Stack:** Rust, Bevy 0.17, WGSL compute shaders, wgpu async buffer mapping

---

## Task 1: Add ContrastResult Types and Channel Infrastructure

**Files:**
- Create: `src/cmax_slam/readback.rs`
- Modify: `src/cmax_slam/mod.rs`

**Step 1: Create the readback module with channel types**

Create `src/cmax_slam/readback.rs`:

```rust
//! Async readback infrastructure for CMax-SLAM contrast values

use bevy::prelude::*;
use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

/// Contrast values from GPU reduction (3 IWE slices)
#[derive(Clone, Debug, Default)]
pub struct ContrastValues {
    pub center: f32,
    pub plus: f32,
    pub minus: f32,
}

/// Main world resource - receives contrast values from render world
#[derive(Resource)]
pub struct ContrastReceiver {
    pub rx: Receiver<ContrastValues>,
}

/// Render world resource - sends contrast values to main world
#[derive(Resource, Clone)]
pub struct ContrastSender {
    pub tx: Arc<Mutex<Sender<ContrastValues>>>,
}

/// GPU-side contrast result buffer layout (matches WGSL struct)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuContrastResult {
    pub sum_sq_center: u32,
    pub sum_sq_plus: u32,
    pub sum_sq_minus: u32,
    pub pixel_count: u32,
}

/// Create channel and return both ends
pub fn create_contrast_channel() -> (ContrastSender, ContrastReceiver) {
    let (tx, rx) = channel();
    (
        ContrastSender {
            tx: Arc::new(Mutex::new(tx)),
        },
        ContrastReceiver { rx },
    )
}
```

**Step 2: Update mod.rs to export new types**

Add to `src/cmax_slam/mod.rs` after existing exports:

```rust
mod readback;
pub use readback::{
    ContrastValues, ContrastReceiver, ContrastSender,
    GpuContrastResult, create_contrast_channel,
};
```

**Step 3: Verify it compiles**

Run: `cargo check 2>&1 | tail -10`
Expected: Compiles with no errors in cmax_slam module

**Step 4: Commit**

```bash
git add src/cmax_slam/readback.rs src/cmax_slam/mod.rs
git commit -m "feat(cmax_slam): add contrast channel infrastructure for async readback"
```

---

## Task 2: Update CmaxSlamState with Optimizer Fields

**Files:**
- Modify: `src/cmax_slam/mod.rs` (CmaxSlamState struct)

**Step 1: Add new fields to CmaxSlamState**

In `src/cmax_slam/mod.rs`, update the `CmaxSlamState` struct:

```rust
/// Runtime state for CMax-SLAM optimization
#[derive(Resource)]
pub struct CmaxSlamState {
    /// Current angular velocity estimate (rad/μs)
    pub omega: f32,
    /// Rotation center
    pub centroid: Vec2,
    /// Current contrast value (for display)
    pub contrast: f32,
    /// Whether optimizer has converged
    pub converged: bool,

    // New fields for Phase 1
    /// Perturbation size for numerical gradient (rad/μs)
    pub delta_omega: f32,
    /// Recent omega values for convergence detection
    pub omega_history: VecDeque<f32>,
    /// Whether cold start initialization has occurred
    pub initialized: bool,
}

impl Default for CmaxSlamState {
    fn default() -> Self {
        Self {
            omega: 0.0,
            centroid: Vec2::new(640.0, 360.0),
            contrast: 0.0,
            converged: false,
            delta_omega: 1e-6,
            omega_history: VecDeque::with_capacity(16),
            initialized: false,
        }
    }
}
```

**Step 2: Add learning_rate to CmaxSlamParams**

Update `CmaxSlamParams` in the same file:

```rust
/// Parameters for CMax-SLAM pipeline
#[derive(Resource)]
pub struct CmaxSlamParams {
    /// Enable CMax-SLAM processing
    pub enabled: bool,
    /// Edge weighting factor
    pub edge_weight: f32,
    /// Smoothing factor for omega updates
    pub smoothing_alpha: f32,
    /// Learning rate for gradient optimizer
    pub learning_rate: f32,
}

impl Default for CmaxSlamParams {
    fn default() -> Self {
        Self {
            enabled: true,
            edge_weight: 1.0,
            smoothing_alpha: 0.1,
            learning_rate: 0.5,
        }
    }
}
```

**Step 3: Add VecDeque import**

Add at top of `src/cmax_slam/mod.rs`:

```rust
use std::collections::VecDeque;
```

**Step 4: Verify it compiles**

Run: `cargo check 2>&1 | tail -10`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add src/cmax_slam/mod.rs
git commit -m "feat(cmax_slam): add optimizer fields to CmaxSlamState"
```

---

## Task 3: Create Reduction Shader

**Files:**
- Create: `assets/shaders/cmax_slam_reduce.wgsl`

**Step 1: Write the reduction shader**

Create `assets/shaders/cmax_slam_reduce.wgsl`:

```wgsl
// CMax-SLAM: Reduce IWE slices to contrast values (sum of squares)
// Uses workgroup reduction to minimize atomic contention

struct ContrastResult {
    sum_sq_center: atomic<u32>,
    sum_sq_plus: atomic<u32>,
    sum_sq_minus: atomic<u32>,
    pixel_count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> iwe_buffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: ContrastResult;

const WIDTH: u32 = 1280u;
const HEIGHT: u32 = 720u;
const SLICE_SIZE: u32 = WIDTH * HEIGHT;
const WORKGROUP_SIZE: u32 = 256u;

// Workgroup shared memory for local reduction
var<workgroup> local_center: array<u32, WORKGROUP_SIZE>;
var<workgroup> local_plus: array<u32, WORKGROUP_SIZE>;
var<workgroup> local_minus: array<u32, WORKGROUP_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let global_idx = gid.x;
    let local_idx = lid.x;

    // Load pixel values (shift right by 12 to prevent overflow)
    // Raw values are 0-65280 (u8 * 256 from bilinear), >> 12 gives 0-15
    var val_c = 0u;
    var val_p = 0u;
    var val_m = 0u;

    if global_idx < SLICE_SIZE {
        val_c = iwe_buffer[global_idx] >> 12u;
        val_p = iwe_buffer[global_idx + SLICE_SIZE] >> 12u;
        val_m = iwe_buffer[global_idx + 2u * SLICE_SIZE] >> 12u;
    }

    // Square values (max 15^2 = 225)
    local_center[local_idx] = val_c * val_c;
    local_plus[local_idx] = val_p * val_p;
    local_minus[local_idx] = val_m * val_m;

    // Synchronize workgroup
    workgroupBarrier();

    // Tree reduction within workgroup
    var stride = WORKGROUP_SIZE / 2u;
    while stride > 0u {
        if local_idx < stride {
            local_center[local_idx] += local_center[local_idx + stride];
            local_plus[local_idx] += local_plus[local_idx + stride];
            local_minus[local_idx] += local_minus[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 adds workgroup result to global buffer
    if local_idx == 0u {
        atomicAdd(&result.sum_sq_center, local_center[0]);
        atomicAdd(&result.sum_sq_plus, local_plus[0]);
        atomicAdd(&result.sum_sq_minus, local_minus[0]);

        // Count pixels for debugging (one per workgroup)
        if global_idx < SLICE_SIZE {
            let pixels_in_workgroup = min(WORKGROUP_SIZE, SLICE_SIZE - wid.x * WORKGROUP_SIZE);
            atomicAdd(&result.pixel_count, pixels_in_workgroup);
        }
    }
}
```

**Step 2: Verify shader syntax**

Run: `cargo build 2>&1 | grep -i "cmax_slam_reduce\|error" | head -10`
Expected: No shader compilation errors (shader loaded at runtime, but syntax issues would show)

**Step 3: Commit**

```bash
git add assets/shaders/cmax_slam_reduce.wgsl
git commit -m "feat(shaders): add cmax_slam_reduce workgroup reduction shader"
```

---

## Task 4: Add Reduction Pipeline and Bind Groups

**Files:**
- Modify: `src/cmax_slam/pipeline.rs`

**Step 1: Read current pipeline.rs**

Run: `head -100 src/cmax_slam/pipeline.rs`

**Step 2: Add reduction pipeline to CmaxSlamPipeline struct**

Add the reduce pipeline and bind group layout. The exact location depends on current structure, but add these fields to `CmaxSlamPipeline`:

```rust
pub reduce_pipeline: CachedComputePipelineId,
pub reduce_layout: BindGroupLayout,
```

**Step 3: Create reduction pipeline in FromWorld**

In the `FromWorld` implementation, add after existing pipeline creation:

```rust
// Reduction shader pipeline
let reduce_shader = world.load_asset("shaders/cmax_slam_reduce.wgsl");

let reduce_layout = render_device.create_bind_group_layout(
    "cmax_slam_reduce_layout",
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
        // Contrast result (read_write)
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
    ],
);

let reduce_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some("cmax_slam_reduce_pipeline".into()),
    layout: vec![reduce_layout.clone()],
    shader: reduce_shader,
    shader_defs: vec![],
    entry_point: Some("main".into()),
    push_constant_ranges: vec![],
    zero_initialize_workgroup_memory: true,
});
```

**Step 4: Add reduce bind group to CmaxSlamBindGroups**

```rust
pub struct CmaxSlamBindGroups {
    pub warp_contrast: Option<BindGroup>,
    pub output: Option<BindGroup>,
    pub reduce: Option<BindGroup>,  // New
}
```

**Step 5: Verify it compiles**

Run: `cargo check 2>&1 | tail -15`
Expected: May have errors about missing buffer - that's OK, we'll add it in Task 5

**Step 6: Commit (if compiles) or note for Task 5**

```bash
git add src/cmax_slam/pipeline.rs
git commit -m "feat(cmax_slam): add reduction pipeline and bind group layout"
```

---

## Task 5: Add Contrast Result Buffer and Prepare Bind Group

**Files:**
- Modify: `src/cmax_slam/pipeline.rs` (CmaxSlamBuffers)
- Modify: `src/cmax_slam/systems.rs` (prepare system)

**Step 1: Add contrast_result buffer to CmaxSlamBuffers**

In `pipeline.rs`, update `CmaxSlamBuffers`:

```rust
#[derive(Resource, Clone)]
pub struct CmaxSlamBuffers {
    pub params: Buffer,
    pub iwe: Buffer,
    pub contrast: Buffer,
    pub contrast_result: Buffer,  // New - for reduction output
    pub contrast_staging: Buffer, // New - for async readback
}
```

**Step 2: Create buffers in prepare_cmax_slam system**

In `systems.rs`, in the buffer creation section:

```rust
// Contrast result buffer (4 x u32 = 16 bytes)
let contrast_result_size = std::mem::size_of::<GpuContrastResult>() as u64;

let contrast_result = render_device.create_buffer(&BufferDescriptor {
    label: Some("cmax_slam_contrast_result"),
    size: contrast_result_size,
    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
});

// Staging buffer for async readback
let contrast_staging = render_device.create_buffer(&BufferDescriptor {
    label: Some("cmax_slam_contrast_staging"),
    size: contrast_result_size,
    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

**Step 3: Create reduce bind group**

```rust
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
```

**Step 4: Verify it compiles**

Run: `cargo check 2>&1 | tail -15`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add src/cmax_slam/pipeline.rs src/cmax_slam/systems.rs
git commit -m "feat(cmax_slam): add contrast result buffers and reduce bind group"
```

---

## Task 6: Add Reduction Pass to Render Node

**Files:**
- Modify: `src/cmax_slam/pipeline.rs` (CmaxSlamNode::run)

**Step 1: Read current node implementation**

Run: `grep -n "impl Node for CmaxSlamNode" -A 100 src/cmax_slam/pipeline.rs | head -80`

**Step 2: Add reduction pass after warp pass**

In the `run` method of `CmaxSlamNode`, after the warp compute pass, add:

```rust
// Clear contrast result buffer before reduction
encoder.clear_buffer(&buffers.contrast_result, 0, None);

// Reduction pass - compute sum of squares for each IWE slice
{
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("cmax_slam_reduce"),
        timestamp_writes: None,
    });

    let reduce_pipeline = pipeline_cache
        .get_compute_pipeline(pipelines.reduce_pipeline)
        .expect("Reduce pipeline not ready");

    pass.set_pipeline(reduce_pipeline);
    pass.set_bind_group(0, reduce_bind_group, &[]);

    // Dispatch enough workgroups to cover all pixels
    // SLICE_SIZE = 1280 * 720 = 921600
    // Workgroups needed = ceil(921600 / 256) = 3600
    let workgroups = (921600 + 255) / 256;
    pass.dispatch_workgroups(workgroups, 1, 1);
}

// Copy result to staging buffer for async readback
encoder.copy_buffer_to_buffer(
    &buffers.contrast_result,
    0,
    &buffers.contrast_staging,
    0,
    std::mem::size_of::<GpuContrastResult>() as u64,
);
```

**Step 3: Verify it compiles**

Run: `cargo check 2>&1 | tail -15`
Expected: Compiles successfully

**Step 4: Commit**

```bash
git add src/cmax_slam/pipeline.rs
git commit -m "feat(cmax_slam): add reduction pass to render node"
```

---

## Task 7: Implement Async Readback System

**Files:**
- Modify: `src/cmax_slam/readback.rs`
- Modify: `src/cmax_slam/systems.rs`

**Step 1: Add readback trigger function to readback.rs**

```rust
use bevy::render::renderer::RenderDevice;
use wgpu::{Buffer, MapMode, BufferSlice};

/// Trigger async readback of contrast values
pub fn trigger_contrast_readback(
    staging_buffer: &Buffer,
    sender: &ContrastSender,
    render_device: &RenderDevice,
) {
    let slice = staging_buffer.slice(..);
    let tx = sender.tx.clone();

    slice.map_async(MapMode::Read, move |result| {
        if result.is_ok() {
            // Note: We can't access the buffer data in this callback directly
            // The actual read happens in a polling system
        }
    });

    // Poll to drive the async operation
    render_device.poll(wgpu::Maintain::Poll);
}

/// Check if readback is ready and send values
pub fn poll_contrast_readback(
    staging_buffer: &Buffer,
    sender: &ContrastSender,
) -> bool {
    let slice = staging_buffer.slice(..);

    // Try to get mapped range
    if let Ok(data) = slice.get_mapped_range_mut() {
        let result: &GpuContrastResult = bytemuck::from_bytes(&data[..16]);

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

        drop(data);
        staging_buffer.unmap();
        return true;
    }

    false
}
```

**Step 2: Add receive system to systems.rs**

```rust
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
            1000.0 * std::f32::consts::TAU / 60.0 / 1e6
        };
        state.delta_omega = (state.omega.abs() * 0.01).max(1e-8);
        state.initialized = true;
        info!("CMax-SLAM initialized with omega={:.2e} rad/μs", state.omega);
    }

    // Check for new contrast values
    if let Ok(contrast) = receiver.rx.try_recv() {
        state.contrast = contrast.center;

        let v_c = contrast.center;
        let v_p = contrast.plus;
        let v_m = contrast.minus;

        // Skip if no data
        if v_c < 1.0 {
            return;
        }

        // Numerical gradient: dV/dω ≈ (V+ - V-) / 2δ
        let gradient = (v_p - v_m) / (2.0 * state.delta_omega);

        // Parabolic interpolation for optimal step
        let denominator = 2.0 * (v_p - 2.0 * v_c + v_m);

        let step = if denominator.abs() > 1e-6 {
            // Parabolic fit: jump to estimated peak
            let raw_step = -(v_p - v_m) / (2.0 * denominator) * state.delta_omega;
            raw_step
        } else {
            // Fallback: gradient ascent
            params.learning_rate * gradient.signum() * state.delta_omega
        };

        // Clamp step (max 5% change per frame)
        let max_step = state.delta_omega * 5.0;
        let clamped_step = step.clamp(-max_step, max_step);

        // Update omega
        state.omega += clamped_step;

        // Update delta for next frame (1% of omega)
        state.delta_omega = (state.omega.abs() * 0.01).max(1e-8);

        // Track convergence
        state.omega_history.push_back(state.omega);
        if state.omega_history.len() > 10 {
            state.omega_history.pop_front();

            let mean: f32 = state.omega_history.iter().sum::<f32>()
                / state.omega_history.len() as f32;
            let variance: f32 = state.omega_history.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / state.omega_history.len() as f32;

            state.converged = variance < (state.omega * 0.001).powi(2);
        }
    }
}
```

**Step 3: Verify it compiles**

Run: `cargo check 2>&1 | tail -15`
Expected: Compiles successfully

**Step 4: Commit**

```bash
git add src/cmax_slam/readback.rs src/cmax_slam/systems.rs
git commit -m "feat(cmax_slam): implement async readback and gradient optimizer"
```

---

## Task 8: Wire Up Channel in Plugin

**Files:**
- Modify: `src/cmax_slam/mod.rs` (CmaxSlamPlugin)

**Step 1: Read current plugin setup**

Run: `grep -n "impl Plugin" -A 50 src/cmax_slam/mod.rs | head -60`

**Step 2: Add channel setup to plugin**

In the `CmaxSlamPlugin` build method:

```rust
fn build(&self, app: &mut App) {
    // Create channel for contrast values
    let (sender, receiver) = create_contrast_channel();

    app.insert_resource(receiver)
        .init_resource::<CmaxSlamParams>()
        .init_resource::<CmaxSlamState>()
        .add_systems(Update, receive_contrast_results);

    // Add sender to render app
    let render_app = app.sub_app_mut(RenderApp);
    render_app.insert_resource(sender);

    // ... existing render world setup
}
```

**Step 3: Add system import**

Make sure `receive_contrast_results` is exported from systems.rs and imported in mod.rs.

**Step 4: Verify it compiles**

Run: `cargo check 2>&1 | tail -15`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add src/cmax_slam/mod.rs
git commit -m "feat(cmax_slam): wire up contrast channel in plugin"
```

---

## Task 9: Update UI to Show Optimizer State

**Files:**
- Modify: `src/compare/ui.rs`

**Step 1: Update CMax-SLAM panel to show optimizer status**

Find the CMax-SLAM section in `draw_edge_controls` and update:

```rust
// In the CMax-SLAM collapsing section
if let Some(state) = cmax_state.as_ref() {
    ui.separator();
    ui.heading("Optimizer");

    // Show current omega and estimated RPM
    let est_rpm = state.omega.abs() * 60.0 / std::f32::consts::TAU * 1e6;

    ui.horizontal(|ui| {
        ui.label("Est RPM:");
        ui.label(
            egui::RichText::new(format!("{:.1}", est_rpm))
                .size(20.0)
                .strong()
        );
    });

    ui.label(format!("Omega: {:.6e} rad/μs", state.omega));
    ui.label(format!("Delta: {:.6e}", state.delta_omega));
    ui.label(format!("Contrast: {:.0}", state.contrast));

    // Convergence status
    let conv_color = if state.converged {
        egui::Color32::GREEN
    } else {
        egui::Color32::YELLOW
    };
    ui.label(
        egui::RichText::new(format!("Converged: {}", state.converged))
            .color(conv_color)
    );

    // Initialization status
    ui.label(format!("Initialized: {}", state.initialized));
}
```

**Step 2: Verify it compiles**

Run: `cargo check 2>&1 | tail -10`
Expected: Compiles successfully

**Step 3: Commit**

```bash
git add src/compare/ui.rs
git commit -m "feat(ui): show CMax-SLAM optimizer state in compare_live"
```

---

## Task 10: Integration Test - Visual Verification

**Files:** None (manual test)

**Step 1: Generate test data**

Run: `cargo run --release --bin generate_synthetic -- --rpm 1200 --blades 3`

**Step 2: Run compare_live**

Run: `cargo run --release --bin compare_live -- data/synthetic/fan_test.dat`

**Step 3: Observe**

Expected behavior:
- RPM starts at ~1320 (1200 * 1.1 = 10% offset)
- RPM converges toward 1200 over 10-30 frames
- "Converged: true" appears when stable
- Contrast value updates each frame

**Step 4: Document results**

Note any issues for debugging.

---

## Task 11: Run Regression Tests

**Files:** None (test execution)

**Step 1: Run regression test**

Run: `cargo test --test regression_cmax_slam -- --ignored test_all_metrics_no_regression 2>&1`

**Step 2: Verify no regression**

Expected: All metrics pass baseline thresholds

**Step 3: Commit if all tests pass**

```bash
git add -A
git commit -m "feat(cmax_slam): complete Phase 1 auto-RPM implementation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Channel infrastructure | readback.rs, mod.rs |
| 2 | State/params updates | mod.rs |
| 3 | Reduction shader | cmax_slam_reduce.wgsl |
| 4 | Pipeline setup | pipeline.rs |
| 5 | Buffer creation | pipeline.rs, systems.rs |
| 6 | Render node update | pipeline.rs |
| 7 | Async readback | readback.rs, systems.rs |
| 8 | Plugin wiring | mod.rs |
| 9 | UI updates | ui.rs |
| 10 | Visual test | (manual) |
| 11 | Regression test | (test execution) |

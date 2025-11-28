# Phase 1: Auto-RPM Design

## Overview

Close the CMax-SLAM optimization loop to estimate RPM autonomously without ground truth.

**Branch:** `phase1-auto-rpm`
**Date:** 2025-11-28

## Architecture

```
Frame N:
┌─────────────────────────────────────────────────────────────────┐
│ GPU                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ cmax_warp    │───▶│ cmax_reduce  │───▶│ Contrast[3]  │       │
│  │ (3 IWEs)     │    │ (workgroup)  │    │ V_c, V+, V-  │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
└───────────────────────────────────────────────────│─────────────┘
                                                    │ async readback
                                              ┌─────▼─────┐
                                              │   CPU     │
                                              │ Gradient  │
                                              │ ω_new =   │
                                              │ ω + α*∇   │
                                              └─────┬─────┘
                                                    │ write uniform
Frame N+1:                                          ▼
┌────────────────────────────────────────────────────────────────┐
│ GPU uses updated ω in cmax_warp                                 │
└────────────────────────────────────────────────────────────────┘
```

## Design Decisions

### 1. Hybrid GPU/CPU Approach
- **GPU:** Warp events + reduce to 3 contrast values
- **CPU:** Gradient computation + omega update
- **Rationale:** Easy to debug optimizer on CPU, heavy lifting stays on GPU

### 2. Workgroup Reduction
- Each workgroup (256 threads) reduces locally using shared memory
- Thread 0 atomically adds to global result
- Reduces memory contention by 256x vs naive global atomics

### 3. Overflow Protection
- IWE values shifted right by 12 bits before squaring
- Max pixel value: 15, Max squared: 225
- Max total sum: ~207M (safe for u32)

### 4. Async Readback
- Uses `std::sync::mpsc::channel` to bridge render/main worlds
- Non-blocking `try_recv()` in main world
- 1-2 frame latency acceptable (fan inertia)

### 5. Cold Start
- Use ground truth + 10% offset when available (tests optimizer convergence)
- Fall back to 1000 RPM default otherwise

## Components

### New: `assets/shaders/cmax_slam_reduce.wgsl`

```wgsl
struct ContrastResult {
    sum_sq_center: atomic<u32>,
    sum_sq_plus: atomic<u32>,
    sum_sq_minus: atomic<u32>,
    pixel_count: atomic<u32>,
}

// Workgroup shared memory for local reduction
var<workgroup> local_sums: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;

    // Load and square (with overflow protection)
    var val_c = 0u;
    var val_p = 0u;
    var val_m = 0u;

    if idx < SLICE_SIZE {
        val_c = (iwe_buffer[idx] >> 12u);
        val_p = (iwe_buffer[idx + SLICE_SIZE] >> 12u);
        val_m = (iwe_buffer[idx + 2u * SLICE_SIZE] >> 12u);
    }

    // Square values
    let sq_c = val_c * val_c;
    let sq_p = val_p * val_p;
    let sq_m = val_m * val_m;

    // Workgroup reduction for each slice...
    // Thread 0 adds to global atomic
}
```

### New: `src/cmax_slam/readback.rs`

```rust
use std::sync::mpsc::{Sender, Receiver, channel};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct ContrastValues {
    pub center: f32,
    pub plus: f32,
    pub minus: f32,
}

// Main world resource
#[derive(Resource)]
pub struct ContrastReceiver {
    pub rx: Receiver<ContrastValues>,
}

// Render world resource
#[derive(Resource, Clone)]
pub struct ContrastSender {
    pub tx: Arc<Mutex<Sender<ContrastValues>>>,
}

pub fn setup_contrast_channel(app: &mut App) {
    let (tx, rx) = channel();
    app.insert_resource(ContrastReceiver { rx });
    // ContrastSender inserted in render app
}
```

### Modified: `src/cmax_slam/systems.rs`

```rust
pub fn receive_contrast_results(
    receiver: Res<ContrastReceiver>,
    mut state: ResMut<CmaxSlamState>,
    params: Res<CmaxSlamParams>,
) {
    if let Ok(contrast) = receiver.rx.try_recv() {
        // Store for UI display
        state.contrast = contrast.center;

        // Compute gradient
        let v_c = contrast.center;
        let v_p = contrast.plus;
        let v_m = contrast.minus;

        // Numerical gradient: dV/dω ≈ (V+ - V-) / 2δ
        let gradient = (v_p - v_m) / (2.0 * state.delta_omega);

        // Parabolic interpolation for step size
        let denominator = 2.0 * (v_p - 2.0 * v_c + v_m);

        let step = if denominator.abs() > 1e-6 {
            -gradient * state.delta_omega / denominator.abs().max(1e-6)
        } else {
            params.learning_rate * gradient
        };

        // Clamp step (max 5% change)
        let clamped_step = step.clamp(
            -state.delta_omega * 5.0,
            state.delta_omega * 5.0
        );

        state.omega += clamped_step;

        // Update delta for next frame
        state.delta_omega = (state.omega.abs() * 0.01).max(1e-8);

        // Track convergence
        update_convergence(&mut state);
    }
}

fn update_convergence(state: &mut CmaxSlamState) {
    state.omega_history.push_back(state.omega);
    if state.omega_history.len() > 10 {
        state.omega_history.pop_front();

        let mean = state.omega_history.iter().sum::<f32>()
                   / state.omega_history.len() as f32;
        let variance = state.omega_history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / state.omega_history.len() as f32;

        // Converged if variance < 0.1% of omega
        state.converged = variance < (state.omega * 0.001).powi(2);
    }
}
```

### Modified: `src/cmax_slam/pipeline.rs`

Add reduction pass after warp pass:

```rust
impl Node for CmaxSlamNode {
    fn run(&self, ...) {
        // 1. Clear buffers
        // 2. Run warp pass (existing)
        // 3. Run reduce pass (new)
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cmax_slam_reduce"),
                ..default()
            });
            pass.set_pipeline(&pipelines.reduce);
            pass.set_bind_group(0, &bind_groups.reduce, &[]);
            pass.dispatch_workgroups(
                (SLICE_SIZE + 255) / 256, 1, 1
            );
        }
        // 4. Trigger async readback
    }
}
```

## State Changes

```rust
pub struct CmaxSlamState {
    pub omega: f32,
    pub centroid: Vec2,
    pub contrast: f32,
    pub converged: bool,

    // New fields
    pub delta_omega: f32,
    pub learning_rate: f32,
    pub omega_history: VecDeque<f32>,
    pub initialized: bool,
}

pub struct CmaxSlamParams {
    pub enabled: bool,
    pub edge_weight: f32,

    // New fields
    pub learning_rate: f32,  // Default: 0.5
    pub use_ground_truth: bool,  // For A/B testing
}
```

## Success Criteria

1. **RPM converges** from +10% offset to within 1% of ground truth
2. **No regression** in existing metrics (run regression tests)
3. **Stable** omega after convergence (variance < 0.1%)
4. **Real-time** performance maintained (>30 FPS)

## Testing Plan

1. Run `cargo test --test regression_cmax_slam -- --ignored` before/after
2. Visual test with `compare_live`: watch RPM converge in UI
3. Headless test: modify `evaluate_cmax_slam` to use optimizer instead of GT

## Files Changed

| File | Change |
|------|--------|
| `assets/shaders/cmax_slam_reduce.wgsl` | **New** - Reduction shader |
| `src/cmax_slam/mod.rs` | Add new types, re-exports |
| `src/cmax_slam/readback.rs` | **New** - Channel infrastructure |
| `src/cmax_slam/pipeline.rs` | Add reduce pass, readback trigger |
| `src/cmax_slam/systems.rs` | Add optimizer system, cold start |
| `src/compare/ui.rs` | Display optimizer state in UI |

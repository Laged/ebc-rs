# Phase 2: Centroid Tracking Implementation Plan

**Date:** 2025-11-28
**Branch:** `phase2-centroid-tracking`
**Depends on:** Phase 1 (merged to main)

## Overview

Extend CMax-SLAM to jointly optimize angular velocity (ω) and rotation center (cx, cy), enabling tracking of a moving/drifting fan.

**Architecture:** Full Numerical (7 IWE slices)
**Success Criterion:** Maintain F1 > 0.99 while tracking drift

---

## Implementation Tasks

### Task 1: Extend GPU Buffer Layout

**Files:** `src/cmax_slam/resources.rs`, `src/cmax_slam/pipeline.rs`

**Changes:**

1. Update `GpuCmaxSlamParams`:
```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuCmaxSlamParams {
    pub centroid_x: f32,      // 0-3
    pub centroid_y: f32,      // 4-7
    pub t_ref: f32,           // 8-11
    pub omega: f32,           // 12-15
    pub delta_omega: f32,     // 16-19
    pub delta_pos: f32,       // 20-23  NEW
    pub edge_weight: f32,     // 24-27
    pub window_start: u32,    // 28-31
    pub window_end: u32,      // 32-35
    pub event_count: u32,     // 36-39
    pub _pad: [u32; 2],       // 40-47 (align to 48 bytes)
}
```

2. Update `GpuCmaxSlamResult` for 7 contrast values:
```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct GpuCmaxSlamResult {
    pub contrast_center: f32,
    pub contrast_omega_plus: f32,
    pub contrast_omega_minus: f32,
    pub contrast_cx_plus: f32,
    pub contrast_cx_minus: f32,
    pub contrast_cy_plus: f32,
    pub contrast_cy_minus: f32,
    pub pixel_count: f32,
}
```

3. Update IWE buffer size in `pipeline.rs`:
```rust
// Old: 3 slices
// let iwe_size = (3 * WIDTH * HEIGHT * 4) as u64;

// New: 7 slices
let iwe_size = (7 * WIDTH * HEIGHT * 4) as u64;
```

**Verification:** `cargo check` passes

---

### Task 2: Update Warp Shader

**File:** `assets/shaders/cmax_slam_warp.wgsl`

**Changes:**

1. Add slice index constants:
```wgsl
const IDX_CENTER: u32 = 0u;
const IDX_OMEGA_PLUS: u32 = 1u;
const IDX_OMEGA_MINUS: u32 = 2u;
const IDX_CX_PLUS: u32 = 3u;
const IDX_CX_MINUS: u32 = 4u;
const IDX_CY_PLUS: u32 = 5u;
const IDX_CY_MINUS: u32 = 6u;
```

2. Update params struct:
```wgsl
struct CmaxSlamParams {
    centroid_x: f32,
    centroid_y: f32,
    t_ref: f32,
    omega: f32,
    delta_omega: f32,
    delta_pos: f32,       // NEW
    edge_weight: f32,
    window_start: u32,
    window_end: u32,
    event_count: u32,
    _pad0: u32,
    _pad1: u32,
}
```

3. Modify `warp_event` to accept centroid as parameter:
```wgsl
fn warp_event(ex: f32, ey: f32, dt: f32, omega: f32, cx: f32, cy: f32) -> vec2<f32> {
    let dx = ex - cx;
    let dy = ey - cy;
    let r = sqrt(dx * dx + dy * dy);

    if r < 1.0 {
        return vec2<f32>(-1.0, -1.0);
    }

    let theta = atan2(dy, dx);
    let theta_warped = theta - omega * dt;

    return vec2<f32>(cx + r * cos(theta_warped), cy + r * sin(theta_warped));
}
```

4. Update main to compute 7 warps:
```wgsl
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ... event loading (unchanged) ...

    let cx = params.centroid_x;
    let cy = params.centroid_y;
    let omega = params.omega;
    let d_omega = params.delta_omega;
    let d_pos = params.delta_pos;

    // 7 warps
    let pos0 = warp_event(ex, ey, dt, omega, cx, cy);
    let pos1 = warp_event(ex, ey, dt, omega + d_omega, cx, cy);
    let pos2 = warp_event(ex, ey, dt, omega - d_omega, cx, cy);
    let pos3 = warp_event(ex, ey, dt, omega, cx + d_pos, cy);
    let pos4 = warp_event(ex, ey, dt, omega, cx - d_pos, cy);
    let pos5 = warp_event(ex, ey, dt, omega, cx, cy + d_pos);
    let pos6 = warp_event(ex, ey, dt, omega, cx, cy - d_pos);

    // Accumulate to 7 slices
    if pos0.x >= 0.0 { accumulate_bilinear(pos0, IDX_CENTER * SLICE_SIZE); }
    if pos1.x >= 0.0 { accumulate_bilinear(pos1, IDX_OMEGA_PLUS * SLICE_SIZE); }
    if pos2.x >= 0.0 { accumulate_bilinear(pos2, IDX_OMEGA_MINUS * SLICE_SIZE); }
    if pos3.x >= 0.0 { accumulate_bilinear(pos3, IDX_CX_PLUS * SLICE_SIZE); }
    if pos4.x >= 0.0 { accumulate_bilinear(pos4, IDX_CX_MINUS * SLICE_SIZE); }
    if pos5.x >= 0.0 { accumulate_bilinear(pos5, IDX_CY_PLUS * SLICE_SIZE); }
    if pos6.x >= 0.0 { accumulate_bilinear(pos6, IDX_CY_MINUS * SLICE_SIZE); }
}
```

**Verification:** Shader compiles (checked at runtime)

---

### Task 3: Update Reduction Shader

**File:** `assets/shaders/cmax_slam_reduce.wgsl`

**Changes:**

1. Update result struct:
```wgsl
struct ContrastResult {
    contrast_center: atomic<u32>,
    contrast_omega_plus: atomic<u32>,
    contrast_omega_minus: atomic<u32>,
    contrast_cx_plus: atomic<u32>,
    contrast_cx_minus: atomic<u32>,
    contrast_cy_plus: atomic<u32>,
    contrast_cy_minus: atomic<u32>,
    pixel_count: atomic<u32>,
}
```

2. Add 7 workgroup arrays:
```wgsl
var<workgroup> local_center: array<u32, 256>;
var<workgroup> local_omega_p: array<u32, 256>;
var<workgroup> local_omega_m: array<u32, 256>;
var<workgroup> local_cx_p: array<u32, 256>;
var<workgroup> local_cx_m: array<u32, 256>;
var<workgroup> local_cy_p: array<u32, 256>;
var<workgroup> local_cy_m: array<u32, 256>;
```

3. Update reduction loop to handle all 7 slices.

4. Update atomic adds for all 7 results.

**Verification:** Shader compiles, workgroup memory < 32KB

---

### Task 4: Update CPU State & Optimizer

**File:** `src/cmax_slam/resources.rs`, `src/cmax_slam/systems.rs`

**Changes to `CmaxSlamState`:**
```rust
pub struct CmaxSlamState {
    // Existing fields (unchanged)
    pub omega: f32,
    pub centroid: Vec2,
    pub contrast: f32,
    pub converged: bool,
    pub delta_omega: f32,
    pub omega_history: VecDeque<f32>,
    pub initialized: bool,
    pub ema_alpha: f32,
    pub max_step_fraction: f32,
    pub last_raw_step: f32,
    pub step_was_clamped: bool,

    // NEW fields
    pub delta_pos: f32,
    pub centroid_history: VecDeque<Vec2>,
    pub lr_omega: f32,
    pub lr_centroid: f32,
}

impl Default for CmaxSlamState {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            delta_pos: 3.0,
            centroid_history: VecDeque::with_capacity(16),
            lr_omega: 0.5,
            lr_centroid: 0.1,
        }
    }
}
```

**Changes to `ContrastValues`:**
```rust
#[derive(Clone)]
pub struct ContrastValues {
    pub center: f32,
    pub omega_plus: f32,
    pub omega_minus: f32,
    pub cx_plus: f32,
    pub cx_minus: f32,
    pub cy_plus: f32,
    pub cy_minus: f32,
    pub pixel_count: u32,
}
```

**Changes to `receive_contrast_results`:**
- Compute 3 gradients (ω, cx, cy)
- Apply parabolic step for ω
- Apply gradient ascent for centroid
- Apply EMA smoothing to all steps
- Clamp: ω by max_step_fraction, centroid by 5 pixels
- Update convergence tracking for both ω and centroid

**Verification:** `cargo test` passes

---

### Task 5: Update Readback

**File:** `src/cmax_slam/readback.rs`, `src/cmax_slam/pipeline.rs`

**Changes:**

1. Update `ContrastValues` to include all 7 values (see Task 4)

2. Update readback buffer mapping to read 8 floats (7 + pixel_count)

3. Update sender to pack all 7 contrast values

**Verification:** Readback values are non-zero in logs

---

### Task 6: Generate Synthetic Test Data

**File:** `src/bin/generate_synthetic.rs`

**Changes:**

1. Add motion configuration:
```rust
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CentroidMotion {
    #[serde(rename = "static")]
    Static,
    #[serde(rename = "linear_drift")]
    LinearDrift { velocity_x: f32, velocity_y: f32 },
    #[serde(rename = "oscillation")]
    Oscillation { amplitude_x: f32, amplitude_y: f32, frequency_hz: f32, phase_offset: f32 },
}
```

2. Update ground truth JSON schema to include motion

3. Modify event generation to compute centroid at each timestamp

4. Add CLI flags: `--drift-vx`, `--drift-vy`, `--oscillate`

**Generate datasets:**
```bash
# Linear drift (50 px/s horizontal, 20 px/s vertical)
cargo run --bin generate_synthetic -- \
    --output data/synthetic/fan_drift.dat \
    --rpm 1200 --blades 3 --duration 2.0 \
    --drift-vx 50 --drift-vy 20

# Oscillation (30px amplitude, 0.5Hz)
cargo run --bin generate_synthetic -- \
    --output data/synthetic/fan_oscillate.dat \
    --rpm 1200 --blades 3 --duration 2.0 \
    --oscillate --osc-amp 30 --osc-freq 0.5
```

**Verification:** Generated files exist with correct GT JSON

---

### Task 7: Update Evaluation Binary

**File:** `src/bin/evaluate_cmax_slam.rs`

**Changes:**

1. Parse motion config from ground truth JSON

2. Add `centroid_at_time(t_us: f32) -> Vec2` helper

3. Update metrics to include centroid error

4. Add test matrix output (static, drift, oscillate)

**Verification:** Evaluation passes on static data (regression)

---

### Task 8: Update UI Panel

**File:** `src/compare/ui.rs`

**Changes:**

1. Display centroid position: `Center: (690.2, 372.5)`

2. Display centroid error (if GT available): `Δpos: 2.3 px`

3. Display estimated velocity: `Vel: (48, 19) px/s`

4. Color-code centroid error (green < 2px, yellow < 5px, red >= 5px)

**Verification:** Visual confirmation in `compare_live`

---

### Task 9: Update Ground Truth Config

**File:** `src/ground_truth.rs`

**Changes:**

1. Add motion field to `GroundTruthConfig`:
```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct GroundTruthConfig {
    pub rpm: f32,
    pub blades: u32,
    pub center_x: f32,
    pub center_y: f32,
    #[serde(default)]
    pub motion: Option<CentroidMotion>,
}
```

2. Add `centroid_at_time(&self, t_us: f32) -> Vec2` method

**Verification:** Existing GT files still parse correctly

---

## Task Dependency Graph

```
Task 1 (Buffers)
    ↓
Task 2 (Warp Shader) ──┬── Task 3 (Reduce Shader)
                       ↓
              Task 5 (Readback)
                       ↓
              Task 4 (Optimizer)
                       ↓
              Task 8 (UI)

Task 6 (Synthetic Data) ─── Task 9 (GT Config)
                                  ↓
                          Task 7 (Evaluation)
```

**Parallelizable:**
- Tasks 2 & 3 (shaders)
- Tasks 6 & 9 (data generation)

---

## Testing Plan

### Unit Tests
- [ ] `GpuCmaxSlamParams` size is 48 bytes
- [ ] `GpuCmaxSlamResult` size is 32 bytes
- [ ] `centroid_at_time` returns correct values for all motion types

### Integration Tests
- [ ] Static fan (`fan_test.dat`): F1 > 0.99, RPM error < 1%
- [ ] Drifting fan (`fan_drift.dat`): F1 > 0.99, centroid error < 5px
- [ ] Oscillating fan (`fan_oscillate.dat`): F1 > 0.99, centroid error < 5px

### Visual Tests
- [ ] `compare_live fan_test.dat`: RPM converges, centroid stable
- [ ] `compare_live fan_drift.dat`: Centroid tracks drift
- [ ] `compare_live fan_oscillate.dat`: Centroid follows oscillation

---

## Estimated Complexity

| Task | Complexity | Lines Changed |
|------|------------|---------------|
| 1. Buffers | Low | ~30 |
| 2. Warp Shader | Medium | ~50 |
| 3. Reduce Shader | Medium | ~60 |
| 4. Optimizer | Medium | ~80 |
| 5. Readback | Low | ~20 |
| 6. Synthetic Data | Medium | ~100 |
| 7. Evaluation | Low | ~40 |
| 8. UI | Low | ~30 |
| 9. GT Config | Low | ~30 |
| **Total** | | **~440 lines** |

---

## Rollback Plan

If Phase 2 causes regressions:
1. The optimizer can fall back to ω-only mode by setting `lr_centroid = 0`
2. Static fan tests must pass before merging
3. Branch can be reverted without affecting main

---

## Success Criteria Checklist

- [ ] Static fan: F1 > 0.99, RPM error < 1% (regression)
- [ ] Drifting fan: F1 > 0.99, centroid error < 5px
- [ ] Oscillating fan: F1 > 0.99, centroid error < 5px
- [ ] Real-time performance: > 30 FPS
- [ ] No memory leaks (7-slice buffer properly sized)

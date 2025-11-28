# Contrast Maximization for RPM Estimation

**Date:** 2025-11-26
**Status:** Design Complete
**Replaces:** Canny edge detector in compare_live Q3

## Overview

Implement Event Contrast Maximization (CM) to estimate fan RPM directly from event data. The CM algorithm warps events by candidate angular velocities and maximizes image sharpness to find the true rotation speed.

**Reference:** Karmokar et al., "Secrets of Edge-Informed Contrast Maximization for Event-Based Vision", WACV 2025

## Requirements

| Requirement | Decision |
|-------------|----------|
| Display | RPM estimation + deblurred IWE visualization |
| RPM range | Auto-detect from event statistics |
| Rotation center | Use existing centroid tracking |
| Architecture | Single-pass GPU polar warp (Approach A) |

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXISTING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  EventData ──► Accumulation ──► Preprocess ──► Sobel/LoG        │
│     │              │                              │              │
│     │              ▼                              ▼              │
│     │         SurfaceImage                   EdgeImages          │
└─────┼───────────────────────────────────────────────────────────┘
      │
      │  Pass raw events directly
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CM PIPELINE (replaces Canny)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌───────────────────┐     ┌────────────┐ │
│  │ GpuEventBuf  │────►│  cm_warp.wgsl     │────►│ IWE Array  │ │
│  │ (existing)   │     │  polar transform  │     │ [N_OMEGA]  │ │
│  └──────────────┘     │  + accumulate     │     └─────┬──────┘ │
│         │             └───────────────────┘           │        │
│         │                                             ▼        │
│  ┌──────┴──────┐     ┌───────────────────┐     ┌────────────┐ │
│  │ CentroidBuf │────►│  cm_contrast.wgsl │────►│ ContrastBuf│ │
│  │ (from CPU)  │     │  gradient-based   │     │ [N_OMEGA]  │ │
│  └─────────────┘     └───────────────────┘     └─────┬──────┘ │
│                                                       │        │
│                      ┌───────────────────┐           │        │
│                      │  cm_select.wgsl   │◄──────────┘        │
│                      │  find best ω      │                    │
│                      └─────────┬─────────┘                    │
│                                │                              │
│                                ▼                              │
│                      ┌───────────────────┐                    │
│                      │ Best IWE → Output │ → CmImage slot     │
│                      │ Best ω → RPM      │ → UI overlay       │
│                      └───────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### Render Graph Position

```
Accumulation → Preprocess → ┬─ Sobel ──┬─► Composite
                            ├─ CM ─────┤
                            └─ LoG ────┘
```

## GPU Shaders

### cm_warp.wgsl - Polar Warp + IWE Construction

```wgsl
struct CmParams {
    centroid: vec2<f32>,      // rotation center (from CPU)
    t_ref: f32,               // reference time (window center)
    omega_min: f32,           // min angular velocity (rad/μs)
    omega_step: f32,          // step between candidates
    n_omega: u32,             // number of candidates (e.g., 64)
    window_start: u32,        // time window
    window_end: u32,
    event_count: u32,
}

@group(0) @binding(0) var<storage, read> events: array<GpuEvent>;
@group(0) @binding(1) var<uniform> params: CmParams;
@group(0) @binding(2) var iwe_array: texture_storage_2d_array<r32float, write>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let event_idx = gid.x;
    if (event_idx >= params.event_count) { return; }

    let event = events[event_idx];
    if (event.timestamp < params.window_start || event.timestamp > params.window_end) { return; }

    // Convert to polar around centroid
    let dx = f32(event.x) - params.centroid.x;
    let dy = f32(event.y) - params.centroid.y;
    let r = sqrt(dx*dx + dy*dy);
    let theta = atan2(dy, dx);
    let dt = f32(event.timestamp) - params.t_ref;

    // For each omega candidate, warp and accumulate
    for (var i = 0u; i < params.n_omega; i++) {
        let omega = params.omega_min + f32(i) * params.omega_step;
        let theta_warped = theta - omega * dt;

        // Convert back to Cartesian
        let x_warped = params.centroid.x + r * cos(theta_warped);
        let y_warped = params.centroid.y + r * sin(theta_warped);

        let ix = i32(x_warped);
        let iy = i32(y_warped);
        if (ix >= 0 && ix < 1280 && iy >= 0 && iy < 720) {
            textureStore(iwe_array, vec2<i32>(ix, iy), i, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        }
    }
}
```

### cm_contrast.wgsl - Gradient-Based Contrast

Computes sum of squared Sobel gradient magnitudes for each IWE slice.

### cm_select.wgsl - Find Best ω + Copy IWE

Parallel reduction to find argmax of contrast array, copies best IWE to output texture.

## Memory Requirements

| Resource | Size | Notes |
|----------|------|-------|
| IWE array (N=64) | 235 MB | 1280×720×64×4 bytes |
| IWE array (N=32) | 118 MB | Reduced option |
| Contrast buffer | 256 B | 64×4 bytes |

## File Structure

### New Files

| File | Purpose |
|------|---------|
| `src/cm/mod.rs` | `CmPlugin` - top-level module |
| `src/cm/pipeline.rs` | `CmPipeline`, `CmNode`, render graph |
| `src/cm/resources.rs` | `CmImage`, `CmParams`, `CmResult` |
| `src/cm/systems.rs` | `prepare_cm`, `extract_cm_params` |
| `assets/shaders/cm_warp.wgsl` | Event warping + IWE |
| `assets/shaders/cm_contrast.wgsl` | Gradient contrast |
| `assets/shaders/cm_select.wgsl` | Argmax + copy |

### Modified Files

| File | Changes |
|------|---------|
| `src/lib.rs` | Add `pub mod cm;` |
| `src/bin/compare_live.rs` | Add `CmPlugin`, remove Canny |
| `src/compare/composite.rs` | Replace `CannyImage` → `CmImage` |
| `src/compare/ui.rs` | Replace CANNY panel → CM + RPM |
| `src/compare/mod.rs` | Update `AllDetectorMetrics` |

## Resource Types

```rust
// src/cm/resources.rs

/// CM output image (replaces Canny in Q3)
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct CmImage {
    pub handle: Handle<Image>,
}

/// CM parameters passed to GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
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
#[derive(Resource, Default)]
pub struct CmResult {
    pub best_omega: f32,     // rad/μs
    pub best_contrast: f32,
    pub rpm: f32,            // omega * 60 / (2π) * 1e6
}
```

## Auto-Detect RPM Range

```rust
fn estimate_omega_range(events: &[GpuEvent], window: (u32, u32)) -> (f32, f32) {
    let count = events.iter()
        .filter(|e| e.timestamp >= window.0 && e.timestamp <= window.1)
        .count();

    // Heuristic: event rate correlates with rotation speed
    let event_rate = count as f32 / (window.1 - window.0) as f32 * 1000.0;
    let estimated_rpm = event_rate * 0.5;
    let omega_center = estimated_rpm * std::f32::consts::TAU / 60.0 / 1e6;

    // Search ±50% around estimate
    (omega_center * 0.5, omega_center * 1.5)
}
```

## UI Design

Q3 panel displays:
- Large RPM value (24pt font)
- Quality indicator (●●● / ●●○ / ●○○)
- Debug omega value

Settings panel:
- Search resolution slider (N_OMEGA: 16-128)
- Remove Canny threshold sliders

## Error Handling

| Scenario | Handling |
|----------|----------|
| No events in window | Skip CM, display "N/A" |
| Centroid outside image | Clamp to bounds, warn |
| All contrasts equal | Low confidence, use previous RPM |
| Very sparse events | Fallback to centroid only |
| IWE allocation fails | Reduce N_OMEGA, log warning |

## Temporal Smoothing

```rust
impl CmResult {
    pub fn update_with_smoothing(&mut self, new_omega: f32, alpha: f32) {
        if self.best_omega > 0.0 {
            self.best_omega = alpha * new_omega + (1.0 - alpha) * self.best_omega;
        } else {
            self.best_omega = new_omega;
        }
        self.rpm = self.best_omega * 60.0 / std::f32::consts::TAU * 1e6;
    }
}
```

## Performance Safeguards

```rust
const MAX_IWE_MEMORY_MB: usize = 256;
const IWE_SIZE_BYTES: usize = 1280 * 720 * 4;

fn safe_n_omega(requested: u32) -> u32 {
    let max_n = (MAX_IWE_MEMORY_MB * 1024 * 1024 / IWE_SIZE_BYTES) as u32;
    requested.min(max_n)
}
```

## Implementation Tasks

1. Create `src/cm/` module structure
2. Implement `CmPlugin` with resource initialization
3. Write `cm_warp.wgsl` shader
4. Write `cm_contrast.wgsl` shader
5. Write `cm_select.wgsl` shader
6. Implement `CmPipeline` and `CmNode`
7. Add render graph integration
8. Update composite to use `CmImage`
9. Update UI with CM panel and RPM display
10. Add auto-detect omega range logic
11. Add temporal smoothing
12. Test with synthetic and real fan data

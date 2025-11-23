# Fan Visualization Accuracy Design

**Date**: 2025-11-23
**Branch**: `fan-rpm-01M4hd29i4StANLrkhbp7pJo`
**Status**: Design Approved

## Overview

This design addresses visualization accuracy issues in the fan RPM tracking feature:
1. **Fan border inaccuracy**: Circle doesn't fit the actual fan activation area
2. **Blade sync mismatch**: Overlay blades don't align with visible event boundaries
3. **Linux Vulkan warnings**: Missing Wayland dependencies
4. **Data loading crashes**: App hangs when file not found
5. **Code quality**: Rust compiler warnings

## Goals

**Primary**: Fix visualization accuracy (blade sync + border fit) using GPU-based event analysis
**Secondary**: Clean up bugs, warnings, and Linux compatibility

## Architecture

### GPU-Centric Approach

All analysis performed on GPU via compute shaders, following the existing architecture pattern:
- Leverage existing GPU event buffer and surface texture
- Use async buffer mapping with channels for GPU→CPU communication
- Maintain performance with large event counts

### New Compute Pipelines

#### 1. Radial Analysis Pipeline
**Purpose**: Detect actual fan radius from accumulated intensity falloff

**Shader**: `assets/shaders/radial_profile.wgsl`
- Input: Surface buffer texture (1280×720), centroid uniform
- Process: Build radial histogram by binning pixel intensities by distance from centroid
- Output: Radial bins array (400 bins, 0-400px radius)
- CPU Analysis: Find 95th percentile of cumulative intensity → fan radius

**Integration**:
- Render node: `RadialProfileNode` (runs after `EventAccumulationNode`)
- Resources: `RadialProfilePipeline`, `RadialGpuResources`
- Data flow: GPU histogram → staging buffer → channel → `FanAnalysis.fan_radius`

#### 2. Angular Histogram Pipeline
**Purpose**: Detect actual blade positions from event distribution

**Shader**: `assets/shaders/angular_histogram.wgsl`
- Input: Event buffer, centroid + radius uniform, time window
- Process: Build polar histogram of events near detected radius
  - Calculate angle: `atan2(y - cy, x - cx)`
  - Filter events: only count if `|distance - radius| < tolerance`
  - Bin into 360 bins (1° resolution)
- Output: Angular bins array (360 u32 counts)
- CPU Analysis: Peak detection to find blade angles
  - Smooth histogram (3-bin window)
  - Find local maxima
  - Select top `blade_count` peaks
  - Store as `Vec<f32>` of angles

**Integration**:
- Render node: `AngularHistogramNode` (runs after `CentroidNode`)
- Resources: `AngularHistogramPipeline`, `AngularGpuResources`
- Data flow: GPU histogram → staging buffer → channel → peak detection → `FanAnalysis.blade_angles`

### Render Graph Order

```
CentroidNode
  ↓ (computes centroid position)
EventAccumulationNode
  ↓ (accumulates events to surface buffer)
RadialProfileNode
  ↓ (analyzes surface for radius)
AngularHistogramNode
  ↓ (analyzes events for blade angles)
CameraDriverLabel
```

**Rationale**:
- Centroid must run first (needed by both other pipelines)
- Accumulation must run before radial analysis (provides surface texture)
- Angular analysis can run after centroid (only needs events + centroid + radius)
- All GPU→CPU readback happens asynchronously, no blocking

## Data Structures

### FanAnalysis Resource Changes

```rust
#[derive(Resource, Clone, ExtractResource)]
pub struct FanAnalysis {
    // Existing fields
    pub is_tracking: bool,
    pub show_borders: bool,
    pub current_rpm: f32,          // Still simulated for now
    pub blade_count: u32,
    pub centroid: Vec2,
    pub tip_velocity: f32,

    // Modified fields
    pub fan_radius: f32,           // FROM: bounding box → TO: radial analysis

    // New fields
    pub blade_angles: Vec<f32>,    // Detected angles from histogram

    // Removed fields
    // pub current_angle: f32,      // No longer needed for visualization
}
```

### GPU Buffers

**RadialResult**:
```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RadialResult {
    radial_bins: [u32; 400],  // Histogram bins
    total_intensity: u32,
    detected_radius: f32,
    _padding: [u32; 2],
}
```

**AngularHistogram**:
```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AngularHistogram {
    bins: [u32; 360],  // 1 degree per bin
}
```

## Component Changes

### src/gizmos.rs
**Current**: Uses `current_angle + i * angle_per_blade` (calculated from simulated RPM)
**New**: Uses `blade_angles[i]` (detected from events)

```rust
// Before
for i in 0..analysis.blade_count {
    let angle = analysis.current_angle + (i as f32 * angle_per_blade);
    // ...
}

// After
for &blade_angle in &analysis.blade_angles {
    let dx = blade_angle.cos() * analysis.fan_radius;
    let dy = blade_angle.sin() * analysis.fan_radius;
    // ... draw blade at detected angle
}
```

### src/analysis.rs
- Add `RadialProfilePipeline`, `RadialGpuResources`
- Add `AngularHistogramPipeline`, `AngularGpuResources`
- Add `RadialProfileNode`, `AngularHistogramNode`
- Add channels and update systems for both pipelines (follow centroid pattern)
- Implement CPU-side peak detection algorithm
- Remove `update_rotation_angle` system (no longer needed)

### src/render.rs
**Data Loading Error Handling**:
```rust
Err(e) => {
    error!("Failed to load data from {}: {:?}", path, e);
    commands.insert_resource(EventData { events: Vec::new() });
}
```
Add UI indicator when `event_data.events.is_empty()`.

### flake.nix
Add to `linuxDeps`:
```nix
wayland-protocols  # Provides missing Wayland symbols for Mesa drivers
```

### Code Quality
- Run `cargo clippy --fix --allow-dirty`
- Remove `mut` from `src/analysis.rs:520`
- Mark or remove unused constants at `src/analysis.rs:18,22`

## Performance Considerations

**Radial Analysis**:
- O(width × height) = ~920k pixels
- Single pass with atomic operations
- Runs on GPU, minimal CPU overhead

**Angular Histogram**:
- O(events_in_window), typically 10k-100k events
- Single pass with atomic operations
- Filter reduces work (only events near radius)

**GPU→CPU Communication**:
- Async buffer mapping (non-blocking)
- Small data transfer (400 u32s + 360 u32s per frame)
- Uses existing channel pattern

**Total Overhead**: ~1-2ms per frame on modern GPU

## Testing Strategy

1. **Visual Validation**:
   - Load `data/fan/fan_const_rpm.dat`
   - Enable "Show Blade Borders"
   - Verify blue circle tightly fits fan activation area
   - Verify green blade lines align with red/blue event boundaries

2. **Radius Accuracy**:
   - Add debug overlay showing radial histogram
   - Verify 95th percentile selection filters outliers
   - Test with different fan speeds

3. **Blade Detection**:
   - Add debug overlay showing angular histogram
   - Verify peaks correspond to blade count
   - Test with 2, 3, 4 blade configurations

4. **Error Handling**:
   - Test with missing data file
   - Verify UI shows error message
   - Verify app doesn't crash

5. **Linux Compatibility**:
   - Test on NixOS 25.11
   - Verify no Vulkan warnings
   - Verify rendering works correctly

## Future Enhancements (Out of Scope)

- Full CMax optimization for real RPM calculation
- Automatic blade count detection
- Multi-fan tracking
- Export of detected parameters

## References

- Original plan: `docs/research/fan-border-rpm-tracking.md`
- Existing centroid pipeline: `src/analysis.rs:252-333`
- Accumulation shader: `assets/shaders/accumulation.wgsl`

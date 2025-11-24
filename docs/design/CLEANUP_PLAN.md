# Repository Cleanup Plan: Edge Detection Focus

## Objective
Clean up the repository to remove all old fan detection code, keeping only the edge detection MVP implementation and its dependencies.

## Current State Analysis

### Source Files Status

#### KEEP (Active Code)
- `src/main.rs` - Entry point using MvpPlugin
- `src/lib.rs` - Library root (needs cleanup of exports)
- `src/loader.rs` - DatLoader used by MVP to load event data
- `src/mvp/mod.rs` - MVP module root
- `src/mvp/gpu.rs` - GPU compute pipeline for edge detection
- `src/mvp/playback.rs` - Playback controls
- `src/mvp/render.rs` - Rendering and UI for edge detection

#### DELETE (Old Fan Detection Code)
- `src/analysis.rs` - Fan RPM analysis, not used by edge detection
- `src/gizmos.rs` - Old visualization gizmos, superseded by MVP
- `src/gpu.rs` - Old GPU code, superseded by `src/mvp/gpu.rs`
- `src/plugins.rs` - Old plugin system, superseded by MVP
- `src/render.rs` - Old render code, superseded by `src/mvp/render.rs`
- `src/synthesis.rs` - Fan synthesis utilities, not used by edge detection
- `src/bin/generate_synthetic_fan.rs` - Fan generator binary, not needed

### Shader Files Status

#### KEEP (Used by MVP)
- `assets/shaders/accumulation.wgsl` - Event accumulation compute shader
- `assets/shaders/spatial_gradient.wgsl` - Sobel edge detection compute shader
- `assets/shaders/visualizer.wgsl` - Fragment shader for visualization

#### DELETE (Old Fan Detection Shaders)
- `assets/shaders/angular_histogram.wgsl` - Fan blade detection
- `assets/shaders/centroid.wgsl` - Fan center detection
- `assets/shaders/cmax_optimization.wgsl` - Fan optimization
- `assets/shaders/radial_profile.wgsl` - Fan radial analysis

### Dependencies
MVP only depends on:
- `loader::DatLoader` - Loads .dat event files
- `EventFilePath` - Resource holding path to event file
- Bevy engine ecosystem

## Cleanup Steps

### Step 1: Delete Old Source Files
```bash
git rm src/analysis.rs
git rm src/gizmos.rs
git rm src/gpu.rs
git rm src/plugins.rs
git rm src/render.rs
git rm src/synthesis.rs
git rm src/bin/generate_synthetic_fan.rs
```

### Step 2: Delete Old Shader Files
```bash
git rm assets/shaders/angular_histogram.wgsl
git rm assets/shaders/centroid.wgsl
git rm assets/shaders/cmax_optimization.wgsl
git rm assets/shaders/radial_profile.wgsl
```

### Step 3: Clean Up lib.rs
Remove old module declarations and exports, keep only:
```rust
use bevy::prelude::*;

pub mod loader;
pub mod mvp;

#[derive(Resource, Clone)]
pub struct EventFilePath(pub String);

impl Default for EventFilePath {
    fn default() -> Self {
        Self("data/fan/fan_const_rpm.dat".to_string())
    }
}

pub use loader::DatLoader;
```

### Step 4: Update Cargo.toml
Remove `generate_synthetic_fan` binary from `[[bin]]` section if present.

### Step 5: Verify Build
```bash
cargo build --bin ebc-rs
cargo clippy
cargo run --bin ebc-rs
```

### Step 6: Commit Changes
```bash
git add -A
git commit -m "chore: remove old fan detection code, keep only edge detection MVP"
```

## Impact Analysis

### What Will Break
- Any external code importing old modules (`analysis`, `gpu`, `render`, etc.) will break
- The `generate_synthetic_fan` binary will no longer be available
- Any documentation referencing old fan detection features will be outdated

### What Will Still Work
- Main edge detection application (`cargo run --bin ebc-rs`)
- Event loading from .dat files
- All edge detection features:
  - Real-time event visualization
  - Sobel edge detection with filters
  - Playback controls
  - Keyboard filter toggles (1/2/3/4)

## Benefits

1. **Clarity** - Repository clearly focuses on edge detection
2. **Maintainability** - Less code to maintain
3. **Performance** - Faster builds, smaller binary
4. **Documentation** - Easier to understand what the project does

## Risks

### Low Risk
- Edge detection MVP is self-contained in `src/mvp/`
- Only dependency is `DatLoader` which we're keeping
- Can easily revert if needed (all changes in git history)

### Mitigation
- Create this cleanup as separate commit(s)
- Test thoroughly after cleanup
- Tag current state before cleanup if desired

## File Size Comparison

### Before Cleanup
- Source files: 10 .rs files
- Shader files: 7 .wgsl files

### After Cleanup
- Source files: 6 .rs files (40% reduction)
- Shader files: 3 .wgsl files (57% reduction)

## Execution Checklist

- [ ] Create backup tag (optional)
- [ ] Delete old source files
- [ ] Delete old shader files
- [ ] Clean up lib.rs
- [ ] Update Cargo.toml
- [ ] Run cargo build
- [ ] Run cargo clippy
- [ ] Test application
- [ ] Commit changes
- [ ] Update README if needed

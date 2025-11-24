# Repository Refactoring Plan: Flatten MVP Structure

## Objective
Remove the `mvp/` namespace and refactor code into a clean, flat structure organized by concern. Delete all old fan detection code.

## Target Structure

```
src/
├── main.rs              # Entry point
├── lib.rs               # Library root with clean exports
├── loader.rs            # Event data loading (KEEP - already good)
├── edge_detection.rs    # Edge detection plugin and core logic
├── event_renderer.rs    # Event visualization and rendering
├── playback.rs          # Playback controls (standalone)
└── gpu/                 # GPU compute infrastructure
    ├── mod.rs           # GPU module root
    ├── types.rs         # GPU data types (GpuEvent, GpuEdgeParams)
    ├── resources.rs     # Bevy resources (EventData, EdgeParams, etc.)
    ├── accumulation.rs  # Event accumulation pipeline
    └── gradient.rs      # Spatial gradient pipeline

assets/shaders/
├── accumulation.wgsl    # Event accumulation compute shader
├── spatial_gradient.wgsl # Sobel edge detection compute shader
└── visualizer.wgsl      # Fragment shader for visualization
```

## Current State (MVP)

```
src/mvp/
├── mod.rs          → Plugin registration
├── gpu.rs          → ALL GPU code (522 lines)
├── playback.rs     → Playback controls
└── render.rs       → UI + rendering + material setup
```

## Refactoring Strategy

### Phase 1: Create New Structure

1. **Create `src/gpu/` module** - Split `mvp/gpu.rs` into logical pieces:
   - `gpu/types.rs` - Pod/Zeroable types (GpuEvent, GpuEdgeParams)
   - `gpu/resources.rs` - Bevy resources (EventData, EdgeParams, SurfaceImage, etc.)
   - `gpu/accumulation.rs` - Event accumulation pipeline
   - `gpu/gradient.rs` - Gradient computation pipeline
   - `gpu/mod.rs` - Module root, re-exports

2. **Create `src/event_renderer.rs`** - From `mvp/render.rs`:
   - EventMaterial setup
   - Scene setup (camera, textures, quad)
   - UI system (egui controls)
   - Material parameter updates
   - EventRendererPlugin

3. **Create `src/edge_detection.rs`** - Coordinate everything:
   - EdgeDetectionPlugin (top-level plugin)
   - Orchestrates GPU pipelines
   - Registers render graph nodes

4. **Move `src/mvp/playback.rs` → `src/playback.rs`** - Already standalone

### Phase 2: Delete Old Code

Delete these files:
- `src/analysis.rs`
- `src/gizmos.rs`
- `src/gpu.rs` (old one, not the new gpu/ module)
- `src/plugins.rs`
- `src/render.rs`
- `src/synthesis.rs`
- `src/bin/generate_synthetic_fan.rs`
- `src/mvp/` (entire directory after content moved)

Delete these shaders:
- `assets/shaders/angular_histogram.wgsl`
- `assets/shaders/centroid.wgsl`
- `assets/shaders/cmax_optimization.wgsl`
- `assets/shaders/radial_profile.wgsl`

### Phase 3: Update Imports

**src/main.rs:**
```rust
use bevy::prelude::*;
use ebc_rs::edge_detection::EdgeDetectionPlugin;
use ebc_rs::EventFilePath;

fn main() {
    let data_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/fan/fan_const_rpm.dat".to_string());

    App::new()
        .insert_resource(EventFilePath(data_path))
        .add_plugins(DefaultPlugins)
        .add_plugins(EdgeDetectionPlugin)
        .run();
}
```

**src/lib.rs:**
```rust
use bevy::prelude::*;

pub mod edge_detection;
pub mod event_renderer;
pub mod gpu;
pub mod loader;
pub mod playback;

#[derive(Resource, Clone)]
pub struct EventFilePath(pub String);

impl Default for EventFilePath {
    fn default() -> Self {
        Self("data/fan/fan_const_rpm.dat".to_string())
    }
}

pub use loader::DatLoader;
```

## Detailed File Mappings

### src/gpu/types.rs (NEW)
Source: `src/mvp/gpu.rs` lines 14-33
```rust
// GpuEdgeParams
// GpuEvent
```

### src/gpu/resources.rs (NEW)
Source: `src/mvp/gpu.rs` lines 35-100
```rust
// EventData
// SurfaceImage
// GradientImage
// GpuEventBuffer
// EdgeParams (+ ExtractResource impl + Default)
```

### src/gpu/accumulation.rs (NEW)
Source: `src/mvp/gpu.rs` lines 102-352
```rust
// EventComputePipeline
// EventBindGroup
// prepare_events()
// queue_bind_group()
// EventLabel
// EventAccumulationNode
```

### src/gpu/gradient.rs (NEW)
Source: `src/mvp/gpu.rs` lines 354-521
```rust
// GradientPipeline
// GradientBindGroup
// EdgeParamsBuffer
// prepare_gradient()
// GradientLabel
// GradientNode
```

### src/event_renderer.rs (NEW)
Source: `src/mvp/render.rs` entire file
```rust
// EventMaterial
// EventParams
// CurrentMaterialHandle
// load_data()
// setup_scene()
// update_material_params()
// ui_system()
// EventRendererPlugin
```

### src/edge_detection.rs (NEW)
Source: `src/mvp/mod.rs` + orchestration
```rust
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::{RenderApp, Render, RenderSystems, ExtractSchedule, Extract};
use bevy::render::render_graph::RenderGraph;

pub struct EdgeDetectionPlugin;

impl Plugin for EdgeDetectionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<crate::gpu::SurfaceImage>()
            .init_resource::<crate::gpu::GradientImage>()
            .init_resource::<crate::playback::PlaybackState>()
            .init_resource::<crate::gpu::EdgeParams>()
            .add_plugins(ExtractResourcePlugin::<crate::gpu::EventData>::default())
            .add_plugins(ExtractResourcePlugin::<crate::gpu::SurfaceImage>::default())
            .add_plugins(ExtractResourcePlugin::<crate::gpu::GradientImage>::default())
            .add_plugins(ExtractResourcePlugin::<crate::playback::PlaybackState>::default())
            .add_plugins(crate::event_renderer::EventRendererPlugin);
    }

    fn finish(&self, app: &mut App) {
        use crate::gpu::*;

        fn extract_edge_params(
            mut commands: Commands,
            edge_params: Extract<Res<EdgeParams>>,
        ) {
            commands.insert_resource(edge_params.clone());
        }

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<EventComputePipeline>()
            .init_resource::<GradientPipeline>()
            .init_resource::<GpuEventBuffer>()
            .add_systems(ExtractSchedule, extract_edge_params)
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

### src/playback.rs (MOVE)
Source: `src/mvp/playback.rs` → `src/playback.rs`
No changes needed, just move.

## Benefits of New Structure

1. **Clear Separation of Concerns**
   - GPU code isolated in `gpu/` module
   - Rendering separated from compute
   - Playback is standalone

2. **Better Discoverability**
   - No more "mvp" ambiguity
   - Each file has clear purpose
   - Logical grouping (gpu subfolder)

3. **Easier to Extend**
   - Add new GPU pipelines in `gpu/`
   - Add new renderers alongside `event_renderer.rs`
   - Add new plugins at top level

4. **Reduced Coupling**
   - GPU types reusable
   - Renderer doesn't know about GPU internals
   - Plugin composition more flexible

## Risk Assessment

### Low Risk
- All code is being moved, not rewritten
- Only changing imports and file locations
- Can test incrementally

### Testing Strategy
1. Create all new files first (compilation will fail, that's OK)
2. Fix imports in new files
3. Update main.rs and lib.rs
4. Delete old files
5. cargo build
6. cargo clippy
7. cargo run

## Execution Checklist

- [ ] Create src/gpu/ directory structure
- [ ] Split mvp/gpu.rs into gpu/{types,resources,accumulation,gradient}.rs
- [ ] Create src/edge_detection.rs from mvp/mod.rs
- [ ] Move mvp/render.rs to src/event_renderer.rs
- [ ] Move mvp/playback.rs to src/playback.rs
- [ ] Update all imports in new files
- [ ] Update src/main.rs
- [ ] Update src/lib.rs
- [ ] Delete src/mvp/ directory
- [ ] Delete old files (analysis, gizmos, gpu, plugins, render, synthesis, bin)
- [ ] Delete old shaders
- [ ] cargo build --bin ebc-rs
- [ ] cargo clippy
- [ ] cargo run --bin ebc-rs
- [ ] Commit changes

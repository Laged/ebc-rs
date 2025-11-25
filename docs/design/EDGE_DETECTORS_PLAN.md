# Implementation Plan: Canny and LoG Edge Detectors

## Goal
Add two new edge detection algorithms (Canny and Laplacian of Gaussian) as separate GPU compute pipelines that can be toggled independently alongside the existing Sobel gradient detector.

## Current State

**Existing:**
- Sobel spatial gradient detector (working)
- Single gradient output texture
- EdgeParams with show_gradient toggle
- Visualizer shows yellow edges when enabled

**Architecture:**
```
Surface Texture (R32Uint) → Sobel Pipeline → Gradient Texture (R32Float) → Visualizer
```

## Target Architecture

**Three Independent Pipelines:**
```
Surface Texture (R32Uint) ─┬→ Sobel Pipeline → Sobel Output (R32Float)
                            ├→ Canny Pipeline → Canny Output (R32Float)
                            └→ LoG Pipeline → LoG Output (R32Float)
                                    ↓
                            All three → Visualizer (blend with colors)
```

## Algorithm Details

### 1. Canny Edge Detection (4-stage)

**Stage 1: Gaussian Blur**
- 5x5 Gaussian kernel convolution
- Reduces noise before edge detection
- Output: Blurred timestamp surface

**Stage 2: Gradient Calculation**
- Sobel Gx/Gy kernels (same as current)
- Compute magnitude: sqrt(Gx² + Gy²)
- Compute direction: atan2(Gy, Gx) quantized to 4 directions (0°, 45°, 90°, 135°)
- Output: Magnitude + Direction

**Stage 3: Non-Maximum Suppression**
- For each pixel, compare magnitude to neighbors in gradient direction
- Keep only local maxima
- Thin edges to single-pixel width
- Output: Thinned edges

**Stage 4: Hysteresis Thresholding**
- Upper threshold: Strong edges (definitely keep)
- Lower threshold: Weak edges (keep only if connected to strong)
- Requires edge tracing (iterative, may need multiple passes)
- Output: Binary edge map

**Parameters:**
- `canny_sigma`: Gaussian blur strength (default: 1.4)
- `canny_low_threshold`: Lower threshold (default: 50.0)
- `canny_high_threshold`: Upper threshold (default: 150.0)

**Challenges:**
- Hysteresis requires connectivity checking (not trivially parallel)
- May need multi-pass approach or approximation
- Consider simplified version: just use double-threshold without connectivity

### 2. Laplacian of Gaussian (LoG)

**Stage 1: Gaussian Smoothing**
- 5x5 Gaussian kernel (same as Canny stage 1)
- Reduces noise sensitivity

**Stage 2: Laplacian (Second Derivative)**
- Apply Laplacian kernel (detects second derivative)
- Common 3x3 kernel:
```
 0  1  0
 1 -4  1
 0  1  0
```
- Or combined LoG kernel (5x5 or larger)

**Stage 3: Zero-Crossing Detection**
- Find pixels where Laplacian changes sign
- Check 4 or 8 neighbors
- Strong zero-crossings (large magnitude) = edges
- Threshold on crossing strength

**Parameters:**
- `log_sigma`: Gaussian blur strength (default: 1.4)
- `log_threshold`: Zero-crossing strength threshold (default: 10.0)

**Advantages:**
- Simpler than Canny (no direction tracking, no hysteresis)
- Good noise robustness
- Isotropic (no directional bias)

## Implementation Plan

### Phase 1: Add New GPU Resources and Parameters

**File: `src/gpu/resources.rs`**

Add new resources:
```rust
// Canny output texture
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct CannyImage {
    pub handle: Handle<Image>,
}

// LoG output texture
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct LogImage {
    pub handle: Handle<Image>,
}
```

Update EdgeParams:
```rust
pub struct EdgeParams {
    pub threshold: f32,

    // Detector toggles
    pub show_sobel: bool,      // Renamed from show_gradient
    pub show_canny: bool,      // New
    pub show_log: bool,        // New
    pub show_raw: bool,

    // Sobel filter toggles (keep existing)
    pub filter_dead_pixels: bool,
    pub filter_density: bool,
    pub filter_bidirectional: bool,
    pub filter_temporal: bool,

    // Canny parameters
    pub canny_sigma: f32,
    pub canny_low_threshold: f32,
    pub canny_high_threshold: f32,

    // LoG parameters
    pub log_sigma: f32,
    pub log_threshold: f32,
}
```

### Phase 2: Create Canny Shader

**File: `assets/shaders/canny.wgsl`**

**Simplified Approach (Single-Pass):**
- Skip full hysteresis (too complex for single pass)
- Implement: Gaussian blur → Gradient → Non-max suppression → Double threshold
- This gives 90% of Canny's benefits

**Structure:**
```wgsl
@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var canny_output: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: CannyParams;

struct CannyParams {
    threshold: f32,
    sigma: f32,
    low_threshold: f32,
    high_threshold: f32,
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 1. Load 5x5 neighborhood for Gaussian
    // 2. Apply Gaussian blur to timestamps
    // 3. Compute Sobel Gx/Gy on blurred values
    // 4. Compute magnitude and direction
    // 5. Non-maximum suppression
    // 6. Double threshold (strong=1.0, weak=0.5, none=0.0)
    // 7. Write output
}
```

### Phase 3: Create LoG Shader

**File: `assets/shaders/log.wgsl`**

**Structure:**
```wgsl
@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var log_output: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: LogParams;

struct LogParams {
    sigma: f32,
    threshold: f32,
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 1. Load 5x5 neighborhood
    // 2. Apply Gaussian blur
    // 3. Apply Laplacian kernel to detect second derivative
    // 4. Check zero-crossings with 8 neighbors
    // 5. Threshold on crossing strength
    // 6. Write binary output (1.0 = edge, 0.0 = no edge)
}
```

### Phase 4: Create GPU Pipeline Modules

**File: `src/gpu/canny.rs`**

Similar structure to `gradient.rs`:
```rust
pub struct CannyPipeline { ... }
pub struct CannyBindGroup(pub BindGroup);
pub struct CannyParamsBuffer(pub Buffer);
pub fn prepare_canny(...) { ... }
pub struct CannyLabel;
pub struct CannyNode;
```

**File: `src/gpu/log.rs`**

Similar structure:
```rust
pub struct LogPipeline { ... }
pub struct LogBindGroup(pub BindGroup);
pub struct LogParamsBuffer(pub Buffer);
pub fn prepare_log(...) { ... }
pub struct LogLabel;
pub struct LogNode;
```

**File: `src/gpu/mod.rs`**

Add modules and re-exports:
```rust
pub mod canny;
pub mod log;

pub use canny::{CannyPipeline, CannyNode, ...};
pub use log::{LogPipeline, LogNode, ...};
```

### Phase 5: Update Visualizer Shader

**File: `assets/shaders/visualizer.wgsl`**

Add new texture bindings:
```wgsl
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var sobel_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var sobel_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var canny_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var canny_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var log_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var log_sampler: sampler;

struct Params {
    ...
    show_sobel: u32,
    show_canny: u32,
    show_log: u32,
}
```

Blend logic:
```wgsl
// Sobel = Yellow (1.0, 1.0, 0.0)
// Canny = Cyan (0.0, 1.0, 1.0)
// LoG = Magenta (1.0, 0.0, 1.0)

if (params.show_sobel == 1u) {
    let sobel_val = textureSample(sobel_texture, sobel_sampler, in.uv).r;
    if (sobel_val > 0.0) {
        output_color = mix(output_color, vec3(1.0, 1.0, 0.0), 0.5);
    }
}
// Similar for canny and log with different colors
```

### Phase 6: Update Event Renderer

**File: `src/event_renderer.rs`**

Update EventMaterial:
```rust
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct EventMaterial {
    #[uniform(0)]
    params: EventParams,
    #[texture(1, sample_type = "u_int")]
    surface_texture: Handle<Image>,
    #[texture(2)]
    #[sampler(3)]
    sobel_texture: Handle<Image>,  // Renamed
    #[texture(4)]
    #[sampler(5)]
    canny_texture: Handle<Image>,
    #[texture(6)]
    #[sampler(7)]
    log_texture: Handle<Image>,
}
```

Update setup_scene to create all three textures:
```rust
fn setup_scene(...) {
    // Create sobel_image, canny_image, log_image
    // All R32Float format
    // Store handles in SobelImage, CannyImage, LogImage resources
}
```

Update UI:
```rust
fn ui_system(...) {
    egui::Window::new("Edge Detection").show(ctx, |ui| {
        ui.checkbox(&mut edge_params.show_raw, "Show Raw Data");
        ui.separator();

        ui.label("Edge Detectors:");
        ui.checkbox(&mut edge_params.show_sobel, "Sobel (Yellow)");
        ui.checkbox(&mut edge_params.show_canny, "Canny (Cyan)");
        ui.checkbox(&mut edge_params.show_log, "LoG (Magenta)");

        ui.separator();

        if edge_params.show_sobel {
            ui.label("Sobel Settings:");
            ui.add(egui::Slider::new(&mut edge_params.threshold, 0.0..=10_000.0));
            // ... existing filters
        }

        if edge_params.show_canny {
            ui.label("Canny Settings:");
            ui.add(egui::Slider::new(&mut edge_params.canny_low_threshold, 0.0..=500.0));
            ui.add(egui::Slider::new(&mut edge_params.canny_high_threshold, 0.0..=1000.0));
        }

        if edge_params.show_log {
            ui.label("LoG Settings:");
            ui.add(egui::Slider::new(&mut edge_params.log_threshold, 0.0..=100.0));
        }
    });
}
```

### Phase 7: Update Edge Detection Plugin

**File: `src/edge_detection.rs`**

Initialize new resources:
```rust
impl Plugin for EdgeDetectionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SurfaceImage>()
            .init_resource::<SobelImage>()    // Renamed
            .init_resource::<CannyImage>()    // New
            .init_resource::<LogImage>()      // New
            .init_resource::<EdgeParams>()
            ...
    }

    fn finish(&self, app: &mut App) {
        render_app
            .init_resource::<SobelPipeline>()   // Renamed
            .init_resource::<CannyPipeline>()   // New
            .init_resource::<LogPipeline>()     // New
            ...
            .add_systems(Render, prepare_sobel.in_set(RenderSystems::Queue))
            .add_systems(Render, prepare_canny.in_set(RenderSystems::Queue))
            .add_systems(Render, prepare_log.in_set(RenderSystems::Queue));

        // Render graph: Event → Sobel → Canny → LoG → Camera
        render_graph.add_node(SobelLabel, SobelNode::default());
        render_graph.add_node(CannyLabel, CannyNode::default());
        render_graph.add_node(LogLabel, LogNode::default());
        render_graph.add_node_edge(EventLabel, SobelLabel);
        render_graph.add_node_edge(SobelLabel, CannyLabel);
        render_graph.add_node_edge(CannyLabel, LogLabel);
        render_graph.add_node_edge(LogLabel, CameraDriverLabel);
    }
}
```

### Phase 8: Rename Gradient → Sobel

Since we're adding multiple detectors, rename for clarity:
- `gradient.rs` → `sobel.rs`
- `GradientImage` → `SobelImage`
- `GradientPipeline` → `SobelPipeline`
- `spatial_gradient.wgsl` → `sobel.wgsl`
- Update all imports

## Testing Strategy

1. **Verify existing Sobel still works** after rename
2. **Test Canny pipeline** independently with toggle
3. **Test LoG pipeline** independently with toggle
4. **Test all three together** with different colors
5. **Test parameter tuning** for each detector
6. **Verify performance** (three pipelines shouldn't significantly impact FPS)

## Success Criteria

- [ ] All three detectors can be toggled independently
- [ ] Each detector has distinct color in visualizer
- [ ] Parameters are adjustable per detector
- [ ] Build succeeds without warnings
- [ ] Application runs at acceptable FPS (>30)
- [ ] Edge detection quality is visually good for all three

## Estimated Complexity

**Total Tasks:** ~15-20 distinct files/changes
**Risk Areas:**
- Canny hysteresis (simplified approach should work)
- Shader complexity (may need optimization)
- Memory usage (three output textures)

**Recommended Approach:**
1. Start with LoG (simpler than Canny)
2. Then add Canny
3. Test thoroughly at each step
4. Commit after each working detector

## Future Enhancements

- Add more detectors (Prewitt, Roberts, etc.)
- Multi-pass Canny with full hysteresis
- Adaptive thresholds
- Edge thinning/morphology
- Combine detectors (ensemble methods)

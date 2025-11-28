# Implementation Plan: compare_live UI and Parameter Controls

## Date: 2025-11-26

## Problem Statement

The `compare_live` binary has multiple UI/parameter control issues:
1. **Detector toggle checkboxes don't work** - Cannot disable LoG or other detector layers
2. **Threshold sliders appear to have no effect** - Changes don't visibly affect output
3. **Config values may not be reflected in UI** - Loaded parameters not visible in controls

## Root Cause Analysis

### Architecture Overview

```
compare_live.rs
├── EdgeDetectionPlugin (includes EventRendererPlugin)
│   ├── EventRendererPlugin::setup_scene()
│   │   ├── Camera at z=1000
│   │   └── EventMaterial mesh at z=0 (uses visualizer.wgsl)
│   └── EventRendererPlugin::ui_system()
│       └── Modifies EdgeParams (threshold sliders, show_* checkboxes)
├── CompositeRenderPlugin
│   └── setup_composite_texture()
│       └── StandardMaterial mesh at z=1 (displays composite texture)
└── CompareUiPlugin
    └── draw_metrics_overlay()
        └── Only displays metrics, NO parameter controls
```

### Issue #1: Detector Toggles Don't Work

**Data Flow:**
1. `ui_system()` modifies `edge_params.show_sobel/show_canny/show_log`
2. `update_material_params()` copies these to `EventMaterial::params`
3. `visualizer.wgsl` uses show_* to blend detector outputs

**Problem:** The composite mesh (z=1) completely occludes the EventMaterial mesh (z=0). The `composite.wgsl` shader has NO toggle parameters - it always renders all 4 quadrants unconditionally.

**Evidence:**
- `composite.wgsl` lines 46-67: Always reads all 4 textures, no conditionals based on show_* flags
- `CompositeBindGroup` has no uniform buffer for parameters
- `CompositePipeline` layout has no uniform binding

### Issue #2: Threshold Sliders Appear Ineffective

**Data Flow:**
1. `ui_system()` modifies `edge_params.sobel_threshold/canny_low/high/log_threshold`
2. `extract_edge_params()` extracts EdgeParams to render world every frame
3. `prepare_sobel()` packs EdgeParams into GpuParams and writes to uniform buffer
4. `sobel.wgsl` reads `params.sobel_threshold` and uses it for thresholding

**Finding:** This data flow IS working correctly. The thresholds ARE being applied.

**Problem:** The threshold sliders affect the edge detection compute shaders (Sobel/Canny/LoG), which write to their respective textures. These ARE being updated. However:
- The initial config values from `config/detectors.toml` are already optimized
- Small threshold changes may not produce visible differences
- The user expects to see threshold changes in the composite view, which DOES show the updated edge detection output

### Issue #3: Config Not Reflected in UI

**Data Flow:**
1. `enable_all_detectors()` runs at Startup
2. Reads from `Res<CompareConfig>` and writes to `ResMut<EdgeParams>`
3. `ui_system()` reads from `Res<EdgeParams>` to populate sliders

**Finding:** This SHOULD work. The sliders should show config values after the first frame.

## Implementation Plan

### Task 1: Add Parameter Controls to CompareUiPlugin

**File:** `src/compare/ui.rs`

Add detector parameter controls to the compare_live UI:

```rust
/// Draw edge detection parameter controls
pub fn draw_edge_controls(
    mut contexts: EguiContexts,
    mut edge_params: ResMut<EdgeParams>,
) {
    let ctx = contexts.ctx_mut().expect("Failed to get egui context");

    egui::Window::new("Edge Detection")
        .default_pos([10.0, 100.0])
        .show(ctx, |ui| {
            // Detector visibility toggles
            ui.heading("Visibility");
            ui.checkbox(&mut edge_params.show_raw, "Show Raw (Q1)");
            ui.checkbox(&mut edge_params.show_sobel, "Show Sobel (Q2)");
            ui.checkbox(&mut edge_params.show_canny, "Show Canny (Q3)");
            ui.checkbox(&mut edge_params.show_log, "Show LoG (Q4)");

            ui.separator();
            ui.heading("Thresholds");

            // Sobel threshold
            ui.add(egui::Slider::new(&mut edge_params.sobel_threshold, 0.0..=10_000.0)
                .text("Sobel"));

            // Canny thresholds
            ui.add(egui::Slider::new(&mut edge_params.canny_low_threshold, 0.0..=5_000.0)
                .text("Canny Low"));
            ui.add(egui::Slider::new(&mut edge_params.canny_high_threshold, 0.0..=10_000.0)
                .text("Canny High"));

            // LoG threshold
            ui.add(egui::Slider::new(&mut edge_params.log_threshold, 0.0..=10_000.0)
                .text("LoG"));

            ui.separator();
            ui.heading("Filters");
            ui.checkbox(&mut edge_params.filter_dead_pixels, "Filter Dead Pixels");
            ui.checkbox(&mut edge_params.filter_density, "Filter Low Density");
            ui.checkbox(&mut edge_params.filter_bidirectional, "Bidirectional");
        });
}
```

**Changes to `CompareUiPlugin`:**
```rust
impl Plugin for CompareUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DataFileState>()
            .add_systems(EguiPrimaryContextPass, (
                draw_metrics_overlay,
                draw_edge_controls,  // NEW
            ))
            .add_systems(Update, handle_file_input);
    }
}
```

### Task 2: Add Visibility Toggles to Composite Shader

**File:** `assets/shaders/composite.wgsl`

Add a uniform buffer for visibility flags:

```wgsl
struct CompositeParams {
    show_raw: u32,
    show_sobel: u32,
    show_canny: u32,
    show_log: u32,
}

@group(0) @binding(5) var<uniform> params: CompositeParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // ... existing quadrant logic ...

    if (!is_right && !is_bottom) {
        // Top-left: Raw events
        if (params.show_raw == 1u) {
            let raw_value = textureLoad(raw_texture, source_coords, 0).r;
            intensity = select(0.0, 1.0, raw_value > 0u);
            color = RAW_COLOR;
        } else {
            intensity = 0.0;
            color = vec3<f32>(0.1, 0.1, 0.1); // Dark background when disabled
        }
    } else if (is_right && !is_bottom) {
        // Top-right: Sobel
        if (params.show_sobel == 1u) {
            intensity = textureLoad(sobel_texture, source_coords, 0).r;
            color = SOBEL_COLOR;
        } else {
            intensity = 0.0;
            color = vec3<f32>(0.1, 0.1, 0.1);
        }
    }
    // ... similar for canny and log ...
}
```

**File:** `src/compare/composite.rs`

Add uniform buffer and bind group entry:

1. Create `CompositeParams` struct matching WGSL
2. Add binding 5 to `BindGroupLayout`
3. Create `CompositeParamsBuffer` resource
4. Update `prepare_composite` to read from `EdgeParams` and write to buffer

### Task 3: Extract EdgeParams in CompositeRenderPlugin

**File:** `src/compare/plugin.rs`

Add extraction for EdgeParams so composite shader can access it:

```rust
// In finish():
render_app
    .add_systems(ExtractSchedule, extract_edge_params_for_composite)
    // ... existing systems ...
```

### Task 4: Remove Duplicate EventMaterial Mesh in compare_live

**Option A:** Disable EventRendererPlugin in compare_live mode
- Add a marker resource `CompareMode` that EventRendererPlugin checks
- Skip spawning EventMaterial mesh when in CompareMode

**Option B:** Hide EventMaterial mesh
- After Startup, query for EventMaterial meshes and set `Visibility::Hidden`

**Option C (Recommended):** Create separate `CompareMaterialPlugin`
- Don't include EventRendererPlugin in compare_live
- Create a minimal plugin that only sets up the compute pipeline without the EventMaterial visualization
- This cleanly separates the two modes

### Task 5: Verify Threshold Changes Are Working

After implementing the above:
1. Add debug logging to `prepare_sobel` to confirm threshold values are updating
2. Use extreme threshold values (0.0 or 100000.0) to verify visual changes
3. If still no visible change, investigate whether the edge detection output textures are being properly written

## Implementation Order

1. **Task 1** - Add parameter controls to CompareUiPlugin (enables user to adjust values)
2. **Task 2 + 3** - Add visibility toggles to composite shader (enables detector toggling)
3. **Task 4** - Remove duplicate EventMaterial mesh (cleanup)
4. **Task 5** - Verify and debug threshold propagation

## Estimated Changes

| File | Changes |
|------|---------|
| `src/compare/ui.rs` | +50 lines (edge controls UI) |
| `src/compare/composite.rs` | +40 lines (params buffer, bind group) |
| `src/compare/plugin.rs` | +10 lines (extraction) |
| `assets/shaders/composite.wgsl` | +30 lines (params struct, conditionals) |
| `src/bin/compare_live.rs` | -5 lines (remove EventMaterial spawn) |

## Testing

1. Run `cargo run --bin compare_live -- data/synthetic/fan_test.dat`
2. Verify Edge Detection panel appears with sliders
3. Toggle each detector checkbox - corresponding quadrant should go dark
4. Adjust Sobel threshold from 0 to 10000 - Q2 edge density should change dramatically
5. Save config changes (future enhancement)

# Ground Truth Pipeline Design

## Overview

Add a GPU-based ground truth rendering pipeline for synthetic fan data validation. The ground truth shader analytically computes exact blade positions per frame, enabling pixel-perfect comparison with edge detectors (Sobel, Canny, LoG).

## Requirements

| Requirement | Decision |
|-------------|----------|
| Edge type | Both boundaries + interior fill (edge pixels marked differently) |
| Computation | GPU compute shader (real-time per frame) |
| Parameters source | JSON sidecar file (fan_test_truth.json) |
| Architecture | New standalone pipeline (ground_truth.wgsl) |
| Texture encoding | Two-channel: R=edge pixels, G=interior pixels |

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYNTHETIC DATA LOADING                           │
│  1. Load fan_test.dat (events)                                     │
│  2. Detect sidecar: fan_test_truth.json                            │
│  3. Parse GroundTruthConfig {center, radius, blade_count, rpm...}  │
│  4. Store in Bevy Resource: GroundTruthConfig                       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU GROUND TRUTH PIPELINE                        │
│  Shader: ground_truth.wgsl (compute)                               │
│  Input:  GroundTruthParams uniform buffer + current_time           │
│  Output: ground_truth_texture (Rgba8Unorm)                          │
│          - R channel: edge pixels (1.0)                             │
│          - G channel: interior pixels (1.0)                         │
│          - B,A: reserved                                            │
│  Runs: Every frame in render graph (parallel to edge detectors)    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VISUALIZER + METRICS                             │
│  1. Visualizer: show_ground_truth toggle → render as Green          │
│  2. Metrics: Compare ground_truth_texture vs sobel/canny/log        │
│     - Precision = TP / (TP + FP)                                    │
│     - Recall    = TP / (TP + FN)                                    │
│     - F1 Score  = 2 * P * R / (P + R)                               │
│     - IoU       = TP / (TP + FP + FN)                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Shader Algorithm

The `ground_truth.wgsl` compute shader analytically computes blade positions:

```wgsl
// For each pixel (x, y):
// 1. Convert to polar coords relative to fan center
let dx = f32(x) - params.center_x;
let dy = f32(y) - params.center_y;
let r = sqrt(dx*dx + dy*dy);
let theta = atan2(dy, dx);

// 2. Skip pixels outside fan radius
if (r < params.r_min || r > params.r_max) { return; }

// 3. Calculate current blade rotation angle
let base_angle = params.angular_velocity * params.current_time;

// 4. For each blade, check if pixel is on blade
for (blade in 0..blade_count) {
    let blade_angle = base_angle + (blade * 2π / blade_count);

    // Logarithmic spiral: compensate for blade sweep
    let sweep_angle = sweep_k * ln(r / r_min);
    let blade_center = blade_angle + sweep_angle;

    // Blade width varies with radius (wider at root)
    let half_width = lerp(width_root, width_tip, (r - r_min) / (r_max - r_min));

    // Check if pixel angle is within blade
    let angle_diff = wrap_angle(theta - blade_center);

    if (abs(angle_diff) < half_width) {
        // Interior pixel
        output.g = 1.0;

        // Edge pixel (within threshold of boundary)
        if (abs(abs(angle_diff) - half_width) < edge_thickness / r) {
            output.r = 1.0;
        }
    }
}
```

Key insight: Edge detection in polar coordinates - a pixel is an "edge" if it's within `edge_thickness` (1-2px) of the blade boundary angle.

## File Structure

```
src/
├── gpu/
│   ├── ground_truth.rs      # NEW: Pipeline, BindGroup, Node, prepare system
│   ├── resources.rs         # ADD: GroundTruthImage, GroundTruthParams
│   └── mod.rs               # Export new module
├── ground_truth.rs          # NEW: GroundTruthConfig resource, JSON loader
├── loader.rs                # MODIFY: Auto-detect and load sidecar JSON
└── lib.rs                   # Export ground_truth module

assets/shaders/
├── ground_truth.wgsl        # NEW: Compute shader for blade geometry
└── visualizer.wgsl          # MODIFY: Add ground_truth_texture binding

data/synthetic/
├── fan_test.dat             # Events (existing)
└── fan_test_truth.json      # MODIFY: Add blade geometry params
```

## New Types

### GroundTruthConfig (CPU-side resource)

```rust
// src/ground_truth.rs
#[derive(Resource, Deserialize, Default)]
pub struct GroundTruthConfig {
    pub enabled: bool,
    pub center: Vec2,
    pub radius_min: f32,
    pub radius_max: f32,
    pub blade_count: u32,
    pub rpm: f32,
    pub sweep_k: f32,           // Logarithmic spiral curvature (0.5)
    pub width_root_rad: f32,    // Blade angular width at root (0.5 rad)
    pub width_tip_rad: f32,     // Blade angular width at tip (0.3 rad)
    pub edge_thickness_px: f32, // Edge detection band (1-2px)
}
```

### GroundTruthParams (GPU uniform buffer)

```rust
// src/gpu/resources.rs
#[derive(ShaderType, Clone, Copy)]
pub struct GroundTruthParams {
    pub center_x: f32,
    pub center_y: f32,
    pub r_min: f32,
    pub r_max: f32,
    pub blade_count: u32,
    pub angular_velocity: f32,  // rad/s = rpm * 2π / 60
    pub current_time: f32,
    pub sweep_k: f32,
    pub width_root: f32,
    pub width_tip: f32,
    pub edge_thickness: f32,
    pub _padding: f32,
}
```

### EdgeMetrics (comparison results)

```rust
pub struct EdgeMetrics {
    pub true_positives: u32,   // Detector=1, GT=1
    pub false_positives: u32,  // Detector=1, GT=0
    pub false_negatives: u32,  // Detector=0, GT=1
    pub precision: f32,        // TP / (TP + FP)
    pub recall: f32,           // TP / (TP + FN)
    pub f1_score: f32,         // 2*P*R / (P+R)
    pub iou: f32,              // TP / (TP + FP + FN)
}
```

## Integration

### Render Graph

```
Event → Preprocess → ┬─ Sobel ─────┬─→ Visualizer
                     ├─ Canny ─────┤
                     ├─ LoG ───────┤
                     └─ GroundTruth┘   (parallel, no dependency on filtered events)
```

### Visualizer Changes

- Add binding slots 8-9: `ground_truth_texture` + sampler
- Add `show_ground_truth: u32` to Params struct
- Render as **Green** with 50% alpha blend (unused color)
- UI checkbox: "Show Ground Truth (Green)"

### Conditional Activation

- Ground truth pipeline only runs when `GroundTruthConfig.enabled = true`
- Auto-enabled when loading synthetic data with valid sidecar JSON
- Disabled for real fan data (no ground truth available)

## Updated JSON Schema

The `fan_test_truth.json` sidecar file will be extended:

```json
{
  "params": {
    "center_x": 640.0,
    "center_y": 360.0,
    "radius_min": 50.0,
    "radius_max": 200.0,
    "blade_count": 3,
    "rpm": 1200.0,
    "sweep_k": 0.5,
    "width_root_rad": 0.5,
    "width_tip_rad": 0.3,
    "edge_thickness_px": 2.0
  },
  "frames": [
    {"time": 0.01, "angle": 1.2566, ...},
    ...
  ]
}
```

## Success Criteria

For synthetic data with perfect edge detection:

| Metric | 100% Match Value |
|--------|------------------|
| Precision | 1.0 |
| Recall | 1.0 |
| F1 Score | 1.0 |
| IoU | 1.0 |
| Centroid error | 0.0 px |
| Radius error | 0.0 px |

## Implementation Tasks

1. Update synthesis.rs to output extended JSON with blade geometry params
2. Create src/ground_truth.rs with GroundTruthConfig and JSON loader
3. Create src/gpu/ground_truth.rs with pipeline infrastructure
4. Create assets/shaders/ground_truth.wgsl compute shader
5. Update src/gpu/resources.rs with GroundTruthImage and GroundTruthParams
6. Update visualizer.wgsl with ground truth texture binding
7. Update event_renderer.rs with show_ground_truth toggle
8. Add EdgeMetrics computation to hypertest for accuracy measurement
9. Update hypersearch to report precision/recall/F1 metrics

# Hyperparameter Architecture Refactor Design

**Date:** 2025-01-25
**Status:** Approved
**Goal:** Separate pre/post processing, enable systematic hyperparameter optimization

---

## 1. Architecture Overview

### Current Pipeline
```
Events → Accumulation → Sobel (with filters) → Canny → LoG → Readback
```

### New Pipeline
```
Events → Accumulation → PREPROCESS → Sobel → Canny → LoG → Readback
              ↓              ↓
         SurfaceImage   FilteredSurfaceImage
```

### New Render Graph
```
EventLabel → PreprocessLabel → SobelLabel → CannyLabel → LogLabel → ReadbackLabel → CameraDriver
```

---

## 2. New Components

### 2.1 FilteredSurfaceImage Resource
```rust
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct FilteredSurfaceImage {
    pub handle: Handle<Image>,
}
```
- Same format as SurfaceImage (R32Uint)
- Output of preprocess shader
- Input to all three detectors

### 2.2 Preprocess Shader (`assets/shaders/preprocess.wgsl`)
```wgsl
@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var filtered_output: texture_storage_2d<r32uint, write>;
@group(0) @binding(2) var<uniform> params: GpuParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Apply filters, write to filtered_output
    // - Dead pixel filter
    // - Density filter
    // - Temporal filter
}
```

### 2.3 PreprocessNode & PreprocessPipeline
```rust
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct PreprocessLabel;

#[derive(Default)]
pub struct PreprocessNode;

impl Node for PreprocessNode { ... }

#[derive(Resource)]
pub struct PreprocessPipeline {
    pub pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
}
```

---

## 3. Unified Parameter Buffer

### 3.1 GPU Struct (WGSL-compatible)
```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuParams {
    // Accumulation
    pub window_size_us: f32,

    // Pre-processing
    pub filter_dead_pixels: u32,
    pub filter_density: u32,
    pub filter_temporal: u32,
    pub min_density_count: u32,       // Was hardcoded: 5
    pub min_temporal_spread: f32,     // Was hardcoded: 500.0

    // Sobel
    pub sobel_threshold: f32,

    // Canny
    pub canny_low_threshold: f32,
    pub canny_high_threshold: f32,

    // LoG
    pub log_threshold: f32,

    // Post-processing
    pub filter_bidirectional: u32,
    pub bidirectional_ratio: f32,     // Was hardcoded: 0.3

    // Padding
    pub _padding: [f32; 2],
}
```

### 3.2 Rust-side Resource
```rust
#[derive(Resource, Clone)]
pub struct EdgeParams {
    // Accumulation
    pub window_size_us: f32,

    // Pre-processing
    pub filter_dead_pixels: bool,
    pub filter_density: bool,
    pub filter_temporal: bool,
    pub min_density_count: u32,
    pub min_temporal_spread_us: f32,

    // Detector toggles (UI only, not sent to GPU)
    pub show_sobel: bool,
    pub show_canny: bool,
    pub show_log: bool,
    pub show_raw: bool,

    // Sobel
    pub sobel_threshold: f32,

    // Canny
    pub canny_low_threshold: f32,
    pub canny_high_threshold: f32,

    // LoG
    pub log_threshold: f32,

    // Post-processing
    pub filter_bidirectional: bool,
    pub bidirectional_ratio: f32,
}

impl Default for EdgeParams {
    fn default() -> Self {
        Self {
            window_size_us: 100.0,
            filter_dead_pixels: true,
            filter_density: false,
            filter_temporal: false,
            min_density_count: 5,
            min_temporal_spread_us: 500.0,
            show_sobel: true,
            show_canny: false,
            show_log: false,
            show_raw: false,
            sobel_threshold: 1000.0,
            canny_low_threshold: 50.0,
            canny_high_threshold: 150.0,
            log_threshold: 10.0,
            filter_bidirectional: false,
            bidirectional_ratio: 0.3,
        }
    }
}
```

---

## 4. Detector Shader Changes

### 4.1 Sobel Changes
- Remove filter logic (lines 44-104 in current sobel.wgsl)
- Change input from `surface_texture` to `filtered_texture`
- Keep only Sobel kernel computation + thresholding

### 4.2 Canny Changes
- Change input from `surface_texture` to `filtered_texture`
- Now benefits from pre-filters automatically

### 4.3 LoG Changes
- Change input from `surface_texture` to `filtered_texture`
- Now benefits from pre-filters automatically

---

## 5. Hyperparameter Test Infrastructure

### 5.1 CLI Binary (`src/bin/hypertest.rs`)
```rust
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    data: PathBuf,

    #[arg(long, default_value = "sobel")]
    detector: String,

    #[arg(long, default_value = "100")]
    window_size: f32,

    #[arg(long, default_value = "1000")]
    threshold: f32,

    #[arg(long)]
    filter_dead_pixels: bool,

    #[arg(long)]
    filter_density: bool,

    #[arg(long)]
    filter_temporal: bool,

    #[arg(long, default_value = "50")]
    frames: usize,
}

fn main() {
    let args = Args::parse();
    let config = HyperConfig::from(&args);
    let result = run_headless_test(&config);
    println!("{}", serde_json::to_string(&result).unwrap());
}
```

### 5.2 Grid Search Binary (`src/bin/hypersearch.rs`)
```rust
use rayon::prelude::*;

fn main() {
    let configs = generate_grid_configs();

    let results: Vec<HyperResult> = configs.par_iter()
        .map(|config| {
            let output = Command::new("cargo")
                .args(["run", "--release", "--bin", "hypertest", "--"])
                .args(config.to_cli_args())
                .output()
                .expect("subprocess failed");

            serde_json::from_slice(&output.stdout).unwrap()
        })
        .collect();

    export_csv(&results, &args.output);
    let best = select_best(&results);
    save_best_config(&best, "results/best.json");
}
```

### 5.3 Result Structures
```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HyperConfig {
    pub detector: String,
    pub window_size_us: f32,
    pub filter_dead_pixels: bool,
    pub filter_density: bool,
    pub filter_temporal: bool,
    pub min_density_count: u32,
    pub min_temporal_spread_us: f32,
    pub threshold: f32,
    pub canny_low: f32,
    pub canny_high: f32,
    pub filter_bidirectional: bool,
    pub bidirectional_ratio: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HyperResult {
    pub config: HyperConfig,
    pub avg_edge_count: f32,
    pub edge_density: f32,
    pub centroid_stability: f32,
    pub radius_stability: f32,
    pub circle_fit_error: f32,
    pub inlier_ratio: f32,
    pub detected_blade_count: f32,
    pub runtime_ms: f32,
}
```

---

## 6. Optimization Workflow

### Phase A: Coarse Grid Search
```bash
cargo run --release --bin hypersearch -- \
    --data data/fan/fan_const_rpm.dat \
    --output results/coarse_real.csv \
    --window-sizes 10,50,100,500,1000,5000 \
    --thresholds 100,500,1000,2000,5000 \
    --filters all-combinations \
    --frames 30
```

### Phase B: Fine Grid Search
```bash
cargo run --release --bin hypersearch -- \
    --data data/fan/fan_const_rpm.dat \
    --output results/fine_real.csv \
    --window-sizes 100,150,200,250,300 \
    --thresholds 800,900,1000,1100,1200 \
    --frames 100
```

### Phase C: Validation
```bash
for data in data/fan/*.dat; do
    cargo run --release --bin hypertest -- \
        --config results/best.json \
        --data "$data" \
        --frames 200
done
```

### Selection Criteria (Priority Order)
1. **Centroid stability** - Lower is better (primary)
2. **Inlier ratio** - Higher is better
3. **Edge count in range** - 5k-50k edges
4. **Runtime** - Tiebreaker

---

## 7. File Structure

```
src/
├── gpu/
│   ├── mod.rs              # Add preprocess exports
│   ├── preprocess.rs       # NEW: PreprocessNode, PreprocessPipeline
│   ├── resources.rs        # Update EdgeParams, add GpuParams
│   ├── types.rs            # Update/replace GpuEdgeParams
│   ├── sobel.rs            # Remove filters, use filtered texture
│   ├── canny.rs            # Use filtered texture
│   └── log.rs              # Use filtered texture
├── bin/
│   ├── hypertest.rs        # NEW: Single config test runner
│   └── hypersearch.rs      # NEW: Grid search orchestrator
└── hyperparams.rs          # NEW: HyperConfig, HyperResult, grid generation

assets/shaders/
├── preprocess.wgsl         # NEW: Pre-processing filters
├── sobel.wgsl              # Simplified (no filters)
├── canny.wgsl              # Use filtered input
└── log.wgsl                # Use filtered input

results/                    # NEW: Output directory
├── coarse_real.csv
├── fine_real.csv
├── best_sobel.json
├── best_canny.json
└── best_log.json
```

---

## 8. Migration Notes

### Breaking Changes
- `GpuEdgeParams` renamed to `GpuParams` with new fields
- Sobel filters moved to preprocess stage
- Detectors now read from `FilteredSurfaceImage`

### Backwards Compatibility
- Default values preserve current behavior
- UI toggles still work (just control different stage now)
- Existing tests need filter texture setup

---

## 9. Success Criteria

1. All three detectors benefit from pre-filters
2. Hyperparameter search completes in <30 min for coarse grid
3. Find configs with centroid stability <1px on real data
4. CSV export works for external analysis
5. Best configs documented per detector

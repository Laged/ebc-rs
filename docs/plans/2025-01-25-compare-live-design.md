# Compare Live: Side-by-Side Detector Visualization

## Overview

A new binary `compare_live` that displays all edge detectors (Raw, Sobel, Canny, LoG) in a 2x2 grid with real-time metrics for each.

## Requirements

- **Layout:** 2x2 grid showing Raw Events | Sobel | Canny | LoG
- **Config:** TOML config file at `config/detectors.toml`
- **Metrics:** Standard metrics per detector (edge count, tolerance precision/recall/F1, avg distance)
- **Architecture:** GPU composite shader combining 4 textures
- **Multi-file:** Accept multiple data files, switch with N/P keys

## CLI Interface

```bash
cargo run --bin compare_live -- [OPTIONS] <DATA_FILES>...

Options:
  --config <PATH>     Config file (default: config/detectors.toml)
  --window <SIZE>     Override window size for all detectors
  --no-gt             Disable ground truth metrics computation
```

Examples:
```bash
# Single file
cargo run --bin compare_live -- data/fan/fan_const_rpm.dat

# Multiple files with glob
cargo run --bin compare_live -- data/synthetic/*.dat

# Custom config
cargo run --bin compare_live -- --config my_config.toml data/*.dat
```

## Configuration Format

`config/detectors.toml`:
```toml
[sobel]
threshold = 50.0
window_size_us = 100000.0
filter_dead_pixels = true

[canny]
low_threshold = 50.0
high_threshold = 150.0
window_size_us = 100000.0

[log]
threshold = 100.0
window_size_us = 100000.0

[display]
show_ground_truth = true
metrics_update_hz = 10
```

Config loading priority:
1. CLI `--config` flag
2. `config/detectors.toml` if exists
3. `results/best_*.json` files as fallback
4. Built-in defaults

## Architecture

### GPU Composite Shader

Output resolution: 2560x1440 (2x base 1280x720)

```
+------------------+------------------+
|                  |                  |
|   Raw Events     |     Sobel        |
|   (top-left)     |   (top-right)    |
|                  |                  |
+------------------+------------------+
|                  |                  |
|     Canny        |      LoG         |
| (bottom-left)    | (bottom-right)   |
|                  |                  |
+------------------+------------------+
```

Shader bindings:
- `@group(0) @binding(0)` - Raw/filtered surface texture
- `@group(0) @binding(1)` - Sobel output texture
- `@group(0) @binding(2)` - Canny output texture
- `@group(0) @binding(3)` - LoG output texture
- `@group(0) @binding(4)` - Output composite texture (write)

### Render Graph

```
Existing: Event -> Preprocess -> Sobel -> Canny -> LoG -> Readback
New:                          \      \      \      \
                               +------+------+-------> Composite -> Camera
```

### Metrics Pipeline

Per-detector metrics struct:
```rust
struct DetectorMetrics {
    edge_count: u32,
    tolerance_precision: f32,
    tolerance_recall: f32,
    tolerance_f1: f32,
    avg_distance: f32,
}
```

Display via egui overlay in each quadrant corner:
```
+-----------------------------+
| SOBEL                       |
| Edges: 4,931                |
| Prec: 100% | Rec: 49.2%     |
| F1: 65.9% | Dist: 3.6px     |
+-----------------------------+
```

Update rate: Metrics computed every frame, display updates at 10Hz.

### Multi-Readback

Extend `EdgeReadbackBuffer` to store results from all 4 textures:
```rust
struct MultiReadbackBuffer {
    raw_buffer: Buffer,
    sobel_buffer: Buffer,
    canny_buffer: Buffer,
    log_buffer: Buffer,
}
```

## File Structure

```
src/
├── bin/
│   └── compare_live.rs      # Binary entry point
├── compare/
│   ├── mod.rs
│   ├── composite.rs         # GPU composite shader + node
│   ├── config.rs            # TOML config loading
│   ├── multi_readback.rs    # Extended readback for 4 textures
│   └── ui.rs                # Egui overlay for metrics
assets/
└── shaders/
    └── composite.wgsl       # 2x2 grid composite shader
config/
└── detectors.toml           # Default config (created by hypersearch)
```

## Plugin Architecture

```rust
pub struct CompareLivePlugin;

impl Plugin for CompareLivePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EdgeDetectionPlugin)  // Reuse existing
           .add_plugins(CompareUiPlugin)
           .init_resource::<DataFileList>()
           .init_resource::<AllMetrics>()
           .add_systems(Update, handle_file_switch)
           .add_systems(Update, compute_all_metrics);
    }

    fn finish(&self, app: &mut App) {
        // Add composite node to render graph
    }
}
```

## Multi-File Support

Resources:
```rust
#[derive(Resource)]
struct DataFileList {
    files: Vec<PathBuf>,
    current_index: usize,
}
```

Key bindings:
- `N`: Next file
- `P`: Previous file
- Dropdown in egui sidebar

On file switch:
1. Pause playback
2. Load new events via `DatLoader::load()`
3. Update `EventData` resource
4. Load ground truth sidecar if available
5. Reset playback time to 0
6. Resume playback

## Integration Notes

Reuses from existing codebase:
- `EdgeDetectionPlugin` - All GPU pipelines
- `HyperConfig` - Config parsing
- `GroundTruthConfig` / `GroundTruthMetrics` - GT comparison
- `DatLoader` / `EventData` - Data loading
- `PlaybackState` - Time control

The existing `ebc-rs` binary remains unchanged.

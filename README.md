# Event-Based Camera Visualizer (ebc-rs)

<img src="fan-demo.gif" width="100%" alt="Fan Demo">

## Project Goal
This project is a high-performance visualizer for event-based camera data, built with Rust and Bevy. It is designed to efficiently load, process, and render millions of events using a GPU-accelerated pipeline.

## Data Format (.dat)
The application expects data in a custom binary format:

1.  **Header**: Text lines starting with `%`.
2.  **Event Type & Size**: 2 bytes (Type + Size=8).
3.  **Binary Data**: Sequence of 8-byte events.

### Event Structure (8 bytes)
*   **Timestamp (4 bytes)**: `u32` (microseconds).
*   **Data (4 bytes)**: `u32` containing packed fields:
    *   **X**: Bits 0-13 (14 bits)
    *   **Y**: Bits 14-27 (14 bits)
    *   **Polarity**: Bits 28-31 (4 bits)

## Rendering Pipeline
The visualization pipeline leverages WGPU compute shaders for high performance:

1.  **Binary Load**: `DatLoader` reads the `.dat` file and parses events into `GpuEvent` structs.
2.  **GPU Upload**: Events are uploaded to a `StorageBuffer` on the GPU (`GpuEventBuffer`).
3.  **Compute Shader**:
    *   A compute shader (`accumulation.wgsl`) processes the events.
    *   It iterates through the event buffer.
    *   Events falling within the current time window are accumulated onto a `SurfaceBuffer`.
4.  **Texture Copy**: The `SurfaceBuffer` is copied to a `GpuImage` texture.
5.  **Visualization**: A custom material (`EventMaterial`) renders the texture onto a quad, applying color mapping and decay effects based on the accumulated values.

## Controls
*   **Play/Pause**: Toggle playback.
*   **Loop**: Toggle looping.
*   **Time Slider**: Scrub through the dataset.
*   **Window Slider**: Adjust the integration time window (accumulation duration).
*   **Speed Slider**: Adjust playback speed.

## Motion Analysis
*   **Enable RPM Tracking**: Activate real-time analysis.
*   **Show Blade Borders**: Visualize detected fan geometry:
    *   **Blue Circle**: Auto-detected fan radius from intensity falloff
    *   **Green Lines**: Blade positions from angular event distribution
    *   **Yellow Dots**: Blade tip markers
    *   **Red Cross**: Centroid (fan center)
*   **Blade Count**: Number of blades to detect (2-8).

### How It Works
The visualizer uses GPU compute shaders to analyze event patterns:
1. **Centroid Tracking**: Calculates spatial mean of events
2. **Radial Analysis**: Builds intensity histogram to find 95th percentile radius
3. **Angular Histogram**: Detects blade positions via peak finding in polar distribution

See `docs/design/2025-11-23-fan-visualization-accuracy-design.md` for technical details.

## Quick Start with Nix

```bash
# Enter development shell
nix develop

# Generate synthetic test data
nix run .#generate_data

# Optimize detector parameters
nix run .#optimise_params

# Compare detectors live
nix run .#compare_live
```

## Workflow: Generating Synthetic Data

Create synthetic fan data with known ground truth for testing edge detectors:

```bash
cargo run --bin generate_synthetic
```

This creates:
- `data/synthetic/fan_test.dat` - Event data file
- `data/synthetic/fan_test_truth.json` - Ground truth configuration

The synthetic data simulates a rotating fan with precise blade positions, enabling accurate detector evaluation.

## Workflow: Hyperparameter Optimization

Find optimal edge detector parameters using grid search:

```bash
cargo run --release --bin hypersearch -- \
  --data data/synthetic/fan_test.dat \
  --output results/search.csv \
  --window-sizes 500,1000,2000,5000 \
  --thresholds 0.5,1.0,2.0,4.0
```

**What it does:**
1. Tests all combinations of detectors (Sobel, Canny, LoG) with specified window sizes and thresholds
2. Runs parallel evaluation using the `hypertest` subprocess
3. Saves results to CSV and selects best config per detector
4. **Auto-exports optimized settings to `config/detectors.toml`**

**Output files:**
- `results/search.csv` - Full results table
- `results/best_sobel.json` - Best Sobel parameters
- `results/best_canny.json` - Best Canny parameters
- `results/best_log.json` - Best LoG parameters
- `config/detectors.toml` - Combined config for `compare_live`

## Workflow: Live Detector Comparison

Compare all edge detectors side-by-side with real-time metrics:

```bash
cargo run --bin compare_live -- data/synthetic/fan_test.dat
# or with custom config:
cargo run --bin compare_live -- --config config/detectors.toml data/fan/fan_const_rpm.dat
```

**The 2x2 grid layout:**
| Top-Left (RAW) | Top-Right (SOBEL) |
|----------------|-------------------|
| Bottom-Left (CANNY) | Bottom-Right (LoG) |

**Controls:**
- `N` / `P` - Next/Previous data file
- `Space` - Pause playback
- Egui panels provide threshold sliders and detector toggles

**Metrics displayed per detector:**
- Edge count
- Precision / Recall (if ground truth available)
- F1 Score
- Average distance to ground truth

## Available Data

| Path | Description |
|------|-------------|
| `data/fan/fan_const_rpm.dat` | Real fan at constant RPM (211 MB) |
| `data/fan/fan_varying_rpm.dat` | Real fan with varying speed (512 MB) |
| `data/synthetic/fan_test.dat` | Synthetic fan with ground truth (1.6 MB) |
| `data/drone_idle/drone_idle.dat` | Stationary drone propellers |
| `data/drone_moving/drone_moving.dat` | Moving drone |

## Edge Detectors

All detectors use **binary event presence** (1.0 if event exists, 0.0 if not) for edge detection. This approach detects boundaries between event-active and event-inactive regions.

| Detector | Description | Threshold Range | Recommended |
|----------|-------------|-----------------|-------------|
| **Sobel** | 3x3 gradient kernels | 0-6 | 1.0 |
| **Canny** | Hysteresis with NMS | low: 0-3, high: 0-6 | 0.5 / 2.0 |
| **LoG** | 5x5 Laplacian of Gaussian | 0-16 | 2.0 |

**Performance on synthetic data (5000μs window):**
| Detector | Edges | Precision | Recall | F1 |
|----------|-------|-----------|--------|-----|
| LoG | 497 | 100% | 21.6% | 35.5% |
| Canny | 37 | 100% | 1.1% | 2.3% |
| Sobel | 40 | ~80% | <1% | <1% |

LoG performs best due to its larger 5x5 kernel being more robust to sparse event data.

All detectors run as GPU compute shaders in the render graph.

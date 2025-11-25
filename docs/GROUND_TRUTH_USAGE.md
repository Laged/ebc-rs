# Ground Truth Comparison Guide

This document explains how to compare edge detectors (Sobel, Canny, LoG) against ground truth for synthetic fan data.

## Overview

The ground truth system provides:
1. **Visual comparison** - Green overlay showing exact blade edges
2. **Programmatic comparison** - Precision/Recall/F1/IoU metrics via hypersearch

## Quick Start

### Generate Synthetic Data with Ground Truth

```bash
# Generate synthetic fan data (3 blades, 1200 RPM)
cargo run --bin generate_synthetic

# Output files:
# - data/synthetic/fan_test.dat        (event data)
# - data/synthetic/fan_test_truth.json (ground truth parameters)
```

### Visual Comparison

```bash
# Run the visualizer with synthetic data
cargo run --release --bin ebc-rs -- data/synthetic/fan_test.dat
```

**In the UI:**
1. Open "Edge Detection" panel (top-left)
2. Check the detector checkboxes to compare:
   - **Show Sobel (Yellow)** - Sobel edge detection
   - **Show Canny (Cyan)** - Canny edge detection
   - **Show LoG (Magenta)** - Laplacian of Gaussian
   - **Show Ground Truth (Green)** - Analytical blade edges

**What to look for:**
- **Perfect match**: Detector color overlaps exactly with green
- **False positives**: Detector color without green underneath
- **False negatives**: Green without detector color overlay

### Programmatic Comparison (Grid Search)

```bash
# Run hyperparameter search on synthetic data
cargo run --release --bin hypersearch -- \
    --data data/synthetic/fan_test.dat \
    --output results/synthetic_comparison.csv \
    --detectors sobel,canny,log \
    --window-sizes 50,100,200,500 \
    --thresholds 500,1000,2000,5000 \
    --frames 30
```

**Output:**
- `results/synthetic_comparison.csv` - All configurations tested
- `results/best_sobel.json` - Best Sobel configuration
- `results/best_canny.json` - Best Canny configuration
- `results/best_log.json` - Best LoG configuration

## Ground Truth Parameters

The `fan_test_truth.json` file contains:

```json
{
  "params": {
    "center_x": 640.0,      // Fan center X (pixels)
    "center_y": 360.0,      // Fan center Y (pixels)
    "radius_min": 50.0,     // Blade root radius (pixels)
    "radius_max": 200.0,    // Blade tip radius (pixels)
    "blade_count": 3,       // Number of blades
    "rpm": 1200.0,          // Rotations per minute
    "sweep_k": 0.50,        // Logarithmic spiral curvature
    "width_root_rad": 0.50, // Blade angular width at root (radians)
    "width_tip_rad": 0.30,  // Blade angular width at tip (radians)
    "edge_thickness_px": 2.0 // Edge detection band (pixels)
  },
  "frames": [...]
}
```

## Metrics Explained

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | How many detected edges are real edges |
| **Recall** | TP / (TP + FN) | How many real edges were detected |
| **F1 Score** | 2 * P * R / (P + R) | Harmonic mean of precision and recall |
| **IoU** | TP / (TP + FP + FN) | Intersection over Union |

Where:
- **TP** (True Positive): Pixel detected as edge AND is a ground truth edge
- **FP** (False Positive): Pixel detected as edge BUT is NOT a ground truth edge
- **FN** (False Negative): Pixel NOT detected as edge BUT IS a ground truth edge

**Ideal values:** All metrics = 1.0 (100%)

## CSV Output Columns

The hypersearch CSV includes:

```
detector,window_size,threshold,...,precision,recall,f1,iou,score
sobel,100,1000,...,0.85,0.72,0.78,0.64,12.5
canny,100,50/150,...,0.91,0.68,0.78,0.64,11.2
log,100,10,...,0.79,0.81,0.80,0.67,10.8
```

## Interpreting Results

### Good Configuration Signs
- **High Precision** (> 0.8): Few false edges detected
- **High Recall** (> 0.8): Most real edges found
- **Low Centroid Stability** (< 2.0 px): Stable fan center detection
- **High Inlier Ratio** (> 0.9): Good circle fit to blade tips

### Detector Characteristics

| Detector | Typical Strengths | Typical Weaknesses |
|----------|-------------------|-------------------|
| **Sobel** | Fast, stable centroid | Thicker edges, more FP |
| **Canny** | Thin edges, good precision | Can miss weak edges |
| **LoG** | Good at blob detection | Noisier, more parameters |

## Example Workflow

### 1. Generate Custom Synthetic Data

Edit `src/synthesis.rs` to change fan parameters:
```rust
let rpm = 2400.0;        // Faster rotation
let blade_count = 5;     // More blades
let radius = 300.0;      // Larger fan
```

Then regenerate:
```bash
cargo run --bin generate_synthetic
```

### 2. Find Optimal Parameters

```bash
# Coarse search
cargo run --release --bin hypersearch -- \
    --data data/synthetic/fan_test.dat \
    --output results/coarse.csv \
    --window-sizes 50,100,200,500,1000 \
    --thresholds 100,500,1000,2000,5000

# Fine search around best values
cargo run --release --bin hypersearch -- \
    --data data/synthetic/fan_test.dat \
    --output results/fine.csv \
    --window-sizes 80,90,100,110,120 \
    --thresholds 800,900,1000,1100,1200
```

### 3. Apply Best Config to Real Data

```bash
# Use the best configuration from synthetic testing
cat results/best_sobel.json
# {"detector":"sobel","window_size_us":100,"threshold":1000,...}

# Apply to real fan data
cargo run --release --bin ebc-rs -- data/fan/fan_const_rpm.dat
# Manually set the window size and threshold in UI
```

## Troubleshooting

### Ground Truth Not Showing
- Ensure you're using synthetic data (real data has no ground truth)
- Check that `fan_test_truth.json` exists alongside `fan_test.dat`
- Verify the checkbox "Show Ground Truth (Green)" is checked

### Metrics All Zero
- The ground truth compute pipeline only runs for synthetic data
- Check log output for "Loaded ground truth config: X blades..."

### Visual Mismatch
- Adjust window_size to match blade motion speed
- Increase edge_thickness_px in truth JSON for looser matching

## Architecture Reference

```
                    Synthetic Data Flow

┌─────────────────┐     ┌──────────────────┐
│ generate_       │────▶│ fan_test.dat     │
│ synthetic       │     │ fan_test_truth.  │
└─────────────────┘     │ json             │
                        └────────┬─────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   ebc-rs / hypertest    │
                    │                         │
                    │  ┌─────────────────┐    │
                    │  │ Event           │    │
                    │  │ Accumulation    │    │
                    │  └────────┬────────┘    │
                    │           │             │
                    │  ┌────────▼────────┐    │
                    │  │ Preprocess      │    │
                    │  │ (filters)       │    │
                    │  └────────┬────────┘    │
                    │           │             │
                    │     ┌─────┴─────┐       │
                    │     ▼     ▼     ▼       │
                    │  Sobel Canny  LoG       │
                    │     │     │     │       │
                    │     └─────┬─────┘       │
                    │           │             │
                    │  ┌────────▼────────┐    │
                    │  │ Ground Truth    │◀───┼── From JSON params
                    │  │ (analytical)    │    │
                    │  └────────┬────────┘    │
                    │           │             │
                    │  ┌────────▼────────┐    │
                    │  │ Visualizer      │    │   Colors:
                    │  │ (blend layers)  │────┼── Yellow: Sobel
                    │  └─────────────────┘    │   Cyan: Canny
                    │                         │   Magenta: LoG
                    │  ┌─────────────────┐    │   Green: Ground Truth
                    │  │ EdgeMetrics     │    │
                    │  │ (P/R/F1/IoU)    │────┼── To CSV
                    │  └─────────────────┘    │
                    └─────────────────────────┘
```

## Files Reference

| File | Purpose |
|------|---------|
| `src/synthesis.rs` | Synthetic data generator |
| `src/ground_truth.rs` | Config loader |
| `src/gpu/ground_truth.rs` | GPU pipeline |
| `assets/shaders/ground_truth.wgsl` | Compute shader |
| `src/hyperparams.rs` | EdgeMetrics, HyperResult |
| `src/bin/hypertest.rs` | Single config tester |
| `src/bin/hypersearch.rs` | Grid search runner |

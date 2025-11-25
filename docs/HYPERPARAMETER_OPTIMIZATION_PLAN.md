# Hyperparameter Optimization Plan for Edge Detection Pipeline

## Current Parameter Landscape

### Layer 0: Event Accumulation (`accumulation.wgsl`)
| Parameter | Source | Default | Description |
|-----------|--------|---------|-------------|
| `window_size` | `PlaybackState` | 100 μs | Time window for event accumulation |
| `current_time` | `PlaybackState` | - | Center of time window |

**Notes:**
- Events outside `[current_time, current_time + window_size]` are filtered
- Uses `atomicMax` to keep only the **latest** event per pixel (not accumulation!)
- `window_size` is critical but currently only tested at default value

### Layer 1: Sobel Edge Detection (`sobel.wgsl`)
| Parameter | Source | Default | Description |
|-----------|--------|---------|-------------|
| `threshold` | `EdgeParams` | 1000.0 | Gradient magnitude threshold |
| `filter_dead_pixels` | `EdgeParams` | **ON** | Skip pixels with no events at center |
| `filter_density` | `EdgeParams` | OFF | Require ≥5/9 active pixels in 3x3 |
| `filter_bidirectional` | `EdgeParams` | OFF | Require both Gx and Gy > 0.3×threshold |
| `filter_temporal` | `EdgeParams` | OFF | Require ≥500μs timestamp spread |

**Hardcoded values in shader:**
- Density threshold: `5u` (5/9 pixels)
- Temporal spread: `500.0` μs
- Bidirectional ratio: `0.3`

### Layer 2: Canny Edge Detection (`canny.wgsl`)
| Parameter | Source | Default | Used? | Description |
|-----------|--------|---------|-------|-------------|
| `canny_sigma` | `EdgeParams` | 1.4 | **NO** | Gaussian sigma (hardcoded kernel) |
| `canny_low_threshold` | `EdgeParams` | 50.0 | Yes | Weak edge threshold |
| `canny_high_threshold` | `EdgeParams` | 150.0 | Yes | Strong edge threshold |

**Hardcoded values in shader:**
- 5×5 Gaussian kernel (σ≈1.4)
- Direction quantization to 4 angles (0°, 45°, 90°, 135°)

### Layer 3: LoG Edge Detection (`log.wgsl`)
| Parameter | Source | Default | Used? | Description |
|-----------|--------|---------|-------|-------------|
| `log_sigma` | `EdgeParams` | 1.4 | **NO** | LoG sigma (hardcoded kernel) |
| `log_threshold` | `EdgeParams` | 10.0 | Yes | Edge strength threshold |

**Hardcoded values in shader:**
- 5×5 LoG kernel (σ≈1.4)

---

## Identified Issues

### 1. Unused Parameters
- `canny_sigma` and `log_sigma` are defined but **never used**
- Kernels are hardcoded for σ=1.4

### 2. Pre-filters Only Apply to Sobel
The 4 Sobel filters are valuable preprocessing steps:
- Dead pixel filter
- Event density filter
- Temporal variance filter
- Bidirectional gradient filter

**Problem:** Canny and LoG don't benefit from these filters!

### 3. Hardcoded Magic Numbers
Several thresholds are hardcoded in shaders:
- Sobel density: `5u` pixels
- Sobel temporal: `500.0` μs
- Sobel bidirectional: `0.3` ratio

These should be configurable parameters.

### 4. Accumulation Strategy
Current: `atomicMax` keeps only the **latest** event per pixel.
Alternative strategies:
- Event count accumulation
- Polarity-weighted accumulation
- Timestamp-weighted accumulation

### 5. Time Window Impact
`window_size` is critical but untested:
- Too small: sparse events, noisy edges
- Too large: motion blur, multiple blade positions overlap

---

## Proposed Architecture: Separation of Concerns

### New Pipeline Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 0: EVENT ACCUMULATION                  │
│  Input: Raw events   →   Output: Surface texture (timestamp)    │
│  Parameters: window_size, accumulation_mode                     │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: PRE-PROCESSING                      │
│  Shared filters applicable to ALL detectors:                    │
│  - Dead pixel filter (min activity threshold)                   │
│  - Event density filter (neighborhood activity)                 │
│  - Temporal variance filter (timestamp spread)                  │
│  Output: Filtered surface texture                               │
└─────────────────────────────────────────────────────────────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│   SOBEL DETECTOR  │ │   CANNY DETECTOR  │ │    LoG DETECTOR   │
│  - kernel: 3×3    │ │  - gaussian: 5×5  │ │  - kernel: 5×5    │
│  - threshold      │ │  - low_threshold  │ │  - threshold      │
│                   │ │  - high_threshold │ │                   │
│                   │ │  - NMS            │ │                   │
└───────────────────┘ └───────────────────┘ └───────────────────┘
                 │                │                │
                 └────────────────┼────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER N: POST-PROCESSING                     │
│  - Bidirectional filter (require both Gx, Gy)                   │
│  - Non-maximum suppression (already in Canny)                   │
│  - Edge thinning / skeletonization                              │
│  - Connected component filtering                                │
└─────────────────────────────────────────────────────────────────┘
```

### New Parameter Structure

```rust
/// Pre-processing parameters (shared across all detectors)
pub struct PreProcessParams {
    /// Filter 1: Minimum activity at center pixel
    pub min_center_activity: u32,        // default: 1 (any event)
    /// Filter 2: Minimum active pixels in 3×3 neighborhood
    pub min_density: u32,                // default: 0 (disabled), range: 1-9
    /// Filter 3: Minimum timestamp spread in neighborhood
    pub min_temporal_spread_us: f32,     // default: 0.0 (disabled)
}

/// Detector-specific parameters
pub struct SobelParams {
    pub threshold: f32,
}

pub struct CannyParams {
    pub sigma: f32,                      // Actually USE this to generate kernel
    pub low_threshold: f32,
    pub high_threshold: f32,
}

pub struct LogParams {
    pub sigma: f32,                      // Actually USE this to generate kernel
    pub threshold: f32,
}

/// Post-processing parameters
pub struct PostProcessParams {
    /// Require significant gradient in both X and Y
    pub bidirectional_filter: bool,
    pub bidirectional_ratio: f32,        // default: 0.3
    /// Morphological operations
    pub edge_thinning: bool,
    /// Minimum edge length (connected component filter)
    pub min_edge_length: u32,
}

/// Time window parameters
pub struct AccumulationParams {
    pub window_size_us: f32,             // default: 100.0
    pub mode: AccumulationMode,          // Latest, Count, WeightedSum
}
```

---

## Hyperparameter Testing Methodology

### 1. Grid Search Framework

```rust
/// A hyperparameter configuration to test
#[derive(Clone, Debug)]
pub struct HyperConfig {
    // Accumulation
    pub window_size_us: f32,

    // Pre-processing
    pub filter_dead_pixels: bool,
    pub min_density: u32,
    pub min_temporal_spread_us: f32,

    // Detector-specific
    pub detector: DetectorType,
    pub threshold: f32,  // Sobel/LoG
    pub canny_low: f32,
    pub canny_high: f32,

    // Post-processing
    pub bidirectional_filter: bool,
}

/// Results from a single configuration
#[derive(Clone, Debug)]
pub struct HyperResult {
    pub config: HyperConfig,
    pub avg_edge_count: f32,
    pub centroid_stability: f32,
    pub circle_fit_error: f32,
    pub inlier_ratio: f32,
    pub detected_blade_count: f32,
    pub processing_time_ms: f32,
}
```

### 2. Test Matrix Design

#### Phase 1: Window Size Sweep
Most critical parameter - affects all detectors.

| window_size_us | Expected Behavior |
|----------------|-------------------|
| 10 | Very sparse, noisy |
| 50 | Moderate density |
| 100 | Default |
| 200 | More events, potential blur |
| 500 | Dense, likely motion blur |
| 1000 | Very dense |
| 5000 | Heavy blur |

#### Phase 2: Pre-filter Combinations
Test each filter independently, then combinations.

| dead_pixels | density | temporal | Expected |
|-------------|---------|----------|----------|
| OFF | OFF | OFF | Baseline (noisy) |
| ON | OFF | OFF | Current default |
| ON | ON | OFF | Cleaner edges |
| ON | OFF | ON | Motion-aware |
| ON | ON | ON | Maximum filtering |

#### Phase 3: Detector Thresholds
For each detector, sweep the threshold range.

**Sobel:**
| threshold | Expected |
|-----------|----------|
| 100 | Many edges (noisy) |
| 500 | Moderate |
| 1000 | Default |
| 2000 | Fewer edges |
| 5000 | Only strong edges |

**Canny:**
| low | high | Expected |
|-----|------|----------|
| 20 | 60 | Sensitive |
| 50 | 150 | Default |
| 100 | 300 | Conservative |
| 150 | 450 | Very selective |

**LoG:**
| threshold | Expected |
|-----------|----------|
| 1 | Many edges |
| 5 | Moderate |
| 10 | Default |
| 20 | Selective |
| 50 | Very selective |

#### Phase 4: Post-filter Effects
| bidirectional | ratio | Expected |
|---------------|-------|----------|
| OFF | - | All edges |
| ON | 0.1 | Light filtering |
| ON | 0.3 | Default |
| ON | 0.5 | Strong filtering |

### 3. Metrics for Comparison

| Metric | Description | Good Value |
|--------|-------------|------------|
| `edge_pixel_count` | Total edges detected | Task-dependent |
| `edge_density` | edges / total_pixels | 0.01-0.05 typically |
| `centroid_stability` | Std dev of centroid position | **Lower is better** |
| `radius_stability` | Std dev of detected radius | Lower is better |
| `circle_fit_error` | RANSAC fit residual | Lower is better |
| `inlier_ratio` | % points fitting circle | Higher is better |
| `blade_count_variance` | Consistency of blade detection | Lower is better |
| `processing_time_ms` | GPU compute time | Lower is better |

### 4. Implementation Plan

```rust
/// Run a grid search over hyperparameters
pub fn hyperparameter_search(
    data_path: &Path,
    configs: &[HyperConfig],
    frames_per_config: usize,
) -> Vec<HyperResult> {
    // Due to Bevy's global state, we need to run each config
    // in a separate process or use subprocess spawning

    configs.iter().map(|config| {
        run_single_config(data_path, config, frames_per_config)
    }).collect()
}

/// Save results to CSV for analysis
pub fn export_results(results: &[HyperResult], path: &Path) {
    // CSV format for easy analysis in Python/R/Excel
}
```

---

## Implementation Tasks

### Task 1: Separate Pre-processing into Dedicated Shader
Create `preprocess.wgsl` that outputs a filtered surface texture.

**Changes:**
- New shader file
- New pipeline registration
- Update render graph: Event → Preprocess → Sobel/Canny/LoG

### Task 2: Add Missing Parameters to GPU Buffers
- Make `canny_sigma` and `log_sigma` actually generate kernels
- Add `min_density`, `min_temporal_spread` as configurable params
- Add `bidirectional_ratio` parameter

### Task 3: Create Hyperparameter Testing Framework
- `tests/hyperparameter_search.rs`
- `HyperConfig` and `HyperResult` structs
- Grid search runner
- CSV export for results

### Task 4: Window Size Investigation
Priority test: How does `window_size` affect edge quality?
- Run sweep: [10, 50, 100, 200, 500, 1000, 5000] μs
- For each: measure edge count, stability, circle fit

### Task 5: Pre-filter Effectiveness Study
For real fan data:
- Compare: no filters vs dead_pixel vs all_filters
- Measure impact on each detector

### Task 6: Threshold Optimization
For each detector:
- Find optimal threshold range for fan detection
- Create auto-tuning based on edge density target

---

## Expected Deliverables

1. **Refactored Pipeline** - Separated pre/post processing
2. **Hyperparameter Test Suite** - Automated grid search
3. **Results CSV** - All configurations tested
4. **Optimal Configurations** - Recommended params per use case
5. **Documentation** - Parameter tuning guide

---

## Priority Order

1. **Window size sweep** - Most impactful, quick to test
2. **Pre-filter separation** - Architecture improvement
3. **Threshold sweeps** - Per-detector optimization
4. **Dynamic kernel generation** - Use sigma params properly
5. **Advanced post-processing** - Edge thinning, connected components

---

## Notes

### Why Sobel has Best Stability on Real Data
Our tests showed Sobel (0.16px stability) >> Canny (3.26px) >> LoG (4.74px).

Hypotheses:
1. **Dead pixel filter** - Only Sobel has this enabled by default
2. **Threshold tuning** - Sobel threshold (1000) may be better calibrated
3. **Kernel size** - Sobel uses 3×3 vs Canny/LoG using 5×5

Test: Run Canny/LoG with dead_pixel filter and compare.

### Event Camera Characteristics
- Events are sparse and asynchronous
- Traditional image processing assumptions may not apply
- Temporal information (timestamps) is unique to event cameras
- Filters should leverage temporal data, not just spatial

### Future Considerations
- Machine learning-based hyperparameter tuning
- Dataset-specific profiles (different fans, speeds, lighting)
- Real-time adaptive thresholding

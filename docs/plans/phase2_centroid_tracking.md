# Phase 2 Implementation Plan: Centroid Tracking

## 1. Overview

**Goal**: Extend CMax-SLAM to track a moving fan (drifting centroid) autonomously.
**Core Mechanic**: 3D Optimization $[\omega, c_x, c_y]$ using 7-slice IWE variance maximization.
**Architecture**: Approach A (Full Numerical Gradient).

---

## 2. Synthetic Data Generation

### 2.1 Update `src/synthesis.rs`
-   Add `DriftConfig` and `OscillationConfig` structs.
-   Update `generate_fan_event` to calculate dynamic centroid $c(t)$.
-   Update `generate_fan_data_with_config` to write motion params to JSON header.

### 2.2 Update `src/bin/generate_synthetic.rs`
-   Add CLI args: `--drift-x`, `--drift-y`, `--osc-amp`, `--osc-freq`.
-   Generate `data/synthetic/fan_drift.dat` and `data/synthetic/fan_oscillate.dat`.

---

## 3. GPU Pipeline Updates

### 3.1 Shader: `assets/shaders/cmax_slam_warp.wgsl`
-   Define 7 slice indices.
-   Update `warp_event` signature to take `cx, cy`.
-   Implement 7 warp calls in `main` (Center, $\omega\pm$, $c_x\pm$, $c_y\pm$).

### 3.2 Shader: `assets/shaders/cmax_slam_reduce.wgsl`
-   Update `ContrastResult` struct to 8 fields (7 sums + count).
-   Update reduction logic to process 7 slices per thread.
-   Update workgroup shared memory to 7 arrays.

### 3.3 Rust: `src/cmax_slam/resources.rs`
-   Update `GpuCmaxSlamParams` with `delta_pos` and padding.
-   Update `GpuContrastResult` to match shader struct.
-   Update `ContrastValues` to hold 7 floats.

### 3.4 Rust: `src/cmax_slam/pipeline.rs`
-   Update `iwe_size` calculation (multiply by 7).
-   Update `contrast_result_size` calculation.

---

## 4. CPU System Updates

### 4.1 Rust: `src/cmax_slam/systems.rs`
-   Update `ExtractedCmaxSlamParams` to include `delta_pos`.
-   Update `CmaxSlamState` with `delta_pos`, `lr_omega`, `lr_centroid`, `centroid_history`.
-   Implement `receive_contrast_results`:
    -   Compute gradients for $\omega, c_x, c_y$.
    -   Apply Parabolic Step for $\omega$.
    -   Apply Gradient Ascent for $c_x, c_y$.
    -   Clamp and update state.
    -   Update convergence logic (3D variance).

### 4.2 Rust: `src/cmax_slam/readback.rs`
-   Update `ContrastValues` struct.
-   Update `readback_contrast_results` to read 8 u32s.

---

## 5. UI & Evaluation

### 5.1 UI: `src/compare/ui.rs`
-   Add Centroid position display.
-   Add Position Error (if GT available).
-   Add Velocity estimate.

### 5.2 Evaluation: `src/bin/evaluate_cmax_slam.rs`
-   Update `GroundTruthConfig` to support dynamic centroids.
-   Implement `evaluate_with_motion` loop.
-   Add metrics: Centroid Error (px), Tracking Latency.

---

## 6. Execution Steps

1.  **Data**: Generate `fan_drift.dat` and `fan_oscillate.dat`.
2.  **Shaders**: Modify `warp.wgsl` and `reduce.wgsl`.
3.  **Pipeline**: Update buffer sizes and structs in Rust.
4.  **Optimizer**: Implement the 3D update logic.
5.  **Verify**: Run `compare_live` on `fan_drift.dat` and watch the green box track the fan.

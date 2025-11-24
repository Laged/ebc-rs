# Fan Visualization Accuracy Testing Plan

This document outlines the strategy for programmatically testing the accuracy of the fan visualization and tracking algorithms.

## 1. Objectives

The primary goal is to ensure the `ebc-rs` application accurately tracks:
1.  **Centroid Position**: The center of the fan.
2.  **Fan Radius**: The physical boundary of the blades.
3.  **Blade Angles**: The instantaneous angular position of each blade.
4.  **RPM**: The rotational speed (derived from blade angles).

We need to quantify "jitter" and "drift" to ensure the visualization is stable and accurate.

## 2. Testing Methodology

We will use a **Data-Driven** testing approach using synthetic and recorded datasets with known (or estimated) ground truth.

### 2.1. Synthetic Data Generation

To rigorously test accuracy, we need data where the *exact* parameters are known. We will create a synthetic event generator (`src/bin/generate_synthetic_fan.rs`) that produces `.dat` files.

**Parameters to Control:**
*   **Resolution**: 1280x720 (standard).
*   **Blade Count**: 3.
*   **RPM**: Constant (e.g., 1200) or Variable (Ramp up/down).
*   **Radius**: Fixed (e.g., 200px).
*   **Centroid**: Fixed (Center) or Moving (Linear/Circular path).
*   **Noise Level**: Add random "salt and pepper" events to simulate sensor noise.
*   **Jitter**: Add random perturbations to event timestamps or positions.

**Output:** `tests/data/synthetic_fan_rpm1200.dat`

### 2.2. Ground Truth Definition

For synthetic data, the ground truth is inherent in the generation parameters.
For real data (`data/fan/fan_const_rpm.dat`), we will establish a "Silver Standard" by manually annotating a few frames or using a high-precision offline pass to generate a reference trajectory.

### 2.3. Test Harness (`src/tests.rs` or `tests/accuracy_test.rs`)

We will create an integration test that runs the `AnalysisPlugin` in headless mode (no window) against the test data.

**Workflow:**
1.  **Setup**: Initialize Bevy app with `AnalysisPlugin`, `EventsPlugin`, and a custom `HeadlessPlaybackPlugin`.
2.  **Load Data**: Load the `.dat` file.
3.  **Step**: Advance the simulation frame-by-frame (e.g., 16.6ms steps).
4.  **Assert**: At each frame, read the `FanAnalysis` resource and compare it with the Ground Truth.
5.  **Log**: Record the error delta (Detected - Truth) for each parameter.

## 3. Metrics

We will measure the following metrics:

| Metric | Description | Target / Tolerance |
| :--- | :--- | :--- |
| **Centroid Error** | Euclidean distance between detected and true center. | < 5.0 pixels |
| **Radius Error** | Absolute difference between detected and true radius. | < 5.0 pixels |
| **Angle Error** | Angular distance (shortest path) between detected blade and true blade. | < 5.0 degrees |
| **Jitter (Stability)** | Standard deviation of the error over a sliding window (e.g., 60 frames). | < 1.0 pixel/degree |
| **Convergence Time** | Time taken for the tracker to settle within tolerance from a cold start. | < 1.0 second |

## 4. Implementation Plan

### Phase 1: Synthetic Generator
- Create a binary that generates `.dat` files with mathematically perfect spirals of events.
- Include metadata (JSON) alongside the `.dat` file describing the ground truth functions $x(t), y(t), \theta(t)$.

### Phase 2: Headless Test Runner
- Implement a Bevy system that drives the `Time` resource manually.
- Mock the GPU readback (or run on actual GPU if available in CI) or refactor `AnalysisPlugin` to be testable without a window. *Note: Since our pipeline relies on Compute Shaders, we need a wgpu context. Bevy can run in headless mode with wgpu, but CI environments might need software rendering (Lavapipe).*

### Phase 3: CI Integration
- Run these tests on every PR.
- Fail if accuracy metrics regress beyond a threshold.

## 5. Example Test Case (Pseudocode)

```rust
#[test]
fn test_synthetic_fan_accuracy() {
    let mut app = create_headless_app();
    app.load_events("tests/data/synthetic_fan.dat");
    
    let ground_truth = load_ground_truth("tests/data/synthetic_fan.json");
    
    for frame in 0..600 { // 10 seconds at 60fps
        app.update();
        
        let analysis = app.world.resource::<FanAnalysis>();
        let time = app.world.resource::<Time>().elapsed_seconds();
        
        let expected = ground_truth.at(time);
        
        assert!((analysis.centroid - expected.centroid).length() < 5.0, 
            "Centroid drifted at t={}", time);
            
        // Check jitter
        if frame > 60 {
             let jitter = calculate_recent_jitter(&app);
             assert!(jitter < 1.0, "Output too jittery at t={}", time);
        }
    }
}
```

# Next Steps: Edge Detection to RPM Calculation

## Overview

This document outlines the path from our current edge detection implementation to robust, automated fan RPM calculation. The key insight is that we need **programmatic optimization** - the ability to measure edge quality without visual inspection.

## Current State (as of 2025-11-25)

### What We Have
- **Three GPU edge detectors**: Sobel (yellow), Canny (cyan), LoG (magenta)
- **Visual overlay**: All three can be toggled and viewed simultaneously
- **Parameter controls**: UI sliders for thresholds
- **Playback system**: Time navigation through event data

### What's Missing
1. **No GPU readback** - Edge textures stay on GPU, can't analyze in Rust
2. **No quality metrics** - Can't measure "how good" edges are
3. **No ground truth** - Nothing to compare against
4. **No automation** - Must manually tune parameters by eye
5. **No geometry extraction** - Can't convert edges to centroid/radius/RPM

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Event Data → Surface Texture → [Sobel|Canny|LoG] → Visualizer → Display   │
│                                       ↓                                      │
│                              (edges stay on GPU)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
                                       ↓ Phase 1
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TARGET PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Event Data → Surface Texture → [Sobel|Canny|LoG] → Edge Textures          │
│                                                           ↓                  │
│                                                    GPU Readback              │
│                                                           ↓                  │
│                                                    Edge Metrics              │
│                                                           ↓                  │
│                                                    Parameter Optimizer       │
│                                                           ↓                  │
│                                                    Geometry Extraction       │
│                                                           ↓                  │
│                                                    RPM Calculation           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: GPU Readback Infrastructure

**Goal**: Read edge detection results back to CPU for analysis

### 1.1 Add Readback Buffer Resource

**File**: `src/gpu/resources.rs`

```rust
/// Buffer for reading edge texture data back to CPU
#[derive(Resource, Default)]
pub struct EdgeReadbackBuffer {
    pub sobel_buffer: Option<Buffer>,
    pub canny_buffer: Option<Buffer>,
    pub log_buffer: Option<Buffer>,
    pub dimensions: UVec2,
    pub sobel_data: Vec<f32>,
    pub canny_data: Vec<f32>,
    pub log_data: Vec<f32>,
    pub ready: bool,
}
```

### 1.2 Add Texture-to-Buffer Copy

**File**: `src/gpu/readback.rs` (new file)

```rust
pub struct ReadbackNode;

impl Node for ReadbackNode {
    fn run(&self, ...) {
        // Copy edge textures to staging buffers
        encoder.copy_texture_to_buffer(
            sobel_texture.as_image_copy(),
            ImageCopyBuffer {
                buffer: &readback.sobel_buffer,
                layout: ImageDataLayout { ... }
            },
            extent
        );
        // Similar for canny, log
    }
}

pub fn map_readback_buffers(readback: ResMut<EdgeReadbackBuffer>) {
    // Async map buffers and copy to Vec<f32>
    // This runs on CPU after GPU work completes
}
```

### 1.3 Integrate into Render Graph

```
Event → Sobel → Canny → LoG → Readback → Camera
```

### Success Criteria
- [ ] Can read Sobel edge pixels as `Vec<f32>` in main world
- [ ] Can read Canny edge pixels as `Vec<f32>` in main world
- [ ] Can read LoG edge pixels as `Vec<f32>` in main world
- [ ] Readback doesn't significantly impact FPS (< 5% drop)

---

## Phase 2: Edge Quality Metrics

**Goal**: Quantify edge detection quality without visual inspection

### 2.1 Core Metrics Module

**File**: `src/metrics.rs` (new file)

```rust
#[derive(Debug, Clone, Default)]
pub struct EdgeMetrics {
    // Basic counts
    pub edge_pixel_count: u32,
    pub total_pixels: u32,
    pub edge_density: f32,  // edge_pixel_count / total_pixels

    // Spatial distribution
    pub centroid: Vec2,           // Center of mass of edge pixels
    pub std_dev: Vec2,            // Spread of edge pixels
    pub bounding_box: Rect,       // Min/max of edge pixels

    // Circular fit (for fan detection)
    pub circle_center: Vec2,
    pub circle_radius: f32,
    pub circle_fit_error: f32,    // Lower = better fit
    pub circle_inlier_ratio: f32, // Fraction of edges on circle

    // Angular distribution (for blade detection)
    pub angular_histogram: [u32; 360],  // Edges binned by angle from center
    pub angular_peaks: Vec<f32>,        // Detected blade angles
    pub angular_uniformity: f32,        // 0 = all in one direction, 1 = uniform

    // Temporal stability (across frames)
    pub frame_to_frame_iou: f32,  // Intersection over Union with previous frame
}

impl EdgeMetrics {
    pub fn compute(edge_data: &[f32], width: u32, height: u32) -> Self { ... }
    pub fn compute_with_previous(edge_data: &[f32], prev: &EdgeMetrics) -> Self { ... }
}
```

### 2.2 Circular Fit Algorithm (RANSAC)

```rust
/// RANSAC circle fitting for fan border detection
pub fn fit_circle_ransac(
    edge_pixels: &[(f32, f32)],
    iterations: u32,
    inlier_threshold: f32,
) -> Option<(Vec2, f32, f32)> {  // (center, radius, error)
    // 1. Random sample 3 points
    // 2. Fit circle through 3 points
    // 3. Count inliers (points within threshold of circle)
    // 4. Keep best model
    // 5. Refine with all inliers
}
```

### 2.3 Angular Histogram for Blade Detection

```rust
/// Build histogram of edge pixel angles relative to center
pub fn angular_histogram(
    edge_pixels: &[(f32, f32)],
    center: Vec2,
) -> [u32; 360] {
    // For each edge pixel:
    //   angle = atan2(y - center.y, x - center.x)
    //   histogram[angle_degrees] += 1
}

/// Find peaks in angular histogram (blade positions)
pub fn find_angular_peaks(
    histogram: &[u32; 360],
    min_prominence: u32,
) -> Vec<f32> {
    // Find local maxima with sufficient prominence
    // Return angles in radians
}
```

### 2.4 Display Metrics in UI

```rust
fn ui_system(..., metrics: Res<EdgeMetrics>) {
    egui::Window::new("Edge Metrics").show(ctx, |ui| {
        ui.label(format!("Edge pixels: {}", metrics.edge_pixel_count));
        ui.label(format!("Density: {:.4}", metrics.edge_density));
        ui.label(format!("Circle fit error: {:.2}", metrics.circle_fit_error));
        ui.label(format!("Detected blades: {}", metrics.angular_peaks.len()));
        // ... more metrics
    });
}
```

### Success Criteria
- [ ] EdgeMetrics computed every frame
- [ ] Circular fit detects fan border (when visible)
- [ ] Angular peaks detect blade count
- [ ] Metrics displayed in UI
- [ ] Frame-to-frame stability measured

---

## Phase 3: Automated Parameter Optimization

**Goal**: Find optimal detector parameters without manual tuning

### 3.1 Optimization Objective Function

```rust
/// Score a set of edge detection parameters
/// Higher = better for fan detection
pub fn score_parameters(metrics: &EdgeMetrics, config: &OptimizationConfig) -> f32 {
    let mut score = 0.0;

    // Reward: Good circular fit (fan border detected)
    if metrics.circle_fit_error < config.max_circle_error {
        score += config.weight_circle * (1.0 - metrics.circle_fit_error / config.max_circle_error);
    }

    // Reward: Correct number of blades
    let blade_diff = (metrics.angular_peaks.len() as i32 - config.expected_blades as i32).abs();
    score += config.weight_blades * (1.0 / (1.0 + blade_diff as f32));

    // Penalty: Too many edge pixels (noise)
    if metrics.edge_density > config.max_density {
        score -= config.penalty_density * (metrics.edge_density - config.max_density);
    }

    // Penalty: Too few edge pixels (missing edges)
    if metrics.edge_density < config.min_density {
        score -= config.penalty_sparse * (config.min_density - metrics.edge_density);
    }

    // Reward: Temporal stability
    score += config.weight_stability * metrics.frame_to_frame_iou;

    score
}
```

### 3.2 Parameter Sweep System

**File**: `src/optimization.rs` (new file)

```rust
#[derive(Resource)]
pub struct ParameterSweep {
    pub enabled: bool,
    pub current_params: EdgeParams,
    pub param_grid: Vec<EdgeParams>,
    pub results: Vec<(EdgeParams, f32)>,  // (params, score)
    pub best_params: Option<EdgeParams>,
    pub best_score: f32,
    pub current_index: usize,
    pub frames_per_config: u32,
    pub frame_count: u32,
}

impl ParameterSweep {
    pub fn new_grid_search(
        sobel_thresholds: &[f32],
        canny_low: &[f32],
        canny_high: &[f32],
        log_thresholds: &[f32],
    ) -> Self {
        // Generate all combinations
    }

    pub fn advance(&mut self, metrics: &EdgeMetrics, config: &OptimizationConfig) {
        self.frame_count += 1;

        if self.frame_count >= self.frames_per_config {
            // Score current configuration
            let score = score_parameters(metrics, config);
            self.results.push((self.current_params.clone(), score));

            if score > self.best_score {
                self.best_score = score;
                self.best_params = Some(self.current_params.clone());
            }

            // Move to next configuration
            self.current_index += 1;
            self.frame_count = 0;

            if self.current_index < self.param_grid.len() {
                self.current_params = self.param_grid[self.current_index].clone();
            } else {
                self.enabled = false;
                info!("Sweep complete. Best score: {}", self.best_score);
            }
        }
    }
}
```

### 3.3 UI for Optimization

```rust
fn optimization_ui(...) {
    egui::Window::new("Parameter Optimization").show(ctx, |ui| {
        if ui.button("Start Grid Search").clicked() {
            sweep.enabled = true;
            sweep.reset();
        }

        ui.label(format!("Progress: {}/{}", sweep.current_index, sweep.param_grid.len()));
        ui.label(format!("Best score: {:.3}", sweep.best_score));

        if let Some(best) = &sweep.best_params {
            ui.label(format!("Best Sobel threshold: {}", best.threshold));
            // ... show other params

            if ui.button("Apply Best").clicked() {
                *edge_params = best.clone();
            }
        }
    });
}
```

### Success Criteria
- [ ] Grid search over parameter space works
- [ ] Objective function rewards good fan detection
- [ ] Best parameters are found automatically
- [ ] Results can be exported for analysis
- [ ] UI shows progress and allows applying best params

---

## Phase 4: Geometry Extraction Pipeline

**Goal**: Convert edges to fan geometry (centroid, radius, blades)

### 4.1 Fan Geometry Resource

```rust
#[derive(Resource, Default)]
pub struct FanGeometry {
    // Core geometry
    pub centroid: Vec2,
    pub radius: f32,
    pub blade_count: u32,
    pub blade_angles: Vec<f32>,  // Current angles in radians

    // Confidence scores
    pub centroid_confidence: f32,
    pub radius_confidence: f32,
    pub blade_confidence: f32,

    // Temporal tracking
    pub angular_velocity: f32,   // rad/s
    pub rpm: f32,
    pub rpm_history: VecDeque<f32>,  // For smoothing

    // Detection state
    pub detected: bool,
    pub frames_since_detection: u32,
}
```

### 4.2 Geometry Extraction System

```rust
fn extract_geometry(
    readback: Res<EdgeReadbackBuffer>,
    mut geometry: ResMut<FanGeometry>,
    edge_params: Res<EdgeParams>,
) {
    if !readback.ready { return; }

    // 1. Get edge pixels from best detector
    let edge_data = select_best_detector(&readback, &edge_params);
    let edge_pixels = extract_edge_pixels(edge_data, readback.dimensions);

    // 2. Fit circle (RANSAC)
    if let Some((center, radius, error)) = fit_circle_ransac(&edge_pixels, 100, 5.0) {
        geometry.centroid = center;
        geometry.radius = radius;
        geometry.centroid_confidence = 1.0 - error.min(1.0);
        geometry.radius_confidence = 1.0 - error.min(1.0);
    }

    // 3. Build angular histogram from center
    let histogram = angular_histogram(&edge_pixels, geometry.centroid);

    // 4. Detect blades from peaks
    let peaks = find_angular_peaks(&histogram, 10);
    geometry.blade_angles = peaks;
    geometry.blade_count = peaks.len() as u32;

    geometry.detected = geometry.centroid_confidence > 0.5;
}
```

### 4.3 Visualize Extracted Geometry

```rust
fn draw_geometry_gizmos(
    mut gizmos: Gizmos,
    geometry: Res<FanGeometry>,
) {
    if !geometry.detected { return; }

    // Draw detected circle (green)
    gizmos.circle_2d(geometry.centroid, geometry.radius, Color::GREEN);

    // Draw centroid cross
    gizmos.line_2d(
        geometry.centroid - Vec2::X * 20.0,
        geometry.centroid + Vec2::X * 20.0,
        Color::RED
    );
    gizmos.line_2d(
        geometry.centroid - Vec2::Y * 20.0,
        geometry.centroid + Vec2::Y * 20.0,
        Color::RED
    );

    // Draw detected blade lines
    for angle in &geometry.blade_angles {
        let direction = Vec2::new(angle.cos(), angle.sin());
        let start = geometry.centroid;
        let end = geometry.centroid + direction * geometry.radius;
        gizmos.line_2d(start, end, Color::YELLOW);
    }
}
```

### Success Criteria
- [ ] Centroid detected from edge pixels
- [ ] Radius detected from circular fit
- [ ] Blade count and angles detected from histogram
- [ ] Geometry overlaid on visualization
- [ ] Confidence scores indicate detection quality

---

## Phase 5: RPM Calculation

**Goal**: Calculate rotation speed from blade angle tracking

### 5.1 Blade Tracking Across Frames

```rust
/// Track blade angles across frames to calculate rotation
pub struct BladeTracker {
    pub previous_angles: Vec<f32>,
    pub angle_history: VecDeque<(f64, Vec<f32>)>,  // (timestamp, angles)
    pub angular_velocity_estimate: f32,
    pub rpm_estimate: f32,
    pub kalman_state: KalmanState,  // For smooth estimation
}

impl BladeTracker {
    pub fn update(&mut self, timestamp: f64, current_angles: &[f32]) {
        if self.previous_angles.is_empty() {
            self.previous_angles = current_angles.to_vec();
            return;
        }

        // Match current blades to previous blades
        let (matches, delta_angle) = match_blades(&self.previous_angles, current_angles);

        // Calculate angular velocity
        let dt = timestamp - self.angle_history.back().map(|(t, _)| *t).unwrap_or(timestamp);
        if dt > 0.0 {
            let omega = delta_angle / dt as f32;

            // Update Kalman filter
            self.kalman_state.update(omega);
            self.angular_velocity_estimate = self.kalman_state.estimate;

            // Convert to RPM: omega (rad/s) * 60 / (2*PI)
            self.rpm_estimate = self.angular_velocity_estimate * 60.0 / (2.0 * std::f32::consts::PI);
        }

        self.previous_angles = current_angles.to_vec();
        self.angle_history.push_back((timestamp, current_angles.to_vec()));

        // Keep limited history
        while self.angle_history.len() > 100 {
            self.angle_history.pop_front();
        }
    }
}

/// Match blades between frames (handles rotation ambiguity)
fn match_blades(prev: &[f32], curr: &[f32]) -> (Vec<(usize, usize)>, f32) {
    // Hungarian algorithm or simple greedy matching
    // Account for blade periodicity (e.g., 3 blades = 120° symmetry)
}
```

### 5.2 RPM Display and Export

```rust
fn rpm_ui(...) {
    egui::Window::new("RPM").show(ctx, |ui| {
        ui.heading(format!("{:.0} RPM", geometry.rpm));

        // RPM history plot
        egui::plot::Plot::new("rpm_plot")
            .show(ui, |plot_ui| {
                let points: Vec<[f64; 2]> = geometry.rpm_history
                    .iter()
                    .enumerate()
                    .map(|(i, rpm)| [i as f64, *rpm as f64])
                    .collect();
                plot_ui.line(egui::plot::Line::new(points));
            });

        ui.label(format!("Angular velocity: {:.2} rad/s", geometry.angular_velocity));
        ui.label(format!("Blade count: {}", geometry.blade_count));

        if ui.button("Export RPM Data").clicked() {
            export_rpm_json(&geometry);
        }
    });
}
```

### 5.3 Ground Truth Comparison (for Testing)

```rust
#[derive(Resource)]
pub struct GroundTruth {
    pub entries: Vec<GroundTruthEntry>,
}

#[derive(Debug, Clone)]
pub struct GroundTruthEntry {
    pub timestamp: f64,
    pub rpm: f32,
    pub centroid: Vec2,
    pub radius: f32,
    pub blade_angles: Vec<f32>,
}

fn compare_to_ground_truth(
    geometry: &FanGeometry,
    truth: &GroundTruth,
    current_time: f64,
) -> AccuracyMetrics {
    // Find closest ground truth entry
    let truth_entry = truth.entries.iter()
        .min_by_key(|e| ((e.timestamp - current_time).abs() * 1000.0) as i64)
        .unwrap();

    AccuracyMetrics {
        rpm_error: (geometry.rpm - truth_entry.rpm).abs(),
        rpm_error_percent: (geometry.rpm - truth_entry.rpm).abs() / truth_entry.rpm * 100.0,
        centroid_error: geometry.centroid.distance(truth_entry.centroid),
        radius_error: (geometry.radius - truth_entry.radius).abs(),
        blade_count_correct: geometry.blade_count == truth_entry.blade_angles.len() as u32,
    }
}
```

### Success Criteria
- [ ] Blade angles tracked across frames
- [ ] Angular velocity calculated correctly
- [ ] RPM displayed in UI
- [ ] RPM history plotted
- [ ] Kalman filter smooths noisy estimates
- [ ] Export to JSON works
- [ ] Accuracy measured against ground truth

---

## Implementation Order

### Sprint 1: Foundation (GPU Readback)
1. Add `EdgeReadbackBuffer` resource
2. Implement `ReadbackNode` for texture-to-buffer copy
3. Add buffer mapping system
4. Verify data accessible in main world
5. Add simple edge count display to UI

### Sprint 2: Metrics
1. Implement `EdgeMetrics` struct
2. Add basic metrics (count, density, centroid)
3. Implement RANSAC circle fit
4. Implement angular histogram
5. Display all metrics in UI

### Sprint 3: Optimization
1. Implement `ParameterSweep` resource
2. Define objective function
3. Add grid search logic
4. Add optimization UI
5. Test on synthetic data

### Sprint 4: Geometry
1. Implement `FanGeometry` resource
2. Add geometry extraction system
3. Add gizmo visualization
4. Test on real fan data

### Sprint 5: RPM
1. Implement `BladeTracker`
2. Add Kalman filter
3. Add RPM UI with plot
4. Add JSON export
5. Add ground truth comparison

---

## Testing Strategy

### Unit Tests
- Circle fitting with known circles
- Angular histogram with known angles
- Blade matching with known rotations
- RPM calculation with known angular velocity

### Integration Tests
- Full pipeline with synthetic fan data
- Accuracy vs ground truth at various RPMs
- Performance (FPS) with readback enabled

### Visual Tests (Manual)
- Geometry overlay matches visible fan
- RPM display matches visual rotation speed
- Parameter optimization improves detection

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU readback too slow | Use async mapping, only read when needed |
| RANSAC fails on sparse edges | Fall back to center-of-mass, use multiple detectors |
| Blade matching ambiguous | Use blade count constraint, temporal smoothing |
| RPM jitter | Kalman filter, longer averaging window |
| Memory pressure from history | Limit history size, compress older data |

---

## Dependencies

- Bevy 0.17 (render graph, gizmos)
- wgpu (buffer mapping)
- nalgebra or glam (linear algebra for RANSAC)
- serde_json (export)
- egui (UI)

---

## Success Metrics (End Goal)

| Metric | Target |
|--------|--------|
| RPM accuracy | < 5% error |
| Centroid accuracy | < 10 pixels |
| Radius accuracy | < 5% error |
| Blade count | 100% correct |
| Frame rate | > 30 FPS |
| Latency | < 100ms from event to RPM |

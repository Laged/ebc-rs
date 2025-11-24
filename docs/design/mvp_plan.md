# Gold Standard Architecture & MVP Plan

## 1. The "Gold Standard" Architecture

We are establishing a modular, research-backed architecture that separates **Data Ingestion**, **Processing Layers**, and **Visualization**.

### Core Concept: Multi-Layer Visualization
The system will support multiple parallel "Analysis Layers" that process the raw event stream and produce a renderable overlay.

*   **Layer 0: Raw Data (Red/Blue)**
    *   **Source:** `EventAccumulationNode`
    *   **Technique:** Temporal decay of raw events.
    *   **Purpose:** Ground truth visualization.

*   **Layer 1: Simple Edge Detection (Yellow) - THE MVP**
    *   **Technique:** **Spatial Timestamp Gradient (STG)**.
    *   **Theory:** In a high-speed event stream, the "Time Surface" (accumulated timestamps) forms a smooth manifold where the object moves. The spatial gradient of this surface ($\nabla T$) is highest at the moving edges. By applying a **Sobel Operator** to the timestamp texture, we can cheaply and robustly detect current edges without complex motion compensation.
    *   **Role:** Fast, lightweight, "always on" edge detection.

*   **Layer 2: CMAX Reconstruction (Purple) - FUTURE STEP**
    *   **Technique:** **Contrast Maximization (CMAX)** (as per `docs/research/cmax-theory.md`).
    *   **Theory:** Warping events by a candidate velocity $\omega$ to a reference time $t_{ref}$ and maximizing the variance of the resulting Image of Warped Events (IWE).
    *   **Role:** High-precision, physics-aware reconstruction for RPM estimation.

---

## 2. MVP Implementation Plan: Layer 0 + Layer 1

**Goal:** Implement the **Spatial Timestamp Gradient (STG)** pipeline and composite it (Yellow) over the Raw Data (Red/Blue).

### A. Files to CREATE

#### 1. `assets/shaders/spatial_gradient.wgsl` (The "Simple" Algorithm)
*   **Purpose:** Compute shader for Layer 1.
*   **Inputs:** `surface_texture` (R32Uint, containing timestamps).
*   **Outputs:** `gradient_texture` (R8Unorm, edge magnitude).
*   **Algorithm:**
    ```wgsl
    // 1. Load neighbors (3x3 kernel) from surface_texture
    // 2. Extract timestamps (ignoring polarity)
    // 3. Apply Sobel X and Sobel Y kernels
    // 4. Magnitude = sqrt(Gx^2 + Gy^2)
    // 5. Threshold and write to output (1.0 for edge, 0.0 for background)
    ```

### B. Files to MODIFY

#### 1. `src/plugins.rs`
*   **Action:** Clean slate.
*   **Change:** Remove `AnalysisPlugin`. Ensure only `EventRenderPlugin` is active.

#### 2. `src/gpu.rs`
*   **Action:** Infrastructure for the new layer.
*   **Add:**
    *   `GradientPipeline` struct.
    *   `GradientNode` struct.
    *   `GradientTexture` resource (The destination for the yellow layer).
*   **Keep:** `EventAccumulationNode` (Layer 0).

#### 3. `src/render.rs`
*   **Action:** Render Graph & Visualization.
*   **Render Graph:**
    *   Chain: `EventAccumulationNode` -> `GradientNode` -> `CameraDriverLabel`.
*   **Material (`EventMaterial`):**
    *   Add `gradient_texture` (Binding 2).
    *   Add `show_gradient` (Uniform flag).
*   **UI:**
    *   Add checkbox: "Show Edge Detection (Yellow)".
    *   Keep: Playback controls.
    *   Remove: All "RPM/Analysis" controls.

#### 4. `assets/shaders/visualizer.wgsl`
*   **Action:** Composite Layer 0 and Layer 1.
*   **Logic:**
    ```wgsl
    // Layer 0: Raw
    let raw_color = decode_raw(surface_texture);
    
    // Layer 1: Gradient
    let edge_val = textureLoad(gradient_texture, ...).r;
    
    // Composite
    if (show_gradient && edge_val > 0.0) {
        // Yellow Overlay (1.0, 1.0, 0.0)
        // Mix with raw or draw on top
        return vec4(1.0, 1.0, 0.0, 1.0); 
    }
    return raw_color;
    ```

### C. Files to DELETE / IGNORE

We are removing the "complex" attempts to clear the path for the Gold Standard.

*   `src/analysis.rs` (Delete)
*   `assets/shaders/centroid.wgsl` (Delete)
*   `assets/shaders/radial_profile.wgsl` (Delete)
*   `assets/shaders/angular_histogram.wgsl` (Delete)
*   `assets/shaders/cmax_optimization.wgsl` (Delete)
*   `src/gizmos.rs` (Delete - we are doing pixel-perfect shader rendering, not vector gizmos)

---

## 3. Verification Steps

1.  **Build:** `cargo build` (Ensure no unused code warnings from deleted files).
2.  **Run:** `cargo run -- --data-path data/fan/fan_const_rpm.dat`.
3.  **Test:**
    *   See Red/Blue raw events.
    *   Toggle "Show Edge Detection".
    *   See **Yellow** edges highlighting the fan blades.
    *   Verify performance (should be 60fps+).

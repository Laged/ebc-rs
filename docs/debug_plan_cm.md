# Debug Plan: Contrast Maximization Pipeline

## Objective
Isolate and fix the "spinning green circle" issue and ensure accurate fan tracking using the Edge-Informed CM pipeline.

## Current Symptoms
- Visualization shows a "spinning green circle" instead of a sharp fan.
- Tracking is not locking onto the fan blades.
- Issue persists even when disabling the correlation term.

## Hypotheses
1.  **Input Data Issue**: The `ShortWindowSurfaceImage` (reference) might be empty, noisy, or incorrectly generated.
2.  **Warping Issue**: The `cm_warp.wgsl` shader might be producing garbage IWEs (e.g., all zeros or all ones), or the `t_ref` alignment is still wrong.
3.  **Contrast Issue**: The contrast metric might be dominated by noise or artifacts (e.g., border effects).
4.  **Selection Issue**: The `cm_select.wgsl` shader might be picking a random or default `omega` (e.g., index 0) due to invalid scores (NaN/Inf).
5.  **Visualization Issue**: The "green circle" might be a misleading visualization of the result (e.g., showing the accumulation buffer instead of the warped image).

## Step-by-Step Debugging Strategy

### Step 1: Verify Input Data (Short-Window Reference)
- **Goal**: Confirm that `ShortWindowSurfaceImage` contains a sharp, valid event image.
- **Action**:
    - Modify `src/event_renderer.rs` to display `ShortWindowSurfaceImage` in "Layer 1" (replacing Sobel for now) or a new debug layer.
    - **Expected**: A sharp, sparse image of the fan blades.
    - **If Fail**: The accumulation logic in `accumulation.wgsl` or `prepare_events` is broken.

### Step 2: Verify Basic Warping (Disable Optimization)
- **Goal**: Confirm that `cm_warp.wgsl` can produce a valid IWE for a *known* good omega.
- **Action**:
    - Hardcode `omega` to a known value (e.g., 0.0 or a rough estimate like 10.0 rad/s) in `cm_warp.wgsl` or `systems.rs`.
    - Visualize the resulting IWE in `CmImage`.
    - **Expected**: A recognizable fan image (blurred if omega is 0, sharper if omega is close).
    - **If Fail**: The warping math or coordinate system is wrong.

### Step 3: Verify Contrast Calculation (Pure CM)
- **Goal**: Confirm that the contrast metric peaks at the correct omega.
- **Action**:
    - Disable the "Correlation" term in `cm_contrast.wgsl` (revert to pure contrast).
    - Log the computed contrast values for a range of omegas (using `readback` or temporary debug buffer).
    - **Expected**: A curve with a distinct peak around the true RPM.
    - **If Fail**: The contrast metric is flat, noisy, or dominated by artifacts.

### Step 4: Verify Edge-Informed Correlation
- **Goal**: Confirm that the correlation term aligns the IWE with the reference.
- **Action**:
    - Re-enable the "Correlation" term.
    - Visualize the `Edge Map` (Sobel of Short-Window) and the `IWE` overlaid.
    - **Expected**: The IWE should align with the Edge Map at the correct omega.

### Step 5: Verify Selection Logic
- **Goal**: Confirm that `cm_select.wgsl` picks the index with the maximum score.
- **Action**:
    - Check the `result` buffer (index and score) via readback.
    - **Expected**: The selected index corresponds to the peak observed in Step 3.

## Immediate Action Items
1.  **Visualize Short-Window**: Modify `event_renderer.rs` to show `ShortWindowSurfaceImage` on Layer 1.
2.  **Simplify Shader**: Comment out correlation in `cm_contrast.wgsl` (User already tried, but we will verify).
3.  **Hardcode Omega**: Force `omega = 0` in `cm_warp.wgsl` to see if we get a valid accumulation.

# CMax-SLAM Implementation Review & Guide

## 1. Comparison: Paper vs. Current Implementation

The paper **"CMax-SLAM: Event-based Rotational-Motion Bundle Adjustment and SLAM System using Contrast Maximization"** describes a comprehensive system for estimating 3-DOF rotational motion. Here is how your current `ebc-rs` implementation compares:

| Feature | Paper (CMax-SLAM) | Current `ebc-rs` Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Motion Model** | Continuous B-Spline on SO(3) | Constant Angular Velocity ($\omega$) per window | ⚠️ **Simplified** (Appropriate for fan RPM) |
| **Optimization** | Levenberg-Marquardt / Conjugate Gradient | **Missing** (Currently relies on Ground Truth) | ❌ **Critical Gap** |
| **Objective** | Variance of Image of Warped Events (IWE) | Variance (Planned) | 🟡 **Partially Implemented** |
| **Warping** | 3D Rotation -> Equirectangular Projection | 2D Planar Rotation -> Cartesian | ✅ **Appropriate** (For 2D fan) |
| **Compute** | CPU (C++) | GPU (WGSL Compute Shaders) | ✅ **Superior** (For real-time) |
| **Differentiation**| Analytical Jacobian w.r.t Control Points | Numerical Difference ($\omega \pm \delta$) | 🟡 **Acceptable** (For 1D parameter) |

### Key Findings
1.  **Warping Logic is Correct**: Your `cmax_slam_warp.wgsl` correctly implements the motion compensation logic ($x' = R(\omega t)x$) and bilinear voting. This matches the "Event Warping" phase of the paper.
2.  **Missing the "Max" in CMax**: The current implementation *warps* events but does not *maximize* contrast. It calculates IWEs for $\omega, \omega+\delta, \omega-\delta$ but doesn't actually compute the variance (contrast) of these images to update $\omega$. It currently relies on `gt_config` (Ground Truth) to set $\omega$.
3.  **Numerical vs Analytical Gradient**: The paper derives analytical Jacobians. Your approach of computing 3 parallel IWEs to estimate the gradient numerically is a smart move for a 1D parameter space (RPM only) on the GPU, as it avoids complex derivative math in the shader.

---

## 2. Step-by-Step Implementation Guide

To align with the paper's core value (estimating motion without ground truth), you need to close the optimization loop.

### Step 1: Implement Variance Reduction (The "Contrast" Metric)

The paper maximizes variance: $\sigma^2 = \frac{1}{N} \sum (I_{ij} - \mu)^2$. Since the mean $\mu$ is constant for a fixed number of events, this simplifies to maximizing $\sum I_{ij}^2$.

**Current Gap**: `cmax_slam_warp.wgsl` accumulates the image but doesn't sum the squares.

**Action**: Create a reduction shader to sum the squared pixel values of your IWEs.

#### `assets/shaders/cmax_slam_reduce.wgsl` (New File)
```wgsl
// Reduce IWE to single variance value (sum of squares)
// We need to reduce 3 separate slices: Center, Plus, Minus

struct ContrastResult {
    sum_sq_center: atomic<u32>,
    sum_sq_plus: atomic<u32>,
    sum_sq_minus: atomic<u32>,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> iwe_buffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: ContrastResult;

const SLICE_SIZE: u32 = 1280u * 720u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= SLICE_SIZE { return; }

    // 1. Read values (scaled by 256 from bilinear voting)
    let val_c = iwe_buffer[idx];
    let val_p = iwe_buffer[idx + SLICE_SIZE];
    let val_m = iwe_buffer[idx + 2u * SLICE_SIZE];

    // 2. Square them (careful with overflow - might need to downscale or use float buffer)
    // For safety, let's convert to float or downshift before squaring if values are high
    // Assuming max events per pixel < 255 for short windows
    
    let sq_c = (val_c >> 8u) * (val_c >> 8u); 
    let sq_p = (val_p >> 8u) * (val_p >> 8u);
    let sq_m = (val_m >> 8u) * (val_m >> 8u);

    // 3. Atomic Accumulate
    if (sq_c > 0u) { atomicAdd(&result.sum_sq_center, sq_c); }
    if (sq_p > 0u) { atomicAdd(&result.sum_sq_plus, sq_p); }
    if (sq_m > 0u) { atomicAdd(&result.sum_sq_minus, sq_m); }
}
```

### Step 2: CPU Readback & Optimization Loop

The paper uses Conjugate Gradient. For your 1D case (RPM), a simple Gradient Ascent or Newton step is sufficient.

**Logic**:
1.  Dispatch `warp` shader (generates 3 IWEs).
2.  Dispatch `reduce` shader (computes 3 variances: $V_c, V_+, V_-$).
3.  **Readback** the 3 values to CPU.
4.  **Update $\omega$**:
    $$ \text{Gradient} \approx \frac{V_+ - V_-}{2\delta} $$
    $$ \text{Curvature} \approx \frac{V_+ - 2V_c + V_-}{\delta^2} $$
    $$ \omega_{new} = \omega_{old} + \alpha \cdot \text{Gradient} $$
    *(Or use parabolic interpolation to find the peak)*

### Step 3: Pipeline Integration

Modify `src/cmax_slam/pipeline.rs` and `systems.rs` to include this readback loop.

**Rust System Update (`systems.rs`):**

```rust
pub fn update_omega_system(
    mut state: ResMut<CmaxSlamState>,
    buffers: Res<CmaxSlamBuffers>,
    render_device: Res<RenderDevice>,
) {
    // 1. Map 'contrast' buffer for read
    let slice = buffers.contrast.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    render_device.poll(Maintain::Wait);
    
    // 2. Read data
    let data = slice.get_mapped_range();
    let result: GpuContrastResult = bytemuck::from_bytes(&data).clone();
    drop(data);
    buffers.contrast.unmap();

    // 3. Compute Gradient
    let v_c = result.sum_sq_center as f32;
    let v_p = result.sum_sq_plus as f32;
    let v_m = result.sum_sq_minus as f32;

    // Avoid division by zero
    if v_c == 0.0 { return; }

    // Parabolic fit to find peak
    // y = ax^2 + bx + c
    // We want x where dy/dx = 0
    let numerator = v_p - v_m;
    let denominator = 2.0 * (v_p - 2.0 * v_c + v_m);
    
    if denominator.abs() > 1e-5 {
        let step = -numerator / denominator * state.delta_omega;
        // Limit step size for stability
        let max_step = state.omega * 0.1; 
        state.omega += step.clamp(-max_step, max_step);
    } else {
        // Fallback to simple gradient ascent
        let grad = v_p - v_m;
        state.omega += grad * 0.001; 
    }
}
```

## 3. Recommended Implementation Plan

1.  **Add the Reduction Shader**: Create `cmax_slam_reduce.wgsl` to compute the sum of squares.
2.  **Update Pipeline**: Add the reduction pass to `CmaxSlamNode` in `pipeline.rs`.
3.  **Implement Readback**: Add a system to read the contrast values back to the CPU.
4.  **Enable Optimization**: Switch `update_cmax_slam_omega` from using Ground Truth to using the calculated gradient.

This will convert your current "Ground Truth Visualizer" into a true "CMax-SLAM" estimator that finds the RPM autonomously.

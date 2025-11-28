# Phase 2 Design: Joint Centroid & RPM Optimization

## 1. Overview

**Goal**: Extend the CMax-SLAM estimator to track a moving center of rotation $(c_x, c_y)$ simultaneously with angular velocity $\omega$.
**Motivation**: In real scenarios (drones), the object is never perfectly stationary. A fixed centroid assumption leads to blur and loss of tracking.
**Approach**: 3-Parameter Optimization $[\omega, c_x, c_y]$ using Numerical Gradients on the GPU.

---

## 2. Mathematical Formulation

### 2.1 The Warp Function
For an event $e_k = (x_k, y_k, t_k)$ and reference time $t_{ref}$, we want to find the position $x'_{k}$ at $t_{ref}$.
Assuming constant angular velocity $\omega$ around a center $\mathbf{c} = (c_x, c_y)$:

$$ \mathbf{x}'_k = \mathbf{c} + R(-\omega (t_k - t_{ref})) (\mathbf{x}_k - \mathbf{c}) $$

Where $R(\theta)$ is the 2D rotation matrix.

### 2.2 The Objective
Maximize the variance of the Image of Warped Events (IWE):
$$ \theta^* = \arg\max_{\omega, c_x, c_y} \text{Var}(\text{IWE}(\omega, c_x, c_y)) $$

### 2.3 Gradient Computation (Numerical)
To optimize 3 parameters, we need the gradient $\nabla C$. We use central differences for stability.
This requires evaluating the cost function at **7 points** per frame:
1.  **Center**: $(\omega, c_x, c_y)$
2.  **Omega $\pm$**: $(\omega \pm \delta_\omega, c_x, c_y)$
3.  **Cx $\pm$**: $(\omega, c_x \pm \delta_p, c_y)$
4.  **Cy $\pm$**: $(\omega, c_x, c_y \pm \delta_p)$

---

## 3. Architecture Changes

### 3.1 GPU Memory (IWE Buffer)
We need to store 7 accumulated images (slices) instead of 3.
-   **Size**: $1280 \times 720 \times 7 \times 4 \text{ bytes} \approx 25.8 \text{ MB}$.
-   **Layout**: Flat array, slices offset by `SLICE_SIZE`.

### 3.2 Warp Shader (`cmax_slam_warp.wgsl`)
Modified to compute 7 warped coordinates for each event.
-   **Input**: Event $(x, y, t)$.
-   **Logic**:
    -   Calculate $\Delta t = t - t_{ref}$.
    -   Compute base warp.
    -   Compute perturbed warps (different $\omega$ or different $\mathbf{c}$).
    -   Atomic add to 7 distinct buffer regions.

### 3.3 Reduction Shader (`cmax_slam_reduce.wgsl`)
Modified to reduce 7 slices.
-   **Output**: `ContrastResult` with 7 sums of squares.
-   **Optimization**: Can still use workgroup reduction, just more registers per thread.

### 3.4 Readback & Optimizer
-   **Readback**: Transfer 7 floats (plus pixel count) to CPU.
-   **Optimizer**:
    -   Compute 3 partial derivatives.
    -   Update state vector: $\mathbf{s}_{new} = \mathbf{s}_{old} + \alpha \cdot \nabla C$.
    -   Adaptive step sizes for $\omega$ (rad/us) vs $\mathbf{c}$ (pixels) as they have different scales.

---

## 4. Implementation Details

### 4.1 Shader Logic (Pseudo-code)

```wgsl
// 7 Slices
const IDX_CENTER = 0u;
const IDX_OMEGA_P = 1u;
const IDX_OMEGA_M = 2u;
const IDX_CX_P = 3u;
const IDX_CX_M = 4u;
const IDX_CY_P = 5u;
const IDX_CY_M = 6u;

fn warp_and_accumulate(evt: Event) {
    // 1. Center
    let p0 = warp(evt.pos, params.c, params.omega);
    accumulate(p0, IDX_CENTER);

    // 2. Omega perturbations
    let p_om = warp(evt.pos, params.c, params.omega - params.delta_omega);
    let p_op = warp(evt.pos, params.c, params.omega + params.delta_omega);
    accumulate(p_om, IDX_OMEGA_M);
    accumulate(p_op, IDX_OMEGA_P);

    // 3. Centroid perturbations
    let c_xm = vec2(params.c.x - params.delta_pos, params.c.y);
    let c_xp = vec2(params.c.x + params.delta_pos, params.c.y);
    accumulate(warp(evt.pos, c_xm, params.omega), IDX_CX_M);
    accumulate(warp(evt.pos, c_xp, params.omega), IDX_CX_P);

    // ... same for Cy
}
```

### 4.2 Optimizer Logic

```rust
struct OptimizerState {
    omega: f32,
    centroid: Vec2,
    // Adaptive learning rates
    lr_omega: f32,
    lr_centroid: f32,
}

fn update(state: &mut OptimizerState, res: ContrastResult) {
    // 1. Omega Step (Parabolic)
    let d_omega = compute_parabolic_step(res.omega_m, res.center, res.omega_p);
    state.omega += d_omega;

    // 2. Centroid Step (Gradient Ascent)
    // Parabolic might be unstable for XY if surface is complex, start with Gradient
    let grad_x = (res.cx_p - res.cx_m) / (2.0 * delta_pos);
    let grad_y = (res.cy_p - res.cy_m) / (2.0 * delta_pos);
    
    state.centroid.x += grad_x * state.lr_centroid;
    state.centroid.y += grad_y * state.lr_centroid;
}
```

---

## 5. Risks & Mitigations

| Risk | Mitigation |
| :--- | :--- |
| **Performance** | 7 atomic adds per event might be slow. **Mitigation**: Use `workgroup` caching if needed, or reduce resolution of perturbation slices. |
| **Local Minima** | Centroid optimization is non-convex. **Mitigation**: Initialize with "Center of Mass" of events or Hough Transform (future). For now, rely on manual/GT initialization. |
| **Coupling** | $\omega$ and $\mathbf{c}$ might fight. **Mitigation**: Update $\omega$ and $\mathbf{c}$ in alternating frames? Or use a joint Hessian (Newton method). Start with simultaneous updates. |

## 6. Plan

1.  **Update Shaders**: Expand to 7 slices.
2.  **Update Buffers**: Resize IWE and Contrast buffers.
3.  **Update Readback**: Handle 7 values.
4.  **Update System**: Implement 3D optimizer.
5.  **Verify**: Test with "drifting fan" synthetic data.

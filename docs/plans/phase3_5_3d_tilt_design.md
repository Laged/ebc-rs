# Phase 3.5 Design: Handling 3D Tilt (Homography)

## 1. Overview

**Goal**: Track a drone that is tilting (pitching/rolling) relative to the camera.
**Gap**: Phase 2 and 3 assume the fan remains parallel to the image plane (SE(2) motion). When a drone tilts to move, the circular fan projects as an **ellipse**. The 2D rigid warp cannot align an ellipse to a circle, causing contrast loss and tracking failure.
**Approach**: Upgrade the motion model to a **Planar Homography** or **3D Rigid Body Motion (SE(3))**.

---

## 2. Mathematical Formulation

### 2.1 The Geometry
We model the fan as a planar patch in 3D space with normal vector $\mathbf{n}$.
The relationship between the pixel coordinates in the current frame $\mathbf{x}_k$ and the reference frame $\mathbf{x}_{ref}$ is given by a **Homography** $H$:

$$ \mathbf{x}_{ref} \propto H \cdot \mathbf{x}_k $$

Where $H$ is a $3 \times 3$ matrix.

### 2.2 The Motion Model (SE(3))
Instead of optimizing just $v_x, v_y, \omega_{yaw}$ (Phase 3), we optimize the full 3D velocity twist $\xi \in \mathfrak{se}(3)$:
$$ \xi = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]^T $$

The homography induced by a plane $\pi = [\mathbf{n}^T, d]$ moving by $T = \exp(\xi \Delta t) = [R | \mathbf{t}]$ is:
$$ H = R - \frac{\mathbf{t} \mathbf{n}^T}{d} $$

### 2.3 The Warp Function
The warp function becomes a perspective projection:
$$ \mathbf{u}' = \frac{h_{11}u + h_{12}v + h_{13}}{h_{31}u + h_{32}v + h_{33}}, \quad \mathbf{v}' = \frac{h_{21}u + h_{22}v + h_{23}}{h_{31}u + h_{32}v + h_{33}} $$

---

## 3. Architecture Changes

### 3.1 Optimizer (High Dimensionality)
We are now optimizing **6 to 8 parameters** per frame (depending on if we fix the plane normal).
*   **Numerical Gradient**: Computing $2 \times 6 = 12$ slices per frame is expensive.
*   **Analytical Gradient**: We **must** switch to analytical derivatives here. The cost of 12 atomic-add passes is too high.

### 3.2 Shader (`cmax_slam_warp_homography.wgsl`)
*   **Input**: Event $(u, v, t)$, Twist $\xi$, Plane $\pi$.
*   **Logic**:
    1.  Compute $T(t) = \exp(\xi \Delta t)$.
    2.  Compute $H(t)$.
    3.  Apply perspective warp.
    4.  Accumulate to IWE.
    5.  (Optional) Compute Jacobian $J$ per event and accumulate Hessian $H = J^T J$.

---

## 4. Implementation Strategy

### Step 1: The "Affine" Bridge
Before full Homography, we can try an **Affine Warp** (6 parameters). It handles "shear" which approximates small tilts.
$$ \mathbf{x}' = A \mathbf{x} + \mathbf{t} $$
This is linear and easier to optimize than the rational Homography function.

### Step 2: Full SE(3) Optimization
1.  **State**: $\mathbf{x} = [\text{Pose}(6), \text{Velocity}(6), \text{Plane}(3)]$.
2.  **Solver**: Levenberg-Marquardt on GPU.
    *   We likely need to port the "Lift-Solve-Retract" scheme from the CMax paper.
    *   This is significantly more complex than Phase 2/3.

---

## 5. Comparison: Phase 3 vs 3.5

| Feature | Phase 3 (SE(2)) | Phase 3.5 (SE(3) / Homography) |
| :--- | :--- | :--- |
| **Motion** | Translation + Yaw | Translation + Yaw + **Pitch + Roll** |
| **Shape** | Circle stays Circle | Circle becomes **Ellipse** |
| **Params** | 3 per frame | 6-8 per frame |
| **Compute** | Low (Numerical Grad) | High (Analytical Grad required) |
| **Use Case** | Hovering, Slow flight | Aggressive Maneuvers |

## 6. Recommendation
**Do not skip Phase 2/3.**
Phase 3.5 is a "Pro" feature.
1.  Implement Phase 2 (Centroid) to get robust 2D tracking.
2.  Implement Phase 3 (Splines) to handle smooth trajectories.
3.  **If** tracking fails during aggressive drone tilts, **then** implement Phase 3.5 as a drop-in replacement for the Warp Shader.

The architecture (Sliding Window, IWE, Optimizer) remains the same; only the **Warp Function** and **State Vector** change.

# Phase 3 Design: Full 2D Rigid Body Motion (SE(2))

## 1. Overview

**Goal**: Track a drone performing complex maneuvers (translation + rotation).
**Gap**: Phase 2 assumes pure rotation around a moving center. Drones translate *while* rotating, and the center of rotation is not always the geometric center.
**Approach**: Switch to a **Continuous-Time Trajectory** model using B-Splines on SE(2), optimized via **Sliding Window Bundle Adjustment**.

---

## 2. Mathematical Formulation

### 2.1 Motion Model (B-Spline)
Instead of optimizing a single $\omega$ and $\mathbf{c}$ per frame, we optimize a continuous trajectory $T(t) \in SE(2)$.
We represent $T(t)$ using cumulative B-Splines:

$$ T(t) = \exp(\Omega(t)) $$
$$ \Omega(t) = \sum_{i} B_{i,k}(t) \cdot \mathbf{\phi}_i $$

Where $\mathbf{\phi}_i \in \mathbb{R}^3$ are control points representing pose (x, y, theta) at time $t_i$.

### 2.2 The Warp Function
For an event at time $t_k$, we warp it to the reference time $t_{ref}$:
$$ \mathbf{x}'_k = T(t_{ref})^{-1} \cdot T(t_k) \cdot \mathbf{x}_k $$

### 2.3 Optimization Problem
Maximize contrast over a sliding window of time $[t_{start}, t_{end}]$.
Variables: Control points $\{ \mathbf{\phi}_0, \mathbf{\phi}_1, ... \mathbf{\phi}_N \}$ within the window.

---

## 3. Architecture Changes

### 3.1 Data Structures
-   **Control Points Buffer**: GPU buffer storing the B-Spline control points.
-   **Event Batching**: Events must be batched by time window to map to specific control points.

### 3.2 Warp Shader (`cmax_slam_warp_se2.wgsl`)
-   **Input**: Event $(x, y, t)$, Control Points.
-   **Logic**:
    1.  Find relevant control points for time $t$.
    2.  Evaluate B-Spline to get $T(t)$.
    3.  Compute relative transform $T_{rel} = T(t_{ref})^{-1} T(t)$.
    4.  Warp pixel: $x' = R x + t$.
    5.  Accumulate to IWE.

### 3.3 Gradient Computation
Numerical gradients become too expensive (3 parameters per control point * N points).
**Strategy**:
1.  **Analytical Derivatives**: Implement analytical Jacobian $J = \frac{\partial I}{\partial \mathbf{\phi}}$ in the shader.
2.  **Accumulate Hessian**: Compute $H = J^T J$ and $b = J^T r$ on GPU.
3.  **Solve on CPU**: Read back $H$ and $b$, solve linear system $\delta \mathbf{\phi} = H^{-1} b$.

---

## 4. Implementation Details

### 4.1 B-Spline Evaluation (GPU)
We use a Cubic B-Spline (k=4). The pose at time $t$ depends on 4 local control points.
Matrix formulation:
$$ \mathbf{u}(t)^T M_{BSpline} \begin{bmatrix} \mathbf{\phi}_{i} \\ \mathbf{\phi}_{i+1} \\ \mathbf{\phi}_{i+2} \\ \mathbf{\phi}_{i+3} \end{bmatrix} $$

### 4.2 Sliding Window Backend
-   **Window Size**: e.g., 100ms.
-   **Knot Spacing**: e.g., 10ms (10 control points per window).
-   **Marginalization**: When the window slides, old control points are fixed (marginalized) to maintain consistency.

---

## 5. Risks & Mitigations

| Risk | Mitigation |
| :--- | :--- |
| **Complexity** | B-Splines on Lie Groups are hard. **Mitigation**: Start with B-Splines on $\mathbb{R}^3$ (treat angle as linear) since 2D rotation is commutative (SO(2) is abelian). |
| **Compute Cost** | Evaluating splines per event is heavy. **Mitigation**: Pre-compute trajectory at fine granularity (lookup table) on GPU before warping. |
| **Initialization** | Trajectory optimization needs good initial guess. **Mitigation**: Use Phase 2 (Centroid + Omega) to initialize the linear velocity and angular velocity. |

## 6. Plan

1.  **Spline Library**: Implement B-Spline evaluation in WGSL.
2.  **Warp Shader**: Update to use Spline trajectory.
3.  **Solver**: Implement Schur Complement or simple Cholesky solver on CPU (using `nalgebra`).
4.  **Integration**: Replace Phase 2 optimizer with Sliding Window BA.

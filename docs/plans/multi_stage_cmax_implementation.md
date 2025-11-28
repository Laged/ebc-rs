# Multi-Stage CMax-SLAM Implementation Roadmap

## Overview

This document outlines the phase-by-phase evolution of the `ebc-rs` CMax-SLAM system. The goal is to bridge the gap between the current **static fan RPM estimator** and the **multi-drone tracking system** required by the Sensofusion Challenge.

> [!IMPORTANT]
> The previous review (`docs/reviews/cmax_slam_implementation_review.md`) covers **Phase 1** only. It is a prerequisite for all subsequent phases.

---

## Phase 1: Robust Static RPM (Current Focus)
**Goal**: autonomously estimate RPM for a spinning fan with a *known/fixed* centroid.
**Gap**: Current implementation relies on Ground Truth for $\omega$.
**Deliverable**: A closed-loop optimizer that finds RPM by maximizing variance.

### Steps
1.  **Variance Reduction Shader**: Implement `cmax_slam_reduce.wgsl` to compute contrast score.>
2.  **Gradient Optimization**: Implement CPU-side optimization loop (Gradient Ascent) to update $\omega$.
3.  **Evaluation**: Verify against `docs/design/cmax-slam-evaluation.md` (RPM Accuracy).

---

## Phase 2: Centroid Tracking (Joint Optimization)
**Goal**: Track a fan that is spinning *and* drifting (moving centroid).
**Gap**: Current shader assumes fixed `centroid_x`, `centroid_y`.
**Deliverable**: Joint optimization of $(\omega, c_x, c_y)$.

### Theory
The contrast function $C(\omega, \mathbf{c})$ depends on both angular velocity and rotation center.
-   Wrong center $\to$ blurry edges (arcs don't align).
-   Correct center $\to$ sharp edges.

### Implementation Steps
1.  **Shader Update**: Modify `cmax_slam_warp.wgsl` to warp based on a dynamic centroid.
2.  **Gradient Computation**:
    -   We now need derivatives w.r.t $c_x$ and $c_y$.
    -   **Option A (Numerical)**: Compute 5 IWEs per frame: Center, $\omega \pm \delta$, $c_x \pm \delta$, $c_y \pm \delta$. (Expensive but simple).
    -   **Option B (Analytical)**: Implement analytical derivatives in the shader (Harder, faster).
3.  **Optimizer Update**: Update `systems.rs` to optimize a 3-vector state $[\omega, c_x, c_y]$.

---

## Phase 3: Full 2D Rigid Body Motion (SE(2))
**Goal**: Track a drone performing complex maneuvers (translation + rotation).
**Gap**: Current model assumes pure rotation around a center. Drones translate *while* rotating.
**Deliverable**: Estimation of linear velocity $(v_x, v_y)$ and angular velocity $\omega$.

### Theory
Motion model changes from "Rotation around Center" to "Linear + Angular Velocity".
$$ \mathbf{v}(t) = \mathbf{v}_{lin} + \omega \times (\mathbf{x} - \mathbf{c}) $$
Or simply optimize for the warp parameters of a rigid body motion.

### Implementation Steps
1.  **Motion Model**: Switch to a B-Spline trajectory representation (as per the CMax-SLAM paper).
    -   State: Control points of the trajectory (Pose at time $t_i$).
2.  **Warping**: Update `warp_event` to interpolate pose from B-Spline control points.
3.  **Sliding Window**: Implement the "Backend" from the paper (Sliding Window Bundle Adjustment).

---

## Phase 4: Multi-Instance Tracking
**Goal**: Track multiple drones simultaneously.
**Gap**: CMax-SLAM is a *global* method. It aligns the *entire* image. Multiple moving objects with different motions will conflict.
**Deliverable**: Independent tracking of multiple objects.

### Implementation Steps
1.  **Clustering (The "Tracker")**:
    -   Use a lightweight tracker (e.g., clustering events by timestamp/location or using the existing `centroid.wgsl`) to identify "Regions of Interest" (RoIs).
    -   Assign events to Cluster A or Cluster B.
2.  **Multi-Pipeline Dispatch**:
    -   Spawn a **CMax-SLAM instance** for *each* cluster.
    -   Instance A optimizes Motion A for Cluster A events.
    -   Instance B optimizes Motion B for Cluster B events.
3.  **Data Association**:
    -   Filter events: Pass only relevant events to each CMax solver.
    -   Masking: Use the "sharp" IWE from Phase 1 to define the object mask.

---

## Summary of Work Required

| Phase | Feature | Complexity | Dependency |
| :--- | :--- | :--- | :--- |
| **1** | **Auto-RPM** | ⭐⭐ | None |
| **2** | **Centroid Opt** | ⭐⭐⭐ | Phase 1 |
| **3** | **Translation** | ⭐⭐⭐⭐ | Phase 2 |
| **4** | **Multi-Drone** | ⭐⭐⭐⭐⭐ | Phase 3 + Clustering |

### Immediate Action (Phase 1 Fix)
Follow the guide in `docs/reviews/cmax_slam_implementation_review.md`. It is the foundational block. You cannot do Phase 2-4 without a working contrast maximizer.

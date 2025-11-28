# Implementation Plan: 10k GPU Boids

This document outlines the plan to implement a high-performance flocking simulation using our GPU pipeline.

## Goal
Simulate and render **10,000 autonomous agents** (boids) at 60+ FPS.

## 1. Data Structures

We need a compact representation for the GPU.

### Rust (`src/boids/gpu.rs`)
```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Boid {
    pos: Vec2,
    vel: Vec2,
    // Padding might be needed for alignment (16 bytes is ideal)
}
```

### WGSL (`assets/shaders/boids.wgsl`)
```wgsl
struct Boid {
    pos: vec2<f32>,
    vel: vec2<f32>,
}
@group(0) @binding(0) var<storage, read_write> boids_src: array<Boid>;
@group(0) @binding(1) var<storage, read_write> boids_dst: array<Boid>;
```
*Note: We use double-buffering (ping-pong) to avoid race conditions during updates.*

## 2. The Algorithm: Spatial Hashing

A naive boids implementation is O(N^2), checking every boid against every other. For 10k boids, that's 100M checks/frame—too slow.
We will use **Spatial Hashing** (Grid Optimization) to reduce this to O(N).

### Grid Structure
-   Divide the world into cells of size `interaction_radius`.
-   Each boid only checks neighbors in its own cell and the 8 surrounding cells.

### Compute Passes
1.  **Clear Grid**: Reset the grid counter buffer.
2.  **Populate Grid**:
    -   Calculate cell ID for each boid.
    -   Atomic add to cell counter.
    -   Store boid index in a "Cell List".
3.  **Update Boids**:
    -   For each boid, look up neighbors using the Cell List.
    -   Apply Flocking Rules:
        -   **Separation**: Steer away from crowded neighbors.
        -   **Alignment**: Steer towards average heading of neighbors.
        -   **Cohesion**: Steer towards center of mass of neighbors.
    -   Integrate position: `pos += vel * dt`.
    -   Wrap around screen edges.

## 3. Rendering

We will use **Instanced Rendering** to draw 10k triangles efficiently.

-   **Geometry**: A single triangle mesh (3 vertices).
-   **Instance Buffer**: The `boids_dst` buffer from the compute pass acts as the instance buffer.
-   **Vertex Shader**:
    -   Input: Vertex position (local), Instance position (world), Instance velocity.
    -   Rotate the local triangle to match the velocity direction.
    -   Translate to world position.

## 4. Bevy Integration (`src/boids/mod.rs`)

### `BoidsPlugin`
-   **Startup**: Initialize 10k boids with random positions/velocities. Upload to GPU.
-   **Update**:
    -   Update `BoidParams` (cohesion/alignment weights) from UI/Resources.
    -   Dispatch Compute Shader.
-   **Render**:
    -   Custom `RenderPhase` or `SpecializedRenderPipeline`.

## 5. Roadmap

1.  **Phase 1: Basic Movement**: Implement the double-buffered update loop. Boids move in straight lines.
2.  **Phase 2: Instanced Rendering**: Get the 10k triangles on screen.
3.  **Phase 3: Naive Flocking**: Implement O(N^2) logic for small N (e.g., 500) to verify behavior.
4.  **Phase 4: Spatial Grid**: Implement the grid optimization to scale to 10k+.

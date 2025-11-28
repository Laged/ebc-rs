# Tutorial: The Spinning Triangle

This guide walks through creating a "Hello World" plugin for our GPU-accelerated architecture: a spinning triangle that emits particles.

## Goal
Create a `SpinningTrianglePlugin` that:
1.  Accepts a `RotationSpeed` resource on the CPU.
2.  Uses a **Compute Shader** to update particle positions based on the triangle's rotation.
3.  Uses a **Render Pipeline** to draw the triangle and particles.

## 1. The CPU Side (`src/triangle/mod.rs`)

First, we define our data and the plugin.

```rust
use bevy::prelude::*;

#[derive(Resource)]
pub struct TriangleParams {
    pub speed: f32,
    pub color: Vec3,
}

pub struct SpinningTrianglePlugin;

impl Plugin for SpinningTrianglePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(TriangleParams {
            speed: 1.0,
            color: Vec3::new(1.0, 0.5, 0.0),
        });
        
        // We will add the render setup later
    }
}
```

## 2. GPU Data Structures (`src/triangle/gpu.rs`)

We need a struct that matches our shader's uniform buffer. We use `bytemuck` for memory layout compatibility.

```rust
use bevy::render::render_resource::{ShaderType, UniformBuffer};
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Pod, Zeroable, ShaderType)]
#[repr(C)]
pub struct GpuTriangleUniforms {
    pub rotation: f32,
    pub color: Vec3,
    pub _padding: f32, // Alignment padding
}
```

## 3. The Compute Shader (`assets/shaders/triangle_compute.wgsl`)

This shader calculates the vertex positions.

```wgsl
struct Uniforms {
    rotation: f32,
    color: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= 3u) { return; }

    let angle = uniforms.rotation + (f32(idx) * 2.0944); // 120 degrees offset
    let r = 0.5;
    
    positions[idx] = vec2<f32>(cos(angle) * r, sin(angle) * r);
}
```

## 4. The Render Shader (`assets/shaders/triangle_render.wgsl`)

This shader draws the vertices calculated by the compute shader.

```wgsl
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    // Read position from the buffer written by compute shader
    let pos = positions[in_vertex_index];
    
    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.color = uniforms.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
```

## 5. Wiring it Up (The Hard Part)

To make this work in Bevy, we need to:

1.  **Extract**: Copy `TriangleParams` to the Render World.
2.  **Prepare**: Create `wgpu::Buffer`s for uniforms and positions.
3.  **Queue**: Create `BindGroup`s.
4.  **Graph**: Add a node to the `RenderGraph`.

### The Render Node

```rust
impl render_graph::Node for TriangleNode {
    fn run(&self, graph: &mut RenderGraphContext, render_context: &mut RenderContext, world: &World) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.get_compute_pipeline(pipeline_id).unwrap();
        let bind_group = world.resource::<TriangleBindGroup>();

        let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group.0, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        
        Ok(())
    }
}
```

## Conclusion

This pattern—**CPU Resource -> Extract -> Prepare -> Compute -> Render**—is the core of our high-performance architecture. It allows us to do heavy simulation (like fluid dynamics or particle systems) entirely on the GPU while controlling it from Bevy's ergonomic ECS.

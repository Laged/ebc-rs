# Fan Visualization Accuracy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix fan blade synchronization and border circle accuracy using GPU-based radial analysis and angular histogram for blade detection.

**Architecture:** Add two GPU compute pipelines (RadialProfileNode, AngularHistogramNode) that analyze the accumulated surface buffer and event distribution to detect actual fan radius and blade positions, replacing simulated calculations with data-driven detection.

**Tech Stack:** Rust 1.91, Bevy 0.17, WGPU/WGSL compute shaders, NixOS flake

---

## Task 1: Fix Code Quality Issues

**Files:**
- Modify: `src/analysis.rs:520`
- Modify: `src/analysis.rs:18,22`
- Modify: `flake.nix:35-45`

**Step 1: Run clippy to see current warnings**

Run: `cargo clippy --all-targets`
Expected: 3 warnings about unused mut, dead_code constants

**Step 2: Fix unused mut warning**

In `src/analysis.rs:520`, change:
```rust
// Before
fn read_centroid_result_render(
    mut gpu_resources: ResMut<CentroidGpuResources>,

// After
fn read_centroid_result_render(
    gpu_resources: ResMut<CentroidGpuResources>,
```

**Step 3: Fix dead code warnings**

In `src/analysis.rs:18-22`, remove unused constants:
```rust
// Remove these lines (they will be used later for real CMax):
const EVENT_TIMESTAMP_UNIT_SECONDS: f32 = 1e-7;
const OMEGA_TO_RPM_SCALE: f32 = 60.0 / (2.0 * std::f32::consts::PI * EVENT_TIMESTAMP_UNIT_SECONDS);
```

**Step 4: Add wayland-protocols to flake.nix**

In `flake.nix:35-45`, modify `linuxDeps`:
```nix
linuxDeps = with pkgs; [
  alsa-lib
  udev
  libxkbcommon
  wayland
  wayland-protocols  # ADD THIS LINE
  vulkan-loader
  xorg.libX11
  xorg.libXcursor
  xorg.libXi
  xorg.libXrandr
];
```

**Step 5: Verify clippy passes**

Run: `cargo clippy --all-targets`
Expected: No warnings

**Step 6: Commit**

```bash
git add src/analysis.rs flake.nix
git commit -m "fix: remove unused code and add wayland-protocols for Linux

- Remove unused mut in read_centroid_result_render
- Remove unused timestamp constants
- Add wayland-protocols to fix Vulkan loader warnings on Linux"
```

---

## Task 2: Fix Data Loading Error Handling

**Files:**
- Modify: `src/render.rs:214-229`
- Modify: `src/render.rs:78-182` (ui_system)

**Step 1: Add graceful failure in load_data**

In `src/render.rs:214-229`, replace error handling:
```rust
fn load_data(mut commands: Commands, mut playback_state: ResMut<PlaybackState>) {
    let path = "data/fan/fan_const_rpm.dat";
    match DatLoader::load(path) {
        Ok(events) => {
            info!("Loaded {} events from {}", events.len(), path);
            if let Some(last) = events.last() {
                playback_state.max_timestamp = last.timestamp;
                playback_state.current_time = last.timestamp as f32;
            }
            commands.insert_resource(EventData { events });
        }
        Err(e) => {
            error!("Failed to load data from {}: {:?}", path, e);
            // Insert empty EventData so app doesn't crash
            commands.insert_resource(EventData { events: Vec::new() });
            playback_state.max_timestamp = 0;
            playback_state.current_time = 0.0;
        }
    }
}
```

**Step 2: Add UI error indicator**

In `src/render.rs`, find the `ui_system` function and add event data check:
```rust
fn ui_system(
    mut contexts: EguiContexts,
    mut playback_state: ResMut<PlaybackState>,
    mut fan_analysis: ResMut<FanAnalysis>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    event_data: Res<EventData>,  // ADD THIS
) {
    // ADD THIS AT THE START OF THE FUNCTION
    if event_data.events.is_empty() {
        egui::Window::new("Error").show(
            contexts.ctx_mut().expect("Failed to get egui context"),
            |ui| {
                ui.colored_label(
                    egui::Color32::RED,
                    "Failed to load event data. Check that data/fan/fan_const_rpm.dat exists."
                );
            },
        );
    }

    // ... existing UI code continues
```

**Step 3: Test with missing file**

Run: `mv data/fan/fan_const_rpm.dat data/fan/fan_const_rpm.dat.bak && cargo run`
Expected: App starts, red error window appears, no crash

**Step 4: Test with file restored**

Run: `mv data/fan/fan_const_rpm.dat.bak data/fan/fan_const_rpm.dat && cargo run`
Expected: App loads data normally, no error window

**Step 5: Commit**

```bash
git add src/render.rs
git commit -m "fix: graceful failure when event data file missing

- Insert empty EventData resource on load failure
- Show red error window in UI when no events loaded
- Prevents app hang/crash on missing data file"
```

---

## Task 3: Add FanAnalysis.blade_angles Field

**Files:**
- Modify: `src/analysis.rs:24-58`

**Step 1: Add blade_angles field to FanAnalysis**

In `src/analysis.rs:24-58`, modify the struct:
```rust
#[derive(Resource, Clone, ExtractResource)]
pub struct FanAnalysis {
    pub is_tracking: bool,
    pub show_borders: bool,
    pub current_rpm: f32,
    pub blade_count: u32,
    pub centroid: Vec2,
    pub tip_velocity: f32,
    pub fan_radius: f32,
    pub current_angle: f32,
    pub blade_angles: Vec<f32>,  // ADD THIS LINE
}
```

**Step 2: Update Default impl**

In `src/analysis.rs:45-58`, add initialization:
```rust
impl Default for FanAnalysis {
    fn default() -> Self {
        Self {
            is_tracking: true,
            show_borders: false,
            current_rpm: 0.0,
            blade_count: 3,
            centroid: Vec2::new(640.0, 360.0),
            tip_velocity: 0.0,
            fan_radius: 200.0,
            current_angle: 0.0,
            blade_angles: vec![0.0, 2.094, 4.189],  // ADD THIS: Default 3 blades at 120° spacing
        }
    }
}
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 4: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add blade_angles field to FanAnalysis

Initialize with default 3-blade spacing (120° intervals).
Will be populated by angular histogram in next tasks."
```

---

## Task 4: Create Radial Profile Compute Shader

**Files:**
- Create: `assets/shaders/radial_profile.wgsl`

**Step 1: Create the shader file**

Create `assets/shaders/radial_profile.wgsl`:
```wgsl
// Radial Profile Analysis Shader
// Analyzes accumulated surface buffer to detect fan radius

struct RadialResult {
    radial_bins: array<atomic<u32>, 400>,  // Bins for 0-400px radius
    total_intensity: atomic<u32>,
}

@group(0) @binding(0) var surface_texture: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> result: RadialResult;
@group(0) @binding(2) var<uniform> centroid: vec2<f32>;

const MAX_RADIUS: u32 = 400u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(surface_texture);
    let pos = global_id.xy;

    // Bounds check
    if (pos.x >= dims.x || pos.y >= dims.y) {
        return;
    }

    // Read pixel intensity from surface buffer
    let pixel_value = textureLoad(surface_texture, pos, 0).r;

    if (pixel_value == 0u) {
        return;  // Skip empty pixels
    }

    // Calculate distance from centroid
    let dx = f32(pos.x) - centroid.x;
    let dy = f32(pos.y) - centroid.y;
    let distance = sqrt(dx * dx + dy * dy);

    // Bin the distance
    let bin_index = min(u32(distance), MAX_RADIUS - 1u);

    // Accumulate intensity in the appropriate bin
    atomicAdd(&result.radial_bins[bin_index], pixel_value);
    atomicAdd(&result.total_intensity, pixel_value);
}
```

**Step 2: Verify shader syntax**

Run: `cargo check` (Bevy will validate WGSL at compile time)
Expected: No shader errors

**Step 3: Commit**

```bash
git add assets/shaders/radial_profile.wgsl
git commit -m "feat: add radial profile compute shader

Analyzes surface buffer to build radial intensity histogram.
Uses 400 bins for 0-400px radius from centroid."
```

---

## Task 5: Create Angular Histogram Compute Shader

**Files:**
- Create: `assets/shaders/angular_histogram.wgsl`

**Step 1: Create the shader file**

Create `assets/shaders/angular_histogram.wgsl`:
```wgsl
// Angular Histogram Shader
// Analyzes event distribution around centroid to detect blade positions

struct AngularResult {
    bins: array<atomic<u32>, 360>,  // 1 degree per bin
}

struct GpuEvent {
    timestamp: u32,
    x: u32,
    y: u32,
    polarity: u32,
}

struct AnalysisParams {
    centroid_x: f32,
    centroid_y: f32,
    radius: f32,
    radius_tolerance: f32,
    window_start: u32,
    window_end: u32,
}

@group(0) @binding(0) var<storage, read> events: array<GpuEvent>;
@group(0) @binding(1) var<storage, read_write> result: AngularResult;
@group(0) @binding(2) var<uniform> params: AnalysisParams;

const PI: f32 = 3.14159265359;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let event_index = global_id.x + global_id.y * 65535u;

    if (event_index >= arrayLength(&events)) {
        return;
    }

    let event = events[event_index];

    // Filter by time window
    if (event.timestamp < params.window_start || event.timestamp > params.window_end) {
        return;
    }

    // Calculate distance from centroid
    let dx = f32(event.x) - params.centroid_x;
    let dy = f32(event.y) - params.centroid_y;
    let distance = sqrt(dx * dx + dy * dy);

    // Only count events near the detected radius
    if (abs(distance - params.radius) > params.radius_tolerance) {
        return;
    }

    // Calculate angle (atan2 returns [-π, π], normalize to [0, 2π])
    var angle = atan2(dy, dx);
    if (angle < 0.0) {
        angle = angle + 2.0 * PI;
    }

    // Convert to bin index (0-359 degrees)
    let bin_index = u32((angle / (2.0 * PI)) * 360.0) % 360u;

    // Increment bin
    atomicAdd(&result.bins[bin_index], 1u);
}
```

**Step 2: Verify shader syntax**

Run: `cargo check`
Expected: No shader errors

**Step 3: Commit**

```bash
git add assets/shaders/angular_histogram.wgsl
git commit -m "feat: add angular histogram compute shader

Analyzes event angular distribution to detect blade positions.
Uses 360 bins (1° resolution) and filters events by radius."
```

---

## Task 6: Implement Radial Profile Pipeline (Rust Infrastructure)

**Files:**
- Modify: `src/analysis.rs` (add after CentroidPipeline around line 198)

**Step 1: Add GPU resource structs**

In `src/analysis.rs`, add after the `CentroidGpuResources` struct:
```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct RadialResult {
    radial_bins: [u32; 400],
    total_intensity: u32,
    _padding: [u32; 3],
}

#[derive(Resource, Default)]
pub struct RadialGpuResources {
    pub result_buffer: Option<Buffer>,
    pub staging_buffer: Option<Buffer>,
    pub bind_group: Option<BindGroup>,
    pub pipeline_ready: bool,
    pub map_receiver: Option<std::sync::Mutex<std::sync::mpsc::Receiver<Result<(), BufferAsyncError>>>>,
    pub is_mapped: bool,
}
```

**Step 2: Add pipeline struct**

After `CentroidPipeline`:
```rust
#[derive(Resource)]
pub struct RadialProfilePipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for RadialProfilePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Radial Profile Layout"),
            &[
                // Surface texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Uint,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Result buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Centroid uniform
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let shader = world.resource::<AssetServer>().load("shaders/radial_profile.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Radial Profile Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        RadialProfilePipeline { layout, pipeline }
    }
}
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 4: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add radial profile pipeline infrastructure

Add GPU resources and pipeline for radial intensity analysis.
Follows same pattern as centroid pipeline."
```

---

## Task 7: Implement Radial Profile Render Node

**Files:**
- Modify: `src/analysis.rs` (add after line 247)

**Step 1: Add channel resources**

After `CentroidReceiver`:
```rust
#[derive(Resource)]
struct RadialSender(pub std::sync::mpsc::Sender<f32>);

#[derive(Resource)]
struct RadialReceiver(pub std::sync::Mutex<std::sync::mpsc::Receiver<f32>>);
```

**Step 2: Add render label**

After `CentroidLabel`:
```rust
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct RadialProfileLabel;
```

**Step 3: Add render node**

After `CentroidNode`:
```rust
struct RadialProfileNode;

impl Node for RadialProfileNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let gpu_resources = world.resource::<RadialGpuResources>();
        let pipeline = world.resource::<RadialProfilePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let surface_image = world.resource::<SurfaceImage>();
        let render_queue = world.resource::<RenderQueue>();
        let analysis = world.resource::<FanAnalysis>();

        if !gpu_resources.pipeline_ready {
            return Ok(());
        }
        let Some(bind_group) = &gpu_resources.bind_group else {
            return Ok(());
        };
        let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
            return Ok(());
        };

        // Reset result buffer
        if let Some(result_buffer) = &gpu_resources.result_buffer {
            let reset_data = RadialResult {
                radial_bins: [0; 400],
                total_intensity: 0,
                _padding: [0; 3],
            };
            render_queue.write_buffer(result_buffer, 0, bytemuck::bytes_of(&reset_data));
        }

        {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Radial Profile Pass"),
                    timestamp_writes: None,
                });

            pass.set_pipeline(compute_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            // Dispatch for 1280x720 texture with 16x16 workgroups
            let workgroups_x = (1280 + 15) / 16;
            let workgroups_y = (720 + 15) / 16;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy to staging buffer
        if !gpu_resources.is_mapped {
            if let (Some(result), Some(staging)) =
                (&gpu_resources.result_buffer, &gpu_resources.staging_buffer)
            {
                render_context.command_encoder().copy_buffer_to_buffer(
                    result,
                    0,
                    staging,
                    0,
                    std::mem::size_of::<RadialResult>() as u64,
                );
            }
        }

        Ok(())
    }
}
```

**Step 4: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 5: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add radial profile render node

Executes radial analysis compute shader on surface buffer.
Dispatches 16x16 workgroups for 1280x720 texture."
```

---

## Task 8: Add Radial Profile Bind Group Preparation

**Files:**
- Modify: `src/analysis.rs` (add after `prepare_centroid_bind_group`)

**Step 1: Add preparation system**

```rust
fn prepare_radial_bind_group(
    pipeline: Res<RadialProfilePipeline>,
    render_device: Res<RenderDevice>,
    mut gpu_resources: ResMut<RadialGpuResources>,
    surface_image: Res<SurfaceImage>,
    gpu_images: Res<bevy::render::render_asset::RenderAssets<GpuImage>>,
    analysis: Res<FanAnalysis>,
) {
    let Some(gpu_image) = gpu_images.get(&surface_image.handle) else {
        return;
    };

    // Create result buffer if missing
    if gpu_resources.result_buffer.is_none() {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("Radial Result Buffer"),
            size: std::mem::size_of::<RadialResult>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu_resources.result_buffer = Some(buffer);
    }

    // Create staging buffer
    if gpu_resources.staging_buffer.is_none() {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("Radial Staging Buffer"),
            size: std::mem::size_of::<RadialResult>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu_resources.staging_buffer = Some(buffer);
    }

    // Create centroid uniform
    let centroid_uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Radial Centroid Uniform"),
        contents: bytemuck::bytes_of(&[analysis.centroid.x, analysis.centroid.y]),
        usage: BufferUsages::UNIFORM,
    });

    // Create bind group
    let bind_group = render_device.create_bind_group(
        Some("Radial Profile Bind Group"),
        &pipeline.layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&gpu_image.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: gpu_resources.result_buffer.as_ref().unwrap().as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: centroid_uniform.as_entire_binding(),
            },
        ],
    );

    gpu_resources.bind_group = Some(bind_group);
    gpu_resources.pipeline_ready = true;
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 3: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add radial profile bind group preparation

Creates GPU buffers and bind group for radial analysis.
Passes surface texture and centroid to compute shader."
```

---

## Task 9: Add Radial Profile CPU Readback and Radius Detection

**Files:**
- Modify: `src/analysis.rs`

**Step 1: Add readback system**

After `read_centroid_result_render`:
```rust
fn read_radial_result_render(
    mut gpu_resources: ResMut<RadialGpuResources>,
    sender: Res<RadialSender>,
) {
    let gpu_resources = gpu_resources.into_inner();
    let Some(staging_buffer) = &gpu_resources.staging_buffer else {
        return;
    };

    if let Some(receiver_mutex) = gpu_resources.map_receiver.take() {
        let should_reinsert = if let Ok(receiver) = receiver_mutex.try_lock() {
            match receiver.try_recv() {
                Ok(Ok(())) => {
                    let slice = staging_buffer.slice(..);
                    {
                        let data = slice.get_mapped_range();
                        let result: RadialResult = *bytemuck::from_bytes(&data);

                        // Calculate radius from 95th percentile
                        let target_intensity = (result.total_intensity as f32 * 0.95) as u32;
                        let mut cumulative = 0u32;
                        let mut detected_radius = 200.0; // Default fallback

                        for (i, &bin_value) in result.radial_bins.iter().enumerate() {
                            cumulative += bin_value;
                            if cumulative >= target_intensity {
                                detected_radius = i as f32;
                                break;
                            }
                        }

                        let _ = sender.0.send(detected_radius);
                    }
                    staging_buffer.unmap();
                    gpu_resources.is_mapped = false;
                    false
                }
                Ok(Err(e)) => {
                    error!("Radial buffer map failed: {}", e);
                    gpu_resources.is_mapped = false;
                    false
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => true,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    error!("Radial buffer map channel disconnected");
                    gpu_resources.is_mapped = false;
                    false
                }
            }
        } else {
            true
        };

        if should_reinsert {
            gpu_resources.map_receiver = Some(receiver_mutex);
        }
    } else {
        if !gpu_resources.is_mapped {
            let slice = staging_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(MapMode::Read, move |v| {
                let _ = sender.send(v);
            });
            gpu_resources.map_receiver = Some(std::sync::Mutex::new(receiver));
            gpu_resources.is_mapped = true;
        }
    }
}
```

**Step 2: Add main world update system**

```rust
fn update_radius_from_render(receiver: Res<RadialReceiver>, mut analysis: ResMut<FanAnalysis>) {
    if let Ok(rx) = receiver.0.lock() {
        while let Ok(radius) = rx.try_recv() {
            // Smooth update
            analysis.fan_radius = analysis.fan_radius + (radius - analysis.fan_radius) * 0.1;
        }
    }
}
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 4: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add radial profile readback and radius detection

Reads GPU histogram and calculates 95th percentile radius.
Smoothly updates FanAnalysis.fan_radius via channel."
```

---

## Task 10: Integrate Radial Profile into AnalysisPlugin

**Files:**
- Modify: `src/analysis.rs` (AnalysisPlugin impl)

**Step 1: Update finish() method**

In the `AnalysisPlugin::finish()` method, after centroid setup:
```rust
fn finish(&self, app: &mut App) {
    // Existing centroid channel setup
    let (centroid_sender, centroid_receiver) = std::sync::mpsc::channel();
    app.insert_resource(CentroidReceiver(std::sync::Mutex::new(centroid_receiver)));

    // ADD: Radial channel setup
    let (radial_sender, radial_receiver) = std::sync::mpsc::channel();
    app.insert_resource(RadialReceiver(std::sync::Mutex::new(radial_receiver)));

    let render_app = app.sub_app_mut(RenderApp);
    render_app.insert_resource(CentroidSender(centroid_sender));
    render_app.insert_resource(RadialSender(radial_sender));  // ADD

    render_app
        .init_resource::<CentroidPipeline>()
        .init_resource::<CentroidGpuResources>()
        .init_resource::<RadialProfilePipeline>()      // ADD
        .init_resource::<RadialGpuResources>()          // ADD
        .add_systems(
            Render,
            prepare_centroid_bind_group.in_set(RenderSystems::Prepare),
        )
        .add_systems(
            Render,
            prepare_radial_bind_group.in_set(RenderSystems::Prepare),  // ADD
        )
        .add_systems(
            Render,
            read_centroid_result_render.in_set(RenderSystems::Cleanup),
        )
        .add_systems(
            Render,
            read_radial_result_render.in_set(RenderSystems::Cleanup),  // ADD
        );

    let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
    render_graph.add_node(CentroidLabel, CentroidNode);
    render_graph.add_node(RadialProfileLabel, RadialProfileNode);  // ADD
    render_graph.add_node_edge(CentroidLabel, bevy::render::graph::CameraDriverLabel);
    render_graph.add_node_edge(RadialProfileLabel, bevy::render::graph::CameraDriverLabel);  // ADD
    // ADD: Radial must run after accumulation (needs surface texture)
    render_graph.add_node_edge(EventLabel, RadialProfileLabel);  // ADD

    app.add_systems(Update, update_analysis_from_render);
    app.add_systems(Update, update_radius_from_render);  // ADD
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 3: Test radial detection**

Run: `cargo run`
Expected: App starts, fan radius adjusts based on surface analysis (check UI "Fan Radius" value changes)

**Step 4: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: integrate radial profile pipeline into render graph

Radial analysis now runs every frame after event accumulation.
Fan radius automatically detected from surface intensity."
```

---

## Task 11: Implement Angular Histogram Pipeline (Rust Infrastructure)

**Files:**
- Modify: `src/analysis.rs`

**Step 1: Add GPU resource structs**

After `RadialGpuResources`:
```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct AngularResult {
    bins: [u32; 360],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct AngularParams {
    centroid_x: f32,
    centroid_y: f32,
    radius: f32,
    radius_tolerance: f32,
    window_start: u32,
    window_end: u32,
    _padding: [u32; 2],
}

#[derive(Resource, Default)]
pub struct AngularGpuResources {
    pub result_buffer: Option<Buffer>,
    pub staging_buffer: Option<Buffer>,
    pub bind_group: Option<BindGroup>,
    pub pipeline_ready: bool,
    pub map_receiver: Option<std::sync::Mutex<std::sync::mpsc::Receiver<Result<(), BufferAsyncError>>>>,
    pub is_mapped: bool,
}
```

**Step 2: Add pipeline struct**

```rust
#[derive(Resource)]
pub struct AngularHistogramPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for AngularHistogramPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Angular Histogram Layout"),
            &[
                // Events buffer
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Result buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params uniform
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let shader = world.resource::<AssetServer>().load("shaders/angular_histogram.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Angular Histogram Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        AngularHistogramPipeline { layout, pipeline }
    }
}
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 4: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add angular histogram pipeline infrastructure

Add GPU resources and pipeline for blade angle detection.
Uses 360-bin histogram with 1° resolution."
```

---

## Task 12: Implement Angular Histogram Render Node and Peak Detection

**Files:**
- Modify: `src/analysis.rs`

**Step 1: Add channel resources**

```rust
#[derive(Resource)]
struct AngularSender(pub std::sync::mpsc::Sender<Vec<f32>>);

#[derive(Resource)]
struct AngularReceiver(pub std::sync::Mutex<std::sync::mpsc::Receiver<Vec<f32>>>);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct AngularHistogramLabel;
```

**Step 2: Add render node**

```rust
struct AngularHistogramNode;

impl Node for AngularHistogramNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let gpu_resources = world.resource::<AngularGpuResources>();
        let pipeline = world.resource::<AngularHistogramPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let gpu_event_buffer = world.resource::<crate::gpu::GpuEventBuffer>();
        let render_queue = world.resource::<RenderQueue>();

        if !gpu_resources.pipeline_ready {
            return Ok(());
        }
        let Some(bind_group) = &gpu_resources.bind_group else {
            return Ok(());
        };
        let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
            return Ok(());
        };

        // Reset result buffer
        if let Some(result_buffer) = &gpu_resources.result_buffer {
            let reset_data = AngularResult { bins: [0; 360] };
            render_queue.write_buffer(result_buffer, 0, bytemuck::bytes_of(&reset_data));
        }

        {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Angular Histogram Pass"),
                    timestamp_writes: None,
                });

            pass.set_pipeline(compute_pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            let workgroup_size = 64;
            let count = gpu_event_buffer.count;
            if count > 0 {
                let total_workgroups = (count + workgroup_size - 1) / workgroup_size;
                let max_workgroups_per_dim = 65535;
                let x_workgroups = total_workgroups.min(max_workgroups_per_dim);
                let y_workgroups = (total_workgroups + max_workgroups_per_dim - 1) / max_workgroups_per_dim;
                pass.dispatch_workgroups(x_workgroups, y_workgroups, 1);
            }
        }

        // Copy to staging
        if !gpu_resources.is_mapped {
            if let (Some(result), Some(staging)) =
                (&gpu_resources.result_buffer, &gpu_resources.staging_buffer)
            {
                render_context.command_encoder().copy_buffer_to_buffer(
                    result,
                    0,
                    staging,
                    0,
                    std::mem::size_of::<AngularResult>() as u64,
                );
            }
        }

        Ok(())
    }
}
```

**Step 3: Add peak detection helper**

```rust
fn find_peaks(histogram: &[u32; 360], num_peaks: usize) -> Vec<f32> {
    // Smooth histogram with 3-bin window
    let mut smoothed = [0u32; 360];
    for i in 0..360 {
        let prev = if i == 0 { histogram[359] } else { histogram[i - 1] };
        let next = if i == 359 { histogram[0] } else { histogram[i + 1] };
        smoothed[i] = (prev + histogram[i] * 2 + next) / 4;
    }

    // Find local maxima
    let mut peaks: Vec<(usize, u32)> = Vec::new();
    for i in 0..360 {
        let prev = if i == 0 { smoothed[359] } else { smoothed[i - 1] };
        let next = if i == 359 { smoothed[0] } else { smoothed[i + 1] };

        if smoothed[i] > prev && smoothed[i] > next && smoothed[i] > 10 {
            peaks.push((i, smoothed[i]));
        }
    }

    // Sort by intensity (descending)
    peaks.sort_by(|a, b| b.1.cmp(&a.1));

    // Take top num_peaks and convert to radians
    peaks
        .iter()
        .take(num_peaks)
        .map(|(angle_deg, _)| (*angle_deg as f32) * std::f32::consts::PI / 180.0)
        .collect()
}
```

**Step 4: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 5: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add angular histogram render node and peak detection

Executes angular analysis on event buffer.
CPU peak detection finds blade positions from histogram."
```

---

## Task 13: Add Angular Histogram Bind Group and Readback

**Files:**
- Modify: `src/analysis.rs`

**Step 1: Add preparation system**

```rust
fn prepare_angular_bind_group(
    pipeline: Res<AngularHistogramPipeline>,
    render_device: Res<RenderDevice>,
    mut gpu_resources: ResMut<AngularGpuResources>,
    gpu_event_buffer: Res<crate::gpu::GpuEventBuffer>,
    playback_state: Res<crate::gpu::PlaybackState>,
    analysis: Res<FanAnalysis>,
) {
    let Some(event_buffer) = &gpu_event_buffer.buffer else {
        return;
    };

    // Create buffers
    if gpu_resources.result_buffer.is_none() {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("Angular Result Buffer"),
            size: std::mem::size_of::<AngularResult>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu_resources.result_buffer = Some(buffer);
    }

    if gpu_resources.staging_buffer.is_none() {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("Angular Staging Buffer"),
            size: std::mem::size_of::<AngularResult>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu_resources.staging_buffer = Some(buffer);
    }

    // Create params uniform
    let window_end = playback_state.current_time as u32;
    let window_start = if window_end > playback_state.window_size as u32 {
        window_end - playback_state.window_size as u32
    } else {
        0
    };

    let params = AngularParams {
        centroid_x: analysis.centroid.x,
        centroid_y: analysis.centroid.y,
        radius: analysis.fan_radius,
        radius_tolerance: 30.0,  // Accept events within ±30px of radius
        window_start: window_start * 10,  // Convert to 100ns units
        window_end: window_end * 10,
        _padding: [0; 2],
    };

    let params_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Angular Params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    // Create bind group
    let bind_group = render_device.create_bind_group(
        Some("Angular Histogram Bind Group"),
        &pipeline.layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: event_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: gpu_resources.result_buffer.as_ref().unwrap().as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    );

    gpu_resources.bind_group = Some(bind_group);
    gpu_resources.pipeline_ready = true;
}
```

**Step 2: Add readback system**

```rust
fn read_angular_result_render(
    mut gpu_resources: ResMut<AngularGpuResources>,
    sender: Res<AngularSender>,
    analysis: Res<FanAnalysis>,
) {
    let gpu_resources = gpu_resources.into_inner();
    let Some(staging_buffer) = &gpu_resources.staging_buffer else {
        return;
    };

    if let Some(receiver_mutex) = gpu_resources.map_receiver.take() {
        let should_reinsert = if let Ok(receiver) = receiver_mutex.try_lock() {
            match receiver.try_recv() {
                Ok(Ok(())) => {
                    let slice = staging_buffer.slice(..);
                    {
                        let data = slice.get_mapped_range();
                        let result: AngularResult = *bytemuck::from_bytes(&data);

                        // Detect peaks
                        let blade_angles = find_peaks(&result.bins, analysis.blade_count as usize);

                        let _ = sender.0.send(blade_angles);
                    }
                    staging_buffer.unmap();
                    gpu_resources.is_mapped = false;
                    false
                }
                Ok(Err(e)) => {
                    error!("Angular buffer map failed: {}", e);
                    gpu_resources.is_mapped = false;
                    false
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => true,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    error!("Angular buffer map channel disconnected");
                    gpu_resources.is_mapped = false;
                    false
                }
            }
        } else {
            true
        };

        if should_reinsert {
            gpu_resources.map_receiver = Some(receiver_mutex);
        }
    } else {
        if !gpu_resources.is_mapped {
            let slice = staging_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(MapMode::Read, move |v| {
                let _ = sender.send(v);
            });
            gpu_resources.map_receiver = Some(std::sync::Mutex::new(receiver));
            gpu_resources.is_mapped = true;
        }
    }
}
```

**Step 3: Add main world update**

```rust
fn update_blades_from_render(receiver: Res<AngularReceiver>, mut analysis: ResMut<FanAnalysis>) {
    if let Ok(rx) = receiver.0.lock() {
        while let Ok(blade_angles) = rx.try_recv() {
            analysis.blade_angles = blade_angles;
        }
    }
}
```

**Step 4: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 5: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: add angular histogram bind group and readback

Prepares event buffer with time window and radius filter.
Detects blade angles via peak finding on GPU histogram."
```

---

## Task 14: Integrate Angular Histogram into AnalysisPlugin

**Files:**
- Modify: `src/analysis.rs` (AnalysisPlugin::finish)

**Step 1: Add to finish() method**

```rust
fn finish(&self, app: &mut App) {
    let (centroid_sender, centroid_receiver) = std::sync::mpsc::channel();
    app.insert_resource(CentroidReceiver(std::sync::Mutex::new(centroid_receiver)));

    let (radial_sender, radial_receiver) = std::sync::mpsc::channel();
    app.insert_resource(RadialReceiver(std::sync::Mutex::new(radial_receiver)));

    // ADD: Angular channel setup
    let (angular_sender, angular_receiver) = std::sync::mpsc::channel();
    app.insert_resource(AngularReceiver(std::sync::Mutex::new(angular_receiver)));

    let render_app = app.sub_app_mut(RenderApp);
    render_app.insert_resource(CentroidSender(centroid_sender));
    render_app.insert_resource(RadialSender(radial_sender));
    render_app.insert_resource(AngularSender(angular_sender));  // ADD

    render_app
        .init_resource::<CentroidPipeline>()
        .init_resource::<CentroidGpuResources>()
        .init_resource::<RadialProfilePipeline>()
        .init_resource::<RadialGpuResources>()
        .init_resource::<AngularHistogramPipeline>()    // ADD
        .init_resource::<AngularGpuResources>()         // ADD
        .add_systems(
            Render,
            (
                prepare_centroid_bind_group,
                prepare_radial_bind_group,
                prepare_angular_bind_group,  // ADD
            ).in_set(RenderSystems::Prepare),
        )
        .add_systems(
            Render,
            (
                read_centroid_result_render,
                read_radial_result_render,
                read_angular_result_render,  // ADD
            ).in_set(RenderSystems::Cleanup),
        );

    let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
    render_graph.add_node(CentroidLabel, CentroidNode);
    render_graph.add_node(RadialProfileLabel, RadialProfileNode);
    render_graph.add_node(AngularHistogramLabel, AngularHistogramNode);  // ADD
    render_graph.add_node_edge(CentroidLabel, bevy::render::graph::CameraDriverLabel);
    render_graph.add_node_edge(RadialProfileLabel, bevy::render::graph::CameraDriverLabel);
    render_graph.add_node_edge(AngularHistogramLabel, bevy::render::graph::CameraDriverLabel);  // ADD
    render_graph.add_node_edge(EventLabel, RadialProfileLabel);
    // ADD: Angular runs after centroid (needs centroid + radius)
    render_graph.add_node_edge(CentroidLabel, AngularHistogramLabel);

    app.add_systems(Update, update_analysis_from_render);
    app.add_systems(Update, update_radius_from_render);
    app.add_systems(Update, update_blades_from_render);  // ADD
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 3: Test angular detection**

Run: `cargo run`
Expected: App starts, blade_angles gets populated (check via adding debug print temporarily)

**Step 4: Commit**

```bash
git add src/analysis.rs
git commit -m "feat: integrate angular histogram pipeline into render graph

Angular analysis runs after centroid detection.
Blade angles automatically detected from event distribution."
```

---

## Task 15: Update Gizmo Visualization to Use Detected Blade Angles

**Files:**
- Modify: `src/gizmos.rs:14-74`

**Step 1: Replace blade drawing logic**

In `src/gizmos.rs`, replace the `draw_fan_visualization` function:
```rust
fn draw_fan_visualization(analysis: Res<FanAnalysis>, mut gizmos: Gizmos) {
    if !analysis.show_borders {
        return;
    }

    // Convert centroid to 3D position
    let center = Vec3::new(
        analysis.centroid.x - 640.0,
        -(analysis.centroid.y - 360.0),
        1.0,
    );

    // Draw centroid marker (cross)
    let marker_size = 10.0;
    gizmos.line(
        center + Vec3::new(-marker_size, 0.0, 0.0),
        center + Vec3::new(marker_size, 0.0, 0.0),
        Color::srgb(1.0, 0.0, 0.0),
    );
    gizmos.line(
        center + Vec3::new(0.0, -marker_size, 0.0),
        center + Vec3::new(0.0, marker_size, 0.0),
        Color::srgb(1.0, 0.0, 0.0),
    );

    // Draw blade borders using DETECTED angles
    for &blade_angle in &analysis.blade_angles {
        // Calculate blade edge positions
        let dx = blade_angle.cos() * analysis.fan_radius;
        let dy = blade_angle.sin() * analysis.fan_radius;

        let blade_end = center + Vec3::new(dx, -dy, 0.0);  // Flip Y for screen coords

        // Draw line from center to blade tip
        gizmos.line(
            center,
            blade_end,
            Color::srgb(0.0, 1.0, 0.0),
        );

        // Draw a small circle at the blade tip
        let tip_radius = 5.0;
        gizmos.circle(
            blade_end,
            tip_radius,
            Color::srgb(1.0, 1.0, 0.0),
        );
    }

    // Draw fan radius circle
    gizmos.circle(
        center,
        analysis.fan_radius,
        Color::srgba(0.5, 0.5, 1.0, 0.5),
    );
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 3: Test visualization**

Run: `cargo run`
Actions:
- Load app
- Enable "Show Blade Borders" in UI
- Verify green blade lines align with red/blue event boundaries
- Verify blue circle fits fan activation area
- Adjust "Blade Count" slider and verify detection updates

Expected: Blades now sync with actual event data

**Step 4: Commit**

```bash
git add src/gizmos.rs
git commit -m "feat: use detected blade angles for gizmo visualization

Replace calculated angles with GPU-detected angles from histogram.
Blades now align with actual event boundaries in real-time."
```

---

## Task 16: Remove Obsolete Simulation Code

**Files:**
- Modify: `src/analysis.rs`

**Step 1: Remove update_rotation_angle system**

In `src/analysis.rs`, delete the `update_rotation_angle` function (lines ~424-441):
```rust
// DELETE THIS ENTIRE FUNCTION
fn update_rotation_angle(
    playback_state: Res<crate::gpu::PlaybackState>,
    mut analysis: ResMut<FanAnalysis>,
) {
    // ... entire function body
}
```

**Step 2: Remove from plugin systems**

In `AnalysisPlugin::build()`, remove the system:
```rust
.add_systems(
    Update,
    (
        // DELETE: update_rotation_angle,
        simulate_rpm_detection,
        log_rpm_periodically,
        debug_cpu_centroid,
    ),
)
```

**Step 3: Optionally remove current_angle field**

In `FanAnalysis` struct and Default impl, you can remove `current_angle` (but keep it if you plan to use it for RPM visualization later).

**Step 4: Verify compilation**

Run: `cargo check`
Expected: No errors

**Step 5: Test app**

Run: `cargo run`
Expected: Blades still visualize correctly using detected angles

**Step 6: Commit**

```bash
git add src/analysis.rs
git commit -m "refactor: remove obsolete rotation angle simulation

Blade visualization now uses detected angles exclusively.
Removed update_rotation_angle system (no longer needed)."
```

---

## Task 17: Final Testing and Validation

**Files:**
- None (testing only)

**Step 1: Full build test**

Run: `cargo build --release`
Expected: Clean build, no warnings

**Step 2: Visual validation test**

Run: `cargo run --release`
Actions:
1. Verify data loads (`data/fan/fan_const_rpm.dat`)
2. Open "Motion Analysis" panel
3. Enable "Show Blade Borders"
4. Observe:
   - Blue circle tightly fits fan activation area
   - Green blade lines align with red/blue event boundaries
   - Radius updates smoothly as fan moves
   - Blade angles track actual event patterns

**Step 3: Error handling test**

Run: `mv data/fan/fan_const_rpm.dat /tmp/ && cargo run`
Expected: Red error window appears, app doesn't crash

Restore: `mv /tmp/fan_const_rpm.dat data/fan/`

**Step 4: Linux compatibility test** (if on NixOS)

Run: `nix develop --command cargo run`
Expected: No Vulkan warnings, clean startup

**Step 5: Blade count test**

Run: `cargo run`
Actions:
- Set "Blade Count" to 2, 3, 4, 5
- Verify peak detection adapts to blade count
- Verify visualization shows correct number of blades

**Step 6: Performance test**

Run: `cargo run --release`
Check: FPS in UI should be >60 on modern hardware

**Step 7: Document test results**

Create: `docs/testing/2025-11-23-visualization-accuracy-test-results.md`
Content:
```markdown
# Visualization Accuracy Testing Results

**Date**: 2025-11-23
**Tester**: [Your name]

## Test Results

### Visual Accuracy
- [ ] Fan radius circle fits activation area
- [ ] Blade lines align with event boundaries
- [ ] Smooth tracking during playback
- [ ] Correct blade count detection

### Error Handling
- [ ] Missing file shows error UI
- [ ] App doesn't crash on load failure

### Platform Compatibility
- [ ] macOS: No warnings
- [ ] Linux/NixOS: No Vulkan warnings

### Performance
- FPS: ____
- GPU usage: Normal

## Notes
[Any observations or issues]
```

**Step 8: Final commit**

```bash
git add docs/testing/
git commit -m "test: validate fan visualization accuracy implementation

All tests passing:
- Radius detection accurate
- Blade sync correct
- Error handling works
- Linux/macOS compatible"
```

---

## Task 18: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Add motion analysis section**

In `README.md`, after the "Controls" section:
```markdown
## Controls
*   **Play/Pause**: Toggle playback.
*   **Loop**: Toggle looping.
*   **Time Slider**: Scrub through the dataset.
*   **Window Slider**: Adjust the integration time window (accumulation duration).
*   **Speed Slider**: Adjust playback speed.

## Motion Analysis (NEW)
*   **Enable RPM Tracking**: Activate real-time analysis.
*   **Show Blade Borders**: Visualize detected fan geometry:
    *   **Blue Circle**: Auto-detected fan radius from intensity falloff
    *   **Green Lines**: Blade positions from angular event distribution
    *   **Yellow Dots**: Blade tip markers
    *   **Red Cross**: Centroid (fan center)
*   **Blade Count**: Number of blades to detect (2-8).

### How It Works
The visualizer uses GPU compute shaders to analyze event patterns:
1. **Centroid Tracking**: Calculates spatial mean of events
2. **Radial Analysis**: Builds intensity histogram to find 95th percentile radius
3. **Angular Histogram**: Detects blade positions via peak finding in polar distribution

See `docs/design/2025-11-23-fan-visualization-accuracy-design.md` for details.
```

**Step 2: Verify markdown**

Run: `cat README.md` (visually check formatting)

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README with motion analysis features

Document new blade detection and radius analysis capabilities.
Add explanation of GPU-based analysis pipeline."
```

---

## Summary

**Total Tasks**: 18
**Estimated Time**: 4-6 hours
**Architecture**: GPU-centric compute pipeline
**Key Technologies**: WGSL compute shaders, Bevy 0.17 render graph, async GPU readback

**Testing Strategy**:
- Visual validation (blade alignment, radius fit)
- Error handling (missing files)
- Platform compatibility (Linux Vulkan)
- Performance (FPS monitoring)

**Success Criteria**:
✅ Blade lines align with event boundaries
✅ Radius circle fits activation area
✅ No crashes on missing data
✅ No Vulkan warnings on Linux
✅ Clean clippy build
✅ Documentation updated

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-11-23-fan-visualization-accuracy.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**

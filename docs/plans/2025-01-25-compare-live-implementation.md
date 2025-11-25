# Compare Live Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a binary that displays Raw/Sobel/Canny/LoG detectors in a 2x2 grid with real-time metrics.

**Architecture:** GPU composite shader combines 4 detector textures into single output. Multi-readback extracts all detector outputs for metrics. Egui overlay displays per-quadrant metrics.

**Tech Stack:** Bevy 0.15, WGSL compute shaders, egui, toml, clap

---

### Task 1: TOML Config Module

**Files:**
- Create: `src/compare/mod.rs`
- Create: `src/compare/config.rs`
- Modify: `src/lib.rs` (add module export)

**Step 1: Create module structure**

Create `src/compare/mod.rs`:
```rust
pub mod config;

pub use config::CompareConfig;
```

**Step 2: Write config types**

Create `src/compare/config.rs`:
```rust
//! Configuration for compare_live binary.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Per-detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_window_size")]
    pub window_size_us: f32,
    #[serde(default = "default_true")]
    pub filter_dead_pixels: bool,
}

fn default_threshold() -> f32 { 50.0 }
fn default_window_size() -> f32 { 100000.0 }
fn default_true() -> bool { true }

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            threshold: default_threshold(),
            window_size_us: default_window_size(),
            filter_dead_pixels: default_true(),
        }
    }
}

/// Canny-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CannyConfig {
    #[serde(default = "default_canny_low")]
    pub low_threshold: f32,
    #[serde(default = "default_canny_high")]
    pub high_threshold: f32,
    #[serde(default = "default_window_size")]
    pub window_size_us: f32,
    #[serde(default = "default_true")]
    pub filter_dead_pixels: bool,
}

fn default_canny_low() -> f32 { 50.0 }
fn default_canny_high() -> f32 { 150.0 }

impl Default for CannyConfig {
    fn default() -> Self {
        Self {
            low_threshold: default_canny_low(),
            high_threshold: default_canny_high(),
            window_size_us: default_window_size(),
            filter_dead_pixels: default_true(),
        }
    }
}

/// Display settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    #[serde(default = "default_true")]
    pub show_ground_truth: bool,
    #[serde(default = "default_metrics_hz")]
    pub metrics_update_hz: u32,
}

fn default_metrics_hz() -> u32 { 10 }

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            show_ground_truth: true,
            metrics_update_hz: default_metrics_hz(),
        }
    }
}

/// Complete configuration for compare_live
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompareConfig {
    #[serde(default)]
    pub sobel: DetectorConfig,
    #[serde(default)]
    pub canny: CannyConfig,
    #[serde(default)]
    pub log: DetectorConfig,
    #[serde(default)]
    pub display: DisplayConfig,
}

impl CompareConfig {
    /// Load from TOML file
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: CompareConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Load with fallback chain: path -> config/detectors.toml -> results/best_*.json -> defaults
    pub fn load_with_fallback(path: Option<&Path>) -> Self {
        // Try explicit path first
        if let Some(p) = path {
            if let Ok(config) = Self::load(p) {
                return config;
            }
        }

        // Try default TOML location
        let default_toml = Path::new("config/detectors.toml");
        if default_toml.exists() {
            if let Ok(config) = Self::load(default_toml) {
                return config;
            }
        }

        // Try loading from hypersearch JSON files
        if let Some(config) = Self::from_hypersearch_results() {
            return config;
        }

        // Fall back to defaults
        Self::default()
    }

    /// Load from results/best_*.json files
    fn from_hypersearch_results() -> Option<Self> {
        use crate::HyperConfig;

        let mut config = CompareConfig::default();
        let mut found_any = false;

        // Load Sobel config
        if let Ok(contents) = std::fs::read_to_string("results/best_sobel.json") {
            if let Ok(hyper) = serde_json::from_str::<HyperConfig>(&contents) {
                config.sobel.threshold = hyper.threshold;
                config.sobel.window_size_us = hyper.window_size_us;
                config.sobel.filter_dead_pixels = hyper.filter_dead_pixels;
                found_any = true;
            }
        }

        // Load Canny config
        if let Ok(contents) = std::fs::read_to_string("results/best_canny.json") {
            if let Ok(hyper) = serde_json::from_str::<HyperConfig>(&contents) {
                config.canny.low_threshold = hyper.canny_low;
                config.canny.high_threshold = hyper.canny_high;
                config.canny.window_size_us = hyper.window_size_us;
                config.canny.filter_dead_pixels = hyper.filter_dead_pixels;
                found_any = true;
            }
        }

        // Load LoG config
        if let Ok(contents) = std::fs::read_to_string("results/best_log.json") {
            if let Ok(hyper) = serde_json::from_str::<HyperConfig>(&contents) {
                config.log.threshold = hyper.threshold;
                config.log.window_size_us = hyper.window_size_us;
                config.log.filter_dead_pixels = hyper.filter_dead_pixels;
                found_any = true;
            }
        }

        if found_any { Some(config) } else { None }
    }

    /// Save to TOML file
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let contents = toml::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, contents)?;
        Ok(())
    }
}
```

**Step 3: Add module to lib.rs**

Modify `src/lib.rs` - add after existing modules:
```rust
pub mod compare;
```

**Step 4: Add toml dependency**

Run: `cargo add toml`

**Step 5: Verify it compiles**

Run: `cargo build --lib`
Expected: Compiles without errors

**Step 6: Commit**

```bash
git add src/compare/ src/lib.rs Cargo.toml Cargo.lock
git commit -m "feat(compare): add TOML config module with fallback chain"
```

---

### Task 2: Composite Shader

**Files:**
- Create: `assets/shaders/composite.wgsl`

**Step 1: Write the composite shader**

Create `assets/shaders/composite.wgsl`:
```wgsl
// Composite shader: combines 4 detector outputs into 2x2 grid
// Output: 2560x1440 (2x base resolution of 1280x720)

@group(0) @binding(0) var raw_texture: texture_2d<f32>;
@group(0) @binding(1) var sobel_texture: texture_2d<f32>;
@group(0) @binding(2) var canny_texture: texture_2d<f32>;
@group(0) @binding(3) var log_texture: texture_2d<f32>;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba8unorm, write>;

const BASE_WIDTH: u32 = 1280u;
const BASE_HEIGHT: u32 = 720u;

// Color scheme for each detector
const RAW_COLOR: vec3<f32> = vec3<f32>(0.8, 0.8, 0.8);    // Light gray
const SOBEL_COLOR: vec3<f32> = vec3<f32>(1.0, 0.4, 0.4);  // Red
const CANNY_COLOR: vec3<f32> = vec3<f32>(0.4, 1.0, 0.4);  // Green
const LOG_COLOR: vec3<f32> = vec3<f32>(0.4, 0.4, 1.0);    // Blue

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_x = global_id.x;
    let output_y = global_id.y;

    // Bounds check for 2560x1440
    if (output_x >= BASE_WIDTH * 2u || output_y >= BASE_HEIGHT * 2u) {
        return;
    }

    let coords = vec2<i32>(i32(output_x), i32(output_y));

    // Determine which quadrant we're in
    let is_right = output_x >= BASE_WIDTH;
    let is_bottom = output_y >= BASE_HEIGHT;

    // Map to source texture coordinates
    let source_x = output_x % BASE_WIDTH;
    let source_y = output_y % BASE_HEIGHT;
    let source_coords = vec2<i32>(i32(source_x), i32(source_y));

    var value: f32 = 0.0;
    var color: vec3<f32>;

    if (!is_right && !is_bottom) {
        // Top-left: Raw events
        value = textureLoad(raw_texture, source_coords, 0).r;
        color = RAW_COLOR;
    } else if (is_right && !is_bottom) {
        // Top-right: Sobel
        value = textureLoad(sobel_texture, source_coords, 0).r;
        color = SOBEL_COLOR;
    } else if (!is_right && is_bottom) {
        // Bottom-left: Canny
        value = textureLoad(canny_texture, source_coords, 0).r;
        color = CANNY_COLOR;
    } else {
        // Bottom-right: LoG
        value = textureLoad(log_texture, source_coords, 0).r;
        color = LOG_COLOR;
    }

    // Normalize value for display (edge magnitude -> intensity)
    let intensity = clamp(value / 1000.0, 0.0, 1.0);

    // Draw border between quadrants (2px wide)
    let border_x = output_x == BASE_WIDTH - 1u || output_x == BASE_WIDTH;
    let border_y = output_y == BASE_HEIGHT - 1u || output_y == BASE_HEIGHT;

    var output_color: vec4<f32>;
    if (border_x || border_y) {
        output_color = vec4<f32>(0.3, 0.3, 0.3, 1.0); // Dark gray border
    } else {
        output_color = vec4<f32>(color * intensity, 1.0);
    }

    textureStore(output_texture, coords, output_color);
}
```

**Step 2: Verify shader syntax**

Run: `cargo build` (shaders are validated at runtime, but this ensures no Rust errors)

**Step 3: Commit**

```bash
git add assets/shaders/composite.wgsl
git commit -m "feat(compare): add 2x2 composite shader"
```

---

### Task 3: Composite GPU Pipeline

**Files:**
- Create: `src/compare/composite.rs`
- Modify: `src/compare/mod.rs`

**Step 1: Write the composite pipeline**

Create `src/compare/composite.rs`:
```rust
//! GPU composite pipeline for 2x2 grid rendering.

use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;

use crate::gpu::{FilteredSurfaceImage, SobelImage, CannyImage, LogImage};

/// Output composite image (2560x1440)
#[derive(Resource, Clone, Default)]
pub struct CompositeImage {
    pub handle: Handle<Image>,
}

/// Label for composite render node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct CompositeLabel;

/// Composite compute pipeline
#[derive(Resource)]
pub struct CompositePipeline {
    pub pipeline: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
}

impl FromWorld for CompositePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "composite_bind_group_layout",
            &[
                // Raw/filtered surface (input)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sobel (input)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Canny (input)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // LoG (input)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output composite (output)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        );

        let shader = world.load_asset("shaders/composite.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("composite_pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            shader,
            shader_defs: vec![],
            entry_point: "main".into(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

/// Composite bind group
#[derive(Resource, Default)]
pub struct CompositeBindGroup {
    pub bind_group: Option<BindGroup>,
}

/// System to prepare composite bind group
pub fn prepare_composite(
    mut bind_group: ResMut<CompositeBindGroup>,
    pipeline: Res<CompositePipeline>,
    render_device: Res<RenderDevice>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    filtered_image: Res<FilteredSurfaceImage>,
    sobel_image: Res<SobelImage>,
    canny_image: Res<CannyImage>,
    log_image: Res<LogImage>,
    composite_image: Res<CompositeImage>,
) {
    let (Some(filtered), Some(sobel), Some(canny), Some(log), Some(composite)) = (
        gpu_images.get(&filtered_image.handle),
        gpu_images.get(&sobel_image.handle),
        gpu_images.get(&canny_image.handle),
        gpu_images.get(&log_image.handle),
        gpu_images.get(&composite_image.handle),
    ) else {
        return;
    };

    bind_group.bind_group = Some(render_device.create_bind_group(
        "composite_bind_group",
        &pipeline.bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&filtered.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&sobel.texture_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&canny.texture_view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&log.texture_view),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&composite.texture_view),
            },
        ],
    ));
}

/// Composite render node
#[derive(Default)]
pub struct CompositeNode;

impl Node for CompositeNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_res = world.resource::<CompositePipeline>();
        let bind_group = world.resource::<CompositeBindGroup>();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.pipeline) else {
            return Ok(());
        };

        let Some(ref bind_group) = bind_group.bind_group else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("composite_pass"),
                timestamp_writes: None,
            });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        // Dispatch for 2560x1440 output with 8x8 workgroups
        let workgroups_x = (2560 + 7) / 8;
        let workgroups_y = (1440 + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);

        Ok(())
    }
}
```

**Step 2: Update mod.rs**

Modify `src/compare/mod.rs`:
```rust
pub mod composite;
pub mod config;

pub use composite::{CompositeImage, CompositeLabel, CompositeNode, CompositePipeline, CompositeBindGroup, prepare_composite};
pub use config::CompareConfig;
```

**Step 3: Verify it compiles**

Run: `cargo build --lib`

**Step 4: Commit**

```bash
git add src/compare/composite.rs src/compare/mod.rs
git commit -m "feat(compare): add GPU composite pipeline and render node"
```

---

### Task 4: Multi-Readback for All Detectors

**Files:**
- Create: `src/compare/multi_readback.rs`
- Modify: `src/compare/mod.rs`

**Step 1: Write multi-readback module**

Create `src/compare/multi_readback.rs`:
```rust
//! Multi-detector readback for metrics computation.

use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderDevice;
use std::sync::mpsc::{Sender, Receiver};
use std::sync::Mutex;

/// Metrics for a single detector
#[derive(Debug, Clone, Default)]
pub struct DetectorMetrics {
    pub edge_count: u32,
    pub tolerance_precision: f32,
    pub tolerance_recall: f32,
    pub tolerance_f1: f32,
    pub avg_distance: f32,
}

/// Combined metrics for all detectors
#[derive(Resource, Debug, Clone, Default)]
pub struct AllDetectorMetrics {
    pub raw: DetectorMetrics,
    pub sobel: DetectorMetrics,
    pub canny: DetectorMetrics,
    pub log: DetectorMetrics,
    pub frame_time_ms: f32,
    pub last_update: f64,
}

/// Channel to send metrics from render world to main world
#[derive(Resource)]
pub struct MetricsSender(pub Sender<AllDetectorMetrics>);

#[derive(Resource)]
pub struct MetricsReceiver(pub Mutex<Receiver<AllDetectorMetrics>>);

/// Per-detector edge data for metrics computation
#[derive(Debug, Clone, Default)]
pub struct DetectorEdgeData {
    pub pixels: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Combined edge data from all detectors
#[derive(Resource, Default)]
pub struct AllEdgeData {
    pub raw: DetectorEdgeData,
    pub sobel: DetectorEdgeData,
    pub canny: DetectorEdgeData,
    pub log: DetectorEdgeData,
}

/// System to receive metrics in main world
pub fn receive_metrics(
    receiver: Res<MetricsReceiver>,
    mut metrics: ResMut<AllDetectorMetrics>,
    time: Res<Time>,
) {
    if let Ok(rx) = receiver.0.try_lock() {
        while let Ok(new_metrics) = rx.try_recv() {
            *metrics = new_metrics;
            metrics.last_update = time.elapsed_secs_f64();
        }
    }
}
```

**Step 2: Update mod.rs**

Add to `src/compare/mod.rs`:
```rust
pub mod composite;
pub mod config;
pub mod multi_readback;

pub use composite::{CompositeImage, CompositeLabel, CompositeNode, CompositePipeline, CompositeBindGroup, prepare_composite};
pub use config::CompareConfig;
pub use multi_readback::{AllDetectorMetrics, DetectorMetrics, MetricsSender, MetricsReceiver, AllEdgeData, DetectorEdgeData, receive_metrics};
```

**Step 3: Verify it compiles**

Run: `cargo build --lib`

**Step 4: Commit**

```bash
git add src/compare/multi_readback.rs src/compare/mod.rs
git commit -m "feat(compare): add multi-detector metrics types and channels"
```

---

### Task 5: Egui Metrics Overlay

**Files:**
- Create: `src/compare/ui.rs`
- Modify: `src/compare/mod.rs`

**Step 1: Write UI module**

Create `src/compare/ui.rs`:
```rust
//! Egui overlay for metrics display.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use super::{AllDetectorMetrics, DetectorMetrics};

/// Resource tracking which data file is active
#[derive(Resource, Default)]
pub struct DataFileState {
    pub files: Vec<std::path::PathBuf>,
    pub current_index: usize,
}

impl DataFileState {
    pub fn current_file(&self) -> Option<&std::path::PathBuf> {
        self.files.get(self.current_index)
    }

    pub fn next(&mut self) {
        if !self.files.is_empty() {
            self.current_index = (self.current_index + 1) % self.files.len();
        }
    }

    pub fn prev(&mut self) {
        if !self.files.is_empty() {
            self.current_index = (self.current_index + self.files.len() - 1) % self.files.len();
        }
    }
}

/// Draw metrics overlay for compare_live
pub fn draw_metrics_overlay(
    mut contexts: EguiContexts,
    metrics: Res<AllDetectorMetrics>,
    file_state: Res<DataFileState>,
    time: Res<Time>,
) {
    let ctx = contexts.ctx_mut();

    // Top bar with file info
    egui::TopBottomPanel::top("file_info").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if let Some(path) = file_state.current_file() {
                ui.label(format!("File: {} ({}/{})",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    file_state.current_index + 1,
                    file_state.files.len()
                ));
            }
            ui.separator();
            ui.label(format!("Frame: {:.1}ms", metrics.frame_time_ms));
            ui.separator();
            ui.label("[N]ext [P]rev file | [Space] pause");
        });
    });

    // Metrics panels for each quadrant
    let panel_width = 200.0;
    let panel_height = 100.0;

    // Top-left: Raw
    draw_detector_panel(ctx, "RAW", &metrics.raw, 10.0, 40.0, panel_width, panel_height);

    // Top-right: Sobel
    let screen_width = ctx.screen_rect().width();
    draw_detector_panel(ctx, "SOBEL", &metrics.sobel, screen_width / 2.0 + 10.0, 40.0, panel_width, panel_height);

    // Bottom-left: Canny
    let screen_height = ctx.screen_rect().height();
    draw_detector_panel(ctx, "CANNY", &metrics.canny, 10.0, screen_height / 2.0 + 10.0, panel_width, panel_height);

    // Bottom-right: LoG
    draw_detector_panel(ctx, "LoG", &metrics.log, screen_width / 2.0 + 10.0, screen_height / 2.0 + 10.0, panel_width, panel_height);
}

fn draw_detector_panel(
    ctx: &egui::Context,
    name: &str,
    metrics: &DetectorMetrics,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
) {
    egui::Window::new(name)
        .fixed_pos([x, y])
        .fixed_size([width, height])
        .title_bar(false)
        .frame(egui::Frame::window(&ctx.style()).fill(egui::Color32::from_rgba_unmultiplied(20, 20, 20, 200)))
        .show(ctx, |ui| {
            ui.heading(name);
            ui.separator();
            ui.label(format!("Edges: {}", metrics.edge_count));
            ui.label(format!("Prec: {:.1}% | Rec: {:.1}%",
                metrics.tolerance_precision * 100.0,
                metrics.tolerance_recall * 100.0
            ));
            ui.label(format!("F1: {:.1}% | Dist: {:.1}px",
                metrics.tolerance_f1 * 100.0,
                metrics.avg_distance
            ));
        });
}

/// Handle keyboard input for file switching
pub fn handle_file_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut file_state: ResMut<DataFileState>,
) {
    if keyboard.just_pressed(KeyCode::KeyN) {
        file_state.next();
    }
    if keyboard.just_pressed(KeyCode::KeyP) {
        file_state.prev();
    }
}

/// Plugin for compare UI
pub struct CompareUiPlugin;

impl Plugin for CompareUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DataFileState>()
            .init_resource::<AllDetectorMetrics>()
            .add_systems(Update, (draw_metrics_overlay, handle_file_input));
    }
}
```

**Step 2: Update mod.rs**

Add to `src/compare/mod.rs`:
```rust
pub mod composite;
pub mod config;
pub mod multi_readback;
pub mod ui;

pub use composite::{CompositeImage, CompositeLabel, CompositeNode, CompositePipeline, CompositeBindGroup, prepare_composite};
pub use config::CompareConfig;
pub use multi_readback::{AllDetectorMetrics, DetectorMetrics, MetricsSender, MetricsReceiver, AllEdgeData, DetectorEdgeData, receive_metrics};
pub use ui::{CompareUiPlugin, DataFileState, draw_metrics_overlay, handle_file_input};
```

**Step 3: Verify it compiles**

Run: `cargo build --lib`

**Step 4: Commit**

```bash
git add src/compare/ui.rs src/compare/mod.rs
git commit -m "feat(compare): add egui metrics overlay UI"
```

---

### Task 6: Compare Live Binary

**Files:**
- Create: `src/bin/compare_live.rs`

**Step 1: Write the binary**

Create `src/bin/compare_live.rs`:
```rust
//! Compare Live: Side-by-side detector visualization with real-time metrics.
//!
//! Usage:
//!   cargo run --bin compare_live -- [OPTIONS] <DATA_FILES>...
//!
//! Options:
//!   --config <PATH>     Config file (default: config/detectors.toml)
//!   --window <SIZE>     Override window size for all detectors
//!   --no-gt             Disable ground truth metrics computation

use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::render::render_resource::*;
use bevy::window::WindowResolution;
use bevy_egui::EguiPlugin;
use clap::Parser;
use std::path::PathBuf;

use ebc_rs::compare::{
    CompareConfig, CompareUiPlugin, CompositeImage, DataFileState,
};
use ebc_rs::edge_detection::EdgeDetectionPlugin;
use ebc_rs::EventFilePath;

#[derive(Parser, Debug)]
#[command(name = "compare_live")]
#[command(about = "Side-by-side detector comparison with real-time metrics")]
struct Args {
    /// Data files to visualize
    #[arg(required = true)]
    files: Vec<PathBuf>,

    /// Config file path
    #[arg(long, default_value = "config/detectors.toml")]
    config: PathBuf,

    /// Override window size for all detectors (microseconds)
    #[arg(long)]
    window: Option<f32>,

    /// Disable ground truth metrics computation
    #[arg(long)]
    no_gt: bool,
}

fn main() {
    let args = Args::parse();

    // Validate files exist
    let valid_files: Vec<PathBuf> = args.files.iter()
        .filter(|f| {
            if !f.exists() {
                eprintln!("Warning: File not found: {}", f.display());
                false
            } else {
                true
            }
        })
        .cloned()
        .collect();

    if valid_files.is_empty() {
        eprintln!("Error: No valid data files provided");
        std::process::exit(1);
    }

    // Load config
    let config = CompareConfig::load_with_fallback(
        if args.config.exists() { Some(&args.config) } else { None }
    );
    println!("Config loaded: {:?}", config);

    // Use first file as initial
    let first_file = valid_files[0].clone();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Compare Live - Edge Detector Comparison".to_string(),
                resolution: WindowResolution::new(2560.0, 1440.0),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin)
        .add_plugins(EdgeDetectionPlugin)
        .add_plugins(CompareUiPlugin)
        .insert_resource(EventFilePath(first_file.to_string_lossy().to_string()))
        .insert_resource(DataFileState {
            files: valid_files,
            current_index: 0,
        })
        .insert_resource(config)
        .add_systems(Startup, setup_composite_texture)
        .run();
}

fn setup_composite_texture(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    // Create 2560x1440 composite output texture
    let size = Extent3d {
        width: 2560,
        height: 1440,
        depth_or_array_layers: 1,
    };

    let mut composite = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    composite.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;

    let handle = images.add(composite);
    commands.insert_resource(CompositeImage { handle: handle.clone() });

    // Spawn sprite to display composite
    commands.spawn((
        Sprite {
            image: handle,
            custom_size: Some(Vec2::new(2560.0, 1440.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Camera
    commands.spawn((
        Camera2d,
        Transform::from_xyz(1280.0, 720.0, 100.0),
    ));
}
```

**Step 2: Verify it compiles**

Run: `cargo build --bin compare_live`

**Step 3: Commit**

```bash
git add src/bin/compare_live.rs
git commit -m "feat(compare): add compare_live binary with CLI"
```

---

### Task 7: Integrate Composite into Render Graph

**Files:**
- Modify: `src/compare/composite.rs`
- Create: `src/compare/plugin.rs`
- Modify: `src/compare/mod.rs`

**Step 1: Create plugin module**

Create `src/compare/plugin.rs`:
```rust
//! Plugin that integrates composite rendering into the render graph.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_graph::RenderGraph;
use bevy::render::{Render, RenderApp, RenderSystems};

use super::{
    CompositeBindGroup, CompositeImage, CompositeLabel, CompositeNode,
    CompositePipeline, prepare_composite,
};
use crate::gpu::LogLabel;

/// Plugin that adds composite rendering to the render graph
pub struct CompositeRenderPlugin;

impl Plugin for CompositeRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CompositeImage>()
            .add_plugins(ExtractResourcePlugin::<CompositeImage>::default());
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<CompositePipeline>()
            .init_resource::<CompositeBindGroup>()
            .add_systems(Render, prepare_composite.in_set(RenderSystems::Queue));

        // Add composite node to render graph after LoG
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(CompositeLabel, CompositeNode::default());
        render_graph.add_node_edge(LogLabel, CompositeLabel);
        render_graph.add_node_edge(CompositeLabel, bevy::render::graph::CameraDriverLabel);
    }
}
```

**Step 2: Update mod.rs**

Update `src/compare/mod.rs`:
```rust
pub mod composite;
pub mod config;
pub mod multi_readback;
pub mod plugin;
pub mod ui;

pub use composite::{CompositeImage, CompositeLabel, CompositeNode, CompositePipeline, CompositeBindGroup, prepare_composite};
pub use config::CompareConfig;
pub use multi_readback::{AllDetectorMetrics, DetectorMetrics, MetricsSender, MetricsReceiver, AllEdgeData, DetectorEdgeData, receive_metrics};
pub use plugin::CompositeRenderPlugin;
pub use ui::{CompareUiPlugin, DataFileState, draw_metrics_overlay, handle_file_input};
```

**Step 3: Update compare_live.rs to use plugin**

Modify `src/bin/compare_live.rs` - add the plugin:
```rust
// Add import at top
use ebc_rs::compare::CompositeRenderPlugin;

// In main(), add plugin after EdgeDetectionPlugin:
        .add_plugins(EdgeDetectionPlugin)
        .add_plugins(CompositeRenderPlugin)  // Add this line
        .add_plugins(CompareUiPlugin)
```

**Step 4: Verify it compiles**

Run: `cargo build --bin compare_live`

**Step 5: Commit**

```bash
git add src/compare/plugin.rs src/compare/mod.rs src/bin/compare_live.rs
git commit -m "feat(compare): integrate composite pipeline into render graph"
```

---

### Task 8: Fix Shader Texture Types

**Files:**
- Modify: `assets/shaders/composite.wgsl`

**Step 1: Fix texture types to match actual formats**

The filtered surface is R32Uint, Sobel/Canny/LoG are R32Float. Update shader:

Modify `assets/shaders/composite.wgsl`:
```wgsl
// Composite shader: combines 4 detector outputs into 2x2 grid
// Output: 2560x1440 (2x base resolution of 1280x720)

// Input textures have different formats:
// - raw_texture: R32Uint (event counts)
// - sobel/canny/log: R32Float (edge magnitudes)
@group(0) @binding(0) var raw_texture: texture_2d<u32>;
@group(0) @binding(1) var sobel_texture: texture_2d<f32>;
@group(0) @binding(2) var canny_texture: texture_2d<f32>;
@group(0) @binding(3) var log_texture: texture_2d<f32>;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba8unorm, write>;

const BASE_WIDTH: u32 = 1280u;
const BASE_HEIGHT: u32 = 720u;

// Color scheme for each detector
const RAW_COLOR: vec3<f32> = vec3<f32>(0.8, 0.8, 0.8);    // Light gray
const SOBEL_COLOR: vec3<f32> = vec3<f32>(1.0, 0.4, 0.4);  // Red
const CANNY_COLOR: vec3<f32> = vec3<f32>(0.4, 1.0, 0.4);  // Green
const LOG_COLOR: vec3<f32> = vec3<f32>(0.4, 0.4, 1.0);    // Blue

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_x = global_id.x;
    let output_y = global_id.y;

    // Bounds check for 2560x1440
    if (output_x >= BASE_WIDTH * 2u || output_y >= BASE_HEIGHT * 2u) {
        return;
    }

    let coords = vec2<i32>(i32(output_x), i32(output_y));

    // Determine which quadrant we're in
    let is_right = output_x >= BASE_WIDTH;
    let is_bottom = output_y >= BASE_HEIGHT;

    // Map to source texture coordinates
    let source_x = output_x % BASE_WIDTH;
    let source_y = output_y % BASE_HEIGHT;
    let source_coords = vec2<i32>(i32(source_x), i32(source_y));

    var intensity: f32 = 0.0;
    var color: vec3<f32>;

    if (!is_right && !is_bottom) {
        // Top-left: Raw events (u32 -> normalize)
        let raw_value = textureLoad(raw_texture, source_coords, 0).r;
        intensity = clamp(f32(raw_value) / 10.0, 0.0, 1.0);
        color = RAW_COLOR;
    } else if (is_right && !is_bottom) {
        // Top-right: Sobel
        let value = textureLoad(sobel_texture, source_coords, 0).r;
        intensity = clamp(value / 500.0, 0.0, 1.0);
        color = SOBEL_COLOR;
    } else if (!is_right && is_bottom) {
        // Bottom-left: Canny
        let value = textureLoad(canny_texture, source_coords, 0).r;
        intensity = clamp(value / 500.0, 0.0, 1.0);
        color = CANNY_COLOR;
    } else {
        // Bottom-right: LoG
        let value = textureLoad(log_texture, source_coords, 0).r;
        intensity = clamp(value / 500.0, 0.0, 1.0);
        color = LOG_COLOR;
    }

    // Draw border between quadrants (2px wide)
    let border_x = output_x == BASE_WIDTH - 1u || output_x == BASE_WIDTH;
    let border_y = output_y == BASE_HEIGHT - 1u || output_y == BASE_HEIGHT;

    var output_color: vec4<f32>;
    if (border_x || border_y) {
        output_color = vec4<f32>(0.3, 0.3, 0.3, 1.0); // Dark gray border
    } else {
        output_color = vec4<f32>(color * intensity, 1.0);
    }

    textureStore(output_texture, coords, output_color);
}
```

**Step 2: Update bind group layout to match**

Modify `src/compare/composite.rs` - change binding 0 type:
```rust
                // Raw/filtered surface (input) - R32Uint
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Uint,  // Changed from Float
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
```

**Step 3: Verify it compiles**

Run: `cargo build --bin compare_live`

**Step 4: Commit**

```bash
git add assets/shaders/composite.wgsl src/compare/composite.rs
git commit -m "fix(compare): correct texture types in composite shader"
```

---

### Task 9: Test Run

**Step 1: Generate test data if needed**

Run: `cargo run --bin generate_synthetic`

**Step 2: Run compare_live**

Run: `cargo run --bin compare_live -- data/synthetic/fan_test.dat`

Expected: Window opens with 2x2 grid showing all detectors with metrics overlays.

**Step 3: Test multiple files**

Run: `cargo run --bin compare_live -- data/synthetic/fan_test.dat data/fan/fan_const_rpm.dat`

Expected: Can switch between files with N/P keys.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix(compare): runtime fixes from testing"
```

---

### Task 10: Update Hypersearch to Write TOML

**Files:**
- Modify: `src/bin/hypersearch.rs`

**Step 1: Add TOML export after finding best configs**

Add to `src/bin/hypersearch.rs` after saving JSON files:

```rust
    // Export unified TOML config
    use ebc_rs::compare::CompareConfig;

    let mut config = CompareConfig::default();

    if let Some(best) = best_by_detector.get("sobel") {
        config.sobel.threshold = best.config.threshold;
        config.sobel.window_size_us = best.config.window_size_us;
        config.sobel.filter_dead_pixels = best.config.filter_dead_pixels;
    }
    if let Some(best) = best_by_detector.get("canny") {
        config.canny.low_threshold = best.config.canny_low;
        config.canny.high_threshold = best.config.canny_high;
        config.canny.window_size_us = best.config.window_size_us;
        config.canny.filter_dead_pixels = best.config.filter_dead_pixels;
    }
    if let Some(best) = best_by_detector.get("log") {
        config.log.threshold = best.config.threshold;
        config.log.window_size_us = best.config.window_size_us;
        config.log.filter_dead_pixels = best.config.filter_dead_pixels;
    }

    let toml_path = std::path::Path::new("config/detectors.toml");
    if let Err(e) = config.save(toml_path) {
        eprintln!("Warning: Failed to save TOML config: {}", e);
    } else {
        println!("\nConfig saved to: {}", toml_path.display());
    }
```

**Step 2: Verify it compiles**

Run: `cargo build --bin hypersearch`

**Step 3: Test hypersearch writes TOML**

Run: `cargo run --release --bin hypersearch -- --data data/synthetic/fan_test.dat --frames 10`

Expected: Creates `config/detectors.toml`

**Step 4: Commit**

```bash
git add src/bin/hypersearch.rs
git commit -m "feat(hypersearch): export best configs to TOML"
```

---

## Summary

This plan creates `compare_live` with:
- 2x2 grid GPU composite shader
- Per-detector metrics overlay via egui
- Multi-file support with N/P keys
- TOML config loading with hypersearch integration
- Reuses existing EdgeDetectionPlugin infrastructure

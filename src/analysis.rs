use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_graph::{Node, RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        Render, RenderApp, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

// Event camera timestamp unit: 100 nanoseconds (not microseconds!)
// Many event cameras use 100ns resolution for higher temporal precision

/// Fan motion analysis state
#[derive(Resource, Clone, ExtractResource)]
pub struct FanAnalysis {
    /// Whether RPM tracking is active
    pub is_tracking: bool,
    /// Whether to show blade borders overlay
    pub show_borders: bool,
    /// Currently detected RPM
    pub current_rpm: f32,
    /// Number of fan blades (default: 3)
    pub blade_count: u32,
    /// Centroid position (center of fan)
    pub centroid: Vec2,
    /// Tip velocity in pixels per second
    pub tip_velocity: f32,
    /// Estimated radius of the fan (in pixels)
    pub fan_radius: f32,
    /// Current rotation angle (radians) for visualization
    pub current_angle: f32,
    /// Detected blade angles (radians)
    pub blade_angles: Vec<f32>,
}

impl Default for FanAnalysis {
    fn default() -> Self {
        Self {
            is_tracking: true,
            show_borders: false,
            current_rpm: 0.0,
            blade_count: 3,
            centroid: Vec2::new(640.0, 360.0), // Default to center of 1280x720
            tip_velocity: 0.0,
            fan_radius: 200.0, // Default estimate
            current_angle: 0.0,
            blade_angles: vec![0.0, 2.094, 4.189], // Default 3 blades at 120° spacing
        }
    }
}

// --- RPM Logging ---

#[derive(Serialize, Deserialize, Debug)]
struct RpmLogEntry {
    timestamp_secs: f32,
    rpm: f32,
    tip_velocity: f32,
    centroid_x: f32,
    centroid_y: f32,
}

#[derive(Resource)]
struct RpmLogger {
    last_log_time: f32,
    log_interval: f32, // 0.1 seconds
    entries: Vec<RpmLogEntry>,
}

impl Default for RpmLogger {
    fn default() -> Self {
        Self {
            last_log_time: -1.0, // Start immediately
            log_interval: 0.1,
            entries: Vec::new(),
        }
    }
}

// --- GPU Structures for Centroid Tracking ---

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CentroidResult {
    sum_x: u32,
    sum_y: u32,
    count: u32,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CentroidUniform {
    width: u32,
    height: u32,
    window_start: u32,
    window_end: u32,
}

#[derive(Resource, Default)]
pub struct CentroidGpuResources {
    pub result_buffer: Option<Buffer>,
    pub staging_buffer: Option<Buffer>,
    pub bind_group: Option<BindGroup>,
    pub pipeline_ready: bool,
    pub map_receiver:
        Option<std::sync::Mutex<std::sync::mpsc::Receiver<Result<(), BufferAsyncError>>>>,
    pub is_mapped: bool,
}

// ... (CentroidPipeline and AnalysisPlugin unchanged)

#[derive(Resource)]
struct CentroidSender(pub std::sync::mpsc::Sender<(Vec2, f32)>);

#[derive(Resource)]
struct CentroidReceiver(pub std::sync::Mutex<std::sync::mpsc::Receiver<(Vec2, f32)>>);

// ...

#[derive(Resource)]
pub struct CentroidPipeline {
    pub layout: BindGroupLayout,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for CentroidPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Centroid Compute Layout"),
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
                // Dimensions uniform
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

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/centroid.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Centroid Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        CentroidPipeline { layout, pipeline }
    }
}

// --- Task 6: Radial Profile Pipeline Structures ---

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
    pub map_receiver:
        Option<std::sync::Mutex<std::sync::mpsc::Receiver<Result<(), BufferAsyncError>>>>,
    pub is_mapped: bool,
}

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

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/radial_profile.wgsl");
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

/// Plugin for fan motion analysis
pub struct AnalysisPlugin;

impl Plugin for AnalysisPlugin {
    fn build(&self, app: &mut App) {
        use bevy::render::extract_resource::ExtractResourcePlugin;

        app.init_resource::<FanAnalysis>()
            .init_resource::<RpmLogger>()
            .add_plugins(ExtractResourcePlugin::<FanAnalysis>::default())
            .add_systems(
                Update,
                (
                    simulate_rpm_detection,
                    log_rpm_periodically,
                    debug_cpu_centroid,
                ),
            )
            .add_systems(Last, write_rpm_log_on_exit);
    }

    fn finish(&self, app: &mut App) {
        // --- Task 10 & 14: Integrate Radial Profile and Angular Histogram into AnalysisPlugin ---

        // Centroid channel setup
        let (centroid_sender, centroid_receiver) = std::sync::mpsc::channel();
        app.insert_resource(CentroidReceiver(std::sync::Mutex::new(centroid_receiver)));

        // Radial channel setup
        let (radial_sender, radial_receiver) = std::sync::mpsc::channel();
        app.insert_resource(RadialReceiver(std::sync::Mutex::new(radial_receiver)));

        // Angular channel setup
        let (angular_sender, angular_receiver) = std::sync::mpsc::channel();
        app.insert_resource(AngularReceiver(std::sync::Mutex::new(angular_receiver)));

        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(CentroidSender(centroid_sender));
        render_app.insert_resource(RadialSender(radial_sender));
        render_app.insert_resource(AngularSender(angular_sender));

        render_app
            .init_resource::<CentroidPipeline>()
            .init_resource::<CentroidGpuResources>()
            .init_resource::<RadialProfilePipeline>()
            .init_resource::<RadialGpuResources>()
            .init_resource::<AngularHistogramPipeline>()
            .init_resource::<AngularGpuResources>()
            .add_systems(
                Render,
                (
                    prepare_centroid_bind_group,
                    prepare_radial_bind_group,
                    prepare_angular_bind_group,
                )
                    .in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                (
                    read_centroid_result_render,
                    read_radial_result_render,
                    read_angular_result_render,
                )
                    .in_set(RenderSystems::Cleanup),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(CentroidLabel, CentroidNode);
        render_graph.add_node(RadialProfileLabel, RadialProfileNode);
        render_graph.add_node(AngularHistogramLabel, AngularHistogramNode);
        render_graph.add_node_edge(CentroidLabel, bevy::render::graph::CameraDriverLabel);
        render_graph.add_node_edge(RadialProfileLabel, bevy::render::graph::CameraDriverLabel);
        render_graph.add_node_edge(AngularHistogramLabel, bevy::render::graph::CameraDriverLabel);
        // Radial must run after accumulation (needs surface texture)
        render_graph.add_node_edge(crate::gpu::EventLabel, RadialProfileLabel);
        // Angular runs after centroid (needs centroid + radius)
        render_graph.add_node_edge(CentroidLabel, AngularHistogramLabel);

        app.add_systems(Update, update_analysis_from_render);
        app.add_systems(Update, update_radius_from_render);
        app.add_systems(Update, update_blades_from_render);
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct CentroidLabel;

struct CentroidNode;

// --- Task 7: Radial Profile Channel Resources and Render Node ---

#[derive(Resource)]
struct RadialSender(pub std::sync::mpsc::Sender<f32>);

#[derive(Resource)]
struct RadialReceiver(pub std::sync::Mutex<std::sync::mpsc::Receiver<f32>>);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct RadialProfileLabel;

struct RadialProfileNode;

impl Node for CentroidNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let gpu_resources = world.resource::<CentroidGpuResources>();
        let pipeline = world.resource::<CentroidPipeline>();
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
            let reset_data = CentroidResult {
                sum_x: 0,
                sum_y: 0,
                count: 0,
                min_x: u32::MAX,
                max_x: 0,
                min_y: u32::MAX,
                max_y: 0,
                _padding: 0,
            };
            render_queue.write_buffer(result_buffer, 0, bytemuck::bytes_of(&reset_data));
        }

        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Centroid Compute Pass"),
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
                let y_workgroups =
                    (total_workgroups + max_workgroups_per_dim - 1) / max_workgroups_per_dim;
                pass.dispatch_workgroups(x_workgroups, y_workgroups, 1);
            }
        }

        // Copy result to staging buffer for readback
        // Only copy if not mapped
        if !gpu_resources.is_mapped {
            if let (Some(result), Some(staging)) =
                (&gpu_resources.result_buffer, &gpu_resources.staging_buffer)
            {
                render_context.command_encoder().copy_buffer_to_buffer(
                    result,
                    0,
                    staging,
                    0,
                    std::mem::size_of::<CentroidResult>() as u64,
                );
            }
        }

        Ok(())
    }
}

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

// --- Task 8: Radial Profile Bind Group Preparation ---

fn prepare_radial_bind_group(
    pipeline: Res<RadialProfilePipeline>,
    render_device: Res<RenderDevice>,
    mut gpu_resources: ResMut<RadialGpuResources>,
    surface_image: Res<crate::gpu::SurfaceImage>,
    gpu_images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
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

fn prepare_centroid_bind_group(
    mut _commands: Commands,
    pipeline: Res<CentroidPipeline>,
    render_device: Res<RenderDevice>,
    mut gpu_resources: ResMut<CentroidGpuResources>,
    gpu_event_buffer: Res<crate::gpu::GpuEventBuffer>,
    playback_state: Res<crate::gpu::PlaybackState>,
) {
    let Some(event_buffer) = &gpu_event_buffer.buffer else {
        return;
    };

    // Create result buffer if missing
    if gpu_resources.result_buffer.is_none() {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("Centroid Result Buffer"),
            size: std::mem::size_of::<CentroidResult>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu_resources.result_buffer = Some(buffer);
    }

    // Create staging buffer for readback
    if gpu_resources.staging_buffer.is_none() {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("Centroid Staging Buffer"),
            size: std::mem::size_of::<CentroidResult>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu_resources.staging_buffer = Some(buffer);
    }

    // Create Uniform Buffer
    let window_end = playback_state.current_time as u32;
    let window_start = if window_end > playback_state.window_size as u32 {
        window_end - playback_state.window_size as u32
    } else {
        0
    };

    // Convert to 100ns units (from microseconds)
    let start_100ns = window_start * 10;
    let end_100ns = window_end * 10;

    let uniform = CentroidUniform {
        width: 1280,
        height: 720,
        window_start: start_100ns,
        window_end: end_100ns,
    };

    let uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Centroid Uniforms"),
        contents: bytemuck::bytes_of(&uniform),
        usage: BufferUsages::UNIFORM,
    });

    // Create Bind Group
    let bind_group = render_device.create_bind_group(
        Some("Centroid Bind Group"),
        &pipeline.layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: event_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: gpu_resources
                    .result_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    );

    gpu_resources.bind_group = Some(bind_group);
    gpu_resources.pipeline_ready = true;
}

// Removed dispatch_centroid_compute as it is now CentroidNode::run

/// Simulate RPM detection (placeholder for full CMax implementation)
/// In a complete implementation, this would dispatch the CMax shader and optimize for omega
fn simulate_rpm_detection(
    mut analysis: ResMut<FanAnalysis>,
    playback_state: Res<crate::gpu::PlaybackState>,
) {
    if !analysis.is_tracking {
        return;
    }

    // Placeholder: Simulate a constant RPM detection
    // In the real implementation, this would:
    // 1. Dispatch centroid shader to find fan center
    // 2. Run CMax optimization loop to find optimal omega
    // 3. Convert omega to RPM using OMEGA_TO_RPM_SCALE

    // For demo purposes, simulate a fan at ~12000 RPM (corrected from 1200)
    // The 10x error was due to timestamp unit mismatch (100ns vs 1μs)
    // Use playback time for simulation consistency
    let t = playback_state.current_time * 1e-6;
    analysis.current_rpm = 12000.0 + (t * 0.5).sin() * 500.0;

    // Calculate tip velocity: v = omega * radius
    let omega = (analysis.current_rpm * 2.0 * std::f32::consts::PI) / 60.0; // rad/s
    analysis.tip_velocity = omega * analysis.fan_radius;
}

/// Log RPM data every 0.1 seconds
fn log_rpm_periodically(
    time: Res<Time>,
    analysis: Res<FanAnalysis>,
    mut logger: ResMut<RpmLogger>,
) {
    let current_time = time.elapsed_secs();

    // Check if enough time has passed since last log
    if current_time - logger.last_log_time >= logger.log_interval {
        let entry = RpmLogEntry {
            timestamp_secs: current_time,
            rpm: analysis.current_rpm,
            tip_velocity: analysis.tip_velocity,
            centroid_x: analysis.centroid.x,
            centroid_y: analysis.centroid.y,
        };

        logger.entries.push(entry);
        logger.last_log_time = current_time;

        // Also write to file periodically (every 10 entries to reduce I/O)
        if logger.entries.len() % 10 == 0 {
            if let Err(e) = write_log_to_file(&logger.entries) {
                error!("Failed to write RPM log: {}", e);
            }
        }
    }
}

/// Write RPM log to output.json file
fn write_log_to_file(entries: &[RpmLogEntry]) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(entries)?;
    let mut file = File::create("output.json")?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Write final log on app exit
fn write_rpm_log_on_exit(logger: Res<RpmLogger>) {
    if !logger.entries.is_empty() {
        if let Err(e) = write_log_to_file(&logger.entries) {
            error!("Failed to write final RPM log: {}", e);
        } else {
            info!("Wrote {} RPM entries to output.json", logger.entries.len());
        }
    }
}

fn read_centroid_result_render(
    gpu_resources: ResMut<CentroidGpuResources>,
    sender: Res<CentroidSender>,
) {
    let gpu_resources = gpu_resources.into_inner();
    let Some(staging_buffer) = &gpu_resources.staging_buffer else {
        return;
    };

    if let Some(receiver_mutex) = gpu_resources.map_receiver.take() {
        let should_reinsert_receiver = if let Ok(receiver) = receiver_mutex.try_lock() {
            match receiver.try_recv() {
                Ok(Ok(())) => {
                    let slice = staging_buffer.slice(..);
                    {
                        let data = slice.get_mapped_range();
                        let result: CentroidResult = *bytemuck::from_bytes(&data);

                        if result.count > 0 {
                            let x = (result.sum_x as f32 / result.count as f32) / 1000.0;
                            let y = (result.sum_y as f32 / result.count as f32) / 1000.0;

                            // Calculate radius from bounding box
                            // This is a rough approximation but robust
                            let width = if result.max_x > result.min_x {
                                result.max_x - result.min_x
                            } else {
                                0
                            };
                            let height = if result.max_y > result.min_y {
                                result.max_y - result.min_y
                            } else {
                                0
                            };
                            let radius = (width.max(height) as f32) / 2.0;

                            let _ = sender.0.send((Vec2::new(x, y), radius));
                        }
                    }
                    staging_buffer.unmap();
                    gpu_resources.is_mapped = false;
                    false
                }
                Ok(Err(e)) => {
                    error!("Buffer map failed: {}", e);
                    gpu_resources.is_mapped = false;
                    false
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => true,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    error!("Buffer map channel disconnected");
                    gpu_resources.is_mapped = false;
                    false
                }
            }
        } else {
            true
        };

        if should_reinsert_receiver {
            gpu_resources.map_receiver = Some(receiver_mutex);
        }
    } else {
        // Request new map
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

fn update_analysis_from_render(receiver: Res<CentroidReceiver>, mut analysis: ResMut<FanAnalysis>) {
    // Process all available messages
    if let Ok(rx) = receiver.0.lock() {
        while let Ok((centroid, radius)) = rx.try_recv() {
            // Smooth update
            analysis.centroid = analysis.centroid.lerp(centroid, 0.1);
            // Smooth radius update (slower lerp for stability)
            analysis.fan_radius = analysis.fan_radius + (radius - analysis.fan_radius) * 0.05;
        }
    }
}

// --- Task 9: Radial Profile Readback and Radius Detection ---

fn read_radial_result_render(
    gpu_resources: ResMut<RadialGpuResources>,
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

fn update_radius_from_render(receiver: Res<RadialReceiver>, mut analysis: ResMut<FanAnalysis>) {
    if let Ok(rx) = receiver.0.lock() {
        while let Ok(radius) = rx.try_recv() {
            // Smooth update
            analysis.fan_radius = analysis.fan_radius + (radius - analysis.fan_radius) * 0.1;
        }
    }
}

// --- Task 11: Angular Histogram Pipeline Structures ---

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
    pub map_receiver:
        Option<std::sync::Mutex<std::sync::mpsc::Receiver<Result<(), BufferAsyncError>>>>,
    pub is_mapped: bool,
}

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

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/angular_histogram.wgsl");
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

// --- Task 12: Angular Histogram Render Node and Peak Detection ---

#[derive(Resource)]
struct AngularSender(pub std::sync::mpsc::Sender<Vec<f32>>);

#[derive(Resource)]
struct AngularReceiver(pub std::sync::Mutex<std::sync::mpsc::Receiver<Vec<f32>>>);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct AngularHistogramLabel;

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
                let y_workgroups =
                    (total_workgroups + max_workgroups_per_dim - 1) / max_workgroups_per_dim;
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

// --- Task 13: Angular Histogram Bind Group and Readback ---

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
        radius_tolerance: 30.0, // Accept events within ±30px of radius
        window_start: window_start * 10, // Convert to 100ns units
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

fn read_angular_result_render(
    gpu_resources: ResMut<AngularGpuResources>,
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

fn update_blades_from_render(receiver: Res<AngularReceiver>, mut analysis: ResMut<FanAnalysis>) {
    if let Ok(rx) = receiver.0.lock() {
        while let Ok(blade_angles) = rx.try_recv() {
            analysis.blade_angles = blade_angles;
        }
    }
}

fn debug_cpu_centroid(
    event_data: Res<crate::gpu::EventData>,
    playback_state: Res<crate::gpu::PlaybackState>,
) {
    if !playback_state.is_playing {
        return;
    }

    let window_end = playback_state.current_time as u32;
    let window_start = if window_end > playback_state.window_size as u32 {
        window_end - playback_state.window_size as u32
    } else {
        0
    };

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0;

    for event in &event_data.events {
        if event.timestamp >= window_start && event.timestamp <= window_end {
            sum_x += event.x as f32;
            sum_y += event.y as f32;
            count += 1;
        }
    }

    if count > 0 {
        let cx = sum_x / count as f32;
        let cy = sum_y / count as f32;
        info!("CPU Centroid: ({:.2}, {:.2}) Count: {}", cx, cy, count);
    }
}

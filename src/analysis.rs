use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_graph::{Node, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        Render, RenderApp, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

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
    _padding: u32, // For alignment
}

#[derive(Resource, Default)]
pub struct CentroidGpuResources {
    pub result_buffer: Option<Buffer>,
    pub staging_buffer: Option<Buffer>,
    pub bind_group: Option<BindGroup>,
    pub pipeline_ready: bool,
}

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

        let shader = world.resource::<AssetServer>().load("shaders/centroid.wgsl");
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

/// Plugin for fan motion analysis
pub struct AnalysisPlugin;

impl Plugin for AnalysisPlugin {
    fn build(&self, app: &mut App) {
        use bevy::render::extract_resource::ExtractResourcePlugin;

        app.init_resource::<FanAnalysis>()
            .init_resource::<RpmLogger>()
            .add_plugins(ExtractResourcePlugin::<FanAnalysis>::default())
            .add_systems(Update, (update_rotation_angle, simulate_rpm_detection, log_rpm_periodically))
            .add_systems(Last, write_rpm_log_on_exit);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<CentroidPipeline>()
            .init_resource::<CentroidGpuResources>();
            // Systems will be added later for centroid tracking
    }
}

/// Update the current rotation angle based on RPM for smooth visualization
fn update_rotation_angle(time: Res<Time>, mut analysis: ResMut<FanAnalysis>) {
    if analysis.current_rpm > 0.0 {
        // Convert RPM to radians per second
        let omega = (analysis.current_rpm * 2.0 * std::f32::consts::PI) / 60.0;
        analysis.current_angle += omega * time.delta_secs();

        // Keep angle in 0..2π range
        if analysis.current_angle > 2.0 * std::f32::consts::PI {
            analysis.current_angle -= 2.0 * std::f32::consts::PI;
        }
    }
}

/// Simulate RPM detection (placeholder for full CMax implementation)
/// In a complete implementation, this would dispatch the CMax shader and optimize for omega
fn simulate_rpm_detection(mut analysis: ResMut<FanAnalysis>, time: Res<Time>) {
    if !analysis.is_tracking {
        return;
    }

    // Placeholder: Simulate a constant RPM detection
    // In the real implementation, this would:
    // 1. Dispatch centroid shader to find fan center
    // 2. Run CMax optimization loop to find optimal omega
    // 3. Convert omega to RPM

    // For demo purposes, simulate a fan at ~1200 RPM
    analysis.current_rpm = 1200.0 + (time.elapsed_secs() * 0.5).sin() * 50.0;

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

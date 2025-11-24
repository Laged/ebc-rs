use crate::analysis::FanAnalysis;
use crate::gpu::{
    prepare_events, queue_bind_group, EventAccumulationNode, EventComputePipeline, EventData,
    EventLabel, GpuEventBuffer, PlaybackState, SurfaceImage,
};
use crate::loader::DatLoader;
use bevy::asset::RenderAssetUsages;
use bevy::{
    prelude::*,
    render::render_resource::ShaderType,
    render::{
        extract_resource::ExtractResourcePlugin, render_graph::RenderGraph, render_resource::*,
        Render, RenderApp, RenderSystems,
    },
    shader::ShaderRef,
};
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};

pub struct EventRenderPlugin;

impl Plugin for EventRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<EventData>::default())
            .add_plugins(ExtractResourcePlugin::<SurfaceImage>::default())
            .add_plugins(ExtractResourcePlugin::<PlaybackState>::default())
            .add_plugins(MaterialPlugin::<EventMaterial>::default())
            .add_plugins(EguiPlugin::default())
            .init_resource::<SurfaceImage>()
            .init_resource::<PlaybackState>()
            .add_systems(Startup, (load_data, setup_scene).chain())
            .add_systems(Update, (playback_system, update_material_time))
            .add_systems(EguiPrimaryContextPass, ui_system);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<EventComputePipeline>()
            .init_resource::<GpuEventBuffer>()
            .add_systems(Render, prepare_events.in_set(RenderSystems::Prepare))
            .add_systems(Render, queue_bind_group.in_set(RenderSystems::Queue));

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(EventLabel, EventAccumulationNode::default());
        render_graph.add_node_edge(EventLabel, bevy::render::graph::CameraDriverLabel);
    }
}

// --- Material for Visualization ---

#[derive(ShaderType, Debug, Clone, Copy)]
struct EventParams {
    width: f32,
    height: f32,
    time: f32,
    decay_tau: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct EventMaterial {
    #[uniform(0)]
    params: EventParams,
    #[texture(1, sample_type = "u_int")]
    surface_texture: Handle<Image>,
}

impl Material for EventMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/visualizer.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

fn ui_system(
    mut contexts: EguiContexts,
    mut playback_state: ResMut<PlaybackState>,
    mut fan_analysis: ResMut<FanAnalysis>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    event_data: Res<EventData>,
) {
    // Show error if no events loaded
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

    egui::Window::new("Playback Controls").show(
        contexts.ctx_mut().expect("Failed to get egui context"),
        |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button(if playback_state.is_playing {
                        "Pause"
                    } else {
                        "Play"
                    })
                    .clicked()
                {
                    playback_state.is_playing = !playback_state.is_playing;
                }
                ui.checkbox(&mut playback_state.looping, "Loop");
            });

            let max_time = playback_state.max_timestamp as f32;
            ui.add(
                egui::Slider::new(&mut playback_state.current_time, 0.0..=max_time)
                    .text("Time (us)"),
            );

            // Window: 1us to 100ms (100,000us)
            ui.add(
                egui::Slider::new(&mut playback_state.window_size, 1.0..=100000.0)
                    .text("Window (us)")
                    .logarithmic(true), // Logarithmic scale for wide range
            );

            // Speed: 0.01x to 100x
            ui.add(
                egui::Slider::new(&mut playback_state.playback_speed, 0.01..=100.0)
                    .text("Speed (x)")
                    .logarithmic(true),
            );

            ui.label(format!(
                "Time: {:.2} ms",
                playback_state.current_time / 1000.0
            ));
            ui.label(format!(
                "Window: {:.2} ms",
                playback_state.window_size / 1000.0
            ));

            // FPS Display
            if let Some(fps) = diagnostics.get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS) {
                if let Some(value) = fps.smoothed() {
                    ui.label(format!("FPS: {:.1}", value));
                }
            }
        },
    );

    // Motion Analysis Panel
    egui::Window::new("Motion Analysis").show(
        contexts.ctx_mut().expect("Failed to get egui context"),
        |ui| {
            ui.checkbox(&mut fan_analysis.is_tracking, "Enable RPM Tracking");

            ui.separator();

            // RPM Display
            ui.label(format!("Detected RPM: {:.1}", fan_analysis.current_rpm));
            ui.label(format!(
                "Tip Velocity: {:.1} px/s",
                fan_analysis.tip_velocity
            ));

            ui.separator();

            // Visualization Controls
            ui.checkbox(&mut fan_analysis.show_borders, "Show Blade Borders");

            ui.add(
                egui::Slider::new(&mut fan_analysis.blade_count, 2..=8)
                    .text("Blade Count"),
            );

            ui.add(
                egui::Slider::new(&mut fan_analysis.fan_radius, 50.0..=400.0)
                    .text("Fan Radius (px)"),
            );

            ui.separator();

            // Debug Info
            ui.label(format!(
                "Centroid: ({:.1}, {:.1})",
                fan_analysis.centroid.x, fan_analysis.centroid.y
            ));
            ui.label(format!(
                "Angle: {:.1}°",
                fan_analysis.current_angle.to_degrees()
            ));
        },
    );
}

fn playback_system(time: Res<Time>, mut playback_state: ResMut<PlaybackState>) {
    if playback_state.is_playing {
        // Convert speed to microseconds per second (1x = real time)
        let delta_us = time.delta_secs() * 1_000_000.0 * playback_state.playback_speed;
        playback_state.current_time += delta_us;

        if playback_state.current_time > playback_state.max_timestamp as f32 {
            if playback_state.looping {
                playback_state.current_time = 0.0;
            } else {
                playback_state.current_time = playback_state.max_timestamp as f32;
                playback_state.is_playing = false;
            }
        }
    }
}

// Update material time from playback state
fn update_material_time(
    playback_state: Res<PlaybackState>,
    mut materials: ResMut<Assets<EventMaterial>>,
    query: Query<&MeshMaterial3d<EventMaterial>>,
) {
    for handle in query.iter() {
        if let Some(material) = materials.get_mut(&handle.0) {
            material.params.time = playback_state.current_time;
        }
    }
}

fn load_data(mut commands: Commands, mut playback_state: ResMut<PlaybackState>) {
    let path = "data/fan/fan_const_rpm.dat";
    match DatLoader::load(path) {
        Ok(events) => {
            info!("Loaded {} events from {}", events.len(), path);
            if let Some(first) = events.first() {
                if let Some(last) = events.last() {
                    let span = last.timestamp - first.timestamp;
                    info!("Timestamp range: {} to {} (span: {} units)",
                        first.timestamp, last.timestamp, span);
                    info!("If microseconds: {:.3} seconds", span as f64 / 1_000_000.0);
                    info!("If 100ns units: {:.3} seconds", span as f64 / 10_000_000.0);

                    playback_state.max_timestamp = last.timestamp;
                    playback_state.current_time = last.timestamp as f32; // Start at end
                }
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

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<EventMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut surface_image_res: ResMut<SurfaceImage>,
    event_data: Res<EventData>,
) {
    // Camera 3D
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 1000.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Create the surface image
    let width = 1280;
    let height = 720;
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Uint, // Matches the u32 buffer
        RenderAssetUsages::RENDER_WORLD,
    );
    // Important: allow it to be a copy destination
    image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;

    let image_handle = images.add(image);
    surface_image_res.handle = image_handle.clone();

    // Determine max timestamp for visualization
    // We assume events are sorted, so the last one is the latest.
    let max_timestamp = event_data.events.last().map(|e| e.timestamp).unwrap_or(0);

    info!(
        "Setting visualization time to max timestamp: {}",
        max_timestamp
    );

    // Spawn quad
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(1280.0, 720.0))),
        MeshMaterial3d(materials.add(EventMaterial {
            surface_texture: image_handle,
            params: EventParams {
                width: 1280.0,
                height: 720.0,
                time: 20000.0,      // Initial value, updated by system
                decay_tau: 50000.0, // Tau = 50ms (50000us)
            },
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

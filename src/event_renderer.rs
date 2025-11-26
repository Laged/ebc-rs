use crate::gpu::{EventData, SurfaceImage, FilteredSurfaceImage, SobelImage, CannyImage, LogImage, GroundTruthImage, EdgeParams};
use crate::playback::PlaybackState;
use crate::loader::DatLoader;
use crate::EventFilePath;
use crate::ground_truth::GroundTruthConfig;
use crate::CompareLiveMode;
use bevy::asset::RenderAssetUsages;
use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType, Extent3d, TextureDimension, TextureFormat, TextureUsages},
    shader::ShaderRef,
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy_egui::EguiPrimaryContextPass;

#[derive(ShaderType, Debug, Clone, Copy)]
struct EventParams {
    width: f32,
    height: f32,
    time: f32,
    decay_tau: f32,
    // Order must match visualizer.wgsl Params struct exactly!
    show_sobel: u32,
    show_raw: u32,
    show_canny: u32,
    show_log: u32,
    show_ground_truth: u32,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct EventMaterial {
    #[uniform(0)]
    params: EventParams,
    #[texture(1, sample_type = "u_int")]
    surface_texture: Handle<Image>,
    #[texture(2)]
    #[sampler(3)]
    sobel_texture: Handle<Image>,
    #[texture(4)]
    #[sampler(5)]
    canny_texture: Handle<Image>,
    #[texture(6)]
    #[sampler(7)]
    log_texture: Handle<Image>,
    #[texture(8)]
    #[sampler(9)]
    ground_truth_texture: Handle<Image>,
}

impl Material for EventMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/visualizer.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

#[derive(Resource)]
struct CurrentMaterialHandle(Handle<EventMaterial>);

fn load_data(
    mut commands: Commands,
    mut playback_state: ResMut<PlaybackState>,
    event_file_path: Res<EventFilePath>,
) {
    let path = &event_file_path.0;
    match DatLoader::load(path) {
        Ok(loaded_events) => {
            info!("Loaded {} events from {}", loaded_events.len(), path);

            // Convert from old crate::gpu::GpuEvent to new crate::gpu::GpuEvent
            let events: Vec<crate::gpu::GpuEvent> = loaded_events
                .iter()
                .map(|e| crate::gpu::GpuEvent {
                    timestamp: e.timestamp,
                    x: e.x,
                    y: e.y,
                    polarity: e.polarity,
                })
                .collect();

            if let Some(last) = events.last() {
                playback_state.max_timestamp = last.timestamp;
                playback_state.current_time = last.timestamp as f32;
                info!("Timestamp range: 0 to {}", last.timestamp);
            }
            commands.insert_resource(EventData { events });

            // Try to load ground truth config from sidecar JSON
            let gt_config = GroundTruthConfig::load_from_sidecar(std::path::Path::new(path))
                .unwrap_or_default();

            if gt_config.enabled {
                info!(
                    "Loaded ground truth config: {} blades, {} RPM, center ({}, {})",
                    gt_config.blade_count, gt_config.rpm, gt_config.center_x, gt_config.center_y
                );
            }
            commands.insert_resource(gt_config);
        }
        Err(e) => {
            error!("Failed to load data from {}: {:?}", path, e);
            commands.insert_resource(EventData { events: Vec::new() });
            playback_state.max_timestamp = 0;
            playback_state.current_time = 0.0;
            commands.insert_resource(GroundTruthConfig::default());
        }
    }
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<EventMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut surface_image_res: ResMut<SurfaceImage>,
    mut filtered_surface_res: ResMut<FilteredSurfaceImage>,
    mut sobel_image_res: ResMut<SobelImage>,
    mut canny_image_res: ResMut<CannyImage>,
    mut log_image_res: ResMut<LogImage>,
    mut ground_truth_image_res: ResMut<GroundTruthImage>,
    compare_live_mode: Option<Res<crate::CompareLiveMode>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 1000.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let width = 1280;
    let height = 720;
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    // Surface texture (R32Uint for timestamps)
    let mut surface_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    surface_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    let surface_handle = images.add(surface_image);
    surface_image_res.handle = surface_handle.clone();

    // Filtered surface texture (same format as surface - R32Uint)
    let mut filtered_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    filtered_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    filtered_surface_res.handle = images.add(filtered_image);

    // Sobel texture (R32Float for edge magnitude)
    let mut sobel_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    sobel_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    let sobel_handle = images.add(sobel_image);
    sobel_image_res.handle = sobel_handle.clone();

    // Canny texture (R32Float for edge magnitude)
    let mut canny_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    canny_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    let canny_handle = images.add(canny_image);
    canny_image_res.handle = canny_handle.clone();

    // LoG texture (R32Float for edge magnitude)
    let mut log_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    log_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    let log_handle = images.add(log_image);
    log_image_res.handle = log_handle.clone();

    // Create ground truth texture (RGBA8 for R=edge, G=interior)
    let mut ground_truth_image = Image::new_fill(
        Extent3d {
            width: 1280,
            height: 720,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    ground_truth_image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC;
    let ground_truth_handle = images.add(ground_truth_image);
    ground_truth_image_res.handle = ground_truth_handle.clone();

    // Material
    let material_handle = materials.add(EventMaterial {
        surface_texture: surface_handle,
        sobel_texture: sobel_handle,
        canny_texture: canny_handle,
        log_texture: log_handle,
        ground_truth_texture: ground_truth_handle,
        params: EventParams {
            width: 1280.0,
            height: 720.0,
            time: 20000.0,
            decay_tau: 50000.0,
            show_sobel: 1,
            show_canny: 0,  // Off by default
            show_log: 0,    // Off by default
            show_raw: 0,    // Off by default
            show_ground_truth: 0, // Off by default
        },
    });
    commands.insert_resource(CurrentMaterialHandle(material_handle.clone()));

    // Skip spawning the EventMaterial mesh in compare_live mode
    // The composite mesh will be used instead
    if compare_live_mode.is_none() {
        // Quad
        commands.spawn((
            Mesh3d(meshes.add(Rectangle::new(1280.0, 720.0))),
            MeshMaterial3d(material_handle),
            Transform::from_xyz(0.0, 0.0, 0.0),
        ));
    }
}

fn update_material_params(
    playback_state: Res<PlaybackState>,
    edge_params: Res<EdgeParams>,
    mut materials: ResMut<Assets<EventMaterial>>,
    current_material: Res<CurrentMaterialHandle>,
) {
    if let Some(material) = materials.get_mut(&current_material.0) {
        material.params.time = playback_state.current_time;
        material.params.show_sobel = if edge_params.show_sobel { 1 } else { 0 };
        material.params.show_canny = if edge_params.show_canny { 1 } else { 0 };
        material.params.show_log = if edge_params.show_log { 1 } else { 0 };
        material.params.show_raw = if edge_params.show_raw { 1 } else { 0 };
        material.params.show_ground_truth = if edge_params.show_ground_truth { 1 } else { 0 };
    }
}

fn ui_system(
    mut contexts: EguiContexts,
    mut playback_state: ResMut<PlaybackState>,
    mut edge_params: ResMut<EdgeParams>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    keyboard: Res<ButtonInput<KeyCode>>,
    metrics: Option<Res<crate::metrics::EdgeMetrics>>,
) {
    // Handle keyboard input for filter toggles (1/2/3/4)
    if keyboard.just_pressed(KeyCode::Digit1) {
        edge_params.filter_dead_pixels = !edge_params.filter_dead_pixels;
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        edge_params.filter_density = !edge_params.filter_density;
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        edge_params.filter_bidirectional = !edge_params.filter_bidirectional;
    }
    if keyboard.just_pressed(KeyCode::Digit4) {
        edge_params.filter_temporal = !edge_params.filter_temporal;
    }

    let ctx = contexts.ctx_mut().expect("Failed to get egui context");

    // Playback Controls
    egui::Window::new("Playback Controls").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button(if playback_state.is_playing { "Pause" } else { "Play" }).clicked() {
                playback_state.is_playing = !playback_state.is_playing;
            }
            ui.checkbox(&mut playback_state.looping, "Loop");
        });

        let max_time = playback_state.max_timestamp as f32;
        ui.add(
            egui::Slider::new(&mut playback_state.current_time, 0.0..=max_time)
                .text("Time (μs)"),
        );

        ui.add(
            egui::Slider::new(&mut playback_state.window_size, 1.0..=100_000.0)
                .text("Window (μs)")
                .logarithmic(true),
        );

        ui.add(
            egui::Slider::new(&mut playback_state.playback_speed, 0.01..=100.0)
                .text("Speed (×)")
                .logarithmic(true),
        );

        ui.label(format!("Time: {:.2} ms", playback_state.current_time / 1000.0));
        ui.label(format!("Window: {:.2} ms", playback_state.window_size / 1000.0));

        if let Some(fps) = diagnostics.get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                ui.label(format!("FPS: {:.1}", value));
            }
        }
    });

    // Edge Detection Controls
    egui::Window::new("Edge Detection").show(ctx, |ui| {
        ui.checkbox(&mut edge_params.show_raw, "Show Raw Data (Red/Blue)");
        ui.checkbox(&mut edge_params.show_sobel, "Show Sobel (Yellow)");
        ui.checkbox(&mut edge_params.show_canny, "Show Canny (Cyan)");
        ui.checkbox(&mut edge_params.show_log, "Show LoG (Magenta)");
        ui.checkbox(&mut edge_params.show_ground_truth, "Show Ground Truth (Green)");

        ui.separator();
        ui.label("Sobel Threshold:");
        ui.add(
            egui::Slider::new(&mut edge_params.sobel_threshold, 0.0..=10_000.0)
                .text("Sobel Threshold"),
        );

        ui.separator();
        ui.label("Canny Thresholds:");
        ui.add(
            egui::Slider::new(&mut edge_params.canny_low_threshold, 0.0..=5_000.0)
                .text("Canny Low"),
        );
        ui.add(
            egui::Slider::new(&mut edge_params.canny_high_threshold, 0.0..=10_000.0)
                .text("Canny High"),
        );

        ui.separator();
        ui.label("LoG Threshold:");
        ui.add(
            egui::Slider::new(&mut edge_params.log_threshold, 0.0..=10_000.0)
                .text("LoG Threshold"),
        );

        ui.separator();
        ui.label("Filters (Toggle with 1/2/3/4):");
        ui.horizontal(|ui| {
            ui.label(format!("[1] Dead Pixels: {}", if edge_params.filter_dead_pixels { "ON" } else { "OFF" }));
        });
        ui.horizontal(|ui| {
            ui.label(format!("[2] Density: {}", if edge_params.filter_density { "ON" } else { "OFF" }));
        });
        ui.horizontal(|ui| {
            ui.label(format!("[3] Bidirectional: {}", if edge_params.filter_bidirectional { "ON" } else { "OFF" }));
        });
        ui.horizontal(|ui| {
            ui.label(format!("[4] Temporal: {}", if edge_params.filter_temporal { "ON" } else { "OFF" }));
        });

        ui.separator();
        ui.label("Layers:");
        ui.label("Layer 0: Red/Blue raw events");
        ui.label("Layer 1: Yellow Sobel edges (STG)");
        ui.label("Layer 2: Cyan Canny edges");
        ui.label("Layer 3: Magenta LoG edges");
    });

    // Edge Metrics window
    egui::Window::new("Edge Metrics").show(ctx, |ui| {
        if let Some(metrics) = &metrics {
            ui.heading("Basic");
            ui.label(format!("Edge pixels: {}", metrics.edge_pixel_count));
            ui.label(format!("Density: {:.4}", metrics.edge_density));
            ui.label(format!("Centroid: ({:.1}, {:.1})", metrics.centroid.x, metrics.centroid.y));

            ui.separator();
            ui.heading("Circle Fit");
            ui.label(format!("Center: ({:.1}, {:.1})", metrics.circle_center.x, metrics.circle_center.y));
            ui.label(format!("Radius: {:.1} px", metrics.circle_radius));
            ui.label(format!("Fit error: {:.2} px", metrics.circle_fit_error));
            ui.label(format!("Inlier ratio: {:.1}%", metrics.circle_inlier_ratio * 100.0));

            ui.separator();
            ui.heading("Blade Detection");
            ui.label(format!("Detected blades: {}", metrics.detected_blade_count));
            for (i, angle) in metrics.angular_peaks.iter().enumerate() {
                ui.label(format!("  Blade {}: {:.1}°", i + 1, angle.to_degrees()));
            }
        } else {
            ui.label("No metrics available");
        }
    });
}

pub struct EventRendererPlugin;

impl Plugin for EventRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<EventMaterial>::default())
            .add_plugins(EguiPlugin::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, (load_data, setup_scene).chain())
            .add_systems(Update, crate::playback::playback_system)
            .add_systems(Update, update_material_params)
            // Skip ui_system when in CompareLiveMode (compare_live has its own UI)
            .add_systems(
                EguiPrimaryContextPass,
                ui_system.run_if(not(resource_exists::<CompareLiveMode>)),
            );
    }
}

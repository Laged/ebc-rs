use crate::mvp::gpu::{EventData, SurfaceImage, GradientImage, EdgeParams};
use crate::mvp::playback::PlaybackState;
use crate::loader::DatLoader;
use crate::EventFilePath;
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
    show_gradient: u32,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct EventMaterial {
    #[uniform(0)]
    params: EventParams,
    #[texture(1, sample_type = "u_int")]
    surface_texture: Handle<Image>,
    #[texture(2)]
    #[sampler(3)]
    gradient_texture: Handle<Image>,
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
            info!("MVP: Loaded {} events from {}", loaded_events.len(), path);

            // Convert from crate::gpu::GpuEvent to mvp::gpu::GpuEvent
            let events: Vec<crate::mvp::gpu::GpuEvent> = loaded_events
                .iter()
                .map(|e| crate::mvp::gpu::GpuEvent {
                    timestamp: e.timestamp,
                    x: e.x,
                    y: e.y,
                    polarity: e.polarity,
                })
                .collect();

            if let Some(last) = events.last() {
                playback_state.max_timestamp = last.timestamp;
                playback_state.current_time = last.timestamp as f32;
                info!("MVP: Timestamp range: 0 to {}", last.timestamp);
            }
            commands.insert_resource(EventData { events });
        }
        Err(e) => {
            error!("MVP: Failed to load data from {}: {:?}", path, e);
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
    mut gradient_image_res: ResMut<GradientImage>,
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

    // Gradient texture (R8Unorm for edge magnitude)
    let mut gradient_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0],
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    gradient_image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let gradient_handle = images.add(gradient_image);
    gradient_image_res.handle = gradient_handle.clone();

    // Material
    let material_handle = materials.add(EventMaterial {
        surface_texture: surface_handle,
        gradient_texture: gradient_handle,
        params: EventParams {
            width: 1280.0,
            height: 720.0,
            time: 20000.0,
            decay_tau: 50000.0,
            show_gradient: 1,
        },
    });
    commands.insert_resource(CurrentMaterialHandle(material_handle.clone()));

    // Quad
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(1280.0, 720.0))),
        MeshMaterial3d(material_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

fn update_material_params(
    playback_state: Res<PlaybackState>,
    edge_params: Res<EdgeParams>,
    mut materials: ResMut<Assets<EventMaterial>>,
    current_material: Res<CurrentMaterialHandle>,
) {
    if let Some(material) = materials.get_mut(&current_material.0) {
        material.params.time = playback_state.current_time;
        material.params.show_gradient = if edge_params.show_gradient { 1 } else { 0 };
    }
}

fn ui_system(
    mut contexts: EguiContexts,
    mut playback_state: ResMut<PlaybackState>,
    mut edge_params: ResMut<EdgeParams>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
) {
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
        ui.checkbox(&mut edge_params.show_gradient, "Show Edge Detection (Yellow)");

        ui.add(
            egui::Slider::new(&mut edge_params.threshold, 0.0..=10_000.0)
                .text("Edge Threshold"),
        );

        ui.label("Layer 0: Red/Blue raw events");
        ui.label("Layer 1: Yellow edge detection (Sobel STG)");
    });
}

pub struct EventRenderPlugin;

impl Plugin for EventRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<EventMaterial>::default())
            .add_plugins(EguiPlugin::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, (load_data, setup_scene).chain())
            .add_systems(Update, (
                crate::mvp::playback::playback_system,
                update_material_params,
            ).chain())
            .add_systems(EguiPrimaryContextPass, ui_system);
    }

    fn finish(&self, app: &mut App) {
        use crate::mvp::gpu::*;
        use bevy::render::render_graph::RenderGraph;
        use bevy::render::{RenderApp, Render, RenderSystems};

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<EventComputePipeline>()
            .init_resource::<GradientPipeline>()
            .init_resource::<GpuEventBuffer>()
            .add_systems(Render, prepare_events.in_set(RenderSystems::Prepare))
            .add_systems(Render, queue_bind_group.in_set(RenderSystems::Queue))
            .add_systems(Render, prepare_gradient.in_set(RenderSystems::Queue));

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(EventLabel, EventAccumulationNode::default());
        render_graph.add_node(GradientLabel, GradientNode::default());
        render_graph.add_node_edge(EventLabel, GradientLabel);
        render_graph.add_node_edge(GradientLabel, bevy::render::graph::CameraDriverLabel);
    }
}

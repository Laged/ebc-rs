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

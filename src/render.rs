use crate::gpu::GpuEvent;
use crate::loader::DatLoader;
use bevy::asset::RenderAssetUsages;
use bevy::{
    prelude::*,
    render::render_resource::ShaderType,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{Node, RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
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

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct EventLabel;

#[derive(Resource, ExtractResource, Clone, Default)]
pub struct EventData {
    pub events: Vec<GpuEvent>,
}

#[derive(Resource, ExtractResource, Clone, Default)]
pub struct SurfaceImage {
    pub handle: Handle<Image>,
}

#[derive(Resource, ExtractResource, Clone)]
struct PlaybackState {
    is_playing: bool,
    current_time: f32,   // Current playback time in microseconds
    window_size: f32,    // Integration window in microseconds
    playback_speed: f32, // Speed multiplier (real-time factor)
    looping: bool,
    max_timestamp: u32,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            is_playing: false,
            current_time: 20000.0,
            window_size: 100.0,  // 100us default
            playback_speed: 0.1, // 0.1x default
            looping: true,
            max_timestamp: 1000000, // Default placeholder, updated on load
        }
    }
}

#[derive(Resource, Default)]
struct GpuEventBuffer {
    buffer: Option<Buffer>,
    count: u32,
    surface_buffer: Option<Buffer>,
    dimensions: UVec2,
    dim_buffer: Option<Buffer>,
    uploaded: bool,
    bind_group_ready: bool,
}

#[derive(Resource)]
struct EventComputePipeline {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for EventComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("Event Compute Layout"),
            &[
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
            .load("shaders/accumulation.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Event Accumulation Pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: vec![],
            shader: shader,
            shader_defs: vec![],
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: false,
        });

        EventComputePipeline { layout, pipeline }
    }
}

#[derive(Resource)]
struct EventBindGroup(BindGroup);

fn prepare_events(
    _commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gpu_buffer: ResMut<GpuEventBuffer>,
    event_data: Res<EventData>,
    playback_state: Res<PlaybackState>,
) {
    // Update dimensions buffer with new time window EVERY FRAME if needed
    // But wait, gpu_buffer.uploaded prevents re-upload of EVENTS.
    // We need to update the UNIFORM buffer separately.

    if event_data.events.is_empty() {
        return;
    }

    let width = 1280;
    let height = 720;

    // Calculate window bounds based on playback state
    let window_end = playback_state.current_time as u32;
    let window_start = if window_end > playback_state.window_size as u32 {
        window_end - playback_state.window_size as u32
    } else {
        0
    };

    // Update dimension buffer (every frame to support dynamic window)
    if let Some(dim_buffer) = &gpu_buffer.dim_buffer {
        let dimensions = [width, height, window_start, window_end];
        render_queue.write_buffer(dim_buffer, 0, bytemuck::cast_slice(&dimensions));
    }

    if gpu_buffer.uploaded {
        return;
    }

    // Log every 100 frames or so to avoid spam, or just log once per batch if it's sporadic
    // For now, trace level is better, or info if we want to see it explicitly
    info!(
        "Uploading {} events to GPU (one-time)",
        event_data.events.len()
    );

    let byte_data: &[u8] = bytemuck::cast_slice(&event_data.events);
    gpu_buffer.buffer = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Event Buffer"),
            contents: byte_data,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        }),
    );
    gpu_buffer.count = event_data.events.len() as u32;

    let width = 1280;
    let height = 720;
    gpu_buffer.dimensions = UVec2::new(width, height);

    // Create dimension buffer (one-time creation, updated per-frame)
    // Uniform buffers must be 16-byte aligned in size.
    // dimensions.z = window_start, dimensions.w = window_end
    // Initial values will be overwritten by the per-frame update above
    let dimensions = [width, height, 0u32, 20000u32];
    gpu_buffer.dim_buffer = Some(
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dimensions),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, // Add COPY_DST for updates
        }),
    );

    let size = width * height * 4;

    gpu_buffer.surface_buffer = Some(render_device.create_buffer(&BufferDescriptor {
        label: Some("Surface Buffer"),
        size: size as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    // Initialize with 0
    render_queue.write_buffer(
        gpu_buffer.surface_buffer.as_ref().unwrap(),
        0,
        &vec![0u8; size as usize],
    );

    gpu_buffer.uploaded = true;
}

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<EventComputePipeline>,
    render_device: Res<RenderDevice>,
    mut gpu_buffer: ResMut<GpuEventBuffer>,
) {
    if gpu_buffer.bind_group_ready {
        return;
    }

    if let (Some(events), Some(surface), Some(dim_buffer)) = (
        &gpu_buffer.buffer,
        &gpu_buffer.surface_buffer,
        &gpu_buffer.dim_buffer,
    ) {
        let bind_group = render_device.create_bind_group(
            Some("Event Bind Group"),
            &pipeline.layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: events.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: surface.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dim_buffer.as_entire_binding(),
                },
            ],
        );
        commands.insert_resource(EventBindGroup(bind_group));
        gpu_buffer.bind_group_ready = true;
    }
}

#[derive(Default)]
struct EventAccumulationNode;

impl Node for EventAccumulationNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<EventComputePipeline>();
        let Some(bind_group) = world.get_resource::<EventBindGroup>() else {
            return Ok(());
        };
        let gpu_buffer = world.resource::<GpuEventBuffer>();
        let surface_image = world.resource::<SurfaceImage>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();

        if let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
            // Clear the surface buffer before accumulation
            // This ensures we only see events from the current frame/window
            if let Some(surface_buffer) = &gpu_buffer.surface_buffer {
                render_context
                    .command_encoder()
                    .clear_buffer(surface_buffer, 0, None);
            }

            {
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("Event Accumulation"),
                            timestamp_writes: None,
                        });

                pass.set_pipeline(compute_pipeline);
                pass.set_bind_group(0, &bind_group.0, &[]);

                let workgroup_size = 64;
                let count = gpu_buffer.count;
                if count > 0 {
                    let total_workgroups = (count + workgroup_size - 1) / workgroup_size;
                    let max_workgroups_per_dim = 65535;

                    let x_workgroups = if total_workgroups > max_workgroups_per_dim {
                        max_workgroups_per_dim
                    } else {
                        total_workgroups
                    };

                    let y_workgroups =
                        (total_workgroups + max_workgroups_per_dim - 1) / max_workgroups_per_dim;

                    pass.dispatch_workgroups(x_workgroups, y_workgroups, 1);
                }
            }

            // Copy buffer to texture
            if let Some(surface_buffer) = &gpu_buffer.surface_buffer {
                if let Some(gpu_image) = gpu_images.get(&surface_image.handle) {
                    render_context.command_encoder().copy_buffer_to_texture(
                        TexelCopyBufferInfo {
                            buffer: surface_buffer,
                            layout: TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(gpu_buffer.dimensions.x * 4),
                                rows_per_image: Some(gpu_buffer.dimensions.y),
                            },
                        },
                        TexelCopyTextureInfo {
                            texture: &gpu_image.texture,
                            mip_level: 0,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        Extent3d {
                            width: gpu_buffer.dimensions.x,
                            height: gpu_buffer.dimensions.y,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
        }
        Ok(())
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
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
) {
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
}

fn playback_system(time: Res<Time>, mut playback_state: ResMut<PlaybackState>) {
    if playback_state.is_playing {
        // Convert speed to microseconds per second (1x = real time)
        // But wait, events are in microseconds.
        // If speed is 1.0, we want to advance 1,000,000 microseconds per second of wall time.
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
            println!("Loaded {} events", events.len());
            if let Some(last) = events.last() {
                playback_state.max_timestamp = last.timestamp;
                playback_state.current_time = last.timestamp as f32; // Start at end
            }
            commands.insert_resource(EventData { events });
        }
        Err(e) => {
            eprintln!("Failed to load data: {:?}", e);
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

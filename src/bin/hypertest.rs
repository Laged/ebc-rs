//! Hyperparameter test runner - runs a single config and outputs JSON metrics.
//!
//! Usage:
//!   cargo run --release --bin hypertest -- --data data/fan/fan_const_rpm.dat --detector sobel --frames 30
//!
//! Output is JSON on stdout for easy parsing by hypersearch orchestrator.

use bevy::{
    asset::RenderAssetUsages,
    diagnostic::DiagnosticsPlugin,
    prelude::*,
    render::{
        extract_resource::ExtractResourcePlugin,
        render_graph::RenderGraph,
        render_resource::*,
        settings::{RenderCreation, WgpuSettings},
        Extract, ExtractSchedule, Render, RenderApp, RenderPlugin, RenderSystems,
    },
    tasks::{IoTaskPool, TaskPoolBuilder},
    window::WindowPlugin,
};
use clap::Parser;
use ebc_rs::{
    analysis::{EdgeData, EdgeDataReceiver, EdgeDataSender},
    gpu::{
        ActiveDetector, CannyImage, CannyLabel, CannyNode, CannyPipeline, EdgeParams,
        EdgeReadbackBuffer, EventComputePipeline, EventData, EventLabel, FilteredSurfaceImage,
        GpuEventBuffer, LogImage, LogLabel, LogNode, LogPipeline, PreprocessBindGroup,
        PreprocessLabel, PreprocessNode, PreprocessPipeline, ReadbackLabel, ReadbackNode,
        SobelImage, SobelLabel, SobelNode, SobelPipeline, SurfaceImage,
    },
    hyperparams::{HyperConfig, HyperResult},
    metrics::EdgeMetrics,
    playback::PlaybackState,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "hypertest")]
#[command(about = "Run a single hyperparameter configuration and output JSON metrics")]
struct Args {
    /// Path to event data file
    #[arg(long)]
    data: PathBuf,

    /// Detector to use: sobel, canny, or log
    #[arg(long, default_value = "sobel")]
    detector: String,

    /// Time window in microseconds
    #[arg(long, default_value = "100.0")]
    window_size: f32,

    /// Threshold for Sobel/LoG (or ignored for Canny)
    #[arg(long, default_value = "1000.0")]
    threshold: f32,

    /// Canny low threshold
    #[arg(long, default_value = "50.0")]
    canny_low: f32,

    /// Canny high threshold
    #[arg(long, default_value = "150.0")]
    canny_high: f32,

    /// Enable dead pixel filter
    #[arg(long)]
    filter_dead_pixels: bool,

    /// Enable density filter
    #[arg(long)]
    filter_density: bool,

    /// Enable temporal filter
    #[arg(long)]
    filter_temporal: bool,

    /// Minimum density count (for density filter)
    #[arg(long, default_value = "5")]
    min_density: u32,

    /// Minimum temporal spread in microseconds
    #[arg(long, default_value = "500.0")]
    min_temporal: f32,

    /// Enable bidirectional filter
    #[arg(long)]
    filter_bidirectional: bool,

    /// Bidirectional ratio threshold
    #[arg(long, default_value = "0.3")]
    bidirectional_ratio: f32,

    /// Number of frames to process
    #[arg(long, default_value = "30")]
    frames: usize,
}

impl Args {
    fn to_config(&self) -> HyperConfig {
        HyperConfig {
            detector: self.detector.clone(),
            window_size_us: self.window_size,
            threshold: self.threshold,
            canny_low: self.canny_low,
            canny_high: self.canny_high,
            filter_dead_pixels: self.filter_dead_pixels,
            filter_density: self.filter_density,
            filter_temporal: self.filter_temporal,
            min_density_count: self.min_density,
            min_temporal_spread_us: self.min_temporal,
            filter_bidirectional: self.filter_bidirectional,
            bidirectional_ratio: self.bidirectional_ratio,
        }
    }
}

fn main() {
    let args = Args::parse();
    let config = args.to_config();

    // Validate data file
    if !args.data.exists() {
        eprintln!("Error: Data file not found: {}", args.data.display());
        let result = HyperResult::empty(config);
        println!("{}", serde_json::to_string(&result).unwrap());
        std::process::exit(1);
    }

    // Determine detector
    let detector = match args.detector.as_str() {
        "sobel" => ActiveDetector::Sobel,
        "canny" => ActiveDetector::Canny,
        "log" => ActiveDetector::Log,
        _ => {
            eprintln!("Error: Invalid detector: {}", args.detector);
            let result = HyperResult::empty(config);
            println!("{}", serde_json::to_string(&result).unwrap());
            std::process::exit(1);
        }
    };

    // Run test
    let result = run_hypertest(&args.data, detector, config, args.frames);

    // Output JSON result
    println!("{}", serde_json::to_string(&result).unwrap());
}

fn run_hypertest(
    data_path: &std::path::Path,
    detector: ActiveDetector,
    config: HyperConfig,
    num_frames: usize,
) -> HyperResult {
    // Init task pool
    let _ = IoTaskPool::get_or_init(|| TaskPoolBuilder::default().num_threads(1).build());

    // Metrics collection
    let mut edge_counts: Vec<u32> = Vec::new();
    let mut edge_densities: Vec<f32> = Vec::new();
    let mut centroids: Vec<Vec2> = Vec::new();
    let mut circle_radii: Vec<f32> = Vec::new();
    let mut circle_fit_errors: Vec<f32> = Vec::new();
    let mut inlier_ratios: Vec<f32> = Vec::new();
    let mut blade_counts: Vec<u32> = Vec::new();

    // Setup headless Bevy app
    let mut app = App::new();

    app.add_plugins(MinimalPlugins);
    app.add_plugins(TransformPlugin::default());
    app.add_plugins(DiagnosticsPlugin::default());
    app.add_plugins(AssetPlugin::default());
    app.add_plugins(ImagePlugin::default());
    app.init_asset::<Mesh>();
    app.insert_resource(ClearColor::default());
    app.add_plugins(WindowPlugin {
        primary_window: None,
        exit_condition: bevy::window::ExitCondition::DontExit,
        ..default()
    });
    app.add_plugins(RenderPlugin {
        render_creation: RenderCreation::Automatic(WgpuSettings { ..default() }),
        ..default()
    });

    // Add test plugin
    app.add_plugins(HypertestPlugin { active_detector: detector });

    // Analysis resources in main world
    app.init_resource::<EdgeData>();
    app.init_resource::<EdgeMetrics>();

    // Load events
    let events = match ebc_rs::loader::DatLoader::load(data_path) {
        Ok(e) => e,
        Err(_) => {
            return HyperResult::empty(config);
        }
    };
    let max_timestamp = events.last().map(|e| e.timestamp).unwrap_or(0);
    app.insert_resource(EventData { events });

    // Configure playback
    app.insert_resource(PlaybackState {
        is_playing: true,
        current_time: 0.0,
        window_size: config.window_size_us,
        playback_speed: 1.0,
        looping: false,
        max_timestamp,
    });

    // Configure edge params from config
    let edge_params = config.to_edge_params();
    app.insert_resource(edge_params);

    app.finish();
    app.cleanup();

    // Run simulation
    let step_size_us = 16_666.0; // ~60 FPS
    let warmup_frames = 10;

    for i in 0..(num_frames + warmup_frames) {
        // Advance time
        {
            let mut state = app.world_mut().resource_mut::<PlaybackState>();
            state.current_time += step_size_us;
        }

        app.update();

        // Receive edge data from render world
        {
            let mut received_data = None;
            if let Some(receiver) = app.world().get_resource::<EdgeDataReceiver>() {
                if let Ok(rx) = receiver.0.try_lock() {
                    while let Ok(data) = rx.try_recv() {
                        received_data = Some(data);
                    }
                }
            }
            if let Some(data) = received_data {
                let mut edge_data = app.world_mut().resource_mut::<EdgeData>();
                *edge_data = data;
            }
        }

        // Compute and record metrics (after warmup)
        if i >= warmup_frames {
            let edge_data = app.world().resource::<EdgeData>();
            if !edge_data.pixels.is_empty() {
                let metrics =
                    EdgeMetrics::compute_basic(&edge_data.pixels, edge_data.width, edge_data.height);

                edge_counts.push(metrics.edge_pixel_count);
                edge_densities.push(metrics.edge_density);
                centroids.push(metrics.centroid);
                circle_radii.push(metrics.circle_radius);
                circle_fit_errors.push(metrics.circle_fit_error);
                inlier_ratios.push(metrics.circle_inlier_ratio);
                blade_counts.push(metrics.detected_blade_count);
            }
        }
    }

    // Compute summary statistics
    let n = edge_counts.len() as f32;
    if n == 0.0 {
        return HyperResult::empty(config);
    }

    HyperResult {
        config,
        avg_edge_count: edge_counts.iter().sum::<u32>() as f32 / n,
        edge_density: edge_densities.iter().sum::<f32>() / n,
        centroid_stability: compute_centroid_stability(&centroids),
        radius_stability: compute_std_dev(&circle_radii),
        circle_fit_error: circle_fit_errors.iter().sum::<f32>() / n,
        inlier_ratio: inlier_ratios.iter().sum::<f32>() / n,
        detected_blade_count: blade_counts.iter().sum::<u32>() as f32 / n,
        frames_processed: edge_counts.len(),
    }
}

fn compute_centroid_stability(points: &[Vec2]) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    let mean = points.iter().fold(Vec2::ZERO, |acc, p| acc + *p) / points.len() as f32;
    let variance = points
        .iter()
        .map(|p| (*p - mean).length_squared())
        .sum::<f32>()
        / points.len() as f32;
    variance.sqrt()
}

fn compute_std_dev(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

// === Test Plugin (mirrors detector_comparison but parameterized) ===

struct HypertestPlugin {
    active_detector: ActiveDetector,
}

impl Plugin for HypertestPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<EventData>::default())
            .add_plugins(ExtractResourcePlugin::<SurfaceImage>::default())
            .add_plugins(ExtractResourcePlugin::<FilteredSurfaceImage>::default())
            .add_plugins(ExtractResourcePlugin::<SobelImage>::default())
            .add_plugins(ExtractResourcePlugin::<CannyImage>::default())
            .add_plugins(ExtractResourcePlugin::<LogImage>::default())
            .add_plugins(ExtractResourcePlugin::<PlaybackState>::default())
            .init_resource::<SurfaceImage>()
            .init_resource::<FilteredSurfaceImage>()
            .init_resource::<SobelImage>()
            .init_resource::<CannyImage>()
            .init_resource::<LogImage>()
            .add_systems(Startup, setup_test_textures);
    }

    fn finish(&self, app: &mut App) {
        fn extract_edge_params(mut commands: Commands, edge_params: Extract<Res<EdgeParams>>) {
            commands.insert_resource(edge_params.clone());
        }

        // Setup channel for edge data
        let (sender, receiver) = std::sync::mpsc::channel();
        app.insert_resource(EdgeDataReceiver(std::sync::Mutex::new(receiver)));

        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(EdgeDataSender(sender));

        // Initialize render world resources
        render_app
            .init_resource::<EventComputePipeline>()
            .init_resource::<PreprocessPipeline>()
            .init_resource::<PreprocessBindGroup>()
            .init_resource::<SobelPipeline>()
            .init_resource::<CannyPipeline>()
            .init_resource::<LogPipeline>()
            .init_resource::<GpuEventBuffer>()
            .init_resource::<EdgeReadbackBuffer>()
            .add_systems(ExtractSchedule, extract_edge_params);

        // Set active detector
        let detector = self.active_detector;
        render_app.add_systems(
            Render,
            (move |mut readback: ResMut<EdgeReadbackBuffer>| {
                readback.active_detector = detector;
            })
            .in_set(RenderSystems::Prepare),
        );

        // Add render systems
        render_app
            .add_systems(
                Render,
                ebc_rs::gpu::prepare_events.in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                ebc_rs::gpu::prepare_preprocess.in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                ebc_rs::gpu::prepare_readback.in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                ebc_rs::gpu::queue_bind_group.in_set(RenderSystems::Queue),
            )
            .add_systems(
                Render,
                ebc_rs::gpu::prepare_sobel.in_set(RenderSystems::Queue),
            )
            .add_systems(
                Render,
                ebc_rs::gpu::prepare_canny.in_set(RenderSystems::Queue),
            )
            .add_systems(
                Render,
                ebc_rs::gpu::prepare_log.in_set(RenderSystems::Queue),
            )
            .add_systems(
                Render,
                ebc_rs::gpu::read_readback_result.in_set(RenderSystems::Cleanup),
            );

        // Build render graph
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(EventLabel, ebc_rs::gpu::EventAccumulationNode::default());
        render_graph.add_node(PreprocessLabel, PreprocessNode::default());
        render_graph.add_node(SobelLabel, SobelNode::default());
        render_graph.add_node(CannyLabel, CannyNode::default());
        render_graph.add_node(LogLabel, LogNode::default());
        render_graph.add_node(ReadbackLabel, ReadbackNode::default());

        // Event -> Preprocess -> Sobel -> Canny -> LoG -> Readback -> Camera
        render_graph.add_node_edge(EventLabel, PreprocessLabel);
        render_graph.add_node_edge(PreprocessLabel, SobelLabel);
        render_graph.add_node_edge(SobelLabel, CannyLabel);
        render_graph.add_node_edge(CannyLabel, LogLabel);
        render_graph.add_node_edge(LogLabel, ReadbackLabel);
        render_graph.add_node_edge(ReadbackLabel, bevy::render::graph::CameraDriverLabel);
    }
}

fn setup_test_textures(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut surface_image: ResMut<SurfaceImage>,
    mut filtered_image: ResMut<FilteredSurfaceImage>,
    mut sobel_image: ResMut<SobelImage>,
    mut canny_image: ResMut<CannyImage>,
    mut log_image: ResMut<LogImage>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 100.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let size = Extent3d {
        width: 1280,
        height: 720,
        depth_or_array_layers: 1,
    };

    // Surface texture (R32Uint)
    let mut surface = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    surface.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    surface_image.handle = images.add(surface);

    // Filtered surface texture (R32Uint)
    let mut filtered = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    filtered.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    filtered_image.handle = images.add(filtered);

    // Sobel texture (R32Float)
    let mut sobel = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    sobel.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    sobel_image.handle = images.add(sobel);

    // Canny texture (R32Float)
    let mut canny = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    canny.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    canny_image.handle = images.add(canny);

    // LoG texture (R32Float)
    let mut log = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    log.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    log_image.handle = images.add(log);
}

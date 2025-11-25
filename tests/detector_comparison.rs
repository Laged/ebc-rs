//! Detector comparison tests
//!
//! This module provides programmatic testing and comparison of the three
//! edge detection pipelines: Sobel, Canny, and LoG.

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
use ebc_rs::{
    analysis::{EdgeData, EdgeDataReceiver, EdgeDataSender},
    gpu::{
        ActiveDetector, CannyImage, CannyLabel, CannyNode, CannyPipeline, EdgeParams,
        EdgeReadbackBuffer, EventComputePipeline, EventData, EventLabel, FilteredSurfaceImage,
        GpuEventBuffer, LogImage, LogLabel, LogNode, LogPipeline, PreprocessBindGroup,
        PreprocessLabel, PreprocessNode, PreprocessPipeline, ReadbackLabel, ReadbackNode,
        SobelImage, SobelLabel, SobelNode, SobelPipeline, SurfaceImage,
    },
    metrics::EdgeMetrics,
    playback::PlaybackState,
};
use std::path::Path;

/// Results from running a detector on synthetic data
#[derive(Debug, Clone)]
pub struct DetectorResults {
    pub detector_name: String,
    pub edge_pixel_counts: Vec<u32>,
    pub edge_densities: Vec<f32>,
    pub centroids: Vec<Vec2>,
    pub circle_centers: Vec<Vec2>,
    pub circle_radii: Vec<f32>,
    pub circle_fit_errors: Vec<f32>,
    pub circle_inlier_ratios: Vec<f32>,
    pub detected_blade_counts: Vec<u32>,
    pub frame_count: usize,
}

impl DetectorResults {
    pub fn new(name: &str) -> Self {
        Self {
            detector_name: name.to_string(),
            edge_pixel_counts: Vec::new(),
            edge_densities: Vec::new(),
            centroids: Vec::new(),
            circle_centers: Vec::new(),
            circle_radii: Vec::new(),
            circle_fit_errors: Vec::new(),
            circle_inlier_ratios: Vec::new(),
            detected_blade_counts: Vec::new(),
            frame_count: 0,
        }
    }

    pub fn record(&mut self, metrics: &EdgeMetrics) {
        self.edge_pixel_counts.push(metrics.edge_pixel_count);
        self.edge_densities.push(metrics.edge_density);
        self.centroids.push(metrics.centroid);
        self.circle_centers.push(metrics.circle_center);
        self.circle_radii.push(metrics.circle_radius);
        self.circle_fit_errors.push(metrics.circle_fit_error);
        self.circle_inlier_ratios.push(metrics.circle_inlier_ratio);
        self.detected_blade_counts.push(metrics.detected_blade_count);
        self.frame_count += 1;
    }

    /// Compute summary statistics
    pub fn summary(&self) -> DetectorSummary {
        let n = self.frame_count as f32;
        if n == 0.0 {
            return DetectorSummary::default();
        }

        DetectorSummary {
            detector_name: self.detector_name.clone(),
            avg_edge_count: self.edge_pixel_counts.iter().sum::<u32>() as f32 / n,
            avg_edge_density: self.edge_densities.iter().sum::<f32>() / n,
            avg_circle_radius: self.circle_radii.iter().sum::<f32>() / n,
            avg_circle_fit_error: self.circle_fit_errors.iter().sum::<f32>() / n,
            avg_inlier_ratio: self.circle_inlier_ratios.iter().sum::<f32>() / n,
            avg_blade_count: self.detected_blade_counts.iter().sum::<u32>() as f32 / n,
            centroid_stability: compute_stability(&self.centroids),
            radius_stability: compute_std_dev(&self.circle_radii),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DetectorSummary {
    pub detector_name: String,
    pub avg_edge_count: f32,
    pub avg_edge_density: f32,
    pub avg_circle_radius: f32,
    pub avg_circle_fit_error: f32,
    pub avg_inlier_ratio: f32,
    pub avg_blade_count: f32,
    pub centroid_stability: f32, // Lower is more stable
    pub radius_stability: f32,   // Lower is more stable
}

impl std::fmt::Display for DetectorSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== {} ===", self.detector_name)?;
        writeln!(f, "  Avg Edge Count:     {:.0}", self.avg_edge_count)?;
        writeln!(f, "  Avg Edge Density:   {:.4}", self.avg_edge_density)?;
        writeln!(f, "  Avg Circle Radius:  {:.1} px", self.avg_circle_radius)?;
        writeln!(f, "  Avg Fit Error:      {:.2} px", self.avg_circle_fit_error)?;
        writeln!(f, "  Avg Inlier Ratio:   {:.1}%", self.avg_inlier_ratio * 100.0)?;
        writeln!(f, "  Avg Blade Count:    {:.1}", self.avg_blade_count)?;
        writeln!(f, "  Centroid Stability: {:.2} px (std dev)", self.centroid_stability)?;
        writeln!(f, "  Radius Stability:   {:.2} px (std dev)", self.radius_stability)?;
        Ok(())
    }
}

fn compute_stability(points: &[Vec2]) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    let mean = points.iter().fold(Vec2::ZERO, |acc, p| acc + *p) / points.len() as f32;
    let variance = points.iter().map(|p| (*p - mean).length_squared()).sum::<f32>() / points.len() as f32;
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

/// Run a single detector and collect metrics
fn run_detector(
    detector: ActiveDetector,
    data_path: &Path,
    num_frames: usize,
) -> DetectorResults {
    // Init task pool
    let _ = IoTaskPool::get_or_init(|| TaskPoolBuilder::default().num_threads(1).build());

    let detector_name = match detector {
        ActiveDetector::Sobel => "Sobel",
        ActiveDetector::Canny => "Canny",
        ActiveDetector::Log => "LoG",
    };

    let mut results = DetectorResults::new(detector_name);

    // Setup headless app
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

    // Add our GPU plugin with the selected detector
    app.add_plugins(DetectorTestPlugin { active_detector: detector });

    // Analysis resources in main world
    app.init_resource::<EdgeData>();
    app.init_resource::<EdgeMetrics>();

    // Load events
    let events = ebc_rs::loader::DatLoader::load(data_path).expect("Failed to load events");
    let max_timestamp = events.last().map(|e| e.timestamp).unwrap_or(0);
    app.insert_resource(EventData { events });

    // Configure playback
    app.insert_resource(PlaybackState {
        is_playing: true,
        current_time: 0.0,
        window_size: 50_000.0,
        playback_speed: 1.0,
        looping: false,
        max_timestamp,
    });

    // Configure edge params
    let mut edge_params = EdgeParams::default();
    edge_params.show_sobel = detector == ActiveDetector::Sobel;
    edge_params.show_canny = detector == ActiveDetector::Canny;
    edge_params.show_log = detector == ActiveDetector::Log;
    app.insert_resource(edge_params);

    app.finish();
    app.cleanup();

    // Run simulation
    let step_size_us = 16_666.0; // ~60 FPS
    let warmup_frames = 10; // Skip initial frames for GPU warmup

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

        // Compute metrics
        {
            let edge_data = app.world().resource::<EdgeData>();
            if !edge_data.pixels.is_empty() {
                let metrics = EdgeMetrics::compute_basic(&edge_data.pixels, edge_data.width, edge_data.height);

                // Only record after warmup
                if i >= warmup_frames {
                    results.record(&metrics);
                }
            }
        }
    }

    results
}

/// Compare all three detectors
pub fn compare_detectors(data_path: &Path, num_frames: usize) -> Vec<DetectorSummary> {
    let detectors = [
        ActiveDetector::Sobel,
        ActiveDetector::Canny,
        ActiveDetector::Log,
    ];

    let mut summaries = Vec::new();

    for detector in detectors {
        println!("Running {:?} detector...", detector);
        let results = run_detector(detector, data_path, num_frames);
        let summary = results.summary();
        println!("{}", summary);
        summaries.push(summary);
    }

    summaries
}

// === Test Plugin ===

struct DetectorTestPlugin {
    active_detector: ActiveDetector,
}

impl Plugin for DetectorTestPlugin {
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

        // Active detector is set in finish() on the render world
    }

    fn finish(&self, app: &mut App) {
        // Custom extraction system for EdgeParams (same as EdgeDetectionPlugin)
        fn extract_edge_params(
            mut commands: Commands,
            edge_params: Extract<Res<EdgeParams>>,
        ) {
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
            // Add EdgeParams extraction
            .add_systems(ExtractSchedule, extract_edge_params);

        // Set active detector in readback buffer
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

        // Event → Preprocess → Sobel → Canny → LoG → Readback → Camera
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

    // Filtered surface texture (R32Uint) - output of preprocess stage
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

// === Tests ===
//
// Note: Due to Bevy's global state, we can only run one Bevy app per test process.
// Run these tests separately:
//   cargo test test_sobel_detector -- --nocapture
//   cargo test test_canny_detector -- --nocapture
//   cargo test test_log_detector -- --nocapture

fn ensure_synthetic_data() -> &'static Path {
    let data_path = Path::new("data/synthetic/detector_test.dat");
    let truth_path = Path::new("data/synthetic/detector_test_truth.json");

    if !data_path.exists() {
        ebc_rs::synthesis::generate_fan_data(data_path, truth_path)
            .expect("Failed to generate synthetic data");
    }
    data_path
}

fn real_fan_data() -> &'static Path {
    Path::new("data/fan/fan_const_rpm.dat")
}

fn verify_detector_results(summary: &DetectorSummary) {
    println!("{}", summary);

    // All detectors should detect some edges
    assert!(
        summary.avg_edge_count > 0.0,
        "{} found no edges",
        summary.detector_name
    );

    println!(
        "{}: {} avg edges detected",
        summary.detector_name, summary.avg_edge_count
    );
}

#[test]
fn test_sobel_detector() {
    let data_path = ensure_synthetic_data();

    println!("Running Sobel detector...");
    let results = run_detector(ActiveDetector::Sobel, data_path, 30);
    let summary = results.summary();

    verify_detector_results(&summary);
}

#[test]
fn test_canny_detector() {
    let data_path = ensure_synthetic_data();

    println!("Running Canny detector...");
    let results = run_detector(ActiveDetector::Canny, data_path, 30);
    let summary = results.summary();

    verify_detector_results(&summary);
}

#[test]
fn test_log_detector() {
    let data_path = ensure_synthetic_data();

    println!("Running LoG detector...");
    let results = run_detector(ActiveDetector::Log, data_path, 30);
    let summary = results.summary();

    verify_detector_results(&summary);
}

// === Real Data Tests ===
// Run with: cargo test test_sobel_real -- --nocapture

#[test]
fn test_sobel_real() {
    let data_path = real_fan_data();
    if !data_path.exists() {
        println!("Skipping: {} not found", data_path.display());
        return;
    }

    println!("Running Sobel detector on REAL data...");
    let results = run_detector(ActiveDetector::Sobel, data_path, 50);
    let summary = results.summary();

    verify_detector_results(&summary);
}

#[test]
fn test_canny_real() {
    let data_path = real_fan_data();
    if !data_path.exists() {
        println!("Skipping: {} not found", data_path.display());
        return;
    }

    println!("Running Canny detector on REAL data...");
    let results = run_detector(ActiveDetector::Canny, data_path, 50);
    let summary = results.summary();

    verify_detector_results(&summary);
}

#[test]
fn test_log_real() {
    let data_path = real_fan_data();
    if !data_path.exists() {
        println!("Skipping: {} not found", data_path.display());
        return;
    }

    println!("Running LoG detector on REAL data...");
    let results = run_detector(ActiveDetector::Log, data_path, 50);
    let summary = results.summary();

    verify_detector_results(&summary);
}

#[test]
#[ignore] // Run with: cargo test test_detector_comparison_synthetic -- --ignored --nocapture --test-threads=1
fn test_detector_comparison_synthetic() {
    // Generate synthetic data
    let data_path = Path::new("data/synthetic/detector_test.dat");
    let truth_path = Path::new("data/synthetic/detector_test_truth.json");

    ebc_rs::synthesis::generate_fan_data(data_path, truth_path)
        .expect("Failed to generate synthetic data");

    // Note: This test requires --test-threads=1 and may fail due to Bevy global state
    // Compare detectors (reduced frames for faster testing)
    let summaries = compare_detectors(data_path, 30);

    // Basic assertions
    assert_eq!(summaries.len(), 3);

    for summary in &summaries {
        // All detectors should detect some edges
        assert!(summary.avg_edge_count > 0.0, "{} found no edges", summary.detector_name);

        // Circle radius should be roughly correct (200px in synthetic data)
        // Allow wide tolerance since different detectors behave differently
        assert!(
            summary.avg_circle_radius > 50.0 && summary.avg_circle_radius < 400.0,
            "{} radius {} is way off",
            summary.detector_name,
            summary.avg_circle_radius
        );
    }

    // Print comparison table
    println!("\n=== DETECTOR COMPARISON ===");
    println!("{:<10} {:>12} {:>12} {:>12} {:>12}",
             "Detector", "Edge Count", "Fit Error", "Inlier %", "Blades");
    println!("{}", "-".repeat(60));
    for s in &summaries {
        println!("{:<10} {:>12.0} {:>12.2} {:>12.1} {:>12.1}",
                 s.detector_name, s.avg_edge_count, s.avg_circle_fit_error,
                 s.avg_inlier_ratio * 100.0, s.avg_blade_count);
    }
}

#[test]
#[ignore] // Run with --ignored flag for longer test
fn test_detector_stability() {
    let data_path = Path::new("data/synthetic/stability_test.dat");
    let truth_path = Path::new("data/synthetic/stability_test_truth.json");

    ebc_rs::synthesis::generate_fan_data(data_path, truth_path)
        .expect("Failed to generate synthetic data");

    // Run each detector for more frames to measure stability
    let summaries = compare_detectors(data_path, 100);

    println!("\n=== STABILITY COMPARISON ===");
    println!("{:<10} {:>15} {:>15}",
             "Detector", "Centroid Std", "Radius Std");
    println!("{}", "-".repeat(45));
    for s in &summaries {
        println!("{:<10} {:>15.2} {:>15.2}",
                 s.detector_name, s.centroid_stability, s.radius_stability);
    }

    // Most stable detector should have low std dev
    let best_centroid = summaries.iter()
        .min_by(|a, b| a.centroid_stability.partial_cmp(&b.centroid_stability).unwrap())
        .unwrap();
    println!("\nMost stable centroid: {}", best_centroid.detector_name);
}

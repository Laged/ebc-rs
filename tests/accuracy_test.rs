use bevy::{
    diagnostic::DiagnosticsPlugin,
    prelude::*,
    render::{
        extract_resource::ExtractResourcePlugin,
        render_graph::RenderGraph,
        render_resource::*,
        settings::{RenderCreation, WgpuSettings},
        Render, RenderApp, RenderPlugin, RenderSystems,
    },
    tasks::{IoTaskPool, TaskPoolBuilder},
    window::WindowPlugin,
};
use ebc_rs::{AnalysisPlugin, FanAnalysis};
use std::path::Path;
use std::f32::consts::PI;

#[derive(Debug, serde::Deserialize)]
struct GroundTruthEntry {
    time: f32, // in seconds
    angle: f32, // in radians, base angle of blade 0
    rpm: f32,
    centroid_x: f32,
    centroid_y: f32,
    radius: f32, // Added radius
}

#[test]
fn test_fan_convergence_headless() {
    // Init task pool manually to ensure it exists for RenderPlugin
    let _ = IoTaskPool::get_or_init(|| TaskPoolBuilder::default().num_threads(1).build());

    // 1. Generate Data
    let data_path = Path::new("data/synthetic/fan_rpm1200.dat");
    let truth_path = Path::new("data/synthetic/fan_rpm1200_truth.json");
    
    ebc_rs::synthesis::generate_fan_data(data_path, truth_path)
        .expect("Failed to generate synthetic data");
    
    assert!(data_path.exists(), "Data file not found");

    // 2. Setup Headless App
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
        render_creation: RenderCreation::Automatic(WgpuSettings {
             ..default()
        }),
        ..default()
    });
    
    app.add_plugins(TestGpuPlugin); 
    app.add_plugins(AnalysisPlugin);
    
    // Manually load data
    let events = ebc_rs::loader::DatLoader::load(data_path).expect("Failed to load events");
    app.insert_resource(ebc_rs::gpu::EventData { events });
    
    // Set PlaybackState
    app.insert_resource(ebc_rs::gpu::PlaybackState {
        is_playing: true,
        current_time: 0.0, 
        window_size: 50_000.0,
        playback_speed: 1.0,
        looping: false,
        max_timestamp: 2_000_000,
    });
    
    app.finish();
    app.cleanup();
    
    // Run loop
    let mut converged = false;
    let step_size_us = 16_666.0; // ~60 FPS
    let mut stable_frames = 0;
    
    for i in 0..120 {
        let mut state = app.world_mut().resource_mut::<ebc_rs::gpu::PlaybackState>();
        state.current_time += step_size_us;
        
        app.update();
        
        let analysis = app.world().resource::<FanAnalysis>();
        
        let dist = analysis.centroid.distance(Vec2::new(640.0, 360.0));
        let radius_diff = (analysis.fan_radius - 200.0).abs();
        
        if i > 60 {
            if dist < 20.0 && radius_diff < 30.0 {
                stable_frames += 1;
            }
        }
        println!("Frame {:3}: Centroid Dist: {:.2}, Radius Diff: {:.2}, Blades: {}", 
                 i, dist, radius_diff, analysis.blade_angles.len());
    }
    
    if stable_frames >= 30 {
        converged = true;
    }
    
    assert!(converged, "Tracker failed to converge. Stable frames: {}", stable_frames);
}

#[test]
fn test_fan_accuracy_detailed_metrics_headless() {
    // Init task pool manually
    let _ = IoTaskPool::get_or_init(|| TaskPoolBuilder::default().num_threads(1).build());

    // 1. Generate Data
    let data_path = Path::new("data/synthetic/fan_rpm1200.dat");
    let truth_path = Path::new("data/synthetic/fan_rpm1200_truth.json");
    
    ebc_rs::synthesis::generate_fan_data(data_path, truth_path)
        .expect("Failed to generate synthetic data");
    
    assert!(data_path.exists(), "Data file not found");

    // Load ground truth
    let truth_file_content = std::fs::read_to_string(truth_path)
        .expect("Failed to read truth file");
    let ground_truth: Vec<GroundTruthEntry> = serde_json::from_str(&truth_file_content)
        .expect("Failed to parse ground truth JSON");
    
    // Sort ground truth by time for efficient lookup
    let mut ground_truth_sorted = ground_truth;
    ground_truth_sorted.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));

    // 2. Setup Headless App
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
    
    app.add_plugins(TestGpuPlugin); 
    app.add_plugins(AnalysisPlugin);
    
    // Manually load data
    let events = ebc_rs::loader::DatLoader::load(data_path).expect("Failed to load events");
    app.insert_resource(ebc_rs::gpu::EventData { events });
    
    // Set PlaybackState
    app.insert_resource(ebc_rs::gpu::PlaybackState {
        is_playing: true,
        current_time: 0.0, 
        window_size: 50_000.0,
        playback_speed: 1.0,
        looping: false,
        max_timestamp: 2_000_000,
    });
    
    app.finish();
    app.cleanup();

    // 3. Run loop and collect metrics
    let mut centroid_errors = Vec::new();
    let mut radius_errors = Vec::new();
    let mut blade_angle_errors = Vec::new();

    let simulation_duration_secs = 2.0;
    let step_size_us = 16_666.0; // ~60 FPS
    let total_frames = (simulation_duration_secs * 1_000_000.0 / step_size_us) as usize;

    for i in 0..total_frames {
        let mut state = app.world_mut().resource_mut::<ebc_rs::gpu::PlaybackState>();
        state.current_time += step_size_us;
        
        app.update();
        
        let analysis = app.world().resource::<FanAnalysis>();
        let playback_state = app.world().resource::<ebc_rs::gpu::PlaybackState>();
        let current_time_secs = playback_state.current_time / 1_000_000.0; // Current time of the events being analyzed

        // Find closest ground truth entry
        let gt_entry_option = ground_truth_sorted.iter().min_by_key(|entry| {
            (entry.time - current_time_secs).abs().to_bits()
        });

        if let Some(gt_entry) = gt_entry_option {
            // Centroid Error
            let gt_centroid = Vec2::new(gt_entry.centroid_x, gt_entry.centroid_y);
            let centroid_err = analysis.centroid.distance(gt_centroid);
            centroid_errors.push(centroid_err);

            // Radius Error
            let radius_err = (analysis.fan_radius - gt_entry.radius).abs();
            radius_errors.push(radius_err);

            // Blade Angle Error
            let mut expected_angles: Vec<f32> = (0..analysis.blade_count).map(|n| {
                (gt_entry.angle + (n as f32 * 2.0 * PI / analysis.blade_count as f32)) % (2.0 * PI)
            }).collect();
            expected_angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mut current_blade_error_sum = 0.0f32;
            let mut matched_detected_angles = vec![false; analysis.blade_angles.len()];

            for &expected_angle in &expected_angles {
                let mut min_diff = f32::MAX;
                let mut best_match_idx = None;

                for (idx, &detected_angle) in analysis.blade_angles.iter().enumerate() {
                    if !matched_detected_angles[idx] {
                        let diff = angle_diff(expected_angle, detected_angle);
                        if diff < min_diff {
                            min_diff = diff;
                            best_match_idx = Some(idx);
                        }
                    }
                }
                if let Some(idx) = best_match_idx {
                    current_blade_error_sum += min_diff;
                    matched_detected_angles[idx] = true;
                }
            }
            if !expected_angles.is_empty() && analysis.blade_angles.len() > 0 {
                 blade_angle_errors.push(current_blade_error_sum / expected_angles.len() as f32);
            } else {
                blade_angle_errors.push(0.0f32); // Not enough detected blades or mismatch in count
            }
        }
    }

    // 4. Report Metrics
    let avg_centroid_err = centroid_errors.iter().sum::<f32>() / centroid_errors.len() as f32;
    let max_centroid_err: f32 = centroid_errors.iter().fold(0.0f32, |acc, x| acc.max(*x));
    
    let avg_radius_err = radius_errors.iter().sum::<f32>() / radius_errors.len() as f32;
    let max_radius_err: f32 = radius_errors.iter().fold(0.0f32, |acc, x| acc.max(*x));

    let avg_blade_angle_err = blade_angle_errors.iter().sum::<f32>() / blade_angle_errors.len() as f32;
    let max_blade_angle_err: f32 = blade_angle_errors.iter().fold(0.0f32, |acc, x| acc.max(*x));

    println!("\n--- Detailed Accuracy Metrics ---");
    println!("Avg Centroid Error: {:.2} px", avg_centroid_err);
    println!("Max Centroid Error: {:.2} px", max_centroid_err);
    println!("Avg Radius Error: {:.2} px", avg_radius_err);
    println!("Max Radius Error: {:.2} px", max_radius_err);
    println!("Avg Blade Angle Error: {:.2} rad ({:.2} deg)", avg_blade_angle_err, avg_blade_angle_err.to_degrees());
    println!("Max Blade Angle Error: {:.2} rad ({:.2} deg)", max_blade_angle_err, max_blade_angle_err.to_degrees());

    // Assertions (example tolerances)
    assert!(avg_centroid_err < 10.0, "Average centroid error too high");
    assert!(max_centroid_err < 30.0, "Max centroid error too high");
    assert!(avg_radius_err < 10.0, "Average radius error too high");
    assert!(max_radius_err < 30.0, "Max radius error too high");
    assert!(avg_blade_angle_err < 0.5, "Average blade angle error too high"); // ~28 degrees
    assert!(max_blade_angle_err < 1.0, "Max blade angle error too high"); // ~57 degrees
}

// Helper function for angular difference handling wrap-around
fn angle_diff(a: f32, b: f32) -> f32 {
    let mut diff = (b - a).abs();
    if diff > PI {
        diff = 2.0 * PI - diff;
    }
    diff
}

struct TestGpuPlugin;

impl Plugin for TestGpuPlugin {
    fn build(&self, app: &mut App) {
        use ebc_rs::gpu::*;
        
        app.add_plugins(ExtractResourcePlugin::<EventData>::default())
           .add_plugins(ExtractResourcePlugin::<SurfaceImage>::default())
           .add_plugins(ExtractResourcePlugin::<PlaybackState>::default())
           .init_resource::<SurfaceImage>() 
           .add_systems(Startup, setup_dummy_camera);
    }
    
    fn finish(&self, app: &mut App) {
        use ebc_rs::gpu::*;
        
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

fn setup_dummy_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut surface_image_res: ResMut<ebc_rs::gpu::SurfaceImage>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 100.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    
    let size = Extent3d {
        width: 1280,
        height: 720,
        depth_or_array_layers: 1,
    };
    let mut surface = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Uint, 
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    surface.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    let surface_handle = images.add(surface);
    surface_image_res.handle = surface_handle;
}

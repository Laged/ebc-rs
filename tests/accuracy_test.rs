use bevy::{
    app::AppExit,
    prelude::*,
    render::{
        settings::{RenderCreation, WgpuSettings},
        RenderPlugin,
    },
};
use ebc_rs::{AnalysisPlugin, FanAnalysis};
use std::process::Command;
use std::path::Path;

#[test]
fn test_fan_accuracy_headless() {
    // 1. Generate Data
    let gen_status = Command::new("cargo")
        .args(["run", "--bin", "generate_synthetic_fan"])
        .status()
        .expect("Failed to run data generator");
    
    assert!(gen_status.success(), "Data generation failed");
    
    let data_path = Path::new("data/synthetic/fan_rpm1200.dat");
    assert!(data_path.exists(), "Data file not found");

    // 2. Setup Headless App
    let mut app = App::new();
    
    // Use MinimalPlugins to avoid Window creation
    app.add_plugins(
        MinimalPlugins.set(bevy::app::ScheduleRunnerPlugin::run_once())
    );
    app.add_plugins(AssetPlugin::default());
    
    // Add RenderPlugin with Headless/CI settings
    // We hope the machine has a GPU or software rasterizer available
    app.add_plugins(RenderPlugin {
        render_creation: RenderCreation::Automatic(WgpuSettings {
             // Force Vulkan or defaults, usually Automatic is fine
             ..default()
        }),
        ..default()
    });
    
    // Add our analysis plugin
    app.add_plugins(AnalysisPlugin);
    
    // Add a custom plugin to load data and drive the test
    app.add_plugins(TestDriverPlugin);
    
    // 3. Run
    // Since we use ScheduleRunnerPlugin::run_once, update() runs once per call.
    // But we actually want to step through time.
    // We'll use a custom loop instead of app.run()
    
    // Actually, MinimalPlugins with ScheduleRunnerPlugin usually runs continuously until exit.
    // Let's just call app.update() manually in a loop.
    // Re-create app without ScheduleRunnerPlugin for manual control
    
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin::default());
    app.add_plugins(TypeRegistrationPlugin::default());
    app.add_plugins(FrameCountPlugin::default());
    app.add_plugins(TimePlugin::default());
    app.add_plugins(TransformPlugin::default());
    app.add_plugins(HierarchyPlugin::default());
    app.add_plugins(DiagnosticsPlugin::default());
    app.add_plugins(AssetPlugin::default());
    
    // RenderPlugin requires an event loop to be fully happy usually, 
    // but for Compute Shaders we might get away with it.
    // NOTE: On a true headless CI without GPU, this WILL fail unless software rendering is set up.
    app.add_plugins(RenderPlugin::default()); 
    
    app.add_plugins(AnalysisPlugin);
    app.add_plugins(ebc_rs::gpu::GpuPlugin); // Need to load the data
    
    // Manually load data
    // We need to insert the EventData resource
    let events = ebc_rs::loader::DatLoader::load(data_path).unwrap();
    app.insert_resource(ebc_rs::gpu::EventData { events });
    
    // Set PlaybackState
    app.insert_resource(ebc_rs::gpu::PlaybackState {
        is_playing: true,
        current_time: 0, // microseconds
        window_size: 50_000,
        speed: 1.0,
        loop_enabled: false,
        total_duration: 2_000_000, // 2s
    });
    
    // Run loop
    let mut converged = false;
    let step_size_us = 16_666; // ~60 FPS
    
    for i in 0..120 { // 2 seconds of simulation
        // Advance time manually
        let mut state = app.world_mut().resource_mut::<ebc_rs::gpu::PlaybackState>();
        state.current_time += step_size_us;
        
        // Update app
        app.update();
        
        // Check results
        let analysis = app.world().resource::<FanAnalysis>();
        
        // Centroid check (should be near 640, 360)
        let dist = analysis.centroid.distance(Vec2::new(640.0, 360.0));
        
        // Radius check (should be near 200)
        let radius_diff = (analysis.fan_radius - 200.0).abs();
        
        println!("Frame {}: Centroid Dist: {:.2}, Radius Diff: {:.2}", i, dist, radius_diff);
        
        if i > 60 && dist < 20.0 && radius_diff < 20.0 {
            converged = true;
        }
    }
    
    assert!(converged, "Tracker failed to converge on synthetic data");
}

struct TestDriverPlugin;

impl Plugin for TestDriverPlugin {
    fn build(&self, app: &mut App) {
        // Setup systems
    }
}

//! Compare Live: Side-by-side detector visualization with real-time metrics.
//!
//! Usage:
//!   cargo run --bin compare_live -- [OPTIONS] <DATA_FILES>...
//!
//! Options:
//!   --config <PATH>     Config file (default: config/detectors.toml)
//!   --window <SIZE>     Override window size for all detectors
//!   --no-gt             Disable ground truth metrics computation

use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::render::render_resource::*;
use bevy::window::WindowResolution;
use clap::Parser;
use std::path::PathBuf;

use ebc_rs::compare::{
    CompareConfig, CompareUiPlugin, CompositeImage, CompositeRenderPlugin, DataFileState,
};
use ebc_rs::edge_detection::EdgeDetectionPlugin;
use ebc_rs::EventFilePath;

#[derive(Parser, Debug)]
#[command(name = "compare_live")]
#[command(about = "Side-by-side detector comparison with real-time metrics")]
struct Args {
    /// Data files to visualize
    #[arg(required = true)]
    files: Vec<PathBuf>,

    /// Config file path
    #[arg(long, default_value = "config/detectors.toml")]
    config: PathBuf,

    /// Override window size for all detectors (microseconds)
    #[arg(long)]
    window: Option<f32>,

    /// Disable ground truth metrics computation
    #[arg(long)]
    no_gt: bool,
}

fn main() {
    let args = Args::parse();

    // Validate files exist
    let valid_files: Vec<PathBuf> = args.files.iter()
        .filter(|f| {
            if !f.exists() {
                eprintln!("Warning: File not found: {}", f.display());
                false
            } else {
                true
            }
        })
        .cloned()
        .collect();

    if valid_files.is_empty() {
        eprintln!("Error: No valid data files provided");
        std::process::exit(1);
    }

    // Load config
    let config = CompareConfig::load_with_fallback(
        if args.config.exists() { Some(&args.config) } else { None }
    );
    println!("Config loaded: {:?}", config);

    // Use first file as initial
    let first_file = valid_files[0].clone();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Compare Live - Edge Detector Comparison".to_string(),
                resolution: WindowResolution::new(2560, 1440),
                ..default()
            }),
            ..default()
        }))
        // Note: EguiPlugin is added by EdgeDetectionPlugin (via EventRendererPlugin)
        .add_plugins(EdgeDetectionPlugin)
        .add_plugins(CompositeRenderPlugin)
        .add_plugins(CompareUiPlugin)
        .insert_resource(EventFilePath(first_file.to_string_lossy().to_string()))
        .insert_resource(DataFileState {
            files: valid_files,
            current_index: 0,
        })
        .insert_resource(config)
        .add_systems(Startup, setup_composite_texture)
        .run();
}

fn setup_composite_texture(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    // Create 2560x1440 composite output texture
    let size = Extent3d {
        width: 2560,
        height: 1440,
        depth_or_array_layers: 1,
    };

    let mut composite = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    composite.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;

    let handle = images.add(composite);
    commands.insert_resource(CompositeImage { handle: handle.clone() });

    // Note: We don't spawn the sprite or camera here.
    // The existing EdgeDetectionPlugin handles rendering the event surface.
    // The composite shader will combine all detector outputs in the render graph.
}

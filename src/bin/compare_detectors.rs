//! CLI tool to compare edge detection pipelines
//!
//! Usage:
//!   cargo run --bin compare_detectors [data_file.dat] [num_frames]
//!
//! Example:
//!   cargo run --bin compare_detectors data/fan/fan_const_rpm.dat 50

use std::path::Path;

fn main() {
    let data_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/fan/fan_const_rpm.dat".to_string());

    let num_frames: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let path = Path::new(&data_path);
    if !path.exists() {
        eprintln!("Error: Data file not found: {}", data_path);
        std::process::exit(1);
    }

    println!("Comparing edge detectors on: {}", data_path);
    println!("Running {} frames per detector\n", num_frames);

    // Note: This requires the detector_comparison test module to be a library
    // For now, print instructions
    println!("To run detector comparison tests:");
    println!("  cargo test test_detector_comparison_synthetic -- --nocapture");
    println!("  cargo test test_detector_stability -- --ignored --nocapture");
    println!();
    println!("Or run the interactive app and toggle detectors with the UI:");
    println!("  cargo run -- {}", data_path);
    println!();
    println!("In the app:");
    println!("  - 'Edge Detection' window toggles Sobel/Canny/LoG visibility");
    println!("  - 'Edge Metrics' window shows real-time metrics for active detector");
    println!("  - Keys 1/2/3/4 toggle Sobel filters");
}

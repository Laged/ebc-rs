//! Grid search orchestrator - spawns hypertest subprocesses in parallel.
//!
//! Usage:
//!   cargo run --release --bin hypersearch -- \
//!     --data data/fan/fan_const_rpm.dat \
//!     --output results/coarse_search.csv \
//!     --window-sizes 10,50,100,500,1000 \
//!     --thresholds 100,500,1000,2000,5000 \
//!     --frames 30
//!
//! Output: CSV file with all results + best config JSON files per detector

use clap::Parser;
use rayon::prelude::*;
use std::path::PathBuf;
use std::process::Command;

use ebc_rs::hyperparams::{export_csv, generate_grid_configs, select_best, HyperConfig, HyperResult};

#[derive(Parser, Debug)]
#[command(name = "hypersearch")]
#[command(about = "Grid search over hyperparameters using parallel subprocess execution")]
struct Args {
    /// Path to event data file
    #[arg(long)]
    data: PathBuf,

    /// Output CSV file path
    #[arg(long)]
    output: PathBuf,

    /// Window sizes to test (comma-separated)
    #[arg(long, value_delimiter = ',', default_value = "50,100,200,500")]
    window_sizes: Vec<f32>,

    /// Thresholds to test (comma-separated)
    #[arg(long, value_delimiter = ',', default_value = "500,1000,2000")]
    thresholds: Vec<f32>,

    /// Detectors to test (comma-separated)
    #[arg(long, default_value = "sobel,canny,log")]
    detectors: String,

    /// Number of frames per config
    #[arg(long, default_value = "30")]
    frames: usize,

    /// Test all filter combinations (slower but more thorough)
    #[arg(long)]
    all_filters: bool,

    /// Maximum parallel jobs (default: number of CPUs)
    #[arg(long)]
    jobs: Option<usize>,
}

fn main() {
    let args = Args::parse();

    // Set rayon thread pool size if specified
    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .expect("Failed to set thread pool size");
    }

    // Validate data file
    if !args.data.exists() {
        eprintln!("Error: Data file not found: {}", args.data.display());
        std::process::exit(1);
    }

    // Create output directory if needed
    if let Some(parent) = args.output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).expect("Failed to create output directory");
        }
    }

    // Parse detectors
    let detectors: Vec<String> = args
        .detectors
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    // Generate config grid
    let configs = generate_grid_configs(
        &detectors,
        &args.window_sizes,
        &args.thresholds,
        args.all_filters,
    );

    println!("=== Hyperparameter Grid Search ===");
    println!("Data file: {}", args.data.display());
    println!("Output: {}", args.output.display());
    println!("Detectors: {:?}", detectors);
    println!("Window sizes: {:?}", args.window_sizes);
    println!("Thresholds: {:?}", args.thresholds);
    println!("Filter combinations: {}", if args.all_filters { "all" } else { "default only" });
    println!("Frames per config: {}", args.frames);
    println!("Total configs to test: {}", configs.len());
    println!();

    // Run configs in parallel
    println!("Starting grid search...");
    let data_path = args.data.clone();
    let frames = args.frames;

    let results: Vec<HyperResult> = configs
        .par_iter()
        .enumerate()
        .map(|(i, config)| {
            let result = run_hypertest_subprocess(&data_path, config, frames);

            // Progress indicator (may be out of order due to parallelism)
            eprintln!(
                "[{}/{}] {} window={} threshold={} -> edges={:.0} stability={:.2}",
                i + 1,
                configs.len(),
                config.detector,
                config.window_size_us,
                config.threshold,
                result.avg_edge_count,
                result.centroid_stability
            );

            result
        })
        .collect();

    // Export all results to CSV
    export_csv(&results, &args.output).expect("Failed to write CSV");
    println!("\nResults written to: {}", args.output.display());

    // Find and save best config per detector
    println!("\n=== Best Configurations ===");
    for detector in &detectors {
        let detector_results: Vec<&HyperResult> = results
            .iter()
            .filter(|r| r.config.detector == *detector)
            .collect();

        if let Some(best) = select_best(&detector_results.iter().map(|r| (*r).clone()).collect::<Vec<_>>()) {
            println!("\n{} ({})", detector.to_uppercase(), detector_results.len());
            println!("  Window size:   {:.0} us", best.config.window_size_us);
            println!("  Threshold:     {:.0}", best.config.threshold);
            println!("  Filters:       dead={} density={} temporal={}",
                best.config.filter_dead_pixels,
                best.config.filter_density,
                best.config.filter_temporal);
            println!("  Avg edges:     {:.0}", best.avg_edge_count);
            println!("  Centroid std:  {:.2} px", best.centroid_stability);
            if best.f1_score > 0.0 {
                println!("  Precision:     {:.1}%", best.precision * 100.0);
                println!("  Recall:        {:.1}%", best.recall * 100.0);
                println!("  F1 Score:      {:.1}%", best.f1_score * 100.0);
                println!("  IoU:           {:.1}%", best.iou * 100.0);
            } else {
                println!("  Inlier ratio:  {:.1}%", best.inlier_ratio * 100.0);
            }
            println!("  Score:         {:.2}", best.score());

            // Save best config to JSON
            let best_path = args.output.with_file_name(format!("best_{}.json", detector));
            std::fs::write(
                &best_path,
                serde_json::to_string_pretty(&best.config).unwrap(),
            )
            .expect("Failed to write best config");
            println!("  Saved to:      {}", best_path.display());
        }
    }

    // Overall best
    if let Some(best) = select_best(&results) {
        println!("\n=== OVERALL BEST ===");
        println!("Detector:       {}", best.config.detector);
        println!("Window size:    {:.0} us", best.config.window_size_us);
        println!("Threshold:      {:.0}", best.config.threshold);
        println!("Centroid std:   {:.2} px", best.centroid_stability);
        if best.f1_score > 0.0 {
            println!("Precision:      {:.1}%", best.precision * 100.0);
            println!("Recall:         {:.1}%", best.recall * 100.0);
            println!("F1 Score:       {:.1}%", best.f1_score * 100.0);
            println!("IoU:            {:.1}%", best.iou * 100.0);
        } else {
            println!("Inlier ratio:   {:.1}%", best.inlier_ratio * 100.0);
        }
        println!("Score:          {:.2}", best.score());
    }
}

fn run_hypertest_subprocess(
    data_path: &PathBuf,
    config: &HyperConfig,
    frames: usize,
) -> HyperResult {
    // Build command
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--release", "--bin", "hypertest", "--"])
        .arg("--data")
        .arg(data_path)
        .arg("--frames")
        .arg(frames.to_string());

    // Add config args
    cmd.args(config.to_cli_args());

    // Run subprocess
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Failed to spawn subprocess: {}", e);
            return HyperResult::empty(config.clone());
        }
    };

    if !output.status.success() {
        eprintln!(
            "Subprocess failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HyperResult::empty(config.clone());
    }

    // Parse JSON output
    let stdout = String::from_utf8_lossy(&output.stdout);
    match serde_json::from_str(&stdout) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Failed to parse subprocess output: {}", e);
            eprintln!("Output was: {}", stdout);
            HyperResult::empty(config.clone())
        }
    }
}

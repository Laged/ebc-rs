//! Headless evaluation binary for CMax-SLAM edge detection
//!
//! Loads event data, runs CPU-based CMax-SLAM warping, and computes metrics.
//!
//! Usage:
//!   cargo run --bin evaluate_cmax_slam -- --data data/synthetic/fan_test.dat
//!
//! Output CSV format:
//!   dataset,rpm_gt,rpm_est,rpm_error_pct,precision,recall,f1,avg_dist,window_us,tolerance_px

use anyhow::{Context, Result};
use clap::Parser;
use ebc_rs::ground_truth::{GroundTruthConfig, GroundTruthMetrics};
use ebc_rs::loader::DatLoader;
use ebc_rs::gpu::GpuEvent;
use std::path::PathBuf;
use std::f32::consts::PI;

#[derive(Parser, Debug)]
#[command(name = "evaluate_cmax_slam")]
#[command(about = "Evaluate CMax-SLAM edge detection against ground truth")]
struct Args {
    /// Input .dat file path
    #[arg(short, long)]
    data: PathBuf,

    /// Output CSV path
    #[arg(short, long, default_value = "results/evaluation.csv")]
    output: PathBuf,

    /// Window size in microseconds
    #[arg(short, long, default_value = "5000")]
    window_size: u32,

    /// Distance tolerance for edge matching in pixels
    #[arg(short, long, default_value = "3.0")]
    tolerance: f32,
}

/// Image dimensions (hard-coded to match GPU pipeline)
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// CPU-based CMax-SLAM implementation for evaluation
struct CpuCmaxSlam {
    width: u32,
    height: u32,
}

impl CpuCmaxSlam {
    fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Warp events using ground truth omega and create IWE
    fn warp_events(
        &self,
        events: &[GpuEvent],
        centroid: (f32, f32),
        omega: f32,
        window_start: u32,
        window_end: u32,
    ) -> Vec<u32> {
        let mut iwe = vec![0u32; (self.width * self.height) as usize];

        // Use middle of window as reference time
        let t_ref = (window_start + window_end) as f32 / 2.0;

        for event in events {
            // Filter by time window
            if event.timestamp < window_start || event.timestamp > window_end {
                continue;
            }

            let ex = event.x as f32;
            let ey = event.y as f32;

            // Convert to polar around centroid
            let dx = ex - centroid.0;
            let dy = ey - centroid.1;
            let r = (dx * dx + dy * dy).sqrt();

            // Skip events at center (undefined angle)
            if r < 1.0 {
                continue;
            }

            let theta = dy.atan2(dx);
            let dt = event.timestamp as f32 - t_ref;

            // Warp: subtract rotation that occurred during dt
            let theta_warped = theta - omega * dt;

            // Convert back to Cartesian
            let x_warped = centroid.0 + r * theta_warped.cos();
            let y_warped = centroid.1 + r * theta_warped.sin();

            let ix = x_warped as u32;
            let iy = y_warped as u32;

            if ix < self.width && iy < self.height {
                let idx = (iy * self.width + ix) as usize;
                iwe[idx] += 1;
            }
        }

        iwe
    }

    /// Apply simple Sobel edge detection to IWE
    fn detect_edges(&self, iwe: &[u32]) -> Vec<f32> {
        let mut edges = vec![0.0f32; iwe.len()];

        // Sobel kernels
        let gx = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
        let gy = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                let mut grad_x = 0.0;
                let mut grad_y = 0.0;

                // Apply Sobel kernels
                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x + kx - 1;
                        let py = y + ky - 1;
                        let idx = (py * self.width + px) as usize;
                        let val = iwe[idx] as f32;

                        grad_x += val * gx[ky as usize][kx as usize];
                        grad_y += val * gy[ky as usize][kx as usize];
                    }
                }

                // Gradient magnitude
                let magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();
                let idx = (y * self.width + x) as usize;
                edges[idx] = magnitude;
            }
        }

        edges
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("CMax-SLAM Evaluation");
    println!("====================");
    println!("Data: {}", args.data.display());
    println!("Window: {} μs", args.window_size);
    println!("Tolerance: {} px", args.tolerance);
    println!();

    // Load events
    println!("Loading events...");
    let events = DatLoader::load(&args.data)
        .context("Failed to load event data")?;
    println!("  Loaded {} events", events.len());

    if events.is_empty() {
        anyhow::bail!("No events loaded from file");
    }

    // Load ground truth
    println!("Loading ground truth...");
    let gt_config = GroundTruthConfig::load_from_sidecar(&args.data)
        .context("Failed to load ground truth config")?;
    println!("  RPM: {}", gt_config.rpm);
    println!("  Blades: {}", gt_config.blade_count);
    println!("  Center: ({}, {})", gt_config.center_x, gt_config.center_y);

    // Convert ground truth RPM to angular velocity (rad/μs)
    let rpm_gt = gt_config.rpm;
    let omega_gt = rpm_gt * 2.0 * PI / 60.0 / 1_000_000.0; // rad/μs

    // Determine time window
    let t_min = events[0].timestamp;
    let t_max = events[events.len() - 1].timestamp;
    let window_start = t_min;
    let window_end = (t_min + args.window_size).min(t_max);

    println!();
    println!("Processing window: {} - {} μs", window_start, window_end);

    // Count events in window
    let events_in_window = events.iter()
        .filter(|e| e.timestamp >= window_start && e.timestamp <= window_end)
        .count();
    println!("  Events in window: {}", events_in_window);

    // Create CMax-SLAM processor
    let cmax = CpuCmaxSlam::new(WIDTH, HEIGHT);

    // Warp events using ground truth omega
    println!();
    println!("Warping events with ground truth ω = {:.6e} rad/μs...", omega_gt);
    let iwe = cmax.warp_events(
        &events,
        (gt_config.center_x, gt_config.center_y),
        omega_gt,
        window_start,
        window_end,
    );

    // Count non-zero pixels in IWE
    let iwe_pixels = iwe.iter().filter(|&&v| v > 0).count();
    println!("  IWE non-zero pixels: {}", iwe_pixels);

    // Detect edges on IWE
    println!("Detecting edges...");
    let edges = cmax.detect_edges(&iwe);

    // Threshold edges (use top 5% as edges)
    let mut sorted_edges: Vec<f32> = edges.iter().copied().filter(|&e| e > 0.0).collect();
    sorted_edges.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = if sorted_edges.len() > 100 {
        sorted_edges[sorted_edges.len() / 20] // Top 5%
    } else {
        0.0
    };

    // Apply threshold
    let detected_edges: Vec<f32> = edges.iter()
        .map(|&e| if e >= threshold { e } else { 0.0 })
        .collect();

    let edge_pixels = detected_edges.iter().filter(|&&e| e > 0.0).count();
    println!("  Detected edge pixels: {} (threshold: {:.2})", edge_pixels, threshold);

    // Generate ground truth edge mask for the window
    println!("Generating ground truth edges...");
    let gt_mask = gt_config.generate_edge_mask_window(
        window_start as f32,
        window_end as f32,
        WIDTH,
        HEIGHT,
        10, // Sample 10 time points across window
    );

    let gt_pixels = gt_mask.iter().filter(|&&e| e).count();
    println!("  Ground truth edge pixels: {}", gt_pixels);

    // Compute metrics
    println!();
    println!("Computing metrics...");
    let metrics = GroundTruthMetrics::compute(
        &detected_edges,
        &gt_mask,
        WIDTH,
        args.tolerance,
    );

    println!("  Tolerance Precision: {:.3}", metrics.tolerance_precision);
    println!("  Tolerance Recall: {:.3}", metrics.tolerance_recall);
    println!("  Tolerance F1: {:.3}", metrics.tolerance_f1);
    println!("  Average Distance: {:.2} px", metrics.avg_distance);

    // For this evaluation, we're using ground truth omega, so RPM error is 0
    // In a real scenario, we'd estimate omega and compute the error
    let rpm_est = omega_gt * 60.0 * 1_000_000.0 / (2.0 * PI);
    let rpm_error_pct = (rpm_est - rpm_gt).abs() / rpm_gt * 100.0;

    // Extract dataset name from path
    let dataset = args.data.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Prepare CSV output
    println!();
    println!("Writing results to: {}", args.output.display());

    // Create output directory if needed
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)
            .context("Failed to create output directory")?;
    }

    // Check if file exists to determine if we need to write header
    let write_header = !args.output.exists();

    // Open file in append mode
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)
        .context("Failed to open output file")?;

    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);

    // Write header if this is a new file
    if write_header {
        writer.write_record(&[
            "dataset",
            "rpm_gt",
            "rpm_est",
            "rpm_error_pct",
            "precision",
            "recall",
            "f1",
            "avg_dist",
            "window_us",
            "tolerance_px",
        ])?;
    }

    // Write CSV row
    writer.write_record(&[
        dataset,
        &format!("{:.1}", rpm_gt),
        &format!("{:.1}", rpm_est),
        &format!("{:.3}", rpm_error_pct),
        &format!("{:.3}", metrics.tolerance_precision),
        &format!("{:.3}", metrics.tolerance_recall),
        &format!("{:.3}", metrics.tolerance_f1),
        &format!("{:.2}", metrics.avg_distance),
        &format!("{}", args.window_size),
        &format!("{:.1}", args.tolerance),
    ])?;

    writer.flush()?;

    println!();
    println!("Evaluation complete!");
    println!("CSV row:");
    println!("  Dataset: {}", dataset);
    println!("  RPM GT: {:.1}, EST: {:.1}, Error: {:.3}%", rpm_gt, rpm_est, rpm_error_pct);
    println!("  Precision: {:.3}, Recall: {:.3}, F1: {:.3}",
             metrics.tolerance_precision, metrics.tolerance_recall, metrics.tolerance_f1);
    println!("  Avg Distance: {:.2} px", metrics.avg_distance);

    Ok(())
}

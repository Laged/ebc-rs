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

    /// Count blades from motion-compensated IWE using angular histogram analysis
    fn count_blades_from_iwe(
        iwe: &[u32],
        width: u32,
        height: u32,
        centroid: (f32, f32),
        radius_min: f32,
        radius_max: f32,
    ) -> (u32, f32) {
        const NUM_BINS: usize = 360;
        let mut angular_hist = vec![0.0f32; NUM_BINS];
        let mut pixels_in_annulus = 0u32;

        // 1. Create angular histogram
        for y in 0..height {
            for x in 0..width {
                let val = iwe[(y * width + x) as usize];
                if val == 0 {
                    continue;
                }

                let dx = x as f32 - centroid.0;
                let dy = y as f32 - centroid.1;
                let r = (dx * dx + dy * dy).sqrt();

                // Only accumulate pixels within blade region
                if r < radius_min || r > radius_max {
                    continue;
                }

                pixels_in_annulus += 1;

                // Calculate angle from centroid
                let theta = dy.atan2(dx);
                // Map [-π, π] to [0, 2π]
                let angle_normalized = if theta < 0.0 { theta + 2.0 * PI } else { theta };
                // Map to bin index
                let bin = ((angle_normalized / (2.0 * PI) * NUM_BINS as f32) as usize) % NUM_BINS;
                angular_hist[bin] += val as f32;
            }
        }

        println!("    Pixels in annulus (r=[{:.0},{:.0}]): {}", radius_min, radius_max, pixels_in_annulus);

        // 2. Smooth histogram with simple moving average (5-bin window)
        let window_size = 5;
        let half_window = window_size / 2;
        let mut smoothed = vec![0.0f32; NUM_BINS];

        for i in 0..NUM_BINS {
            let mut sum = 0.0;
            for j in 0..window_size {
                let idx = (i + NUM_BINS + j - half_window) % NUM_BINS;
                sum += angular_hist[idx];
            }
            smoothed[i] = sum / window_size as f32;
        }

        // 3. Find peaks (local maxima)
        // Calculate threshold based on mean + std deviation
        let mean_val = smoothed.iter().sum::<f32>() / NUM_BINS as f32;
        let variance = smoothed.iter().map(|x| (x - mean_val).powi(2)).sum::<f32>() / NUM_BINS as f32;
        let std_dev = variance.sqrt();
        // Use mean + 1*std as threshold (more adaptive than 2x mean)
        let threshold = mean_val + std_dev;
        let min_spacing = 20; // Minimum 20 bins between peaks (~20 degrees) - blades are ~120° apart for 3 blades

        println!("    Angular hist: mean={:.1}, std={:.1}, threshold={:.1}", mean_val, std_dev, threshold);

        let mut peaks = Vec::new();
        for i in 0..NUM_BINS {
            let prev = smoothed[(i + NUM_BINS - 1) % NUM_BINS];
            let curr = smoothed[i];
            let next = smoothed[(i + 1) % NUM_BINS];

            // Check if local maximum above threshold
            if curr > threshold && curr > prev && curr > next {
                // Check minimum spacing from previous peaks
                let too_close = peaks.iter().any(|&p: &usize| {
                    let diff = (i as i32 - p as i32).abs();
                    let dist = diff.min(NUM_BINS as i32 - diff);
                    dist < min_spacing
                });

                if !too_close {
                    peaks.push(i);
                }
            }
        }

        println!("    Found {} raw peaks above threshold", peaks.len());
        if !peaks.is_empty() {
            println!("    Peak positions (degrees): {:?}", peaks.iter().map(|&p| p).collect::<Vec<_>>());
        }

        // 4. Calculate peak prominence (how distinct are the peaks)
        let peak_prominence = if peaks.is_empty() {
            0.0
        } else {
            let peak_heights: Vec<f32> = peaks.iter().map(|&i| smoothed[i]).collect();
            let avg_peak_height = peak_heights.iter().sum::<f32>() / peak_heights.len() as f32;
            // Prominence = avg peak height / mean background
            if mean_val > 0.0 {
                avg_peak_height / mean_val
            } else {
                0.0
            }
        };

        // 5. For blade counting: each blade produces 2 edge peaks (leading + trailing)
        // For 3 blades, expect 6 peaks spaced roughly evenly around 360°
        // The simplest approach: count total peaks and divide by 2
        // But we should validate that peaks are reasonably spaced
        let blade_count = if peaks.is_empty() {
            0
        } else {
            // For a fan with N blades rotating, we expect 2N edge peaks (leading + trailing per blade)
            // These should be roughly evenly distributed around the circle
            // Simply divide by 2 to get blade count
            let estimated_blades = (peaks.len() as u32 + 1) / 2; // Round up
            estimated_blades
        };

        (blade_count, peak_prominence)
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

            // Warp: SUBTRACT rotation to compensate for rotation that occurred during dt
            // If event occurred before t_ref (dt < 0), the blade was at an earlier angle,
            // so we need to advance it (subtract negative = add)
            // If event occurred after t_ref (dt > 0), the blade was at a later angle,
            // so we need to bring it back (subtract positive)
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

    // Detect edges from IWE using Sobel for real-world applicability
    println!("Detecting edges using Sobel...");
    let edges = cmax.detect_edges(&iwe);

    // Threshold edges - keep any pixel with gradient magnitude > 0
    // The IWE with correct motion compensation should have sharp edges
    let detected_edges: Vec<f32> = edges.iter()
        .map(|&e| if e > 0.5 { e } else { 0.0 })
        .collect();

    let edge_pixels = detected_edges.iter().filter(|&&e| e > 0.0).count();
    println!("  Detected edge pixels: {}", edge_pixels);

    // Count blades from IWE
    println!("Counting blades from angular histogram...");
    let (blades_detected, peak_prominence) = CpuCmaxSlam::count_blades_from_iwe(
        &iwe,
        WIDTH,
        HEIGHT,
        (gt_config.center_x, gt_config.center_y),
        gt_config.radius_min,
        gt_config.radius_max,
    );
    let blades_gt = gt_config.blade_count;
    let blades_correct = blades_detected == blades_gt;
    println!("  Ground truth blades: {}", blades_gt);
    println!("  Detected blades: {}", blades_detected);
    println!("  Correct: {}", blades_correct);
    println!("  Peak prominence: {:.2}", peak_prominence);

    // Generate ground truth edge mask at the reference time
    // For CMax-SLAM, the IWE warps all events to a single reference time (t_ref),
    // so we compare against GT edges at that same reference time.
    // This is the correct comparison because motion compensation should align
    // all edges to their position at t_ref.
    let t_ref = (window_start + window_end) as f32 / 2.0;
    println!("Generating ground truth edges at t_ref = {:.0} μs...", t_ref);
    let gt_mask = gt_config.generate_edge_mask(t_ref, WIDTH, HEIGHT);

    let gt_pixels = gt_mask.iter().filter(|&&e| e).count();
    println!("  Ground truth edge pixels: {}", gt_pixels);

    // Debug: check overlap between IWE and GT
    let mut overlap_count = 0u32;
    for (&is_gt, &iwe_val) in gt_mask.iter().zip(iwe.iter()) {
        if is_gt && iwe_val > 0 {
            overlap_count += 1;
        }
    }
    println!("  GT pixels with IWE overlap: {} / {} ({:.1}%)",
             overlap_count, gt_pixels, 100.0 * overlap_count as f32 / gt_pixels as f32);

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
            "blades_gt",
            "blades_det",
            "blades_correct",
            "peak_prominence",
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
        &format!("{}", blades_gt),
        &format!("{}", blades_detected),
        &format!("{}", blades_correct),
        &format!("{:.2}", peak_prominence),
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
    println!("  Blades GT: {}, DET: {}, Correct: {}, Prominence: {:.2}",
             blades_gt, blades_detected, blades_correct, peak_prominence);

    Ok(())
}

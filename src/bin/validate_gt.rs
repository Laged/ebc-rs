//! Validate ground truth implementation by comparing raw synthesis events
//! directly against the ground truth mask.
//!
//! If the synthesis and ground truth are consistent, raw events should
//! fall exactly on ground truth edge pixels (modulo jitter).
//!
//! Usage:
//!   cargo run --release --bin validate_gt -- --data data/synthetic/fan_test.dat

use clap::Parser;
use ebc_rs::ground_truth::{GroundTruthConfig, GroundTruthMetrics};
use ebc_rs::loader::DatLoader;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "validate_gt")]
#[command(about = "Validate ground truth by comparing raw events to GT mask")]
struct Args {
    /// Path to synthetic event data file
    #[arg(long)]
    data: PathBuf,

    /// Time window in microseconds
    #[arg(long, default_value = "50000.0")]
    window_size: f32,

    /// Number of frames to test
    #[arg(long, default_value = "30")]
    frames: usize,
}

fn main() {
    let args = Args::parse();

    // Load events
    let events = match DatLoader::load(&args.data) {
        Ok(e) => {
            println!("Loaded {} events", e.len());
            println!("First event timestamp: {} µs", e.first().map(|ev| ev.timestamp).unwrap_or(0));
            println!("Last event timestamp: {} µs", e.last().map(|ev| ev.timestamp).unwrap_or(0));
            e
        }
        Err(e) => {
            eprintln!("Failed to load events: {}", e);
            std::process::exit(1);
        }
    };

    // Load ground truth config
    let gt_config = match GroundTruthConfig::load_from_sidecar(&args.data) {
        Some(c) => {
            println!("Loaded ground truth config:");
            println!("  Center: ({}, {})", c.center_x, c.center_y);
            println!("  Radius: {} - {}", c.radius_min, c.radius_max);
            println!("  Blades: {}", c.blade_count);
            println!("  RPM: {}", c.rpm);
            println!("  Sweep k: {}", c.sweep_k);
            println!("  Width: {} - {} rad", c.width_root_rad, c.width_tip_rad);
            println!("  Edge thickness: {} px", c.edge_thickness_px);
            c
        }
        None => {
            eprintln!("No ground truth config found for: {}", args.data.display());
            eprintln!("Expected: {}_truth.json", args.data.with_extension("").display());
            std::process::exit(1);
        }
    };

    let width = 1280u32;
    let height = 720u32;
    let step_size_us = args.window_size;

    println!("\n=== Testing with window size: {} µs ===\n", args.window_size);

    let mut all_metrics: Vec<GroundTruthMetrics> = Vec::new();

    for frame in 0..args.frames {
        let current_time = (frame as f32 + 1.0) * step_size_us;

        // Find events in this time window
        let window_start = current_time - args.window_size;
        let window_end = current_time;

        let window_events: Vec<_> = events
            .iter()
            .filter(|e| e.timestamp as f32 >= window_start && (e.timestamp as f32) < window_end)
            .collect();

        if window_events.is_empty() {
            println!("Frame {}: No events in window [{}, {}]", frame, window_start, window_end);
            continue;
        }

        // Create surface from raw events (like the GPU accumulator does)
        let mut surface = vec![0.0f32; (width * height) as usize];
        for event in &window_events {
            if (event.x as u32) < width && (event.y as u32) < height {
                let idx = (event.y as u32 * width + event.x as u32) as usize;
                surface[idx] = 1.0; // Mark as edge pixel
            }
        }

        // Generate windowed ground truth mask that accounts for fan rotation
        let angular_velocity = gt_config.angular_velocity();
        let window_duration_s = args.window_size / 1_000_000.0;
        let rotation_degrees = angular_velocity * window_duration_s * 180.0 / std::f32::consts::PI;
        let num_samples = (rotation_degrees.ceil() as u32).max(10).min(360);

        let gt_mask = gt_config.generate_edge_mask_window(
            window_start,
            window_end,
            width,
            height,
            num_samples,
        );

        // Count GT edge pixels
        let gt_count = gt_mask.iter().filter(|&&x| x).count();
        let event_count = window_events.len();

        // Compute metrics
        let metrics = GroundTruthMetrics::compute(&surface, &gt_mask, width, 3.0);

        println!(
            "Frame {:2}: events={:5}, gt_edges={:5}, exact_precision={:.1}%, exact_recall={:.1}%, tol_precision={:.1}%, tol_recall={:.1}%, avg_dist={:.2}px",
            frame,
            event_count,
            gt_count,
            metrics.precision * 100.0,
            metrics.recall * 100.0,
            metrics.tolerance_precision * 100.0,
            metrics.tolerance_recall * 100.0,
            metrics.avg_distance,
        );

        all_metrics.push(metrics);
    }

    // Event-level validation: check each event against GT at its exact timestamp
    println!("\n=== Per-event validation (using exact event timestamps) ===");
    let pi = std::f32::consts::PI;
    let angular_velocity = gt_config.rpm * 2.0 * pi / 60.0;
    let blade_spacing = 2.0 * pi / gt_config.blade_count as f32;

    let mut within_threshold = 0u32;
    let mut total_checked = 0u32;
    let mut distances: Vec<f32> = Vec::new();

    for event in events.iter() {
        let t_secs = event.timestamp as f32 / 1_000_000.0;
        let x = event.x as f32;
        let y = event.y as f32;

        // Compute polar coords from event position
        let dx = x - gt_config.center_x;
        let dy = y - gt_config.center_y;
        let r = (dx * dx + dy * dy).sqrt();

        // Skip events outside blade radius range
        if r < gt_config.radius_min || r > gt_config.radius_max {
            continue;
        }

        let event_theta = dy.atan2(dx);

        // What angle should the blade be at this exact time?
        let base_angle = angular_velocity * t_secs;

        // Expected blade center angle at this radius (logarithmic spiral)
        let sweep_angle = gt_config.sweep_k * (r / gt_config.radius_min).ln();

        // Find closest blade edge
        let mut min_edge_dist_px = f32::MAX;
        for blade in 0..gt_config.blade_count {
            let blade_center = base_angle + (blade as f32 * blade_spacing) + sweep_angle;
            let mut angle_diff = event_theta - blade_center;
            while angle_diff > pi { angle_diff -= 2.0 * pi; }
            while angle_diff < -pi { angle_diff += 2.0 * pi; }

            // Blade width at this radius
            let r_norm = (r - gt_config.radius_min) / (gt_config.radius_max - gt_config.radius_min);
            let blade_width = gt_config.width_root_rad + (gt_config.width_tip_rad - gt_config.width_root_rad) * r_norm;
            let half_width = blade_width * 0.5;

            // Distance to edge in angular terms, converted to pixels
            let dist_to_leading = (angle_diff - half_width).abs() * r;
            let dist_to_trailing = (angle_diff + half_width).abs() * r;
            let edge_dist_px = dist_to_leading.min(dist_to_trailing);

            if edge_dist_px < min_edge_dist_px {
                min_edge_dist_px = edge_dist_px;
            }
        }

        distances.push(min_edge_dist_px);
        if min_edge_dist_px <= gt_config.edge_thickness_px {
            within_threshold += 1;
        }
        total_checked += 1;
    }

    // Compute statistics
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_dist = distances.iter().sum::<f32>() / distances.len() as f32;
    let median_dist = distances[distances.len() / 2];
    let p95_dist = distances[(distances.len() as f32 * 0.95) as usize];
    let max_dist = distances.last().copied().unwrap_or(0.0);

    println!("Events checked: {} (within radius range)", total_checked);
    println!("Within {} px of edge: {} ({:.1}%)",
        gt_config.edge_thickness_px,
        within_threshold,
        100.0 * within_threshold as f32 / total_checked as f32);
    println!("Distance statistics:");
    println!("  Average:  {:.2} px", avg_dist);
    println!("  Median:   {:.2} px", median_dist);
    println!("  P95:      {:.2} px", p95_dist);
    println!("  Max:      {:.2} px", max_dist);

    // Debug first few events
    println!("\nFirst 5 events:");
    for (i, event) in events.iter().take(5).enumerate() {
        let t_secs = event.timestamp as f32 / 1_000_000.0;
        let x = event.x as f32;
        let y = event.y as f32;
        let dx = x - gt_config.center_x;
        let dy = y - gt_config.center_y;
        let r = (dx * dx + dy * dy).sqrt();
        let event_theta = dy.atan2(dx);
        let base_angle = angular_velocity * t_secs;
        let sweep_angle = gt_config.sweep_k * (r / gt_config.radius_min).ln();

        let mut closest_blade = 0u32;
        let mut closest_diff = f32::MAX;
        for blade in 0..gt_config.blade_count {
            let blade_center = base_angle + (blade as f32 * blade_spacing) + sweep_angle;
            let mut angle_diff = event_theta - blade_center;
            while angle_diff > pi { angle_diff -= 2.0 * pi; }
            while angle_diff < -pi { angle_diff += 2.0 * pi; }
            if angle_diff.abs() < closest_diff.abs() {
                closest_diff = angle_diff;
                closest_blade = blade;
            }
        }

        let r_norm = (r - gt_config.radius_min) / (gt_config.radius_max - gt_config.radius_min);
        let blade_width = gt_config.width_root_rad + (gt_config.width_tip_rad - gt_config.width_root_rad) * r_norm;
        let half_width = blade_width * 0.5;
        let dist_to_leading = (closest_diff - half_width).abs() * r;
        let dist_to_trailing = (closest_diff + half_width).abs() * r;

        println!("  Event {}: t={:.6}s pos=({:.0},{:.0}) r={:.1}px blade={} edge_dist={:.2}px (lead={:.2}, trail={:.2})",
            i, t_secs, x, y, r, closest_blade, dist_to_leading.min(dist_to_trailing), dist_to_leading, dist_to_trailing);
    }

    // Summary
    if !all_metrics.is_empty() {
        let n = all_metrics.len() as f32;
        println!("\n=== Summary ===");
        println!(
            "Exact precision:     {:.1}%",
            all_metrics.iter().map(|m| m.precision).sum::<f32>() / n * 100.0
        );
        println!(
            "Exact recall:        {:.1}%",
            all_metrics.iter().map(|m| m.recall).sum::<f32>() / n * 100.0
        );
        println!(
            "Tolerance precision: {:.1}%",
            all_metrics.iter().map(|m| m.tolerance_precision).sum::<f32>() / n * 100.0
        );
        println!(
            "Tolerance recall:    {:.1}%",
            all_metrics.iter().map(|m| m.tolerance_recall).sum::<f32>() / n * 100.0
        );
        println!(
            "Average distance:    {:.2} px",
            all_metrics.iter().map(|m| m.avg_distance).sum::<f32>() / n
        );
        println!(
            "Median distance:     {:.2} px",
            all_metrics.iter().map(|m| m.median_distance).sum::<f32>() / n
        );

        // Check if we're getting near-perfect match
        let avg_tol_precision = all_metrics.iter().map(|m| m.tolerance_precision).sum::<f32>() / n;
        let avg_tol_recall = all_metrics.iter().map(|m| m.tolerance_recall).sum::<f32>() / n;

        println!("\n=== Diagnosis ===");
        if avg_tol_precision < 0.9 {
            println!("⚠ Low tolerance precision ({:.1}%): Many events are NOT near ground truth edges!", avg_tol_precision * 100.0);
            println!("  This suggests synthesis is generating events in wrong locations.");
        }
        if avg_tol_recall < 0.5 {
            println!("⚠ Low tolerance recall ({:.1}%): Many ground truth edges have NO nearby events!", avg_tol_recall * 100.0);
            println!("  This could mean:");
            println!("  - Ground truth is computing different edge positions than synthesis");
            println!("  - Window size is too small to capture enough events");
            println!("  - Time synchronization issue between events and ground truth");
        }
        if avg_tol_precision > 0.9 && avg_tol_recall > 0.9 {
            println!("✓ Synthesis and ground truth are well-aligned!");
        }
    }
}

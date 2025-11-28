//! Regression tests for CMax-SLAM evaluation metrics
//!
//! These tests ensure that changes to the CMax-SLAM pipeline don't regress
//! the baseline metrics established on the synthetic fan dataset.
//!
//! Baseline captured: 2024-11-28
//! - Precision: 0.718
//! - Recall: 0.282
//! - F1: 0.405
//! - Avg Distance: 5.52 px

use std::process::Command;
use std::str::FromStr;

/// Baseline metrics from current implementation
/// These values should only improve, never regress
mod baseline {
    pub const PRECISION_MIN: f32 = 0.65; // Allow 10% regression from 0.718
    pub const RECALL_MIN: f32 = 0.25;    // Allow 10% regression from 0.282
    pub const F1_MIN: f32 = 0.36;        // Allow 10% regression from 0.405
    pub const AVG_DIST_MAX: f32 = 6.5;   // Allow ~1px regression from 5.52
    pub const RPM_ERROR_MAX: f32 = 1.0;  // Max 1% RPM error allowed
}

#[derive(Debug)]
struct EvalMetrics {
    precision: f32,
    recall: f32,
    f1: f32,
    avg_dist: f32,
    rpm_error_pct: f32,
}

impl EvalMetrics {
    fn from_csv_line(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 8 {
            return None;
        }
        Some(Self {
            rpm_error_pct: f32::from_str(parts[3]).ok()?,
            precision: f32::from_str(parts[4]).ok()?,
            recall: f32::from_str(parts[5]).ok()?,
            f1: f32::from_str(parts[6]).ok()?,
            avg_dist: f32::from_str(parts[7]).ok()?,
        })
    }
}

fn run_evaluation() -> EvalMetrics {
    // Ensure test data exists
    let data_path = "data/synthetic/fan_test.dat";
    if !std::path::Path::new(data_path).exists() {
        // Generate test data if missing
        let gen_output = Command::new("cargo")
            .args(["run", "--release", "--bin", "generate_synthetic", "--",
                   "--rpm", "1200", "--blades", "3", "--output", data_path])
            .output()
            .expect("Failed to generate test data");

        if !gen_output.status.success() {
            panic!("Failed to generate test data: {}",
                   String::from_utf8_lossy(&gen_output.stderr));
        }
    }

    // Run evaluation
    let output = Command::new("cargo")
        .args(["run", "--release", "--bin", "evaluate_cmax_slam", "--",
               "--data", data_path,
               "--output", "/tmp/regression_test.csv"])
        .output()
        .expect("Failed to run evaluation");

    if !output.status.success() {
        panic!("Evaluation failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    // Parse results
    let csv = std::fs::read_to_string("/tmp/regression_test.csv")
        .expect("Failed to read results CSV");

    let lines: Vec<&str> = csv.lines().collect();
    if lines.len() < 2 {
        panic!("CSV has no data rows");
    }

    EvalMetrics::from_csv_line(lines[1])
        .expect("Failed to parse CSV metrics")
}

#[test]
#[ignore] // Run with: cargo test --test regression_cmax_slam -- --ignored
fn test_precision_no_regression() {
    let metrics = run_evaluation();
    assert!(
        metrics.precision >= baseline::PRECISION_MIN,
        "Precision regressed: {:.3} < {:.3} (baseline min)",
        metrics.precision, baseline::PRECISION_MIN
    );
}

#[test]
#[ignore]
fn test_recall_no_regression() {
    let metrics = run_evaluation();
    assert!(
        metrics.recall >= baseline::RECALL_MIN,
        "Recall regressed: {:.3} < {:.3} (baseline min)",
        metrics.recall, baseline::RECALL_MIN
    );
}

#[test]
#[ignore]
fn test_f1_no_regression() {
    let metrics = run_evaluation();
    assert!(
        metrics.f1 >= baseline::F1_MIN,
        "F1 score regressed: {:.3} < {:.3} (baseline min)",
        metrics.f1, baseline::F1_MIN
    );
}

#[test]
#[ignore]
fn test_avg_distance_no_regression() {
    let metrics = run_evaluation();
    assert!(
        metrics.avg_dist <= baseline::AVG_DIST_MAX,
        "Avg distance regressed: {:.2} > {:.2} (baseline max)",
        metrics.avg_dist, baseline::AVG_DIST_MAX
    );
}

#[test]
#[ignore]
fn test_rpm_accuracy() {
    let metrics = run_evaluation();
    assert!(
        metrics.rpm_error_pct <= baseline::RPM_ERROR_MAX,
        "RPM error too high: {:.3}% > {:.3}% (max allowed)",
        metrics.rpm_error_pct, baseline::RPM_ERROR_MAX
    );
}

#[test]
#[ignore]
fn test_all_metrics_no_regression() {
    let metrics = run_evaluation();

    println!("=== CMax-SLAM Regression Test Results ===");
    println!("Precision: {:.3} (min: {:.3}) {}",
             metrics.precision, baseline::PRECISION_MIN,
             if metrics.precision >= baseline::PRECISION_MIN { "✓" } else { "✗" });
    println!("Recall:    {:.3} (min: {:.3}) {}",
             metrics.recall, baseline::RECALL_MIN,
             if metrics.recall >= baseline::RECALL_MIN { "✓" } else { "✗" });
    println!("F1:        {:.3} (min: {:.3}) {}",
             metrics.f1, baseline::F1_MIN,
             if metrics.f1 >= baseline::F1_MIN { "✓" } else { "✗" });
    println!("Avg Dist:  {:.2} (max: {:.2}) {}",
             metrics.avg_dist, baseline::AVG_DIST_MAX,
             if metrics.avg_dist <= baseline::AVG_DIST_MAX { "✓" } else { "✗" });
    println!("RPM Error: {:.3}% (max: {:.3}%) {}",
             metrics.rpm_error_pct, baseline::RPM_ERROR_MAX,
             if metrics.rpm_error_pct <= baseline::RPM_ERROR_MAX { "✓" } else { "✗" });

    assert!(metrics.precision >= baseline::PRECISION_MIN, "Precision regressed");
    assert!(metrics.recall >= baseline::RECALL_MIN, "Recall regressed");
    assert!(metrics.f1 >= baseline::F1_MIN, "F1 regressed");
    assert!(metrics.avg_dist <= baseline::AVG_DIST_MAX, "Avg distance regressed");
    assert!(metrics.rpm_error_pct <= baseline::RPM_ERROR_MAX, "RPM error too high");

    println!("\n✓ All regression tests passed!");
}

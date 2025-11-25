//! Hyperparameter configuration and result types for grid search optimization.

use serde::{Deserialize, Serialize};

/// Metrics for comparing detector output against ground truth
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct EdgeMetrics {
    pub true_positives: u32,
    pub false_positives: u32,
    pub false_negatives: u32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub iou: f32,
}

impl EdgeMetrics {
    /// Compute metrics from edge counts
    pub fn compute(detector_edges: &[f32], ground_truth_edges: &[f32], threshold: f32) -> Self {
        let mut tp = 0u32;
        let mut fp = 0u32;
        let mut fn_ = 0u32;

        for (det, gt) in detector_edges.iter().zip(ground_truth_edges.iter()) {
            let det_edge = *det > threshold;
            let gt_edge = *gt > 0.5;

            match (det_edge, gt_edge) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }

        let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f32 / (tp + fn_) as f32 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        let iou = if tp + fp + fn_ > 0 { tp as f32 / (tp + fp + fn_) as f32 } else { 0.0 };

        Self { true_positives: tp, false_positives: fp, false_negatives: fn_, precision, recall, f1_score, iou }
    }
}

/// Configuration for a single hyperparameter test run.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HyperConfig {
    pub detector: String,
    pub window_size_us: f32,
    pub threshold: f32,
    pub canny_low: f32,
    pub canny_high: f32,
    pub filter_dead_pixels: bool,
    pub filter_density: bool,
    pub filter_temporal: bool,
    pub min_density_count: u32,
    pub min_temporal_spread_us: f32,
    pub filter_bidirectional: bool,
    pub bidirectional_ratio: f32,
}

impl Default for HyperConfig {
    fn default() -> Self {
        Self {
            detector: "sobel".to_string(),
            window_size_us: 100.0,
            threshold: 1000.0,
            canny_low: 50.0,
            canny_high: 150.0,
            filter_dead_pixels: true,
            filter_density: false,
            filter_temporal: false,
            min_density_count: 5,
            min_temporal_spread_us: 500.0,
            filter_bidirectional: false,
            bidirectional_ratio: 0.3,
        }
    }
}

impl HyperConfig {
    /// Convert this config to EdgeParams for use in Bevy app.
    pub fn to_edge_params(&self) -> crate::gpu::EdgeParams {
        crate::gpu::EdgeParams {
            filter_dead_pixels: self.filter_dead_pixels,
            filter_density: self.filter_density,
            filter_temporal: self.filter_temporal,
            min_density_count: self.min_density_count,
            min_temporal_spread_us: self.min_temporal_spread_us,
            sobel_threshold: self.threshold,
            threshold: self.threshold, // Backwards compat field
            canny_low_threshold: self.canny_low,
            canny_high_threshold: self.canny_high,
            log_threshold: self.threshold,
            filter_bidirectional: self.filter_bidirectional,
            bidirectional_ratio: self.bidirectional_ratio,
            show_sobel: self.detector == "sobel",
            show_canny: self.detector == "canny",
            show_log: self.detector == "log",
            show_raw: false,
            show_ground_truth: false,
        }
    }

    /// Generate CLI args for subprocess invocation.
    pub fn to_cli_args(&self) -> Vec<String> {
        let mut args = vec![
            "--detector".to_string(),
            self.detector.clone(),
            "--window-size".to_string(),
            self.window_size_us.to_string(),
            "--threshold".to_string(),
            self.threshold.to_string(),
            "--canny-low".to_string(),
            self.canny_low.to_string(),
            "--canny-high".to_string(),
            self.canny_high.to_string(),
            "--min-density".to_string(),
            self.min_density_count.to_string(),
            "--min-temporal".to_string(),
            self.min_temporal_spread_us.to_string(),
            "--bidirectional-ratio".to_string(),
            self.bidirectional_ratio.to_string(),
        ];

        if self.filter_dead_pixels {
            args.push("--filter-dead-pixels".to_string());
        }
        if self.filter_density {
            args.push("--filter-density".to_string());
        }
        if self.filter_temporal {
            args.push("--filter-temporal".to_string());
        }
        if self.filter_bidirectional {
            args.push("--filter-bidirectional".to_string());
        }

        args
    }
}

/// Results from a single hyperparameter test run.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HyperResult {
    pub config: HyperConfig,
    pub avg_edge_count: f32,
    pub edge_density: f32,
    pub centroid_stability: f32,
    pub radius_stability: f32,
    pub circle_fit_error: f32,
    pub inlier_ratio: f32,
    pub detected_blade_count: f32,
    pub frames_processed: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub iou: f32,
}

impl HyperResult {
    /// Create a result with zero metrics (for placeholder/error cases).
    pub fn empty(config: HyperConfig) -> Self {
        Self {
            config,
            avg_edge_count: 0.0,
            edge_density: 0.0,
            centroid_stability: f32::MAX,
            radius_stability: f32::MAX,
            circle_fit_error: f32::MAX,
            inlier_ratio: 0.0,
            detected_blade_count: 0.0,
            frames_processed: 0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            iou: 0.0,
        }
    }

    /// Score this result (lower is better). Used for selecting best config.
    /// Primary: centroid_stability, Secondary: inlier_ratio (inverted)
    pub fn score(&self) -> f32 {
        // Lower centroid_stability is better (stable)
        // Higher inlier_ratio is better (good circle fit)
        // Weight: 70% stability, 30% inlier
        let stability_score = self.centroid_stability;
        let inlier_penalty = 1.0 - self.inlier_ratio; // Invert so lower is better

        0.7 * stability_score + 0.3 * inlier_penalty * 100.0
    }
}

/// Generate a grid of configurations to test.
pub fn generate_grid_configs(
    detectors: &[String],
    window_sizes: &[f32],
    thresholds: &[f32],
    filter_combinations: bool,
) -> Vec<HyperConfig> {
    let mut configs = Vec::new();

    let filter_options: Vec<(bool, bool, bool)> = if filter_combinations {
        vec![
            (false, false, false), // No filters
            (true, false, false),  // Dead pixel only
            (true, true, false),   // Dead pixel + density
            (true, false, true),   // Dead pixel + temporal
            (true, true, true),    // All filters
        ]
    } else {
        vec![(true, false, false)] // Default: just dead pixel
    };

    for detector in detectors {
        for &window_size in window_sizes {
            for &threshold in thresholds {
                for &(dead, density, temporal) in &filter_options {
                    let config = HyperConfig {
                        detector: detector.clone(),
                        window_size_us: window_size,
                        threshold,
                        filter_dead_pixels: dead,
                        filter_density: density,
                        filter_temporal: temporal,
                        ..Default::default()
                    };
                    configs.push(config);
                }
            }
        }
    }

    configs
}

/// Export results to CSV format.
pub fn export_csv(results: &[HyperResult], path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    // Header
    writeln!(
        file,
        "detector,window_size,threshold,canny_low,canny_high,\
         filter_dead,filter_density,filter_temporal,filter_bidir,\
         avg_edges,density,centroid_stab,radius_stab,fit_error,\
         inlier_ratio,blades,frames,precision,recall,f1_score,iou,score"
    )?;

    // Data rows
    for r in results {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{:.0},{:.4},{:.2},{:.2},{:.2},{:.2},{:.1},{},{:.4},{:.4},{:.4},{:.4},{:.2}",
            r.config.detector,
            r.config.window_size_us,
            r.config.threshold,
            r.config.canny_low,
            r.config.canny_high,
            r.config.filter_dead_pixels as u8,
            r.config.filter_density as u8,
            r.config.filter_temporal as u8,
            r.config.filter_bidirectional as u8,
            r.avg_edge_count,
            r.edge_density,
            r.centroid_stability,
            r.radius_stability,
            r.circle_fit_error,
            r.inlier_ratio,
            r.detected_blade_count,
            r.frames_processed,
            r.precision,
            r.recall,
            r.f1_score,
            r.iou,
            r.score()
        )?;
    }

    Ok(())
}

/// Select the best result based on score.
pub fn select_best(results: &[HyperResult]) -> Option<&HyperResult> {
    results
        .iter()
        .filter(|r| r.frames_processed > 0)
        .min_by(|a, b| a.score().partial_cmp(&b.score()).unwrap())
}

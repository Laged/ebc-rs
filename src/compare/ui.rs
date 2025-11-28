//! Egui overlay for metrics display and controls for compare_live.

use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};

use super::{AllDetectorMetrics, DetectorMetrics};
use crate::gpu::EdgeParams;
use crate::playback::PlaybackState;
use crate::cm::{CmParams, CmResult};
use crate::cmax_slam::{CmaxSlamParams, CmaxSlamState};
use crate::ground_truth::GroundTruthConfig;

/// Resource tracking which data file is active
#[derive(Resource, Default)]
pub struct DataFileState {
    pub files: Vec<std::path::PathBuf>,
    pub current_index: usize,
}

impl DataFileState {
    pub fn current_file(&self) -> Option<&std::path::PathBuf> {
        self.files.get(self.current_index)
    }

    pub fn next(&mut self) {
        if !self.files.is_empty() {
            self.current_index = (self.current_index + 1) % self.files.len();
        }
    }

    pub fn prev(&mut self) {
        if !self.files.is_empty() {
            self.current_index = (self.current_index + self.files.len() - 1) % self.files.len();
        }
    }
}

/// Draw playback controls
pub fn draw_playback_controls(
    mut contexts: EguiContexts,
    mut playback_state: ResMut<PlaybackState>,
    diagnostics: Res<DiagnosticsStore>,
) {
    let ctx = contexts.ctx_mut().expect("Failed to get egui context");

    egui::Window::new("Playback")
        .default_pos([10.0, 10.0])
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button(if playback_state.is_playing { "Pause" } else { "Play" }).clicked() {
                    playback_state.is_playing = !playback_state.is_playing;
                }
                ui.checkbox(&mut playback_state.looping, "Loop");
            });

            let max_time = playback_state.max_timestamp as f32;
            ui.add(
                egui::Slider::new(&mut playback_state.current_time, 0.0..=max_time)
                    .text("Time (μs)"),
            );

            ui.add(
                egui::Slider::new(&mut playback_state.window_size, 1.0..=100_000.0)
                    .text("Window (μs)")
                    .logarithmic(true),
            );

            ui.add(
                egui::Slider::new(&mut playback_state.playback_speed, 0.01..=100.0)
                    .text("Speed (×)")
                    .logarithmic(true),
            );

            ui.label(format!("Time: {:.2} ms", playback_state.current_time / 1000.0));
            ui.label(format!("Window: {:.2} ms", playback_state.window_size / 1000.0));

            if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
                if let Some(value) = fps.smoothed() {
                    ui.label(format!("FPS: {:.1}", value));
                }
            }
        });
}

/// Draw edge detection parameter controls
pub fn draw_edge_controls(
    mut contexts: EguiContexts,
    mut edge_params: ResMut<EdgeParams>,
    mut cm_params: ResMut<CmParams>,
    cmax_params: Option<ResMut<CmaxSlamParams>>,
    cmax_state: Option<Res<CmaxSlamState>>,
    gt_config: Option<Res<GroundTruthConfig>>,
) {
    let ctx = contexts.ctx_mut().expect("Failed to get egui context");

    egui::Window::new("Detector Settings")
        .default_pos([10.0, 250.0])
        .show(ctx, |ui| {
            // Detector visibility toggles
            ui.heading("Visibility");
            ui.checkbox(&mut edge_params.show_raw, "Show Raw (Q1)");
            ui.checkbox(&mut edge_params.show_sobel, "Show CMax-SLAM (Q2)");
            ui.checkbox(&mut edge_params.show_canny, "Show CM (Q3)");
            ui.checkbox(&mut edge_params.show_log, "Show LoG (Q4)");

            ui.separator();
            ui.heading("Thresholds");

            // Sobel threshold (binary inputs: magnitude 0-5.66)
            ui.add(egui::Slider::new(&mut edge_params.sobel_threshold, 0.0..=6.0)
                .text("Sobel"));

            // LoG threshold (binary inputs: response 0-16)
            ui.add(egui::Slider::new(&mut edge_params.log_threshold, 0.0..=16.0)
                .text("LoG"));

            ui.separator();
            ui.heading("CM Settings");

            // CM search resolution
            let mut n_omega_float = cm_params.n_omega as f32;
            ui.add(egui::Slider::new(&mut n_omega_float, 16.0..=128.0)
                .text("Search Resolution")
                .step_by(16.0));
            cm_params.n_omega = n_omega_float as u32;

            ui.checkbox(&mut cm_params.enabled, "Enable CM");

            ui.separator();
            ui.heading("Filters");
            ui.checkbox(&mut edge_params.filter_dead_pixels, "Filter Dead Pixels");
            ui.checkbox(&mut edge_params.filter_density, "Filter Low Density");
            ui.checkbox(&mut edge_params.filter_bidirectional, "Bidirectional");

            // CMax-SLAM controls
            if let Some(mut params) = cmax_params {
                ui.separator();
                ui.collapsing("CMax-SLAM", |ui| {
                    ui.checkbox(&mut params.enabled, "Enable");

                    ui.add(egui::Slider::new(&mut params.learning_rate, 0.0001..=0.01)
                        .text("Learning Rate")
                        .logarithmic(true));

                    ui.add(egui::Slider::new(&mut params.smoothing_alpha, 0.0..=1.0)
                        .text("Smoothing Alpha"));

                    if let Some(state) = cmax_state.as_ref() {
                        ui.separator();
                        ui.heading("Estimation");

                        // Calculate estimated RPM
                        let est_rpm = state.omega.abs() * 60.0 / std::f32::consts::TAU * 1e6;

                        // Display estimated RPM prominently
                        ui.horizontal(|ui| {
                            ui.label("Est RPM:");
                            ui.label(
                                egui::RichText::new(format!("{:.1}", est_rpm))
                                    .size(20.0)
                                    .strong()
                            );
                        });

                        // Display omega
                        ui.label(format!("Omega: {:.6} rad/μs", state.omega));

                        // Display convergence status with color coding
                        let convergence_color = if state.converged {
                            egui::Color32::GREEN
                        } else {
                            egui::Color32::YELLOW
                        };
                        ui.label(
                            egui::RichText::new(format!("Converged: {}", state.converged))
                                .color(convergence_color)
                        );

                        // Display contrast value
                        ui.label(format!("Contrast: {:.3}", state.contrast));

                        // Ground truth comparison (if available)
                        if let Some(gt) = gt_config.as_ref() {
                            if gt.enabled && gt.rpm > 0.0 {
                                ui.separator();
                                ui.heading("Ground Truth Comparison");

                                // Display ground truth RPM
                                ui.label(format!("GT RPM: {:.1}", gt.rpm));
                                ui.label(format!("GT Blades: {}", gt.blade_count));

                                // Calculate and display RPM error
                                let rpm_error_pct = ((est_rpm - gt.rpm).abs() / gt.rpm) * 100.0;

                                // Color code the error based on thresholds
                                let error_color = if rpm_error_pct < 1.0 {
                                    egui::Color32::from_rgb(0, 255, 0)  // Green < 1%
                                } else if rpm_error_pct < 5.0 {
                                    egui::Color32::from_rgb(255, 200, 0)  // Yellow < 5%
                                } else {
                                    egui::Color32::from_rgb(255, 0, 0)  // Red >= 5%
                                };

                                ui.horizontal(|ui| {
                                    ui.label("RPM Error:");
                                    ui.label(
                                        egui::RichText::new(format!("{:.2}%", rpm_error_pct))
                                            .color(error_color)
                                            .size(18.0)
                                            .strong()
                                    );
                                });
                            }
                        }
                    }
                });
            }
        });
}

/// Draw metrics overlay for compare_live
pub fn draw_metrics_overlay(
    mut contexts: EguiContexts,
    metrics: Res<AllDetectorMetrics>,
    file_state: Res<DataFileState>,
) {
    let ctx = contexts.ctx_mut().expect("Failed to get egui context");

    // Top bar with file info
    egui::TopBottomPanel::top("file_info").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if let Some(path) = file_state.current_file() {
                ui.label(format!("File: {} ({}/{})",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    file_state.current_index + 1,
                    file_state.files.len()
                ));
            }
            ui.separator();
            ui.label(format!("Frame: {:.1}ms", metrics.frame_time_ms));
            ui.separator();
            ui.label("[N]ext [P]rev file | [Space] pause");
        });
    });

    // Metrics panels positioned near center where quadrants meet
    // Window size: 2560x1440, each quadrant is 1280x720
    // Panels should be positioned near the center dividing lines
    let panel_width = 180.0;
    let panel_height = 90.0;
    let center_x = 1280.0;  // Horizontal divider
    let center_y = 720.0;   // Vertical divider
    let margin = 10.0;      // Gap from center lines

    // Q1 Top-left (RAW): position at bottom-right of quadrant (near center)
    draw_detector_panel(
        ctx, "RAW", &metrics.raw,
        center_x - panel_width - margin,  // Left of center line
        center_y - panel_height - margin, // Above center line
        panel_width, panel_height
    );

    // Q2 Top-right (CMax-SLAM): position at bottom-left of quadrant (near center)
    draw_detector_panel(
        ctx, "CMax-SLAM", &metrics.sobel,
        center_x + margin,                // Right of center line
        center_y - panel_height - margin, // Above center line
        panel_width, panel_height
    );

    // Q3 Bottom-left (CM): position at top-right of quadrant (near center)
    draw_cm_panel(
        ctx, &metrics.cm,
        center_x - panel_width - margin,  // Left of center line
        center_y + margin,                // Below center line
        panel_width, panel_height
    );

    // Q4 Bottom-right (LoG): position at top-left of quadrant (near center)
    draw_detector_panel(
        ctx, "LoG", &metrics.log,
        center_x + margin,                // Right of center line
        center_y + margin,                // Below center line
        panel_width, panel_height
    );
}

fn draw_detector_panel(
    ctx: &egui::Context,
    name: &str,
    metrics: &DetectorMetrics,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
) {
    egui::Window::new(name)
        .fixed_pos([x, y])
        .fixed_size([width, height])
        .title_bar(false)
        .frame(egui::Frame::window(&ctx.style()).fill(egui::Color32::from_rgba_unmultiplied(20, 20, 20, 200)))
        .show(ctx, |ui| {
            ui.heading(name);
            ui.separator();
            ui.label(format!("Edges: {}", metrics.edge_count));
            ui.label(format!("Prec: {:.1}% | Rec: {:.1}%",
                metrics.tolerance_precision * 100.0,
                metrics.tolerance_recall * 100.0
            ));
            ui.label(format!("F1: {:.1}% | Dist: {:.1}px",
                metrics.tolerance_f1 * 100.0,
                metrics.avg_distance
            ));
        });
}

fn draw_cm_panel(
    ctx: &egui::Context,
    cm_result: &CmResult,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
) {
    egui::Window::new("CM")
        .fixed_pos([x, y])
        .fixed_size([width, height])
        .title_bar(false)
        .frame(egui::Frame::window(&ctx.style()).fill(egui::Color32::from_rgba_unmultiplied(20, 20, 20, 200)))
        .show(ctx, |ui| {
            ui.heading("CM");
            ui.separator();

            // Display RPM prominently with large font
            ui.horizontal(|ui| {
                ui.label("RPM:");
                ui.label(
                    egui::RichText::new(format!("{:.0}", cm_result.rpm))
                        .size(24.0)
                        .strong()
                );
            });

            // Quality indicator based on confidence
            let quality_text = if cm_result.confidence > 0.8 {
                "Quality: Excellent"
            } else if cm_result.confidence > 0.5 {
                "Quality: Good"
            } else if cm_result.confidence > 0.2 {
                "Quality: Fair"
            } else {
                "Quality: Poor"
            };

            let quality_color = if cm_result.confidence > 0.8 {
                egui::Color32::from_rgb(0, 255, 0)
            } else if cm_result.confidence > 0.5 {
                egui::Color32::from_rgb(150, 255, 0)
            } else if cm_result.confidence > 0.2 {
                egui::Color32::from_rgb(255, 200, 0)
            } else {
                egui::Color32::from_rgb(255, 0, 0)
            };

            ui.label(egui::RichText::new(quality_text).color(quality_color));

            // Additional details
            ui.label(format!("ω: {:.2e} rad/μs", cm_result.best_omega));
        });
}

/// Handle keyboard input for file switching
pub fn handle_file_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut file_state: ResMut<DataFileState>,
) {
    if keyboard.just_pressed(KeyCode::KeyN) {
        file_state.next();
    }
    if keyboard.just_pressed(KeyCode::KeyP) {
        file_state.prev();
    }
}

/// Plugin for compare UI
pub struct CompareUiPlugin;

impl Plugin for CompareUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DataFileState>()
            // Note: AllDetectorMetrics is initialized by CompositeRenderPlugin
            // Use EguiPrimaryContextPass schedule for egui systems
            .add_systems(EguiPrimaryContextPass, (
                draw_playback_controls,
                draw_edge_controls,
                draw_metrics_overlay,
            ))
            .add_systems(Update, handle_file_input);
    }
}

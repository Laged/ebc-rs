//! Egui overlay for metrics display.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};

use super::{AllDetectorMetrics, DetectorMetrics};
use crate::gpu::EdgeParams;

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

/// Draw edge detection parameter controls
pub fn draw_edge_controls(
    mut contexts: EguiContexts,
    mut edge_params: ResMut<EdgeParams>,
) {
    let ctx = contexts.ctx_mut().expect("Failed to get egui context");

    egui::Window::new("Edge Detection")
        .default_pos([10.0, 100.0])
        .show(ctx, |ui| {
            // Detector visibility toggles
            ui.heading("Visibility");
            ui.checkbox(&mut edge_params.show_raw, "Show Raw (Q1)");
            ui.checkbox(&mut edge_params.show_sobel, "Show Sobel (Q2)");
            ui.checkbox(&mut edge_params.show_canny, "Show Canny (Q3)");
            ui.checkbox(&mut edge_params.show_log, "Show LoG (Q4)");

            ui.separator();
            ui.heading("Thresholds");

            // Sobel threshold
            ui.add(egui::Slider::new(&mut edge_params.sobel_threshold, 0.0..=10_000.0)
                .text("Sobel"));

            // Canny thresholds
            ui.add(egui::Slider::new(&mut edge_params.canny_low_threshold, 0.0..=5_000.0)
                .text("Canny Low"));
            ui.add(egui::Slider::new(&mut edge_params.canny_high_threshold, 0.0..=10_000.0)
                .text("Canny High"));

            // LoG threshold
            ui.add(egui::Slider::new(&mut edge_params.log_threshold, 0.0..=10_000.0)
                .text("LoG"));

            ui.separator();
            ui.heading("Filters");
            ui.checkbox(&mut edge_params.filter_dead_pixels, "Filter Dead Pixels");
            ui.checkbox(&mut edge_params.filter_density, "Filter Low Density");
            ui.checkbox(&mut edge_params.filter_bidirectional, "Bidirectional");
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

    // Q2 Top-right (SOBEL): position at bottom-left of quadrant (near center)
    draw_detector_panel(
        ctx, "SOBEL", &metrics.sobel,
        center_x + margin,                // Right of center line
        center_y - panel_height - margin, // Above center line
        panel_width, panel_height
    );

    // Q3 Bottom-left (CANNY): position at top-right of quadrant (near center)
    draw_detector_panel(
        ctx, "CANNY", &metrics.canny,
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
                draw_metrics_overlay,
                draw_edge_controls,
            ))
            .add_systems(Update, handle_file_input);
    }
}

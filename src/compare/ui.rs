//! Egui overlay for metrics display.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use super::{AllDetectorMetrics, DetectorMetrics};

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

    // Metrics panels for each quadrant
    let panel_width = 200.0;
    let panel_height = 100.0;

    // Top-left: Raw
    draw_detector_panel(ctx, "RAW", &metrics.raw, 10.0, 40.0, panel_width, panel_height);

    // Top-right: Sobel
    let screen_width = ctx.viewport_rect().width();
    draw_detector_panel(ctx, "SOBEL", &metrics.sobel, screen_width / 2.0 + 10.0, 40.0, panel_width, panel_height);

    // Bottom-left: Canny
    let screen_height = ctx.viewport_rect().height();
    draw_detector_panel(ctx, "CANNY", &metrics.canny, 10.0, screen_height / 2.0 + 10.0, panel_width, panel_height);

    // Bottom-right: LoG
    draw_detector_panel(ctx, "LoG", &metrics.log, screen_width / 2.0 + 10.0, screen_height / 2.0 + 10.0, panel_width, panel_height);
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
            .init_resource::<AllDetectorMetrics>()
            .add_systems(Update, (draw_metrics_overlay, handle_file_input));
    }
}

use crate::analysis::FanAnalysis;
use bevy::prelude::*;

/// Plugin for gizmo visualization of fan analysis
pub struct FanGizmosPlugin;

impl Plugin for FanGizmosPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_fan_visualization);
    }
}

/// Draw fan borders and centroid visualization
fn draw_fan_visualization(analysis: Res<FanAnalysis>, mut gizmos: Gizmos) {
    if !analysis.show_borders {
        return;
    }

    // Convert centroid to 3D position (Z = 0 to match the event visualization plane)
    let center = Vec3::new(
        analysis.centroid.x - 640.0,    // Center coordinate system
        -(analysis.centroid.y - 360.0), // Flip Y axis (screen coords to world coords)
        1.0,                            // Slightly in front of the event plane
    );

    // Draw centroid marker (cross)
    let marker_size = 10.0;
    gizmos.line(
        center + Vec3::new(-marker_size, 0.0, 0.0),
        center + Vec3::new(marker_size, 0.0, 0.0),
        Color::srgb(1.0, 0.0, 0.0), // Red
    );
    gizmos.line(
        center + Vec3::new(0.0, -marker_size, 0.0),
        center + Vec3::new(0.0, marker_size, 0.0),
        Color::srgb(1.0, 0.0, 0.0), // Red
    );

    // Draw blade borders
    let blade_count = analysis.blade_count as f32;
    let angle_per_blade = 2.0 * std::f32::consts::PI / blade_count;

    for i in 0..analysis.blade_count {
        let angle = analysis.current_angle + (i as f32 * angle_per_blade);

        // Calculate blade edge positions
        let dx = angle.cos() * analysis.fan_radius;
        let dy = angle.sin() * analysis.fan_radius;

        let blade_end = center + Vec3::new(dx, dy, 0.0);

        // Draw line from center to blade tip
        gizmos.line(
            center,
            blade_end,
            Color::srgb(0.0, 1.0, 0.0), // Green
        );

        // Draw a small circle at the blade tip
        let tip_radius = 5.0;
        gizmos.circle(
            blade_end,
            tip_radius,
            Color::srgb(1.0, 1.0, 0.0), // Yellow
        );
    }

    // Draw fan radius circle
    gizmos.circle(
        center,
        analysis.fan_radius,
        Color::srgba(0.5, 0.5, 1.0, 0.5), // Semi-transparent blue
    );
}

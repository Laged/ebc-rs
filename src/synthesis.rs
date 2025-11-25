//! # Synthetic Fan Data Generation
//!
//! This module generates synthetic event camera data simulating a rotating fan
//! with multiple blades. It produces both the binary event data (.dat) and
//! ground truth JSON for validation.
//!
//! ## Fan Model
//! - Blades follow a logarithmic spiral profile for realistic shape
//! - Events are generated at blade edges (leading/trailing)
//! - Leading edges (front of rotation) have polarity 1 (ON events)
//! - Trailing edges (back of rotation) have polarity 0 (OFF events)
//! - Position and angular jitter simulate sensor noise

use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Generate synthetic fan event data
///
/// # Arguments
/// * `output_path` - Path to write the binary .dat file
/// * `truth_path` - Path to write the ground truth JSON file
///
/// # Returns
/// * `Ok(())` on success
/// * `Err` if file operations fail
pub fn generate_fan_data(output_path: &Path, truth_path: &Path) -> std::io::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = truth_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(output_path)?;
    let mut truth_file = File::create(truth_path)?;

    // Write header lines
    writeln!(file, "% SYNTHETIC FAN DATA")?;
    writeln!(file, "% RPM: 1200")?;
    writeln!(file, "% BLADES: 3")?;
    writeln!(file, "% RESOLUTION: 1280x720")?;
    writeln!(file, "% DURATION: 2 seconds")?;

    // Write event type (0x00) and size (8 bytes)
    file.write_all(&[0x00, 0x08])?;

    // Fan parameters
    let duration_secs = 2.0;
    let rpm = 1200.0;
    let rps = rpm / 60.0; // revolutions per second = 20 Hz
    let angular_velocity = rps * 2.0 * PI; // radians per second
    let blade_count = 3;
    let radius = 200.0; // pixels
    let center_x = 640.0; // center of 1280x720 frame
    let center_y = 360.0;
    let events_per_sec = 100_000;
    let total_events = (events_per_sec as f32 * duration_secs) as usize;

    let mut truth_entries = Vec::new();
    let mut time_step_us = (1_000_000.0 / events_per_sec as f32) as u32;
    if time_step_us == 0 {
        time_step_us = 1;
    }

    let mut current_time_us = 0u32;

    // Simple Linear Congruential Generator for deterministic randomness
    let mut seed: u32 = 12345;
    let rand = |seed: &mut u32| -> f32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (*seed as f32) / (u32::MAX as f32)
    };

    for _ in 0..total_events {
        current_time_us += time_step_us;
        let t_secs = current_time_us as f32 / 1_000_000.0;

        // Calculate current rotation angle
        let base_angle = angular_velocity * t_secs;

        // Randomly select a blade
        let blade_idx = (rand(&mut seed) * blade_count as f32) as usize;
        let blade_spacing = 2.0 * PI / blade_count as f32;
        let root_angle = base_angle + (blade_idx as f32 * blade_spacing);

        // Randomly select radius along blade (from 50px to full radius)
        let r_min = 50.0;
        let r_max = radius;
        let r = r_min + rand(&mut seed) * (r_max - r_min);

        // Blade shape: logarithmic spiral for swept-back design
        // This creates a realistic fan blade curvature
        let sweep_k = 0.5; // curvature parameter
        let sweep_angle = sweep_k * (r / r_min).ln();
        let center_angle = root_angle + sweep_angle;

        // Blade width distribution (wider at root, narrower at tip)
        let r_norm = (r - r_min) / (r_max - r_min);
        let width_rad_root = 0.5;
        let width_rad_tip = 0.3;
        let blade_width = width_rad_root + (width_rad_tip - width_rad_root) * r_norm;
        let half_width = blade_width * 0.5;

        // Randomly assign event to leading or trailing edge
        // Leading edge: front of rotation (polarity 1 - ON/Red)
        // Trailing edge: back of rotation (polarity 0 - OFF/Blue)
        let is_leading_edge = rand(&mut seed) > 0.5;

        let (edge_offset, polarity) = if is_leading_edge {
            (half_width, 1)
        } else {
            (-half_width, 0)
        };

        // Calculate final angle with edge offset and jitter
        let angular_jitter = (rand(&mut seed) - 0.5) * 0.02;
        let theta = center_angle + edge_offset + angular_jitter;

        // Add position noise to simulate sensor imperfections
        let jitter_x = (rand(&mut seed) - 0.5) * 1.0;
        let jitter_y = (rand(&mut seed) - 0.5) * 1.0;

        let x = center_x + r * theta.cos() + jitter_x;
        let y = center_y + r * theta.sin() + jitter_y;

        // Bounds check (1280x720 resolution)
        if x < 0.0 || x >= 1280.0 || y < 0.0 || y >= 720.0 {
            continue;
        }

        let x_u32 = x as u32;
        let y_u32 = y as u32;

        // Encode event in binary format
        // w32: bits 0-13 = x, bits 14-27 = y, bits 28-31 = polarity
        let w32 = (x_u32 & 0x3FFF) | ((y_u32 & 0x3FFF) << 14) | ((polarity & 0xF) << 28);

        // Write timestamp (4 bytes, little-endian)
        file.write_all(&current_time_us.to_le_bytes())?;
        // Write encoded data (4 bytes, little-endian)
        file.write_all(&w32.to_le_bytes())?;

        // Save ground truth every 10ms (10,000 microseconds)
        if current_time_us % 10_000 == 0 {
            truth_entries.push(format!(
                "{{\"time\": {:.4}, \"angle\": {:.4}, \"rpm\": {:.1}, \"centroid_x\": {:.1}, \"centroid_y\": {:.1}, \"radius\": {:.1}}}",
                t_secs,
                base_angle % (2.0 * PI),
                rpm,
                center_x,
                center_y,
                radius
            ));
        }
    }

    // Blade geometry constants used in generation
    let r_min = 50.0_f32;
    let sweep_k = 0.5_f32;
    let width_root_rad = 0.5_f32;
    let width_tip_rad = 0.3_f32;

    // Write ground truth JSON with params header
    writeln!(truth_file, "{{")?;
    writeln!(truth_file, "  \"params\": {{")?;
    writeln!(truth_file, "    \"center_x\": {:.1},", center_x)?;
    writeln!(truth_file, "    \"center_y\": {:.1},", center_y)?;
    writeln!(truth_file, "    \"radius_min\": {:.1},", r_min)?;
    writeln!(truth_file, "    \"radius_max\": {:.1},", radius)?;
    writeln!(truth_file, "    \"blade_count\": {},", blade_count)?;
    writeln!(truth_file, "    \"rpm\": {:.1},", rpm)?;
    writeln!(truth_file, "    \"sweep_k\": {:.2},", sweep_k)?;
    writeln!(truth_file, "    \"width_root_rad\": {:.2},", width_root_rad)?;
    writeln!(truth_file, "    \"width_tip_rad\": {:.2},", width_tip_rad)?;
    writeln!(truth_file, "    \"edge_thickness_px\": 2.0")?;
    writeln!(truth_file, "  }},")?;
    writeln!(truth_file, "  \"frames\": [")?;
    writeln!(truth_file, "{}", truth_entries.join(",\n"))?;
    writeln!(truth_file, "  ]")?;
    writeln!(truth_file, "}}")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::DatLoader;

    #[test]
    fn test_generate_fan_data() -> std::io::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let output_path = temp_dir.path().join("test_fan.dat");
        let truth_path = temp_dir.path().join("test_truth.json");

        generate_fan_data(&output_path, &truth_path)?;

        // Verify files exist
        assert!(output_path.exists());
        assert!(truth_path.exists());

        // Verify file sizes are reasonable
        let dat_metadata = std::fs::metadata(&output_path)?;
        let json_metadata = std::fs::metadata(&truth_path)?;

        assert!(dat_metadata.len() > 0);
        assert!(json_metadata.len() > 0);

        // Verify .dat file has proper header
        let dat_contents = std::fs::read(&output_path)?;
        assert!(dat_contents.starts_with(b"% SYNTHETIC FAN DATA"));

        // Verify JSON is valid (now has params + frames structure)
        let json_contents = std::fs::read_to_string(&truth_path)?;
        assert!(json_contents.starts_with("{"));
        assert!(json_contents.contains("\"params\""));
        assert!(json_contents.contains("\"frames\""));
        assert!(json_contents.ends_with("}\n"));

        Ok(())
    }

    #[test]
    fn test_generated_data_loadable() {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("test_fan.dat");
        let truth_path = temp_dir.path().join("test_truth.json");

        generate_fan_data(&output_path, &truth_path).unwrap();

        // Verify DatLoader can load the generated data
        let events = DatLoader::load(&output_path).expect("Should load generated data");

        // Should have close to 200,000 events (2 sec * 100,000/sec, minus bounds filtering)
        assert!(events.len() > 100_000, "Expected >100k events, got {}", events.len());
        assert!(events.len() < 250_000, "Expected <250k events, got {}", events.len());

        // Verify event coordinates are in bounds
        for event in &events {
            assert!(event.x < 1280, "X out of bounds: {}", event.x);
            assert!(event.y < 720, "Y out of bounds: {}", event.y);
            assert!(event.polarity <= 1, "Invalid polarity: {}", event.polarity);
        }

        // Verify timestamps are increasing (loader sorts them)
        for window in events.windows(2) {
            assert!(window[0].timestamp <= window[1].timestamp, "Events not sorted");
        }

        // Verify events are centered around fan center (640, 360)
        let avg_x: f32 = events.iter().map(|e| e.x as f32).sum::<f32>() / events.len() as f32;
        let avg_y: f32 = events.iter().map(|e| e.y as f32).sum::<f32>() / events.len() as f32;
        assert!((avg_x - 640.0).abs() < 50.0, "Avg X too far from center: {}", avg_x);
        assert!((avg_y - 360.0).abs() < 50.0, "Avg Y too far from center: {}", avg_y);
    }
}

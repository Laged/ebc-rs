use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::f32::consts::PI;

pub fn generate_fan_data(output_path: &Path, truth_path: &Path) -> std::io::Result<()> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let mut file = File::create(output_path)?;
    let mut truth_file = File::create(truth_path)?;
    
    // 1. Write Header
    writeln!(file, "% SYNTHETIC FAN DATA")?;
    writeln!(file, "% RPM: 1200")?;
    writeln!(file, "% BLADES: 3")?;
    writeln!(file, "% RESOLUTION: 1280x720")?;
    
    // 2. Write Event Type & Size
    file.write_all(&[0x00, 0x08])?;
    
    // Parameters
    let duration_secs = 2.0;
    let rpm = 1200.0;
    let rps = rpm / 60.0; // 20 Hz
    let angular_velocity = rps * 2.0 * PI; // rad/s
    let blade_count = 3;
    let radius = 200.0;
    let center_x = 640.0;
    let center_y = 360.0;
    let events_per_sec = 100_000;
    let total_events = (events_per_sec as f32 * duration_secs) as usize;
    
    let mut truth_entries = Vec::new();
    let mut time_step_us = (1_000_000.0 / events_per_sec as f32) as u32;
    if time_step_us == 0 { time_step_us = 1; }
    
    let mut current_time_us = 0u32;
    
    // Simple Random Number Generator (LCG)
    let mut seed: u32 = 12345;
    let rand = |seed: &mut u32| -> f32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (*seed as f32) / (u32::MAX as f32)
    };
    
    for _ in 0..total_events {
        current_time_us += time_step_us;
        let t_secs = current_time_us as f32 / 1_000_000.0;
        
        // Calculate current rotation
        let base_angle = angular_velocity * t_secs;
        
        // Pick a random blade
        let blade_idx = (rand(&mut seed) * blade_count as f32) as usize;
        let blade_spacing = 2.0 * PI / blade_count as f32;
        let root_angle = base_angle + (blade_idx as f32 * blade_spacing);
        
        // Pick a random radius (50px to radius)
        let r_min = 50.0;
        let r_max = radius;
        let r = r_min + rand(&mut seed) * (r_max - r_min);
        
        // --- Blade Shape Design ---
        // "Fan rotor blade shapes are defined by complex mathematical formulas...
        // based on a logarithmic spiral for a simplified design."
        // Logarithmic Spiral: theta = k * ln(r / r0)
        // We define a sweep angle that lags the rotation (swept back).
        let sweep_k = 0.5; // Controls curvature
        let sweep_angle = sweep_k * (r / r_min).ln();
        let center_angle = root_angle + sweep_angle;
        
        // Thickness/Width distribution
        // We want the events to appear at the EDGES of the blade to show depth.
        // Width: 0.5 rad at root, 0.3 rad at tip (Increased for visibility)
        // This ensures "Blue" and "Red" lines are separated by the blade body.
        let r_norm = (r - r_min) / (r_max - r_min);
        let width_rad_root = 0.5; 
        let width_rad_tip = 0.3;
        let blade_width = width_rad_root + (width_rad_tip - width_rad_root) * r_norm;
        let half_width = blade_width * 0.5;

        // Randomly pick which edge this event belongs to:
        // Leading Edge (Front of rotation) -> Polarity 1 (Red/ON) -> Angle + half_width
        // Trailing Edge (Back of rotation) -> Polarity 0 (Blue/OFF) -> Angle - half_width
        let is_leading_edge = rand(&mut seed) > 0.5;
        
        let (edge_offset, polarity) = if is_leading_edge {
            (half_width, 1)
        } else {
            (-half_width, 0)
        };
        
        // Final angle for this event
        // Add small angular jitter to simulate edge fuzziness/sensor noise
        let angular_jitter = (rand(&mut seed) - 0.5) * 0.02; 
        let theta = center_angle + edge_offset + angular_jitter;
        
        // Add some noise to position (sensor noise)
        let jitter_x = (rand(&mut seed) - 0.5) * 1.0; // +/- 0.5px
        let jitter_y = (rand(&mut seed) - 0.5) * 1.0;
        
        let x = center_x + r * theta.cos() + jitter_x;
        let y = center_y + r * theta.sin() + jitter_y;
        
        // Bounds check
        if x < 0.0 || x >= 1280.0 || y < 0.0 || y >= 720.0 {
            continue;
        }
        
        let x_u32 = x as u32;
        let y_u32 = y as u32;
        let polarity = if rand(&mut seed) > 0.5 { 1 } else { 0 };
        
        // Encode
        // t32: little endian
        // w32: x | y<<14 | p<<28
        let w32 = (x_u32 & 0x3FFF) | ((y_u32 & 0x3FFF) << 14) | ((polarity & 0xF) << 28);
        
        file.write_all(&current_time_us.to_le_bytes())?;
        file.write_all(&w32.to_le_bytes())?;
        
        // Save ground truth every 10ms (10,000us)
        if current_time_us % 10_000 == 0 {
            truth_entries.push(format!(
                "{{\"time\": {:.4}, \"angle\": {:.4}, \"rpm\": {:.1}, \"centroid_x\": {:.1}, \"centroid_y\": {:.1}, \"radius\": {:.1}}}",
                t_secs, base_angle % (2.0 * PI), rpm, center_x, center_y, radius
            ));
        }
    }
    
    // Write Truth JSON
    writeln!(truth_file, "[")?;
    writeln!(truth_file, "{}", truth_entries.join(",\n"))?;
    writeln!(truth_file, "]")?;
    
    Ok(())
}

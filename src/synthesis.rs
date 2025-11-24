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
        let blade_angle = base_angle + (blade_idx as f32 * 2.0 * PI / blade_count as f32);
        
        // Pick a random distance along the blade (50px to radius)
        let r = 50.0 + rand(&mut seed) * (radius - 50.0);
        
        // Add some noise to position
        let jitter_x = (rand(&mut seed) - 0.5) * 2.0; // +/- 1px
        let jitter_y = (rand(&mut seed) - 0.5) * 2.0;
        
        let x = center_x + r * blade_angle.cos() + jitter_x;
        let y = center_y + r * blade_angle.sin() + jitter_y;
        
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

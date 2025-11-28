//! Generate synthetic fan data for testing
//!
//! Usage: cargo run --bin generate_synthetic -- [OPTIONS]

use clap::Parser;
use std::path::PathBuf;
use ebc_rs::synthesis::{generate_fan_data_with_config, SynthConfig};

fn validate_noise(s: &str) -> Result<f32, String> {
    let value: f32 = s.parse().map_err(|_| format!("'{}' is not a valid number", s))?;
    if value < 0.0 || value > 1.0 {
        Err(format!("noise must be between 0.0 and 1.0, got {}", value))
    } else {
        Ok(value)
    }
}

#[derive(Parser, Debug)]
#[command(name = "generate_synthetic")]
#[command(about = "Generate synthetic fan event data for testing", long_about = None)]
struct Args {
    /// Fan rotation speed in revolutions per minute
    #[arg(long, default_value_t = 1200.0)]
    rpm: f32,

    /// Number of blades
    #[arg(long, default_value_t = 3)]
    blades: u32,

    /// Position jitter fraction (0.0 = none, 1.0 = max noise)
    #[arg(long, default_value_t = 0.0, value_parser = validate_noise)]
    noise: f32,

    /// Output path for the .dat file
    #[arg(long, default_value = "data/synthetic/fan_test.dat")]
    output: PathBuf,

    /// Duration of generated data in seconds
    #[arg(long, default_value_t = 2.0)]
    duration: f32,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    // Create config from CLI arguments
    let config = SynthConfig {
        rpm: args.rpm,
        blade_count: args.blades,
        noise: args.noise,
        duration_secs: args.duration,
    };

    // Auto-derive truth JSON filename from output path
    // e.g., fan_test.dat -> fan_test_truth.json
    let truth_path = if let Some(stem) = args.output.file_stem() {
        let stem_str = stem.to_string_lossy();
        args.output.with_file_name(format!("{}_truth.json", stem_str))
    } else {
        args.output.with_extension("json")
    };

    println!("Generating synthetic fan data...");
    println!("  RPM:      {}", config.rpm);
    println!("  Blades:   {}", config.blade_count);
    println!("  Noise:    {}", config.noise);
    println!("  Duration: {}s", config.duration_secs);
    println!("  Output:   {}", args.output.display());
    println!("  Truth:    {}", truth_path.display());

    generate_fan_data_with_config(&args.output, &truth_path, &config)?;

    // Report file sizes
    let dat_size = std::fs::metadata(&args.output)?.len();
    let json_size = std::fs::metadata(&truth_path)?.len();

    println!("\nGenerated:");
    println!(
        "  {} - {:.2} MB",
        args.output.display(),
        dat_size as f64 / 1_000_000.0
    );
    println!(
        "  {} - {:.2} KB",
        truth_path.display(),
        json_size as f64 / 1_000.0
    );
    println!("\nRun with: cargo run -- {}", args.output.display());

    Ok(())
}

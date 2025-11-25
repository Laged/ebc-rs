//! Generate synthetic fan data for testing
//!
//! Usage: cargo run --bin generate_synthetic

use std::path::Path;
use ebc_rs::synthesis::generate_fan_data;

fn main() -> std::io::Result<()> {
    let output_path = Path::new("data/synthetic/fan_test.dat");
    let truth_path = Path::new("data/synthetic/fan_test_truth.json");

    println!("Generating synthetic fan data...");
    println!("  Output: {}", output_path.display());
    println!("  Truth:  {}", truth_path.display());

    generate_fan_data(output_path, truth_path)?;

    // Report file sizes
    let dat_size = std::fs::metadata(output_path)?.len();
    let json_size = std::fs::metadata(truth_path)?.len();

    println!("\nGenerated:");
    println!(
        "  {} - {:.2} MB",
        output_path.display(),
        dat_size as f64 / 1_000_000.0
    );
    println!(
        "  {} - {:.2} KB",
        truth_path.display(),
        json_size as f64 / 1_000.0
    );
    println!("\nRun with: cargo run -- data/synthetic/fan_test.dat");

    Ok(())
}

use ebc_rs::synthesis::generate_fan_data;
use std::path::Path;

fn main() -> std::io::Result<()> {
    let output_path = Path::new("data/synthetic/fan_rpm1200.dat");
    let truth_path = Path::new("data/synthetic/fan_rpm1200_truth.json");
    
    println!("Generating synthetic fan data to {:?}...", output_path);
    generate_fan_data(output_path, truth_path)?;
    println!("Done.");
    Ok(())
}
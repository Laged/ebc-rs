use bevy::prelude::*;
use clap::Parser;
use ebc_rs::plugins::CorePlugins;
use ebc_rs::EventFilePath; // Import the new resource

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to the event data file (.dat)
    #[clap(short, long, default_value = "data/fan/fan_const_rpm.dat")]
    data_path: String,
}

fn main() {
    let args = Args::parse();

    App::new()
        .add_plugins(CorePlugins)
        .insert_resource(EventFilePath(args.data_path)) // Insert the resource
        .run();
}
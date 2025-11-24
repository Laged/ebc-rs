use bevy::prelude::*;
use ebc_rs::edge_detection::EdgeDetectionPlugin;
use ebc_rs::EventFilePath;

fn main() {
    let data_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/fan/fan_const_rpm.dat".to_string());

    App::new()
        .insert_resource(EventFilePath(data_path))
        .add_plugins(DefaultPlugins)
        .add_plugins(EdgeDetectionPlugin)
        .run();
}

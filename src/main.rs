use bevy::prelude::*;

mod analysis;
mod gizmos;
mod gpu;
mod loader;
mod plugins;
mod render;
#[cfg(test)]
mod tests;

fn main() {
    App::new().add_plugins(plugins::CorePlugins).run();
}

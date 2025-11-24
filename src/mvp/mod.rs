pub mod gpu;
pub mod playback;
pub mod render;

use bevy::prelude::*;

pub struct MvpPlugin;

impl Plugin for MvpPlugin {
    fn build(&self, app: &mut App) {
        info!("MvpPlugin: Building MVP visualization pipeline");
    }
}
